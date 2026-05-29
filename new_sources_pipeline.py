"""
New Sources Pipeline — IQAir & SCI Monitoring
==============================================
Fetches PM10 and PM2.5 from IQAir and SCI Monitoring sensors, applies the
same four-tier imputation chain as the main pipeline, saves batches and
per-sensor monthly ML files, and uploads to a dedicated XYZT dataset.

Entry points (called from air_quality_pipeline.py):
  run()           Fetch, process, and save for the next time window.
  upload(token)   Upload any pending batch files to XYZT.
"""

import os
import time
import warnings
from datetime import datetime, timezone
from io import StringIO

import numpy as np
import pandas as pd
import pytz
import requests
from dateutil import parser as dtparser
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

warnings.filterwarnings(
    "ignore", message="Mean of empty slice", category=RuntimeWarning
)
warnings.filterwarnings(
    "ignore", message="Degrees of freedom <= 0", category=RuntimeWarning
)


# -- Config -------------------------------------------------------------------

PLATFORM_URL = os.getenv("PLATFORM_URL", "https://api.platform-xyzt.ai/")
NEW_SOURCE_DATASET_ID = os.getenv("XYZT_SCI_DATASET_ID", "")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "data_batches_new_sources")
ML_LOCATION_DIR = os.path.join(BASE_DIR, "ml_data_new_sources", "by_location")
BATCH_COUNTER_FILE = os.path.join(BASE_DIR, "batch_counter_new_sources.txt")
UPLOADED_FILES_LOG = os.path.join(BASE_DIR, "uploaded_files_new_sources.txt")

FETCH_WINDOW_MINUTES = 300
SLEEP_BETWEEN_UPLOADS = 20
RETRY_MAX_ATTEMPTS = 4
RETRY_BACKOFF_BASE = 5

RESAMPLE_FREQ = "5min"
SHORT_GAP_MINUTES = 30
MEDIUM_GAP_MINUTES = 360
LARGE_GAP_MINUTES = 1440
SPATIAL_MIN_CORRELATION = 0.70
SPATIAL_MIN_PERIODS = 100

# Columns present in both sources
# IQAir-only:  PM1, CO2, pressure
# SCI-only:    NO2, O3, SO2, CO, NOX, TSP, VOC, wind_speed, wind_direction
NUMERIC_SENSOR_COLS = [
    "humidity",
    "temperature",
    "PM1",
    "PM2_5",
    "PM10",
    "CO2",
    "pressure",
    "NO2",
    "O3",
    "SO2",
    "CO",
    "NOX",
    "TSP",
    "VOC",
    "wind_speed",
    "wind_direction",
]

SRC_RAW = "raw"
SRC_INTERPOLATED = "interpolated"
SRC_SPATIAL = "spatial"
SRC_CLIMATOLOGY = "climatology"
SRC_SENSOR_DOWN = "sensor_down"

# SCI Monitoring expects EAT local time strings, not ISO 8601
SCI_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
EAT = pytz.timezone("Africa/Nairobi")

SENSORS = [
    {
        "id": 4974,
        "name": "Nakuru MET Station",
        "source": "iqair",
        "device_id": "6856994e52fb712759ff48a7",
    },
    {
        "id": 4975,
        "name": "Nakuru County Office",
        "source": "iqair",
        "device_id": "62b9caaf8e60f0ce6659e0c6",
    },
    {
        "id": 4976,
        "name": "Nakuru Athletics Club",
        "source": "sci",
        "code": "XHI308A20022200005",
        "key": "7cf9af47753d9ff986215c438c3ec4",
    },
    {
        "id": 4977,
        "name": "Nakuru Level 6 Hospital",
        "source": "sci",
        "code": "XHI308A20022200011",
        "key": "7cf9af47753d9ff986215c438c3ec4",
    },
    {
        "id": 4978,
        "name": "Nakuru High School Girls",
        "source": "sci",
        "code": "XHI308A20022200026",
        "key": "7cf9af47753d9ff986215c438c3ec4",
    },
]


# -- Utilities ----------------------------------------------------------------


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [NEW_SOURCES] {msg}")


def parse_timestamp(ts: str) -> datetime:
    return dtparser.isoparse(ts).astimezone(timezone.utc).replace(tzinfo=timezone.utc)


def with_retry(fn, *args, label: str = "operation", **kwargs):
    last_exc = None
    for attempt in range(RETRY_MAX_ATTEMPTS):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt < RETRY_MAX_ATTEMPTS - 1:
                wait = RETRY_BACKOFF_BASE * (2**attempt)
                log(
                    f"  {label} failed (attempt {attempt + 1}/{RETRY_MAX_ATTEMPTS}): {exc}"
                )
                log(f"  Retrying in {wait}s...")
                time.sleep(wait)
            else:
                log(f"  {label} failed after {RETRY_MAX_ATTEMPTS} attempts: {exc}")
    raise last_exc


# -- Batch tracking -----------------------------------------------------------


def get_batch_number() -> int:
    if os.path.exists(BATCH_COUNTER_FILE):
        with open(BATCH_COUNTER_FILE, "r") as f:
            content = f.read().strip()
            if content:
                return int(content)
    return 1


def increment_batch_number() -> None:
    with open(BATCH_COUNTER_FILE, "w") as f:
        f.write(str(get_batch_number() + 1))


def load_uploaded_files() -> set:
    if os.path.exists(UPLOADED_FILES_LOG):
        with open(UPLOADED_FILES_LOG, "r") as f:
            return {line.strip() for line in f if line.strip()}
    return set()


def mark_file_uploaded(filename: str) -> None:
    with open(UPLOADED_FILES_LOG, "a") as f:
        f.write(f"{filename}\n")


# -- IQAir fetcher ------------------------------------------------------------


def _flatten_iqair_record(record: dict) -> dict:
    """
    Normalise a single IQAir record to {ts, PM2_5, PM10} regardless of device type.

    Device type A — MET Station:   {"pm25": {"conc": 13}, "pm10": {"conc": 23}}
    Device type B — County Office:  instant {"p2": 6, "p1": 7}
                                    hourly  {"p2_sum": 14.8, "p2_count": 1, ...}
    """
    flat: dict = {"ts": record.get("ts")}

    if isinstance(record.get("pm25"), dict):
        flat["PM2_5"] = record["pm25"].get("conc")
    elif "p2" in record:
        flat["PM2_5"] = record["p2"]
    elif "p2_sum" in record and record.get("p2_count"):
        flat["PM2_5"] = round(record["p2_sum"] / record["p2_count"], 4)
    else:
        flat["PM2_5"] = np.nan

    if isinstance(record.get("pm10"), dict):
        flat["PM10"] = record["pm10"].get("conc")
    elif "p1" in record:
        flat["PM10"] = record["p1"]
    elif "p1_sum" in record and record.get("p1_count"):
        flat["PM10"] = round(record["p1_sum"] / record["p1_count"], 4)
    else:
        flat["PM10"] = np.nan

    # PM1 — MET Station uses "pm1", County Office uses "p01" / "p01_sum"
    if "pm1" in record:
        flat["PM1"] = record["pm1"]
    elif "p01" in record:
        flat["PM1"] = record["p01"]
    elif "p01_sum" in record and record.get("p01_count"):
        flat["PM1"] = round(record["p01_sum"] / record["p01_count"], 4)
    else:
        flat["PM1"] = np.nan

    # CO2 — MET Station field is "co2", County Office field is "co" (not carbon monoxide)
    if "co2" in record:
        flat["CO2"] = record["co2"]
    elif "co" in record:
        flat["CO2"] = record["co"]
    elif "co_sum" in record and record.get("co_count"):
        flat["CO2"] = round(record["co_sum"] / record["co_count"], 4)
    else:
        flat["CO2"] = np.nan

    # Pressure (Pa) — present on MET Station, may be absent on County Office
    flat["pressure"] = record.get("pr", np.nan)

    flat["temperature"] = record.get("tp", np.nan)
    flat["humidity"] = record.get("hm", np.nan)

    return flat


def fetch_iqair_data(device_id: str, start_ts: str, end_ts: str) -> pd.DataFrame:
    """
    IQAir device endpoints don't accept date-range params so we fetch all
    available data (instant + hourly), filter to the window, and deduplicate.
    Instant readings take priority over hourly at shared timestamps.
    """
    response = requests.get(f"https://device.iqair.com/v2/{device_id}", timeout=30)
    response.raise_for_status()

    hist = response.json().get("historical", {})
    records = [
        _flatten_iqair_record(r)
        for tier in ("instant", "hourly")
        for r in hist.get(tier, [])
    ]

    if not records:
        log(f"  IQAir {device_id}: API returned no records.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    df = df.drop(columns=["ts"]).dropna(subset=["timestamp"])
    df["PM2_5"] = pd.to_numeric(df["PM2_5"], errors="coerce")
    df["PM10"] = pd.to_numeric(df["PM10"], errors="coerce")

    start_dt = parse_timestamp(start_ts)
    end_dt = parse_timestamp(end_ts)
    df = df[(df["timestamp"] >= start_dt) & (df["timestamp"] <= end_dt)]
    df = (
        df.sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="first")
        .reset_index(drop=True)
    )

    log(f"  IQAir {device_id}: {len(df)} records in window.")
    return df


# -- SCI Monitoring fetcher ---------------------------------------------------


def fetch_sci_data(code: str, key: str, start_ts: str, end_ts: str) -> pd.DataFrame:
    """
    SCI API returns a CSV with EAT-local timestamps. PM columns are discovered
    by scanning the header for common name variants so firmware renames don't crash the pipeline.
    """
    start_eat = parse_timestamp(start_ts).astimezone(EAT).strftime(SCI_TIME_FORMAT)
    end_eat = parse_timestamp(end_ts).astimezone(EAT).strftime(SCI_TIME_FORMAT)

    response = requests.get(
        "http://sensor.sci-monitoring.com/Api/Data/getMiniteHisDataForCsvFileV2",
        params={"code": code, "key": key, "starttime": start_eat, "endtime": end_eat},
        timeout=60,
    )
    response.raise_for_status()

    raw_text = response.text.strip()
    if not raw_text:
        log(f"  SCI {code}: empty response.")
        return pd.DataFrame()

    df = pd.read_csv(StringIO(raw_text))
    if df.empty:
        log(f"  SCI {code}: no rows in CSV response.")
        return pd.DataFrame()

    # Timestamp column is named TIMESTAMP; convert EAT local → UTC
    df["timestamp"] = (
        pd.to_datetime(df["TIMESTAMP"], errors="coerce")
        .dt.tz_localize(EAT, ambiguous="infer", nonexistent="shift_forward")
        .dt.tz_convert("UTC")
    )
    df = df.dropna(subset=["timestamp"])

    # Column names confirmed from live API response
    out = pd.DataFrame({"timestamp": df["timestamp"]})
    out["PM2_5"] = pd.to_numeric(df.get("PM2_5"), errors="coerce")
    out["PM10"] = pd.to_numeric(df.get("PM10"), errors="coerce")
    out["temperature"] = pd.to_numeric(df.get("TEMPERATURE"), errors="coerce")
    out["humidity"] = pd.to_numeric(df.get("HUMIDITY"), errors="coerce")
    out["NO2"] = pd.to_numeric(df.get("NO2"), errors="coerce")
    out["O3"] = pd.to_numeric(df.get("O3"), errors="coerce")
    out["SO2"] = pd.to_numeric(df.get("SO2"), errors="coerce")
    out["CO"] = pd.to_numeric(df.get("CO"), errors="coerce")
    out["NOX"] = pd.to_numeric(df.get("NOX"), errors="coerce")
    out["TSP"] = pd.to_numeric(df.get("TSP"), errors="coerce")
    out["VOC"] = pd.to_numeric(df.get("PIDVOC"), errors="coerce")
    out["wind_speed"] = pd.to_numeric(df.get("WINDSPEED"), errors="coerce")
    out["wind_direction"] = pd.to_numeric(df.get("WINDDIRECTION"), errors="coerce")

    log(f"  SCI {code}: {len(out)} records in window.")
    return out.sort_values("timestamp").reset_index(drop=True)


# -- Schema normalisation -----------------------------------------------------


def normalise_to_pipeline_schema(
    sensor_df: pd.DataFrame, sensor_id: int
) -> pd.DataFrame:
    """Attach metadata columns expected by the imputation chain."""
    if sensor_df.empty:
        return pd.DataFrame()

    df = sensor_df.copy()
    df["sensor"] = sensor_id
    df["location_id"] = sensor_id
    df["location_country"] = "Kenya"
    df["id"] = pd.NA
    df["sampling_rate"] = pd.NA
    df["software_version"] = pd.NA

    # Columns present only in IQAir — will be NaN for SCI rows
    for col in ["PM1", "CO2", "pressure"]:
        if col not in df.columns:
            df[col] = np.nan

    # Columns present only in SCI — will be NaN for IQAir rows
    for col in [
        "NO2",
        "O3",
        "SO2",
        "CO",
        "NOX",
        "TSP",
        "VOC",
        "wind_speed",
        "wind_direction",
    ]:
        if col not in df.columns:
            df[col] = np.nan

    # Columns always expected from both sources
    for col in ["PM2_5", "PM10", "humidity", "temperature"]:
        if col not in df.columns:
            df[col] = np.nan

    ordered = [
        "id",
        "sampling_rate",
        "timestamp",
        "sensor",
        "software_version",
        "location_id",
        "location_country",
        "humidity",
        "temperature",
        "PM1",
        "PM2_5",
        "PM10",
        "CO2",
        "pressure",
        "NO2",
        "O3",
        "SO2",
        "CO",
        "NOX",
        "TSP",
        "VOC",
        "wind_speed",
        "wind_direction",
    ]
    for col in ordered:
        if col not in df.columns:
            df[col] = pd.NA

    return df[ordered]


# -- Imputation chain (mirrors air_quality_pipeline.py exactly) ---------------


def resample_per_sensor(clean_df: pd.DataFrame) -> pd.DataFrame:
    if clean_df.empty:
        return clean_df

    clean_df = clean_df.copy()
    clean_df["timestamp"] = pd.to_datetime(
        clean_df["timestamp"], errors="coerce", utc=True
    )
    clean_df = clean_df.dropna(subset=["timestamp"])
    clean_df["timestamp"] = clean_df["timestamp"].dt.tz_convert(EAT)

    metadata_cols = [
        "id",
        "sampling_rate",
        "sensor",
        "software_version",
        "location_id",
        "location_country",
    ]
    freq_td = pd.Timedelta(RESAMPLE_FREQ)
    short_gap_steps = int(pd.Timedelta(minutes=SHORT_GAP_MINUTES) / freq_td)
    sensors = clean_df["sensor"].dropna().unique()

    if len(sensors) == 0:
        log("No valid sensor IDs found in this batch.")
        return pd.DataFrame()

    null_rows = clean_df["sensor"].isna().sum()
    if null_rows > 0:
        log(f"Warning: discarding {null_rows} records with null sensor ID.")

    log(f"Resampling {len(sensors)} sensors…")
    processed = []

    for sensor_id in sensors:
        sdf = (
            clean_df[clean_df["sensor"] == sensor_id]
            .copy()
            .set_index("timestamp")
            .sort_index()
        )

        for col in NUMERIC_SENSOR_COLS:
            if col not in sdf.columns:
                sdf[col] = pd.NA

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            resampled_values = (
                sdf[NUMERIC_SENSOR_COLS].resample(RESAMPLE_FREQ).mean(numeric_only=True)
            )

        has_raw = (
            sdf[NUMERIC_SENSOR_COLS].resample(RESAMPLE_FREQ).count().max(axis=1) > 0
        )
        filled = resampled_values.interpolate(
            method="time", limit=short_gap_steps, limit_direction="both"
        ).round(2)
        interp_mask = ~has_raw & filled.notna().any(axis=1)

        data_source = pd.Series(SRC_SENSOR_DOWN, index=filled.index, dtype=str)
        data_source[has_raw] = SRC_RAW
        data_source[interp_mask] = SRC_INTERPOLATED

        available_meta = [c for c in metadata_cols if c in sdf.columns]
        resampled_meta = (
            sdf[available_meta].resample(RESAMPLE_FREQ).first().ffill().bfill()
            if available_meta
            else pd.DataFrame(index=filled.index)
        )

        sensor_out = pd.concat([resampled_meta, filled], axis=1)
        sensor_out["data_source"] = data_source
        sensor_out["timestamp_utc"] = sensor_out.index.tz_convert("UTC").strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        sensor_out = sensor_out.reset_index()
        processed.append(sensor_out)

        log(
            f"  Sensor {sensor_id}: {len(sensor_out)} slots | {has_raw.sum()} raw | {interp_mask.sum()} interpolated | {filled.isna().all(axis=1).sum()} still missing"
        )

    if not processed:
        return pd.DataFrame()

    out = (
        pd.concat(processed, ignore_index=True)
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    expected = [
        "id",
        "sampling_rate",
        "timestamp",
        "timestamp_utc",
        "sensor",
        "software_version",
        "location_id",
        "location_country",
        "humidity",
        "temperature",
        "PM1",
        "PM2_5",
        "PM10",
        "CO2",
        "pressure",
        "NO2",
        "O3",
        "SO2",
        "CO",
        "NOX",
        "TSP",
        "VOC",
        "wind_speed",
        "wind_direction",
        "data_source",
    ]
    for col in expected:
        if col not in out.columns:
            out[col] = pd.NA

    for col in NUMERIC_SENSOR_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in ["id", "sensor", "location_id"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")

    return out


def spatial_impute(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    medium_steps = int(
        pd.Timedelta(minutes=MEDIUM_GAP_MINUTES) / pd.Timedelta(RESAMPLE_FREQ)
    )
    short_steps = int(
        pd.Timedelta(minutes=SHORT_GAP_MINUTES) / pd.Timedelta(RESAMPLE_FREQ)
    )
    total_filled = 0

    for col in NUMERIC_SENSOR_COLS:
        if col not in df.columns:
            continue

        pivot = df.pivot_table(
            index="timestamp", columns="sensor", values=col, aggfunc="mean"
        )
        if pivot.shape[1] < 2:
            continue

        corr_matrix = pivot.corr(min_periods=SPATIAL_MIN_PERIODS)
        filled_pivot = pivot.copy()

        for sensor_id in pivot.columns:
            nan_mask = pivot[sensor_id].isna()
            run_ids = nan_mask.ne(nan_mask.shift()).cumsum()
            run_sizes = nan_mask.groupby(run_ids).transform("sum")
            eligible = (
                nan_mask & (run_sizes > short_steps) & (run_sizes <= medium_steps)
            )

            if not eligible.any() or sensor_id not in corr_matrix.index:
                continue

            donors = corr_matrix[sensor_id].drop(sensor_id, errors="ignore").dropna()
            donors = donors[donors >= SPATIAL_MIN_CORRELATION]
            if donors.empty:
                continue

            eligible_ts = eligible[eligible].index
            donor_vals = pivot.loc[eligible_ts, donors.index.tolist()]
            available = donor_vals.notna().values
            w_matrix = np.where(available, donors.values[np.newaxis, :], 0.0)
            w_sum = w_matrix.sum(axis=1)
            imputed_values = np.where(
                w_sum > 0,
                (donor_vals.fillna(0).values * w_matrix).sum(axis=1) / w_sum,
                np.nan,
            )

            actually_filled = (
                pd.Series(imputed_values, index=eligible_ts).dropna().index
            )
            filled_pivot.loc[actually_filled, sensor_id] = pd.Series(
                imputed_values, index=eligible_ts
            )[actually_filled].round(2)
            total_filled += len(actually_filled)

        melted = filled_pivot.reset_index().melt(
            id_vars="timestamp", var_name="sensor", value_name=f"__sp_{col}"
        )
        melted["sensor"] = pd.to_numeric(melted["sensor"], errors="coerce").astype(
            "Int64"
        )
        df = df.merge(melted, on=["timestamp", "sensor"], how="left")

        filled_here = df[col].isna() & df[f"__sp_{col}"].notna()
        df.loc[filled_here, col] = df.loc[filled_here, f"__sp_{col}"]
        df.loc[filled_here, "data_source"] = SRC_SPATIAL
        df = df.drop(columns=[f"__sp_{col}"])

    log(f"Spatial imputation filled {total_filled} sensor-column slots.")
    return df


def climatology_impute(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    large_steps = int(
        pd.Timedelta(minutes=LARGE_GAP_MINUTES) / pd.Timedelta(RESAMPLE_FREQ)
    )
    df["_hour"] = df["timestamp"].dt.hour
    df["_dow"] = df["timestamp"].dt.dayofweek
    clim_filled = 0
    down_flagged = 0

    for col in NUMERIC_SENSOR_COLS:
        if col not in df.columns:
            continue

        for sensor_id, sdf_idx in df.groupby("sensor").groups.items():
            sdf = df.loc[sdf_idx]
            nan_mask = sdf[col].isna()
            if not nan_mask.any():
                continue

            run_ids = nan_mask.ne(nan_mask.shift()).cumsum()
            run_sizes = nan_mask.groupby(run_ids).transform("sum")

            eligible_idx = sdf.index[nan_mask & (run_sizes <= large_steps)]
            beyond_idx = sdf.index[nan_mask & (run_sizes > large_steps)]

            if len(eligible_idx) > 0:
                clim = df.groupby(["sensor", "_hour", "_dow"])[col].transform(
                    lambda x: x.median(skipna=True)
                )
                df.loc[eligible_idx, col] = clim.loc[eligible_idx].values
                actually_filled = eligible_idx[df.loc[eligible_idx, col].notna()]
                df.loc[actually_filled, "data_source"] = SRC_CLIMATOLOGY
                clim_filled += len(actually_filled)

            if len(beyond_idx) > 0:
                df.loc[beyond_idx, "data_source"] = SRC_SENSOR_DOWN
                down_flagged += len(beyond_idx)

    df = df.drop(columns=["_hour", "_dow"])
    log(
        f"Climatology filled {clim_filled} slots. {down_flagged} slots marked sensor_down."
    )
    return df


# -- Save ---------------------------------------------------------------------


def save_batch(df: pd.DataFrame, start: str, end: str) -> tuple:
    os.makedirs(SAVE_DIR, exist_ok=True)
    batch_num = get_batch_number()
    increment_batch_number()
    filename = f"{SAVE_DIR}/new_source_batch_{batch_num:05d}_{start.replace(':', '-')}_to_{end.replace(':', '-')}.csv"
    df.to_csv(filename, index=False)
    log(f"Batch #{batch_num} saved → {len(df)} rows → {filename}")
    return filename, batch_num


def save_ml_by_sensor_and_month(df: pd.DataFrame) -> None:
    if df.empty:
        log("No data to write for ML monthly files.")
        return

    os.makedirs(ML_LOCATION_DIR, exist_ok=True)
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["_year"] = df["timestamp"].dt.year
    df["_month"] = df["timestamp"].dt.month

    for (sensor_id, year, month), group in df.groupby(["sensor", "_year", "_month"]):
        sensor_str = (
            "sensor_unknown" if pd.isna(sensor_id) else f"sensor_{int(sensor_id)}"
        )
        month_str = f"{int(year)}-{int(month):02d}"
        folder = os.path.join(ML_LOCATION_DIR, sensor_str)
        os.makedirs(folder, exist_ok=True)
        file_path = os.path.join(folder, f"{sensor_str}_{month_str}.csv")
        to_save = group.drop(columns=["_year", "_month"])

        if os.path.exists(file_path):
            to_save.to_csv(file_path, mode="a", header=False, index=False)
        else:
            to_save.sort_values("timestamp").to_csv(file_path, index=False)

        log(f"  {sensor_str} {month_str}: +{len(to_save)} rows → {file_path}")

    log("Per-sensor monthly ML files updated.")


# -- Public entry points ------------------------------------------------------


def run(start_ts: str, end_ts: str) -> None:
    """Fetch, process, and save for the given time window.
    start_ts and end_ts are passed in from air_quality_pipeline.py
    so both pipelines always operate on the exact same window.
    """
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(ML_LOCATION_DIR, exist_ok=True)

    log(f"Fetching window {start_ts} → {end_ts}")
    all_dfs = []

    for sensor in SENSORS:
        sid, sname = sensor["id"], sensor["name"]
        log(f"  Fetching: {sname} (id={sid})…")
        try:
            if sensor["source"] == "iqair":
                raw = with_retry(
                    fetch_iqair_data,
                    sensor["device_id"],
                    start_ts,
                    end_ts,
                    label=f"IQAir fetch {sname}",
                )
            else:
                raw = with_retry(
                    fetch_sci_data,
                    sensor["code"],
                    sensor["key"],
                    start_ts,
                    end_ts,
                    label=f"SCI fetch {sname}",
                )
        except Exception as exc:
            log(f"  {sname}: skipping after all retries — {exc}")
            continue

        if not raw.empty:
            all_dfs.append(normalise_to_pipeline_schema(raw, sid))
        else:
            log(f"  {sname}: no data in this window.")

    if not all_dfs:
        log("No data retrieved from any sensor in this window.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    log(f"Combined: {len(combined)} raw rows across {len(all_dfs)} sensor(s).")

    df = resample_per_sensor(combined)
    if df.empty:
        log("Resampled DataFrame is empty; nothing to save.")
        return

    df = spatial_impute(df)
    df = climatology_impute(df)
    save_batch(df, start_ts, end_ts)
    save_ml_by_sensor_and_month(df)


def upload(token: str) -> None:
    """Upload pending batch files to XYZT. Token is provided by the main pipeline."""
    if not os.path.exists(SAVE_DIR):
        log("Save directory does not exist; nothing to upload.")
        return

    uploaded = load_uploaded_files()
    pending = [
        f
        for f in sorted(os.listdir(SAVE_DIR))
        if f.startswith("new_source_batch_")
        and f.endswith(".csv")
        and f not in uploaded
    ]

    if not pending:
        log("No pending files to upload.")
        return

    total = len(pending)
    success = 0
    log(f"Uploading {total} file(s) to dataset {NEW_SOURCE_DATASET_ID}…")

    for i, filename in enumerate(pending, start=1):
        try:
            batch_num = int(filename.split("_")[3])
        except (IndexError, ValueError):
            log(
                f"  [{i}/{total}] Could not parse batch number from '{filename}'; skipping."
            )
            continue

        log(f"  [{i}/{total}] Uploading {filename} (batch #{batch_num})…")
        try:
            with open(os.path.join(SAVE_DIR, filename), "rb") as fh:
                response = requests.post(
                    f"{PLATFORM_URL}public/api/datasets/{NEW_SOURCE_DATASET_ID}/data/upload?batch={batch_num}",
                    files={"file": (filename, fh, "text/csv")},
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=120,
                )
            if response.status_code in (200, 201):
                mark_file_uploaded(filename)
                success += 1
                log(f"  [{i}/{total}] ✓ Upload successful.")
            else:
                log(
                    f"  [{i}/{total}] ✗ Upload failed ({response.status_code}) — stopping to preserve order."
                )
                break
        except Exception as exc:
            log(f"  [{i}/{total}] ✗ Exception: {exc} — stopping.")
            break

        time.sleep(SLEEP_BETWEEN_UPLOADS)

    log(f"Upload complete: {success}/{total} successful.")


if __name__ == "__main__":
    log(
        "Running standalone (fetch + save only; upload handled by air_quality_pipeline.py)."
    )
    # When running standalone, set your desired window here
    TEST_START = "2026-03-01T00:00:00Z"
    TEST_END = "2026-03-01T05:00:00Z"
    run(TEST_START, TEST_END)
    log("Done.")
