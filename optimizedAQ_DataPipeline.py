"""
Air Quality Data Pipeline
=========================

Fetches sensor data from sensors.africa, cleans it, applies tiered imputation,
saves per-sensor monthly ML files, and uploads processed batches to the XYZT
platform.

New sources (IQAir + SCI Monitoring) are handled entirely by new_sources_pipeline.py
which is called at the end of each processing loop and manages its own fetch,
save, and upload cycle.


Imputation tiers (applied in order):
  Tier 1  – Linear interpolation       gaps ≤ SHORT_GAP_MINUTES (30 min)
  Tier 2  – Spatial cross-sensor fill  gaps ≤ MEDIUM_GAP_MINUTES (6 hrs)
  Tier 3  – Climatology median         gaps ≤ LARGE_GAP_MINUTES (24 hrs)
  Tier 4  – sensor_down flag           gaps > LARGE_GAP_MINUTES  (no fill)

Every output row carries a 'data_source' column so downstream consumers can
always distinguish real observations from imputed ones.
"""

import os
import time
import traceback
import warnings
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytz
import requests
from dateutil import parser as dtparser
from dotenv import load_dotenv

load_dotenv()

warnings.filterwarnings(
    "ignore", message="Mean of empty slice", category=RuntimeWarning
)
warnings.filterwarnings(
    "ignore", message="Degrees of freedom <= 0", category=RuntimeWarning
)


# === SENSORS.AFRICA CONFIGURATION ===
SENSORSAFRICA_USERNAME = os.getenv("SENSORSAFRICA_USERNAME", "")
SENSORSAFRICA_PASSWORD = os.getenv("SENSORSAFRICA_PASSWORD", "")
CITY = os.getenv("CITY", "Nakuru")
COUNTRY = os.getenv("COUNTRY", "Kenya")

# === PLATFORM-XYZT CONFIGURATION ===
PLATFORM_USER = os.getenv("XYZT_USER", "")
PLATFORM_PASSWORD = os.getenv("XYZT_PASSWORD", "")
PLATFORM_URL = os.getenv("PLATFORM_URL", "https://api.platform-xyzt.ai/")
DATASET_ID = os.getenv("XYZT_SENSORSAFRICA_DATASET_ID", "")  # sensor readings dataset

# =====================================================================================
# === PATH CONFIGURATION ===
# =====================================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "data_batches")
ML_DATA_DIR = os.path.join(BASE_DIR, "ml_data")
ML_LOCATION_DIR = os.path.join(ML_DATA_DIR, "by_location")

LAST_TIMESTAMP_FILE = os.path.join(BASE_DIR, "last_timestamp.txt")
BATCH_COUNTER_FILE = os.path.join(BASE_DIR, "batch_counter.txt")
UPLOADED_FILES_LOG = os.path.join(BASE_DIR, "uploaded_files.txt")
LAST_HEALTH_UPLOAD_FILE = os.path.join(BASE_DIR, "last_health_upload.txt")
LOCKFILE = os.path.join(BASE_DIR, "pipeline.lock")

# =====================================================================================
# === TIMING CONFIGURATION ===
# =====================================================================================
FETCH_WINDOW_MINUTES = 300  # how far ahead of last timestamp to fetch
SLEEP_BETWEEN_FETCH_LOOPS = 1800  # seconds to sleep between main loop iterations
SLEEP_BETWEEN_UPLOADS = 20  # seconds to pause between individual file uploads
FORCED_START_DATE = "2025-07-01T00:00:00Z"

# =====================================================================================
# === RETRY / BACKOFF CONFIGURATION ===
# =====================================================================================
RETRY_MAX_ATTEMPTS = 4  # total attempts (1 original + 3 retries)
RETRY_BACKOFF_BASE = 5  # seconds; wait = RETRY_BACKOFF_BASE * 2^attempt
#   attempt 0 → immediate
#   attempt 1 → wait  5 s
#   attempt 2 → wait 10 s
#   attempt 3 → wait 20 s

# =====================================================================================
# === RESAMPLING & IMPUTATION CONFIGURATION ===
# =====================================================================================
RESAMPLE_FREQ = "5min"
# Tier boundaries (minutes)
SHORT_GAP_MINUTES = 30  # Tier 1: linear interpolation limit
MEDIUM_GAP_MINUTES = 360  # Tier 2: spatial imputation limit  (6 hours)
LARGE_GAP_MINUTES = 1440  # Tier 3: climatology limit         (24 hours)
# Beyond LARGE_GAP_MINUTES  → Tier 4: mark as sensor_down, leave NaN

# Spatial imputation thresholds
SPATIAL_MIN_CORRELATION = 0.70  # Pearson r must be at least this to act as a donor
SPATIAL_MIN_PERIODS = 100  # min overlapping non-NaN rows to compute a valid correlation


# =====================================================================================
# === PM COLUMN NAMING
# sensors.africa field names do NOT match particle sizes numerically:
#   P0 = PM1    (finest particles)
#   P1 = PM10   (coarsest particles)
#   P2 = PM2.5  (intermediate)
# We rename immediately after unpacking so all downstream code uses clear names.
# =====================================================================================
SENSOR_VALUE_RENAME = {"P0": "PM1", "P1": "PM10", "P2": "PM2_5"}
PM_COLS = ["PM1", "PM10", "PM2_5"]
NUMERIC_SENSOR_COLS = ["humidity", "temperature"] + PM_COLS


# =====================================================================================
# === DATA SOURCE LABELS
# Every processed row carries one of these to record how its values were produced.
# =====================================================================================
SRC_RAW = "raw"  # Real API observation
SRC_INTERPOLATED = "interpolated"  # Linear interpolation across a short gap
SRC_SPATIAL = "spatial"  # Imputed from correlated nearby sensors
SRC_CLIMATOLOGY = "climatology"  # Imputed from historical median (hour × day-of-week)
SRC_SENSOR_DOWN = "sensor_down"  # Gap too long; sensor considered offline


# =====================================================================================
# === TIMEZONE ===
# =====================================================================================
EAT = pytz.timezone("Africa/Nairobi")


# =====================================================================================
# === LOGGER ===
# =====================================================================================
def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


# =====================================================================================
# === RETRY WRAPPER ===
# =====================================================================================
# === PROCESS LOCK (prevents two instances running simultaneously) ===
# =====================================================================================
def acquire_lock() -> None:
    """
    Write the current PID to LOCKFILE. If the file already exists and the
    recorded PID belongs to a running process, abort immediately.
    On Windows os.kill(pid, 0) raises OSError if the process does not exist,
    so this works as a liveness check on all platforms.
    """
    if os.path.exists(LOCKFILE):
        with open(LOCKFILE, "r") as f:
            old_pid_str = f.read().strip()
        if old_pid_str.isdigit():
            old_pid = int(old_pid_str)
            try:
                os.kill(old_pid, 0)
                log(
                    f"Another instance is already running (PID {old_pid}). "
                    "Exiting to prevent duplicate processing."
                )
                raise SystemExit(1)
            except OSError:
                log(f"Removing stale lock file (PID {old_pid} is no longer running).")

    with open(LOCKFILE, "w") as f:
        f.write(str(os.getpid()))
    log(f"Lock acquired (PID {os.getpid()}).")


def release_lock() -> None:
    if os.path.exists(LOCKFILE):
        os.remove(LOCKFILE)
        log("Lock released.")


# =====================================================================================
def with_retry(fn, *args, label: str = "operation", **kwargs):
    """
    Call fn(*args, **kwargs) with exponential backoff on any exception.

    Retries up to RETRY_MAX_ATTEMPTS times total. Waits RETRY_BACKOFF_BASE * 2^i
    seconds between attempts (5 s, 10 s, 20 s by default).
    Returns the function's return value on success, or raises the last exception
    after all attempts are exhausted.

    Usage:
        data = with_retry(get_sensor_data, token, start, end, CITY, COUNTRY,
                          label="sensors.africa fetch")
    """
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


# =====================================================================================
# === BATCH COUNTER ===
# =====================================================================================
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


# =====================================================================================
# === UPLOADED FILES TRACKING ===
# =====================================================================================
def load_uploaded_files() -> set:
    if os.path.exists(UPLOADED_FILES_LOG):
        with open(UPLOADED_FILES_LOG, "r") as f:
            return {line.strip() for line in f if line.strip()}
    return set()


def mark_file_uploaded(filename: str) -> None:
    with open(UPLOADED_FILES_LOG, "a") as f:
        f.write(f"{filename}\n")


# =====================================================================================
# === TIMESTAMP TRACKING ===
# =====================================================================================
def get_last_timestamp() -> str:
    if os.path.exists(LAST_TIMESTAMP_FILE):
        with open(LAST_TIMESTAMP_FILE, "r") as f:
            ts = f.read().strip()
            if ts:
                return ts
    return FORCED_START_DATE


def update_last_timestamp(new_ts: str) -> None:
    with open(LAST_TIMESTAMP_FILE, "w") as f:
        f.write(new_ts)


def parse_timestamp(ts: str) -> datetime:
    """Parse any ISO 8601 string (with or without fractional seconds, with any UTC offset)."""
    return dtparser.isoparse(ts).astimezone(timezone.utc).replace(tzinfo=timezone.utc)


# =====================================================================================
# === SENSORS.AFRICA AUTHENTICATION ===
# =====================================================================================
def get_sensorsafrica_token(username: str, password: str) -> str:
    url = "https://api.sensors.africa/get-auth-token/"
    response = requests.post(
        url, json={"username": username, "password": password}, timeout=30
    )
    response.raise_for_status()
    return response.json()["token"]


# =====================================================================================
# === DATA FETCHER ===
# =====================================================================================
def get_sensor_data(
    token: str, start_date: str, end_date: str, city: str, country: str
) -> list:
    url = "https://api.sensors.africa/v2/data/"
    headers = {"Authorization": f"Token {token}"}
    params = {
        "sensor__public": 1,
        "location__country": country,
        "location__city": city,
        "timestamp__gte": start_date,
        "timestamp__lte": end_date,
    }
    all_results = []
    next_url = url
    page = 1

    while next_url:
        log(f"  Fetching page {page}…")
        response = requests.get(
            next_url,
            headers=headers,
            params=params if next_url == url else None,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        all_results.extend(data.get("results", []))
        next_url = data.get("next")
        page += 1

    log(f"Fetched {len(all_results)} raw records.")
    return all_results


# =====================================================================================
# === CLEANING ===
# =====================================================================================
def clean_dataframe(raw_data: list) -> pd.DataFrame:
    """
    Unpack raw API records into a flat DataFrame.

    Key behaviours:
      - Renames PM columns immediately (P0→PM1, P1→PM10, P2→PM2_5).
      - Averages duplicate value_type entries within the same record (rather
        than silently taking the last one).
      - Extracts location_id and location_country from nested location dict.
    """
    df = pd.DataFrame(raw_data)
    if df.empty:
        return df

    def unpack_values(values: list) -> dict:
        """Average multiple readings of the same type within one record."""
        if not isinstance(values, list):
            return {}
        accumulator: dict[str, list] = {}
        for v in values:
            vtype = v.get("value_type")
            val = v.get("value")
            if vtype is None or val is None:
                continue
            try:
                accumulator.setdefault(vtype, []).append(float(val))
            except (ValueError, TypeError):
                continue
        return {k: round(sum(v) / len(v), 4) for k, v in accumulator.items() if v}

    def unpack_location(loc) -> dict:
        if isinstance(loc, dict):
            return {
                "location_id": loc.get("id"),
                "location_country": loc.get("country"),
            }
        return {"location_id": None, "location_country": None}

    expanded_values = pd.DataFrame(df["sensordatavalues"].apply(unpack_values).tolist())
    expanded_values = expanded_values.rename(columns=SENSOR_VALUE_RENAME)

    expanded_loc = pd.DataFrame(df["location"].apply(unpack_location).tolist())

    return pd.concat(
        [
            df.drop(columns=["sensordatavalues", "location"], errors="ignore"),
            expanded_values,
            expanded_loc,
        ],
        axis=1,
    )


# =====================================================================================
# === TIER 1: RESAMPLE PER SENSOR + SHORT-GAP LINEAR INTERPOLATION ===
# =====================================================================================
def resample_per_sensor(clean_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each sensor:
      1. Convert UTC timestamps to EAT.
      2. Resample to RESAMPLE_FREQ grid (mean within each 5-min slot).
      3. Tag slots that had at least one real observation as SRC_RAW.
      4. Linearly interpolate across gaps ≤ SHORT_GAP_MINUTES → SRC_INTERPOLATED.
      5. Leave longer gaps as NaN (handled by Tiers 2–4).

    Returns a tidy, per-row DataFrame with a 'data_source' column.
    """
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

    null_sensor_rows = clean_df["sensor"].isna().sum()
    if null_sensor_rows > 0:
        log(f"Warning: discarding {null_sensor_rows} records with null sensor ID.")

    log(f"Resampling {len(sensors)} sensors…")
    processed = []

    for sensor_id in sensors:
        sdf = clean_df[clean_df["sensor"] == sensor_id].copy()
        sdf = sdf.set_index("timestamp").sort_index()

        for col in NUMERIC_SENSOR_COLS:
            if col not in sdf.columns:
                sdf[col] = pd.NA

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            resampled_values = (
                sdf[NUMERIC_SENSOR_COLS].resample(RESAMPLE_FREQ).mean(numeric_only=True)
            )

        # Track which grid slots had at least one real observation
        has_raw = (
            sdf[NUMERIC_SENSOR_COLS].resample(RESAMPLE_FREQ).count().max(axis=1) > 0
        )

        # Tier 1: linear interpolation for short gaps only
        filled = resampled_values.interpolate(
            method="time",
            limit=short_gap_steps,
            limit_direction="both",
        ).round(2)

        # Build per-slot data_source
        data_source = pd.Series(SRC_SENSOR_DOWN, index=filled.index, dtype=str)
        data_source[has_raw] = SRC_RAW
        interp_mask = ~has_raw & filled.notna().any(axis=1)
        data_source[interp_mask] = SRC_INTERPOLATED

        # Metadata: forward/back fill is fine (sensor id, location don't change mid-life)
        available_meta = [c for c in metadata_cols if c in sdf.columns]
        if available_meta:
            resampled_meta = (
                sdf[available_meta].resample(RESAMPLE_FREQ).first().ffill().bfill()
            )
        else:
            resampled_meta = pd.DataFrame(index=filled.index)

        sensor_out = pd.concat([resampled_meta, filled], axis=1)
        sensor_out["data_source"] = data_source
        sensor_out = sensor_out.reset_index()

        # Add a UTC reference column alongside the EAT timestamp so downstream
        # consumers always have an unambiguous reference without needing to know
        # the local timezone.  (Report item: "EAT timestamps stored with no UTC reference")
        sensor_out["timestamp_utc"] = (
            sensor_out["timestamp"]
            .dt.tz_convert("UTC")
            .dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        )
        processed.append(sensor_out)

        n_raw = has_raw.sum()
        n_interp = interp_mask.sum()
        n_missing = filled.isna().all(axis=1).sum()
        log(
            f"  Sensor {sensor_id}: {len(sensor_out)} slots | "
            f"{n_raw} raw | {n_interp} interpolated | {n_missing} still missing"
        )

    if not processed:
        return pd.DataFrame()

    out = (
        pd.concat(processed, ignore_index=True)
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # Ensure all expected columns exist
    expected = [
        "id",
        "sampling_rate",
        "timestamp",
        "timestamp_utc",
        "sensor",
        "software_version",
        "humidity",
        "temperature",
        "PM2_5",
        "PM10",
        "PM1",
        "location_id",
        "location_country",
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


# =====================================================================================
# === TIER 2: SPATIAL (CROSS-SENSOR) IMPUTATION ===
# =====================================================================================
def spatial_impute(df: pd.DataFrame) -> pd.DataFrame:
    """
    For slots still missing after Tier 1 (gap between SHORT_GAP_MINUTES and
    MEDIUM_GAP_MINUTES), impute using sensors that ARE reporting at that timestamp.

    Strategy per variable column:
      1. Pivot the DataFrame to (timestamp × sensor).
      2. Compute Pearson correlation matrix across sensors (min SPATIAL_MIN_PERIODS
         overlapping non-NaN rows to filter out spurious correlations).
      3. For each sensor+timestamp that needs filling, identify donor sensors with
         r ≥ SPATIAL_MIN_CORRELATION that have a reading at that timestamp.
      4. Impute as a correlation-weighted average of donor values.
      5. Only fill contiguous NaN runs ≤ MEDIUM_GAP_MINUTES long; longer runs are
         left for Tier 3.

    Why this is better than climatology for medium gaps:
      Climatology returns the 'average day', systematically missing real pollution
      spikes or unusually clean air episodes that correlated neighbours DO capture.
    """
    if df.empty:
        return df

    df = df.copy()
    freq_td = pd.Timedelta(RESAMPLE_FREQ)
    medium_steps = int(pd.Timedelta(minutes=MEDIUM_GAP_MINUTES) / freq_td)
    short_steps = int(pd.Timedelta(minutes=SHORT_GAP_MINUTES) / freq_td)

    total_filled = 0

    for col in NUMERIC_SENSOR_COLS:
        if col not in df.columns:
            continue

        pivot = df.pivot_table(
            index="timestamp", columns="sensor", values=col, aggfunc="mean"
        )

        if pivot.shape[1] < 2:
            continue  # nothing to correlate

        corr_matrix = pivot.corr(min_periods=SPATIAL_MIN_PERIODS)

        filled_pivot = pivot.copy()

        for sensor_id in pivot.columns:
            nan_mask = pivot[sensor_id].isna()
            if not nan_mask.any():
                continue

            # Identify contiguous NaN runs; only fill those within the medium gap
            run_ids = nan_mask.ne(nan_mask.shift()).cumsum()
            run_sizes = nan_mask.groupby(run_ids).transform("sum")
            # Must be missing AND in a run that's (short_steps, medium_steps]
            eligible = (
                nan_mask & (run_sizes > short_steps) & (run_sizes <= medium_steps)
            )

            if not eligible.any():
                continue

            if sensor_id not in corr_matrix.index:
                continue

            correlations = (
                corr_matrix[sensor_id].drop(sensor_id, errors="ignore").dropna()
            )
            donors = correlations[correlations >= SPATIAL_MIN_CORRELATION]

            if donors.empty:
                continue

            donor_ids = donors.index.tolist()
            weights = donors.values  # shape (D,)

            eligible_ts = eligible[eligible].index
            donor_vals = pivot.loc[eligible_ts, donor_ids]  # shape (T, D)

            available = donor_vals.notna().values  # (T, D) bool
            w_matrix = np.where(available, weights[np.newaxis, :], 0.0)
            w_sum = w_matrix.sum(axis=1)  # (T,)

            imputed_values = np.where(
                w_sum > 0,
                (donor_vals.fillna(0).values * w_matrix).sum(axis=1) / w_sum,
                np.nan,
            )

            imputed_series = pd.Series(imputed_values, index=eligible_ts)
            actually_filled = imputed_series.dropna().index
            filled_pivot.loc[actually_filled, sensor_id] = imputed_series[
                actually_filled
            ].round(2)
            total_filled += len(actually_filled)

        # Melt pivot back and join onto df
        melted = filled_pivot.reset_index().melt(
            id_vars="timestamp", var_name="sensor", value_name=f"__sp_{col}"
        )
        melted["sensor"] = pd.to_numeric(melted["sensor"], errors="coerce").astype(
            "Int64"
        )

        df = df.merge(melted, on=["timestamp", "sensor"], how="left")

        was_nan = df[col].isna()
        new_val = df[f"__sp_{col}"]
        filled_here = was_nan & new_val.notna()
        df.loc[filled_here, col] = new_val[filled_here]
        df.loc[filled_here, "data_source"] = SRC_SPATIAL
        df = df.drop(columns=[f"__sp_{col}"])

    log(f"Spatial imputation filled {total_filled} sensor-column slots.")
    return df


# =====================================================================================
# === TIER 3: CLIMATOLOGY IMPUTATION + TIER 4: SENSOR DOWN FLAGGING ===
# =====================================================================================
def climatology_impute(df: pd.DataFrame) -> pd.DataFrame:
    """
    For gaps that survived Tiers 1 & 2:
      - Gaps ≤ LARGE_GAP_MINUTES  → fill with median(sensor, hour, day-of-week)
                                     and tag as SRC_CLIMATOLOGY.
      - Gaps > LARGE_GAP_MINUTES  → leave as NaN and tag as SRC_SENSOR_DOWN.

    Climatology is computed from all confirmed non-NaN values in the current
    batch (raw + interpolated + spatial). For sensors with very long outages
    this sample may be thin — downstream ML pipelines should check
    'data_source == sensor_down' and handle accordingly.
    """
    if df.empty:
        return df

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    freq_td = pd.Timedelta(RESAMPLE_FREQ)
    large_steps = int(pd.Timedelta(minutes=LARGE_GAP_MINUTES) / freq_td)

    df["_hour"] = df["timestamp"].dt.hour
    df["_dow"] = df["timestamp"].dt.dayofweek  # 0 = Monday

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
                # Climatology: median per (sensor, hour, day-of-week) from current batch
                clim = df.groupby(["sensor", "_hour", "_dow"])[col].transform(
                    lambda x: x.median(skipna=True)
                )
                fill_vals = clim.loc[eligible_idx]
                df.loc[eligible_idx, col] = fill_vals.values
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


# =====================================================================================
# === SAVE BATCH ===
# =====================================================================================
def save_batch(df: pd.DataFrame, start: str, end: str) -> tuple:
    """
    Save processed batch to SAVE_DIR.
    Returns (file_path, batch_number) so the batch number can be passed
    directly to upload_data_file without re-reading the counter.

    The counter is incremented BEFORE the file is written so that if the
    process is killed mid-save, the next run claims a new number and never
    overwrites a partially written file with a recycled batch number.
    """
    os.makedirs(SAVE_DIR, exist_ok=True)
    batch_num = get_batch_number()
    increment_batch_number()  # claim the number before touching disk
    safe_start = start.replace(":", "-")
    safe_end = end.replace(":", "-")
    filename = f"{SAVE_DIR}/batch_{batch_num:05d}_{safe_start}_to_{safe_end}.csv"
    df.to_csv(filename, index=False)
    log(f"Batch #{batch_num} saved → {len(df)} rows → {filename}")
    return filename, batch_num


# =====================================================================================
# === SAVE ML DATA BY SENSOR & MONTH ===
# =====================================================================================
def save_ml_by_sensor_and_month(df: pd.DataFrame) -> None:
    """
    Write per-sensor monthly CSV files to ML_LOCATION_DIR:
      by_location/sensor_<id>/sensor_<id>_YYYY-MM.csv

    Normal path (fast): simply appends new rows to the existing file.
    The timestamp window always advances forward so duplicates should not
    occur in normal operation.

    Safe path (slow, first-write-of-month only): if the file was just created
    this call (i.e. it did not exist before), nothing extra is needed.
    A full deduplication pass is triggered ONLY when the caller explicitly
    passes force_dedup=True — useful as a one-off repair tool.
    """
    if df.empty:
        log("No data to write for ML monthly files.")
        return

    os.makedirs(ML_LOCATION_DIR, exist_ok=True)
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["_year"] = df["timestamp"].dt.year
    df["_month"] = df["timestamp"].dt.month

    groups = list(df.groupby(["sensor", "_year", "_month"]))
    log(f"Writing ML files for {len(groups)} sensor-month group(s)…")

    for (sensor_id, year, month), group in groups:
        sensor_str = (
            "sensor_unknown" if pd.isna(sensor_id) else f"sensor_{int(sensor_id)}"
        )
        month_str = f"{int(year)}-{int(month):02d}"

        folder = os.path.join(ML_LOCATION_DIR, sensor_str)
        os.makedirs(folder, exist_ok=True)
        file_path = os.path.join(folder, f"{sensor_str}_{month_str}.csv")

        to_save = group.drop(columns=["_year", "_month"])

        if os.path.exists(file_path):
            # Fast path: append only — no read, no dedup, no sort.
            # parse_dates on large files is very slow; avoid it in the hot path.
            to_save.to_csv(file_path, mode="a", header=False, index=False)
        else:
            # First write for this sensor+month — write header + sorted rows.
            to_save.sort_values("timestamp").to_csv(file_path, index=False)

        log(f"  {sensor_str} {month_str}: +{len(to_save)} rows → {file_path}")

    log("Per-sensor monthly ML files updated.")


# =====================================================================================
# === PLATFORM AUTHENTICATION ===
# =====================================================================================
def request_platform_token() -> str:
    """
    Request a XYZT JWT token. Raises RuntimeError on any failure so that
    with_retry() can catch it and back off appropriately.
    """
    payload = {"userName": PLATFORM_USER, "password": PLATFORM_PASSWORD}
    response = requests.post(
        f"{PLATFORM_URL}public/api/tokens", json=payload, timeout=30
    )
    if response.status_code == 200:
        token = response.json().get("jwtToken", "")
        if token:
            log("Platform token obtained.")
            return token
        raise RuntimeError("Token endpoint returned 200 but jwtToken was empty.")
    raise RuntimeError(
        f"Platform token request failed ({response.status_code}): {response.text}"
    )


# =====================================================================================
# === UPLOAD ===
# =====================================================================================
def upload_data_file(
    dataset_id: str, file_path: str, token: str, batch_num: int
) -> bool:
    """
    Upload a single batch CSV to the XYZT platform.
    batch_num is passed in (not re-read from disk) to avoid the off-by-one
    that occurred when increment_batch_number() ran between save and upload.
    """
    log(f"Uploading {os.path.basename(file_path)} (batch #{batch_num})…")
    headers = {"Authorization": f"Bearer {token}"}
    upload_url = (
        f"{PLATFORM_URL}public/api/datasets/{dataset_id}/data/upload?batch={batch_num}"
    )

    try:
        with open(file_path, "rb") as fh:
            files = {"file": (os.path.basename(file_path), fh, "text/csv")}
            response = requests.post(
                upload_url, files=files, headers=headers, timeout=120
            )

        if response.status_code in (200, 201):
            log(f"Upload successful (batch #{batch_num}).")
            mark_file_uploaded(os.path.basename(file_path))
            return True

        log(f"Upload failed ({response.status_code}): {response.text}")
        return False

    except Exception as exc:
        log(f"Exception during upload: {exc}")
        return False


def upload_new_files(dataset_id: str, token: str) -> None:
    """
    Upload every batch CSV in SAVE_DIR that has not yet been uploaded.
    Token is passed in so it is shared with the health upload session.
    """
    if not os.path.exists(SAVE_DIR):
        log("Save directory does not exist; nothing to upload.")
        return

    uploaded = load_uploaded_files()
    all_files = sorted(
        f for f in os.listdir(SAVE_DIR) if f.startswith("batch_") and f.endswith(".csv")
    )
    new_files = [f for f in all_files if f not in uploaded]

    if not new_files:
        log("No new files to upload.")
        return

    log(f"Uploading {len(new_files)} new file(s)…")
    success = 0
    for filename in new_files:
        file_path = os.path.join(SAVE_DIR, filename)
        try:
            # Parse the batch number directly from the filename (batch_00001_...)
            batch_num = int(filename.split("_")[1])
        except (IndexError, ValueError):
            log(f"Could not parse batch number from {filename}; skipping.")
            continue

        try:
            if upload_data_file(dataset_id, file_path, token, batch_num):
                success += 1
        except Exception as exc:
            log(f"Error uploading {filename}: {exc}")

        time.sleep(SLEEP_BETWEEN_UPLOADS)

    log(f"Upload session done: {success}/{len(new_files)} successful.")


if __name__ == "__main__":
    import new_sources_pipeline

    log("Air quality pipeline starting…")
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(ML_LOCATION_DIR, exist_ok=True)

    acquire_lock()
    try:
        sensorsafrica_token = with_retry(
            get_sensorsafrica_token,
            SENSORSAFRICA_USERNAME,
            SENSORSAFRICA_PASSWORD,
            label="sensors.africa authentication",
        )
        log("sensors.africa authentication successful.")

        while True:
            try:
                # ── Determine fetch window ────────────────────────────────────────
                start_ts = get_last_timestamp()
                start_dt = parse_timestamp(start_ts)
                now_utc = datetime.now(tz=timezone.utc)

                if start_dt >= now_utc:
                    log("Already caught up to current time. Sleeping…")
                    time.sleep(SLEEP_BETWEEN_FETCH_LOOPS)
                    continue

                end_dt = min(
                    start_dt + timedelta(minutes=FETCH_WINDOW_MINUTES), now_utc
                )
                end_ts = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

                # ── sensors.africa: fetch → process → save ────────────────────────
                log(f"Fetching sensors.africa {start_ts} → {end_ts}")
                raw_data = with_retry(
                    get_sensor_data,
                    sensorsafrica_token,
                    start_ts,
                    end_ts,
                    CITY,
                    COUNTRY,
                    label="sensors.africa data fetch",
                )

                if raw_data:
                    clean_df = clean_dataframe(raw_data)
                    df = resample_per_sensor(clean_df)

                    if not df.empty:
                        df = spatial_impute(df)
                        df = climatology_impute(df)
                        save_batch(df, start_ts, end_ts)
                        save_ml_by_sensor_and_month(df)
                    else:
                        log("Resampled DataFrame is empty; advancing window.")
                else:
                    log("No sensors.africa data in this window; advancing.")

                update_last_timestamp(end_ts)
                log(f"Timestamp advanced to {end_ts}.")

                # ── New sources: IQAir + SCI Monitoring ──────────────────────────
                new_sources_pipeline.run(start_ts, end_ts)

                # # ── Upload ────────────────────────────────────────────────────────
                # log("Checking for files to upload…")
                # try:
                #     upload_token = with_retry(
                #         request_platform_token,
                #         label="XYZT platform token",
                #     )
                #     upload_new_files(DATASET_ID, upload_token)
                #     new_sources_pipeline.upload(upload_token)
                #     time.sleep(SLEEP_BETWEEN_UPLOADS)
                # except Exception as token_exc:
                #     log(f"Skipping uploads after all retries failed: {token_exc}")

                log(
                    "No Upload Done - Disable as uploading new files would break the live routine"
                )
                log("Speak to Respira-AQM if you'd to upload files")

            except KeyboardInterrupt:
                log("KeyboardInterrupt received — shutting down cleanly.")
                raise
            except BaseException as exc:
                log(f"Error in main loop ({type(exc).__name__}): {exc}")
                traceback.print_exc()

            log(f"Sleeping {SLEEP_BETWEEN_FETCH_LOOPS}s…")
            time.sleep(SLEEP_BETWEEN_FETCH_LOOPS)

    finally:
        release_lock()
