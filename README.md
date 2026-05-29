# Air Quality Data Pipeline
Continuously fetches Respira's project air quality sensor readings for Nakuru, Kenya, cleans them, applies a four-tier imputation chain, and uploads processed batches to the XYZT platform.

Two cooperating scripts:

| Script | Role |
|---|---|
| `optimizedAQ_DataPipeline.py` | **Orchestrator.** Runs in an infinite loop. Authenticates with sensors.africa, fetches the next time window, imputes gaps, saves batches and monthly ML files, then calls the new-sources pipeline and uploads everything. |
| `new_sources_pipeline.py` | **IQAir + SCI Monitoring sensors.** Imported and driven by the orchestrator. Has the same imputation chain but a separate batch counter, save folder, and XYZT dataset. Can also be run standalone for testing. |

---

## Folder layout


```
respira-xyzt-integration/
├── optimizedAQ_DataPipeline.py     ← main orchestrator (entry point)
├── new_sources_pipeline.py         ← IQAir + SCI module
├── requirements.txt
├── README.md                       ← you are here
│
├── data_batches/                   ← sensors.africa batch CSVs (created at runtime)
├── data_batches_new_sources/       ← IQAir + SCI batch CSVs (created at runtime)
├── ml_data/
│   └── by_location/                ← per-sensor monthly CSVs for sensors.africa
│       └── sensor_<id>/
│           └── sensor_<id>_YYYY-MM.csv
└── ml_data_new_sources/
    └── by_location/                ← per-sensor monthly CSVs for IQAir + SCI
        └── sensor_<id>/
            └── sensor_<id>_YYYY-MM.csv
```

Runtime state files also created automatically in the root folder:

| File | Purpose |
|---|---|
| `last_timestamp.txt` | Last UTC timestamp successfully processed by the main pipeline. The next loop iteration starts here. Delete it to start from `FORCED_START_DATE`. |
| `batch_counter.txt` | Monotonically increasing batch number for sensors.africa. |
| `batch_counter_new_sources.txt` | Same, for IQAir + SCI. |
| `uploaded_files.txt` | One filename per line — batches already pushed to XYZT (skipped on retry). |
| `uploaded_files_new_sources.txt` | Same, for the new sources. |
| `pipeline.lock` | Contains the PID of the running process. Prevents two instances from running at once. |

---

## Requirements

- **Python 3.9 or newer** (the code uses `dict[str, list]` style annotations, which need 3.9+).
- Network access to: `api.sensors.africa`, `device.iqair.com`, `sensor.sci-monitoring.com`, `api.platform-xyzt.ai`.
- Working credentials for sensors.africa and the XYZT platform  `optimizedAQ_DataPipeline.py`.

Python packages: `numpy`, `pandas`, `pytz`, `requests`, `python-dateutil`.

---

## Setup

From the root folder

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r "requirements.txt"
```

---

## Running

### Run the full pipeline (the normal case)

```bash
# c

# Start the orchestrator
python "optimizedAQ_DataPipeline.py"
```


What happens once it's running:

1. Acquires the lockfile (refuses to start if another instance is already running).
2. Authenticates against sensors.africa.
3. Loops forever:
   - Reads `last_timestamp.txt` → fetches the next `FETCH_WINDOW_MINUTES` (default 300 min = 5 hours).
   - Cleans, resamples to 5-min slots, runs the four imputation tiers.
   - Writes a batch CSV and appends to the relevant monthly ML files.
   - Calls `new_sources_pipeline.run(...)` for the same window.
   - Requests a fresh XYZT token, uploads any new batches from both pipelines.
   - Sleeps `SLEEP_BETWEEN_FETCH_LOOPS` (default 1800 s = 30 min) and repeats.

### Stopping it

Press `Ctrl+C`. The `finally` block releases the lockfile. The next run will pick up exactly where this one left off (it reads `last_timestamp.txt`).

If the process is killed hard (power loss, SIGKILL) and the lockfile is left behind, the next startup detects that the recorded PID is no longer alive and removes the stale lock automatically.

### Run the new-sources module on its own (for testing)

```bash
cd /path/to/parent
python "new_sources_pipeline.py"
```

In standalone mode it fetches one fixed test window (see `TEST_START` / `TEST_END` at the bottom of the file), saves a batch and ML files, and exits. **It does not upload** when run standalone — uploads only happen via the orchestrator, which provides the XYZT token.

---

## Configuration

All knobs live near the top of each file as plain module-level constants.

### `optimizedAQ_DataPipeline.py`

| Constant | Default | Meaning |
|---|---|---|
| `SENSORSAFRICA_USERNAME` / `_PASSWORD` | sensors.africa API credentials |
| `CITY`, `COUNTRY` | `"Nakuru"`, `"Kenya"` | Sensor location filter |
| `PLATFORM_USER` / `_PASSWORD` | XYZT platform credentials |
| `DATASET_ID` | XYZT dataset for sensors.africa data |
| `FETCH_WINDOW_MINUTES` | `300` | How much time to ingest per loop iteration |
| `SLEEP_BETWEEN_FETCH_LOOPS` | `1800` | Seconds between iterations (30 min) |
| `FORCED_START_DATE` | `"2025-07-01T00:00:00Z"` | Used **only** when `last_timestamp.txt` is missing or empty |
| `SHORT_GAP_MINUTES` | `30` | Tier 1 (linear interpolation) limit |
| `MEDIUM_GAP_MINUTES` | `360` | Tier 2 (spatial cross-sensor) limit |
| `LARGE_GAP_MINUTES` | `1440` | Tier 3 (climatology) limit; beyond this → `sensor_down` |
| `SPATIAL_MIN_CORRELATION` | `0.70` | Minimum Pearson r for a donor sensor |
| `SPATIAL_MIN_PERIODS` | `100` | Minimum overlapping non-NaN rows to trust a correlation |

### `new_sources_pipeline.py`

| Constant | Meaning |
|---|---|
| `NEW_SOURCE_DATASET_ID` | XYZT dataset for IQAir + SCI data |
| `SENSORS` | List of dicts — one per sensor, with `id`, `name`, `source` (`iqair`/`sci`), and the source-specific credentials |
| Imputation knobs | Same names and defaults as the main pipeline (intentionally — both pipelines must use the same gap thresholds so the resulting CSVs are directly comparable) |

## Output files

### Batch CSVs

Two parallel streams. Each row is one sensor at one 5-minute slot, with an `EAT` `timestamp` column and a UTC equivalent `timestamp_utc`. Every row carries a `data_source` value so downstream consumers can tell real readings from imputed ones:

| `data_source` | Meaning |
|---|---|
| `raw` | At least one real observation fell in this slot |
| `interpolated` | Tier 1 — linearly interpolated across a gap ≤ 30 min |
| `spatial` | Tier 2 — imputed from correlated neighbour sensors |
| `climatology` | Tier 3 — filled with the median for this (sensor, hour, day-of-week) |
| `sensor_down` | Tier 4 — gap > 24 h, left as NaN |

Filenames:

- `data_batches/batch_00001_2025-07-01T00-00-00Z_to_2025-07-01T05-00-00Z.csv`
- `data_batches_new_sources/new_source_batch_00001_…csv`

### Per-sensor monthly ML files

Accumulate across batches. New rows are appended on the fast path; deduplication is only done explicitly (the window always advances forward, so duplicates shouldn't appear in normal operation):

- `ml_data/by_location/sensor_4123/sensor_4123_2025-07.csv`
- `ml_data_new_sources/by_location/sensor_4974/sensor_4974_2025-07.csv`

---

## Troubleshooting

**"Another instance is already running (PID …). Exiting."**
Either another copy really is running, or a previous run died without cleaning up. Check with `ps -p <PID>` (Linux/macOS) or Task Manager (Windows). If the PID is gone, just delete `pipeline.lock` and start again.

**Pipeline keeps logging "Already caught up to current time."**
You've ingested up to "now". Expected behaviour — it will sleep `SLEEP_BETWEEN_FETCH_LOOPS` and check again.

**Want to re-process a time range?**
Stop the pipeline, edit `last_timestamp.txt` to your desired start time (ISO 8601 UTC, e.g. `2025-07-01T00:00:00Z`), and restart. The batch counter will keep advancing — old batches won't be overwritten.

**Upload returns 401 / 403**
The XYZT token has expired or the credentials are wrong. The pipeline requests a fresh token at the start of every upload cycle, so transient expirations resolve themselves. Persistent failures mean credentials need updating in `optimizedAQ_DataPipeline.py`.

**A sensor's column is all NaN with `data_source == sensor_down`**
The sensor has been silent for more than `LARGE_GAP_MINUTES` (24 h). This is by design — climatology over a multi-day outage would be misleading. Investigate the sensor directly.
