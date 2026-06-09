"""
Dataset ingestion and stream construction for the clamped w-event DP pipeline.

Scope (everything the experimental runner needs from a raw dataset on disk):

  1.  Per-dataset loaders that read the source CSV(s).
  2.  Per-dataset stream builders that resample raw rows into the pair
      (aggregate stream a_tau, publisher-count stream n_tau) plus the
      per-publisher table used for per-topic experiments.
  3.  A ``DATASETS`` registry that binds each dataset to its loader,
      stream builders, topic hierarchy, and clamp metadata.
  4.  The Definition 3.2 clamp options (static / DP-released) applied on
      per-publisher streams.
  5.  A ``prepare_dataset`` helper that performs the entire
      load -> build -> clamp pipeline and returns a ``PreparedDataset``.

``run_experiment.py`` imports exclusively from here when it needs data;
no experimental / plotting / tuning code lives in this module.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def _stable_hash(s: str) -> int:
    """Deterministic, process-independent hash of a string -> [0, 10000).

    Python's built-in ``hash()`` is salted per process (PYTHONHASHSEED), so
    using it to derive a per-sensor RNG seed makes Option-B DP-released clamps
    NON-reproducible across runs/processes.  md5 gives a stable digest so the
    same sensor always seeds the same clamp noise.
    """
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16) % 10000

logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════
#  Paths
# ═════════════════════════════════════════════════════════════════════════

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
ENERGY_CSV = os.path.join(
    DATA_DIR,
    "Multi-Circuit Electric Consumption Data for Application of Energy Disaggregation.csv",
)
TRAFFIC_DIR = os.path.join(DATA_DIR, "2024_12_20")
WEARABLE_CSV = os.path.join(DATA_DIR, "Wearable IoT Health Dataset.csv")
PUNE_CSV = os.path.join(DATA_DIR, "Pune_SmartCity_Test_Dataset.csv")
MOBILITY_CSV = os.path.join(DATA_DIR, "smart_mobility_dataset.csv")
MANUFACTURING_CSV = os.path.join(DATA_DIR, "Manufacturing_dataset.csv")


# ═════════════════════════════════════════════════════════════════════════
#  Energy dataset (MCEC-Thai)
# ═════════════════════════════════════════════════════════════════════════

ENERGY_CIRCUITS = [
    "CT1", "CT2", "CT3", "CT4", "CT5",
    "CT6", "CT7", "CT8", "CT9", "CT10",
    "CT13", "CT17",
]
ENERGY_SENSORS = {
    "power_kw":       {"col_suffix": "_kW",    "unit": "kW"},
    "voltage":        {"col_suffix": "_V",     "unit": "V"},
    "current":        {"col_suffix": "_A",     "unit": "A"},
    "reactive_power": {"col_suffix": "_kVar+", "unit": "kVar"},
    "power_factor":   {"col_suffix": "_PF",    "unit": "PF"},
}


def load_energy_dataset(max_timestamps: int | None = None) -> pd.DataFrame:
    logger.info(f"Loading energy dataset from {ENERGY_CSV}")
    df = pd.read_csv(ENERGY_CSV, low_memory=False)
    df["Time"] = pd.to_datetime(df["Time"], format="mixed", dayfirst=False)
    df = df.sort_values("Time").reset_index(drop=True)
    if max_timestamps and len(df) > max_timestamps:
        df = df.iloc[:max_timestamps]
    logger.info(f"  Loaded {len(df)} timestamps, {df['Time'].min()} to {df['Time'].max()}")
    return df


def build_energy_streams(df, sensor_type="power_kw", window_minutes=5):
    col_suffix = ENERGY_SENSORS[sensor_type]["col_suffix"]
    circuit_cols = [f"{ct}{col_suffix}" for ct in ENERGY_CIRCUITS
                    if f"{ct}{col_suffix}" in df.columns]
    if not circuit_cols:
        raise ValueError(f"No columns found for sensor type {sensor_type}")
    for col in circuit_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.set_index("Time")
    resampled = df[circuit_cols].resample(f"{window_minutes}min")
    aggregates, pub_counts, all_values = [], [], []
    for _, window_df in resampled:
        if window_df.empty:
            continue
        means_per_circuit = window_df.mean()
        active = means_per_circuit.dropna()
        active = active[active != 0]
        if len(active) > 0:
            aggregates.append(float(active.mean()))
            pub_counts.append(len(active))
            all_values.extend(active.values)
        else:
            aggregates.append(0.0); pub_counts.append(0)
    payload_bound = float(np.ptp(all_values)) if all_values else 1.0
    if payload_bound == 0:
        payload_bound = 1.0
    logger.info(f"  Energy [{sensor_type}]: {len(aggregates)} windows, "
                f"B={payload_bound:.4f}, avg pubs={np.mean(pub_counts):.1f}")
    return aggregates, pub_counts, payload_bound


def build_energy_per_publisher(df, sensor_type="power_kw", window_minutes=5):
    col_suffix = ENERGY_SENSORS[sensor_type]["col_suffix"]
    circuit_cols = {ct: f"{ct}{col_suffix}" for ct in ENERGY_CIRCUITS
                    if f"{ct}{col_suffix}" in df.columns}
    if not circuit_cols:
        raise ValueError(f"No columns found for sensor type {sensor_type}")
    for col in circuit_cols.values():
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.set_index("Time")
    resampled = df[list(circuit_cols.values())].resample(f"{window_minutes}min")
    per_pub: dict[str, list[float | None]] = {ct: [] for ct in circuit_cols}
    all_values = []
    for _, window_df in resampled:
        if window_df.empty:
            for ct in circuit_cols:
                per_pub[ct].append(None)
            continue
        for ct, col in circuit_cols.items():
            val = window_df[col].mean()
            if pd.notna(val) and val != 0:
                per_pub[ct].append(float(val)); all_values.append(float(val))
            else:
                per_pub[ct].append(None)
    payload_bound = float(np.ptp(all_values)) if all_values else 1.0
    if payload_bound == 0:
        payload_bound = 1.0
    return per_pub, payload_bound


# ═════════════════════════════════════════════════════════════════════════
#  Traffic dataset (Colorado Springs)
# ═════════════════════════════════════════════════════════════════════════

def load_traffic_dataset(max_rows_per_file: int | None = None):
    logger.info(f"Loading traffic dataset from {TRAFFIC_DIR}")
    sensors = {}
    for fname in sorted(os.listdir(TRAFFIC_DIR)):
        if not fname.endswith(".csv") or fname == "PARAMS.csv":
            continue
        fpath = os.path.join(TRAFFIC_DIR, fname)
        sensor_name = fname.replace(".csv", "")
        df = pd.read_csv(fpath, low_memory=False)
        df["Time"] = pd.to_datetime(df["Time"], format="mixed")
        df = df.sort_values("Time").reset_index(drop=True)
        if max_rows_per_file and len(df) > max_rows_per_file:
            df = df.iloc[:max_rows_per_file]
        sensors[sensor_name] = df
        logger.info(f"  {sensor_name}: {len(df)} detections")
    return sensors


def _traffic_time_bins(sensors, window_seconds):
    all_times = []
    for df in sensors.values():
        all_times.extend(df["Time"].values)
    t_min = pd.Timestamp(min(all_times))
    t_max = pd.Timestamp(max(all_times))
    freq = pd.Timedelta(seconds=window_seconds)
    return pd.date_range(start=t_min, end=t_max + freq, freq=freq)


def _traffic_metric(window_df, metric):
    if metric == "object_count":
        return float(window_df["ObjectId"].nunique())
    if metric == "speed":
        return float(window_df["Speed"].mean())
    if metric == "position_x":
        return float(window_df["PositionX"].mean())
    if metric == "heading":
        col = "HeadingDeg_DERIVED" if "HeadingDeg_DERIVED" in window_df.columns else "HeadingDeg"
        if col not in window_df.columns:
            return None
        return float(window_df[col].mean())
    raise ValueError(f"Unknown traffic metric: {metric}")


def build_traffic_streams(sensors, metric="speed", window_seconds=10):
    bins = _traffic_time_bins(sensors, window_seconds)
    aggregates, pub_counts, all_values = [], [], []
    for i in range(len(bins) - 1):
        w_start, w_end = bins[i], bins[i + 1]
        sensor_values = []
        for df in sensors.values():
            window_df = df[(df["Time"] >= w_start) & (df["Time"] < w_end)]
            if window_df.empty:
                continue
            val = _traffic_metric(window_df, metric)
            if val is not None and np.isfinite(val):
                sensor_values.append(val)
        if sensor_values:
            aggregates.append(float(np.mean(sensor_values)))
            pub_counts.append(len(sensor_values))
            all_values.extend(sensor_values)
        else:
            aggregates.append(0.0); pub_counts.append(0)
    payload_bound = float(np.ptp(all_values)) if all_values else 1.0
    if payload_bound == 0:
        payload_bound = 1.0
    logger.info(f"  Traffic [{metric}]: {len(aggregates)} windows, "
                f"B={payload_bound:.2f}, avg pubs={np.mean(pub_counts):.1f}")
    return aggregates, pub_counts, payload_bound


def build_traffic_per_publisher(sensors, metric="speed", window_seconds=10):
    bins = _traffic_time_bins(sensors, window_seconds)
    per_pub: dict[str, list[float | None]] = {name: [] for name in sensors}
    all_values = []
    for i in range(len(bins) - 1):
        w_start, w_end = bins[i], bins[i + 1]
        for sensor_name, df in sensors.items():
            window_df = df[(df["Time"] >= w_start) & (df["Time"] < w_end)]
            if window_df.empty:
                per_pub[sensor_name].append(None); continue
            val = _traffic_metric(window_df, metric)
            if val is not None and np.isfinite(val):
                per_pub[sensor_name].append(val); all_values.append(val)
            else:
                per_pub[sensor_name].append(None)
    payload_bound = float(np.ptp(all_values)) if all_values else 1.0
    if payload_bound == 0:
        payload_bound = 1.0
    return per_pub, payload_bound


# ═════════════════════════════════════════════════════════════════════════
#  Wearable IoT Healthcare (Kaggle dcsavinod)
# ═════════════════════════════════════════════════════════════════════════

WEARABLE_SENSORS = {
    "heart_rate":      {"col": "Heart_Rate",      "unit": "bpm"},
    "steps":           {"col": "Steps",           "unit": "count"},
    "temperature":     {"col": "Temperature",     "unit": "C"},
    "humidity":        {"col": "Humidity",        "unit": "%"},
    "calories_burned": {"col": "Calories_Burned", "unit": "kcal"},
}


def load_wearable_dataset(max_rows: int | None = None):
    logger.info(f"Loading wearable dataset from {WEARABLE_CSV}")
    df = pd.read_csv(WEARABLE_CSV, low_memory=False)
    df = df.sort_values(["Device_ID", "Timestamp"]).reset_index(drop=True)
    df["tau"] = df.groupby("Device_ID").cumcount()
    if max_rows and len(df) > max_rows:
        df = df.iloc[:max_rows]
    logger.info(f"  {len(df)} readings, {df['Device_ID'].nunique()} devices, "
                f"{df['tau'].nunique()} logical timestamps")
    return df


def build_wearable_streams(df, sensor_type="heart_rate"):
    if sensor_type not in WEARABLE_SENSORS:
        raise ValueError(f"Unknown wearable sensor: {sensor_type}")
    col = WEARABLE_SENSORS[sensor_type]["col"]
    if col not in df.columns:
        raise ValueError(f"Column {col} missing from wearable CSV")
    collapsed = df.groupby(["tau", "Device_ID"])[col].mean().reset_index()
    collapsed = collapsed[pd.to_numeric(collapsed[col], errors="coerce").notna()]
    taus = sorted(collapsed["tau"].unique())
    aggregates, pub_counts, all_values = [], [], []
    for t in taus:
        w = collapsed[collapsed["tau"] == t]
        if w.empty:
            aggregates.append(0.0); pub_counts.append(0); continue
        aggregates.append(float(w[col].mean()))
        pub_counts.append(int(w["Device_ID"].nunique()))
        all_values.extend(w[col].tolist())
    payload_bound = float(np.ptp(all_values)) if all_values else 1.0
    if payload_bound == 0:
        payload_bound = 1.0
    logger.info(f"  Wearable [{sensor_type}]: {len(aggregates)} windows, "
                f"B={payload_bound:.2f}, avg pubs={np.mean(pub_counts):.1f}")
    return aggregates, pub_counts, payload_bound


def build_wearable_per_publisher(df, sensor_type="heart_rate"):
    col = WEARABLE_SENSORS[sensor_type]["col"]
    collapsed = df.groupby(["tau", "Device_ID"])[col].mean().reset_index()
    taus = sorted(collapsed["tau"].unique())
    devices = sorted(df["Device_ID"].unique())
    per_pub: dict[str, list[float | None]] = {d: [] for d in devices}
    all_values = []
    for t in taus:
        w = collapsed[collapsed["tau"] == t].set_index("Device_ID")[col]
        for d in devices:
            v = w.get(d)
            if v is not None and pd.notna(v):
                per_pub[d].append(float(v)); all_values.append(float(v))
            else:
                per_pub[d].append(None)
    payload_bound = float(np.ptp(all_values)) if all_values else 1.0
    if payload_bound == 0:
        payload_bound = 1.0
    return per_pub, payload_bound


# ═════════════════════════════════════════════════════════════════════════
#  Pune Smart City (Kaggle akshman)
# ═════════════════════════════════════════════════════════════════════════

PUNE_SENSORS = {
    "humidity":     {"cols": ["HUMIDITY"],                         "unit": "%"},
    "temperature":  {"cols": ["TEMPRATURE_MAX", "TEMPRATURE_MIN"], "unit": "C"},
    "pm10":         {"cols": ["PM10_MAX", "PM10_MIN"],             "unit": "ug/m3"},
    "pm2":          {"cols": ["PM2_MAX", "PM2_MIN"],               "unit": "ug/m3"},
    "ozone":        {"cols": ["OZONE_MAX", "OZONE_MIN"],           "unit": "ppb"},
    "co2":          {"cols": ["CO2_MAX", "CO2_MIN"],               "unit": "ppm"},
    "sound":        {"cols": ["SOUND"],                            "unit": "dB"},
    "air_pressure": {"cols": ["AIR_PRESSURE"],                     "unit": "atm"},
}


def load_pune_dataset(max_rows: int | None = None):
    logger.info(f"Loading Pune dataset from {PUNE_CSV}")
    df = pd.read_csv(PUNE_CSV, low_memory=False)
    df["Time"] = pd.to_datetime(df["LASTUPDATEDATETIME"], format="%d/%m/%y %H:%M", errors="coerce")
    df = df.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)
    if max_rows and len(df) > max_rows:
        df = df.iloc[:max_rows]
    logger.info(f"  {len(df)} readings, {df['NAME'].nunique()} stations, "
                f"{df['Time'].min()} to {df['Time'].max()}")
    return df


def _pune_metric_values(df, sensor_type):
    cols = PUNE_SENSORS[sensor_type]["cols"]
    present = [c for c in cols if c in df.columns]
    if not present:
        raise ValueError(f"Pune: no columns for sensor {sensor_type} ({cols})")
    return df[present].apply(pd.to_numeric, errors="coerce").mean(axis=1)


def build_pune_streams(df, sensor_type="pm10", window_minutes=60):
    df = df.copy()
    df["_v"] = _pune_metric_values(df, sensor_type)
    df = df.dropna(subset=["_v"])
    df["_bin"] = df["Time"].dt.floor(f"{window_minutes}min")
    binned = df.groupby(["_bin", "NAME"])["_v"].mean().reset_index()
    bins = sorted(binned["_bin"].unique())
    aggregates, pub_counts, all_values = [], [], []
    for b in bins:
        w = binned[binned["_bin"] == b]
        if w.empty:
            aggregates.append(0.0); pub_counts.append(0); continue
        aggregates.append(float(w["_v"].mean()))
        pub_counts.append(int(w["NAME"].nunique()))
        all_values.extend(w["_v"].tolist())
    payload_bound = float(np.ptp(all_values)) if all_values else 1.0
    if payload_bound == 0:
        payload_bound = 1.0
    logger.info(f"  Pune [{sensor_type}]: {len(aggregates)} windows "
                f"({window_minutes}min), B={payload_bound:.2f}, "
                f"avg pubs={np.mean(pub_counts):.1f}")
    return aggregates, pub_counts, payload_bound


def build_pune_per_publisher(df, sensor_type="pm10", window_minutes=60):
    df = df.copy()
    df["_v"] = _pune_metric_values(df, sensor_type)
    df = df.dropna(subset=["_v"])
    df["_bin"] = df["Time"].dt.floor(f"{window_minutes}min")
    binned = df.groupby(["_bin", "NAME"])["_v"].mean().reset_index()
    stations = sorted(df["NAME"].unique())
    bins = sorted(binned["_bin"].unique())
    per_pub: dict[str, list[float | None]] = {s: [] for s in stations}
    all_values = []
    for b in bins:
        w = binned[binned["_bin"] == b].set_index("NAME")["_v"]
        for s in stations:
            v = w.get(s)
            if v is not None and pd.notna(v):
                per_pub[s].append(float(v)); all_values.append(float(v))
            else:
                per_pub[s].append(None)
    payload_bound = float(np.ptp(all_values)) if all_values else 1.0
    if payload_bound == 0:
        payload_bound = 1.0
    return per_pub, payload_bound


# ═════════════════════════════════════════════════════════════════════════
#  Smart Mobility (Kaggle ziya07)
# ═════════════════════════════════════════════════════════════════════════

MOBILITY_SENSORS = {
    "vehicle_count":  {"col": "Vehicle_Count",         "unit": "count"},
    "traffic_speed":  {"col": "Traffic_Speed_kmh",     "unit": "km/h"},
    "road_occupancy": {"col": "Road_Occupancy_%",      "unit": "%"},
    "emission":       {"col": "Emission_Levels_g_km",  "unit": "g/km"},
    "energy":         {"col": "Energy_Consumption_L_h","unit": "L/h"},
}
MOBILITY_GRID = 4


def load_mobility_dataset(max_rows: int | None = None):
    logger.info(f"Loading mobility dataset from {MOBILITY_CSV}")
    df = pd.read_csv(MOBILITY_CSV, low_memory=False)
    df["Time"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)
    lat_bins = pd.qcut(df["Latitude"], MOBILITY_GRID, labels=False, duplicates="drop")
    lon_bins = pd.qcut(df["Longitude"], MOBILITY_GRID, labels=False, duplicates="drop")
    df["cell_id"] = (
        "cell_" + lat_bins.astype("Int64").astype(str)
        + "_" + lon_bins.astype("Int64").astype(str)
    )
    if max_rows and len(df) > max_rows:
        df = df.iloc[:max_rows]
    logger.info(f"  {len(df)} readings, {df['cell_id'].nunique()} virtual cells, "
                f"{df['Time'].min()} to {df['Time'].max()}")
    return df


def _mobility_binned(df, sensor_type, window_minutes):
    if sensor_type not in MOBILITY_SENSORS:
        raise ValueError(f"Unknown mobility sensor: {sensor_type}")
    col = MOBILITY_SENSORS[sensor_type]["col"]
    if col not in df.columns:
        raise ValueError(f"Column {col} missing from mobility CSV")
    df = df.copy()
    df["_v"] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["_v"])
    df["_bin"] = df["Time"].dt.floor(f"{window_minutes}min")
    return df, df.groupby(["_bin", "cell_id"])["_v"].mean().reset_index()


def build_mobility_streams(df, sensor_type="traffic_speed", window_minutes=30):
    df, binned = _mobility_binned(df, sensor_type, window_minutes)
    bins = sorted(binned["_bin"].unique())
    aggregates, pub_counts, all_values = [], [], []
    for b in bins:
        w = binned[binned["_bin"] == b]
        if w.empty:
            aggregates.append(0.0); pub_counts.append(0); continue
        aggregates.append(float(w["_v"].mean()))
        pub_counts.append(int(w["cell_id"].nunique()))
        all_values.extend(w["_v"].tolist())
    payload_bound = float(np.ptp(all_values)) if all_values else 1.0
    if payload_bound == 0:
        payload_bound = 1.0
    logger.info(f"  Mobility [{sensor_type}]: {len(aggregates)} windows "
                f"({window_minutes}min), B={payload_bound:.2f}, "
                f"avg pubs={np.mean(pub_counts):.1f}")
    return aggregates, pub_counts, payload_bound


def build_mobility_per_publisher(df, sensor_type="traffic_speed", window_minutes=30):
    df, binned = _mobility_binned(df, sensor_type, window_minutes)
    cells = sorted(df["cell_id"].unique())
    bins = sorted(binned["_bin"].unique())
    per_pub: dict[str, list[float | None]] = {c: [] for c in cells}
    all_values = []
    for b in bins:
        w = binned[binned["_bin"] == b].set_index("cell_id")["_v"]
        for c in cells:
            v = w.get(c)
            if v is not None and pd.notna(v):
                per_pub[c].append(float(v)); all_values.append(float(v))
            else:
                per_pub[c].append(None)
    payload_bound = float(np.ptp(all_values)) if all_values else 1.0
    if payload_bound == 0:
        payload_bound = 1.0
    return per_pub, payload_bound


# ═════════════════════════════════════════════════════════════════════════
#  Smart Manufacturing (Kaggle programmer3)
# ═════════════════════════════════════════════════════════════════════════

MANUFACTURING_SENSORS = {
    "temperature":   {"match": "Temperature",       "unit": "C"},
    "machine_speed": {"match": "Machine Speed",     "unit": "RPM"},
    "quality":       {"match": "Production Quality","unit": "score"},
    "vibration":     {"match": "Vibration Level",   "unit": "mm/s"},
    "energy":        {"match": "Energy Consumption","unit": "kWh"},
}


def _manufacturing_col(df, sensor_type):
    needle = MANUFACTURING_SENSORS[sensor_type]["match"]
    for c in df.columns:
        if c.startswith(needle):
            return c
    raise ValueError(f"Manufacturing: no column matches {needle!r}")


def load_manufacturing_dataset(max_rows: int | None = None):
    logger.info(f"Loading manufacturing dataset from {MANUFACTURING_CSV}")
    df = pd.read_csv(MANUFACTURING_CSV, low_memory=False)
    df["Time"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)
    if max_rows and len(df) > max_rows:
        df = df.iloc[:max_rows]
    logger.info(f"  {len(df)} minute-rows, {df['Time'].min()} to {df['Time'].max()}")
    return df


def _manufacturing_preprocessed(df, sensor_type, window_minutes):
    col = _manufacturing_col(df, sensor_type)
    df = df.copy()
    df["_v"] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["_v"])
    df["_bin"] = df["Time"].dt.floor(f"{window_minutes}min")
    df["_sub"] = df.groupby("_bin").cumcount().astype(str).radd("sub_")
    return df


def build_manufacturing_streams(df, sensor_type="temperature", window_minutes=10):
    df = _manufacturing_preprocessed(df, sensor_type, window_minutes)
    bins = sorted(df["_bin"].unique())
    aggregates, pub_counts, all_values = [], [], []
    for b in bins:
        w = df[df["_bin"] == b]
        if w.empty:
            aggregates.append(0.0); pub_counts.append(0); continue
        aggregates.append(float(w["_v"].mean()))
        pub_counts.append(int(w["_sub"].nunique()))
        all_values.extend(w["_v"].tolist())
    payload_bound = float(np.ptp(all_values)) if all_values else 1.0
    if payload_bound == 0:
        payload_bound = 1.0
    logger.info(f"  Manufacturing [{sensor_type}]: {len(aggregates)} windows "
                f"({window_minutes}min), B={payload_bound:.2f}, "
                f"avg pubs={np.mean(pub_counts):.1f}")
    return aggregates, pub_counts, payload_bound


def build_manufacturing_per_publisher(df, sensor_type="temperature", window_minutes=10):
    df = _manufacturing_preprocessed(df, sensor_type, window_minutes)
    subs = [f"sub_{i}" for i in range(window_minutes)]
    bins = sorted(df["_bin"].unique())
    per_pub: dict[str, list[float | None]] = {s: [] for s in subs}
    all_values = []
    for b in bins:
        w = df[df["_bin"] == b].set_index("_sub")["_v"]
        for s in subs:
            v = w.get(s)
            if v is not None and pd.notna(v):
                per_pub[s].append(float(v)); all_values.append(float(v))
            else:
                per_pub[s].append(None)
    payload_bound = float(np.ptp(all_values)) if all_values else 1.0
    if payload_bound == 0:
        payload_bound = 1.0
    return per_pub, payload_bound


# ═════════════════════════════════════════════════════════════════════════
#  Dataset registry
# ═════════════════════════════════════════════════════════════════════════
#
# A registry entry binds a dataset to its loader, per-sensor stream builders,
# clamp metadata, and publisher/subscriber topic hierarchy.  Every field is
# data (no experiment logic) so new datasets can be added by appending here.

DATASETS: dict[str, dict] = {
    "energy": {
        "label": "Smart Building Energy (MCEC-Thai)",
        "sensors": ["power_kw", "voltage", "current"],
        "loader": load_energy_dataset,
        "loader_row_arg": "max_timestamps",
        "copy_load_per_sensor": True,
        "build_streams": lambda d, s: build_energy_streams(d, s, window_minutes=5),
        "build_per_pub": lambda d, s: build_energy_per_publisher(d, s, window_minutes=5),
        # Option A operator-declared [a_p, b_p] from residential breaker datasheets.
        "static_clamps": {
            "power_kw": (0.0, 50.0),
            "voltage": (180.0, 260.0),
            "current": (0.0, 100.0),
        },
        # Public fallback M for Option B: loose physical ceiling used only as a
        # data-independent bound on the Laplace noise of the clamp calibration.
        "fallback_M": {"power_kw": 100.0, "voltage": 300.0, "current": 200.0},
        # Normative MQTT topic hierarchy: energy/{site}/{circuit}/{metric}.
        "topic_root": "energy/building01",
        "publisher_topic": lambda pub_id, sensor: f"energy/building01/{pub_id}/{sensor}",
        "subscriber_filters": [
            "energy/building01/#",           # whole-building dashboard
            "energy/building01/+/power_kw",  # load monitor across circuits
            "energy/building01/CT1/#",       # single circuit (main service)
        ],
    },
    "traffic": {
        "label": "Smart City Intersection (Colorado Springs)",
        "sensors": ["speed", "object_count"],
        "loader": load_traffic_dataset,
        "loader_row_arg": "max_rows_per_file",
        "copy_load_per_sensor": False,
        "build_streams": lambda d, s: build_traffic_streams(d, s, window_seconds=10),
        "build_per_pub": lambda d, s: build_traffic_per_publisher(d, s, window_seconds=10),
        "static_clamps": {
            "speed":        (0.0, 50.0),
            "object_count": (0.0, 50.0),
        },
        "fallback_M": {"speed": 100.0, "object_count": 100.0},
        # Hierarchy: traffic/{intersection}/{sensor-class}/{sensor-id}/{metric}.
        "topic_root": "traffic/intersection01",
        "publisher_topic": lambda pub_id, sensor: (
            f"traffic/intersection01/{'radar' if 'RADAR' in pub_id else 'lidar'}/"
            f"{pub_id}/{sensor}"
        ),
        "subscriber_filters": [
            "traffic/intersection01/#",                       # city ops dashboard
            "traffic/intersection01/+/+/speed",               # speed across sensors
            "traffic/intersection01/radar/#",                 # radar-only analytics
            "traffic/intersection01/+/EVO_RADAR_1/#",         # one sensor's full feed
        ],
    },
    "wearable": {
        "label": "Wearable IoT Healthcare (Kaggle dcsavinod)",
        "sensors": ["heart_rate", "steps", "temperature", "calories_burned"],
        "loader": load_wearable_dataset,
        "loader_row_arg": "max_rows",
        "copy_load_per_sensor": False,
        "build_streams": lambda d, s: build_wearable_streams(d, s),
        "build_per_pub": lambda d, s: build_wearable_per_publisher(d, s),
        "static_clamps": {
            "heart_rate":      (30.0, 220.0),
            "steps":           (0.0, 5000.0),
            "temperature":     (15.0, 45.0),
            "calories_burned": (0.0, 50.0),
        },
        "fallback_M": {
            "heart_rate": 250.0, "steps": 10000.0,
            "temperature": 60.0, "calories_burned": 100.0,
        },
        # Hierarchy: health/{hospital}/{device}/{metric}, following PSMark-HC's
        # smart-healthcare deployment (PerCom): continuously-monitored ICU beds
        # spread across 5 regional hospitals (one edge server each).  We map
        # each wearable Device_i to a regional hospital (hospital01..hospital05)
        # so the hierarchy walk pools a hospital's devices before falling back
        # to the cross-hospital root.
        "topic_root": "health",
        "publisher_topic": lambda pub_id, sensor: (
            f"health/hospital{(int(''.join(filter(str.isdigit, str(pub_id))) or 0) % 5) + 1:02d}/"
            f"{pub_id}/{sensor}"
        ),
        "subscriber_filters": [
            "health/#",                          # region-wide RPM dashboard
            "health/+/+/heart_rate",             # cardiac alerts across hospitals
            "health/hospital01/#",               # one hospital's feed
            "health/+/Device_5/#",               # one patient's full telemetry
            "health/+/+/steps",                  # activity analytics
        ],
    },
    "pune": {
        "label": "Pune Smart City Air Quality (Kaggle akshman)",
        "sensors": ["pm10", "pm2", "humidity", "sound", "ozone"],
        "loader": load_pune_dataset,
        "loader_row_arg": "max_rows",
        "copy_load_per_sensor": False,
        "build_streams": lambda d, s: build_pune_streams(d, s, window_minutes=60),
        "build_per_pub": lambda d, s: build_pune_per_publisher(d, s, window_minutes=60),
        "static_clamps": {
            "pm10":     (0.0, 1000.0),
            "pm2":      (0.0, 500.0),
            "humidity": (0.0, 100.0),
            "sound":    (20.0, 140.0),
            "ozone":    (0.0, 500.0),
        },
        "fallback_M": {
            "pm10": 2000.0, "pm2": 1000.0, "humidity": 100.0,
            "sound": 200.0, "ozone": 1000.0,
        },
        # Hierarchy: air_quality/{city}/{station}/{pollutant}.  Station names in
        # the CSV carry spaces/underscores; we slugify them in the topic.
        "topic_root": "air_quality/pune",
        "publisher_topic": lambda pub_id, sensor: (
            f"air_quality/pune/{pub_id.replace(' ', '_')}/{sensor}"
        ),
        "subscriber_filters": [
            "air_quality/pune/#",                  # municipal dashboard
            "air_quality/pune/+/pm10",             # city-wide PM10 alerts
            "air_quality/pune/Hadapsar_Gadital_01/#",  # one station's feed
        ],
    },
    "mobility": {
        "label": "Smart Mobility Traffic (Kaggle ziya07)",
        "sensors": ["traffic_speed", "vehicle_count", "road_occupancy", "emission"],
        "loader": load_mobility_dataset,
        "loader_row_arg": "max_rows",
        "copy_load_per_sensor": False,
        "build_streams": lambda d, s: build_mobility_streams(d, s, window_minutes=30),
        "build_per_pub": lambda d, s: build_mobility_per_publisher(d, s, window_minutes=30),
        "static_clamps": {
            "traffic_speed":  (0.0, 120.0),
            "vehicle_count":  (0.0, 500.0),
            "road_occupancy": (0.0, 100.0),
            "emission":       (0.0, 800.0),
        },
        "fallback_M": {
            "traffic_speed": 200.0, "vehicle_count": 1000.0,
            "road_occupancy": 100.0, "emission": 2000.0,
        },
        # Hierarchy: mobility/{city}/{zone}/{cell}/{metric}.  zone is the
        # coarse quadrant (NW/NE/SW/SE) and cell is the 4x4 grid label.
        "topic_root": "mobility/nyc",
        "publisher_topic": lambda pub_id, sensor: (
            # pub_id = "cell_<lat>_<lon>".  Map lat 0/1 -> south, 2/3 -> north
            # and lon 0/1 -> west, 2/3 -> east for a coarser 2-level scope.
            f"mobility/nyc/"
            f"{'N' if int(pub_id.split('_')[1]) >= 2 else 'S'}"
            f"{'E' if int(pub_id.split('_')[2]) >= 2 else 'W'}/"
            f"{pub_id}/{sensor}"
        ),
        "subscriber_filters": [
            "mobility/nyc/#",                        # city-wide flow
            "mobility/nyc/+/+/traffic_speed",        # speed everywhere
            "mobility/nyc/NE/#",                     # one quadrant
            "mobility/nyc/+/+/emission",             # environmental reporting
        ],
    },
    "manufacturing": {
        "label": "Smart Manufacturing Process (Kaggle programmer3)",
        "sensors": ["temperature", "machine_speed", "quality", "vibration", "energy"],
        "loader": load_manufacturing_dataset,
        "loader_row_arg": "max_rows",
        "copy_load_per_sensor": False,
        "build_streams": lambda d, s: build_manufacturing_streams(d, s, window_minutes=10),
        "build_per_pub": lambda d, s: build_manufacturing_per_publisher(d, s, window_minutes=10),
        "static_clamps": {
            "temperature":   (0.0, 200.0),
            "machine_speed": (0.0, 5000.0),
            "quality":       (0.0, 10.0),
            "vibration":     (0.0, 1.0),
            "energy":        (0.0, 10.0),
        },
        "fallback_M": {
            "temperature": 500.0, "machine_speed": 10000.0,
            "quality": 20.0, "vibration": 5.0, "energy": 50.0,
        },
        # Hierarchy: factory/{line}/{station}/{machine}/{sensor}, following
        # PSMark-F's smart-factory deployment (PerCom): an assembly line whose
        # six instrumented machines group into stations (sorting, processing =
        # oven+milling, warehouse, robotics = gripper+AMR).  The source dataset
        # is a single machine, so we map the sub_i sub-publishers to distinct
        # virtual machines and assign each to a PSMark station, giving the
        # hierarchy walk (Algorithm 1) a realistic machine -> station -> line
        # -> factory ladder.
        "topic_root": "factory/line1",
        "publisher_topic": lambda pub_id, sensor: (
            f"factory/line1/"
            f"{['sorting','processing','warehouse','robotics'][int(pub_id.split('_')[1]) % 4]}/"
            f"{pub_id.replace('sub_', 'machine')}/{sensor}"
        ),
        "subscriber_filters": [
            "factory/line1/#",                          # line-level dashboard
            "factory/line1/+/+/vibration",              # predictive-maintenance
            "factory/line1/processing/#",               # one station's feed
            "factory/line1/+/machine01/#",              # single-machine feed
            "factory/line1/+/+/quality",                # QC rollup
        ],
    },
}


# ═════════════════════════════════════════════════════════════════════════
#  Loading / building public API
# ═════════════════════════════════════════════════════════════════════════

def load_dataset_object(name: str, max_rows: int | None = None):
    """Invoke the registered loader for `name`, wiring the row-cap argument."""
    spec = DATASETS[name]
    kwargs = {}
    if max_rows is not None and spec["loader_row_arg"]:
        kwargs[spec["loader_row_arg"]] = max_rows
    return spec["loader"](**kwargs)


def build_sensor_streams(
    name: str,
    obj,
    sensors: list[str] | None = None,
) -> tuple[dict, dict]:
    """Build aggregate streams and per-publisher tables for every sensor.

    Returns (streams, per_pubs):
      streams[sensor]  -> (aggregates, pub_counts, payload_bound)
      per_pubs[sensor] -> (per_publisher_dict, payload_bound)

    Sensors that fail to build, or that yield < 10 windows / B ~ 0, are
    dropped from `streams`.  `per_pubs` is kept even for short streams because
    intro figures and per-publisher experiments want every available series.
    """
    spec = DATASETS[name]
    sensors = sensors or spec["sensors"]
    streams: dict = {}
    per_pubs: dict = {}
    for sensor in sensors:
        arg = obj.copy() if spec["copy_load_per_sensor"] else obj
        try:
            agg, pub, B = spec["build_streams"](arg, sensor)
            if len(agg) > 10 and B > 0.01:
                streams[sensor] = (agg, pub, B)
        except Exception as e:
            logger.warning(f"  Skipping {name}/{sensor} (streams): {e}")
        try:
            arg2 = obj.copy() if spec["copy_load_per_sensor"] else obj
            pp, Bpp = spec["build_per_pub"](arg2, sensor)
            per_pubs[sensor] = (pp, Bpp)
        except Exception as e:
            logger.warning(f"  Skipping {name}/{sensor} (per-pub): {e}")
    return streams, per_pubs


def build_topic_manifest(name: str, per_pubs: dict) -> pd.DataFrame:
    """Render the full topic hierarchy as a concrete publisher x sensor table.

    Each row is one publisher x sensor -> MQTT topic mapping so the operator
    can see every topic the mechanism will publish on.
    """
    spec = DATASETS[name]
    topic_of = spec.get("publisher_topic")
    if topic_of is None:
        return pd.DataFrame()
    rows = []
    for sensor, (pp, _B) in per_pubs.items():
        for pub_id in pp.keys():
            try:
                topic = topic_of(pub_id, sensor)
            except Exception:
                topic = f"{spec.get('topic_root','unknown')}/{pub_id}/{sensor}"
            rows.append({
                "dataset": name,
                "publisher_id": pub_id,
                "sensor": sensor,
                "raw_topic": topic,
                "protected_topic": topic.replace(spec.get("topic_root", ""), "").lstrip("/"),
            })
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════
#  Definition 3.2 clamp options (Option A static / Option B DP-released)
# ═════════════════════════════════════════════════════════════════════════
#
# Option A ("static"): operator-declared [a_p, b_p] from datasheets / schema.
#   No budget charged; clamp is a public constant.
# Option B ("dp_released"): per-publisher min/max released under Laplace
#   noise scaled by a public fallback M and a separate budget eps_clip.
#   After release the noisy (a_hat, b_hat) are treated as public constants
#   (DP post-processing), and the clamp is applied to every payload.  Total
#   DP cost composes to eps + eps_clip; the effective per-release budget
#   that enters the w-event accounting is (eps - eps_clip) when we subtract.

def _clamp_static(per_pub, static_range):
    lo, hi = static_range
    out = {}
    for p, series in per_pub.items():
        out[p] = [min(max(v, lo), hi) if v is not None else None for v in series]
    R = float(hi - lo)
    meta = {"mode": "static", "a_global": float(lo), "b_global": float(hi),
            "R": R, "eps_clip": 0.0}
    return out, R, meta


def _clamp_dp_released(per_pub, fallback_M, eps_clip, seed=0):
    """Option B: per-publisher DP-released min/max with Laplace(M/eps_clip)."""
    rng = np.random.default_rng(seed)
    M = float(fallback_M)
    out = {}
    per_pub_clamps = {}
    max_width = 0.0
    for p, series in per_pub.items():
        vals = [v for v in series if v is not None]
        if not vals:
            # Publisher never emitted: it contributes nothing to any release, so
            # it must NOT inflate the global range R.  Give it the widest public
            # interval for clamping consistency but EXCLUDE it from max_width.
            out[p] = series
            per_pub_clamps[p] = (-M, M)
            continue
        true_min = float(min(vals)); true_max = float(max(vals))
        a_hat = true_min + rng.laplace(scale=M / eps_clip)
        b_hat = true_max + rng.laplace(scale=M / eps_clip)
        a_hat = max(-M, min(a_hat, M))
        b_hat = max(a_hat, min(b_hat, M))
        per_pub_clamps[p] = (a_hat, b_hat)
        max_width = max(max_width, b_hat - a_hat)
        out[p] = [min(max(v, a_hat), b_hat) if v is not None else None for v in series]
    meta = {"mode": "dp_released", "M": M, "eps_clip": float(eps_clip),
            "per_pub_clamps": per_pub_clamps, "R": float(max_width)}
    return out, float(max_width), meta


def apply_clamp_option(
    per_pub: dict[str, list[float | None]],
    sensor_type: str,
    dataset_spec: dict,
    mode: str,
    eps_clip: float = 0.1,
    seed: int = 0,
) -> tuple[dict, float, dict]:
    """Apply the operator's clamp choice from Definition 3.2."""
    if mode == "static":
        static = dataset_spec["static_clamps"].get(sensor_type)
        if static is None:
            raise ValueError(f"No static clamp configured for {sensor_type}")
        return _clamp_static(per_pub, static)
    if mode == "dp_released":
        M = dataset_spec["fallback_M"].get(sensor_type)
        if M is None:
            raise ValueError(f"No fallback_M configured for {sensor_type}")
        return _clamp_dp_released(per_pub, M, eps_clip, seed=seed)
    raise ValueError(f"Unknown clamp mode: {mode}")


def aggregate_from_per_pub(per_pub: dict[str, list[float | None]]):
    """Re-derive (aggregates, pub_counts) from a per-publisher table.

    Used after applying clamp options so the downstream sweep sees the clamped
    values rather than the raw ones.
    """
    T = len(next(iter(per_pub.values())))
    agg, cnt = [], []
    for tau in range(T):
        vals = [per_pub[p][tau] for p in per_pub if per_pub[p][tau] is not None]
        agg.append(float(np.mean(vals)) if vals else 0.0)
        cnt.append(len(vals))
    return agg, cnt


def reclamp_dataset(
    streams: dict,
    per_pubs: dict,
    dataset_spec: dict,
    clamp_mode: str,
    eps_clip: float,
    seed: int = 0,
) -> tuple[dict, dict, dict]:
    """Apply the chosen clamp option to every sensor of a dataset.

    For each sensor:
      1. ``apply_clamp_option`` yields clamped per_pub + updated payload bound R.
      2. Re-derive (aggregates, pub_counts) from the clamped per_pub so the
         downstream sweep sees clamped values.

    Returns (clamped_streams, clamped_per_pubs, clamp_meta_per_sensor).
    Sensors where the clamp fails are omitted from the result and warned about.
    """
    clamped_streams: dict = {}
    clamped_per_pubs: dict = {}
    meta: dict = {}
    for sensor, (pp, _B_raw) in per_pubs.items():
        try:
            new_pp, R, m = apply_clamp_option(
                pp, sensor, dataset_spec, clamp_mode,
                eps_clip=eps_clip, seed=seed + _stable_hash(sensor),
            )
        except ValueError as e:
            logger.warning(f"    clamp {clamp_mode} failed for {sensor}: {e}")
            continue
        meta[sensor] = m
        clamped_per_pubs[sensor] = (new_pp, R)
        new_agg, new_cnt = aggregate_from_per_pub(new_pp)
        if len(new_agg) > 10 and R > 0.0:
            clamped_streams[sensor] = (new_agg, new_cnt, R)
    return clamped_streams, clamped_per_pubs, meta


# ═════════════════════════════════════════════════════════════════════════
#  End-to-end prepared dataset
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class PreparedDataset:
    """Everything a downstream experiment needs about a dataset.

    Fields:
      name, spec:        registry key and the entry dict from DATASETS.
      streams:           {sensor -> (aggregates, pub_counts, payload_bound)} after clamp.
      per_pubs:          {sensor -> (per_pub_dict, payload_bound)} after clamp.
      clamp_meta:        {sensor -> clamp metadata dict} (mode, R, etc).
      clamp_mode:        "static" or "dp_released".
      raw_streams:       pre-clamp streams (occasionally useful for diagnostics).
      raw_per_pubs:      pre-clamp per-publisher tables.
    """
    name: str
    spec: dict
    streams: dict
    per_pubs: dict
    clamp_meta: dict
    clamp_mode: str
    raw_streams: dict = field(default_factory=dict)
    raw_per_pubs: dict = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.streams

    def has_per_pub(self, sensor: str) -> bool:
        return sensor in self.per_pubs


# Process-local cache: the same (dataset, clamp_mode, ...) is prepared by
# several experiments in one process (e.g. F/G/H + grid within one cluster
# shard).  Loading + clamping the CSV once and reusing it is a large speedup;
# the cache is keyed on every argument that affects the result so it is safe.
_PREPARE_CACHE: dict = {}


def prepare_dataset(
    name: str,
    *,
    clamp_mode: str = "static",
    eps_clip: float = 0.1,
    seed: int = 0,
    max_rows: int | None = None,
    sensors: list[str] | None = None,
) -> PreparedDataset | None:
    """Load, build, and clamp one dataset end-to-end.

    Returns ``None`` if the dataset source cannot be opened.  Returns a
    ``PreparedDataset`` with ``is_empty == True`` if streams all failed the
    clamp step — callers should check that rather than silently proceeding.

    Result is memoized per process on the full argument tuple (see
    ``_PREPARE_CACHE``); deterministic in all args including ``seed``, so the
    cache never changes behaviour, only avoids re-reading the CSV.
    """
    cache_key = (name, clamp_mode, round(float(eps_clip), 9), int(seed),
                 max_rows, tuple(sensors) if sensors is not None else None)
    if cache_key in _PREPARE_CACHE:
        return _PREPARE_CACHE[cache_key]
    spec = DATASETS[name]
    try:
        obj = load_dataset_object(name, max_rows=max_rows)
    except FileNotFoundError as e:
        logger.warning(f"Skipping {name}: {e}")
        _PREPARE_CACHE[cache_key] = None
        return None

    raw_streams, raw_per_pubs = build_sensor_streams(name, obj, sensors=sensors)
    streams, per_pubs, meta = reclamp_dataset(
        raw_streams, raw_per_pubs, spec, clamp_mode, eps_clip, seed=seed,
    )
    prepared = PreparedDataset(
        name=name, spec=spec,
        streams=streams, per_pubs=per_pubs, clamp_meta=meta,
        clamp_mode=clamp_mode,
        raw_streams=raw_streams, raw_per_pubs=raw_per_pubs,
    )
    _PREPARE_CACHE[cache_key] = prepared
    return prepared
