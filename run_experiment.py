#!/usr/bin/env python3
"""
End-to-end experimental pipeline for clamped w-event DP with P-allocation,
run exclusively on real-world public datasets (no synthetic data).

For every dataset registered in `run_real_data_experiment.DATASETS` this script:
  1. Runs the full parameter sweep (all strategies x P x eps x w x sensor).
  2. Produces the paper's four intro figures (two extremes + U-shape + KL bar).
  3. Runs Section 5.7 two-stage hyperparameter tuning across ALL strategies
     (offline grid search over (P, Delta_t, A) + online greedy scope walk).
  4. Runs the n-weighted spotlight, collusion experiment, K_ext sweep, and
     per-dataset Figure-1 reproduction.

Cross-dataset, it also:
  5. Combines every per-dataset sweep into one table + comparison plot.
  6. Averages the per-dataset U-shape into a single Figure-1 across datasets.
  7. Picks the best (strategy, P, Delta_t) per dataset from the tuning grids.

Every experiment writes raw data to CSV alongside its PNG, so downstream
analysis never has to re-run the sweep.  Output layout:

  results/
    <dataset>/
      sweep/    <-- main sweep CSV + plots
      intro/    <-- the four intro figures
      tuning/   <-- offline (P, dt, A) grid
      extras/   <-- n-weighted, collusion, K_ext
    cross_dataset/
      combined_*.csv, figure1_all_datasets.csv/.png, ...

Usage:
  python run_experiment.py                 # full pipeline, every dataset
  python run_experiment.py --quick         # reduced grid for testing
  python run_experiment.py --dataset energy
  python run_experiment.py --tune-only     # just the hyperparameter tuning
"""

from __future__ import annotations

import argparse
import itertools
import logging
import os
import sys
from collections import defaultdict
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dp_engine import (
    BudgetStrategy,
    PrivacyConfig,
    StreamState,
    attribution_advantage,
    compute_global_utility,
    compute_kl_divergence,
    compute_utility_metrics,
    compute_windowed_kl_divergence,
    is_p_gated,
)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

ALL_STRATEGIES = [
    "uniform",
    "sample",
    "budget_distribution",
    "budget_absorption",
    "p_gated_ba",
    "n_weighted",
]

# ═════════════════════════════════════════════════════════════════════════
#  Data loaders for every real-world dataset we evaluate on
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


# ── Energy dataset (MCEC-Thai) ────────────────────────────────────────────

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


# ── Traffic dataset (Colorado Springs) ───────────────────────────────────

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


def build_traffic_streams(sensors, metric="speed", window_seconds=10):
    all_times = []
    for df in sensors.values():
        all_times.extend(df["Time"].values)
    t_min = pd.Timestamp(min(all_times))
    t_max = pd.Timestamp(max(all_times))
    freq = pd.Timedelta(seconds=window_seconds)
    bins = pd.date_range(start=t_min, end=t_max + freq, freq=freq)
    aggregates, pub_counts, all_values = [], [], []
    for i in range(len(bins) - 1):
        w_start, w_end = bins[i], bins[i + 1]
        sensor_values = []
        for sensor_name, df in sensors.items():
            window_df = df[(df["Time"] >= w_start) & (df["Time"] < w_end)]
            if window_df.empty:
                continue
            if metric == "object_count":
                val = float(window_df["ObjectId"].nunique())
            elif metric == "speed":
                val = float(window_df["Speed"].mean())
            elif metric == "position_x":
                val = float(window_df["PositionX"].mean())
            elif metric == "heading":
                col = "HeadingDeg_DERIVED" if "HeadingDeg_DERIVED" in window_df.columns else "HeadingDeg"
                if col not in window_df.columns:
                    continue
                val = float(window_df[col].mean())
            else:
                raise ValueError(f"Unknown traffic metric: {metric}")
            if np.isfinite(val):
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
    all_times = []
    for df in sensors.values():
        all_times.extend(df["Time"].values)
    t_min = pd.Timestamp(min(all_times))
    t_max = pd.Timestamp(max(all_times))
    freq = pd.Timedelta(seconds=window_seconds)
    bins = pd.date_range(start=t_min, end=t_max + freq, freq=freq)
    per_pub: dict[str, list[float | None]] = {name: [] for name in sensors}
    all_values = []
    for i in range(len(bins) - 1):
        w_start, w_end = bins[i], bins[i + 1]
        for sensor_name, df in sensors.items():
            window_df = df[(df["Time"] >= w_start) & (df["Time"] < w_end)]
            if window_df.empty:
                per_pub[sensor_name].append(None); continue
            if metric == "object_count":
                val = float(window_df["ObjectId"].nunique())
            elif metric == "speed":
                val = float(window_df["Speed"].mean())
            elif metric == "position_x":
                val = float(window_df["PositionX"].mean())
            else:
                per_pub[sensor_name].append(None); continue
            if np.isfinite(val):
                per_pub[sensor_name].append(val); all_values.append(val)
            else:
                per_pub[sensor_name].append(None)
    payload_bound = float(np.ptp(all_values)) if all_values else 1.0
    if payload_bound == 0:
        payload_bound = 1.0
    return per_pub, payload_bound


# ── Wearable IoT Healthcare (Kaggle dcsavinod) ────────────────────────────

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


# ── Pune Smart City (Kaggle akshman) ─────────────────────────────────────

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


# ── Smart Mobility (Kaggle ziya07) ────────────────────────────────────────

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


def build_mobility_streams(df, sensor_type="traffic_speed", window_minutes=30):
    if sensor_type not in MOBILITY_SENSORS:
        raise ValueError(f"Unknown mobility sensor: {sensor_type}")
    col = MOBILITY_SENSORS[sensor_type]["col"]
    if col not in df.columns:
        raise ValueError(f"Column {col} missing from mobility CSV")
    df = df.copy()
    df["_v"] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["_v"])
    df["_bin"] = df["Time"].dt.floor(f"{window_minutes}min")
    binned = df.groupby(["_bin", "cell_id"])["_v"].mean().reset_index()
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
    col = MOBILITY_SENSORS[sensor_type]["col"]
    df = df.copy()
    df["_v"] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["_v"])
    df["_bin"] = df["Time"].dt.floor(f"{window_minutes}min")
    binned = df.groupby(["_bin", "cell_id"])["_v"].mean().reset_index()
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


# ── Smart Manufacturing (Kaggle programmer3) ──────────────────────────────

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


def build_manufacturing_streams(df, sensor_type="temperature", window_minutes=10):
    col = _manufacturing_col(df, sensor_type)
    df = df.copy()
    df["_v"] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["_v"])
    df["_bin"] = df["Time"].dt.floor(f"{window_minutes}min")
    df["_sub"] = df.groupby("_bin").cumcount().astype(str).radd("sub_")
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
    col = _manufacturing_col(df, sensor_type)
    df = df.copy()
    df["_v"] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["_v"])
    df["_bin"] = df["Time"].dt.floor(f"{window_minutes}min")
    df["_sub"] = df.groupby("_bin").cumcount().astype(str).radd("sub_")
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

DATASETS = {
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
        # Hierarchy: health/{site}/{device}/{metric}.
        "topic_root": "health/clinic01",
        "publisher_topic": lambda pub_id, sensor: f"health/clinic01/{pub_id}/{sensor}",
        "subscriber_filters": [
            "health/clinic01/#",                 # hospital-wide RPM dashboard
            "health/clinic01/+/heart_rate",      # cardiac alerts across all patients
            "health/clinic01/Device_5/#",        # one patient's full telemetry
            "health/clinic01/+/steps",           # activity analytics
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
        # Hierarchy: factory/{line}/{machine}/{sensor}.  The source dataset
        # is a single machine, so we map the sub_i sub-publishers to distinct
        # virtual machines m01..m10 on the same line.
        "topic_root": "factory/line1",
        "publisher_topic": lambda pub_id, sensor: (
            f"factory/line1/{pub_id.replace('sub_', 'machine')}/{sensor}"
        ),
        "subscriber_filters": [
            "factory/line1/#",                      # line-level dashboard
            "factory/line1/+/vibration",            # predictive-maintenance
            "factory/line1/machine01/#",            # single-machine feed
            "factory/line1/+/quality",              # QC rollup
        ],
    },
}


def load_dataset_object(name: str, max_rows: int | None = None):
    spec = DATASETS[name]
    kwargs = {}
    if max_rows is not None and spec["loader_row_arg"]:
        kwargs[spec["loader_row_arg"]] = max_rows
    return spec["loader"](**kwargs)


def build_sensor_streams(name: str, obj, sensors: list[str] | None = None):
    spec = DATASETS[name]
    sensors = sensors or spec["sensors"]
    streams, per_pubs = {}, {}
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
    """Render the full topic hierarchy for this dataset as a concrete table.

    Each row is one publisher × sensor → topic mapping.  The operator can
    read this CSV to see every MQTT topic the mechanism will publish on.
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
#  Clamp modes (Definition 3.2): Option A static + Option B DP-released
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
            out[p] = series
            per_pub_clamps[p] = (-M, M)
            max_width = max(max_width, 2 * M)
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


def _aggregate_from_per_pub(per_pub: dict[str, list[float | None]]):
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


# ═════════════════════════════════════════════════════════════════════════
#  Core DP runner
# ═════════════════════════════════════════════════════════════════════════

def run_dp_on_stream(
    aggregates: list[float],
    pub_counts: list[int],
    epsilon: float,
    window_size: int,
    min_publishers: int,
    payload_bound: float,
    strategy: str,
    seed: int = 0,
) -> dict:
    """Run the DP engine on a pre-aggregated stream (offline evaluation)."""
    np.random.seed(seed)
    config = PrivacyConfig(
        epsilon=float(epsilon),
        window_size=int(window_size),
        min_publishers=int(min_publishers),
        payload_bound=float(payload_bound),
        strategy=BudgetStrategy(strategy),
    )
    stream = StreamState(config=config)
    for agg, n in zip(aggregates, pub_counts):
        stream.release(agg, n)

    metrics = compute_utility_metrics(stream.true_values, stream.noisy_values)
    metrics["normalized_mae"] = (
        metrics["mae"] / payload_bound if payload_bound > 0 else float("nan")
    )

    # KL on only the timestamps where budget was actually spent.
    true_kl = [t for t, b in zip(stream.true_values, stream.budgets_spent)
                if b > 0 and t is not None]
    noisy_kl = [n for n, b in zip(stream.noisy_values, stream.budgets_spent)
                 if b > 0 and n is not None]
    if len(true_kl) < 10:
        true_kl = stream.true_values
        noisy_kl = stream.noisy_values

    metrics["kl_divergence"] = compute_kl_divergence(true_kl, noisy_kl)
    metrics["kl_global_utility"] = compute_global_utility(
        stream.true_values, stream.noisy_values, config.window_size
    )
    metrics["release_rate"] = (
        stream.releases / (stream.releases + stream.deferrals)
        if stream.releases + stream.deferrals > 0 else float("nan")
    )
    metrics["deferrals"] = stream.deferrals
    metrics["attribution_advantage"] = attribution_advantage(
        stream.pub_counts, [b == 0 for b in stream.budgets_spent]
    )

    return {
        "metrics": metrics,
        "true_values": stream.true_values,
        "noisy_values": stream.noisy_values,
        "budgets_spent": stream.budgets_spent,
        "pub_counts": stream.pub_counts,
        "kl_windowed": compute_windowed_kl_divergence(
            stream.true_values, stream.noisy_values, config.window_size
        ),
    }


# ═════════════════════════════════════════════════════════════════════════
#  Parameter sweep
# ═════════════════════════════════════════════════════════════════════════

def sweep(
    dataset_name: str,
    streams: dict[str, tuple[list[float], list[int], float]],
    s_values: list[int],
    epsilon_values: list[float],
    w_values: list[int],
    strategies: list[str],
) -> pd.DataFrame:
    rows = []
    combos = list(itertools.product(
        streams.keys(), s_values, epsilon_values, w_values, strategies,
    ))

    for i, (sensor, P, eps, w, strat) in enumerate(combos):
        aggregates, pub_counts, B = streams[sensor]
        result = run_dp_on_stream(
            aggregates, pub_counts,
            epsilon=eps, window_size=w, min_publishers=P,
            payload_bound=B, strategy=strat, seed=i,
        )
        m = result["metrics"]
        avg_n = float(np.mean([n for n in pub_counts if n > 0])) if any(pub_counts) else 0.0
        rows.append({
            "dataset": dataset_name,
            "sensor": sensor,
            "P": P,
            "epsilon": eps,
            "w": w,
            "strategy": strat,
            "mae": m["mae"],
            "rmse": m["rmse"],
            "relative_error": m["relative_error"],
            "normalized_mae": m["normalized_mae"],
            "kl_divergence": m["kl_divergence"],
            "kl_global_utility": m["kl_global_utility"],
            "release_rate": m["release_rate"],
            "deferrals": m["deferrals"],
            "attribution_advantage": m["attribution_advantage"],
            # Theoretical Uniform Laplace scale on the released mean:
            #   lambda = R * w / (n_tau * eps);  use avg n_tau for reporting.
            "noise_scale_theoretical": B * w / (max(avg_n, 1.0) * eps),
            "payload_bound": B,
            "num_timestamps": len(aggregates),
            "avg_publishers": float(np.mean(pub_counts)),
        })
        if (i + 1) % 100 == 0:
            logger.info(f"  [{dataset_name}] {i + 1}/{len(combos)}")

    logger.info(f"  [{dataset_name}] sweep complete: {len(combos)} configs")
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════
#  Plotting
# ═════════════════════════════════════════════════════════════════════════

def _hide_unused(axes_grid, n_used, nrows, ncols):
    for idx in range(n_used, nrows * ncols):
        axes_grid[idx // ncols][idx % ncols].set_visible(False)


def plot_results(
    df: pd.DataFrame,
    dataset_name: str,
    output_dir: str,
    streams: dict[str, tuple[list[float], list[int], float]],
):
    os.makedirs(output_dir, exist_ok=True)
    sensors = sorted(df["sensor"].unique())
    strategies = sorted(df["strategy"].unique())
    first = sensors[0]
    w_mid = int(sorted(df["w"].unique())[len(df["w"].unique()) // 2])
    eps_mid = float(sorted(df["epsilon"].unique())[len(df["epsilon"].unique()) // 2])
    p_mid = int(sorted(df["P"].unique())[len(df["P"].unique()) // 2])
    ncols = min(3, len(sensors))
    nrows = (len(sensors) + ncols - 1) // ncols

    # 1 -- MAE vs epsilon, one panel per strategy.
    fig, axes = plt.subplots(1, len(strategies), figsize=(4 * len(strategies), 4.5), sharey=True)
    if len(strategies) == 1:
        axes = [axes]
    sub = df[(df["sensor"] == first) & (df["w"] == w_mid)]
    for ax, strat in zip(axes, strategies):
        sd = sub[sub["strategy"] == strat]
        for P in sorted(sd["P"].unique()):
            d = sd[sd["P"] == P].sort_values("epsilon")
            ax.plot(d["epsilon"], d["mae"], marker="o", label=f"P={P}")
        ax.set(xlabel="eps", ylabel="MAE", title=strat)
        ax.legend(fontsize=7); ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(True, alpha=0.3)
    fig.suptitle(f"{dataset_name}: MAE vs eps [{first}, w={w_mid}]", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_mae_vs_epsilon.png"), dpi=150)
    plt.close()

    # 2 -- MAE vs P, one panel per sensor.
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    sub = df[(df["epsilon"] == eps_mid) & (df["w"] == w_mid) & (df["strategy"] == "uniform")]
    for idx, sensor in enumerate(sensors):
        ax = axes[idx // ncols][idx % ncols]
        sd = sub[sub["sensor"] == sensor].sort_values("P")
        ax.bar(sd["P"].astype(str), sd["mae"], color="steelblue", alpha=0.8)
        ax.set(xlabel="P", ylabel="MAE", title=sensor); ax.grid(True, alpha=0.3, axis="y")
    _hide_unused(axes, len(sensors), nrows, ncols)
    fig.suptitle(f"{dataset_name}: MAE vs P (eps={eps_mid}, w={w_mid}, uniform)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_mae_vs_P.png"), dpi=150)
    plt.close()

    # 3 -- Strategy comparison (grouped bar per sensor).
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    sub = df[(df["epsilon"] == eps_mid) & (df["w"] == w_mid) & (df["P"] == p_mid)]
    palette = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b2", "#937860"]
    for idx, sensor in enumerate(sensors):
        ax = axes[idx // ncols][idx % ncols]
        sd = sub[sub["sensor"] == sensor]
        strats = sorted(sd["strategy"].unique())
        vals = [sd[sd["strategy"] == s]["mae"].mean() for s in strats]
        ax.bar(strats, vals, color=palette[:len(strats)], alpha=0.85)
        ax.set(ylabel="MAE", title=sensor)
        ax.tick_params(axis="x", rotation=30, labelsize=7)
        ax.grid(True, alpha=0.3, axis="y")
    _hide_unused(axes, len(sensors), nrows, ncols)
    fig.suptitle(f"{dataset_name}: Strategy Comparison (eps={eps_mid}, w={w_mid}, P={p_mid})", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_strategy_comparison.png"), dpi=150)
    plt.close()

    # 4 -- Time series for each sensor, all strategies overlaid.
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
    for idx, sensor in enumerate(sensors):
        ax = axes[idx // ncols][idx % ncols]
        agg, cnt, B = streams[sensor]
        show_len = min(200, len(agg))
        ax.plot(range(show_len), agg[:show_len], "k-", alpha=0.8, label="True", lw=1.2)
        for strat, color in zip(["uniform", "n_weighted"], ["tab:red", "tab:green"]):
            if strat not in strategies:
                continue
            res = run_dp_on_stream(agg, cnt, epsilon=eps_mid, window_size=w_mid,
                                   min_publishers=p_mid, payload_bound=B,
                                   strategy=strat, seed=99)
            ny = [(i, v) for i, v in enumerate(res["noisy_values"][:show_len]) if v is not None]
            if ny:
                ax.plot([p[0] for p in ny], [p[1] for p in ny], color=color,
                         alpha=0.5, label=strat, lw=1)
        ax.set(xlabel="Window", ylabel=sensor, title=f"{sensor} (eps={eps_mid}, P={p_mid})")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    _hide_unused(axes, len(sensors), nrows, ncols)
    fig.suptitle(f"{dataset_name}: True vs DP-Protected Streams", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_timeseries.png"), dpi=150)
    plt.close()

    # 5 -- Budget utilization.
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
    for idx, sensor in enumerate(sensors):
        ax = axes[idx // ncols][idx % ncols]
        agg, cnt, B = streams[sensor]
        for strat in strategies:
            res = run_dp_on_stream(agg, cnt, epsilon=eps_mid, window_size=w_mid,
                                   min_publishers=p_mid, payload_bound=B,
                                   strategy=strat, seed=42)
            budgets = res["budgets_spent"]
            wsums = [sum(budgets[max(0, i - w_mid + 1):i + 1]) for i in range(min(200, len(budgets)))]
            ax.plot(range(len(wsums)), wsums, label=strat, alpha=0.8, lw=1)
        ax.axhline(y=eps_mid, color="red", ls="--", alpha=0.5, label=f"eps={eps_mid}")
        ax.set(xlabel="Window", ylabel="Budget spent", title=sensor)
        ax.legend(fontsize=6); ax.grid(True, alpha=0.3)
    _hide_unused(axes, len(sensors), nrows, ncols)
    fig.suptitle(f"{dataset_name}: Budget Utilization per Strategy", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_budget_utilization.png"), dpi=150)
    plt.close()

    # 6 -- KL vs epsilon per strategy.
    fig, axes = plt.subplots(1, len(strategies), figsize=(4 * len(strategies), 4.5), sharey=True)
    if len(strategies) == 1:
        axes = [axes]
    sub = df[(df["sensor"] == first) & (df["w"] == w_mid)]
    for ax, strat in zip(axes, strategies):
        sd = sub[sub["strategy"] == strat]
        for P in sorted(sd["P"].unique()):
            d = sd[sd["P"] == P].sort_values("epsilon")
            ax.plot(d["epsilon"], d["kl_divergence"], marker="o", label=f"P={P}")
        ax.set(xlabel="eps", ylabel="KL divergence", title=strat)
        ax.legend(fontsize=7); ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(True, alpha=0.3)
    fig.suptitle(f"{dataset_name}: KL Divergence vs eps [{first}, w={w_mid}]", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_kl_vs_epsilon.png"), dpi=150)
    plt.close()

    # 7 -- KL heatmap (w, eps) x (P, strategy).
    w_vals = sorted(df["w"].unique())
    eps_vals = sorted(df["epsilon"].unique())
    s_vals = sorted(df["P"].unique())
    agg_df = df.groupby(["w", "epsilon", "P", "strategy"]).agg(
        kl_mean=("kl_divergence", "mean")).reset_index()
    row_labels = [(w, e) for w in w_vals for e in eps_vals]
    col_labels = [(s, st) for s in s_vals for st in strategies]
    grid = np.full((len(row_labels), len(col_labels)), np.nan)
    for i, (w, e) in enumerate(row_labels):
        for j, (s, st) in enumerate(col_labels):
            m = agg_df[(agg_df["w"] == w) & (agg_df["epsilon"] == e) &
                       (agg_df["P"] == s) & (agg_df["strategy"] == st)]
            if len(m) == 1:
                grid[i, j] = m["kl_mean"].values[0]

    fig, ax = plt.subplots(figsize=(max(14, len(col_labels) * 1.1), max(8, len(row_labels) * 0.5)))
    im = ax.imshow(grid, aspect="auto", cmap="YlOrRd")
    fig.colorbar(im, ax=ax, label="Mean KL divergence")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels([f"P={s}\n{st}" for s, st in col_labels], fontsize=6, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels([f"w={w}, eps={e}" for w, e in row_labels], fontsize=8)
    med = np.nanmedian(grid)
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            v = grid[i, j]
            if np.isfinite(v):
                c = "white" if v > med else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5, color=c)
    ax.set_title(f"{dataset_name}: KL Divergence Heatmap (incl. n-weighted)", fontsize=12, pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_kl_heatmap.png"), dpi=200)
    plt.close()

    # 8 -- Per-window KL time series.
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
    for idx, sensor in enumerate(sensors):
        ax = axes[idx // ncols][idx % ncols]
        agg, cnt, B = streams[sensor]
        for strat, color in zip(["uniform", "n_weighted"], ["tab:red", "tab:green"]):
            if strat not in strategies:
                continue
            res = run_dp_on_stream(agg, cnt, epsilon=eps_mid, window_size=w_mid,
                                   min_publishers=p_mid, payload_bound=B,
                                   strategy=strat, seed=99)
            kl_w = res["kl_windowed"]
            n = min(200, len(kl_w))
            ax.plot(range(n), kl_w[:n], color=color, alpha=0.7, lw=1, label=strat)
        ax.set(xlabel="Window", ylabel="U__tau", title=sensor)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    _hide_unused(axes, len(sensors), nrows, ncols)
    fig.suptitle(f"{dataset_name}: Per-Window KL (Uniform vs n-weighted)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_kl_windowed.png"), dpi=150)
    plt.close()

    # 9 -- Release rate vs P per strategy (shows P-allocation deferral cost).
    fig, ax = plt.subplots(figsize=(7, 5))
    for strat in strategies:
        d = df[(df["strategy"] == strat) & (df["sensor"] == first) &
               (df["w"] == w_mid) & (df["epsilon"] == eps_mid)].sort_values("P")
        ax.plot(d["P"], d["release_rate"], marker="o", label=strat)
    ax.set(xlabel="P (publisher threshold)", ylabel="Release rate",
           title=f"{dataset_name}: Deferral Cost of P-allocation")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_release_rate_vs_P.png"), dpi=150)
    plt.close()

    logger.info(f"  All plots saved to {output_dir}/")


# ═════════════════════════════════════════════════════════════════════════
#  n-weighted spotlight: variance of n_tau vs. n-weighted advantage
# ═════════════════════════════════════════════════════════════════════════

def n_weighted_spotlight(
    streams: dict[str, tuple[list[float], list[int], float]],
    dataset_name: str,
    output_dir: str,
    epsilon: float = 1.0,
    w: int = 8,
    P: int = 2,
):
    """
    n-weighted P-allocation is expected to dominate Uniform when n_tau varies
    substantially within a window (paper Section 5.4).  This plot verifies
    empirically: KL_uniform - KL_n_weighted as a function of CV(n_tau).
    """
    rows = []
    for sensor, (agg, cnt, B) in streams.items():
        cv = float(np.std(cnt) / max(np.mean(cnt), 1e-9))
        r_u = run_dp_on_stream(agg, cnt, epsilon, w, P, B, "uniform", seed=7)
        r_n = run_dp_on_stream(agg, cnt, epsilon, w, P, B, "n_weighted", seed=7)
        rows.append({
            "sensor": sensor,
            "cv_n_tau": cv,
            "mean_n_tau": float(np.mean(cnt)),
            "kl_uniform": r_u["metrics"]["kl_divergence"],
            "kl_n_weighted": r_n["metrics"]["kl_divergence"],
            "mae_uniform": r_u["metrics"]["mae"],
            "mae_n_weighted": r_n["metrics"]["mae"],
            "nmae_uniform": r_u["metrics"]["normalized_mae"],
            "nmae_n_weighted": r_n["metrics"]["normalized_mae"],
        })
    out = pd.DataFrame(rows)
    out["kl_advantage"] = out["kl_uniform"] - out["kl_n_weighted"]
    out.to_csv(os.path.join(output_dir, f"{dataset_name}_n_weighted_spotlight.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(out["sensor"], out["kl_uniform"], alpha=0.7, label="Uniform", color="tab:red")
    axes[0].bar(out["sensor"], out["kl_n_weighted"], alpha=0.7,
                label="n-weighted", color="tab:green")
    axes[0].set(title=f"{dataset_name}: KL (Uniform vs n-weighted, P={P}, eps={epsilon})",
                ylabel="KL divergence")
    axes[0].legend(); axes[0].grid(True, alpha=0.3, axis="y")
    axes[0].tick_params(axis="x", rotation=30, labelsize=8)

    axes[1].scatter(out["cv_n_tau"], out["kl_advantage"], s=100, color="purple", alpha=0.8)
    for _, r in out.iterrows():
        axes[1].annotate(r["sensor"], (r["cv_n_tau"], r["kl_advantage"]), fontsize=8)
    axes[1].axhline(y=0, color="k", ls="--", alpha=0.4)
    axes[1].set(xlabel="CV(n__tau)", ylabel="KL_uniform − KL_n_weighted",
                title="n-weighted advantage increases with pool variability")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_n_weighted_spotlight.png"), dpi=150)
    plt.close()
    logger.info(f"  n-weighted spotlight saved (CV range {out['cv_n_tau'].min():.2f}-{out['cv_n_tau'].max():.2f}).")
    return out


# ═════════════════════════════════════════════════════════════════════════
#  Dynamic timestamp-interval extension (K_ext sweep)
# ═════════════════════════════════════════════════════════════════════════

def dynamic_interval_experiment(
    per_pub: dict[str, list[float | None]],
    payload_bound: float,
    dataset_name: str,
    sensor_name: str,
    output_dir: str,
    epsilon: float = 1.0,
    w: int = 8,
    P: int = 3,
    k_ext_values: list[int] = (0, 1, 2, 4),
) -> pd.DataFrame:
    """
    Simulate adaptive timestamp extension (Section 5.6) by merging k_ext
    consecutive raw-timestamp slots whenever the leaf-level publisher count
    is below P.  Reports release rate, KL divergence, and per-release
    wall-clock latency in units of the base interval Delta_t.
    """
    T = len(next(iter(per_pub.values())))
    rows = []

    for k_ext in k_ext_values:
        merged_agg: list[float] = []
        merged_cnt: list[int] = []
        merged_wait: list[int] = []    # how many base intervals elapsed
        pending_vals: list[float] = []
        pending_waits = 0

        for tau in range(T):
            active = [per_pub[p][tau] for p in per_pub if per_pub[p][tau] is not None]
            pending_vals.extend(active)
            pending_waits += 1

            if len(set(range(len(pending_vals)))) >= P or pending_waits > k_ext:
                if pending_vals:
                    merged_agg.append(float(np.mean(pending_vals)))
                    merged_cnt.append(len(pending_vals))
                else:
                    merged_agg.append(0.0)
                    merged_cnt.append(0)
                merged_wait.append(pending_waits)
                pending_vals = []
                pending_waits = 0

        if pending_vals:
            merged_agg.append(float(np.mean(pending_vals)))
            merged_cnt.append(len(pending_vals))
            merged_wait.append(pending_waits)

        res = run_dp_on_stream(
            merged_agg, merged_cnt, epsilon=epsilon, window_size=w,
            min_publishers=P, payload_bound=payload_bound,
            strategy="p_gated_ba", seed=13 + k_ext,
        )
        m = res["metrics"]
        rows.append({
            "K_ext": k_ext,
            "T_max_over_dt": k_ext + 1,
            "num_releases": len(merged_agg),
            "release_rate": m["release_rate"],
            "mean_wait_dt": float(np.mean(merged_wait)) if merged_wait else float("nan"),
            "max_wait_dt": max(merged_wait) if merged_wait else 0,
            "mae": m["mae"],
            "kl_divergence": m["kl_divergence"],
            "normalized_mae": m["normalized_mae"],
        })

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(output_dir, f"{dataset_name}_{sensor_name}_k_ext_sweep.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(out["K_ext"], out["mean_wait_dt"], "o-", color="tab:blue", label="mean")
    axes[0].plot(out["K_ext"], out["max_wait_dt"], "s--", color="tab:purple", label="max")
    axes[0].set(xlabel="K_ext", ylabel="Wait (× dt)",
                title=f"{dataset_name}/{sensor_name}: Latency vs K_ext")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    ax2 = axes[1]; ax3 = ax2.twinx()
    ax2.plot(out["K_ext"], out["kl_divergence"], "o-", color="tab:red", label="KL")
    ax3.plot(out["K_ext"], out["normalized_mae"], "s--", color="tab:green", label="NMAE")
    ax2.set(xlabel="K_ext", ylabel="KL divergence")
    ax3.set_ylabel("Normalized MAE")
    ax2.set_title("Utility vs K_ext"); ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left"); ax3.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_{sensor_name}_k_ext.png"), dpi=150)
    plt.close()
    logger.info(f"  K_ext sweep done ({sensor_name}).")
    return out


# ═════════════════════════════════════════════════════════════════════════
#  Subscriber collusion experiment (paper Section 4.7)
# ═════════════════════════════════════════════════════════════════════════

def collusion_experiment(
    streams: dict[str, tuple[list[float], list[int], float]],
    dataset_name: str,
    output_dir: str,
    epsilon: float = 1.0,
    w: int = 8,
    P: int = 2,
    c_values: list[int] = (1, 2, 4, 8, 16),
    trials_per_c: int = 32,
) -> pd.DataFrame:
    """
    Empirically verify the sqrt(c) noise reduction: c independent noisy
    streams averaged together have noise std reduced by ~sqrt(c).
    """
    sensor = next(iter(streams))
    agg, cnt, B = streams[sensor]

    rows = []
    for c in c_values:
        maes = []
        for trial in range(trials_per_c):
            avg_noisy = None
            for k in range(c):
                res = run_dp_on_stream(agg, cnt, epsilon, w, P, B, "uniform",
                                        seed=1_000 * trial + k)
                arr = np.array([v if v is not None else np.nan for v in res["noisy_values"]],
                               dtype=float)
                avg_noisy = arr if avg_noisy is None else avg_noisy + arr
            avg_noisy /= c
            true_arr = np.array([t if t is not None else np.nan for t in res["true_values"]],
                                 dtype=float)
            mask = np.isfinite(avg_noisy) & np.isfinite(true_arr)
            maes.append(float(np.mean(np.abs(avg_noisy[mask] - true_arr[mask]))))
        rows.append({
            "num_colluders_c": c,
            "mean_mae": float(np.mean(maes)),
            "std_mae": float(np.std(maes)),
            "predicted_mae_ratio": 1.0 / np.sqrt(c),
        })

    out = pd.DataFrame(rows)
    baseline = out.iloc[0]["mean_mae"]
    out["empirical_mae_ratio"] = out["mean_mae"] / baseline
    out.to_csv(os.path.join(output_dir, f"{dataset_name}_collusion.csv"), index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(out["num_colluders_c"], out["empirical_mae_ratio"], "o-", color="tab:red",
            label="Empirical MAE ratio")
    ax.plot(out["num_colluders_c"], out["predicted_mae_ratio"], "s--", color="tab:blue",
            label="Predicted 1/√c")
    ax.set(xlabel="Number of colluding subscribers c", ylabel="MAE ratio vs c=1",
           title=f"{dataset_name}: Collusion shrinks noise by 1/√c (Section 4.7)")
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_xscale("log"); ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_collusion.png"), dpi=150)
    plt.close()
    logger.info(f"  Collusion experiment done.")
    return out


# ═════════════════════════════════════════════════════════════════════════
#  Section 5.7: Two-stage hyperparameter tuning
# ═════════════════════════════════════════════════════════════════════════

def _rebuild_stream_with_dt(
    per_pub: dict[str, list[float | None]],
    dt_multiplier: int,
) -> tuple[list[float], list[int]]:
    """Re-bucket a per-publisher stream into blocks of `dt_multiplier` raw
    timestamps.  dt_multiplier=1 keeps the native Delta_t; dt_multiplier=k
    aggregates k consecutive raw slots into one logical timestamp.
    """
    T = len(next(iter(per_pub.values())))
    agg, cnt = [], []
    for start in range(0, T, dt_multiplier):
        block_vals = []
        contrib_pubs = set()
        for tau in range(start, min(start + dt_multiplier, T)):
            for p, series in per_pub.items():
                v = series[tau]
                if v is not None:
                    block_vals.append(v)
                    contrib_pubs.add(p)
        agg.append(float(np.mean(block_vals)) if block_vals else 0.0)
        # Multiplicity is the number of distinct publishers that contributed
        # any message during the block, matching P_tau in the paper.
        cnt.append(len(contrib_pubs))
    return agg, cnt


def _pareto_front(nmae: np.ndarray, latency: np.ndarray) -> np.ndarray:
    pts = list(zip(nmae, latency))
    is_pareto = []
    for i, (u_i, l_i) in enumerate(pts):
        dominated = any(
            (u_j <= u_i and l_j <= l_i) and (u_j < u_i or l_j < l_i)
            for j, (u_j, l_j) in enumerate(pts) if j != i
        )
        is_pareto.append(not dominated)
    return np.array(is_pareto)


def _evaluate_P(
    per_pub, payload_bound, epsilon, w, P, strategy,
    utility_weight=1.0, latency_weight=0.2, seed=77,
) -> dict:
    """Single-point evaluator; corresponds to paper's Evaluate(trace, P)."""
    agg, cnt = _rebuild_stream_with_dt(per_pub, 1)
    if len(agg) < w + 2:
        return {
            "P": int(P), "strategy": strategy, "normalized_mae": float("nan"),
            "kl_divergence": float("nan"), "release_rate": float("nan"),
            "attribution_advantage": float("nan"),
            "tuning_loss": float("inf"), "avg_n_tau": float("nan"),
            "pr_n_ge_P": float("nan"), "deferrals": 0, "evaluated": False,
        }
    res = run_dp_on_stream(
        agg, cnt, epsilon=epsilon, window_size=w, min_publishers=int(P),
        payload_bound=payload_bound, strategy=strategy, seed=seed,
    )
    m = res["metrics"]
    nmae = m["normalized_mae"]; rr = m["release_rate"]
    nmae_val = float(nmae) if nmae is not None and np.isfinite(nmae) else 1.0
    rr_val = float(rr) if rr is not None and np.isfinite(rr) else 0.0
    loss = utility_weight * nmae_val + latency_weight * (1.0 - rr_val)
    return {
        "P": int(P), "strategy": strategy,
        "normalized_mae": nmae, "kl_divergence": m["kl_divergence"],
        "release_rate": rr, "attribution_advantage": m["attribution_advantage"],
        "tuning_loss": float(loss), "avg_n_tau": float(np.mean(cnt)),
        "pr_n_ge_P": float(np.mean([n >= P for n in cnt])),
        "deferrals": m["deferrals"], "evaluated": True,
    }


def greedy_tune_P(
    per_pub, payload_bound, epsilon, w, strategy, p_max,
    alpha=0.25, I_max=20, utility_weight=1.0, latency_weight=0.2,
    loss_tie_tol: float = 1e-9,
) -> dict:
    """Algorithm 2 from the paper: greedy hill-climb over P.

    Seed P_0 = ceil(1/alpha) (smallest P that meets the attribution-advantage
    target alpha).  At each step, evaluate P-1 and P+1, move to the better if
    it strictly improves (within `loss_tie_tol`), else stop.  On a plateau the
    walk stops at the seed, which is the conservative choice since increasing
    P strengthens identity protection without any loss cost.
    """
    P_seed = max(1, int(np.ceil(1.0 / max(alpha, 1e-9))))
    P = min(P_seed, max(1, p_max))
    trajectory: list[dict] = []
    best = _evaluate_P(per_pub, payload_bound, epsilon, w, P, strategy,
                       utility_weight, latency_weight)
    trajectory.append({**best, "iter": 0, "action": "seed"})
    seen = {P: best["tuning_loss"]}

    for i in range(1, I_max + 1):
        neighbors = []
        for P_nbr in (P - 1, P + 1):
            if P_nbr < 1 or P_nbr > p_max:
                continue
            if P_nbr in seen:
                neighbors.append({"P": P_nbr, "tuning_loss": seen[P_nbr], "_cached": True})
                continue
            r = _evaluate_P(per_pub, payload_bound, epsilon, w, P_nbr, strategy,
                            utility_weight, latency_weight)
            seen[P_nbr] = r["tuning_loss"]
            neighbors.append({**r, "_cached": False})
            trajectory.append({**r, "iter": i, "action": f"probe_P={P_nbr}"})
        if not neighbors:
            break
        # Tie-break: on equal loss prefer the higher P (better identity protection).
        best_nbr = min(neighbors, key=lambda x: (x["tuning_loss"], -x["P"]))
        strict_improvement = best_nbr["tuning_loss"] < best["tuning_loss"] - loss_tie_tol
        if strict_improvement:
            P = best_nbr["P"]
            if "evaluated" in best_nbr:
                best = {k: v for k, v in best_nbr.items() if not k.startswith("_")}
            else:
                best = _evaluate_P(per_pub, payload_bound, epsilon, w, P, strategy,
                                   utility_weight, latency_weight)
            trajectory.append({**best, "iter": i, "action": f"step_to_P={P}"})
        else:
            trajectory.append({**best, "iter": i, "action": "stop_local_optimum"})
            break
    return {
        "strategy": strategy, "best": best, "seed_P": P_seed,
        "trajectory": pd.DataFrame(trajectory),
        "evaluations": sum(1 for t in trajectory if t["action"].startswith(("seed", "probe"))),
    }


def brute_force_tune_P(
    per_pub, payload_bound, epsilon, w, strategy, p_max,
    utility_weight=1.0, latency_weight=0.2,
) -> pd.DataFrame:
    """Naive full enumeration: evaluate every P in [1, p_max] for one strategy."""
    rows = []
    for P in range(1, int(p_max) + 1):
        rows.append(_evaluate_P(per_pub, payload_bound, epsilon, w, P, strategy,
                                utility_weight, latency_weight))
    return pd.DataFrame(rows)


def tune_hyperparameters(
    per_pub: dict[str, list[float | None]],
    payload_bound: float,
    dataset_name: str,
    sensor_name: str,
    output_dir: str,
    epsilon: float,
    w: int,
    strategies: list[str] | str = "p_gated_ba",
    alpha: float = 0.25,
    I_max: int = 20,
    utility_weight: float = 1.0,
    latency_weight: float = 0.2,
    p_max: int | None = None,
) -> dict:
    """
    Stage 1 of Section 5.7: tune P per strategy using the paper's Algorithm 2
    greedy hill-climb, and compare against a naive full enumeration over every
    integer P in [1, p_max].  Delta_t adapts online via the extension mechanism
    of Section 5.6, so it is not tuned here.  Strategies are evaluated in
    parallel (one run each); the paper treats A as selected by subscription
    requirements rather than jointly optimized.

    Writes:
      {ds}_{sensor}_tuning_greedy.csv          -- every evaluated (P, strategy)
      {ds}_{sensor}_tuning_brute_force.csv     -- every P in [1, p_max]
      {ds}_{sensor}_tuning_strategy_summary.csv -- greedy vs brute-force best
      {ds}_{sensor}_tuning.png                 -- per-strategy loss curves
    """
    if isinstance(strategies, str):
        strategies = [strategies]

    # p_max defaults to the maximum observed n_tau on the trace.
    _, raw_cnt = _rebuild_stream_with_dt(per_pub, 1)
    obs_max = max(raw_cnt) if raw_cnt else 1
    if p_max is None:
        p_max = max(2, min(obs_max, len(per_pub)))
    p_max = int(p_max)

    greedy_rows: list[pd.DataFrame] = []
    brute_rows: list[pd.DataFrame] = []
    greedy_results: dict[str, dict] = {}
    for strat in strategies:
        g = greedy_tune_P(per_pub, payload_bound, epsilon, w, strat, p_max,
                          alpha=alpha, I_max=I_max,
                          utility_weight=utility_weight, latency_weight=latency_weight)
        g["trajectory"]["strategy"] = strat
        greedy_rows.append(g["trajectory"])
        greedy_results[strat] = g

        b = brute_force_tune_P(per_pub, payload_bound, epsilon, w, strat, p_max,
                               utility_weight=utility_weight, latency_weight=latency_weight)
        b["strategy"] = strat
        brute_rows.append(b)

    greedy_df = pd.concat(greedy_rows, ignore_index=True) if greedy_rows else pd.DataFrame()
    brute_df = pd.concat(brute_rows, ignore_index=True) if brute_rows else pd.DataFrame()

    greedy_df.to_csv(
        os.path.join(output_dir, f"{dataset_name}_{sensor_name}_tuning_greedy.csv"),
        index=False,
    )
    brute_df.to_csv(
        os.path.join(output_dir, f"{dataset_name}_{sensor_name}_tuning_brute_force.csv"),
        index=False,
    )

    # Per-strategy greedy vs brute-force comparison.
    summary_rows = []
    for strat in strategies:
        g_res = greedy_results[strat]
        g_best = g_res["best"]
        g_evals = g_res["evaluations"]
        b_sub = brute_df[brute_df["strategy"] == strat]
        if b_sub.empty:
            continue
        # Tie-break: on equal loss prefer the higher P (better identity protection).
        b_sub = b_sub.assign(_neg_P=-b_sub["P"]).sort_values(
            ["tuning_loss", "_neg_P"]).drop(columns=["_neg_P"])
        b_best = b_sub.iloc[0]
        gap_loss = float(g_best["tuning_loss"] - b_best["tuning_loss"])
        gap_P = int(g_best["P"] - b_best["P"])
        summary_rows.append({
            "strategy": strat,
            "greedy_P": int(g_best["P"]),
            "greedy_loss": float(g_best["tuning_loss"]),
            "greedy_nmae": g_best["normalized_mae"],
            "greedy_kl": g_best["kl_divergence"],
            "greedy_release_rate": g_best["release_rate"],
            "greedy_evaluations": g_evals,
            "greedy_seed_P": g_res["seed_P"],
            "brute_P": int(b_best["P"]),
            "brute_loss": float(b_best["tuning_loss"]),
            "brute_nmae": b_best["normalized_mae"],
            "brute_kl": b_best["kl_divergence"],
            "brute_release_rate": b_best["release_rate"],
            "brute_evaluations": len(b_sub),
            "gap_loss": gap_loss,
            "gap_P": gap_P,
            "speedup": float(len(b_sub) / max(g_evals, 1)),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        os.path.join(output_dir, f"{dataset_name}_{sensor_name}_tuning_strategy_summary.csv"),
        index=False,
    )

    logger.info(f"  [tuning] {dataset_name}/{sensor_name}  p_max={p_max}, alpha={alpha} -> seed P_0={max(1, int(np.ceil(1.0 / max(alpha, 1e-9))))}")
    logger.info(f"    {'strategy':<20} {'greedy_P':>9} {'brute_P':>8} "
                f"{'greedy_L':>9} {'brute_L':>8} {'gap_L':>8} "
                f"{'gr_evals':>9} {'br_evals':>9} {'speedup':>8}")
    for _, r in summary_df.iterrows():
        logger.info(
            f"    {r['strategy']:<20} {int(r['greedy_P']):>9} {int(r['brute_P']):>8} "
            f"{r['greedy_loss']:>9.4f} {r['brute_loss']:>8.4f} "
            f"{r['gap_loss']:>8.4f} {int(r['greedy_evaluations']):>9} "
            f"{int(r['brute_evaluations']):>9} {r['speedup']:>8.2f}"
        )

    # Per-strategy loss-vs-P plot with greedy trajectory overlay.
    palette = plt.cm.tab10(np.linspace(0, 1, max(len(strategies), 1)))
    ncols = min(3, len(strategies)) or 1
    nrows = (len(strategies) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.0 * nrows),
                             squeeze=False)
    for idx, (color, strat) in enumerate(zip(palette, strategies)):
        ax = axes[idx // ncols][idx % ncols]
        bsub = brute_df[brute_df["strategy"] == strat].sort_values("P")
        if bsub.empty:
            continue
        ax.plot(bsub["P"], bsub["tuning_loss"], "o-", color=color, lw=1.4,
                alpha=0.85, label="brute force (every P)")
        g_traj = greedy_df[(greedy_df["strategy"] == strat)
                           & (greedy_df["action"].str.startswith(("seed", "probe", "step")))]
        ax.scatter(g_traj["P"], g_traj["tuning_loss"], marker="x", s=70,
                   color="black", zorder=5, label="greedy probe")
        # Final greedy best
        g_best = greedy_results[strat]["best"]
        ax.scatter([g_best["P"]], [g_best["tuning_loss"]], marker="*", s=300,
                   color="gold", edgecolor="black", zorder=6,
                   label=f"greedy best (P={int(g_best['P'])})")
        b_best = bsub.sort_values("tuning_loss").iloc[0]
        ax.scatter([b_best["P"]], [b_best["tuning_loss"]], marker="D", s=110,
                   color="red", edgecolor="white", zorder=6,
                   label=f"brute-force best (P={int(b_best['P'])})")
        ax.set(xlabel="P", ylabel="tuning loss", title=strat)
        ax.legend(fontsize=6); ax.grid(True, alpha=0.3)
    for idx in range(len(strategies), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)
    fig.suptitle(f"{dataset_name}/{sensor_name}: Algorithm 2 greedy vs brute-force over P "
                 f"(alpha={alpha}, p_max={p_max})", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_{sensor_name}_tuning.png"), dpi=150)
    plt.close()

    return {"greedy": greedy_df, "brute_force": brute_df, "gap_summary": summary_df}


# ═════════════════════════════════════════════════════════════════════════
#  Figure 1 reproduction: KL divergence vs aggregation scope P
# ═════════════════════════════════════════════════════════════════════════

def figure1_reproduction(
    per_pub_all: dict[str, tuple[dict[str, list[float | None]], float]],
    dataset_name: str,
    output_dir: str,
    epsilon: float = 1.0,
    w: int = 8,
    n_trials: int = 20,
) -> pd.DataFrame:
    """
    Reproduce paper Figure 1 on real data (cross-dataset aggregation input).

    All regimes use **Uniform** budget allocation (paper §1.3: Figure 1 is
    about the effect of AGGREGATION SCOPE, not budget-allocation strategy);
    KL is averaged over `n_trials` seeds per sensor to suppress single-draw
    Laplace variance.  The "global" point is Paper Extreme 1 (one stream per
    system, cross-metric, one Laplace per tau).

    Returns a DataFrame with columns
    (dataset, P_scope, P_label, kl_divergence, epsilon, w, N).
    """
    # Reconstruct the per-sensor aggregate streams from per_pub_all so we
    # don't need the original `streams` dict.
    sensor_streams: dict[str, tuple[list[float], list[int], float]] = {}
    for s, (pp, B) in per_pub_all.items():
        T_s = len(next(iter(pp.values())))
        agg, cnt = [], []
        for tau in range(T_s):
            vals = [pp[p][tau] for p in pp if pp[p][tau] is not None]
            agg.append(float(np.mean(vals)) if vals else 0.0)
            cnt.append(len(vals))
        sensor_streams[s] = (agg, cnt, B)

    sensor_names = list(sensor_streams.keys())
    T = min(len(v[0]) for v in sensor_streams.values())
    N_pubs = max(len(pp) for pp, _ in per_pub_all.values())

    # Per-publisher extreme (P=1): mean KL across every publisher × trial.
    p1_kls = []
    for s in sensor_names:
        pp, B = per_pub_all[s]
        for i, series in enumerate(pp.values()):
            pa = [v if v is not None else 0.0 for v in series]
            pc = [1 if v is not None else 0 for v in series]
            for trial in range(n_trials):
                r = run_dp_on_stream(
                    pa, pc, epsilon=epsilon, window_size=w,
                    min_publishers=1, payload_bound=B, strategy="uniform",
                    seed=500_000 + i * 1000 + trial,
                )
                k = r["metrics"]["kl_divergence"]
                if np.isfinite(k):
                    p1_kls.append(k)
    kl_p1 = float(np.mean(p1_kls)) if p1_kls else float("nan")

    # Intermediate P: clamped aggregate with Uniform, averaged across sensors and trials.
    P_values = [p for p in [2, 3, 4, 6, 8] if p <= N_pubs]
    mid_kls: dict[int, float] = {}
    for P in P_values:
        kls = []
        for s in sensor_names:
            agg, cnt, B = sensor_streams[s]
            if max(cnt) < P:
                continue
            for trial in range(n_trials):
                r = run_dp_on_stream(
                    agg, cnt, epsilon=epsilon, window_size=w,
                    min_publishers=P, payload_bound=B, strategy="uniform",
                    seed=600_000 + P * 1000 + trial,
                )
                k = r["metrics"]["kl_divergence"]
                if np.isfinite(k):
                    kls.append(k)
        mid_kls[P] = float(np.mean(kls)) if kls else float("nan")

    # Global (Extreme 1): one stream per system, single noise per tau with
    # R = sup(R) and n_tau = total publishers across metrics.
    all_B = max(B for _, B in per_pub_all.values())
    cross_metric_true = []
    num_pubs_total = []
    for tau in range(T):
        vals = [sensor_streams[s][0][tau] for s in sensor_names
                if sensor_streams[s][1][tau] > 0]
        cross_metric_true.append(float(np.mean(vals)) if vals else 0.0)
        num_pubs_total.append(sum(sensor_streams[s][1][tau] for s in sensor_names))

    g_kls = []
    for s in sensor_names:
        true_stream = sensor_streams[s][0][:T]
        for trial in range(n_trials):
            r = run_dp_on_stream(
                cross_metric_true, num_pubs_total,
                epsilon=epsilon, window_size=w,
                min_publishers=1, payload_bound=all_B, strategy="uniform",
                seed=700_000 + hash(s) % 10000 + trial * 7919,
            )
            # Compare per-metric true against the cross-metric noisy global.
            nvals = r["noisy_values"]
            true_c = [v for v in true_stream if v is not None]
            nc = [v for v in nvals if v is not None]
            if len(true_c) >= 10 and len(nc) >= 10:
                k = compute_kl_divergence(true_c, nc)
                if np.isfinite(k):
                    g_kls.append(k)
    kl_global = float(np.mean(g_kls)) if g_kls else float("nan")

    rows = [{"dataset": dataset_name, "P_scope": 1, "P_label": "per-pub",
             "kl_divergence": kl_p1, "epsilon": epsilon, "w": w, "N": N_pubs}]
    for P in P_values:
        rows.append({"dataset": dataset_name, "P_scope": P, "P_label": f"P={P}",
                     "kl_divergence": mid_kls[P], "epsilon": epsilon, "w": w, "N": N_pubs})
    rows.append({"dataset": dataset_name, "P_scope": N_pubs + 1, "P_label": "global",
                 "kl_divergence": kl_global, "epsilon": epsilon, "w": w, "N": N_pubs})
    df_fig1 = pd.DataFrame(rows)
    df_fig1.to_csv(os.path.join(output_dir, f"{dataset_name}_figure1_kl_vs_P.csv"),
                   index=False)

    labels = df_fig1["P_label"].tolist()
    values = df_fig1["kl_divergence"].tolist()
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#e74c3c"] + ["#2ecc71"] * len(P_values) + ["#e74c3c"]
    bars = ax.bar(labels, values, color=colors, alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, values):
        if np.isfinite(v):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set(xlabel="Aggregation scope P",
           ylabel="Average KL divergence",
           title=f"{dataset_name}: Distributional Distortion vs. Aggregation Scope"
                 f"  (eps={epsilon}, w={w})")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(output_dir, f"{dataset_name}_figure1_kl_vs_P.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"  Figure-1 reproduction saved: {path}")
    return df_fig1


# ═════════════════════════════════════════════════════════════════════════
#  Paper's motivating intro figures (Section 1.3)
# ═════════════════════════════════════════════════════════════════════════
#
# Produces, per dataset:
#   extreme1_global: one-stream-per-system extreme (paper Section 1.3)
#   extreme2_per_publisher: one-stream-per-publisher extreme
#   kl_extremes_vs_ours: grouped-bar KL across both extremes + our approach
#   u_shaped_curve: the KL-vs-P U-shape (paper Figure 1 on real data)
# All functions write both CSV (where data-like) and PNG (always).

INTRO_EPSILON = 1.0
INTRO_W = 8


def _apply_dp(aggregates, pub_counts, epsilon, w, min_publishers,
              payload_bound, seed=0, strategy=BudgetStrategy.UNIFORM):
    np.random.seed(seed)
    cfg = PrivacyConfig(
        epsilon=epsilon, window_size=w, min_publishers=min_publishers,
        payload_bound=payload_bound, strategy=strategy,
    )
    state = StreamState(config=cfg)
    for a, n in zip(aggregates, pub_counts):
        state.release(float(a), int(n))
    return state.true_values, state.noisy_values


def extreme1_global(sensor_streams, output_dir, dataset_name):
    """Global mean across every publisher/topic.  Destroys topic signal."""
    T = min(len(v[0]) for v in sensor_streams.values())
    global_B = max(v[2] for v in sensor_streams.values())
    num_pubs_per_tau = [sum(v[1][tau] for v in sensor_streams.values()) for tau in range(T)]
    global_agg = []
    for tau in range(T):
        vals = [agg[tau] for (agg, cnt, _) in sensor_streams.values() if cnt[tau] > 0]
        global_agg.append(float(np.mean(vals)) if vals else 0.0)

    _, noisy_global = _apply_dp(global_agg, num_pubs_per_tau,
                                epsilon=INTRO_EPSILON, w=INTRO_W,
                                min_publishers=1, payload_bound=global_B, seed=100)

    pd.DataFrame({
        "t": range(T),
        "global_true": global_agg,
        "global_noisy": [v if v is not None else np.nan for v in noisy_global[:T]],
        "num_pubs_total": num_pubs_per_tau,
    }).to_csv(os.path.join(output_dir, f"{dataset_name}_figure_extreme1_global.csv"),
              index=False)

    t = np.arange(min(T, 300))
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1]},
    )
    palette = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12"]
    for i, (sensor, (agg, _, _)) in enumerate(sensor_streams.items()):
        ax1.plot(t, agg[:len(t)], color=palette[i % len(palette)],
                 alpha=0.6, lw=1.0, label=sensor)
    ax1.plot(t, global_agg[:len(t)], "k-", lw=2.5, alpha=0.9, label="global mean")
    ax1.set_ylabel("Sensor value")
    ax1.set_title("Extreme 1: Global Average Destroys Topic-Level Signal",
                  fontsize=13, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8, ncol=2); ax1.grid(True, alpha=0.3)

    noisy_x = [i for i, v in enumerate(noisy_global[:len(t)]) if v is not None]
    noisy_y = [noisy_global[i] for i in noisy_x]
    ax2.plot(t, global_agg[:len(t)], "b-", lw=1.5, alpha=0.8, label="true global mean")
    ax2.plot(noisy_x, noisy_y, "r-", lw=1.0, alpha=0.6,
             label=f"noisy DP release (eps={INTRO_EPSILON}, w={INTRO_W})")
    ax2.set(xlabel="Time window", ylabel="Global mean",
            title=f"w-event DP on global stream  (Delta_f = R/n_tau, R={global_B:.1f})")
    ax2.legend(loc="upper right", fontsize=8); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, f"{dataset_name}_figure_extreme1_global.png")
    plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()
    logger.info(f"  saved {path}")


def extreme2_per_publisher(per_pub, payload_bound, sensor_label,
                           dataset_name, output_dir):
    """Per-publisher DP: n_tau=1 everywhere, lambda = R*w/eps."""
    pubs = list(per_pub.keys())
    show = pubs[:4]
    noise_scale = payload_bound * INTRO_W / INTRO_EPSILON

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    csv_rows = []
    for ax, pub_id in zip(axes.flat, show):
        raw = per_pub[pub_id]
        pub_agg = [v if v is not None else 0.0 for v in raw]
        pub_cnt = [1 if v is not None else 0 for v in raw]
        true_vals, noisy_vals = _apply_dp(
            pub_agg, pub_cnt,
            epsilon=INTRO_EPSILON, w=INTRO_W, min_publishers=1,
            payload_bound=payload_bound, seed=hash(pub_id) % 10000,
        )
        n = min(300, len(true_vals))
        true_x = [i for i in range(n) if true_vals[i] is not None and true_vals[i] != 0]
        true_y = [true_vals[i] for i in true_x]
        ax.plot(true_x, true_y, "b-", lw=1.2, alpha=0.85, label="true")
        noisy_xy = [(i, noisy_vals[i]) for i in range(n) if noisy_vals[i] is not None]
        if noisy_xy:
            ax.plot([p[0] for p in noisy_xy], [p[1] for p in noisy_xy],
                    "r-", lw=0.8, alpha=0.5, label="DP release")
        ax.set_title(f"publisher: {pub_id}", fontsize=10)
        ax.legend(fontsize=7, loc="upper right"); ax.grid(True, alpha=0.3)
        for i in range(n):
            csv_rows.append({
                "publisher_id": pub_id, "t": i,
                "true": true_vals[i] if i < len(true_vals) else None,
                "noisy": noisy_vals[i] if i < len(noisy_vals) else None,
            })

    pd.DataFrame(csv_rows).to_csv(
        os.path.join(output_dir, f"{dataset_name}_figure_extreme2_per_publisher.csv"),
        index=False,
    )

    axes[1][0].set_xlabel("Time window"); axes[1][1].set_xlabel("Time window")
    axes[0][0].set_ylabel(sensor_label); axes[1][0].set_ylabel(sensor_label)
    fig.suptitle(
        f"Extreme 2: Per-publisher w-event DP  "
        f"(n_tau=1, Delta_f=R={payload_bound:.1f}, lambda=R*w/eps={noise_scale:.0f})\n"
        f"Noise overwhelms the signal; publisher identity is fully exposed",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(output_dir, f"{dataset_name}_figure_extreme2_per_publisher.png")
    plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()
    logger.info(f"  saved {path}")


def kl_extremes_vs_ours(sensor_streams, per_pub_all, P_our,
                        dataset_name, output_dir):
    """Grouped bar of KL for Extreme 1 / Extreme 2 / Our-approach per sensor.

    All three regimes are averaged over `INTRO_N_TRIALS` noise seeds so the
    bars reflect the expected distortion, not a single-draw artefact.
    """
    T = min(len(v[0]) for v in sensor_streams.values())
    sensor_names = list(sensor_streams.keys())
    global_B = max(v[2] for v in sensor_streams.values())
    uniform = BudgetStrategy.UNIFORM

    # Extreme 1 (paper §1.3): one stream per system.  Single Laplace release per
    # tau with n_tau = total publishers across every metric and R = sup R.
    global_true = []
    for tau in range(T):
        vals = [sensor_streams[s][0][tau] for s in sensor_names
                if sensor_streams[s][1][tau] > 0]
        global_true.append(float(np.mean(vals)) if vals else 0.0)
    num_pubs_global = [sum(sensor_streams[s][1][tau] for s in sensor_names)
                       for tau in range(T)]

    e1: dict[str, list[float]] = {s: [] for s in sensor_names}
    for trial in range(INTRO_N_TRIALS):
        _, noisy_global = _apply_dp(global_true, num_pubs_global,
                                    epsilon=INTRO_EPSILON, w=INTRO_W,
                                    min_publishers=1, payload_bound=global_B,
                                    seed=200_000 + trial * 13, strategy=uniform)
        for s in sensor_names:
            k = _kl_of(sensor_streams[s][0][:T], noisy_global)
            if np.isfinite(k):
                e1[s].append(k)
    results = {"Extreme 1\n(global average)":
               {s: float(np.mean(v)) if v else float("nan") for s, v in e1.items()}}

    # Extreme 2: each publisher is its own stream; average KL across publishers.
    e2: dict[str, list[float]] = {s: [] for s in sensor_names}
    for s in sensor_names:
        if s not in per_pub_all:
            continue
        pp, B = per_pub_all[s]
        for i, series in enumerate(pp.values()):
            pa = [v if v is not None else 0.0 for v in series]
            pc = [1 if v is not None else 0 for v in series]
            for trial in range(INTRO_N_TRIALS):
                tv, nv = _apply_dp(pa, pc, epsilon=INTRO_EPSILON, w=INTRO_W,
                                   min_publishers=1, payload_bound=B,
                                   seed=300_000 + i * 1000 + trial,
                                   strategy=uniform)
                k = _kl_of(tv, nv)
                if np.isfinite(k):
                    e2[s].append(k)
    results["Extreme 2\n(per-publisher)"] = {
        s: float(np.mean(v)) if v else float("nan") for s, v in e2.items()
    }

    # Our approach: clamped aggregate with P-gated BA at P_our.
    ours: dict[str, list[float]] = {s: [] for s in sensor_names}
    for s in sensor_names:
        agg, cnt, B = sensor_streams[s]
        for trial in range(INTRO_N_TRIALS):
            tv, nv = _apply_dp(agg, cnt, epsilon=INTRO_EPSILON, w=INTRO_W,
                               min_publishers=P_our, payload_bound=B,
                               seed=400_000 + trial,
                               strategy=BudgetStrategy.P_GATED_BA)
            k = _kl_of(tv, nv)
            if np.isfinite(k):
                ours[s].append(k)
    results[f"Our approach\n(P={P_our} topic pool)"] = {
        s: float(np.mean(v)) if v else float("nan") for s, v in ours.items()
    }

    csv_rows = []
    for regime, per_sensor in results.items():
        for sensor_name, kl in per_sensor.items():
            csv_rows.append({
                "regime": regime.replace("\n", " "),
                "sensor": sensor_name,
                "kl_divergence": kl,
                "P_our": P_our,
            })
    pd.DataFrame(csv_rows).to_csv(
        os.path.join(output_dir, f"{dataset_name}_figure_extremes_vs_ours.csv"),
        index=False,
    )

    regimes = list(results.keys())
    x = np.arange(len(regimes))
    bar_w = 0.15
    palette = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for j, s in enumerate(sensor_names):
        vals = [results[r].get(s, float("nan")) for r in regimes]
        offset = (j - len(sensor_names) / 2 + 0.5) * bar_w
        bars = ax.bar(x + offset, vals, bar_w, label=s,
                      color=palette[j % len(palette)], alpha=0.85,
                      edgecolor="white", lw=0.8)
        for bar, val in zip(bars, vals):
            if np.isfinite(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom",
                        fontsize=7, fontweight="bold")
    avg_vals = [float(np.nanmean([results[r].get(s, np.nan) for s in sensor_names]))
                for r in regimes]
    avg_off = (len(sensor_names) / 2 + 0.5) * bar_w
    ax.bar(x + avg_off, avg_vals, bar_w * 1.2, label="average",
           color="#34495e", alpha=0.7, edgecolor="white", lw=0.8)
    for i, val in enumerate(avg_vals):
        ax.text(x[i] + avg_off, val + 0.01, f"{val:.2f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold", color="#34495e")
    ax.axhline(y=np.log(2), color="gray", ls=":", alpha=0.5)
    ax.text(len(regimes) - 0.5, np.log(2) + 0.01, "ln(2) ~ 0.69",
            fontsize=7, color="gray", ha="right")
    ax.set_xticks(x); ax.set_xticklabels(regimes, fontsize=10)
    ax.set_ylabel("KL divergence  D_KL(P || Q)", fontsize=11)
    ax.set_title(
        f"{dataset_name}: naive extremes vs. our approach  "
        f"(eps={INTRO_EPSILON}, w={INTRO_W})",
        fontsize=13, fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=9, ncol=3); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(output_dir, f"{dataset_name}_figure_extremes_vs_ours.png")
    plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()
    logger.info(f"  saved {path}")
    return results


INTRO_N_TRIALS = 20   # seed average for Figure 1 (Laplace noise is high-variance)


def _kl_of(true_vals, noisy_vals):
    tc = [v for v in true_vals if v is not None]
    nc = [v for v in noisy_vals if v is not None]
    if len(tc) < 10 or len(nc) < 10:
        return float("nan")
    k = compute_kl_divergence(tc, nc)
    return k if np.isfinite(k) else float("nan")


def u_shaped_curve(sensor_streams, per_pub_all, dataset_name, output_dir):
    """Reproduce paper Figure 1 on real data: KL vs aggregation scope P.

    All three regimes use Uniform budget allocation (paper §1.3: Figure 1 is
    about the effect of AGGREGATION SCOPE, not budget-allocation strategy).
    Each (P, sensor) is averaged over INTRO_N_TRIALS noise seeds to suppress
    single-draw Laplace variance.
    """
    sensor_names = list(sensor_streams.keys())
    T = min(len(v[0]) for v in sensor_streams.values())
    uniform = BudgetStrategy.UNIFORM

    # P=1 per-publisher: mean KL over (publisher, trial).
    p1_kls = []
    for s in sensor_names:
        if s not in per_pub_all:
            continue
        pp, B = per_pub_all[s]
        for i, series in enumerate(pp.values()):
            pa = [v if v is not None else 0.0 for v in series]
            pc = [1 if v is not None else 0 for v in series]
            for trial in range(INTRO_N_TRIALS):
                seed = 1_000_000 + i * 1000 + trial
                tv, nv = _apply_dp(pa, pc, epsilon=INTRO_EPSILON, w=INTRO_W,
                                   min_publishers=1, payload_bound=B,
                                   seed=seed, strategy=uniform)
                k = _kl_of(tv, nv)
                if np.isfinite(k):
                    p1_kls.append(k)
    kl_p1 = float(np.mean(p1_kls)) if p1_kls else float("nan")

    # Intermediate P: per-sensor clamped-aggregate with Uniform, averaged over trials.
    sweep_p = [2, 3, 4, 6, 8]
    mid_kls: dict[int, float] = {}
    for P in sweep_p:
        kls = []
        for s in sensor_names:
            agg, cnt, B = sensor_streams[s]
            if max(cnt) < P:
                continue
            for trial in range(INTRO_N_TRIALS):
                seed = 2_000_000 + P * 1000 + trial
                tv, nv = _apply_dp(agg, cnt, epsilon=INTRO_EPSILON, w=INTRO_W,
                                   min_publishers=P, payload_bound=B,
                                   seed=seed, strategy=uniform)
                k = _kl_of(tv, nv)
                if np.isfinite(k):
                    kls.append(k)
        mid_kls[P] = float(np.mean(kls)) if kls else float("nan")

    # Paper Extreme 1 "one stream per system": collapse every publisher across
    # every metric into a SINGLE database and publish one Laplace-noised mean
    # per tau, with R = sup across metrics and n_tau = total active publishers.
    # Every subscriber then gets this same global value regardless of topic.
    all_B = max(v[2] for v in sensor_streams.values())
    num_pubs_total = [sum(v[1][tau] for v in sensor_streams.values())
                      for tau in range(T)]
    cross_metric_true = []
    for tau in range(T):
        vals = [sensor_streams[s][0][tau] for s in sensor_names
                if sensor_streams[s][1][tau] > 0]
        cross_metric_true.append(float(np.mean(vals)) if vals else 0.0)

    g_kls = []
    for s in sensor_names:
        # A subscriber to topic `<s>` wanted the per-metric mean, but under
        # Extreme 1 is handed the cross-metric global noise instead -- that
        # mismatch is exactly the distortion the paper is calling out.
        true_stream = sensor_streams[s][0][:T]
        for trial in range(INTRO_N_TRIALS):
            seed = 5_000_000 + hash(s) % 10000 + trial * 7919
            _, noisy_global = _apply_dp(
                cross_metric_true, num_pubs_total,
                epsilon=INTRO_EPSILON, w=INTRO_W,
                min_publishers=1, payload_bound=all_B,
                seed=seed, strategy=uniform,
            )
            k = _kl_of(true_stream, noisy_global)
            if np.isfinite(k):
                g_kls.append(k)
    kl_global = float(np.mean(g_kls)) if g_kls else float("nan")

    P_labels = ["1\n(per-pub)"] + [str(p) for p in sweep_p] + ["all\n(global)"]
    kl_values = [kl_p1] + [mid_kls[p] for p in sweep_p] + [kl_global]
    pd.DataFrame({
        "P_label": [l.replace("\n", " ") for l in P_labels],
        "kl_divergence": kl_values,
    }).to_csv(
        os.path.join(output_dir, f"{dataset_name}_figure_u_shaped_P_vs_KL.csv"),
        index=False,
    )

    best_mid = int(np.nanargmin(kl_values[1:-1])) + 1
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(kl_values))
    ax.plot(x, kl_values, "k-", lw=2, zorder=4)
    for i in range(len(x)):
        if i in (0, len(x) - 1):
            color = "#e74c3c"
        elif i == best_mid:
            color = "#2ecc71"
        else:
            color = "#555"
        ax.plot(x[i], kl_values[i], "o", color=color, markersize=12, zorder=6,
                markeredgecolor="white", markeredgewidth=1.5)
        if np.isfinite(kl_values[i]):
            off = max(kl_values) * 0.04
            ax.text(x[i], kl_values[i] + off, f"{kl_values[i]:.2f}",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.axvspan(-0.5, 0.5, color="#e74c3c", alpha=0.08, zorder=1)
    ax.axvspan(0.5, len(x) - 1.5, color="#2ecc71", alpha=0.08, zorder=1)
    ax.axvspan(len(x) - 1.5, len(x) - 0.5, color="#e74c3c", alpha=0.08, zorder=1)
    ax.set_xticks(x); ax.set_xticklabels(P_labels, fontsize=10)
    ax.set_xlabel("Aggregation scope  P", fontsize=12)
    ax.set_ylabel("Average KL divergence", fontsize=12)
    ax.set_title(
        f"{dataset_name}: Distributional distortion vs. aggregation scope  "
        f"(eps={INTRO_EPSILON}, w={INTRO_W})",
        fontsize=12, fontweight="bold",
    )
    ax.grid(True, alpha=0.25, axis="y")
    plt.tight_layout()
    path = os.path.join(output_dir, f"{dataset_name}_figure_u_shaped_P_vs_KL.png")
    plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()
    logger.info(f"  saved {path}")


def run_intro_figures(sensor_streams, per_pub_all, dataset_name, output_dir, P_our=4):
    """Produce the four paper-style intro figures for one dataset."""
    if not sensor_streams:
        return
    extreme1_global(sensor_streams, output_dir, dataset_name)
    if per_pub_all:
        pp_sensor = next(iter(per_pub_all))
        pp_data, pp_B = per_pub_all[pp_sensor]
        extreme2_per_publisher(pp_data, pp_B, pp_sensor, dataset_name, output_dir)
    kl_extremes_vs_ours(sensor_streams, per_pub_all, P_our, dataset_name, output_dir)
    u_shaped_curve(sensor_streams, per_pub_all, dataset_name, output_dir)


# ═════════════════════════════════════════════════════════════════════════
#  Summary / dataset runners
# ═════════════════════════════════════════════════════════════════════════

def print_summary(df: pd.DataFrame, name: str):
    print(f"\n{'=' * 90}")
    print(f"RESULTS: {name.upper()}")
    print(f"{'=' * 90}")

    summary = df.groupby(["strategy", "P", "epsilon", "w"]).agg(
        mae=("mae", "mean"),
        nmae=("normalized_mae", "mean"),
        kl=("kl_divergence", "mean"),
        rel_rate=("release_rate", "mean"),
    ).reset_index()

    for strat in sorted(df["strategy"].unique()):
        sd = summary[summary["strategy"] == strat].sort_values("kl")
        print(f"\n--- {strat} (top 5 by KL) ---")
        print(sd[["strategy", "P", "epsilon", "w", "kl", "nmae", "rel_rate"]].head(5).to_string(index=False))

    best = summary.sort_values("kl").head(1).iloc[0]
    print(f"\nBEST (KL): {best['strategy']}, P={int(best['P'])}, "
          f"eps={best['epsilon']}, w={int(best['w'])}")
    print(f"  KL={best['kl']:.6f}, NMAE={best['nmae']:.4f} ({best['nmae']*100:.1f}%)")
    print(f"{'=' * 90}")


def _dataset_max_rows(name: str, args) -> int | None:
    """Map CLI args to the per-dataset row-cap argument."""
    if name == "energy":
        return args.max_energy_timestamps
    if name == "traffic":
        return args.max_traffic_rows
    return args.max_rows


def _dataset_dirs(output_dir: str, name: str, clamp_mode: str) -> dict:
    """Standardized sub-folder layout for one (dataset, clamp_mode) pair."""
    root = os.path.join(output_dir, name, clamp_mode)
    dirs = {
        "root":   root,
        "sweep":  os.path.join(root, "sweep"),
        "intro":  os.path.join(root, "intro"),
        "tuning": os.path.join(root, "tuning"),
        "extras": os.path.join(root, "extras"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def _reclamp_streams_and_pubs(
    streams, per_pubs, dataset_spec, clamp_mode, eps_clip, seed=0,
):
    """
    Apply Definition 3.2 Option A/B to every sensor in a dataset.

    For each sensor:
      1. apply_clamp_option() yields clamped per_pub + updated payload_bound R.
      2. re-derive (aggregates, pub_counts) from the clamped per_pub so the
         downstream sweep sees the clamped values, not the raw ones.

    Returns (clamped_streams, clamped_per_pubs, clamp_meta_per_sensor).
    """
    clamped_streams: dict = {}
    clamped_per_pubs: dict = {}
    meta_per_sensor: dict = {}
    for sensor, (pp, _B_raw) in per_pubs.items():
        try:
            new_pp, R, meta = apply_clamp_option(
                pp, sensor, dataset_spec, clamp_mode,
                eps_clip=eps_clip, seed=seed + hash(sensor) % 10000,
            )
        except ValueError as e:
            logger.warning(f"    clamp {clamp_mode} failed for {sensor}: {e}")
            continue
        meta_per_sensor[sensor] = meta
        clamped_per_pubs[sensor] = (new_pp, R)
        new_agg, new_cnt = _aggregate_from_per_pub(new_pp)
        if len(new_agg) > 10 and R > 0.0:
            clamped_streams[sensor] = (new_agg, new_cnt, R)
    return clamped_streams, clamped_per_pubs, meta_per_sensor


def run_dataset(
    name: str,
    s_values, eps_values, w_values, strategies, output_dir, args,
    clamp_mode: str = "static",
    quick: bool = False, skip_extras: bool = False,
) -> dict:
    """Run the full experiment on one (dataset, clamp_mode) pair.

    Returns a dict of DataFrames for cross-dataset aggregation.
    """
    spec = DATASETS[name]
    logger.info("=" * 72)
    logger.info(f"{name.upper()} / clamp_mode={clamp_mode} :: {spec['label']}")
    logger.info("=" * 72)

    obj = load_dataset_object(name, max_rows=_dataset_max_rows(name, args))
    raw_streams, raw_per_pubs = build_sensor_streams(name, obj)
    if not raw_streams:
        logger.error(f"No valid {name} streams; skipping."); return {}

    streams, per_pubs, clamp_meta = _reclamp_streams_and_pubs(
        raw_streams, raw_per_pubs, spec, clamp_mode, args.eps_clip, seed=args.seed,
    )
    if not streams:
        logger.error(f"No valid {name} streams after {clamp_mode} clamping"); return {}

    # Log the clamp decisions so a reader can audit R per sensor.
    logger.info(f"  clamp[{clamp_mode}] R per sensor:")
    for sensor, meta in clamp_meta.items():
        if meta["mode"] == "static":
            logger.info(f"    {sensor}: [{meta['a_global']:.2f}, {meta['b_global']:.2f}] "
                        f"R={meta['R']:.2f}")
        else:
            logger.info(f"    {sensor}: M={meta['M']:.1f}, eps_clip={meta['eps_clip']:.3f}, "
                        f"sup(b-a)={meta['R']:.2f}")

    dirs = _dataset_dirs(output_dir, name, clamp_mode)

    # Persist the clamp metadata itself so downstream analysis can reproduce
    # exactly which [a_p, b_p] was used per publisher.
    clamp_rows = []
    for sensor, meta in clamp_meta.items():
        if meta["mode"] == "static":
            clamp_rows.append({
                "sensor": sensor, "mode": "static", "eps_clip": 0.0,
                "a": meta["a_global"], "b": meta["b_global"], "R": meta["R"],
                "publisher": "*",
            })
        else:
            for pub_id, (a, b) in meta["per_pub_clamps"].items():
                clamp_rows.append({
                    "sensor": sensor, "mode": "dp_released",
                    "eps_clip": meta["eps_clip"], "M": meta["M"],
                    "a": a, "b": b, "R": meta["R"], "publisher": pub_id,
                })
    if clamp_rows:
        pd.DataFrame(clamp_rows).to_csv(
            os.path.join(dirs["root"], f"{name}_{clamp_mode}_clamps.csv"), index=False,
        )

    # Persist the normative topic manifest for this run so subscribers can see
    # every topic the mechanism publishes on.
    manifest = build_topic_manifest(name, per_pubs)
    if not manifest.empty:
        manifest.to_csv(os.path.join(dirs["root"], f"{name}_topics.csv"), index=False)
        filters = spec.get("subscriber_filters", [])
        if filters:
            pd.DataFrame({"subscriber_filter": filters}).to_csv(
                os.path.join(dirs["root"], f"{name}_subscriber_filters.csv"),
                index=False,
            )

    df = sweep(name, streams, s_values, eps_values, w_values, strategies)
    df["clamp_mode"] = clamp_mode
    df["eps_clip"] = args.eps_clip if clamp_mode == "dp_released" else 0.0
    df.to_csv(os.path.join(dirs["sweep"], "sweep_results.csv"), index=False)
    print_summary(df, f"{spec['label']} [clamp_mode={clamp_mode}]")
    plot_results(df, name, dirs["sweep"], streams)

    results: dict = {"sweep": df}
    run_intro_figures(streams, per_pubs, name, dirs["intro"], P_our=4)

    if not skip_extras:
        w_mid = max(w_values) // 2
        n_weighted_spotlight(streams, name, dirs["extras"], epsilon=1.0, w=w_mid, P=2)
        fig1_df = figure1_reproduction(per_pubs, name, dirs["intro"],
                                       epsilon=1.0, w=w_mid)
        fig1_df = fig1_df.copy()
        fig1_df["clamp_mode"] = clamp_mode
        results["figure1"] = fig1_df
        collusion_experiment(streams, name, dirs["extras"], epsilon=1.0, w=w_mid, P=2,
                             trials_per_c=16 if quick else 32)
        if per_pubs:
            sensor_name = next(iter(per_pubs))
            pp, B = per_pubs[sensor_name]
            dynamic_interval_experiment(
                pp, B, name, sensor_name, dirs["extras"],
                epsilon=1.0, w=w_mid, P=3,
                k_ext_values=(0, 1, 2, 4) if quick else (0, 1, 2, 4, 8),
            )
            tune = tune_hyperparameters(
                pp, B, name, sensor_name, dirs["tuning"],
                epsilon=1.0, w=w_mid,
                strategies=strategies,
                alpha=args.alpha, I_max=args.I_max,
            )
            # Stamp each frame with (dataset, sensor, clamp_mode) for cross-agg.
            for key in ("greedy", "brute_force", "gap_summary"):
                tune[key]["dataset"] = name
                tune[key]["sensor"] = sensor_name
                tune[key]["clamp_mode"] = clamp_mode
            results["tuning_greedy"] = tune["greedy"]
            results["tuning_brute"] = tune["brute_force"]
            results["tuning_gap"] = tune["gap_summary"]
    return results


def cross_dataset_figure1(fig1_dfs: list[pd.DataFrame], output_dir: str):
    """Aggregate per-dataset Figure-1 into one plot + CSV.

    Each row of the combined CSV is one (dataset, P_scope) point.  The plot
    shows KL per P_scope averaged across datasets -- a generalization of paper
    Figure 1 across every real-world stream we evaluate on.
    """
    if not fig1_dfs:
        return
    combined = pd.concat(fig1_dfs, ignore_index=True)
    combined.to_csv(os.path.join(output_dir, "figure1_all_datasets.csv"), index=False)

    # Bucket the middle P-values into canonical labels so averaging lines up.
    def _bucket(row):
        if row["P_label"] == "per-pub":
            return "per-pub"
        if row["P_label"] == "global":
            return "global"
        return f"P={int(row['P_scope'])}"
    combined["bucket"] = combined.apply(_bucket, axis=1)
    order = ["per-pub", "P=2", "P=3", "P=4", "P=6", "P=8", "global"]
    order = [o for o in order if o in combined["bucket"].unique()]

    per_bucket = (
        combined.groupby("bucket")["kl_divergence"]
                .agg(["mean", "std", "count"])
                .reindex(order)
    )
    per_bucket.to_csv(os.path.join(output_dir, "figure1_avg_across_datasets.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    datasets_present = sorted(combined["dataset"].unique())
    palette = plt.cm.tab10(np.linspace(0, 1, max(len(datasets_present), 1)))
    for color, ds in zip(palette, datasets_present):
        sub = combined[combined["dataset"] == ds].set_index("bucket").reindex(order)
        axes[0].plot(sub.index, sub["kl_divergence"],
                     marker="o", label=ds, color=color, alpha=0.85)
    axes[0].set(xlabel="Aggregation scope P", ylabel="KL divergence",
                title="Per-dataset U-shape (higher = more distortion)")
    axes[0].grid(True, alpha=0.3); axes[0].legend(fontsize=8)

    means = per_bucket["mean"]
    stds = per_bucket["std"].fillna(0.0)
    colors = ["#e74c3c"] + ["#2ecc71"] * (len(order) - 2) + ["#e74c3c"]
    bars = axes[1].bar(means.index, means.values, yerr=stds.values,
                       color=colors, alpha=0.85, edgecolor="white", capsize=4)
    for bar, v in zip(bars, means.values):
        if np.isfinite(v):
            axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                         f"{v:.2f}", ha="center", va="bottom", fontsize=9,
                         fontweight="bold")
    axes[1].set(xlabel="Aggregation scope P",
                ylabel="Average KL divergence (mean ± std across datasets)",
                title=f"Figure 1 (avg of {len(datasets_present)} real datasets)")
    axes[1].grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "figure1_all_datasets.png"), dpi=150)
    plt.close()
    logger.info(f"  Cross-dataset Figure-1 saved ({len(datasets_present)} datasets)")


# ═════════════════════════════════════════════════════════════════════════
#  Single-axis experiments (A/B/C): hold every hyperparameter fixed except one
# ═════════════════════════════════════════════════════════════════════════
#
# The paper's Section 5.7 tuning problem has a 4-D hyperparameter space
# (P, Delta_t, A, theta).  The full sweep in run_dataset varies several axes
# at once; these experiments isolate a single axis per run so the effect on
# utility is unambiguous.
#
# A: greedy Algorithm 2 vs naive brute-force enumeration of every P.  For each
#    (strategy, eps, w) combination, records greedy_P / brute_P / speedup
#    across every dataset.
# B: vary w only.  For each (P, eps, strategy) combination, sweep w and record
#    NMAE / KL / release_rate.
# C: vary eps only.  For each (P, w, strategy) combination, sweep eps.

EXPERIMENT_FIXED_COMBOS_A = [
    # For Experiment A these are the (eps, w) points at which we compare
    # greedy-vs-brute; strategy loops through every strategy.
    {"epsilon": 0.5, "w": 8},
    {"epsilon": 1.0, "w": 8},
    {"epsilon": 2.0, "w": 8},
    {"epsilon": 1.0, "w": 4},
    {"epsilon": 1.0, "w": 12},
]

EXPERIMENT_FIXED_COMBOS_B = [
    # Each combo fixes (P, eps, strategy); w is the swept variable.
    {"P": 2, "epsilon": 1.0, "strategy": "uniform"},
    {"P": 4, "epsilon": 1.0, "strategy": "uniform"},
    {"P": 2, "epsilon": 1.0, "strategy": "p_gated_ba"},
    {"P": 4, "epsilon": 1.0, "strategy": "p_gated_ba"},
    {"P": 2, "epsilon": 2.0, "strategy": "n_weighted"},
]
EXPERIMENT_B_W_VALUES = [4, 6, 8, 10, 12, 16]

EXPERIMENT_FIXED_COMBOS_C = [
    # Each combo fixes (P, w, strategy); eps is swept.
    {"P": 2, "w": 8, "strategy": "uniform"},
    {"P": 4, "w": 8, "strategy": "uniform"},
    {"P": 2, "w": 8, "strategy": "p_gated_ba"},
    {"P": 4, "w": 8, "strategy": "p_gated_ba"},
    {"P": 2, "w": 8, "strategy": "n_weighted"},
]
EXPERIMENT_C_EPS_VALUES = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]


def _iter_clamped_per_sensor(datasets, clamp_mode, eps_clip, seed, args):
    """Yield (ds_name, sensor, pp, B, R, streams_agg_cnt) per sensor of each dataset."""
    for ds_name in datasets:
        spec = DATASETS[ds_name]
        try:
            obj = load_dataset_object(ds_name, max_rows=_dataset_max_rows(ds_name, args))
        except FileNotFoundError as e:
            logger.warning(f"Skipping {ds_name}: {e}")
            continue
        raw_streams, raw_per_pubs = build_sensor_streams(ds_name, obj)
        if not raw_per_pubs:
            continue
        streams, per_pubs, _ = _reclamp_streams_and_pubs(
            raw_streams, raw_per_pubs, spec, clamp_mode, eps_clip, seed=seed,
        )
        for sensor in streams:
            agg, cnt, R = streams[sensor]
            pp = per_pubs[sensor][0]
            yield ds_name, sensor, pp, R, (agg, cnt)


def experiment_A_greedy_vs_brute(
    datasets, clamp_mode, output_dir, args,
    fixed_combos=None, strategies=None,
) -> pd.DataFrame:
    """Experiment A: Algorithm 2 greedy hill-climb vs naive full-P enumeration.

    For each (dataset, sensor, strategy, eps, w) point, records both the
    greedy result and the brute-force optimum plus the speedup.
    """
    fixed_combos = fixed_combos or EXPERIMENT_FIXED_COMBOS_A
    strategies = strategies or ALL_STRATEGIES
    rows = []
    for ds_name, sensor, pp, R, _ in _iter_clamped_per_sensor(
            datasets, clamp_mode, args.eps_clip, args.seed, args):
        _, cnt = _rebuild_stream_with_dt(pp, 1)
        p_max = max(2, min(max(cnt), len(pp)))
        for combo in fixed_combos:
            eps = combo["epsilon"]; w = combo["w"]
            for strat in strategies:
                g = greedy_tune_P(pp, R, eps, w, strat, p_max,
                                  alpha=args.alpha, I_max=args.I_max)
                b_df = brute_force_tune_P(pp, R, eps, w, strat, p_max)
                b_best = (b_df.assign(_neg=-b_df["P"])
                          .sort_values(["tuning_loss", "_neg"]).iloc[0])
                g_best = g["best"]
                rows.append({
                    "dataset": ds_name, "sensor": sensor, "strategy": strat,
                    "clamp_mode": clamp_mode,
                    "epsilon": eps, "w": w, "p_max": p_max,
                    "greedy_P": int(g_best["P"]),
                    "greedy_loss": float(g_best["tuning_loss"]),
                    "greedy_nmae": g_best["normalized_mae"],
                    "greedy_kl": g_best["kl_divergence"],
                    "greedy_release_rate": g_best["release_rate"],
                    "greedy_evaluations": g["evaluations"],
                    "greedy_seed_P": g["seed_P"],
                    "brute_P": int(b_best["P"]),
                    "brute_loss": float(b_best["tuning_loss"]),
                    "brute_nmae": b_best["normalized_mae"],
                    "brute_kl": b_best["kl_divergence"],
                    "brute_release_rate": b_best["release_rate"],
                    "brute_evaluations": int(p_max),
                    "gap_loss": float(g_best["tuning_loss"] - b_best["tuning_loss"]),
                    "gap_P": int(g_best["P"] - b_best["P"]),
                    "speedup": float(p_max / max(g["evaluations"], 1)),
                })

    df = pd.DataFrame(rows)
    exp_dir = os.path.join(output_dir, "experiments", "A_greedy_vs_brute")
    os.makedirs(exp_dir, exist_ok=True)
    df.to_csv(os.path.join(exp_dir, "experiment_A_greedy_vs_brute.csv"), index=False)
    logger.info(f"  Experiment A wrote {len(df)} rows -> {exp_dir}")

    # Plot: per-dataset mean speedup (all strategies, all combos).
    if not df.empty:
        datasets_present = sorted(df["dataset"].unique())
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        mean_speedup = df.groupby("dataset")["speedup"].mean().reindex(datasets_present)
        axes[0].bar(mean_speedup.index, mean_speedup.values,
                    color="#3498db", alpha=0.85, edgecolor="white")
        for i, v in enumerate(mean_speedup.values):
            axes[0].text(i, v + 0.05, f"{v:.2f}x", ha="center", fontsize=10, fontweight="bold")
        axes[0].set(ylabel="mean speedup (brute_evals / greedy_evals)",
                    title="Experiment A: Algorithm 2 greedy vs naive enumeration",
                    xlabel="dataset")
        axes[0].grid(True, alpha=0.3, axis="y")
        axes[0].tick_params(axis="x", rotation=30, labelsize=8)

        axes[1].scatter(df["brute_loss"], df["greedy_loss"], s=40, alpha=0.7,
                        c=df["dataset"].astype("category").cat.codes, cmap="tab10")
        mn = min(df["brute_loss"].min(), df["greedy_loss"].min())
        mx = max(df["brute_loss"].max(), df["greedy_loss"].max())
        axes[1].plot([mn, mx], [mn, mx], "k--", lw=0.7, alpha=0.5)
        axes[1].set(xlabel="brute-force optimum loss",
                    ylabel="greedy result loss",
                    title="greedy vs brute-force loss (identity = optimal)")
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, "experiment_A_speedup_and_gap.png"), dpi=150)
        plt.close()
    return df


def experiment_B_vary_w(
    datasets, clamp_mode, output_dir, args,
    fixed_combos=None, w_values=None,
) -> pd.DataFrame:
    """Experiment B: fix (P, eps, strategy), sweep w."""
    fixed_combos = fixed_combos or EXPERIMENT_FIXED_COMBOS_B
    w_values = w_values or EXPERIMENT_B_W_VALUES
    rows = []
    for ds_name, sensor, _, R, (agg, cnt) in _iter_clamped_per_sensor(
            datasets, clamp_mode, args.eps_clip, args.seed, args):
        for combo in fixed_combos:
            for w in w_values:
                res = run_dp_on_stream(
                    agg, cnt, epsilon=combo["epsilon"], window_size=w,
                    min_publishers=combo["P"], payload_bound=R,
                    strategy=combo["strategy"], seed=77,
                )
                m = res["metrics"]
                rows.append({
                    "dataset": ds_name, "sensor": sensor,
                    "clamp_mode": clamp_mode,
                    "strategy": combo["strategy"], "P": combo["P"],
                    "epsilon": combo["epsilon"], "w": w,
                    "normalized_mae": m["normalized_mae"],
                    "kl_divergence": m["kl_divergence"],
                    "release_rate": m["release_rate"],
                    "attribution_advantage": m["attribution_advantage"],
                    "avg_n_tau": float(np.mean(cnt)) if cnt else 0.0,
                })
    df = pd.DataFrame(rows)
    exp_dir = os.path.join(output_dir, "experiments", "B_vary_w")
    os.makedirs(exp_dir, exist_ok=True)
    df.to_csv(os.path.join(exp_dir, "experiment_B_vary_w.csv"), index=False)
    logger.info(f"  Experiment B wrote {len(df)} rows -> {exp_dir}")

    _plot_single_axis_experiment(
        df, x_col="w", x_label="window size  w",
        path=os.path.join(exp_dir, "experiment_B_vary_w.png"),
        title=f"Experiment B [clamp={clamp_mode}]: NMAE and KL vs w",
    )
    return df


def experiment_C_vary_epsilon(
    datasets, clamp_mode, output_dir, args,
    fixed_combos=None, eps_values=None,
) -> pd.DataFrame:
    """Experiment C: fix (P, w, strategy), sweep eps."""
    fixed_combos = fixed_combos or EXPERIMENT_FIXED_COMBOS_C
    eps_values = eps_values or EXPERIMENT_C_EPS_VALUES
    rows = []
    for ds_name, sensor, _, R, (agg, cnt) in _iter_clamped_per_sensor(
            datasets, clamp_mode, args.eps_clip, args.seed, args):
        for combo in fixed_combos:
            for eps in eps_values:
                res = run_dp_on_stream(
                    agg, cnt, epsilon=eps, window_size=combo["w"],
                    min_publishers=combo["P"], payload_bound=R,
                    strategy=combo["strategy"], seed=77,
                )
                m = res["metrics"]
                rows.append({
                    "dataset": ds_name, "sensor": sensor,
                    "clamp_mode": clamp_mode,
                    "strategy": combo["strategy"], "P": combo["P"],
                    "w": combo["w"], "epsilon": eps,
                    "normalized_mae": m["normalized_mae"],
                    "kl_divergence": m["kl_divergence"],
                    "release_rate": m["release_rate"],
                    "attribution_advantage": m["attribution_advantage"],
                    "avg_n_tau": float(np.mean(cnt)) if cnt else 0.0,
                })
    df = pd.DataFrame(rows)
    exp_dir = os.path.join(output_dir, "experiments", "C_vary_epsilon")
    os.makedirs(exp_dir, exist_ok=True)
    df.to_csv(os.path.join(exp_dir, "experiment_C_vary_epsilon.csv"), index=False)
    logger.info(f"  Experiment C wrote {len(df)} rows -> {exp_dir}")

    _plot_single_axis_experiment(
        df, x_col="epsilon", x_label="privacy budget  ε",
        path=os.path.join(exp_dir, "experiment_C_vary_epsilon.png"),
        title=f"Experiment C [clamp={clamp_mode}]: NMAE and KL vs ε",
        logx=True,
    )
    return df


def _plot_single_axis_experiment(df, x_col, x_label, path, title, logx=False):
    """Shared plot for Experiments B/C: one panel per dataset, curves per combo."""
    if df.empty:
        return
    datasets = sorted(df["dataset"].unique())
    ncols = min(3, len(datasets)) or 1
    nrows = (len(datasets) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols * 2, figsize=(5.0 * ncols * 2, 3.6 * nrows),
                             squeeze=False)
    palette = plt.cm.tab10(np.linspace(0, 1, 10))
    for idx, ds_name in enumerate(datasets):
        ax_nmae = axes[idx // ncols][2 * (idx % ncols)]
        ax_kl = axes[idx // ncols][2 * (idx % ncols) + 1]
        sub = df[df["dataset"] == ds_name]
        # One line per (sensor, strategy, P, fixed-others) combo.
        fixed_cols = [c for c in ["strategy", "P", "epsilon", "w"]
                      if c != x_col and c in sub.columns]
        combos = sub[fixed_cols + ["sensor"]].drop_duplicates().to_dict("records")
        for j, combo in enumerate(combos):
            mask = pd.Series(True, index=sub.index)
            for k, v in combo.items():
                mask &= (sub[k] == v)
            line = sub[mask].sort_values(x_col)
            if line.empty:
                continue
            label = f"{combo.get('sensor','')} / {combo.get('strategy','')} P={combo.get('P','')}"
            c = palette[j % 10]
            ax_nmae.plot(line[x_col], line["normalized_mae"], "o-",
                         color=c, lw=1.2, alpha=0.85, label=label)
            ax_kl.plot(line[x_col], line["kl_divergence"], "o-",
                       color=c, lw=1.2, alpha=0.85, label=label)
        for ax, ylabel in [(ax_nmae, "NMAE"), (ax_kl, "KL div.")]:
            ax.set(xlabel=x_label, ylabel=ylabel, title=f"{ds_name} / {ylabel}")
            if logx:
                ax.set_xscale("log")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=5, ncol=2)
    for idx in range(len(datasets), nrows * ncols):
        axes[idx // ncols][2 * (idx % ncols)].set_visible(False)
        axes[idx // ncols][2 * (idx % ncols) + 1].set_visible(False)
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def run_single_axis_experiments(
    datasets, clamp_modes, output_dir, args, strategies, which="ABC",
):
    """Drive Experiments A, B, and C (subset selectable via `which`)."""
    for clamp_mode in clamp_modes:
        logger.info(f"===== Experiments [clamp={clamp_mode}] =====")
        if "A" in which:
            experiment_A_greedy_vs_brute(
                datasets, clamp_mode,
                os.path.join(output_dir, "cross_dataset", clamp_mode),
                args, strategies=strategies,
            )
        if "B" in which:
            experiment_B_vary_w(
                datasets, clamp_mode,
                os.path.join(output_dir, "cross_dataset", clamp_mode),
                args,
            )
        if "C" in which:
            experiment_C_vary_epsilon(
                datasets, clamp_mode,
                os.path.join(output_dir, "cross_dataset", clamp_mode),
                args,
            )


# ═════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════

CLAMP_MODES = ["static", "dp_released"]


def main():
    parser = argparse.ArgumentParser(
        description="Run clamped w-event DP with P-allocation on real-world datasets"
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()) + ["all"],
        default="all",
        help="Dataset to run; 'all' runs every entry in DATASETS",
    )
    parser.add_argument(
        "--clamp-mode",
        choices=CLAMP_MODES + ["both"],
        default="both",
        help="Definition 3.2: 'static' (Option A operator-declared), "
             "'dp_released' (Option B DP-released min/max), 'both' runs each "
             "as a side-by-side experiment.",
    )
    parser.add_argument("--eps-clip", type=float, default=0.1,
                        help="Option B calibration budget epsilon_clip (Def 3.2)")
    parser.add_argument("--alpha", type=float, default=0.25,
                        help="Attribution-advantage target; Algorithm 2 seeds P_0 = ceil(1/alpha)")
    parser.add_argument("--I-max", type=int, default=20,
                        help="Algorithm 2 iteration cap for the greedy hill-climb over P")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for Option B calibration noise")
    parser.add_argument("--quick", action="store_true", help="Reduced sweep for testing")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--max-energy-timestamps", type=int, default=None,
                        help="Cap rows for the energy dataset (MCEC-Thai)")
    parser.add_argument("--max-traffic-rows", type=int, default=None,
                        help="Cap rows per file for the traffic dataset")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Cap rows for wearable/pune/mobility/manufacturing")
    parser.add_argument("--strategies", nargs="+", default=None,
                        help=f"Subset of {ALL_STRATEGIES} to run (defaults to all)")
    parser.add_argument("--skip-extras", action="store_true",
                        help="Skip the n-weighted / collusion / K_ext / tuning experiments")
    parser.add_argument("--tune-only", action="store_true",
                        help="Only run the Section 5.7 hyperparameter tuning")
    parser.add_argument(
        "--experiment",
        choices=["full", "sweep", "tune", "A", "B", "C", "ABC", "none"],
        default="full",
        help="'full' runs sweep + intro + tuning + experiments A/B/C.  "
             "'sweep' is the main grid only.  'tune' is just Algorithm 2 / "
             "brute-force tuning.  A/B/C pick one single-axis experiment.  "
             "'ABC' runs all three single-axis experiments only.",
    )
    args = parser.parse_args()

    if args.quick:
        s_values = [1, 2, 4]
        eps_values = [0.5, 1.0, 2.0]
        w_values = [6, 10]
    else:
        s_values = [1, 2, 4, 6]
        eps_values = [0.5, 1.0, 2.0, 4.0]
        w_values = [4, 8, 10, 12]

    strategies = args.strategies or ALL_STRATEGIES
    invalid = [s for s in strategies if s not in ALL_STRATEGIES]
    if invalid:
        raise SystemExit(f"Unknown strategies: {invalid}.  Valid: {ALL_STRATEGIES}")

    os.makedirs(args.output_dir, exist_ok=True)
    targets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    clamp_modes = CLAMP_MODES if args.clamp_mode == "both" else [args.clamp_mode]

    sweep_frames: list[pd.DataFrame] = []
    figure1_frames: list[pd.DataFrame] = []
    greedy_frames: list[pd.DataFrame] = []
    brute_frames: list[pd.DataFrame] = []
    gap_frames: list[pd.DataFrame] = []

    if args.tune_only:
        for clamp_mode in clamp_modes:
            for name in targets:
                spec = DATASETS[name]
                try:
                    obj = load_dataset_object(name, max_rows=_dataset_max_rows(name, args))
                except FileNotFoundError as e:
                    logger.warning(f"Skipping {name}: {e}"); continue
                first_sensor = spec["sensors"][0]
                raw_streams, raw_per_pubs = build_sensor_streams(
                    name, obj, sensors=[first_sensor])
                if first_sensor not in raw_per_pubs:
                    logger.warning(f"Skipping {name} tuning: no per-pub stream"); continue
                streams_c, per_pubs_c, _ = _reclamp_streams_and_pubs(
                    raw_streams, raw_per_pubs, spec, clamp_mode,
                    args.eps_clip, seed=args.seed,
                )
                if first_sensor not in per_pubs_c:
                    continue
                pp, B = per_pubs_c[first_sensor]
                dirs = _dataset_dirs(args.output_dir, name, clamp_mode)
                tune = tune_hyperparameters(
                    pp, B, name, first_sensor, dirs["tuning"],
                    epsilon=1.0, w=8,
                    strategies=strategies,
                    alpha=args.alpha, I_max=args.I_max,
                )
                for key in ("greedy", "brute_force", "gap_summary"):
                    tune[key]["dataset"] = name
                    tune[key]["sensor"] = first_sensor
                    tune[key]["clamp_mode"] = clamp_mode
                greedy_frames.append(tune["greedy"])
                brute_frames.append(tune["brute_force"])
                gap_frames.append(tune["gap_summary"])
        _write_cross_dataset_tuning(args.output_dir, clamp_modes,
                                    greedy_frames, brute_frames, gap_frames)
        return

    run_main_pipeline = args.experiment in ("full", "sweep")
    run_single_axis = args.experiment in ("full", "ABC", "A", "B", "C")

    if run_main_pipeline:
        for clamp_mode in clamp_modes:
            for name in targets:
                try:
                    results = run_dataset(
                        name, s_values, eps_values, w_values, strategies,
                        args.output_dir, args,
                        clamp_mode=clamp_mode,
                        quick=args.quick, skip_extras=args.skip_extras,
                    )
                except FileNotFoundError as e:
                    logger.warning(f"Skipping {name}: {e}")
                    continue
                if "sweep" in results:
                    sweep_frames.append(results["sweep"])
                if "figure1" in results:
                    figure1_frames.append(results["figure1"])
                if "tuning_greedy" in results:
                    greedy_frames.append(results["tuning_greedy"])
                    brute_frames.append(results["tuning_brute"])
                    gap_frames.append(results["tuning_gap"])

        _write_cross_dataset_sweep_and_fig1(args.output_dir, clamp_modes,
                                            sweep_frames, figure1_frames)
        _write_cross_dataset_tuning(args.output_dir, clamp_modes,
                                    greedy_frames, brute_frames, gap_frames)

    if run_single_axis:
        which = "ABC" if args.experiment in ("full", "ABC") else args.experiment
        run_single_axis_experiments(
            targets, clamp_modes, args.output_dir, args, strategies, which=which,
        )

    logger.info("All experiments complete.")


def _write_cross_dataset_sweep_and_fig1(
    output_dir, clamp_modes, sweep_frames, figure1_frames,
):
    cross_dir = os.path.join(output_dir, "cross_dataset")
    os.makedirs(cross_dir, exist_ok=True)
    if sweep_frames:
        combined = pd.concat(sweep_frames, ignore_index=True)
        combined.to_csv(os.path.join(cross_dir, "combined_sweep_results.csv"),
                        index=False)
        # Per-clamp-mode cross-dataset comparison figure.
        for clamp_mode in clamp_modes:
            sub = combined[combined["clamp_mode"] == clamp_mode]
            if sub.empty:
                continue
            datasets_present = sorted(sub["dataset"].unique())
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            for ax, metric, ylabel in [
                (axes[0], "normalized_mae", "Normalized MAE"),
                (axes[1], "kl_divergence", "KL divergence"),
            ]:
                for ds in datasets_present:
                    g = (sub[sub["dataset"] == ds]
                         .groupby("epsilon")[metric].mean().sort_index())
                    ax.plot(g.index, g.values, marker="o", label=ds)
                ax.set(xlabel="eps", ylabel=ylabel, title=f"{ylabel} vs eps")
                ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
            fig.suptitle(f"Cross-Dataset [clamp={clamp_mode}]", fontsize=13)
            plt.tight_layout()
            plt.savefig(os.path.join(cross_dir, f"cross_dataset_{clamp_mode}.png"),
                        dpi=150)
            plt.close()

    if figure1_frames:
        combined_f1 = pd.concat(figure1_frames, ignore_index=True)
        combined_f1.to_csv(os.path.join(cross_dir, "figure1_all_datasets.csv"),
                           index=False)
        for clamp_mode in clamp_modes:
            sub = combined_f1[combined_f1["clamp_mode"] == clamp_mode]
            if sub.empty:
                continue
            cross_dir_mode = os.path.join(cross_dir, clamp_mode)
            os.makedirs(cross_dir_mode, exist_ok=True)
            cross_dataset_figure1(
                [sub[sub["dataset"] == ds] for ds in sub["dataset"].unique()],
                cross_dir_mode,
            )


def _write_cross_dataset_tuning(output_dir, clamp_modes,
                                greedy_frames, brute_frames, gap_frames):
    cross_dir = os.path.join(output_dir, "cross_dataset")
    os.makedirs(cross_dir, exist_ok=True)
    if greedy_frames:
        pd.concat(greedy_frames, ignore_index=True).to_csv(
            os.path.join(cross_dir, "combined_tuning_greedy.csv"), index=False,
        )
    if brute_frames:
        pd.concat(brute_frames, ignore_index=True).to_csv(
            os.path.join(cross_dir, "combined_tuning_brute_force.csv"), index=False,
        )
    if gap_frames:
        all_gap = pd.concat(gap_frames, ignore_index=True)
        all_gap.to_csv(os.path.join(cross_dir, "combined_tuning_gap_summary.csv"),
                       index=False)
        # Headline best per (dataset, clamp_mode) by brute-force loss; break
        # loss ties by higher P (stronger identity protection on a plateau).
        best_brute = (all_gap.assign(_neg_P=-all_gap["brute_P"])
                      .sort_values(["brute_loss", "_neg_P"])
                      .groupby(["dataset", "clamp_mode"], as_index=False).first()
                      .drop(columns=["_neg_P"]))
        best_brute.to_csv(os.path.join(cross_dir, "tuning_best_per_dataset.csv"),
                          index=False)
        # Cross-mode gap plot: one panel per clamp_mode; x=dataset, y=gap_loss.
        fig, axes = plt.subplots(1, len(clamp_modes), figsize=(6.5 * len(clamp_modes), 4.5),
                                 sharey=True, squeeze=False)
        for idx, clamp_mode in enumerate(clamp_modes):
            ax = axes[0][idx]
            sub = all_gap[all_gap["clamp_mode"] == clamp_mode]
            if sub.empty:
                ax.set_visible(False); continue
            pivot = sub.pivot_table(index="dataset", columns="strategy",
                                    values="gap_loss", aggfunc="mean")
            pivot.plot(kind="bar", ax=ax, width=0.85, alpha=0.85)
            ax.set(title=f"greedy − brute-force loss gap  [{clamp_mode}]",
                   ylabel="loss gap (greedy − brute)")
            ax.axhline(0, color="black", lw=0.8, alpha=0.6)
            ax.grid(True, alpha=0.3, axis="y")
            ax.legend(fontsize=7, loc="best")
        plt.tight_layout()
        plt.savefig(os.path.join(cross_dir, "tuning_greedy_vs_brute_gap.png"), dpi=150)
        plt.close()


if __name__ == "__main__":
    main()
