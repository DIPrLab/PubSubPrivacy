#!/usr/bin/env python3
"""
Data loading utilities for the two real-world IEEE datasets.

  Dataset 1 — MCEC-Thai Electric Consumption
    12 circuit breakers as publishers, 5-minute aggregation windows.

  Dataset 2 — Colorado Springs Traffic Intersection
    4 EVO radars + 2 OS1 lidars as publishers, 10-second windows.

These functions are imported by run_experiment.py and generate_intro_figures.py.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
ENERGY_CSV = os.path.join(
    DATA_DIR,
    "Multi-Circuit Electric Consumption Data for Application of Energy Disaggregation.csv",
)
TRAFFIC_DIR = os.path.join(DATA_DIR, "2024_12_20")


# ── Energy dataset ──────────────────────────────────────────────────────────

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
    """Load and clean the MCEC-Thai electric consumption CSV."""
    logger.info(f"Loading energy dataset from {ENERGY_CSV}")
    df = pd.read_csv(ENERGY_CSV, low_memory=False)
    df["Time"] = pd.to_datetime(df["Time"], format="mixed", dayfirst=False)
    df = df.sort_values("Time").reset_index(drop=True)
    if max_timestamps and len(df) > max_timestamps:
        df = df.iloc[:max_timestamps]
    logger.info(f"  Loaded {len(df)} timestamps, {df['Time'].min()} to {df['Time'].max()}")
    return df


def build_energy_streams(
    df: pd.DataFrame,
    sensor_type: str = "power_kw",
    window_minutes: int = 5,
) -> tuple[list[float], list[int], float]:
    """
    Build aggregate pub/sub streams from the energy dataset.

    Each circuit is a publisher; readings are aggregated per time window.

    Returns (aggregates, pub_counts, payload_bound).
    """
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
            aggregates.append(0.0)
            pub_counts.append(0)

    payload_bound = float(np.ptp(all_values)) if all_values else 1.0
    if payload_bound == 0:
        payload_bound = 1.0

    logger.info(
        f"  Energy [{sensor_type}]: {len(aggregates)} windows, "
        f"B={payload_bound:.4f}, avg pubs={np.mean(pub_counts):.1f}"
    )
    return aggregates, pub_counts, payload_bound


def build_energy_per_publisher(
    df: pd.DataFrame,
    sensor_type: str = "power_kw",
    window_minutes: int = 5,
) -> tuple[dict[str, list[float | None]], float]:
    """
    Build per-publisher (per-circuit) time series for the energy dataset.

    Returns (per_pub, payload_bound) where
      per_pub[circuit_name] = [value_or_None per window].
    """
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
                per_pub[ct].append(float(val))
                all_values.append(float(val))
            else:
                per_pub[ct].append(None)

    payload_bound = float(np.ptp(all_values)) if all_values else 1.0
    if payload_bound == 0:
        payload_bound = 1.0
    return per_pub, payload_bound


# ── Traffic dataset ─────────────────────────────────────────────────────────

TRAFFIC_SENSORS_META = {
    "speed":        {"col": "Speed",    "unit": "m/s"},
    "object_count": {"col": None,       "unit": "count"},
    "position_x":   {"col": "PositionX", "unit": "m"},
    "heading":      {"col": "HeadingDeg_DERIVED", "unit": "deg"},
}


def load_traffic_dataset(
    max_rows_per_file: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Load all radar and lidar CSV files from the traffic dataset."""
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


def build_traffic_streams(
    sensors: dict[str, pd.DataFrame],
    metric: str = "speed",
    window_seconds: int = 10,
) -> tuple[list[float], list[int], float]:
    """
    Build aggregate pub/sub streams from the traffic dataset.

    Each radar/lidar is a publisher; detections are aggregated per time window.

    Returns (aggregates, pub_counts, payload_bound).
    """
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
            aggregates.append(0.0)
            pub_counts.append(0)

    payload_bound = float(np.ptp(all_values)) if all_values else 1.0
    if payload_bound == 0:
        payload_bound = 1.0

    logger.info(
        f"  Traffic [{metric}]: {len(aggregates)} windows, "
        f"B={payload_bound:.2f}, avg pubs={np.mean(pub_counts):.1f}"
    )
    return aggregates, pub_counts, payload_bound


def build_traffic_per_publisher(
    sensors: dict[str, pd.DataFrame],
    metric: str = "speed",
    window_seconds: int = 10,
) -> tuple[dict[str, list[float | None]], float]:
    """
    Build per-publisher (per-sensor) time series for the traffic dataset.

    Returns (per_pub, payload_bound) where
      per_pub[sensor_name] = [value_or_None per window].
    """
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
                per_pub[sensor_name].append(None)
                continue

            if metric == "object_count":
                val = float(window_df["ObjectId"].nunique())
            elif metric == "speed":
                val = float(window_df["Speed"].mean())
            elif metric == "position_x":
                val = float(window_df["PositionX"].mean())
            else:
                per_pub[sensor_name].append(None)
                continue

            if np.isfinite(val):
                per_pub[sensor_name].append(val)
                all_values.append(val)
            else:
                per_pub[sensor_name].append(None)

    payload_bound = float(np.ptp(all_values)) if all_values else 1.0
    if payload_bound == 0:
        payload_bound = 1.0
    return per_pub, payload_bound
