#!/usr/bin/env python3
"""
Real-data experimental evaluation of S-sensitive w-event differential privacy
using two IEEE research datasets:

  1. MCEC-Thai: Multi-Circuit Electric Consumption from a Thailand home
     - 11 circuit breakers publishing voltage, current, power readings
     - Scenario: smart building energy monitoring pub/sub

  2. Colorado Springs Traffic Intersection: Multi-sensor object detection
     - 4 EVO radar + 2 OS1 lidar sensors at a traffic intersection
     - Scenario: smart city traffic monitoring pub/sub

Each dataset is loaded, converted into realistic aggregate pub/sub streams
(multiple publishers -> topic -> aggregate), and run through the DP engine
to evaluate privacy-utility tradeoffs on real-world IoT data.

Usage:
  python run_real_data_experiment.py                # full sweep on both datasets
  python run_real_data_experiment.py --quick         # reduced parameter sweep
  python run_real_data_experiment.py --dataset energy # only energy dataset
  python run_real_data_experiment.py --dataset traffic # only traffic dataset
"""

from __future__ import annotations

import argparse
import itertools
import logging
import os
import sys

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
    compute_utility_metrics,
    compute_kl_divergence,
    compute_windowed_kl_divergence,
    compute_global_utility,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
ENERGY_CSV = os.path.join(
    DATA_DIR,
    "Multi-Circuit Electric Consumption Data for Application of Energy Disaggregation.csv",
)
TRAFFIC_DIR = os.path.join(DATA_DIR, "2024_12_20")


# ---------------------------------------------------------------------------
# Dataset 1: MCEC-Thai Electric Consumption
# ---------------------------------------------------------------------------
# Each circuit (CT1..CT10, CT13, CT17) acts as a separate publisher.
# We extract per-circuit kW readings and aggregate them into pub/sub streams
# grouped by time windows — simulating a building management system where
# circuit breaker sensors publish power readings at 1-minute intervals.

ENERGY_CIRCUITS = [
    "CT1", "CT2", "CT3", "CT4", "CT5",
    "CT6", "CT7", "CT8", "CT9", "CT10",
    "CT13", "CT17",
]

# Sensor topics for the energy scenario
ENERGY_SENSORS = {
    "power_kw":      {"col_suffix": "_kW",    "unit": "kW"},
    "voltage":       {"col_suffix": "_V",     "unit": "V"},
    "current":       {"col_suffix": "_A",     "unit": "A"},
    "reactive_power": {"col_suffix": "_kVar+", "unit": "kVar"},
    "power_factor":  {"col_suffix": "_PF",    "unit": "PF"},
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

    Simulates a pub/sub topic where all circuits publish their readings,
    and the broker aggregates them per time window.

    Returns:
        aggregates: list of mean values per window
        pub_counts: list of active publisher counts per window
        payload_bound: range of the sensor domain
    """
    col_suffix = ENERGY_SENSORS[sensor_type]["col_suffix"]

    # Extract per-circuit columns
    circuit_cols = []
    for ct in ENERGY_CIRCUITS:
        col = f"{ct}{col_suffix}"
        if col in df.columns:
            circuit_cols.append(col)

    if not circuit_cols:
        raise ValueError(f"No columns found for sensor type {sensor_type}")

    # Convert to numeric, coerce errors
    for col in circuit_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Group by time windows and compute aggregates
    df = df.set_index("Time")
    resampled = df[circuit_cols].resample(f"{window_minutes}min")

    aggregates = []
    pub_counts = []
    all_values = []

    for _, window_df in resampled:
        if window_df.empty:
            continue
        # Each circuit that has non-null, non-zero data is an active publisher
        means_per_circuit = window_df.mean()
        active = means_per_circuit.dropna()
        active = active[active != 0]

        n_active = len(active)
        if n_active > 0:
            agg_value = float(active.mean())
            aggregates.append(agg_value)
            pub_counts.append(n_active)
            all_values.extend(active.values)
        else:
            aggregates.append(0.0)
            pub_counts.append(0)

    # Compute payload bound from actual data range
    if all_values:
        payload_bound = float(np.max(all_values) - np.min(all_values))
        if payload_bound == 0:
            payload_bound = 1.0
    else:
        payload_bound = 1.0

    logger.info(
        f"  Energy stream [{sensor_type}]: {len(aggregates)} windows, "
        f"payload_bound={payload_bound:.4f}, "
        f"avg publishers={np.mean(pub_counts):.1f}"
    )
    return aggregates, pub_counts, payload_bound


# ---------------------------------------------------------------------------
# Dataset 2: Colorado Springs Traffic Intersection
# ---------------------------------------------------------------------------
# 4 EVO radars and 2 OS1 lidars monitor a traffic intersection.
# Each sensor acts as a publisher. We aggregate per time window to simulate
# a smart city pub/sub system monitoring vehicle speeds, counts, etc.

TRAFFIC_SENSORS = {
    "speed":        {"col": "Speed",    "unit": "m/s"},
    "object_count": {"col": None,       "unit": "count"},  # derived
    "position_x":   {"col": "PositionX", "unit": "m"},
    "heading":      {"col": "HeadingDeg_DERIVED", "unit": "deg"},  # radar only
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
        logger.info(f"  {sensor_name}: {len(df)} detections, {df['Time'].min()} to {df['Time'].max()}")

    return sensors


def build_traffic_streams(
    sensors: dict[str, pd.DataFrame],
    metric: str = "speed",
    window_seconds: int = 10,
) -> tuple[list[float], list[int], float]:
    """
    Build aggregate pub/sub streams from traffic sensor data.

    Each sensor (radar/lidar) is a publisher. Per time window, we compute
    the mean of the chosen metric across all sensors that reported data.

    For 'object_count', we count unique ObjectIds per sensor per window.

    Returns:
        aggregates: mean metric value per window
        pub_counts: number of sensors with data per window
        payload_bound: range of the metric domain
    """
    # Find the global time range across all sensors
    all_times = []
    for df in sensors.values():
        all_times.extend(df["Time"].values)
    t_min = pd.Timestamp(min(all_times))
    t_max = pd.Timestamp(max(all_times))

    # Create time bins
    freq = pd.Timedelta(seconds=window_seconds)
    bins = pd.date_range(start=t_min, end=t_max + freq, freq=freq)

    aggregates = []
    pub_counts = []
    all_values = []

    for i in range(len(bins) - 1):
        w_start, w_end = bins[i], bins[i + 1]
        sensor_values = []

        for sensor_name, df in sensors.items():
            mask = (df["Time"] >= w_start) & (df["Time"] < w_end)
            window_df = df[mask]

            if window_df.empty:
                continue

            if metric == "object_count":
                # Count unique objects detected by this sensor in this window
                val = float(window_df["ObjectId"].nunique())
            elif metric == "speed":
                val = float(window_df["Speed"].mean())
            elif metric == "position_x":
                val = float(window_df["PositionX"].mean())
            elif metric == "heading":
                col = "HeadingDeg_DERIVED" if "HeadingDeg_DERIVED" in window_df.columns else "HeadingDeg"
                if col in window_df.columns:
                    val = float(window_df[col].mean())
                else:
                    continue
            else:
                raise ValueError(f"Unknown traffic metric: {metric}")

            if np.isfinite(val):
                sensor_values.append(val)

        n_active = len(sensor_values)
        if n_active > 0:
            agg = float(np.mean(sensor_values))
            aggregates.append(agg)
            pub_counts.append(n_active)
            all_values.extend(sensor_values)
        else:
            aggregates.append(0.0)
            pub_counts.append(0)

    if all_values:
        payload_bound = float(np.max(all_values) - np.min(all_values))
        if payload_bound == 0:
            payload_bound = 1.0
    else:
        payload_bound = 1.0

    logger.info(
        f"  Traffic stream [{metric}]: {len(aggregates)} windows, "
        f"payload_bound={payload_bound:.2f}, "
        f"avg publishers={np.mean(pub_counts):.1f}"
    )
    return aggregates, pub_counts, payload_bound


# ---------------------------------------------------------------------------
# DP Engine Runner (shared with run_experiment.py)
# ---------------------------------------------------------------------------

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
    """Run the DP mechanism on a pre-generated aggregate stream."""
    np.random.seed(seed)

    config = PrivacyConfig(
        epsilon=float(epsilon),
        window_size=int(window_size),
        min_publishers=int(min_publishers),
        payload_bound=float(payload_bound),
        strategy=BudgetStrategy(strategy),
    )
    stream = StreamState(config=config)

    for agg, n_pub in zip(aggregates, pub_counts):
        stream.release(agg, n_pub)

    metrics = compute_utility_metrics(stream.true_values, stream.noisy_values)
    metrics["normalized_mae"] = (
        metrics["mae"] / payload_bound if payload_bound > 0 else float("nan")
    )

    # KL: only use timestamps where budget was actually spent
    true_for_kl = []
    noisy_for_kl = []
    for t, n, b in zip(stream.true_values, stream.noisy_values, stream.budgets_spent):
        if b > 0 and t is not None and n is not None:
            true_for_kl.append(t)
            noisy_for_kl.append(n)

    if len(true_for_kl) < 10:
        true_for_kl = stream.true_values
        noisy_for_kl = stream.noisy_values

    kl_global = compute_kl_divergence(true_for_kl, noisy_for_kl)
    kl_windowed = compute_windowed_kl_divergence(
        stream.true_values, stream.noisy_values, window_size
    )
    u_global = compute_global_utility(
        stream.true_values, stream.noisy_values, window_size
    )
    metrics["kl_divergence"] = kl_global
    metrics["kl_global_utility"] = u_global

    return {
        "metrics": metrics,
        "true_values": stream.true_values,
        "noisy_values": stream.noisy_values,
        "budgets_spent": stream.budgets_spent,
        "kl_windowed": kl_windowed,
    }


# ---------------------------------------------------------------------------
# Parameter Sweep
# ---------------------------------------------------------------------------

def sweep_real_data(
    dataset_name: str,
    streams: dict[str, tuple[list[float], list[int], float]],
    s_values: list[int],
    epsilon_values: list[float],
    w_values: list[int],
    strategies: list[str],
) -> pd.DataFrame:
    """
    Run a parameter sweep over real data streams.

    streams: dict mapping sensor_name -> (aggregates, pub_counts, payload_bound)
    """
    rows = []
    combos = list(itertools.product(
        streams.keys(), s_values, epsilon_values, w_values, strategies
    ))
    total = len(combos)

    for i, (sensor_name, s_val, eps, w, strat) in enumerate(combos):
        aggregates, pub_counts, payload_bound = streams[sensor_name]

        result = run_dp_on_stream(
            aggregates, pub_counts,
            epsilon=eps,
            window_size=w,
            min_publishers=s_val,
            payload_bound=payload_bound,
            strategy=strat,
            seed=i,
        )
        rows.append({
            "dataset": dataset_name,
            "sensor": sensor_name,
            "S": s_val,
            "epsilon": eps,
            "w": w,
            "strategy": strat,
            "mae": result["metrics"]["mae"],
            "rmse": result["metrics"]["rmse"],
            "relative_error": result["metrics"]["relative_error"],
            "normalized_mae": result["metrics"]["normalized_mae"],
            "kl_divergence": result["metrics"]["kl_divergence"],
            "kl_global_utility": result["metrics"]["kl_global_utility"],
            "noise_scale_theoretical": payload_bound * w / (s_val * eps),
            "payload_bound": payload_bound,
            "num_timestamps": len(aggregates),
            "avg_publishers": float(np.mean(pub_counts)),
        })

        if (i + 1) % 50 == 0:
            logger.info(f"  [{dataset_name}] Sweep progress: {i + 1}/{total}")

    logger.info(f"  [{dataset_name}] Sweep complete: {total} configurations")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_real_data_results(
    df: pd.DataFrame,
    dataset_name: str,
    output_dir: str,
    streams: dict[str, tuple[list[float], list[int], float]],
):
    """Generate all plots for a single dataset's results."""
    os.makedirs(output_dir, exist_ok=True)
    sensor_names = sorted(df["sensor"].unique())
    strategies = sorted(df["strategy"].unique())

    # --- 1. MAE vs Epsilon by Strategy (first sensor, all S values) ---
    first_sensor = sensor_names[0]
    w_mid = int(sorted(df["w"].unique())[len(df["w"].unique()) // 2])

    fig, axes = plt.subplots(1, len(strategies), figsize=(5 * len(strategies), 5), sharey=True)
    if len(strategies) == 1:
        axes = [axes]
    subset = df[(df["sensor"] == first_sensor) & (df["w"] == w_mid)]

    for ax, strat in zip(axes, strategies):
        strat_data = subset[subset["strategy"] == strat]
        for s_val in sorted(strat_data["S"].unique()):
            s_data = strat_data[strat_data["S"] == s_val].sort_values("epsilon")
            ax.plot(s_data["epsilon"], s_data["mae"], marker="o", label=f"S={s_val}")
        ax.set_xlabel("Privacy Budget (epsilon)")
        ax.set_ylabel("Mean Absolute Error")
        ax.set_title(f"Strategy: {strat}")
        ax.legend()
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{dataset_name}: MAE vs epsilon [{first_sensor}, w={w_mid}]", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_mae_vs_epsilon.png"), dpi=150)
    plt.close()

    # --- 2. MAE vs S by Sensor Type ---
    eps_mid = float(sorted(df["epsilon"].unique())[len(df["epsilon"].unique()) // 2])
    n_sensors = len(sensor_names)
    ncols = min(3, n_sensors)
    nrows = (n_sensors + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    subset = df[(df["epsilon"] == eps_mid) & (df["w"] == w_mid) & (df["strategy"] == "uniform")]

    for idx, sensor_name in enumerate(sensor_names):
        ax = axes[idx // ncols][idx % ncols]
        sensor_data = subset[subset["sensor"] == sensor_name].sort_values("S")
        ax.bar(sensor_data["S"].astype(str), sensor_data["mae"], color="steelblue", alpha=0.8)
        ax.set_xlabel("Sensitivity Parameter (S)")
        ax.set_ylabel("MAE")
        ax.set_title(sensor_name)
        ax.grid(True, alpha=0.3, axis="y")

    # Hide unused axes
    for idx in range(n_sensors, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(f"{dataset_name}: MAE vs S (eps={eps_mid}, w={w_mid}, uniform)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_mae_vs_S.png"), dpi=150)
    plt.close()

    # --- 3. Strategy Comparison ---
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    s_mid = int(sorted(df["S"].unique())[len(df["S"].unique()) // 2])
    subset = df[(df["epsilon"] == eps_mid) & (df["w"] == w_mid) & (df["S"] == s_mid)]
    colors = ["#4c72b0", "#dd8452", "#55a868"]

    for idx, sensor_name in enumerate(sensor_names):
        ax = axes[idx // ncols][idx % ncols]
        sensor_data = subset[subset["sensor"] == sensor_name]
        strats = sensor_data["strategy"].unique()
        mae_vals = [sensor_data[sensor_data["strategy"] == s]["mae"].values[0]
                    for s in strats if len(sensor_data[sensor_data["strategy"] == s]) > 0]
        ax.bar(strats[:len(mae_vals)], mae_vals, color=colors[:len(mae_vals)], alpha=0.8)
        ax.set_ylabel("MAE")
        ax.set_title(sensor_name)
        ax.grid(True, alpha=0.3, axis="y")

    for idx in range(n_sensors, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(f"{dataset_name}: Strategy Comparison (eps={eps_mid}, w={w_mid}, S={s_mid})", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_strategy_comparison.png"), dpi=150)
    plt.close()

    # --- 4. Time Series: True vs Noisy (one per sensor, uniform, mid params) ---
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)

    for idx, sensor_name in enumerate(sensor_names):
        ax = axes[idx // ncols][idx % ncols]
        aggregates, pub_counts, payload_bound = streams[sensor_name]

        result = run_dp_on_stream(
            aggregates, pub_counts,
            epsilon=eps_mid, window_size=w_mid, min_publishers=s_mid,
            payload_bound=payload_bound, strategy="uniform", seed=99,
        )

        # Show a representative slice (up to 200 points)
        n_show = min(200, len(result["true_values"]))
        t = range(n_show)
        ax.plot(t, result["true_values"][:n_show], "b-", alpha=0.7, label="True", linewidth=1.2)
        noisy_slice = result["noisy_values"][:n_show]
        noisy_x = [i for i, v in enumerate(noisy_slice) if v is not None]
        noisy_y = [v for v in noisy_slice if v is not None]
        ax.plot(noisy_x, noisy_y, "r-", alpha=0.5, label="Noisy (DP)", linewidth=1)
        ax.set_xlabel("Time Window")
        ax.set_ylabel(sensor_name)
        ax.set_title(f"{sensor_name} (eps={eps_mid}, S={s_mid})")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for idx in range(n_sensors, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(f"{dataset_name}: True vs DP-Protected Streams", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_timeseries.png"), dpi=150)
    plt.close()

    # --- 5. Budget Utilization ---
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)

    for idx, sensor_name in enumerate(sensor_names):
        ax = axes[idx // ncols][idx % ncols]
        aggregates, pub_counts, payload_bound = streams[sensor_name]

        for strat in strategies:
            result = run_dp_on_stream(
                aggregates, pub_counts,
                epsilon=eps_mid, window_size=w_mid, min_publishers=s_mid,
                payload_bound=payload_bound, strategy=strat, seed=42,
            )
            budgets = result["budgets_spent"]
            window_sums = []
            for i in range(min(200, len(budgets))):
                start = max(0, i - w_mid + 1)
                window_sums.append(sum(budgets[start:i + 1]))
            ax.plot(range(len(window_sums)), window_sums, label=strat, alpha=0.8)

        ax.axhline(y=eps_mid, color="red", linestyle="--", alpha=0.5, label=f"Budget limit (eps={eps_mid})")
        ax.set_xlabel("Time Window")
        ax.set_ylabel("Budget in Window")
        ax.set_title(sensor_name)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    for idx in range(n_sensors, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(f"{dataset_name}: Sliding Window Budget Utilization", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_budget_utilization.png"), dpi=150)
    plt.close()

    # --- 6. KL Divergence vs Epsilon ---
    fig, axes = plt.subplots(1, len(strategies), figsize=(5 * len(strategies), 5), sharey=True)
    if len(strategies) == 1:
        axes = [axes]
    subset = df[(df["sensor"] == first_sensor) & (df["w"] == w_mid)]

    for ax, strat in zip(axes, strategies):
        strat_data = subset[subset["strategy"] == strat]
        for s_val in sorted(strat_data["S"].unique()):
            s_data = strat_data[strat_data["S"] == s_val].sort_values("epsilon")
            ax.plot(s_data["epsilon"], s_data["kl_divergence"], marker="o", label=f"S={s_val}")
        ax.set_xlabel("Privacy Budget (epsilon)")
        ax.set_ylabel("KL Divergence")
        ax.set_title(f"Strategy: {strat}")
        ax.legend()
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{dataset_name}: KL Divergence vs epsilon [{first_sensor}, w={w_mid}]", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_kl_vs_epsilon.png"), dpi=150)
    plt.close()

    # --- 7. KL vs MAE Tradeoff Scatter ---
    fig, ax = plt.subplots(figsize=(10, 7))
    subset = df[(df["w"] == w_mid) & (df["strategy"] == "uniform")]
    cmap = plt.cm.tab10
    for i, sensor_name in enumerate(sensor_names):
        s_data = subset[subset["sensor"] == sensor_name]
        ax.scatter(
            s_data["mae"], s_data["kl_divergence"],
            c=[cmap(i)] * len(s_data), alpha=0.6, s=s_data["S"] * 15,
            label=sensor_name, edgecolors="white", linewidth=0.5,
        )

    ax.set_xlabel("Mean Absolute Error (MAE)")
    ax.set_ylabel("KL Divergence")
    ax.set_title(f"{dataset_name}: Privacy-Utility Tradeoff (w={w_mid}, uniform; dot size = S)")
    ax.legend(fontsize=7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_kl_vs_mae_tradeoff.png"), dpi=150)
    plt.close()

    # --- 8. KL Comparison Heatmap ---
    w_values = sorted(df["w"].unique())
    eps_values = sorted(df["epsilon"].unique())
    s_values = sorted(df["S"].unique())

    agg = df.groupby(["w", "epsilon", "S", "strategy"]).agg(
        kl_mean=("kl_divergence", "mean"),
    ).reset_index()

    row_labels = [(w, e) for w in w_values for e in eps_values]
    col_labels = [(s, st) for s in s_values for st in strategies]

    grid = np.full((len(row_labels), len(col_labels)), np.nan)
    for i, (w, e) in enumerate(row_labels):
        for j, (s, st) in enumerate(col_labels):
            match = agg[(agg["w"] == w) & (agg["epsilon"] == e) &
                        (agg["S"] == s) & (agg["strategy"] == st)]
            if len(match) == 1:
                grid[i, j] = match["kl_mean"].values[0]

    fig, ax = plt.subplots(
        figsize=(max(14, len(col_labels) * 1.6), max(8, len(row_labels) * 0.5))
    )
    im = ax.imshow(grid, aspect="auto", cmap="YlOrRd")
    fig.colorbar(im, ax=ax, label="Mean KL Divergence")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels([f"S={s}\n{st}" for s, st in col_labels], fontsize=7, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels([f"w={w}, eps={e}" for w, e in row_labels], fontsize=8)

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = grid[i, j]
            if np.isfinite(val):
                color = "white" if val > np.nanmedian(grid) else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=6, color=color, fontweight="bold")

    ax.set_title(f"{dataset_name}: KL Divergence Across All Scenarios", fontsize=13, pad=12)
    ax.set_xlabel("Sensitivity (S) x Strategy")
    ax.set_ylabel("Window Size (w) x Privacy Budget (epsilon)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_kl_heatmap.png"), dpi=200)
    plt.close()

    # --- 9. Per-window KL time series ---
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)

    for idx, sensor_name in enumerate(sensor_names):
        ax = axes[idx // ncols][idx % ncols]
        aggregates, pub_counts, payload_bound = streams[sensor_name]

        result = run_dp_on_stream(
            aggregates, pub_counts,
            epsilon=eps_mid, window_size=w_mid, min_publishers=s_mid,
            payload_bound=payload_bound, strategy="uniform", seed=99,
        )

        kl_w = result["kl_windowed"]
        n_show = min(200, len(kl_w))
        ax.plot(range(n_show), kl_w[:n_show], "m-", alpha=0.7, linewidth=1.2)
        valid_kl = [k for k in kl_w if np.isfinite(k)]
        if valid_kl:
            mean_kl = np.mean(valid_kl)
            ax.axhline(y=mean_kl, color="red", linestyle="--", alpha=0.5,
                       label=f"U_global={mean_kl:.4f}")
        ax.set_xlabel("Window Position")
        ax.set_ylabel("KL Divergence")
        ax.set_title(sensor_name)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for idx in range(n_sensors, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(f"{dataset_name}: Per-Window KL Divergence Over Time", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_kl_windowed.png"), dpi=150)
    plt.close()

    logger.info(f"  All plots saved to {output_dir}/")


# ---------------------------------------------------------------------------
# Summary Table
# ---------------------------------------------------------------------------

def print_summary_table(df: pd.DataFrame, dataset_name: str):
    """Print a summary table of results to console."""
    print(f"\n{'=' * 90}")
    print(f"REAL DATA EXPERIMENT RESULTS: {dataset_name.upper()}")
    print(f"{'=' * 90}")

    summary = df.groupby(["strategy", "S", "epsilon", "w"]).agg(
        mae_mean=("mae", "mean"),
        nmae_mean=("normalized_mae", "mean"),
        rmse_mean=("rmse", "mean"),
        kl_div_mean=("kl_divergence", "mean"),
        kl_global_mean=("kl_global_utility", "mean"),
    ).reset_index()

    for strat in sorted(df["strategy"].unique()):
        strat_data = summary[summary["strategy"] == strat].sort_values("kl_div_mean")
        print(f"\n--- Strategy: {strat} (top 5 configs by KL divergence) ---")
        cols = ["strategy", "S", "epsilon", "w", "kl_div_mean", "kl_global_mean", "nmae_mean"]
        print(strat_data[cols].head(5).to_string(index=False))

    best_kl = summary.sort_values("kl_div_mean").head(1).iloc[0]
    print(f"\n{'=' * 90}")
    print(f"BEST CONFIG (by KL): strategy={best_kl['strategy']}, S={int(best_kl['S'])}, "
          f"epsilon={best_kl['epsilon']}, w={int(best_kl['w'])}")
    print(f"  Mean KL Divergence: {best_kl['kl_div_mean']:.6f}, "
          f"Mean U_global: {best_kl['kl_global_mean']:.6f}")
    print(f"  Mean Normalized MAE: {best_kl['nmae_mean']:.4f} "
          f"({best_kl['nmae_mean'] * 100:.1f}% of payload range)")

    best_mae = summary.sort_values("nmae_mean").head(1).iloc[0]
    print(f"\nBEST CONFIG (by NMAE): strategy={best_mae['strategy']}, S={int(best_mae['S'])}, "
          f"epsilon={best_mae['epsilon']}, w={int(best_mae['w'])}")
    print(f"  Mean Normalized MAE: {best_mae['nmae_mean']:.4f} "
          f"({best_mae['nmae_mean'] * 100:.1f}% of payload range)")
    print(f"  Mean KL Divergence: {best_mae['kl_div_mean']:.6f}")
    print(f"{'=' * 90}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_energy_experiment(
    s_values, epsilon_values, w_values, strategies,
    max_timestamps, output_dir,
):
    """Run the full energy dataset experiment."""
    logger.info("=" * 60)
    logger.info("ENERGY DATASET: Smart Building Pub/Sub Scenario")
    logger.info("=" * 60)

    df_raw = load_energy_dataset(max_timestamps=max_timestamps)

    # Build streams for multiple sensor types
    energy_streams = {}
    for sensor_type in ["power_kw", "voltage", "current"]:
        try:
            agg, pub, bound = build_energy_streams(df_raw.copy(), sensor_type, window_minutes=5)
            if len(agg) > 10 and bound > 0.01:
                energy_streams[sensor_type] = (agg, pub, bound)
        except Exception as e:
            logger.warning(f"  Skipping {sensor_type}: {e}")

    if not energy_streams:
        logger.error("No valid energy streams could be built")
        return None

    df_results = sweep_real_data(
        "energy", energy_streams,
        s_values, epsilon_values, w_values, strategies,
    )

    out = os.path.join(output_dir, "energy")
    os.makedirs(out, exist_ok=True)
    df_results.to_csv(os.path.join(out, "sweep_results.csv"), index=False)

    print_summary_table(df_results, "Energy (MCEC-Thai)")
    plot_real_data_results(df_results, "energy", out, energy_streams)
    return df_results


def run_traffic_experiment(
    s_values, epsilon_values, w_values, strategies,
    max_rows_per_file, output_dir,
):
    """Run the full traffic dataset experiment."""
    logger.info("=" * 60)
    logger.info("TRAFFIC DATASET: Smart City Intersection Pub/Sub Scenario")
    logger.info("=" * 60)

    sensors = load_traffic_dataset(max_rows_per_file=max_rows_per_file)

    # Build streams for multiple metrics
    traffic_streams = {}
    for metric in ["speed", "object_count"]:
        try:
            agg, pub, bound = build_traffic_streams(sensors, metric, window_seconds=10)
            if len(agg) > 10:
                traffic_streams[metric] = (agg, pub, bound)
        except Exception as e:
            logger.warning(f"  Skipping {metric}: {e}")

    if not traffic_streams:
        logger.error("No valid traffic streams could be built")
        return None

    df_results = sweep_real_data(
        "traffic", traffic_streams,
        s_values, epsilon_values, w_values, strategies,
    )

    out = os.path.join(output_dir, "traffic")
    os.makedirs(out, exist_ok=True)
    df_results.to_csv(os.path.join(out, "sweep_results.csv"), index=False)

    print_summary_table(df_results, "Traffic (Colorado Springs)")
    plot_real_data_results(df_results, "traffic", out, traffic_streams)
    return df_results


def main():
    parser = argparse.ArgumentParser(
        description="Real-data DP experiment using MCEC-Thai energy + Colorado Springs traffic datasets"
    )
    parser.add_argument(
        "--dataset", choices=["energy", "traffic", "both"], default="both",
        help="Which dataset to run (default: both)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Reduced parameter sweep for faster testing",
    )
    parser.add_argument("--output-dir", default="results_real_data")
    parser.add_argument(
        "--max-energy-timestamps", type=int, default=None,
        help="Limit energy dataset rows (default: all ~178K)",
    )
    parser.add_argument(
        "--max-traffic-rows", type=int, default=None,
        help="Limit traffic dataset rows per sensor file (default: all)",
    )
    args = parser.parse_args()

    if args.quick:
        s_values = [1, 3, 6]
        epsilon_values = [0.5, 1.0, 2.0]
        w_values = [5, 10]
        strategies = ["uniform", "budget_absorption"]
    else:
        s_values = [1, 2, 4, 6]
        epsilon_values = [0.5, 1.0, 2.0, 4.0]
        w_values = [4, 8, 10, 12]
        strategies = ["uniform", "sample", "budget_absorption"]

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = []

    if args.dataset in ("energy", "both"):
        df_energy = run_energy_experiment(
            s_values, epsilon_values, w_values, strategies,
            max_timestamps=args.max_energy_timestamps,
            output_dir=args.output_dir,
        )
        if df_energy is not None:
            all_results.append(df_energy)

    if args.dataset in ("traffic", "both"):
        df_traffic = run_traffic_experiment(
            s_values, epsilon_values, w_values, strategies,
            max_rows_per_file=args.max_traffic_rows,
            output_dir=args.output_dir,
        )
        if df_traffic is not None:
            all_results.append(df_traffic)

    # Combined cross-dataset comparison
    if len(all_results) == 2:
        df_combined = pd.concat(all_results, ignore_index=True)
        combined_path = os.path.join(args.output_dir, "combined_sweep_results.csv")
        df_combined.to_csv(combined_path, index=False)

        # Cross-dataset comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for ax, metric, ylabel in [
            (axes[0], "normalized_mae", "Normalized MAE (% of range)"),
            (axes[1], "kl_divergence", "KL Divergence"),
        ]:
            for dataset_name in ["energy", "traffic"]:
                ds = df_combined[df_combined["dataset"] == dataset_name]
                grouped = ds.groupby("epsilon")[metric].mean().sort_index()
                ax.plot(grouped.index, grouped.values, marker="o", label=dataset_name)
            ax.set_xlabel("Privacy Budget (epsilon)")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{ylabel} vs epsilon (averaged across sensors, S, w)")
            ax.legend()
            ax.grid(True, alpha=0.3)

        fig.suptitle("Cross-Dataset Comparison: Energy vs Traffic", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "cross_dataset_comparison.png"), dpi=150)
        plt.close()
        logger.info(f"Saved cross-dataset comparison to {args.output_dir}/")

    logger.info("All experiments complete.")


if __name__ == "__main__":
    main()
