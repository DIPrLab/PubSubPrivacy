#!/usr/bin/env python3
"""
Experimental evaluation script for S-sensitive w-event differential privacy
on a realistic factory IoT pub/sub scenario.

This script:
  1. Launches a Mosquitto MQTT broker (or connects to an existing one)
  2. Starts the Privacy Plugin with configurable (epsilon, w, S, strategy)
  3. Runs the Factory IoT simulator publishing realistic sensor data
  4. Collects true vs. noisy aggregate values
  5. Computes utility metrics (MAE, RMSE, relative error)
  6. Sweeps over parameter combinations and plots results

Usage:
  python run_experiment.py                  # full sweep (requires MQTT broker)
  python run_experiment.py --offline        # offline simulation (no broker needed)
  python run_experiment.py --quick          # quick demo with fewer params
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import os
import sys
import time
from collections import defaultdict

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
from factory_iot import (
    MachineState,
    SensorSpec,
    OPERATING_PROFILES,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SENSOR_SPECS = [
    SensorSpec("temperature", "celsius", 15.0, 120.0, 2.0, 0.5),
    SensorSpec("vibration", "mm/s", 0.0, 50.0, 2.0, 0.3),
    SensorSpec("power_draw", "kW", 0.0, 500.0, 2.0, 2.0),
    SensorSpec("humidity", "percent", 10.0, 95.0, 2.0, 0.8),
]


def generate_synthetic_stream(
    sensor: SensorSpec,
    num_publishers: int,
    num_timestamps: int,
    seed: int = 42,
) -> tuple[list[float], list[int]]:
    rng = np.random.default_rng(seed)
    val_range = sensor.max_val - sensor.min_val

    # Each publisher has its own state and phase
    states = [MachineState.RUNNING] * num_publishers
    phases = rng.uniform(0, 2 * math.pi, num_publishers)

    aggregates = []
    pub_counts = []

    for tau in range(num_timestamps):
        elapsed = tau * sensor.publish_interval

        active_values = []
        for i in range(num_publishers):
            r = rng.random()
            if states[i] == MachineState.RUNNING:
                if r < 0.02:
                    states[i] = MachineState.IDLE
                elif r < 0.025:
                    states[i] = MachineState.MAINTENANCE
            elif states[i] == MachineState.IDLE:
                if r < 0.15:
                    states[i] = MachineState.RUNNING
            elif states[i] == MachineState.MAINTENANCE:
                if r < 0.05:
                    states[i] = MachineState.RUNNING

            if states[i] != MachineState.RUNNING:
                continue

            profile = OPERATING_PROFILES.get(sensor.sensor_type, {}).get(
                states[i], {"base_frac": 0.5, "variation": 0.1}
            )
            base = sensor.min_val + profile["base_frac"] * val_range
            drift = profile["variation"] * val_range * math.sin(
                2 * math.pi * elapsed / 120.0 + phases[i]
            )
            warmup = min(1.0, elapsed / 30.0)
            noise = rng.normal(0, sensor.noise_std)
            value = np.clip(base * warmup + drift + noise, sensor.min_val, sensor.max_val)
            active_values.append(float(value))

        n_active = len(active_values)
        if n_active > 0:
            aggregates.append(float(np.mean(active_values)))
        else:
            aggregates.append(0.0)
        pub_counts.append(n_active)

    return aggregates, pub_counts


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
    """
    Run the DP mechanism on a pre-generated aggregate stream.

    Returns dict with utility metrics and per-timestamp details.
    """
    np.random.seed(seed)

    config = PrivacyConfig(
        epsilon=epsilon,
        window_size=window_size,
        min_publishers=min_publishers,
        payload_bound=payload_bound,
        strategy=BudgetStrategy(strategy),
    )
    stream = StreamState(config=config)

    released_values = []
    for agg, n_pub in zip(aggregates, pub_counts):
        released = stream.release(agg, n_pub)
        released_values.append(released)

    metrics = compute_utility_metrics(stream.true_values, stream.noisy_values)
    metrics["normalized_mae"] = metrics["mae"] / payload_bound if payload_bound > 0 else float("nan")

    # For KL divergence, only use timestamps where a fresh value was actually
    # released (budget > 0), filtering out repeated-last-value entries that
    # create artificial point masses in the distribution under Sample/BA.
    true_for_kl = []
    noisy_for_kl = []
    for t, n, b in zip(stream.true_values, stream.noisy_values, stream.budgets_spent):
        if b > 0 and t is not None and n is not None:
            true_for_kl.append(t)
            noisy_for_kl.append(n)

    # Fall back to full streams if filtering leaves too few points
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


def run_live_experiment(
    broker_host: str,
    broker_port: int,
    duration: float,
    epsilon: float,
    window_size: int,
    min_publishers: int,
    strategy: str,
) -> dict:
    """Run a live experiment through an actual MQTT broker."""
    from plugin import PrivacyPlugin
    from factory_iot import FactorySimulator

    sensor_bounds = {s.sensor_type: (s.min_val, s.max_val) for s in SENSOR_SPECS}

    plugin = PrivacyPlugin(
        broker_host=broker_host,
        broker_port=broker_port,
        epsilon=epsilon,
        window_size=window_size,
        min_publishers=min_publishers,
        strategy=strategy,
        timestamp_interval=2.0,
        sensor_bounds=sensor_bounds,
    )
    plugin.start()

    factory = FactorySimulator(
        broker_host=broker_host,
        broker_port=broker_port,
        num_lines=2,
        machines_per_line=4,
        sensor_specs=SENSOR_SPECS,
        publish_interval=2.0,
    )
    sim_thread = factory.start(duration=duration, blocking=False)

    sim_thread.join()
    time.sleep(3)
    plugin.stop()

    results = {}
    for topic, stream_state in plugin.get_stream_states().items():
        metrics = compute_utility_metrics(
            stream_state.true_values, stream_state.noisy_values
        )
        results[topic] = {
            "metrics": metrics,
            "true_values": stream_state.true_values,
            "noisy_values": stream_state.noisy_values,
        }
    return results

def sweep_offline(
    num_publishers: int = 8,
    num_timestamps: int = 100,
    s_values: list[int] | None = None,
    epsilon_values: list[float] | None = None,
    w_values: list[int] | None = None,
    strategies: list[str] | None = None,
) -> pd.DataFrame:
    if s_values is None:
        s_values = [1, 2, 3, 5, 8]
    if epsilon_values is None:
        epsilon_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    if w_values is None:
        w_values = [5, 10, 20]
    if strategies is None:
        strategies = ["uniform", "sample", "budget_absorption"]

    rows = []
    total = len(SENSOR_SPECS) * len(s_values) * len(epsilon_values) * len(w_values) * len(strategies)
    done = 0

    for sensor in SENSOR_SPECS:
        payload_bound = sensor.max_val - sensor.min_val
        aggregates, pub_counts = generate_synthetic_stream(
            sensor, num_publishers, num_timestamps, seed=42
        )

        for s_val, eps, w, strat in itertools.product(s_values, epsilon_values, w_values, strategies):
            result = run_dp_on_stream(
                aggregates, pub_counts,
                epsilon=eps,
                window_size=w,
                min_publishers=s_val,
                payload_bound=payload_bound,
                strategy=strat,
                seed=done,
            )
            rows.append({
                "sensor": sensor.sensor_type,
                "S": s_val,
                "epsilon": eps,
                "w": w,
                "strategy": strat,
                "mae": result["metrics"]["mae"],
                "rmse": result["metrics"]["rmse"],
                "relative_error": result["metrics"]["relative_error"],
                "noise_scale_theoretical": payload_bound * w / (s_val * eps),
                "normalized_mae": result["metrics"]["normalized_mae"],
                "kl_divergence": result["metrics"]["kl_divergence"],
                "kl_global_utility": result["metrics"]["kl_global_utility"],
            })
            done += 1
            if done % 50 == 0:
                logger.info(f"  Sweep progress: {done}/{total}")

    return pd.DataFrame(rows)


def plot_results(df: pd.DataFrame, output_dir: str = "results"):
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    subset = df[(df["sensor"] == "temperature") & (df["w"] == 10)]

    for ax, strat in zip(axes, ["uniform", "sample", "budget_absorption"]):
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

    fig.suptitle("Temperature Sensor: MAE vs epsilon (w=10)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mae_vs_epsilon.png"), dpi=150)
    plt.close()
    logger.info(f"Saved: {output_dir}/mae_vs_epsilon.png")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    subset = df[(df["epsilon"] == 1.0) & (df["w"] == 10) & (df["strategy"] == "uniform")]

    for ax, sensor in zip(axes.flat, ["temperature", "vibration", "power_draw", "humidity"]):
        sensor_data = subset[subset["sensor"] == sensor].sort_values("S")
        ax.bar(sensor_data["S"].astype(str), sensor_data["mae"], color="steelblue", alpha=0.8)
        ax.set_xlabel("Sensitivity Parameter (S)")
        ax.set_ylabel("MAE")
        ax.set_title(f"{sensor.replace('_', ' ').title()}")
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("MAE vs S by Sensor Type (epsilon=1.0, w=10, uniform)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mae_vs_S.png"), dpi=150)
    plt.close()
    logger.info(f"Saved: {output_dir}/mae_vs_S.png")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    subset = df[(df["epsilon"] == 1.0) & (df["w"] == 10) & (df["S"] == 3)]

    for ax, sensor in zip(axes.flat, ["temperature", "vibration", "power_draw", "humidity"]):
        sensor_data = subset[subset["sensor"] == sensor]
        strategies = sensor_data["strategy"].unique()
        mae_vals = [sensor_data[sensor_data["strategy"] == s]["mae"].values[0] for s in strategies]
        bars = ax.bar(strategies, mae_vals, color=["#4c72b0", "#dd8452", "#55a868"], alpha=0.8)
        ax.set_ylabel("MAE")
        ax.set_title(f"{sensor.replace('_', ' ').title()}")
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Strategy Comparison (epsilon=1.0, w=10, S=3)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "strategy_comparison.png"), dpi=150)
    plt.close()
    logger.info(f"Saved: {output_dir}/strategy_comparison.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    subset = df[(df["epsilon"] == 1.0) & (df["w"] == 10) & (df["strategy"] == "uniform")]
    for sensor in ["temperature", "vibration", "power_draw", "humidity"]:
        sensor_data = subset[subset["sensor"] == sensor].sort_values("S")
        ax.plot(
            sensor_data["S"],
            sensor_data["noise_scale_theoretical"],
            marker="s",
            label=sensor.replace("_", " ").title(),
        )
    ax.set_xlabel("Sensitivity Parameter (S)")
    ax.set_ylabel("Theoretical Noise Scale (lambda = Bw / Se)")
    ax.set_title("Theoretical Noise Scale vs S (epsilon=1.0, w=10)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "noise_scale_vs_S.png"), dpi=150)
    plt.close()
    logger.info(f"Saved: {output_dir}/noise_scale_vs_S.png")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sensor = SENSOR_SPECS[0]  # temperature
    payload_bound = sensor.max_val - sensor.min_val
    aggregates, pub_counts = generate_synthetic_stream(sensor, 8, 100, seed=42)

    configs = [
        ("S=1, eps=1.0", 1, 1.0),
        ("S=3, eps=1.0", 3, 1.0),
        ("S=3, eps=0.1", 3, 0.1),
        ("S=8, eps=1.0", 8, 1.0),
    ]

    for ax, (label, s_val, eps) in zip(axes.flat, configs):
        result = run_dp_on_stream(
            aggregates, pub_counts,
            epsilon=eps, window_size=10, min_publishers=s_val,
            payload_bound=payload_bound, strategy="uniform", seed=99,
        )
        timestamps = range(len(result["true_values"]))
        ax.plot(timestamps, result["true_values"], "b-", alpha=0.7, label="True", linewidth=1.5)
        noisy_x = [t for t, v in zip(timestamps, result["noisy_values"]) if v is not None]
        noisy_y = [v for v in result["noisy_values"] if v is not None]
        ax.plot(noisy_x, noisy_y, "r-", alpha=0.5, label="Noisy (DP)", linewidth=1)
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Temperature (C)")
        ax.set_title(f"Temperature: {label}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("True vs. DP-Protected Temperature Stream", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "timeseries_comparison.png"), dpi=150)
    plt.close()
    logger.info(f"Saved: {output_dir}/timeseries_comparison.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    sensor = SENSOR_SPECS[0]
    payload_bound = sensor.max_val - sensor.min_val
    aggregates, pub_counts = generate_synthetic_stream(sensor, 8, 100, seed=42)

    for strat in ["uniform", "sample", "budget_absorption"]:
        result = run_dp_on_stream(
            aggregates, pub_counts,
            epsilon=1.0, window_size=10, min_publishers=3,
            payload_bound=payload_bound, strategy=strat, seed=42,
        )
        budgets = result["budgets_spent"]
        w = 10
        window_sums = []
        for i in range(len(budgets)):
            start = max(0, i - w + 1)
            window_sums.append(sum(budgets[start:i + 1]))
        ax.plot(range(len(window_sums)), window_sums, label=strat, alpha=0.8)

    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="Budget limit (epsilon=1.0)")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Cumulative Budget in Window")
    ax.set_title("Sliding Window Budget Utilization by Strategy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "budget_utilization.png"), dpi=150)
    plt.close()
    logger.info(f"Saved: {output_dir}/budget_utilization.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    subset = df[(df["sensor"] == "temperature") & (df["w"] == 10)]

    for ax, strat in zip(axes, ["uniform", "sample", "budget_absorption"]):
        strat_data = subset[subset["strategy"] == strat]
        for s_val in sorted(strat_data["S"].unique()):
            s_data = strat_data[strat_data["S"] == s_val].sort_values("epsilon")
            ax.plot(s_data["epsilon"], s_data["kl_divergence"], marker="o", label=f"S={s_val}")
        ax.set_xlabel("Privacy Budget (epsilon)")
        ax.set_ylabel("KL Divergence D_KL(P || Q)")
        ax.set_title(f"Strategy: {strat}")
        ax.legend()
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Temperature: KL Divergence vs epsilon (w=10)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "kl_vs_epsilon.png"), dpi=150)
    plt.close()
    logger.info(f"Saved: {output_dir}/kl_vs_epsilon.png")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    subset = df[(df["epsilon"] == 1.0) & (df["w"] == 10) & (df["strategy"] == "uniform")]

    for ax, sensor_name in zip(axes.flat, ["temperature", "vibration", "power_draw", "humidity"]):
        sensor_data = subset[subset["sensor"] == sensor_name].sort_values("S")
        ax.bar(sensor_data["S"].astype(str), sensor_data["kl_divergence"],
               color="darkorange", alpha=0.8)
        ax.set_xlabel("Sensitivity Parameter (S)")
        ax.set_ylabel("KL Divergence")
        ax.set_title(f"{sensor_name.replace('_', ' ').title()}")
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("KL Divergence vs S by Sensor Type (epsilon=1.0, w=10, uniform)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "kl_vs_S.png"), dpi=150)
    plt.close()
    logger.info(f"Saved: {output_dir}/kl_vs_S.png")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    subset = df[(df["epsilon"] == 1.0) & (df["w"] == 10) & (df["S"] == 3)]

    for ax, sensor_name in zip(axes.flat, ["temperature", "vibration", "power_draw", "humidity"]):
        sensor_data = subset[subset["sensor"] == sensor_name]
        strategies = sensor_data["strategy"].unique()
        kl_vals = [sensor_data[sensor_data["strategy"] == s]["kl_divergence"].values[0]
                   for s in strategies]
        ax.bar(strategies, kl_vals, color=["#4c72b0", "#dd8452", "#55a868"], alpha=0.8)
        ax.set_ylabel("KL Divergence")
        ax.set_title(f"{sensor_name.replace('_', ' ').title()}")
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("KL Divergence: Strategy Comparison (epsilon=1.0, w=10, S=3)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "kl_strategy_comparison.png"), dpi=150)
    plt.close()
    logger.info(f"Saved: {output_dir}/kl_strategy_comparison.png")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sensor = SENSOR_SPECS[0]  # temperature
    payload_bound = sensor.max_val - sensor.min_val
    aggregates, pub_counts = generate_synthetic_stream(sensor, 8, 100, seed=42)

    configs = [
        ("S=1, eps=1.0", 1, 1.0),
        ("S=3, eps=1.0", 3, 1.0),
        ("S=3, eps=0.1", 3, 0.1),
        ("S=8, eps=1.0", 8, 1.0),
    ]

    for ax, (label, s_val, eps) in zip(axes.flat, configs):
        result = run_dp_on_stream(
            aggregates, pub_counts,
            epsilon=eps, window_size=10, min_publishers=s_val,
            payload_bound=payload_bound, strategy="uniform", seed=99,
        )
        kl_w = result["kl_windowed"]
        ax.plot(range(len(kl_w)), kl_w, "m-", alpha=0.7, linewidth=1.2)
        ax.axhline(y=np.mean([k for k in kl_w if np.isfinite(k)]),
                    color="red", linestyle="--", alpha=0.5, label=f"U_global={np.mean([k for k in kl_w if np.isfinite(k)]):.4f}")
        ax.set_xlabel("Window Position (tau)")
        ax.set_ylabel("U_tau = D_KL(P_tau || Q_tau)")
        ax.set_title(f"Temperature: {label}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Per-Window KL Divergence Over Time", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "kl_windowed_timeseries.png"), dpi=150)
    plt.close()
    logger.info(f"Saved: {output_dir}/kl_windowed_timeseries.png")

    fig, ax = plt.subplots(figsize=(10, 6))
    subset = df[(df["epsilon"] == 1.0) & (df["strategy"] == "uniform") & (df["sensor"] == "temperature")]

    for s_val in sorted(subset["S"].unique()):
        s_data = subset[subset["S"] == s_val].sort_values("w")
        ax.plot(s_data["w"], s_data["kl_global_utility"], marker="o", label=f"S={s_val}")

    ax.set_xlabel("Window Size (w)")
    ax.set_ylabel("Global Utility Loss U_global")
    ax.set_title("Global Distributional Utility Loss vs Window Size (temperature, epsilon=1.0)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "kl_global_vs_w.png"), dpi=150)
    plt.close()
    logger.info(f"Saved: {output_dir}/kl_global_vs_w.png")

    fig, ax = plt.subplots(figsize=(10, 7))
    subset = df[(df["w"] == 10) & (df["strategy"] == "uniform")]
    colors = {"temperature": "#e41a1c", "vibration": "#377eb8",
              "power_draw": "#4daf4a", "humidity": "#984ea3"}

    for sensor_name, color in colors.items():
        s_data = subset[subset["sensor"] == sensor_name]
        scatter = ax.scatter(
            s_data["mae"], s_data["kl_divergence"],
            c=color, alpha=0.6, s=s_data["S"] * 15,
            label=sensor_name.replace("_", " ").title(),
            edgecolors="white", linewidth=0.5,
        )

    ax.set_xlabel("Mean Absolute Error (MAE)")
    ax.set_ylabel("KL Divergence D_KL(P || Q)")
    ax.set_title("Privacy-Utility Tradeoff: KL Divergence vs MAE\n(w=10, uniform; dot size = S)")
    ax.legend()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "kl_vs_mae_tradeoff.png"), dpi=150)
    plt.close()
    logger.info(f"Saved: {output_dir}/kl_vs_mae_tradeoff.png")


def plot_kl_comparison_chart(df: pd.DataFrame, output_dir: str = "results"):
    os.makedirs(output_dir, exist_ok=True)

    w_values = sorted(df["w"].unique())
    eps_values = sorted(df["epsilon"].unique())
    s_values = sorted(df["S"].unique())
    strategies = sorted(df["strategy"].unique())
    sensors = sorted(df["sensor"].unique())

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

    fig, ax = plt.subplots(figsize=(max(14, len(col_labels) * 1.6), max(8, len(row_labels) * 0.5)))
    im = ax.imshow(grid, aspect="auto", cmap="YlOrRd")
    cbar = fig.colorbar(im, ax=ax, label="Mean KL Divergence D_KL(P || Q)")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels([f"S={s}\n{st}" for s, st in col_labels], fontsize=7, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels([f"w={w}, eps={e}" for w, e in row_labels], fontsize=8)

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = grid[i, j]
            if np.isfinite(val):
                color = "white" if val > np.nanmedian(grid) else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color=color, fontweight="bold")

    ax.set_title("KL Divergence Across All Scenarios (averaged over sensors)", fontsize=13, pad=12)
    ax.set_xlabel("Sensitivity (S) x Strategy")
    ax.set_ylabel("Window Size (w) x Privacy Budget (epsilon)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "kl_comparison_heatmap.png"), dpi=200)
    plt.close()
    logger.info(f"Saved: {output_dir}/kl_comparison_heatmap.png")

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    for ax, sensor_name in zip(axes.flat, sensors):
        s_agg = df[df["sensor"] == sensor_name].groupby(
            ["w", "epsilon", "S", "strategy"]
        ).agg(kl_mean=("kl_divergence", "mean")).reset_index()

        grid_s = np.full((len(row_labels), len(col_labels)), np.nan)
        for i, (w, e) in enumerate(row_labels):
            for j, (s, st) in enumerate(col_labels):
                match = s_agg[(s_agg["w"] == w) & (s_agg["epsilon"] == e) &
                              (s_agg["S"] == s) & (s_agg["strategy"] == st)]
                if len(match) == 1:
                    grid_s[i, j] = match["kl_mean"].values[0]

        im = ax.imshow(grid_s, aspect="auto", cmap="YlOrRd")
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels([f"S={s}\n{st}" for s, st in col_labels], fontsize=6, rotation=45, ha="right")
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels([f"w={w}, e={e}" for w, e in row_labels], fontsize=6)
        ax.set_title(sensor_name.replace("_", " ").title(), fontsize=11)

        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                val = grid_s[i, j]
                if np.isfinite(val):
                    color = "white" if val > np.nanmedian(grid_s) else "black"
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                            fontsize=5, color=color)

    fig.suptitle("KL Divergence by Sensor Type Across All Scenarios", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "kl_comparison_per_sensor.png"), dpi=200)
    plt.close()
    logger.info(f"Saved: {output_dir}/kl_comparison_per_sensor.png")

    uniform = df[df["strategy"] == "uniform"].groupby(
        ["w", "epsilon", "S"]
    ).agg(kl_mean=("kl_divergence", "mean")).reset_index()

    fig, axes = plt.subplots(1, len(w_values), figsize=(5 * len(w_values), 6), sharey=True)
    if len(w_values) == 1:
        axes = [axes]

    bar_colors = {1: "#e41a1c", 2: "#377eb8", 4: "#4daf4a"}
    bar_width = 0.25

    for ax, w in zip(axes, w_values):
        x_labels = [f"eps={e}" for e in eps_values]
        x = np.arange(len(eps_values))

        for idx, s_val in enumerate(s_values):
            vals = []
            for e in eps_values:
                match = uniform[(uniform["w"] == w) & (uniform["epsilon"] == e) &
                                (uniform["S"] == s_val)]
                vals.append(match["kl_mean"].values[0] if len(match) == 1 else 0)
            offset = (idx - len(s_values) / 2 + 0.5) * bar_width
            ax.bar(x + offset, vals, bar_width, label=f"S={s_val}",
                   color=bar_colors.get(s_val, "#999999"), alpha=0.85)

        ax.set_xlabel("Privacy Budget")
        ax.set_ylabel("Mean KL Divergence")
        ax.set_title(f"w = {w}")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("KL Divergence: epsilon vs S for Each Window Size (uniform strategy)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "kl_grouped_bars.png"), dpi=150)
    plt.close()
    logger.info(f"Saved: {output_dir}/kl_grouped_bars.png")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    strategy_styles = {"uniform": "-o", "sample": "--s", "budget_absorption": "-.^"}

    for ax, w in zip(axes.flat, w_values):
        for s_val in s_values:
            for strat, style in strategy_styles.items():
                sub = agg[(agg["w"] == w) & (agg["S"] == s_val) &
                          (agg["strategy"] == strat)].sort_values("epsilon")
                if len(sub) > 0:
                    ax.plot(sub["epsilon"], sub["kl_mean"], style, alpha=0.7,
                            label=f"S={s_val}, {strat}", markersize=5)
        ax.set_xlabel("epsilon")
        ax.set_ylabel("KL Divergence")
        ax.set_title(f"w = {w}")
        ax.grid(True, alpha=0.3)

    # Single legend for all subplots
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize=7, bbox_to_anchor=(1.18, 0.5))
    fig.suptitle("KL Divergence vs epsilon by (S, Strategy) for Each w", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "kl_lines_by_w.png"), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {output_dir}/kl_lines_by_w.png")

    pivot = df.groupby(["w", "epsilon", "S", "strategy", "sensor"]).agg(
        kl=("kl_divergence", "mean"),
        kl_global=("kl_global_utility", "mean"),
        mae=("mae", "mean"),
    ).reset_index()
    pivot_path = os.path.join(output_dir, "kl_all_scenarios.csv")
    pivot.to_csv(pivot_path, index=False)
    logger.info(f"Saved: {output_dir}/kl_all_scenarios.csv")

    print("\n" + "=" * 100)
    print("KL DIVERGENCE COMPARISON ACROSS ALL SCENARIOS (mean over sensors)")
    print("=" * 100)
    summary = df.groupby(["w", "epsilon", "S", "strategy"]).agg(
        kl_mean=("kl_divergence", "mean"),
        kl_std=("kl_divergence", "std"),
        nmae_mean=("normalized_mae", "mean"),
        mae_mean=("mae", "mean"),
    ).reset_index().sort_values(["w", "epsilon", "S", "strategy"])

    print(f"{'w':>3} {'eps':>5} {'S':>3} {'strategy':>20} {'KL mean':>10} {'KL std':>10} {'NMAE':>10} {'MAE':>10}")
    print("-" * 110)
    for _, row in summary.iterrows():
        print(f"{int(row['w']):>3} {row['epsilon']:>5.1f} {int(row['S']):>3} "
              f"{row['strategy']:>20} {row['kl_mean']:>10.4f} {row['kl_std']:>10.4f} "
              f"{row['nmae_mean']:>10.4f} {row['mae_mean']:>10.2f}")
    print("=" * 110)


def print_summary_table(df: pd.DataFrame):
    """Print a summary table of results to console."""
    print("\n" + "=" * 80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 80)

    summary = df.groupby(["strategy", "S", "epsilon", "w"]).agg(
        mae_mean=("mae", "mean"),
        nmae_mean=("normalized_mae", "mean"),
        rmse_mean=("rmse", "mean"),
        rel_err_mean=("relative_error", "mean"),
        kl_div_mean=("kl_divergence", "mean"),
        kl_global_mean=("kl_global_utility", "mean"),
    ).reset_index()

    for strat in ["uniform", "sample", "budget_absorption"]:
        strat_data = summary[summary["strategy"] == strat].sort_values("kl_div_mean")
        print(f"\n--- Strategy: {strat} (top 5 configs by KL divergence) ---")
        cols = ["strategy", "S", "epsilon", "w", "kl_div_mean", "kl_global_mean", "nmae_mean"]
        print(strat_data[cols].head(5).to_string(index=False))

    # Overall best by KL divergence
    best_kl = summary.sort_values("kl_div_mean").head(1).iloc[0]
    print(f"\n{'=' * 80}")
    print(f"BEST CONFIG (by KL): strategy={best_kl['strategy']}, S={best_kl['S']}, "
          f"epsilon={best_kl['epsilon']}, w={best_kl['w']}")
    print(f"  Mean KL Divergence: {best_kl['kl_div_mean']:.6f}, "
          f"Mean U_global: {best_kl['kl_global_mean']:.6f}")
    print(f"  Mean Normalized MAE: {best_kl['nmae_mean']:.4f} "
          f"({best_kl['nmae_mean']*100:.1f}% of payload range)")

    # Best by normalized MAE
    best = summary.sort_values("nmae_mean").head(1).iloc[0]
    print(f"\nBEST CONFIG (by NMAE): strategy={best['strategy']}, S={best['S']}, "
          f"epsilon={best['epsilon']}, w={best['w']}")
    print(f"  Mean Normalized MAE: {best['nmae_mean']:.4f} "
          f"({best['nmae_mean']*100:.1f}% of payload range)")
    print(f"  Mean KL Divergence: {best['kl_div_mean']:.6f}")
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(
        description="PubSubPrivacy Experiment: S-sensitive w-event DP for factory IoT"
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="Run offline simulation (no MQTT broker needed)",
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Run live experiment through MQTT broker",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick demo with reduced parameter sweep",
    )
    parser.add_argument("--broker-host", default="localhost")
    parser.add_argument("--broker-port", type=int, default=1883)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--num-publishers", type=int, default=16)
    parser.add_argument("--num-timestamps", type=int, default=200)
    args = parser.parse_args()

    if args.live:
        logger.info("Running LIVE experiment through MQTT broker...")
        results = run_live_experiment(
            args.broker_host, args.broker_port, args.duration,
            epsilon=1.0, window_size=10, min_publishers=3, strategy="uniform",
        )
        for topic, res in results.items():
            print(f"\n{topic}: {res['metrics']}")
        return

    # Default to offline sweep
    logger.info("Running OFFLINE parameter sweep...")

    if args.quick:
        s_values = [1, 3, 5]
        epsilon_values = [0.5, 1.0, 2.0]
        w_values = [10]
        strategies = ["uniform", "budget_absorption"]
    else:
        s_values = [1, 2, 4]
        epsilon_values = [1.0, 2.0, 3.0, 4.0]
        w_values = [4, 8, 10, 12]
        strategies = ["uniform", "sample", "budget_absorption"]

    df = sweep_offline(
        num_publishers=args.num_publishers,
        num_timestamps=args.num_timestamps,
        s_values=s_values,
        epsilon_values=epsilon_values,
        w_values=w_values,
        strategies=strategies,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "sweep_results.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Results saved to {csv_path}")

    print_summary_table(df)

    logger.info("Generating plots...")
    plot_results(df, args.output_dir)

    logger.info("Generating KL divergence comparison chart...")
    plot_kl_comparison_chart(df, args.output_dir)
    logger.info(f"All plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
