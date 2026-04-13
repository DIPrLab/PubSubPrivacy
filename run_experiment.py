#!/usr/bin/env python3
"""
Experimental evaluation of S-sensitive w-event differential privacy
on two real-world IEEE datasets (energy + traffic).

Runs the parameter sweep from Section 6.2 of the paper and generates
all result tables and figures.

Usage:
  python run_experiment.py                # full sweep on both datasets
  python run_experiment.py --quick        # reduced sweep for testing
  python run_experiment.py --dataset energy
  python run_experiment.py --dataset traffic
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
from run_real_data_experiment import (
    load_energy_dataset,
    build_energy_streams,
    load_traffic_dataset,
    build_traffic_streams,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# ── DP runner ───────────────────────────────────────────────────────────────

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
    for agg, n in zip(aggregates, pub_counts):
        stream.release(agg, n)

    metrics = compute_utility_metrics(stream.true_values, stream.noisy_values)
    metrics["normalized_mae"] = (
        metrics["mae"] / payload_bound if payload_bound > 0 else float("nan")
    )

    # KL: only use timestamps where budget was actually spent
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

    return {
        "metrics": metrics,
        "true_values": stream.true_values,
        "noisy_values": stream.noisy_values,
        "budgets_spent": stream.budgets_spent,
        "kl_windowed": compute_windowed_kl_divergence(
            stream.true_values, stream.noisy_values, config.window_size
        ),
    }


# ── Parameter sweep ─────────────────────────────────────────────────────────

def sweep(
    dataset_name: str,
    streams: dict[str, tuple[list[float], list[int], float]],
    s_values: list[int],
    epsilon_values: list[float],
    w_values: list[int],
    strategies: list[str],
) -> pd.DataFrame:
    """Run a parameter sweep over real data streams."""
    rows = []
    combos = list(itertools.product(
        streams.keys(), s_values, epsilon_values, w_values, strategies,
    ))

    for i, (sensor, s, eps, w, strat) in enumerate(combos):
        aggregates, pub_counts, B = streams[sensor]
        result = run_dp_on_stream(
            aggregates, pub_counts,
            epsilon=eps, window_size=w, min_publishers=s,
            payload_bound=B, strategy=strat, seed=i,
        )
        rows.append({
            "dataset": dataset_name,
            "sensor": sensor,
            "S": s,
            "epsilon": eps,
            "w": w,
            "strategy": strat,
            "mae": result["metrics"]["mae"],
            "rmse": result["metrics"]["rmse"],
            "relative_error": result["metrics"]["relative_error"],
            "normalized_mae": result["metrics"]["normalized_mae"],
            "kl_divergence": result["metrics"]["kl_divergence"],
            "kl_global_utility": result["metrics"]["kl_global_utility"],
            "noise_scale_theoretical": B * w / (s * eps),
            "payload_bound": B,
            "num_timestamps": len(aggregates),
            "avg_publishers": float(np.mean(pub_counts)),
        })
        if (i + 1) % 50 == 0:
            logger.info(f"  [{dataset_name}] {i+1}/{len(combos)}")

    logger.info(f"  [{dataset_name}] sweep complete: {len(combos)} configs")
    return pd.DataFrame(rows)


# ── Plotting ────────────────────────────────────────────────────────────────

def plot_results(
    df: pd.DataFrame,
    dataset_name: str,
    output_dir: str,
    streams: dict[str, tuple[list[float], list[int], float]],
):
    """Generate all result figures for a dataset."""
    os.makedirs(output_dir, exist_ok=True)
    sensors = sorted(df["sensor"].unique())
    strategies = sorted(df["strategy"].unique())
    first = sensors[0]
    w_mid = int(sorted(df["w"].unique())[len(df["w"].unique()) // 2])
    eps_mid = float(sorted(df["epsilon"].unique())[len(df["epsilon"].unique()) // 2])
    s_mid = int(sorted(df["S"].unique())[len(df["S"].unique()) // 2])
    ncols = min(3, len(sensors))
    nrows = (len(sensors) + ncols - 1) // ncols

    def _hide_unused(axes_grid, n_used):
        for idx in range(n_used, nrows * ncols):
            axes_grid[idx // ncols][idx % ncols].set_visible(False)

    # 1. MAE vs epsilon
    fig, axes = plt.subplots(1, len(strategies), figsize=(5 * len(strategies), 5), sharey=True)
    if len(strategies) == 1:
        axes = [axes]
    sub = df[(df["sensor"] == first) & (df["w"] == w_mid)]
    for ax, strat in zip(axes, strategies):
        sd = sub[sub["strategy"] == strat]
        for s in sorted(sd["S"].unique()):
            d = sd[sd["S"] == s].sort_values("epsilon")
            ax.plot(d["epsilon"], d["mae"], marker="o", label=f"S={s}")
        ax.set(xlabel="ε", ylabel="MAE", title=strat)
        ax.legend(); ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(True, alpha=0.3)
    fig.suptitle(f"{dataset_name}: MAE vs ε [{first}, w={w_mid}]", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_mae_vs_epsilon.png"), dpi=150)
    plt.close()

    # 2. MAE vs S
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    sub = df[(df["epsilon"] == eps_mid) & (df["w"] == w_mid) & (df["strategy"] == "uniform")]
    for idx, sensor in enumerate(sensors):
        ax = axes[idx // ncols][idx % ncols]
        sd = sub[sub["sensor"] == sensor].sort_values("S")
        ax.bar(sd["S"].astype(str), sd["mae"], color="steelblue", alpha=0.8)
        ax.set(xlabel="S", ylabel="MAE", title=sensor); ax.grid(True, alpha=0.3, axis="y")
    _hide_unused(axes, len(sensors))
    fig.suptitle(f"{dataset_name}: MAE vs S (ε={eps_mid}, w={w_mid}, uniform)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_mae_vs_S.png"), dpi=150)
    plt.close()

    # 3. Strategy comparison
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    sub = df[(df["epsilon"] == eps_mid) & (df["w"] == w_mid) & (df["S"] == s_mid)]
    colors = ["#4c72b0", "#dd8452", "#55a868"]
    for idx, sensor in enumerate(sensors):
        ax = axes[idx // ncols][idx % ncols]
        sd = sub[sub["sensor"] == sensor]
        strats = list(sd["strategy"].unique())
        vals = [sd[sd["strategy"] == s]["mae"].values[0] for s in strats if len(sd[sd["strategy"] == s]) > 0]
        ax.bar(strats[:len(vals)], vals, color=colors[:len(vals)], alpha=0.8)
        ax.set(ylabel="MAE", title=sensor); ax.grid(True, alpha=0.3, axis="y")
    _hide_unused(axes, len(sensors))
    fig.suptitle(f"{dataset_name}: Strategy Comparison (ε={eps_mid}, w={w_mid}, S={s_mid})", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_strategy_comparison.png"), dpi=150)
    plt.close()

    # 4. Time series
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
    for idx, sensor in enumerate(sensors):
        ax = axes[idx // ncols][idx % ncols]
        agg, cnt, B = streams[sensor]
        res = run_dp_on_stream(agg, cnt, epsilon=eps_mid, window_size=w_mid,
                               min_publishers=s_mid, payload_bound=B, strategy="uniform", seed=99)
        n = min(200, len(res["true_values"]))
        ax.plot(range(n), res["true_values"][:n], "b-", alpha=0.7, label="True", lw=1.2)
        ny = [(i, v) for i, v in enumerate(res["noisy_values"][:n]) if v is not None]
        if ny:
            ax.plot([p[0] for p in ny], [p[1] for p in ny], "r-", alpha=0.5, label="Noisy", lw=1)
        ax.set(xlabel="Window", ylabel=sensor, title=f"{sensor} (ε={eps_mid}, S={s_mid})")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    _hide_unused(axes, len(sensors))
    fig.suptitle(f"{dataset_name}: True vs DP-Protected Streams", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_timeseries.png"), dpi=150)
    plt.close()

    # 5. Budget utilization
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
    for idx, sensor in enumerate(sensors):
        ax = axes[idx // ncols][idx % ncols]
        agg, cnt, B = streams[sensor]
        for strat in strategies:
            res = run_dp_on_stream(agg, cnt, epsilon=eps_mid, window_size=w_mid,
                                   min_publishers=s_mid, payload_bound=B, strategy=strat, seed=42)
            budgets = res["budgets_spent"]
            wsums = [sum(budgets[max(0, i - w_mid + 1):i + 1]) for i in range(min(200, len(budgets)))]
            ax.plot(range(len(wsums)), wsums, label=strat, alpha=0.8)
        ax.axhline(y=eps_mid, color="red", ls="--", alpha=0.5, label=f"ε={eps_mid}")
        ax.set(xlabel="Window", ylabel="Budget", title=sensor)
        ax.legend(fontsize=6); ax.grid(True, alpha=0.3)
    _hide_unused(axes, len(sensors))
    fig.suptitle(f"{dataset_name}: Budget Utilization", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_budget_utilization.png"), dpi=150)
    plt.close()

    # 6. KL vs epsilon
    fig, axes = plt.subplots(1, len(strategies), figsize=(5 * len(strategies), 5), sharey=True)
    if len(strategies) == 1:
        axes = [axes]
    sub = df[(df["sensor"] == first) & (df["w"] == w_mid)]
    for ax, strat in zip(axes, strategies):
        sd = sub[sub["strategy"] == strat]
        for s in sorted(sd["S"].unique()):
            d = sd[sd["S"] == s].sort_values("epsilon")
            ax.plot(d["epsilon"], d["kl_divergence"], marker="o", label=f"S={s}")
        ax.set(xlabel="ε", ylabel="KL Divergence", title=strat)
        ax.legend(); ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(True, alpha=0.3)
    fig.suptitle(f"{dataset_name}: KL Divergence vs ε [{first}, w={w_mid}]", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_kl_vs_epsilon.png"), dpi=150)
    plt.close()

    # 7. KL heatmap
    w_vals = sorted(df["w"].unique())
    eps_vals = sorted(df["epsilon"].unique())
    s_vals = sorted(df["S"].unique())
    agg_df = df.groupby(["w", "epsilon", "S", "strategy"]).agg(
        kl_mean=("kl_divergence", "mean")).reset_index()
    row_labels = [(w, e) for w in w_vals for e in eps_vals]
    col_labels = [(s, st) for s in s_vals for st in strategies]
    grid = np.full((len(row_labels), len(col_labels)), np.nan)
    for i, (w, e) in enumerate(row_labels):
        for j, (s, st) in enumerate(col_labels):
            m = agg_df[(agg_df["w"] == w) & (agg_df["epsilon"] == e) &
                       (agg_df["S"] == s) & (agg_df["strategy"] == st)]
            if len(m) == 1:
                grid[i, j] = m["kl_mean"].values[0]

    fig, ax = plt.subplots(figsize=(max(14, len(col_labels) * 1.6), max(8, len(row_labels) * 0.5)))
    im = ax.imshow(grid, aspect="auto", cmap="YlOrRd")
    fig.colorbar(im, ax=ax, label="Mean KL Divergence")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels([f"S={s}\n{st}" for s, st in col_labels], fontsize=7, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels([f"w={w}, ε={e}" for w, e in row_labels], fontsize=8)
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            v = grid[i, j]
            if np.isfinite(v):
                c = "white" if v > np.nanmedian(grid) else "black"
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=6, color=c, fontweight="bold")
    ax.set_title(f"{dataset_name}: KL Divergence Heatmap", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_kl_heatmap.png"), dpi=200)
    plt.close()

    # 8. Per-window KL time series
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
    for idx, sensor in enumerate(sensors):
        ax = axes[idx // ncols][idx % ncols]
        agg, cnt, B = streams[sensor]
        res = run_dp_on_stream(agg, cnt, epsilon=eps_mid, window_size=w_mid,
                               min_publishers=s_mid, payload_bound=B, strategy="uniform", seed=99)
        kl_w = res["kl_windowed"]
        n = min(200, len(kl_w))
        ax.plot(range(n), kl_w[:n], "m-", alpha=0.7, lw=1.2)
        valid = [k for k in kl_w if np.isfinite(k)]
        if valid:
            ax.axhline(y=np.mean(valid), color="red", ls="--", alpha=0.5,
                       label=f"U_global={np.mean(valid):.4f}")
        ax.set(xlabel="Window", ylabel="KL", title=sensor)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    _hide_unused(axes, len(sensors))
    fig.suptitle(f"{dataset_name}: Per-Window KL Divergence", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_kl_windowed.png"), dpi=150)
    plt.close()

    logger.info(f"  All plots saved to {output_dir}/")


# ── Summary ─────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame, name: str):
    print(f"\n{'=' * 90}")
    print(f"RESULTS: {name.upper()}")
    print(f"{'=' * 90}")

    summary = df.groupby(["strategy", "S", "epsilon", "w"]).agg(
        mae=("mae", "mean"), nmae=("normalized_mae", "mean"),
        kl=("kl_divergence", "mean"), kl_g=("kl_global_utility", "mean"),
    ).reset_index()

    for strat in sorted(df["strategy"].unique()):
        sd = summary[summary["strategy"] == strat].sort_values("kl")
        print(f"\n--- {strat} (top 5 by KL) ---")
        print(sd[["strategy", "S", "epsilon", "w", "kl", "kl_g", "nmae"]].head(5).to_string(index=False))

    best = summary.sort_values("kl").head(1).iloc[0]
    print(f"\nBEST (KL): {best['strategy']}, S={int(best['S'])}, eps={best['epsilon']}, w={int(best['w'])}")
    print(f"  KL={best['kl']:.6f}, NMAE={best['nmae']:.4f} ({best['nmae']*100:.1f}%)")
    print(f"{'=' * 90}")


# ── Dataset runners ─────────────────────────────────────────────────────────

def run_energy(s_values, eps_values, w_values, strategies, output_dir, max_ts=None):
    logger.info("=" * 60)
    logger.info("ENERGY DATASET: Smart Building Pub/Sub")
    logger.info("=" * 60)

    df_raw = load_energy_dataset(max_timestamps=max_ts)
    streams = {}
    for sensor in ["power_kw", "voltage", "current"]:
        try:
            agg, pub, B = build_energy_streams(df_raw.copy(), sensor, window_minutes=5)
            if len(agg) > 10 and B > 0.01:
                streams[sensor] = (agg, pub, B)
        except Exception as e:
            logger.warning(f"  Skipping {sensor}: {e}")

    if not streams:
        logger.error("No valid energy streams"); return None

    df = sweep("energy", streams, s_values, eps_values, w_values, strategies)
    out = os.path.join(output_dir, "energy")
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, "sweep_results.csv"), index=False)
    print_summary(df, "Energy (MCEC-Thai)")
    plot_results(df, "energy", out, streams)
    return df


def run_traffic(s_values, eps_values, w_values, strategies, output_dir, max_rows=None):
    logger.info("=" * 60)
    logger.info("TRAFFIC DATASET: Smart City Intersection")
    logger.info("=" * 60)

    sensors = load_traffic_dataset(max_rows_per_file=max_rows)
    streams = {}
    for metric in ["speed", "object_count"]:
        try:
            agg, pub, B = build_traffic_streams(sensors, metric, window_seconds=10)
            if len(agg) > 10:
                streams[metric] = (agg, pub, B)
        except Exception as e:
            logger.warning(f"  Skipping {metric}: {e}")

    if not streams:
        logger.error("No valid traffic streams"); return None

    df = sweep("traffic", streams, s_values, eps_values, w_values, strategies)
    out = os.path.join(output_dir, "traffic")
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, "sweep_results.csv"), index=False)
    print_summary(df, "Traffic (Colorado Springs)")
    plot_results(df, "traffic", out, streams)
    return df


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run S-sensitive w-event DP experiments on real IEEE datasets"
    )
    parser.add_argument("--dataset", choices=["energy", "traffic", "both"], default="both")
    parser.add_argument("--quick", action="store_true", help="Reduced sweep for testing")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--max-energy-timestamps", type=int, default=None)
    parser.add_argument("--max-traffic-rows", type=int, default=None)
    args = parser.parse_args()

    if args.quick:
        s_values = [1, 3, 6]
        eps_values = [0.5, 1.0, 2.0]
        w_values = [5, 10]
        strategies = ["uniform", "budget_absorption"]
    else:
        # Paper Section 6.2 parameters
        s_values = [1, 2, 4]
        eps_values = [1.0, 2.0, 3.0, 4.0]
        w_values = [4, 8, 10, 12]
        strategies = ["uniform", "sample", "budget_absorption"]

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = []

    if args.dataset in ("energy", "both"):
        df = run_energy(s_values, eps_values, w_values, strategies,
                        args.output_dir, args.max_energy_timestamps)
        if df is not None:
            all_results.append(df)

    if args.dataset in ("traffic", "both"):
        df = run_traffic(s_values, eps_values, w_values, strategies,
                         args.output_dir, args.max_traffic_rows)
        if df is not None:
            all_results.append(df)

    # Cross-dataset comparison
    if len(all_results) == 2:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(os.path.join(args.output_dir, "combined_sweep_results.csv"), index=False)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, metric, ylabel in [
            (axes[0], "normalized_mae", "Normalized MAE"),
            (axes[1], "kl_divergence", "KL Divergence"),
        ]:
            for ds in ["energy", "traffic"]:
                g = combined[combined["dataset"] == ds].groupby("epsilon")[metric].mean().sort_index()
                ax.plot(g.index, g.values, marker="o", label=ds)
            ax.set(xlabel="ε", ylabel=ylabel, title=f"{ylabel} vs ε")
            ax.legend(); ax.grid(True, alpha=0.3)
        fig.suptitle("Cross-Dataset: Energy vs Traffic", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "cross_dataset_comparison.png"), dpi=150)
        plt.close()

    logger.info("All experiments complete.")


if __name__ == "__main__":
    main()
