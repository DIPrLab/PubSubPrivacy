#!/usr/bin/env python3
"""
End-to-end experimental pipeline for clamped w-event DP with P-allocation,
run exclusively on real-world public datasets (no synthetic data).

All dataset ingestion, stream construction, and clamp application live in
``data_streams.py``.  This module is just the experimental runner: sweeps,
figures, tuning, and cross-dataset aggregation.

For every dataset registered in ``data_streams.DATASETS`` this script:
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
import asyncio
import itertools
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

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
from data_streams import (
    DATASETS,
    build_topic_manifest,
    prepare_dataset,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

ALL_STRATEGIES = [
    "uniform",
    "sample",
    "budget_distribution",
    "budget_absorption",
    "p_gated_uniform",
    "p_gated_sample",
    "p_gated_ba",
    "n_weighted",
]

# ═════════════════════════════════════════════════════════════════════════
#  Data loaders for every real-world dataset we evaluate on
# ═════════════════════════════════════════════════════════════════════════
# Dataset loading, stream construction, and the clamp options of
# Definition 3.2 all live in ``data_streams.py``.  See the imports above.


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
    epsilon_count: float = 0.0,
    max_publishers: int | None = None,
) -> dict:
    """Run the DP engine on a pre-aggregated stream (offline evaluation).

    ``epsilon_count`` > 0 makes the P-allocation release gate compare a
    differentially private count |P_tau| (Laplace scale 1/eps_count, paper Sec.
    6.3 step 1) against P_min instead of the exact count; ``max_publishers``
    caps the multiplicity that enters the mean sensitivity (P_max, Sec. 6.5).
    """
    np.random.seed(seed)
    config = PrivacyConfig(
        epsilon=float(epsilon),
        window_size=int(window_size),
        min_publishers=int(min_publishers),
        payload_bound=float(payload_bound),
        strategy=BudgetStrategy(strategy),
        epsilon_count=float(epsilon_count),
        max_publishers=int(max_publishers) if max_publishers else None,
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
        stream.pub_counts, stream.deferred_flags
    )
    # Composed DP cost: the w-event budget epsilon plus the eps_count paid on
    # every DP publisher-count release (paper Table 3 / Sec. 6.3 step 1).
    metrics["eps_count_spent"] = stream.eps_count_spent
    metrics["dp_count_releases"] = stream.dp_counts

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
#  Parallel execution helpers
# ═════════════════════════════════════════════════════════════════════════
#
# Each hot experiment (sweep, A, B, C) is a loop of independent calls into
# ``run_dp_on_stream`` or the greedy/brute tuners.  We ship those tasks to
# a ``ProcessPoolExecutor`` so a full run finishes in roughly 1 / n_workers
# of the serial time.  Workers are plain Python processes (spawn on Windows)
# and the task functions below are module-level so they pickle cleanly.
#
# Large read-only payloads (``streams``) are passed through ``initializer``
# and stashed in a worker-process global so each task args tuple only has to
# carry small identifiers (sensor name, combo values).

_WORKER_STREAMS: dict | None = None


def _init_streams_worker(streams):
    """Pool initializer: cache ``streams`` in a worker-local global."""
    global _WORKER_STREAMS
    _WORKER_STREAMS = streams


def _default_workers(requested: int | None) -> int:
    """Clamp the requested worker count to [1, os.cpu_count()]."""
    if requested is None or requested <= 0:
        cpu = os.cpu_count() or 1
        return max(1, cpu - 1)
    return int(requested)


def _run_parallel_tasks(
    tasks: list,
    fn,
    *,
    workers: int,
    initializer=None,
    initargs: tuple = (),
    progress_label: str | None = None,
    progress_every: int = 100,
) -> list:
    """Run ``fn(task)`` for each item in ``tasks`` with optional parallelism.

    Results are returned in the same order as ``tasks``.  When ``workers`` is
    1 the loop runs serially in-process (still calling ``initializer`` so
    task functions that rely on worker globals keep working).  Otherwise a
    ``ProcessPoolExecutor`` is created, tasks are submitted with
    ``as_completed`` for progress logging, and results are re-sorted to the
    original index before returning.
    """
    total = len(tasks)
    if total == 0:
        return []

    if workers <= 1:
        if initializer is not None:
            initializer(*initargs)
        results = []
        for idx, task in enumerate(tasks):
            results.append(fn(task))
            if progress_label and (idx + 1) % progress_every == 0:
                logger.info(f"{progress_label} {idx + 1}/{total}")
        return results

    results: list = [None] * total
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=initializer,
        initargs=initargs,
    ) as pool:
        future_to_idx = {
            pool.submit(fn, task): idx for idx, task in enumerate(tasks)
        }
        done = 0
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            results[idx] = fut.result()
            done += 1
            if progress_label and done % progress_every == 0:
                logger.info(f"{progress_label} {done}/{total}")
    return results


# Generic named-stream pool used by every post-sweep phase (figure1, u-shape,
# extremes, collusion, k_ext, tuning).  Streams keyed by an arbitrary hashable
# tag are stashed in a worker global so each task only ships the tag + params.
_WORKER_NAMED_STREAMS: dict | None = None


def _init_named_streams_worker(named_streams):
    global _WORKER_NAMED_STREAMS
    _WORKER_NAMED_STREAMS = named_streams


def _plot_dp_task(task):
    """DP run for plot_results: returns the three arrays the plots consume
    (noisy_arr, budgets_spent, kl_windowed), all as float32 to keep IPC cheap.
    task = (idx, key, strategy, P, eps, w, seed).
    """
    idx, key, strategy, P, eps, w, seed = task
    agg, cnt, B = _WORKER_NAMED_STREAMS[key]
    res = run_dp_on_stream(
        agg, cnt,
        epsilon=float(eps), window_size=int(w), min_publishers=int(P),
        payload_bound=float(B), strategy=strategy, seed=int(seed),
    )
    nv = res["noisy_values"]
    return {
        "idx": idx,
        "key": key,
        "strategy": strategy,
        "seed": int(seed),
        "noisy_arr": np.array(
            [v if v is not None else np.nan for v in nv], dtype=np.float32,
        ),
        "budgets_spent": np.asarray(res["budgets_spent"], dtype=np.float32),
        "kl_windowed": np.asarray(res["kl_windowed"], dtype=np.float32),
    }


def _dp_named_task(task):
    """Run run_dp_on_stream on a worker-cached (agg, cnt, B) tuple.

    task = (idx, key, strategy, P, epsilon, w, seed, return_mode).
    return_mode ∈ {"metrics", "self_kl", "noisy"}:
      metrics -- return only the built-in metrics KL (filter: budget_spent>0).
      self_kl -- also return _kl_of-style KL (filter: non-None only); needed for
                 u_shaped_curve / kl_extremes_vs_ours numerical parity.
      noisy   -- additionally return a compact float32 noisy_values array so the
                 caller can compute KL against a different reference stream
                 (used by the global/extreme regimes and collusion).
    """
    idx, key, strategy, P, eps, w, seed, return_mode = task
    agg, cnt, B = _WORKER_NAMED_STREAMS[key]
    res = run_dp_on_stream(
        agg, cnt,
        epsilon=float(eps), window_size=int(w), min_publishers=int(P),
        payload_bound=float(B), strategy=strategy, seed=int(seed),
    )
    m = res["metrics"]
    out = {
        "idx": idx,
        "key": key,
        "kl": m["kl_divergence"],
        "mae": m["mae"],
        "nmae": m["normalized_mae"],
        "release_rate": m["release_rate"],
        "deferrals": m["deferrals"],
        "attribution_advantage": m["attribution_advantage"],
    }
    if return_mode in ("self_kl", "noisy"):
        out["kl_self"] = _kl_of(res["true_values"], res["noisy_values"])
    if return_mode == "noisy":
        out["noisy_arr"] = np.array(
            [v if v is not None else np.nan for v in res["noisy_values"]],
            dtype=np.float32,
        )
    return out


# ═════════════════════════════════════════════════════════════════════════
#  Parameter sweep
# ═════════════════════════════════════════════════════════════════════════

def _sweep_task(task):
    """One sweep combo: runs the DP engine and builds the result row.

    When ``log_messages`` is set on the task tuple, the worker also returns
    per-release message records (true aggregate, noisy value, n_tau,
    eps_tau, lambda_tau, deferred, ...) for that configuration.  The main
    process consolidates these into a single ``sweep_messages.csv`` per
    dataset -- one row per logical timestamp per config.
    """
    (i, dataset_name, sensor, P, eps, w, strat,
     clamp_mode, log_messages) = task
    aggregates, pub_counts, B = _WORKER_STREAMS[sensor]
    result = run_dp_on_stream(
        aggregates, pub_counts,
        epsilon=eps, window_size=w, min_publishers=P,
        payload_bound=B, strategy=strat, seed=i,
    )
    m = result["metrics"]
    avg_n = float(np.mean([n for n in pub_counts if n > 0])) if any(pub_counts) else 0.0
    out = {
        "dataset": dataset_name,
        "clamp_mode": clamp_mode,
        "sensor": sensor,
        "P": P,
        "epsilon": eps,
        "w": w,
        "strategy": strat,
        "seed": i,
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
    }
    if log_messages:
        from message_logger import build_message_rows
        out["_messages"] = build_message_rows(
            result,
            dataset=dataset_name, clamp_mode=clamp_mode, sensor=sensor,
            strategy=strat, P=P, epsilon=eps, w=w,
            payload_bound=B, seed=i, experiment="sweep",
        )
    return out


def sweep(
    dataset_name: str,
    streams: dict[str, tuple[list[float], list[int], float]],
    s_values: list[int],
    epsilon_values: list[float],
    w_values: list[int],
    strategies: list[str],
    workers: int = 1,
    clamp_mode: str = "static",
    log_messages: bool = True,
    messages_csv_path: str | None = None,
) -> pd.DataFrame:
    combos = list(itertools.product(
        streams.keys(), s_values, epsilon_values, w_values, strategies,
    ))
    tasks = [
        (i, dataset_name, sensor, P, eps, w, strat, clamp_mode, log_messages)
        for i, (sensor, P, eps, w, strat) in enumerate(combos)
    ]
    rows = _run_parallel_tasks(
        tasks, _sweep_task,
        workers=workers,
        initializer=_init_streams_worker,
        initargs=(streams,),
        progress_label=f"  [{dataset_name}]",
    )

    # Split out the per-release messages before building the summary frame.
    all_messages: list[dict] = []
    summary_rows: list[dict] = []
    for r in rows:
        if isinstance(r, dict) and "_messages" in r:
            all_messages.extend(r.pop("_messages"))
        summary_rows.append(r)

    if log_messages and messages_csv_path and all_messages:
        from message_logger import write_messages_csv
        n = write_messages_csv(all_messages, messages_csv_path, append=False)
        logger.info(f"  [{dataset_name}] sweep messages -> {messages_csv_path} "
                    f"({n} rows)")

    logger.info(f"  [{dataset_name}] sweep complete: {len(tasks)} configs")
    return pd.DataFrame(summary_rows)


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
    workers: int = 1,
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

    # Pre-dispatch every DP run needed by panels 4 / 5 / 8 so the plot loop
    # reads from a cache instead of running ~30 DP passes serially.  Panel 4
    # and panel 8 share (strategy, seed=99); panel 5 uses seed=42 across all
    # strategies.
    panel_4_8_strats = [s for s in ("uniform", "n_weighted") if s in strategies]
    plot_tasks: list = []
    idx = 0
    task_index: dict[tuple, int] = {}  # (sensor, strategy, seed) -> task idx
    for sensor in sensors:
        for strat in panel_4_8_strats:
            key = (sensor, strat, 99)
            if key not in task_index:
                task_index[key] = idx
                plot_tasks.append((idx, sensor, strat, p_mid,
                                   eps_mid, w_mid, 99))
                idx += 1
        for strat in strategies:
            key = (sensor, strat, 42)
            if key not in task_index:
                task_index[key] = idx
                plot_tasks.append((idx, sensor, strat, p_mid,
                                   eps_mid, w_mid, 42))
                idx += 1

    logger.info(f"  [{dataset_name}] plot_results: dispatching "
                f"{len(plot_tasks)} DP runs (panels 4/5/8) across "
                f"{min(workers, len(plot_tasks)) if plot_tasks else 1} worker(s)")
    plot_results_arr = _run_parallel_tasks(
        plot_tasks, _plot_dp_task,
        workers=min(workers, max(len(plot_tasks), 1)),
        initializer=_init_named_streams_worker,
        initargs=(streams,),
        progress_label=f"  [{dataset_name}] plot-dp",
        progress_every=max(5, len(plot_tasks) // 10 or 1),
    )
    plot_cache: dict[tuple, dict] = {
        k: plot_results_arr[i] for k, i in task_index.items()
    }

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
            noisy_arr = plot_cache[(sensor, strat, 99)]["noisy_arr"][:show_len]
            ny = [(i, float(noisy_arr[i])) for i in range(len(noisy_arr))
                  if np.isfinite(noisy_arr[i])]
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
            budgets = plot_cache[(sensor, strat, 42)]["budgets_spent"]
            n = min(200, len(budgets))
            # Trailing-window sum of eps_tau over the last w_mid timestamps.
            wsums = np.zeros(n, dtype=float)
            for i in range(n):
                wsums[i] = float(budgets[max(0, i - w_mid + 1):i + 1].sum())
            ax.plot(range(n), wsums, label=strat, alpha=0.8, lw=1)
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
        for strat, color in zip(["uniform", "n_weighted"], ["tab:red", "tab:green"]):
            if strat not in strategies:
                continue
            kl_w = plot_cache[(sensor, strat, 99)]["kl_windowed"]
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

_WORKER_PER_PUB: dict | None = None


def _init_per_pub_worker(per_pub):
    global _WORKER_PER_PUB
    _WORKER_PER_PUB = per_pub


def _k_ext_task(task):
    """One k_ext merge + DP run.  per_pub is cached in worker global."""
    k_ext, epsilon, w, P, payload_bound, seed = task
    per_pub = _WORKER_PER_PUB
    T = len(next(iter(per_pub.values())))
    merged_agg: list[float] = []
    merged_cnt: list[int] = []
    merged_wait: list[int] = []
    pending_vals: list[float] = []
    pending_pubs: set[str] = set()
    pending_waits = 0

    for tau in range(T):
        for p, series in per_pub.items():
            v = series[tau]
            if v is not None:
                pending_vals.append(v)
                pending_pubs.add(p)
        pending_waits += 1

        if len(pending_pubs) >= P or pending_waits > k_ext:
            merged_agg.append(float(np.mean(pending_vals)) if pending_vals else 0.0)
            merged_cnt.append(len(pending_pubs))
            merged_wait.append(pending_waits)
            pending_vals = []
            pending_pubs = set()
            pending_waits = 0

    if pending_vals:
        merged_agg.append(float(np.mean(pending_vals)))
        merged_cnt.append(len(pending_pubs))
        merged_wait.append(pending_waits)

    res = run_dp_on_stream(
        merged_agg, merged_cnt, epsilon=epsilon, window_size=w,
        min_publishers=P, payload_bound=payload_bound,
        strategy="p_gated_ba", seed=seed,
    )
    m = res["metrics"]
    return {
        "K_ext": k_ext,
        "T_max_over_dt": k_ext + 1,
        "num_releases": len(merged_agg),
        "release_rate": m["release_rate"],
        "mean_wait_dt": float(np.mean(merged_wait)) if merged_wait else float("nan"),
        "max_wait_dt": max(merged_wait) if merged_wait else 0,
        "mae": m["mae"],
        "kl_divergence": m["kl_divergence"],
        "normalized_mae": m["normalized_mae"],
    }


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
    workers: int = 1,
) -> pd.DataFrame:
    """
    Simulate adaptive timestamp extension (Section 5.6) by merging k_ext
    consecutive raw-timestamp slots whenever the leaf-level publisher count
    is below P.  Reports release rate, KL divergence, and per-release
    wall-clock latency in units of the base interval Delta_t.
    """
    tasks = [(int(k_ext), float(epsilon), int(w), int(P), float(payload_bound),
              13 + int(k_ext))
             for k_ext in k_ext_values]
    logger.info(f"  [{dataset_name}/{sensor_name}] K_ext sweep: dispatching "
                f"{len(tasks)} tasks across {min(workers, len(tasks))} worker(s)")
    # K_ext tasks are few (4-5); cap worker count to avoid idle processes.
    rows = _run_parallel_tasks(
        tasks, _k_ext_task,
        workers=min(workers, len(tasks)),
        initializer=_init_per_pub_worker,
        initargs=(per_pub,),
        progress_label=f"  [{dataset_name}/{sensor_name}] k_ext",
        progress_every=1,
    )

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
    workers: int = 1,
) -> pd.DataFrame:
    """
    Empirically verify the sqrt(c) noise reduction: c independent noisy
    streams averaged together have noise std reduced by ~sqrt(c).
    """
    sensor = next(iter(streams))
    agg, cnt, B = streams[sensor]

    # Each (trial, k) pair runs one independent Laplace-noise DP stream; we
    # then fold them into `c`-sized collusion averages for every c >= k+1.
    # The upper bound of unique DP runs is trials_per_c * max(c) because
    # original seeds were 1000*trial + k, shared across c values.
    max_c = max(c_values)
    named = {("collusion",): (agg, cnt, B)}
    tasks: list = []
    idx = 0
    for trial in range(trials_per_c):
        for k in range(max_c):
            seed = 1_000 * trial + k
            tasks.append((idx, ("collusion",), "uniform", P,
                          epsilon, w, seed, "noisy"))
            idx += 1

    logger.info(f"  [{dataset_name}] collusion_experiment: dispatching "
                f"{len(tasks)} DP runs across {workers} worker(s)")
    results = _run_parallel_tasks(
        tasks, _dp_named_task,
        workers=workers,
        initializer=_init_named_streams_worker,
        initargs=(named,),
        progress_label=f"  [{dataset_name}] collusion",
        progress_every=max(50, len(tasks) // 20),
    )

    # Index (trial, k) -> noisy_arr for fast lookup.
    noisy_by_tk: dict[tuple[int, int], np.ndarray] = {}
    t_idx = 0
    for trial in range(trials_per_c):
        for k in range(max_c):
            noisy_by_tk[(trial, k)] = results[t_idx]["noisy_arr"]
            t_idx += 1

    true_arr = np.array(
        [t if t is not None else np.nan for t in agg], dtype=float,
    )
    rows = []
    for c in c_values:
        maes = []
        for trial in range(trials_per_c):
            avg_noisy = np.zeros_like(true_arr, dtype=float)
            for k in range(c):
                avg_noisy += noisy_by_tk[(trial, k)]
            avg_noisy /= c
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


def _evaluate_stream(
    agg, cnt, payload_bound, epsilon, w, P, strategy,
    utility_weight=1.0, latency_weight=0.2, seed=77,
) -> dict:
    """Stream-level evaluator; caller supplies the (agg, cnt) rebuild once."""
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


def _evaluate_P(
    per_pub, payload_bound, epsilon, w, P, strategy,
    utility_weight=1.0, latency_weight=0.2, seed=77,
) -> dict:
    """Single-point evaluator; corresponds to paper's Evaluate(trace, P).

    Compatibility shim: rebuilds the stream from per_pub, then delegates to
    ``_evaluate_stream``.  Prefer the stream variant when the rebuild is
    already cached (``tune_hyperparameters`` does this).
    """
    agg, cnt = _rebuild_stream_with_dt(per_pub, 1)
    return _evaluate_stream(agg, cnt, payload_bound, epsilon, w, P, strategy,
                            utility_weight, latency_weight, seed)


# Worker-side state for hyperparameter tuning: cached (agg, cnt, B, eps, w,
# utility_weight, latency_weight).  Shared by brute-force and per-strategy
# greedy walks so the rebuild runs exactly once.
_WORKER_TUNE_CTX: tuple | None = None


def _init_tune_worker(agg, cnt, payload_bound, epsilon, w,
                      utility_weight, latency_weight):
    global _WORKER_TUNE_CTX
    _WORKER_TUNE_CTX = (agg, cnt, payload_bound, epsilon, w,
                        utility_weight, latency_weight)


def _tune_eval_task(task):
    """Brute-force evaluator: (strategy, P, seed) -> row dict."""
    strategy, P, seed = task
    agg, cnt, B, eps, w, uw, lw = _WORKER_TUNE_CTX
    return _evaluate_stream(agg, cnt, B, eps, w, P, strategy, uw, lw, seed)


def _greedy_walk_task(task):
    """One strategy's Algorithm-2 greedy walk, end-to-end in the worker."""
    (strategy, p_max, alpha, I_max, loss_tie_tol, seed,
     n_restarts, restart_rng_seed) = task
    agg, cnt, B, eps, w, uw, lw = _WORKER_TUNE_CTX
    return _greedy_walk_on_stream(
        agg, cnt, B, eps, w, strategy, p_max,
        alpha=alpha, I_max=I_max,
        utility_weight=uw, latency_weight=lw,
        loss_tie_tol=loss_tie_tol, seed=seed,
        n_restarts=n_restarts, restart_rng_seed=restart_rng_seed,
    )


def _intelligent_restart_seeds(
    cnt: list[int], alpha: float, p_max: int,
    n_restarts: int, rng_seed: int,
) -> list[int]:
    """Pick P values for Algorithm 2 multi-start.

    The first seed is always P_0 = ceil(1/alpha), the paper's
    identity-protection seed (Section 6.7).  The remaining n_restarts-1 seeds
    are sampled *intelligently* rather than uniformly: we draw from the
    empirical distribution of n_tau so the hill-climb is seeded at plausible
    publisher-count levels the stream actually exhibits.  Concretely, we take
    quantiles of the non-zero n_tau distribution (p25, median, p75, ...) so
    the restarts cover the pool-density regimes that matter for the loss.

    All returned values are clamped to [1, p_max] and de-duplicated while
    preserving order.
    """
    P_0 = max(1, int(np.ceil(1.0 / max(alpha, 1e-9))))
    P_0 = max(1, min(P_0, max(1, p_max)))
    seeds: list[int] = [P_0]
    if n_restarts <= 1:
        return seeds

    nonzero = [int(c) for c in cnt if c is not None and c > 0]
    k_extra = max(0, n_restarts - 1)
    extras: list[int] = []
    rng = np.random.default_rng(rng_seed)

    if nonzero:
        # Symmetric quantile grid avoiding 0 and 1 (equivalent to equally spaced
        # probabilities strictly inside the distribution's support).
        qs = np.linspace(1.0 / (k_extra + 1), k_extra / (k_extra + 1), k_extra)
        quantile_candidates = [int(np.clip(round(q), 1, p_max))
                               for q in np.quantile(nonzero, qs)]
        extras.extend(quantile_candidates)

    # Fill any remaining slots with rng-sampled draws from the observed n_tau
    # so the restart surface is not degenerate when the stream's n_tau support
    # is thin (common for traffic/manufacturing sub-pubs).
    while len(extras) < k_extra:
        if nonzero:
            extras.append(int(np.clip(rng.choice(nonzero), 1, p_max)))
        else:
            extras.append(int(rng.integers(1, max(2, p_max) + 1)))

    for s in extras:
        s = int(np.clip(s, 1, max(1, p_max)))
        if s not in seeds:
            seeds.append(s)
    return seeds


def _greedy_hillclimb_from(
    agg, cnt, payload_bound, epsilon, w, strategy, p_max,
    P_start: int, utility_weight, latency_weight,
    I_max: int, loss_tie_tol: float, seed: int,
    restart_idx: int, action_prefix: str = "",
) -> tuple[list[dict], dict, int]:
    """One hill-climb trajectory from ``P_start``.  Returns (trajectory rows,
    best evaluated row, evaluation count for this run).
    """
    evaluations = 0
    trajectory: list[dict] = []
    P = max(1, min(P_start, p_max))
    best = _evaluate_stream(agg, cnt, payload_bound, epsilon, w, P, strategy,
                            utility_weight, latency_weight, seed)
    evaluations += 1
    trajectory.append({
        **best, "iter": 0,
        "action": f"{action_prefix}seed",
        "restart_idx": restart_idx,
    })
    seen = {P: best["tuning_loss"]}
    for i in range(1, I_max + 1):
        neighbors = []
        for P_nbr in (P - 1, P + 1):
            if P_nbr < 1 or P_nbr > p_max:
                continue
            if P_nbr in seen:
                neighbors.append({"P": P_nbr, "tuning_loss": seen[P_nbr],
                                  "_cached": True})
                continue
            r = _evaluate_stream(agg, cnt, payload_bound, epsilon, w, P_nbr, strategy,
                                 utility_weight, latency_weight, seed)
            evaluations += 1
            seen[P_nbr] = r["tuning_loss"]
            neighbors.append({**r, "_cached": False})
            trajectory.append({
                **r, "iter": i,
                "action": f"{action_prefix}probe_P={P_nbr}",
                "restart_idx": restart_idx,
            })
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
                best = _evaluate_stream(agg, cnt, payload_bound, epsilon, w, P, strategy,
                                        utility_weight, latency_weight, seed)
                evaluations += 1
            trajectory.append({
                **best, "iter": i,
                "action": f"{action_prefix}step_to_P={P}",
                "restart_idx": restart_idx,
            })
        else:
            trajectory.append({
                **best, "iter": i,
                "action": f"{action_prefix}stop_local_optimum",
                "restart_idx": restart_idx,
            })
            break
    return trajectory, best, evaluations


def _greedy_walk_on_stream(
    agg, cnt, payload_bound, epsilon, w, strategy, p_max,
    alpha=0.25, I_max=20, utility_weight=1.0, latency_weight=0.2,
    loss_tie_tol: float = 1e-9, seed: int = 77,
    n_restarts: int = 3, restart_rng_seed: int = 12345,
) -> dict:
    """Stream-level Algorithm 2 walk with intelligent multi-start.

    The first restart seeds at P_0 = ceil(1/alpha) (paper Section 6.7).
    Additional restarts are sampled from quantiles of the observed n_tau
    distribution, so the hill-climb explores the pool-density regimes the
    stream actually exhibits instead of always starting at the same point.
    Each restart runs an independent greedy walk; we return the best local
    optimum across restarts.
    """
    P_0 = max(1, int(np.ceil(1.0 / max(alpha, 1e-9))))
    P_seeds = _intelligent_restart_seeds(
        cnt, alpha=alpha, p_max=max(1, p_max),
        n_restarts=max(1, n_restarts), rng_seed=restart_rng_seed,
    )

    all_traj: list[dict] = []
    best_overall: dict | None = None
    total_evals = 0
    for r_idx, P_start in enumerate(P_seeds):
        prefix = "" if r_idx == 0 else f"r{r_idx}_"
        traj, best_here, evals = _greedy_hillclimb_from(
            agg, cnt, payload_bound, epsilon, w, strategy, p_max,
            P_start=P_start, utility_weight=utility_weight,
            latency_weight=latency_weight, I_max=I_max,
            loss_tie_tol=loss_tie_tol, seed=seed,
            restart_idx=r_idx, action_prefix=prefix,
        )
        all_traj.extend(traj)
        total_evals += evals
        if best_overall is None or (
            best_here["tuning_loss"]
            < best_overall["tuning_loss"] - loss_tie_tol
        ) or (
            abs(best_here["tuning_loss"] - best_overall["tuning_loss"])
            <= loss_tie_tol
            and best_here["P"] > best_overall["P"]
        ):
            best_overall = best_here

    return {
        "strategy": strategy,
        "best": best_overall,
        "seed_P": P_0,
        "restart_seeds_P": P_seeds,
        "trajectory": pd.DataFrame(all_traj),
        "evaluations": total_evals,
        "n_restarts": len(P_seeds),
    }


def greedy_tune_P(
    per_pub, payload_bound, epsilon, w, strategy, p_max,
    alpha=0.25, I_max=20, utility_weight=1.0, latency_weight=0.2,
    loss_tie_tol: float = 1e-9,
    n_restarts: int = 3, restart_rng_seed: int = 12345,
) -> dict:
    """Algorithm 2 from the paper: greedy hill-climb over P with intelligent
    multi-start (seeded at P_0 = ceil(1/alpha) plus additional restarts drawn
    from quantiles of the observed n_tau distribution).

    Compatibility shim: rebuilds the stream from ``per_pub`` and delegates
    to ``_greedy_walk_on_stream``.  Callers with a cached stream should use
    that helper directly.
    """
    agg, cnt = _rebuild_stream_with_dt(per_pub, 1)
    return _greedy_walk_on_stream(
        agg, cnt, payload_bound, epsilon, w, strategy, p_max,
        alpha=alpha, I_max=I_max,
        utility_weight=utility_weight, latency_weight=latency_weight,
        loss_tie_tol=loss_tie_tol,
        n_restarts=n_restarts, restart_rng_seed=restart_rng_seed,
    )


def brute_force_tune_P(
    per_pub, payload_bound, epsilon, w, strategy, p_max,
    utility_weight=1.0, latency_weight=0.2,
) -> pd.DataFrame:
    """Naive full enumeration: evaluate every P in [1, p_max] for one strategy.

    Compatibility shim: rebuilds the stream once and delegates per-P to
    ``_evaluate_stream``.  Callers with a cached stream should use the
    parallel brute-force path in ``tune_hyperparameters`` instead.
    """
    agg, cnt = _rebuild_stream_with_dt(per_pub, 1)
    rows = []
    for P in range(1, int(p_max) + 1):
        rows.append(_evaluate_stream(agg, cnt, payload_bound, epsilon, w, P, strategy,
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
    workers: int = 1,
    n_restarts: int = 3,
    restart_rng_seed: int = 12345,
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

    # Rebuild the stream exactly once; every greedy probe + brute-force eval
    # reuses the same cached (agg, cnt).  Worker processes see the same cached
    # tuple via the initializer, so each task is a pure run_dp_on_stream call.
    logger.info(f"  [{dataset_name}/{sensor_name}] tune_hyperparameters: rebuilding stream...")
    agg, cnt = _rebuild_stream_with_dt(per_pub, 1)
    obs_max = max(cnt) if cnt else 1
    if p_max is None:
        p_max = max(2, min(obs_max, len(per_pub)))
    p_max = int(p_max)

    tune_init_args = (agg, cnt, payload_bound, epsilon, w,
                      utility_weight, latency_weight)

    # Brute-force: one task per (strategy, P).
    brute_tasks = [(strat, P, 77)
                   for strat in strategies
                   for P in range(1, p_max + 1)]
    logger.info(f"  [{dataset_name}/{sensor_name}] brute-force: dispatching "
                f"{len(brute_tasks)} (strategy, P) tasks across {workers} worker(s)")
    brute_rows_flat = _run_parallel_tasks(
        brute_tasks, _tune_eval_task,
        workers=workers,
        initializer=_init_tune_worker,
        initargs=tune_init_args,
        progress_label=f"  [{dataset_name}/{sensor_name}] brute",
        progress_every=max(10, len(brute_tasks) // 10),
    )
    brute_df = pd.DataFrame(brute_rows_flat)

    # Greedy: one task per strategy (walks are serial within, but across
    # strategies they're independent).  Each task carries its n_restarts and
    # restart-rng seed so workers can reproduce the intelligent multi-start
    # seed schedule.
    greedy_tasks = [(strat, p_max, alpha, I_max, 1e-9, 77,
                     n_restarts, restart_rng_seed + i)
                    for i, strat in enumerate(strategies)]
    logger.info(f"  [{dataset_name}/{sensor_name}] greedy: dispatching "
                f"{len(greedy_tasks)} walks across "
                f"{min(workers, len(greedy_tasks))} worker(s)")
    greedy_out = _run_parallel_tasks(
        greedy_tasks, _greedy_walk_task,
        workers=min(workers, len(greedy_tasks)),
        initializer=_init_tune_worker,
        initargs=tune_init_args,
        progress_label=f"  [{dataset_name}/{sensor_name}] greedy",
        progress_every=1,
    )

    greedy_rows: list[pd.DataFrame] = []
    greedy_results: dict[str, dict] = {}
    for strat, g in zip(strategies, greedy_out):
        g["trajectory"]["strategy"] = strat
        greedy_rows.append(g["trajectory"])
        greedy_results[strat] = g

    greedy_df = pd.concat(greedy_rows, ignore_index=True) if greedy_rows else pd.DataFrame()

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
        restart_seeds = g_res.get("restart_seeds_P", [g_res.get("seed_P", 1)])
        summary_rows.append({
            "strategy": strat,
            "greedy_P": int(g_best["P"]),
            "greedy_loss": float(g_best["tuning_loss"]),
            "greedy_nmae": g_best["normalized_mae"],
            "greedy_kl": g_best["kl_divergence"],
            "greedy_release_rate": g_best["release_rate"],
            "greedy_evaluations": g_evals,
            "greedy_seed_P": g_res["seed_P"],
            "greedy_n_restarts": int(g_res.get("n_restarts", 1)),
            "greedy_restart_seeds_P": ";".join(str(p) for p in restart_seeds),
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
    workers: int = 1,
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
    logger.info(f"  [{dataset_name}] figure1_reproduction: building tasks...")
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
    P_values = [p for p in [2, 3, 4, 6, 8] if p <= N_pubs]

    # Global (Extreme 1) stream: one stream per system, single noise per tau
    # with R = sup(R) and n_tau = total publishers across metrics.
    all_B = max(B for _, B in per_pub_all.values())
    cross_metric_true: list[float] = []
    num_pubs_total: list[int] = []
    for tau in range(T):
        vals = [sensor_streams[s][0][tau] for s in sensor_names
                if sensor_streams[s][1][tau] > 0]
        cross_metric_true.append(float(np.mean(vals)) if vals else 0.0)
        num_pubs_total.append(sum(sensor_streams[s][1][tau] for s in sensor_names))

    # Pack every stream the workers might need into one keyed dict.
    named: dict = {}
    for s, (pp, B) in per_pub_all.items():
        for pub_id, series in pp.items():
            pa = [v if v is not None else 0.0 for v in series]
            pc = [1 if v is not None else 0 for v in series]
            named[("pub", s, pub_id)] = (pa, pc, B)
        named[("agg", s)] = sensor_streams[s]
    named[("global",)] = (cross_metric_true, num_pubs_total, all_B)

    # Build the task list.  Each task carries an "op" tag so we can bucket the
    # results (p1 / mid[P] / global[sensor]) when everything finishes.
    tasks: list = []
    ops: list = []  # parallel to `tasks`
    idx = 0
    # Per-publisher extreme (P=1)
    for s in sensor_names:
        pp, _ = per_pub_all[s]
        for i, pub_id in enumerate(pp.keys()):
            for trial in range(n_trials):
                seed = 500_000 + i * 1000 + trial
                tasks.append((idx, ("pub", s, pub_id), "uniform", 1,
                              epsilon, w, seed, "metrics"))
                ops.append(("p1",))
                idx += 1
    # Intermediate P on clamped aggregate
    for P in P_values:
        for s in sensor_names:
            agg, cnt, _ = sensor_streams[s]
            if max(cnt) < P:
                continue
            for trial in range(n_trials):
                seed = 600_000 + P * 1000 + trial
                tasks.append((idx, ("agg", s), "uniform", P,
                              epsilon, w, seed, "metrics"))
                ops.append(("mid", P))
                idx += 1
    # Global (Extreme 1) — need noisy_values back for custom KL vs per-metric truth
    for s in sensor_names:
        for trial in range(n_trials):
            seed = 700_000 + hash(s) % 10000 + trial * 7919
            tasks.append((idx, ("global",), "uniform", 1,
                          epsilon, w, seed, "noisy"))
            ops.append(("global", s))
            idx += 1

    logger.info(f"  [{dataset_name}] figure1_reproduction: dispatching "
                f"{len(tasks)} DP runs across {workers} worker(s)")
    results = _run_parallel_tasks(
        tasks, _dp_named_task,
        workers=workers,
        initializer=_init_named_streams_worker,
        initargs=(named,),
        progress_label=f"  [{dataset_name}] figure1",
        progress_every=max(50, len(tasks) // 20),
    )

    # Aggregate by bucket.
    p1_kls: list[float] = []
    mid_kls_buf: dict[int, list[float]] = {P: [] for P in P_values}
    g_kls: list[float] = []
    for op, res in zip(ops, results):
        kind = op[0]
        if kind == "p1":
            k = res["kl"]
            if np.isfinite(k):
                p1_kls.append(k)
        elif kind == "mid":
            _, P = op
            k = res["kl"]
            if np.isfinite(k):
                mid_kls_buf[P].append(k)
        else:  # "global"
            _, s = op
            true_stream = sensor_streams[s][0][:T]
            nvals = res["noisy_arr"]
            true_c = [v for v in true_stream if v is not None]
            nc = [float(v) for v in nvals if np.isfinite(v)]
            if len(true_c) >= 10 and len(nc) >= 10:
                k = compute_kl_divergence(true_c, nc)
                if np.isfinite(k):
                    g_kls.append(k)

    kl_p1 = float(np.mean(p1_kls)) if p1_kls else float("nan")
    mid_kls: dict[int, float] = {
        P: (float(np.mean(v)) if v else float("nan"))
        for P, v in mid_kls_buf.items()
    }
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
INTRO_N_TRIALS = 20   # seed average for Figure 1 (Laplace noise is high-variance)


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
                        dataset_name, output_dir, workers: int = 1):
    """Grouped bar of KL for Extreme 1 / Extreme 2 / Our-approach per sensor.

    All three regimes are averaged over `INTRO_N_TRIALS` noise seeds so the
    bars reflect the expected distortion, not a single-draw artefact.
    """
    logger.info(f"  [{dataset_name}] kl_extremes_vs_ours: building tasks...")
    T = min(len(v[0]) for v in sensor_streams.values())
    sensor_names = list(sensor_streams.keys())
    global_B = max(v[2] for v in sensor_streams.values())

    # Extreme 1 (paper §1.3): one stream per system.
    global_true: list[float] = []
    for tau in range(T):
        vals = [sensor_streams[s][0][tau] for s in sensor_names
                if sensor_streams[s][1][tau] > 0]
        global_true.append(float(np.mean(vals)) if vals else 0.0)
    num_pubs_global = [sum(sensor_streams[s][1][tau] for s in sensor_names)
                       for tau in range(T)]

    # Pack every keyed stream into the worker pool.
    named: dict = {}
    for s in sensor_names:
        if s in per_pub_all:
            pp, B = per_pub_all[s]
            for pub_id, series in pp.items():
                pa = [v if v is not None else 0.0 for v in series]
                pc = [1 if v is not None else 0 for v in series]
                named[("pub", s, pub_id)] = (pa, pc, B)
        named[("agg", s)] = sensor_streams[s]
    named[("global",)] = (global_true, num_pubs_global, global_B)

    tasks: list = []
    ops: list = []
    idx = 0
    # Extreme 1: INTRO_N_TRIALS global DP runs; KL per-sensor computed in caller.
    for trial in range(INTRO_N_TRIALS):
        seed = 200_000 + trial * 13
        tasks.append((idx, ("global",), "uniform", 1,
                      INTRO_EPSILON, INTRO_W, seed, "noisy"))
        ops.append(("e1",))
        idx += 1
    # Extreme 2: per-publisher DP
    for s in sensor_names:
        if s not in per_pub_all:
            continue
        pp, _ = per_pub_all[s]
        for i, pub_id in enumerate(pp.keys()):
            for trial in range(INTRO_N_TRIALS):
                seed = 300_000 + i * 1000 + trial
                tasks.append((idx, ("pub", s, pub_id), "uniform", 1,
                              INTRO_EPSILON, INTRO_W, seed, "self_kl"))
                ops.append(("e2", s))
                idx += 1
    # Our approach: P-gated BA at P_our on each sensor's clamped aggregate
    for s in sensor_names:
        for trial in range(INTRO_N_TRIALS):
            seed = 400_000 + trial
            tasks.append((idx, ("agg", s), "p_gated_ba", P_our,
                          INTRO_EPSILON, INTRO_W, seed, "self_kl"))
            ops.append(("ours", s))
            idx += 1

    logger.info(f"  [{dataset_name}] kl_extremes_vs_ours: dispatching "
                f"{len(tasks)} DP runs across {workers} worker(s)")
    run_results = _run_parallel_tasks(
        tasks, _dp_named_task,
        workers=workers,
        initializer=_init_named_streams_worker,
        initargs=(named,),
        progress_label=f"  [{dataset_name}] extremes",
        progress_every=max(50, len(tasks) // 20),
    )

    e1: dict[str, list[float]] = {s: [] for s in sensor_names}
    e2: dict[str, list[float]] = {s: [] for s in sensor_names}
    ours: dict[str, list[float]] = {s: [] for s in sensor_names}
    for op, res in zip(ops, run_results):
        kind = op[0]
        if kind == "e1":
            noisy_arr = res["noisy_arr"]
            noisy_list = [float(v) if np.isfinite(v) else None for v in noisy_arr]
            for s in sensor_names:
                k = compute_kl_divergence(
                    list(sensor_streams[s][0][:T]), noisy_list,
                )
                if np.isfinite(k):
                    e1[s].append(k)
        elif kind == "e2":
            _, s = op
            k = res["kl_self"]
            if np.isfinite(k):
                e2[s].append(k)
        else:  # "ours"
            _, s = op
            k = res["kl_self"]
            if np.isfinite(k):
                ours[s].append(k)

    results = {
        "Extreme 1\n(global average)":
            {s: float(np.mean(v)) if v else float("nan") for s, v in e1.items()},
        "Extreme 2\n(per-publisher)":
            {s: float(np.mean(v)) if v else float("nan") for s, v in e2.items()},
        f"Our approach\n(P={P_our} topic pool)":
            {s: float(np.mean(v)) if v else float("nan") for s, v in ours.items()},
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


def _kl_of(true_vals, noisy_vals):
    tc = [v for v in true_vals if v is not None]
    nc = [v for v in noisy_vals if v is not None]
    if len(tc) < 10 or len(nc) < 10:
        return float("nan")
    k = compute_kl_divergence(tc, nc)
    return k if np.isfinite(k) else float("nan")


def u_shaped_curve(sensor_streams, per_pub_all, dataset_name, output_dir,
                   workers: int = 1):
    """Reproduce paper Figure 1 on real data: KL vs aggregation scope P.

    All three regimes use Uniform budget allocation (paper §1.3: Figure 1 is
    about the effect of AGGREGATION SCOPE, not budget-allocation strategy).
    Each (P, sensor) is averaged over INTRO_N_TRIALS noise seeds to suppress
    single-draw Laplace variance.
    """
    logger.info(f"  [{dataset_name}] u_shaped_curve: building tasks...")
    sensor_names = list(sensor_streams.keys())
    T = min(len(v[0]) for v in sensor_streams.values())
    sweep_p = [2, 3, 4, 6, 8]

    # Cross-metric global stream (Extreme 1 / "one stream per system").
    all_B = max(v[2] for v in sensor_streams.values())
    num_pubs_total = [sum(v[1][tau] for v in sensor_streams.values())
                      for tau in range(T)]
    cross_metric_true: list[float] = []
    for tau in range(T):
        vals = [sensor_streams[s][0][tau] for s in sensor_names
                if sensor_streams[s][1][tau] > 0]
        cross_metric_true.append(float(np.mean(vals)) if vals else 0.0)

    # Pack every stream keyed for the worker pool.
    named: dict = {}
    for s in sensor_names:
        if s in per_pub_all:
            pp, B = per_pub_all[s]
            for pub_id, series in pp.items():
                pa = [v if v is not None else 0.0 for v in series]
                pc = [1 if v is not None else 0 for v in series]
                named[("pub", s, pub_id)] = (pa, pc, B)
        named[("agg", s)] = sensor_streams[s]
    named[("global",)] = (cross_metric_true, num_pubs_total, all_B)

    tasks: list = []
    ops: list = []
    idx = 0
    # P=1 per-publisher uses kl_self (_kl_of semantics)
    for s in sensor_names:
        if s not in per_pub_all:
            continue
        pp, _ = per_pub_all[s]
        for i, pub_id in enumerate(pp.keys()):
            for trial in range(INTRO_N_TRIALS):
                seed = 1_000_000 + i * 1000 + trial
                tasks.append((idx, ("pub", s, pub_id), "uniform", 1,
                              INTRO_EPSILON, INTRO_W, seed, "self_kl"))
                ops.append(("p1",))
                idx += 1
    # Intermediate P on clamped aggregate
    for P in sweep_p:
        for s in sensor_names:
            agg, cnt, _ = sensor_streams[s]
            if max(cnt) < P:
                continue
            for trial in range(INTRO_N_TRIALS):
                seed = 2_000_000 + P * 1000 + trial
                tasks.append((idx, ("agg", s), "uniform", P,
                              INTRO_EPSILON, INTRO_W, seed, "self_kl"))
                ops.append(("mid", P))
                idx += 1
    # Global (Extreme 1) — compare sensor's true vs global-DP noisy
    for s in sensor_names:
        for trial in range(INTRO_N_TRIALS):
            seed = 5_000_000 + hash(s) % 10000 + trial * 7919
            tasks.append((idx, ("global",), "uniform", 1,
                          INTRO_EPSILON, INTRO_W, seed, "noisy"))
            ops.append(("global", s))
            idx += 1

    logger.info(f"  [{dataset_name}] u_shaped_curve: dispatching "
                f"{len(tasks)} DP runs across {workers} worker(s)")
    results = _run_parallel_tasks(
        tasks, _dp_named_task,
        workers=workers,
        initializer=_init_named_streams_worker,
        initargs=(named,),
        progress_label=f"  [{dataset_name}] u_shape",
        progress_every=max(50, len(tasks) // 20),
    )

    p1_kls: list[float] = []
    mid_kls_buf: dict[int, list[float]] = {P: [] for P in sweep_p}
    g_kls: list[float] = []
    for op, res in zip(ops, results):
        kind = op[0]
        if kind == "p1":
            k = res["kl_self"]
            if np.isfinite(k):
                p1_kls.append(k)
        elif kind == "mid":
            _, P = op
            k = res["kl_self"]
            if np.isfinite(k):
                mid_kls_buf[P].append(k)
        else:  # "global": compare sensor's true vs cross-metric noisy
            _, s = op
            true_stream = sensor_streams[s][0][:T]
            noisy_arr = res["noisy_arr"]
            # Replicate _kl_of: compute_kl_divergence handles None/NaN filtering.
            k = compute_kl_divergence(list(true_stream),
                                      [float(v) if np.isfinite(v) else None
                                       for v in noisy_arr])
            if np.isfinite(k):
                g_kls.append(k)

    kl_p1 = float(np.mean(p1_kls)) if p1_kls else float("nan")
    mid_kls: dict[int, float] = {
        P: (float(np.mean(v)) if v else float("nan"))
        for P, v in mid_kls_buf.items()
    }
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


def run_intro_figures(sensor_streams, per_pub_all, dataset_name, output_dir,
                      P_our=4, workers: int = 1):
    """Produce the four paper-style intro figures for one dataset."""
    if not sensor_streams:
        return
    extreme1_global(sensor_streams, output_dir, dataset_name)
    if per_pub_all:
        pp_sensor = next(iter(per_pub_all))
        pp_data, pp_B = per_pub_all[pp_sensor]
        extreme2_per_publisher(pp_data, pp_B, pp_sensor, dataset_name, output_dir)
    kl_extremes_vs_ours(sensor_streams, per_pub_all, P_our, dataset_name, output_dir,
                        workers=workers)
    u_shaped_curve(sensor_streams, per_pub_all, dataset_name, output_dir,
                   workers=workers)


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


def run_dataset(
    name: str,
    s_values, eps_values, w_values, strategies, output_dir, args,
    clamp_mode: str = "static",
    quick: bool = False, skip_extras: bool = False,
    workers: int = 1,
) -> dict:
    """Run the full experiment on one (dataset, clamp_mode) pair.

    Returns a dict of DataFrames for cross-dataset aggregation.
    """
    spec = DATASETS[name]
    logger.info("=" * 72)
    logger.info(f"{name.upper()} / clamp_mode={clamp_mode} :: {spec['label']}")
    logger.info("=" * 72)

    prepared = prepare_dataset(
        name,
        clamp_mode=clamp_mode,
        eps_clip=args.eps_clip,
        seed=args.seed,
        max_rows=_dataset_max_rows(name, args),
    )
    if prepared is None:
        return {}
    if not prepared.raw_streams:
        logger.error(f"No valid {name} streams; skipping."); return {}
    if prepared.is_empty:
        logger.error(f"No valid {name} streams after {clamp_mode} clamping"); return {}
    streams, per_pubs, clamp_meta = prepared.streams, prepared.per_pubs, prepared.clamp_meta

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

    log_messages = getattr(args, "log_messages", True)
    messages_dir = os.path.join(dirs["root"], "messages")
    os.makedirs(messages_dir, exist_ok=True)
    sweep_messages_csv = os.path.join(messages_dir, "sweep_messages.csv")
    df = sweep(
        name, streams, s_values, eps_values, w_values, strategies,
        workers=workers,
        clamp_mode=clamp_mode,
        log_messages=log_messages,
        messages_csv_path=sweep_messages_csv if log_messages else None,
    )
    df["clamp_mode"] = clamp_mode
    df["eps_clip"] = args.eps_clip if clamp_mode == "dp_released" else 0.0
    df.to_csv(os.path.join(dirs["sweep"], "sweep_results.csv"), index=False)
    print_summary(df, f"{spec['label']} [clamp_mode={clamp_mode}]")
    if getattr(args, "generate_plots", False):
        plot_results(df, name, dirs["sweep"], streams, workers=workers)

    results: dict = {"sweep": df}
    logger.info(f"  [{name}/{clamp_mode}] --- phase: intro figures ---")
    run_intro_figures(streams, per_pubs, name, dirs["intro"], P_our=4,
                      workers=workers)

    if not skip_extras:
        w_mid = max(w_values) // 2
        logger.info(f"  [{name}/{clamp_mode}] --- phase: n-weighted spotlight ---")
        n_weighted_spotlight(streams, name, dirs["extras"], epsilon=1.0, w=w_mid, P=2)
        logger.info(f"  [{name}/{clamp_mode}] --- phase: figure1 reproduction ---")
        fig1_df = figure1_reproduction(per_pubs, name, dirs["intro"],
                                       epsilon=1.0, w=w_mid,
                                       workers=workers)
        fig1_df = fig1_df.copy()
        fig1_df["clamp_mode"] = clamp_mode
        results["figure1"] = fig1_df
        logger.info(f"  [{name}/{clamp_mode}] --- phase: collusion ---")
        collusion_experiment(streams, name, dirs["extras"], epsilon=1.0, w=w_mid, P=2,
                             trials_per_c=16 if quick else 32,
                             workers=workers)
        if per_pubs:
            sensor_name = next(iter(per_pubs))
            pp, B = per_pubs[sensor_name]
            logger.info(f"  [{name}/{clamp_mode}] --- phase: K_ext sweep ({sensor_name}) ---")
            dynamic_interval_experiment(
                pp, B, name, sensor_name, dirs["extras"],
                epsilon=1.0, w=w_mid, P=3,
                k_ext_values=(0, 1, 2, 4) if quick else (0, 1, 2, 4, 8),
                workers=workers,
            )
            logger.info(f"  [{name}/{clamp_mode}] --- phase: hyperparameter tuning ({sensor_name}) ---")
            tune = tune_hyperparameters(
                pp, B, name, sensor_name, dirs["tuning"],
                epsilon=1.0, w=w_mid,
                strategies=strategies,
                alpha=args.alpha, I_max=args.I_max,
                workers=workers,
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


def _iter_clamped_by_dataset(datasets, clamp_mode, eps_clip, seed, args):
    """Yield (ds_name, [(sensor, pp, R, (agg, cnt)), ...]) grouped per dataset.

    One list per dataset so callers can launch a fresh worker pool per
    dataset.  Keeps memory bounded: only one dataset is held at a time.
    """
    for ds_name in datasets:
        prepared = prepare_dataset(
            ds_name,
            clamp_mode=clamp_mode,
            eps_clip=eps_clip,
            seed=seed,
            max_rows=_dataset_max_rows(ds_name, args),
        )
        if prepared is None or not prepared.raw_per_pubs:
            continue
        entries = []
        for sensor, (agg, cnt, R) in prepared.streams.items():
            pp = prepared.per_pubs[sensor][0]
            entries.append((sensor, pp, R, (agg, cnt)))
        if entries:
            yield ds_name, entries


def _experiment_A_task(task):
    """One (sensor, combo, strategy) Experiment-A point: greedy + brute."""
    (ds_name, sensor, pp, R, combo, strat, alpha, I_max, p_max,
     clamp_mode, n_restarts, restart_rng_seed) = task
    eps = combo["epsilon"]
    w = combo["w"]
    g = greedy_tune_P(pp, R, eps, w, strat, p_max,
                      alpha=alpha, I_max=I_max,
                      n_restarts=n_restarts,
                      restart_rng_seed=restart_rng_seed)
    b_df = brute_force_tune_P(pp, R, eps, w, strat, p_max)
    b_best = (b_df.assign(_neg=-b_df["P"])
              .sort_values(["tuning_loss", "_neg"]).iloc[0])
    g_best = g["best"]
    return {
        "dataset": ds_name, "sensor": sensor, "strategy": strat,
        "clamp_mode": clamp_mode,
        "epsilon": eps, "w": w, "p_max": p_max,
        "payload_bound": R,
        "alpha": alpha,
        "I_max": I_max,
        "greedy_P": int(g_best["P"]),
        "greedy_loss": float(g_best["tuning_loss"]),
        "greedy_nmae": g_best["normalized_mae"],
        "greedy_kl": g_best["kl_divergence"],
        "greedy_release_rate": g_best["release_rate"],
        "greedy_attribution_advantage": g_best["attribution_advantage"],
        "greedy_evaluations": g["evaluations"],
        "greedy_seed_P": g["seed_P"],
        "greedy_n_restarts": int(g.get("n_restarts", 1)),
        "greedy_restart_seeds_P": ";".join(
            str(p) for p in g.get("restart_seeds_P", [g.get("seed_P", 1)])
        ),
        "brute_P": int(b_best["P"]),
        "brute_loss": float(b_best["tuning_loss"]),
        "brute_nmae": b_best["normalized_mae"],
        "brute_kl": b_best["kl_divergence"],
        "brute_release_rate": b_best["release_rate"],
        "brute_attribution_advantage": b_best["attribution_advantage"],
        "brute_evaluations": int(p_max),
        "gap_loss": float(g_best["tuning_loss"] - b_best["tuning_loss"]),
        "gap_P": int(g_best["P"] - b_best["P"]),
        "speedup": float(p_max / max(g["evaluations"], 1)),
    }


def _experiment_single_axis_task(task):
    """Shared worker for Experiment B (vary w) and C (vary eps)."""
    (ds_name, sensor, agg, cnt, R, strategy, P, eps, w,
     clamp_mode, log_messages, experiment_tag) = task
    res = run_dp_on_stream(
        agg, cnt, epsilon=eps, window_size=w,
        min_publishers=P, payload_bound=R,
        strategy=strategy, seed=77,
    )
    m = res["metrics"]
    elig_n = [n for n in cnt if n > 0]
    avg_n = float(np.mean(elig_n)) if elig_n else 0.0
    # Paper Thm 5.1 + Uniform allocation: lambda = R * w / (n * eps);
    # expected NMAE = lambda / R = w / (n * eps).
    pred_lambda = R * w / (avg_n * eps) if avg_n > 0 else float("nan")
    pred_nmae = w / (avg_n * eps) if avg_n > 0 else float("nan")
    out = {
        "dataset": ds_name, "sensor": sensor,
        "clamp_mode": clamp_mode,
        "strategy": strategy, "P": P,
        "epsilon": eps, "w": w,
        "payload_bound": R,
        "normalized_mae": m["normalized_mae"],
        "predicted_nmae_uniform": pred_nmae,
        "predicted_laplace_scale": pred_lambda,
        "kl_divergence": m["kl_divergence"],
        "release_rate": m["release_rate"],
        "attribution_advantage": m["attribution_advantage"],
        "avg_n_tau_eligible": avg_n,
        "avg_n_tau": float(np.mean(cnt)) if cnt else 0.0,
        "mae": m["mae"],
        "deferrals": m["deferrals"],
        "num_timestamps": len(agg),
        "seed": 77,
    }
    if log_messages:
        from message_logger import build_message_rows
        out["_messages"] = build_message_rows(
            res,
            dataset=ds_name, clamp_mode=clamp_mode, sensor=sensor,
            strategy=strategy, P=P, epsilon=eps, w=w,
            payload_bound=R, seed=77, experiment=experiment_tag,
        )
    return out


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
    workers = _default_workers(getattr(args, "workers", None))
    rows = []
    for ds_name, entries in _iter_clamped_by_dataset(
            datasets, clamp_mode, args.eps_clip, args.seed, args):
        tasks = []
        for sensor, pp, R, _ in entries:
            _, cnt = _rebuild_stream_with_dt(pp, 1)
            p_max = max(2, min(max(cnt), len(pp)))
            for combo in fixed_combos:
                for strat in strategies:
                    tasks.append((
                        ds_name, sensor, pp, R, combo, strat,
                        args.alpha, args.I_max, p_max, clamp_mode,
                        int(getattr(args, "n_restarts", 3)),
                        int(getattr(args, "restart_rng_seed", 12345))
                        + hash(strat) % 1000,
                    ))
        batch = _run_parallel_tasks(
            tasks, _experiment_A_task, workers=workers,
            progress_label=f"  [exp A {ds_name}/{clamp_mode}]",
            progress_every=20,
        )
        rows.extend(batch)

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


def _split_messages_from_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Pop ``_messages`` off every returned task row; return (summary, msgs)."""
    messages: list[dict] = []
    summaries: list[dict] = []
    for r in rows:
        if isinstance(r, dict) and "_messages" in r:
            messages.extend(r.pop("_messages"))
        summaries.append(r)
    return summaries, messages


def experiment_B_vary_w(
    datasets, clamp_mode, output_dir, args,
    fixed_combos=None, w_values=None,
) -> pd.DataFrame:
    """Experiment B: fix (P, eps, strategy), sweep w."""
    fixed_combos = fixed_combos or EXPERIMENT_FIXED_COMBOS_B
    w_values = w_values or EXPERIMENT_B_W_VALUES
    workers = _default_workers(getattr(args, "workers", None))
    log_messages = getattr(args, "log_messages", True)
    rows = []
    for ds_name, entries in _iter_clamped_by_dataset(
            datasets, clamp_mode, args.eps_clip, args.seed, args):
        tasks = []
        for sensor, _, R, (agg, cnt) in entries:
            for combo in fixed_combos:
                for w in w_values:
                    tasks.append((
                        ds_name, sensor, agg, cnt, R,
                        combo["strategy"], combo["P"], combo["epsilon"], w,
                        clamp_mode, log_messages, "B_vary_w",
                    ))
        batch = _run_parallel_tasks(
            tasks, _experiment_single_axis_task, workers=workers,
            progress_label=f"  [exp B {ds_name}/{clamp_mode}]",
            progress_every=50,
        )
        rows.extend(batch)
    summaries, messages = _split_messages_from_rows(rows)
    df = pd.DataFrame(summaries)
    exp_dir = os.path.join(output_dir, "experiments", "B_vary_w")
    os.makedirs(exp_dir, exist_ok=True)
    df.to_csv(os.path.join(exp_dir, "experiment_B_vary_w.csv"), index=False)
    if log_messages and messages:
        from message_logger import write_messages_csv
        n = write_messages_csv(
            messages,
            os.path.join(exp_dir, "experiment_B_messages.csv"),
        )
        logger.info(f"  Experiment B wrote {n} per-release messages")
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
    workers = _default_workers(getattr(args, "workers", None))
    log_messages = getattr(args, "log_messages", True)
    rows = []
    for ds_name, entries in _iter_clamped_by_dataset(
            datasets, clamp_mode, args.eps_clip, args.seed, args):
        tasks = []
        for sensor, _, R, (agg, cnt) in entries:
            for combo in fixed_combos:
                for eps in eps_values:
                    tasks.append((
                        ds_name, sensor, agg, cnt, R,
                        combo["strategy"], combo["P"], eps, combo["w"],
                        clamp_mode, log_messages, "C_vary_epsilon",
                    ))
        batch = _run_parallel_tasks(
            tasks, _experiment_single_axis_task, workers=workers,
            progress_label=f"  [exp C {ds_name}/{clamp_mode}]",
            progress_every=50,
        )
        rows.extend(batch)
    summaries, messages = _split_messages_from_rows(rows)
    df = pd.DataFrame(summaries)
    exp_dir = os.path.join(output_dir, "experiments", "C_vary_epsilon")
    os.makedirs(exp_dir, exist_ok=True)
    df.to_csv(os.path.join(exp_dir, "experiment_C_vary_epsilon.csv"), index=False)
    if log_messages and messages:
        from message_logger import write_messages_csv
        n = write_messages_csv(
            messages,
            os.path.join(exp_dir, "experiment_C_messages.csv"),
        )
        logger.info(f"  Experiment C wrote {n} per-release messages")
    logger.info(f"  Experiment C wrote {len(df)} rows -> {exp_dir}")

    _plot_single_axis_experiment(
        df, x_col="epsilon", x_label="privacy budget  ε",
        path=os.path.join(exp_dir, "experiment_C_vary_epsilon.png"),
        title=f"Experiment C [clamp={clamp_mode}]: NMAE and KL vs ε",
        logx=True,
    )
    return df


# ═════════════════════════════════════════════════════════════════════════
#  Experiment D: plugin end-to-end via _on_message (in-process, stubbed MQTT)
# ═════════════════════════════════════════════════════════════════════════
#
# Drives the broker-side `PrivacyPlugin` without an actual MQTT broker:
#   1. Swaps `plugin._client` with a stub that captures publishes.
#   2. Monkey-patches `plugin.time.time` so interval timing is deterministic.
#   3. For every logical timestamp tau, calls `_on_message` once per active
#      publisher (clamp + buffer) and then `_flush_and_release` (DP release).
# Two scenarios per dataset:
#   - pooled    : every publisher on a shared leaf topic; no scope walk, single
#                 StreamState.  Under the same seed the plugin's released
#                 values must match `run_dp_on_stream` byte-for-byte, which
#                 verifies the DP math on the plugin path.
#   - hierarchy : per-publisher leaves under the dataset's normative topic
#                 tree; P>1 forces Algorithm 1 walk-ups on every release.
#                 Verifies the scope walk fires, walked t_start is wall-clock,
#                 and no release happens with n_tau < P.

from plugin import PrivacyPlugin  # noqa: E402  (imported here to keep the
                                   # main pipeline import-light; plugin pulls in paho)


class _MockMQTTClient:
    """Stub for paho.mqtt.client.Client used by Experiment D."""

    def __init__(self):
        self.published: list[dict] = []
        self.subscribed: list[str] = []

    def connect(self, *_a, **_kw): pass
    def disconnect(self, *_a, **_kw): pass
    def loop_start(self, *_a, **_kw): pass
    def loop_stop(self, *_a, **_kw): pass

    def subscribe(self, topic):
        self.subscribed.append(topic)

    def publish(self, topic, payload):
        data = json.loads(payload) if isinstance(payload, (str, bytes)) else dict(payload)
        self.published.append({"topic": topic, **data})


class _FakeMsg:
    """Minimal paho MQTTMessage stand-in; plugin uses `.topic` and `.payload`."""

    __slots__ = ("topic", "payload")

    def __init__(self, topic: str, payload: bytes):
        self.topic = topic
        self.payload = payload


def _drive_plugin_scenario(
    per_pub: dict[str, list[float | None]],
    sensor: str,
    dataset_spec: dict,
    scenario: str,
    *,
    epsilon: float,
    w: int,
    P: int,
    delta_t: float,
    n_steps: int,
    seed: int,
    strategy: str = "p_gated_uniform",
    k_ext: int = 0,
    enable_hierarchy_walk: bool = True,
    epsilon_count: float = 0.0,
    max_publishers: int | None = None,
    force_P: int | None = None,
):
    """Feed a per-publisher trace into a PrivacyPlugin; return the plugin, the
    mock client, and the per-tau true-aggregate log for post-hoc comparison.
    """
    import plugin as _plug  # local for monkey-patching

    if scenario not in ("pooled", "hierarchy"):
        raise ValueError(f"unknown scenario: {scenario}")

    root = dataset_spec.get("topic_root") or f"{scenario}/ds"
    lo, hi = dataset_spec["static_clamps"][sensor]

    if scenario == "pooled":
        # All publishers emit on one shared leaf; plugin pools them there.
        def _leaf_of(_pub_id: str) -> str:
            return f"{root}/{sensor}"
        plugin_P = 1  # never triggers walk
    else:
        topic_of = dataset_spec["publisher_topic"]
        def _leaf_of(pub_id: str) -> str:
            return topic_of(pub_id, sensor)
        plugin_P = max(2, P)  # force walk-up (each leaf has n=1)

    if force_P is not None:
        plugin_P = int(force_P)

    # p_gated_uniform so the plugin's gate + Algorithm 1 walk logic actually
    # fires in the hierarchy scenario.  For pooled (plugin_P=1) the gate is a
    # no-op and the allocation dispatches to plain Uniform, so offline
    # comparison against strategy="uniform" still matches byte-for-byte.
    plugin = PrivacyPlugin(
        raw_prefix="raw",
        protected_prefix="protected",
        epsilon=epsilon,
        window_size=w,
        min_publishers=plugin_P,
        strategy=strategy,
        timestamp_interval=delta_t,
        k_ext=k_ext,
        epsilon_count=epsilon_count,
        max_publishers=max_publishers,
        enable_hierarchy_walk=enable_hierarchy_walk,
        sensor_bounds={sensor: (lo, hi)},
    )
    mock = _MockMQTTClient()
    plugin._client = mock

    # Deterministic clock.
    clock = [0.0]
    orig_time = _plug.time.time
    _plug.time.time = lambda: clock[0]

    tau_log: list[dict] = []
    try:
        np.random.seed(seed)
        publishers = list(per_pub.keys())
        T = min(n_steps, len(per_pub[publishers[0]]))

        for tau in range(T):
            clock[0] = tau * delta_t
            for pub_id in publishers:
                v = per_pub[pub_id][tau]
                if v is None:
                    continue
                topic = f"raw/{_leaf_of(pub_id)}"
                payload = json.dumps({"publisher_id": str(pub_id),
                                      "value": float(v)}).encode("utf-8")
                plugin._on_message(None, None, _FakeMsg(topic, payload))
            clock[0] = (tau + 1) * delta_t
            plugin._flush_and_release()

            # True clamped aggregate for this tau (comparison only).
            active = [min(max(per_pub[p][tau], lo), hi)
                      for p in publishers if per_pub[p][tau] is not None]
            tau_log.append({
                "tau": tau,
                "true_clamped_mean": float(np.mean(active)) if active else 0.0,
                "n_active": len(active),
                "t_start_expected": tau * delta_t,
            })
    finally:
        _plug.time.time = orig_time

    return {"plugin": plugin, "mock": mock, "tau_log": tau_log, "plugin_P": plugin_P}


def experiment_D_plugin_path(
    datasets,
    clamp_mode,
    output_dir,
    args,
    *,
    epsilon: float = 1.0,
    w: int = 8,
    P: int = 2,
    delta_t: float = 1.0,
    n_steps: int = 200,
    seed: int = 123,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exp D: plugin end-to-end vs offline DP engine on a stubbed MQTT path.

    Per dataset × scenario in {pooled, hierarchy} we record every release's
    (tau, t_start, scope, value, n_tau, walk_up, deferred), plus a per-run
    summary including:

      - ``release_rate`` / ``walkup_rate`` / ``deferred_count``
      - ``t_start_monotonic`` & spacing stats (paper Def 3.1 wall-clock)
      - ``p_gate_violations``: any released message with n_tau < plugin_P
        (must be 0 — otherwise the release gate is broken)
      - ``max_abs_diff_vs_offline`` for pooled only: the byte-level match
        between plugin release values and ``run_dp_on_stream`` under the
        same seed.  Near-zero means the plugin's DP math on the full path
        agrees with the standalone DP engine.
    """
    rows, summary_rows = [], []
    for ds_name in datasets:
        prepared = prepare_dataset(
            ds_name,
            clamp_mode=clamp_mode,
            eps_clip=args.eps_clip,
            seed=args.seed,
            max_rows=_dataset_max_rows(ds_name, args),
        )
        if prepared is None or not prepared.per_pubs:
            continue

        # Pick first sensor with >=2 publishers AND a configured static clamp.
        sensor = next(
            (s for s in prepared.spec["sensors"]
             if s in prepared.per_pubs
             and s in prepared.spec["static_clamps"]
             and len(prepared.per_pubs[s][0]) >= 2),
            None,
        )
        if sensor is None:
            logger.warning(f"[exp D] no suitable sensor in {ds_name}; skip")
            continue

        per_pub = prepared.per_pubs[sensor][0]

        for scenario in ("pooled", "hierarchy"):
            result = _drive_plugin_scenario(
                per_pub, sensor, prepared.spec, scenario,
                epsilon=epsilon, w=w, P=P,
                delta_t=delta_t, n_steps=n_steps, seed=seed,
            )
            plugin = result["plugin"]
            mock = result["mock"]
            tau_log = result["tau_log"]
            log = plugin.release_log

            released = [r for r in log if not r["deferred"] and r["released_value"] is not None]
            deferred = [r for r in log if r["deferred"]]
            walkups = [r for r in log if r["walk_up"]]
            p_violations = sum(1 for r in released if r["n_tau"] < result["plugin_P"])

            # Wall-clock t_start sanity: every published msg carries a t_start
            # that matches (tau-1)*delta_t for its interval and is monotone
            # non-decreasing across publications.
            published_t = [m["t_start"] for m in mock.published]
            monotonic = all(published_t[i+1] >= published_t[i]
                            for i in range(len(published_t) - 1))
            spacings = [published_t[i+1] - published_t[i]
                        for i in range(len(published_t) - 1)]

            # Offline comparison (pooled only — hierarchy has one stream per
            # publisher, which diverges from run_dp_on_stream's single-stream
            # semantics).
            max_diff_offline = float("nan")
            if scenario == "pooled" and released:
                # Reconstruct the pooled aggregate stream from tau_log and
                # replay offline with the SAME seed.  Only include taus up to
                # the simulation horizon; n_tau per tau is the count of active
                # publishers (post-clamp, as the plugin sees it).
                agg = [e["true_clamped_mean"] for e in tau_log]
                cnt = [e["n_active"] for e in tau_log]
                # Payload bound: static-clamp width for this sensor.
                lo, hi = prepared.spec["static_clamps"][sensor]
                B = float(hi - lo)
                offline = run_dp_on_stream(
                    agg, cnt, epsilon=epsilon, window_size=w,
                    min_publishers=result["plugin_P"],
                    payload_bound=B, strategy="uniform", seed=seed,
                )
                # Pair plugin releases with offline's noisy_values by tau.
                offline_by_tau = {i: v for i, v in enumerate(offline["noisy_values"])
                                  if v is not None}
                diffs = []
                for r in released:
                    # plugin's current_tau counts from 1; tau_log entries from 0.
                    i = r["tau"] - 1
                    if i in offline_by_tau and offline_by_tau[i] is not None:
                        diffs.append(abs(r["released_value"] - offline_by_tau[i]))
                max_diff_offline = max(diffs) if diffs else float("nan")

            # Per-release rows.
            for r in log:
                rows.append({
                    "dataset": ds_name,
                    "scenario": scenario,
                    "sensor": sensor,
                    "clamp_mode": clamp_mode,
                    "tau": r["tau"],
                    "t_start": r["t_start"],
                    "leaf_topic": r["leaf_topic"],
                    "release_scope": r["release_scope"],
                    "true_aggregate": r["true_aggregate"],
                    "released_value": r["released_value"],
                    "n_tau": r["n_tau"],
                    "walk_up": r["walk_up"],
                    "deferred": r["deferred"],
                })

            summary_rows.append({
                "dataset": ds_name,
                "scenario": scenario,
                "sensor": sensor,
                "clamp_mode": clamp_mode,
                "plugin_P": result["plugin_P"],
                "num_taus": len(log),
                "num_releases": len(released),
                "num_deferrals": len(deferred),
                "num_walkups": len(walkups),
                "release_rate": len(released) / max(1, len(log)),
                "walkup_rate": len(walkups) / max(1, len(log)),
                "t_start_monotonic": monotonic,
                "t_start_spacing_mean": float(np.mean(spacings)) if spacings else float("nan"),
                "t_start_spacing_std": float(np.std(spacings)) if spacings else float("nan"),
                "p_gate_violations": p_violations,
                "max_abs_diff_vs_offline": max_diff_offline,
                "num_published": len(mock.published),
            })

            logger.info(
                f"[exp D] {ds_name}/{sensor} scenario={scenario}: "
                f"releases={len(released)}/{len(log)} "
                f"walkups={len(walkups)} "
                f"p_violations={p_violations} "
                f"max_diff_offline={max_diff_offline}"
            )

    exp_dir = os.path.join(output_dir, "experiments", "D_plugin_path")
    os.makedirs(exp_dir, exist_ok=True)
    df_rel = pd.DataFrame(rows)
    df_sum = pd.DataFrame(summary_rows)
    df_rel.to_csv(os.path.join(exp_dir, "experiment_D_plugin_releases.csv"),
                  index=False)
    df_sum.to_csv(os.path.join(exp_dir, "experiment_D_plugin_summary.csv"),
                  index=False)
    logger.info(f"  Experiment D wrote {len(df_rel)} release rows, "
                f"{len(df_sum)} summary rows -> {exp_dir}")

    if not df_sum.empty:
        datasets_present = sorted(df_sum["dataset"].unique())
        scenarios = ["pooled", "hierarchy"]
        x = np.arange(len(datasets_present))
        bw = 0.38
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for i, (metric, ylabel) in enumerate([
            ("release_rate", "release rate"),
            ("walkup_rate", "walk-up rate (Algorithm 1 fires)"),
            ("max_abs_diff_vs_offline", "|plugin - offline|  (pooled only)"),
        ]):
            ax = axes[i]
            for j, scen in enumerate(scenarios):
                vals = []
                for ds in datasets_present:
                    sub = df_sum[(df_sum["dataset"] == ds) & (df_sum["scenario"] == scen)]
                    vals.append(float(sub[metric].iloc[0]) if not sub.empty else float("nan"))
                ax.bar(x + (j - 0.5) * bw, vals, bw, label=scen, alpha=0.85)
            ax.set(xticks=x, xlabel="dataset", ylabel=ylabel, title=ylabel)
            ax.set_xticklabels(datasets_present, rotation=30, fontsize=8)
            ax.grid(True, alpha=0.3, axis="y")
            ax.legend(fontsize=8)
        fig.suptitle(f"Experiment D: plugin end-to-end [clamp={clamp_mode}]",
                     fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, "experiment_D_plugin_path.png"), dpi=150)
        plt.close()

    return df_rel, df_sum


# ═════════════════════════════════════════════════════════════════════════
#  Experiment F (paper Sec. 7.8): incremental-module ablation
# ═════════════════════════════════════════════════════════════════════════
#
# The ablation adds mechanism modules one at a time and measures the utility
# impact of each, per the user-specified design:
#
#   M1  P-gated allocation only        gate on n_tau >= P_min, defer otherwise
#   M2  + subscription rewriting       + adaptive interval extension (Sec. 6.7):
#                                        hold an under-P scope open up to
#                                        K_ext * Delta_t to pool more publishers
#   M3  + walking up the tree          + Algorithm 1 hierarchy walk (Sec. 6.5):
#                                        rewrite scope to the nearest ancestor
#                                        whose range-compatible pool meets P
#
# It runs FULLY OFFLINE (no MQTT broker, no plugin object): each module is
# modelled directly as a transform of the per-publisher trace into the
# (aggregate, count) stream the subscriber's w-event stream would observe,
# then scored with ``run_dp_on_stream``.  Two subscription scopes are reported
# because the two rewriting modules dominate in different sparsity regimes:
#
#   * scope="leaf"   : the subscriber binds to ONE publisher's leaf.  n_tau in
#                      {0,1}, so M1/M2 starve (interval extension cannot add
#                      publishers to a single-pub leaf); only M3's walk-up,
#                      which pools range-compatible siblings, restores utility.
#                      Isolates the WALK-UP module (spatial sparsity).
#   * scope="pooled" : the subscriber binds to the whole-sensor scope and the
#                      gate is active (P_min > 1).  Temporal gaps make some
#                      Delta_t buckets fall below P; M2's interval extension
#                      merges consecutive buckets to recover them, while M3's
#                      walk-up adds nothing (the pooled scope is already the
#                      ancestor).  Isolates the INTERVAL-EXTENSION module
#                      (temporal sparsity).
#
# Together the two scopes give the complete incremental picture the paper's
# Sec. 6.6 utility analysis predicts (utility is dominated by the topic
# hierarchy and by range-compatible publisher availability).

def _adaptive_interval_rebuild(
    per_pub: dict[str, list[float | None]],
    P: int,
    k_ext: int,
    subset: list[str] | None = None,
    base_dt: int = 1,
) -> tuple[list[float], list[int]]:
    """Re-bucket a per-publisher trace with adaptive interval extension
    (paper Sec. 6.7).  A logical timestamp accumulates messages over one base
    Delta_t (``base_dt`` raw slots); if fewer than P distinct publishers
    contributed it merges the next Delta_t block (up to k_ext extensions)
    before emitting (aggregate, count).

    ``base_dt`` sets the base wall-clock interval in raw-slot units (the Delta_t
    hyperparameter); k_ext == 0 reduces to the fixed-Delta_t stream.  ``subset``
    restricts the pooled publisher set (used for the leaf/group scopes); None
    pools all.
    """
    pubs = subset if subset is not None else list(per_pub.keys())
    if not pubs:
        return [], []
    base_dt = max(1, int(base_dt))
    T = len(per_pub[pubs[0]])
    agg, cnt = [], []
    tau = 0
    while tau < T:
        vals: list[float] = []
        active: set[str] = set()
        end = tau
        ext = 0
        while True:
            block_end = min(end + base_dt, T)
            for slot in range(end, block_end):
                for p in pubs:
                    v = per_pub[p][slot]
                    if v is not None:
                        vals.append(v)
                        active.add(p)
            end = block_end
            if len(active) >= P or ext >= k_ext or end >= T:
                break
            ext += 1
        agg.append(float(np.mean(vals)) if vals else 0.0)
        cnt.append(len(active))
        tau = end
    return agg, cnt


# Worker context for the parallel grid search: precomputed (agg, cnt) streams
# keyed by (base_dt, k_ext, p_min), plus the fixed scoring params.
_WORKER_GRID_CTX: tuple | None = None


def _init_grid_worker(streams_by_key, payload_bound, epsilon, w, epsilon_count, seed):
    global _WORKER_GRID_CTX
    _WORKER_GRID_CTX = (streams_by_key, payload_bound, epsilon, w, epsilon_count, seed)


def _grid_eval_task(task):
    """Score one (strategy, base_dt, k_ext, p_min, p_max) grid cell."""
    strat, base_dt, k_ext, p_min, p_max = task
    streams_by_key, B, eps, w, eps_count, seed = _WORKER_GRID_CTX
    agg, cnt = streams_by_key[(base_dt, k_ext, p_min)]
    res = run_dp_on_stream(
        agg, cnt, epsilon=eps, window_size=w, min_publishers=p_min,
        payload_bound=B, strategy=strat, seed=seed,
        epsilon_count=eps_count, max_publishers=p_max,
    )
    m = res["metrics"]
    return {
        "strategy": strat, "epsilon": eps, "w": w,
        "P_min": p_min, "P_max": p_max, "delta_t": base_dt, "k_ext": k_ext,
        "epsilon_count": eps_count,
        "mae": m.get("mae"), "normalized_mae": m.get("normalized_mae"),
        "kl_divergence": m.get("kl_divergence"),
        "release_rate": m.get("release_rate"),
        "avg_n_tau": float(np.mean(cnt)) if cnt else float("nan"),
        "n_logical_timestamps": len(agg),
    }


def grid_search_hyperparameters(
    per_pub: dict[str, list[float | None]],
    payload_bound: float,
    dataset_name: str,
    sensor_name: str,
    output_dir: str,
    *,
    epsilon: float,
    w: int,
    strategies: list[str],
    p_min_grid: list[int],
    p_max_grid: list[int | None],
    dt_grid: list[int],
    k_ext_grid: list[int],
    epsilon_count: float = 0.0,
    seed: int = 77,
    workers: int = 1,
) -> pd.DataFrame:
    """Paper Sec. 7.5 utility-hyperparameter grid search.

    For a fixed (epsilon, dataset, sensor) it sweeps the full utility grid
    (P_min x P_max x Delta_t x K_ext) per budget strategy A and scores each
    configuration by MAE, then records the MAE-optimal configuration per
    strategy.  The (agg, cnt) stream depends only on (base_dt, k_ext, p_min),
    so it is rebuilt ONCE per such key (not per strategy/p_max) and the DP
    scoring is fanned out over ``workers`` processes.  Returns the full grid as
    a DataFrame and writes both the grid and the per-strategy optimum.
    """
    # 1) Precompute every distinct stream once (rebuild is the costly part).
    streams_by_key: dict = {}
    for base_dt in dt_grid:
        for k_ext in k_ext_grid:
            for p_min in p_min_grid:
                agg, cnt = _adaptive_interval_rebuild(
                    per_pub, p_min, k_ext, base_dt=base_dt)
                if len(agg) >= w + 2:
                    streams_by_key[(base_dt, k_ext, p_min)] = (agg, cnt)
    # 2) Build the task list (skip infeasible P_max < P_min and missing streams).
    tasks = [
        (strat, base_dt, k_ext, p_min, p_max)
        for strat in strategies
        for (base_dt, k_ext, p_min) in streams_by_key
        for p_max in p_max_grid
        if not (p_max is not None and p_max < p_min)
    ]
    # 3) Score every cell (parallel when workers > 1).
    if workers and workers > 1 and len(tasks) > 1:
        scored = _run_parallel_tasks(
            tasks, _grid_eval_task, workers=workers,
            initializer=_init_grid_worker,
            initargs=(streams_by_key, payload_bound, epsilon, w, epsilon_count, seed),
            progress_label=f"  [{dataset_name}/{sensor_name}] grid",
            progress_every=max(20, len(tasks) // 10),
        )
    else:
        _init_grid_worker(streams_by_key, payload_bound, epsilon, w, epsilon_count, seed)
        scored = [_grid_eval_task(t) for t in tasks]
    rows = [{"dataset": dataset_name, "sensor": sensor_name, **r} for r in scored]
    df = pd.DataFrame(rows)
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(
        os.path.join(output_dir, f"{dataset_name}_{sensor_name}_gridsearch.csv"),
        index=False,
    )
    # MAE-optimal configuration per strategy (the canonical config the paper
    # carries into the downstream experiments).
    best_rows = []
    if not df.empty:
        finite = df[df["mae"].notna() & np.isfinite(df["mae"])]
        for strat in strategies:
            sub = finite[finite["strategy"] == strat]
            if sub.empty:
                continue
            best_rows.append(sub.loc[sub["mae"].idxmin()].to_dict())
    best_df = pd.DataFrame(best_rows)
    best_df.to_csv(
        os.path.join(output_dir,
                     f"{dataset_name}_{sensor_name}_gridsearch_best.csv"),
        index=False,
    )
    logger.info(
        f"  [{dataset_name}/{sensor_name}] grid search: {len(df)} configs, "
        f"{len(best_df)} per-strategy optima -> {output_dir}")
    return df


# ═════════════════════════════════════════════════════════════════════════
#  Canonical config: the grid search fixes the params for every later experiment
# ═════════════════════════════════════════════════════════════════════════
#
# Paper Sec. 7.5: "We utilize these optimized values as the canonical fixed
# values for the following experiments."  The grid search writes one
# grid_canonical.json at the run's output root, keyed by
# (dataset, clamp_mode, strategy, epsilon) -> {P_min, P_max, delta_t, k_ext}.
# Downstream experiments resolve their (P_min, P_max, K_ext, ...) from it when a
# --use-grid-config path is supplied, falling back to CLI defaults otherwise.

def _grid_canonical_path(output_dir: str) -> str:
    return os.path.join(output_dir, "grid_canonical.json")


def _write_grid_canonical(output_dir: str, best_records: list[dict]) -> str:
    """Persist the per-(dataset,clamp,strategy,epsilon) MAE-optimal configs."""
    path = _grid_canonical_path(output_dir)
    payload = []
    for r in best_records:
        payload.append({
            "dataset": r.get("dataset"),
            "clamp_mode": r.get("clamp_mode"),
            "strategy": r.get("strategy"),
            "epsilon": float(r.get("epsilon")),
            "P_min": int(r.get("P_min")) if r.get("P_min") is not None else None,
            "P_max": (int(r["P_max"]) if r.get("P_max") is not None
                      and not (isinstance(r.get("P_max"), float) and np.isnan(r["P_max"]))
                      else None),
            "delta_t": int(r.get("delta_t", 1)),
            "k_ext": int(r.get("k_ext", 0)),
            "mae": float(r.get("mae")) if r.get("mae") is not None else None,
        })
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    logger.info(f"  wrote canonical grid config ({len(payload)} entries) -> {path}")
    return path


def _load_grid_config(path: str | None) -> dict | None:
    """Load grid_canonical.json into a lookup dict keyed by
    (dataset, clamp_mode, strategy, round(epsilon, 4))."""
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            records = json.load(fh)
    except Exception as exc:  # pragma: no cover
        logger.warning(f"could not read grid config {path}: {exc}")
        return None
    cfg = {}
    for r in records:
        key = (r.get("dataset"), r.get("clamp_mode"), r.get("strategy"),
               round(float(r.get("epsilon", 0.0)), 4))
        cfg[key] = r
    logger.info(f"loaded canonical grid config ({len(cfg)} entries) from {path}")
    return cfg


def _resolve_params(grid_config: dict | None, dataset: str, clamp_mode: str,
                    strategy: str, epsilon: float, defaults: dict) -> dict:
    """Return {P_min, P_max, k_ext, delta_t} from the canonical grid config for
    this (dataset, clamp, strategy, epsilon), else the supplied defaults."""
    out = dict(defaults)
    if grid_config is None:
        return out
    rec = grid_config.get(
        (dataset, clamp_mode, strategy, round(float(epsilon), 4)))
    if rec is None:
        # Strategy-agnostic fallback: any entry for this (dataset, clamp, eps).
        for (d, c, _s, e), r in grid_config.items():
            if d == dataset and c == clamp_mode and e == round(float(epsilon), 4):
                rec = r
                break
    if rec is None:
        return out
    for k_cfg, k_out in (("P_min", "P_min"), ("P_max", "P_max"),
                         ("k_ext", "k_ext"), ("delta_t", "delta_t")):
        if rec.get(k_cfg) is not None:
            out[k_out] = rec[k_cfg]
    return out


def _ablation_module_streams(
    per_pub: dict[str, list[float | None]],
    P: int,
    k_ext: int,
    scope: str,
) -> dict[str, tuple[list[float], list[int]]]:
    """Build the (aggregate, count) stream each cumulative module produces for
    one subscription scope.  Returns {module_name -> (agg, cnt)}.
    """
    pubs = list(per_pub.keys())
    if scope == "leaf":
        # Subscriber binds to the single busiest publisher's leaf.
        leaf_pub = max(pubs, key=lambda p: sum(v is not None for v in per_pub[p]))
        leaf_subset = [leaf_pub]
        # M1: gate only, native Delta_t, single-pub leaf (n in {0,1}).
        m1 = _adaptive_interval_rebuild(per_pub, P, 0, subset=leaf_subset)
        # M2: + interval extension on that leaf (still a single publisher).
        m2 = _adaptive_interval_rebuild(per_pub, P, k_ext, subset=leaf_subset)
        # M3: + walk-up pools the range-compatible siblings (whole sensor).
        m3 = _adaptive_interval_rebuild(per_pub, P, k_ext, subset=pubs)
        return {"M1_pgate": m1, "M2_interval_ext": m2, "M3_walk_up": m3}
    # scope == "pooled": gate active on the whole-sensor scope.
    m1 = _adaptive_interval_rebuild(per_pub, P, 0, subset=pubs)
    m2 = _adaptive_interval_rebuild(per_pub, P, k_ext, subset=pubs)
    # Walk-up has no ancestor above the pooled root, so M3 == M2 here.
    m3 = m2
    return {"M1_pgate": m1, "M2_interval_ext": m2, "M3_walk_up": m3}


def ablation_experiment(
    datasets,
    clamp_mode,
    output_dir,
    args,
    *,
    epsilon: float = 1.0,
    w: int = 8,
    P: int = 3,
    k_ext: int = 3,
    epsilon_count: float = 0.0,
    strategy: str = "p_gated_uniform",
    seed: int = 77,
    grid_config: dict | None = None,
) -> pd.DataFrame:
    """Paper Sec. 7.8 incremental-module ablation (fully offline).

    For each dataset (first suitable sensor) and each subscription scope in
    {leaf, pooled}, scores the three cumulative module sets M1/M2/M3 and writes
    NMAE / KL / release-rate / avg n_tau so the utility delta of each added
    module is read directly off the CSV.  When ``grid_config`` is supplied the
    per-dataset (P_min, P_max, K_ext) come from the Sec. 7.5 grid optimum.
    """
    MODULES = ["M1_pgate", "M2_interval_ext", "M3_walk_up"]
    rows = []
    for ds_name in datasets:
        prepared = prepare_dataset(
            ds_name, clamp_mode=clamp_mode, eps_clip=args.eps_clip,
            seed=args.seed, max_rows=_dataset_max_rows(ds_name, args),
        )
        if prepared is None or not prepared.per_pubs:
            continue
        sensor = next(
            (s for s in prepared.spec["sensors"]
             if s in prepared.per_pubs
             and s in prepared.spec["static_clamps"]
             and len(prepared.per_pubs[s][0]) >= 2),
            None,
        )
        if sensor is None:
            logger.warning(f"[exp F/ablation] no suitable sensor in {ds_name}; skip")
            continue
        prm = _resolve_params(grid_config, ds_name, clamp_mode, strategy, epsilon,
                              {"P_min": P, "P_max": None, "k_ext": k_ext})
        P_ds, P_max_ds, k_ext_ds = prm["P_min"], prm["P_max"], prm["k_ext"]
        per_pub, B = prepared.per_pubs[sensor]
        for scope in ("leaf", "pooled"):
            streams = _ablation_module_streams(per_pub, P_ds, k_ext_ds, scope)
            for m_idx, module in enumerate(MODULES, start=1):
                agg, cnt = streams[module]
                if len(agg) < w + 2:
                    res_metrics = {"normalized_mae": float("nan"),
                                   "kl_divergence": float("nan"),
                                   "release_rate": float("nan"), "mae": float("nan")}
                else:
                    res = run_dp_on_stream(
                        agg, cnt, epsilon=epsilon, window_size=w,
                        min_publishers=P_ds, payload_bound=B, strategy=strategy,
                        seed=seed, epsilon_count=epsilon_count,
                        max_publishers=P_max_ds,
                    )
                    res_metrics = res["metrics"]
                rows.append({
                    "dataset": ds_name, "sensor": sensor, "clamp_mode": clamp_mode,
                    "scope": scope, "module": module, "module_idx": m_idx,
                    "epsilon": epsilon, "w": w, "P": P_ds, "P_max": P_max_ds,
                    "k_ext": k_ext_ds,
                    "epsilon_count": epsilon_count, "strategy": strategy,
                    "payload_bound": B,
                    "normalized_mae": res_metrics.get("normalized_mae"),
                    "mae": res_metrics.get("mae"),
                    "kl_divergence": res_metrics.get("kl_divergence"),
                    "release_rate": res_metrics.get("release_rate"),
                    "avg_n_tau": float(np.mean(cnt)) if cnt else float("nan"),
                    "n_logical_timestamps": len(agg),
                })
            # Per-scope module deltas (utility gained by adding each module).
            scope_rows = [r for r in rows if r["dataset"] == ds_name
                          and r["scope"] == scope]
            base_rr = scope_rows[0]["release_rate"]
            for r in scope_rows:
                rr = r["release_rate"]
                r["release_rate_gain_vs_M1"] = (
                    (rr - base_rr) if rr is not None and base_rr is not None
                    and np.isfinite(rr) and np.isfinite(base_rr) else float("nan")
                )
    exp_dir = os.path.join(output_dir, "experiments", "F_ablation")
    os.makedirs(exp_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(exp_dir, "experiment_F_ablation.csv"), index=False)
    logger.info(f"  Experiment F (ablation) wrote {len(df)} rows -> {exp_dir}")

    if not df.empty and getattr(args, "generate_plots", False):
        _plot_single_axis_experiment(
            df[df["scope"] == "leaf"], "module_idx", "module (cumulative)",
            os.path.join(exp_dir, "experiment_F_ablation_leaf.png"),
            f"Ablation (leaf scope) [clamp={clamp_mode}]",
        )
    return df


# ═════════════════════════════════════════════════════════════════════════
#  Local Differential Privacy (the P_min = 1 regime)
# ═════════════════════════════════════════════════════════════════════════
#
# Paper Sec. 1 (Extreme 2) & Sec. 6.6: when P_min = 1 the mechanism dissolves
# to LOCAL differential privacy.  Each publisher is its own stream element with
# a single contributor (n_tau = 1), so the mean sensitivity is Delta_f = R and
# the Laplace scale under Uniform allocation is lambda = R * w / eps -- noise of
# magnitude proportional to the full payload range R, injected per publisher
# BEFORE any pooling (input / local privacy) rather than on a pooled aggregate
# (output privacy).  This is the canonical LDP baseline used by Figure 1's
# rightmost (per-publisher) point and by the Sec. 7.9 overhead comparison.

def run_ldp_on_per_pub(
    per_pub: dict[str, list[float | None]],
    payload_bound: float,
    epsilon: float,
    window_size: int,
    seed: int = 0,
) -> dict:
    """Local DP baseline (P_min = 1): perturb each publisher's clamped payload
    independently with Laplace(R * w / eps) (sensitivity R, Uniform w-event
    share eps/w), then form the per-tau mean of the noisy inputs.

    Returns the same metric dict shape as ``run_dp_on_stream`` so callers can
    compare LDP to the pooled output-DP mechanism directly.  The per-publisher
    perturbation is the defining feature of LOCAL DP: noise is added to inputs,
    so the released mean's error does not shrink as 1/n the way the pooled
    output-DP release does.
    """
    rng = np.random.default_rng(seed)
    pubs = list(per_pub.keys())
    if not pubs:
        return {"metrics": {"mae": float("nan"), "normalized_mae": float("nan"),
                            "kl_divergence": float("nan"), "release_rate": float("nan"),
                            "attribution_advantage": 1.0}}
    T = len(per_pub[pubs[0]])
    scale = payload_bound * window_size / epsilon if epsilon > 0 else float("inf")
    true_means, noisy_means = [], []
    for tau in range(T):
        true_vals, noisy_vals = [], []
        for p in pubs:
            v = per_pub[p][tau]
            if v is None:
                continue
            true_vals.append(v)
            # Local perturbation: each publisher noises its OWN value.
            noisy_vals.append(v + float(rng.laplace(loc=0.0, scale=scale)))
        if true_vals:
            true_means.append(float(np.mean(true_vals)))
            noisy_means.append(float(np.mean(noisy_vals)))
    metrics = compute_utility_metrics(true_means, noisy_means)
    metrics["normalized_mae"] = (
        metrics["mae"] / payload_bound if payload_bound > 0 else float("nan")
    )
    metrics["kl_divergence"] = compute_kl_divergence(true_means, noisy_means)
    metrics["release_rate"] = 1.0  # LDP releases every populated timestamp
    # LDP exposes every contributor (n_tau = 1 stream elements): attribution
    # advantage is 1 (the subscriber sees a single publisher's value per leaf).
    metrics["attribution_advantage"] = 1.0
    metrics["ldp_noise_scale"] = scale
    return {"metrics": metrics, "true_values": true_means, "noisy_values": noisy_means}


# ═════════════════════════════════════════════════════════════════════════
#  Experiment G (paper Sec. 7.9): overhead / privacy-utility comparison
# ═════════════════════════════════════════════════════════════════════════
#
# Compares our clamped w-event DP with P-allocation against the baselines the
# paper contrasts it with (Sec. 7.9 + the two extremes of Sec. 1):
#
#   classic         no privacy: the broker releases the true aggregate.  Sets
#                   the utility ceiling (NMAE = 0, KL = 0) and the throughput
#                   reference (no Laplace draws, no eps_count).
#   ldp             P_min = 1 LOCAL DP: per-publisher input perturbation,
#                   lambda = R*w/eps (run_ldp_on_per_pub).  Strongest privacy,
#                   worst utility (Extreme 2).
#   per_type_wevent one stream per sensor TYPE (Extreme 1.1): output-DP on the
#                   per-type mean, Delta_f = R/n_type, topic semantics collapsed.
#   ours            clamped w-event DP with P-allocation at the topic scope
#                   (p_gated / n_weighted), Delta_f = R/n_tau.
#
# Utility/privacy (NMAE, KL, release rate, attribution advantage) is fully
# offline-measurable and reported here.  A compute-overhead proxy (mean
# wall-clock per released element and the eps_count surcharge) is also
# recorded; true broker throughput/latency under concurrency is measured by
# the live-broker Experiment E, which this experiment cross-references.

def overhead_experiment(
    datasets,
    clamp_mode,
    output_dir,
    args,
    *,
    epsilon: float = 1.0,
    w: int = 8,
    P: int = 3,
    epsilon_count: float = 0.0,
    our_strategy: str = "p_gated_uniform",
    seed: int = 77,
    grid_config: dict | None = None,
) -> pd.DataFrame:
    """Paper Sec. 7.9 overhead / privacy-utility comparison (offline).

    When ``grid_config`` is supplied, the 'ours' approach uses the Sec. 7.5
    grid-optimal (P_min, P_max) per dataset.
    """
    rows = []
    for ds_name in datasets:
        prepared = prepare_dataset(
            ds_name, clamp_mode=clamp_mode, eps_clip=args.eps_clip,
            seed=args.seed, max_rows=_dataset_max_rows(ds_name, args),
        )
        if prepared is None or not prepared.per_pubs:
            continue
        sensor = next(
            (s for s in prepared.spec["sensors"]
             if s in prepared.per_pubs and s in prepared.streams
             and s in prepared.spec["static_clamps"]),
            None,
        )
        if sensor is None:
            continue
        prm = _resolve_params(grid_config, ds_name, clamp_mode, our_strategy,
                              epsilon, {"P_min": P, "P_max": None})
        P_ds, P_max_ds = prm["P_min"], prm["P_max"]
        per_pub, B = prepared.per_pubs[sensor]
        agg, cnt, _B = prepared.streams[sensor]

        def _emit(approach, metrics, compute_s, extra=None):
            row = {
                "dataset": ds_name, "sensor": sensor, "clamp_mode": clamp_mode,
                "approach": approach, "epsilon": epsilon, "w": w, "P": P,
                "payload_bound": B,
                "normalized_mae": metrics.get("normalized_mae"),
                "mae": metrics.get("mae"),
                "kl_divergence": metrics.get("kl_divergence"),
                "release_rate": metrics.get("release_rate"),
                "attribution_advantage": metrics.get("attribution_advantage"),
                "eps_count_spent": metrics.get("eps_count_spent", 0.0),
                "compute_ms_per_element": 1000.0 * compute_s / max(1, len(agg)),
            }
            if extra:
                row.update(extra)
            rows.append(row)

        # classic: no privacy (true aggregate released verbatim).
        _emit("classic",
              {"normalized_mae": 0.0, "mae": 0.0, "kl_divergence": 0.0,
               "release_rate": 1.0, "attribution_advantage": float("nan")},
              0.0)

        # ldp: P_min = 1 local DP (per-publisher input perturbation).
        t0 = time.perf_counter()
        ldp = run_ldp_on_per_pub(per_pub, B, epsilon, w, seed=seed)
        _emit("ldp", ldp["metrics"], time.perf_counter() - t0,
              {"noise_scale": ldp["metrics"].get("ldp_noise_scale")})

        # per_type_wevent: one stream per sensor type (Extreme 1.1) -- here the
        # single sensor's pooled mean over ALL its publishers, output-DP with
        # Uniform allocation, no P-gate (topic scope collapsed to the type).
        t0 = time.perf_counter()
        pt = run_dp_on_stream(agg, cnt, epsilon=epsilon, window_size=w,
                              min_publishers=1, payload_bound=B,
                              strategy="uniform", seed=seed)
        _emit("per_type_wevent", pt["metrics"], time.perf_counter() - t0)

        # ours: clamped w-event DP with P-allocation (P-gate + eps_count),
        # at the Sec. 7.5 grid-optimal (P_min, P_max).
        t0 = time.perf_counter()
        ours = run_dp_on_stream(agg, cnt, epsilon=epsilon, window_size=w,
                                min_publishers=P_ds, payload_bound=B,
                                strategy=our_strategy, seed=seed,
                                epsilon_count=epsilon_count,
                                max_publishers=P_max_ds)
        _emit("ours", ours["metrics"], time.perf_counter() - t0,
              {"strategy": our_strategy, "P_min": P_ds, "P_max": P_max_ds})

    exp_dir = os.path.join(output_dir, "experiments", "G_overhead")
    os.makedirs(exp_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(exp_dir, "experiment_G_overhead.csv"), index=False)
    logger.info(f"  Experiment G (overhead) wrote {len(df)} rows -> {exp_dir}")
    return df


# ═════════════════════════════════════════════════════════════════════════
#  Experiment H (paper Sec. 7.11): average-case utility
# ═════════════════════════════════════════════════════════════════════════
#
# Paper Sec. 6.6 / 7.11: average-case utility is governed by (i) the proportion
# of publishers range-compatible with a subscription, |P_R| / |P| in [0, 1],
# and (ii) the topic-hierarchy depth h (how many levels a subscription may have
# to walk up).  As the range-compatible fraction -> 1 utility improves; deeper
# trees cost more eps_count on the walk.  This experiment measures both
# structural quantities per dataset and pairs them with the realized utility.

def _topic_depth(spec: dict, per_pubs: dict) -> int:
    """Number of levels in the dataset's topic tree (max '/'-segment count over
    a sample of publisher topics)."""
    topic_of = spec.get("publisher_topic")
    if topic_of is None:
        return 0
    depths = []
    for sensor, (pp, _B) in per_pubs.items():
        for pub_id in list(pp.keys())[:5]:
            try:
                depths.append(len(topic_of(pub_id, sensor).split("/")))
            except Exception:
                continue
        break
    return max(depths) if depths else 0


def _range_compatible_fraction(spec: dict, sensors: list[str], ref_sensor: str) -> float:
    """Fraction of the dataset's sensor types whose static clamp is
    range-compatible with the reference sensor (hull width <= R), a proxy for
    |P_R| / |P| at the cross-type pooling scope (Definition 6.3)."""
    clamps = spec.get("static_clamps", {})
    if ref_sensor not in clamps:
        return float("nan")
    a_ref, b_ref = clamps[ref_sensor]
    R = b_ref - a_ref
    compatible, total = 0, 0
    for s in sensors:
        if s not in clamps:
            continue
        total += 1
        a, b = clamps[s]
        hull = max(b_ref, b) - min(a_ref, a)
        if hull <= R + 1e-9:
            compatible += 1
    return compatible / total if total else float("nan")


def average_case_utility_experiment(
    datasets,
    clamp_mode,
    output_dir,
    args,
    *,
    epsilon: float = 1.0,
    w: int = 8,
    P: int = 3,
    epsilon_count: float = 0.5,
    strategy: str = "p_gated_uniform",
    seed: int = 77,
    grid_config: dict | None = None,
) -> pd.DataFrame:
    """Paper Sec. 7.11 average-case utility: relate range-compatible fraction
    and topic-hierarchy depth h to realized utility, per dataset.  Uses the
    Sec. 7.5 grid-optimal (P_min, P_max) per dataset when ``grid_config`` is
    supplied."""
    rows = []
    for ds_name in datasets:
        prepared = prepare_dataset(
            ds_name, clamp_mode=clamp_mode, eps_clip=args.eps_clip,
            seed=args.seed, max_rows=_dataset_max_rows(ds_name, args),
        )
        if prepared is None or not prepared.per_pubs:
            continue
        sensors = [s for s in prepared.spec["sensors"] if s in prepared.streams]
        if not sensors:
            continue
        ref_sensor = sensors[0]
        depth_h = _topic_depth(prepared.spec, prepared.per_pubs)
        frac = _range_compatible_fraction(prepared.spec, sensors, ref_sensor)
        prm = _resolve_params(grid_config, ds_name, clamp_mode, strategy, epsilon,
                              {"P_min": P, "P_max": None})
        P_ds, P_max_ds = prm["P_min"], prm["P_max"]
        agg, cnt, B = prepared.streams[ref_sensor]
        res = run_dp_on_stream(
            agg, cnt, epsilon=epsilon, window_size=w, min_publishers=P_ds,
            payload_bound=B, strategy=strategy, seed=seed,
            epsilon_count=epsilon_count, max_publishers=P_max_ds,
        )
        m = res["metrics"]
        rows.append({
            "dataset": ds_name, "ref_sensor": ref_sensor, "clamp_mode": clamp_mode,
            "range_compatible_fraction": frac,
            "topic_hierarchy_depth_h": depth_h,
            "n_sensor_types": len(sensors),
            "avg_n_tau": float(np.mean(cnt)) if cnt else float("nan"),
            "normalized_mae": m.get("normalized_mae"),
            "kl_divergence": m.get("kl_divergence"),
            "release_rate": m.get("release_rate"),
            "eps_count_spent": m.get("eps_count_spent", 0.0),
            "dp_count_releases": m.get("dp_count_releases", 0),
            "epsilon": epsilon, "w": w, "P": P_ds, "P_max": P_max_ds,
            "epsilon_count": epsilon_count,
        })
    exp_dir = os.path.join(output_dir, "experiments", "H_average_case")
    os.makedirs(exp_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(exp_dir, "experiment_H_average_case.csv"), index=False)
    logger.info(f"  Experiment H (average-case utility) wrote {len(df)} rows -> {exp_dir}")
    return df


# ═════════════════════════════════════════════════════════════════════════
#  Experiment E: live MQTT broker sanity subset
# ═════════════════════════════════════════════════════════════════════════
#
# Unlike Experiment D (stubbed MQTT + deterministic clock), Experiment E runs
# every config end-to-end through a real MQTT broker (default localhost:1883).
# For each config:
#
#   1. Spin up a PrivacyPlugin connected to the broker with a unique
#      paho client_id and unique raw/protected topic prefixes.
#   2. Spin up a paho subscriber on the protected prefix that logs every
#      delivered release.
#   3. Spin up a paho publisher that emits each tau's per-publisher readings
#      on the raw prefix at a wall-clock cadence of `live_dt` seconds.
#   4. Wait for the plugin's timer loop to drain the final window, then stop.
#   5. Metrics are computed from plugin.release_log (the canonical in-process
#      record, same as Experiment D).  The subscriber count is checked against
#      the plugin's non-deferred release count to validate the broker path.
#
# Safe parallel execution: every config generates a UUID and uses it in both
# its topic prefixes and its paho client_ids, so N configs can share a single
# broker without cross-talk.  NMAE / KL / release_rate / attribution_advantage
# are computed per config and compared to an offline `run_dp_on_stream` call
# under the same seed; the delta is reported as a sanity metric.

def _live_raw_topic(prefix: str, leaf: str) -> str:
    return f"{prefix}/{leaf}"


class _EmbeddedBroker:
    """Embedded amqtt (pure-Python) MQTT broker, started on a background thread.

    Used by ``--experiment E`` when no broker is already listening on the
    requested (host, port).  Runs an asyncio loop in a daemon thread hosting
    the ``amqtt.broker.Broker`` instance; ``stop()`` cleanly shuts both down.
    """

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self._loop: asyncio.AbstractEventLoop | None = None
        self._broker = None
        self._thread: threading.Thread | None = None
        self._started = threading.Event()
        self._start_error: BaseException | None = None

    def start(self, timeout: float = 10.0) -> None:
        import asyncio as _asyncio
        try:
            from amqtt.broker import Broker  # noqa: F401 (import-check only)
        except ImportError as exc:
            raise RuntimeError(
                "amqtt is not installed; either install it "
                "(`pip install amqtt`) or start an external MQTT broker "
                "(e.g., mosquitto) before running --experiment E."
            ) from exc

        def _runner():
            try:
                loop = _asyncio.new_event_loop()
                _asyncio.set_event_loop(loop)
                self._loop = loop
                from amqtt.broker import Broker
                config = {
                    "listeners": {
                        "default": {
                            "type": "tcp",
                            "bind": f"{self.host}:{self.port}",
                        }
                    },
                    "auth": {"allow-anonymous": True},
                    # Disable the $SYS plugin's periodic reporting; it tries
                    # to compare sys_interval (None) to 0 and spams warnings.
                    "sys_interval": 0,
                }

                async def _bring_up():
                    # Broker.__init__ and start() both need a running loop.
                    self._broker = Broker(config=config)
                    await self._broker.start()

                loop.run_until_complete(_bring_up())
                self._started.set()
                loop.run_forever()
            except BaseException as exc:  # pragma: no cover
                self._start_error = exc
                self._started.set()

        self._thread = threading.Thread(
            target=_runner, name="pubsubpriv-embedded-broker", daemon=True,
        )
        self._thread.start()
        if not self._started.wait(timeout=timeout):
            raise RuntimeError(
                f"Embedded broker did not start within {timeout}s"
            )
        if self._start_error is not None:
            raise self._start_error
        # Give paho clients a beat to be able to connect reliably.
        time.sleep(0.3)
        logger.info(f"[exp E] embedded amqtt broker listening at {self.host}:{self.port}")

    def stop(self, timeout: float = 5.0) -> None:
        if self._loop is None:
            return
        loop = self._loop

        async def _shutdown():
            try:
                if self._broker is not None:
                    await self._broker.shutdown()
            except Exception:
                pass

        try:
            fut = asyncio.run_coroutine_threadsafe(_shutdown(), loop)
            fut.result(timeout=timeout)
        except Exception:
            pass
        loop.call_soon_threadsafe(loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._loop = None
        logger.info("[exp E] embedded broker stopped")


def _broker_is_listening(host: str, port: int, timeout: float = 0.75) -> bool:
    import socket
    s = socket.socket()
    s.settimeout(timeout)
    try:
        s.connect((host, port))
        return True
    except Exception:
        return False
    finally:
        try:
            s.close()
        except Exception:
            pass


def _ensure_broker(host: str, port: int, auto_start: bool) -> "_EmbeddedBroker | None":
    """If a broker is already listening, return None.  Otherwise, start an
    embedded amqtt broker on (host, port) and return a handle whose ``stop()``
    tears it down when the experiment finishes.  Raises if ``auto_start`` is
    false and no external broker is found.
    """
    if _broker_is_listening(host, port):
        logger.info(f"[exp E] using existing broker at {host}:{port}")
        return None
    if not auto_start:
        raise RuntimeError(
            f"No MQTT broker listening at {host}:{port} and --no-auto-broker "
            f"was specified.  Start mosquitto (or similar) first."
        )
    logger.info(
        f"[exp E] no broker at {host}:{port}; starting embedded amqtt broker"
    )
    eb = _EmbeddedBroker(host, port)
    eb.start()
    return eb


class _LiveSubscriber:
    """paho subscriber that captures delivered releases on the protected prefix."""

    def __init__(self, broker_host, broker_port, protected_prefix, client_id):
        import paho.mqtt.client as mqtt  # local import: paho may not be installed
        self._mqtt = mqtt
        self.prefix = protected_prefix
        self.received: list[dict] = []
        self._ready = threading.Event()
        self._client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=client_id,
        )

        def _on_connect(client, *_a, **_kw):
            client.subscribe(f"{self.prefix}/#")
            self._ready.set()

        def _on_message(_c, _u, msg):
            try:
                body = json.loads(msg.payload.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return
            self.received.append({
                "topic": msg.topic,
                "t_start": float(body.get("t_start", 0.0)),
                "value": float(body.get("value", 0.0)),
                "wall_clock_recv": time.time(),
            })

        self._client.on_connect = _on_connect
        self._client.on_message = _on_message
        self._broker_host = broker_host
        self._broker_port = broker_port

    def start(self, connect_timeout: float = 5.0):
        self._client.connect(self._broker_host, self._broker_port)
        self._client.loop_start()
        if not self._ready.wait(timeout=connect_timeout):
            raise RuntimeError("Subscriber did not connect within timeout")

    def stop(self):
        self._client.loop_stop()
        try:
            self._client.disconnect()
        except Exception:
            pass


class _LivePublisher:
    """paho publisher that emits each tau's per-publisher readings at a wall-clock cadence."""

    def __init__(self, broker_host, broker_port, raw_prefix, client_id):
        import paho.mqtt.client as mqtt
        self.prefix = raw_prefix
        self._client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=client_id,
        )
        self._ready = threading.Event()
        self._client.on_connect = lambda *_a, **_kw: self._ready.set()
        self._broker_host = broker_host
        self._broker_port = broker_port
        self.num_published = 0

    def start(self, connect_timeout: float = 5.0):
        self._client.connect(self._broker_host, self._broker_port)
        self._client.loop_start()
        if not self._ready.wait(timeout=connect_timeout):
            raise RuntimeError("Publisher did not connect within timeout")

    def publish_one(self, leaf: str, publisher_id: str, value: float):
        topic = _live_raw_topic(self.prefix, leaf)
        payload = json.dumps({"publisher_id": str(publisher_id),
                              "value": float(value)}).encode("utf-8")
        # QoS 1 so the plugin reliably sees every publish even under load.
        info = self._client.publish(topic, payload, qos=1)
        info.wait_for_publish(timeout=2.0)
        self.num_published += 1

    def stop(self):
        self._client.loop_stop()
        try:
            self._client.disconnect()
        except Exception:
            pass


def _drive_live_config(
    per_pub: dict[str, list[float | None]],
    sensor: str,
    dataset_spec: dict,
    *,
    strategy: str,
    epsilon: float,
    w: int,
    P: int,
    broker_host: str,
    broker_port: int,
    live_dt: float,
    n_steps: int,
    seed: int,
    drain_ticks: int = 3,
    scenario: str = "pooled",
    n_subscribers: int = 1,
) -> dict:
    """Run one (strategy, epsilon, w, P, scenario) config end-to-end
    through a live broker.

    Scenarios:

      * ``pooled``    — every publisher emits on the SAME leaf topic so the
                        plugin aggregates all n_tau active publishers into a
                        single release per tau.  Matches the offline
                        ``run_dp_on_stream`` reference exactly (single stream
                        over the mean of all active publishers).  Exercises
                        the many-publishers path under a shared-leaf
                        subscription.
      * ``hierarchy`` — each publisher emits on its own leaf under the
                        dataset's normative MQTT tree (so each leaf's
                        n_tau = 1).  P >= 2 therefore forces Algorithm 1's
                        clamp-compatible walk-up on every release, and the
                        plugin's ``walk_up`` flag should be True for every
                        released record.

    Every component (plugin, publishers, subscribers) gets unique paho
    client_ids and a unique topic-prefix UUID so parallel configs can share
    the broker safely.  ``n_subscribers`` concurrent ``_LiveSubscriber``
    clients attach to the protected prefix to verify the broker correctly
    fans each release out to every subscribing client — each subscriber's
    received-count is reported alongside the plugin's published-count, so
    the caller can verify subscriber-fan-out integrity.

    Returns the plugin's release log, every subscriber's received releases
    (aggregated), per-subscriber counts, tau-truth log, and timing metadata.
    """
    import uuid as _uuid

    run_id = _uuid.uuid4().hex[:8]
    raw_prefix = f"pubsubpriv/{run_id}/raw"
    protected_prefix = f"pubsubpriv/{run_id}/protected"

    lo, hi = dataset_spec["static_clamps"][sensor]
    if scenario == "pooled":
        pooled_leaf = f"{sensor}"

        def _leaf_of(_pub_id: str) -> str:
            return pooled_leaf
        plugin_P = int(P)
    elif scenario == "hierarchy":
        # Each publisher gets its own leaf under the dataset's normative
        # topic tree.  With P > 1 the plugin's release gate will fire
        # on every leaf (n_tau=1 < P) and Algorithm 1's clamp-compatible
        # walk-up must climb to an ancestor before any release can emit.
        topic_of = dataset_spec["publisher_topic"]

        def _leaf_of(pub_id: str) -> str:
            return topic_of(pub_id, sensor)
        plugin_P = max(2, int(P))  # force walk-up (every leaf has n=1)
    else:
        raise ValueError(f"unknown scenario: {scenario}")

    plugin = PrivacyPlugin(
        broker_host=broker_host,
        broker_port=broker_port,
        raw_prefix=raw_prefix,
        protected_prefix=protected_prefix,
        epsilon=epsilon,
        window_size=w,
        min_publishers=plugin_P,
        strategy=strategy,
        timestamp_interval=live_dt,
        k_ext=0,
        sensor_bounds={sensor: (lo, hi)},
        client_id=f"plugin-{run_id}",
    )
    n_subscribers = max(1, int(n_subscribers))
    subscribers = [
        _LiveSubscriber(broker_host, broker_port, protected_prefix,
                        client_id=f"sub-{run_id}-{i}")
        for i in range(n_subscribers)
    ]
    publisher = _LivePublisher(broker_host, broker_port, raw_prefix,
                               client_id=f"pub-{run_id}")

    np.random.seed(seed)
    publishers = list(per_pub.keys())
    T = min(n_steps, len(per_pub[publishers[0]]))
    tau_truth: list[dict] = []
    t0_wall = None

    try:
        for sub in subscribers:
            sub.start()
        plugin.start()
        publisher.start()

        # Give the broker a moment to finish subscription ack-and-route before
        # the first publish.  This avoids a race where tau=0 messages arrive
        # before any subscription is live.
        time.sleep(max(0.2, live_dt))

        t0_wall = time.time()
        for tau in range(T):
            tau_start_wall = time.time()
            active_values = []
            for pub_id in publishers:
                v = per_pub[pub_id][tau]
                if v is None:
                    continue
                publisher.publish_one(_leaf_of(pub_id), pub_id, float(v))
                active_values.append(max(lo, min(hi, float(v))))
            tau_truth.append({
                "tau": tau,
                "true_clamped_mean": float(np.mean(active_values)) if active_values else 0.0,
                "n_active": len(active_values),
                "wall_clock_publish": tau_start_wall,
            })
            # Pace to one publish burst per live_dt so the plugin timer groups
            # this tau's messages into one flush.
            elapsed = time.time() - tau_start_wall
            if elapsed < live_dt:
                time.sleep(live_dt - elapsed)

        # Drain the tail: let the plugin timer fire a few more times so the
        # final window is released.
        time.sleep(drain_ticks * live_dt + 0.2)
    finally:
        try:
            plugin.stop()
        finally:
            publisher.stop()
            for sub in subscribers:
                try:
                    sub.stop()
                except Exception:
                    pass

    # Aggregate across every subscriber so downstream metrics still see one
    # canonical delivery list; also record per-subscriber counts so the
    # caller can verify broker fan-out was correct (every subscriber should
    # receive every release).
    subscriber_received: list[dict] = []
    per_subscriber_counts: list[int] = []
    for sub in subscribers:
        recv = list(sub.received)
        subscriber_received.extend(recv)
        per_subscriber_counts.append(len(recv))

    return {
        "plugin_log": list(plugin.release_log),
        "subscriber_received": subscriber_received,
        "per_subscriber_counts": per_subscriber_counts,
        "num_subscribers": n_subscribers,
        "scenario": scenario,
        "plugin_P": plugin_P,
        "tau_truth": tau_truth,
        "run_id": run_id,
        "raw_prefix": raw_prefix,
        "protected_prefix": protected_prefix,
        "num_published": publisher.num_published,
        "t0_wall": t0_wall,
    }


def experiment_E_live_broker(
    dataset_name: str,
    output_dir: str,
    args,
    *,
    broker_host: str = "localhost",
    broker_port: int = 1883,
    live_dt: float = 0.1,
    n_steps: int = 120,
    seed: int = 123,
    auto_start_broker: bool = True,
) -> pd.DataFrame:
    """Exp E: small sanity grid through a live MQTT broker.

    Grid (9 configs): 3 strategies × 3 epsilons × 1 (dataset, sensor, w, P).
    Intended for the smallest datasets ('wearable' default); do NOT run on
    'energy' (~36k windows per sensor -> hours of broker traffic).

    If no broker is listening at (broker_host, broker_port) and
    ``auto_start_broker`` is True, spins up an embedded amqtt broker on the
    same (host, port) for the duration of the experiment.
    """
    if dataset_name == "energy":
        logger.warning(
            "[exp E] 'energy' has ~36k windows/sensor; that is ~1 hour per "
            "config at live_dt=0.1s.  Strongly recommend 'wearable' (119 "
            "windows) or 'manufacturing' (1000)."
        )

    embedded_broker = _ensure_broker(broker_host, broker_port, auto_start_broker)
    try:
        return _experiment_E_run(
            dataset_name, output_dir, args,
            broker_host=broker_host, broker_port=broker_port,
            live_dt=live_dt, n_steps=n_steps, seed=seed,
        )
    finally:
        if embedded_broker is not None:
            embedded_broker.stop()


def _experiment_E_run(
    dataset_name: str,
    output_dir: str,
    args,
    *,
    broker_host: str,
    broker_port: int,
    live_dt: float,
    n_steps: int,
    seed: int,
) -> pd.DataFrame:
    prepared = prepare_dataset(
        dataset_name,
        clamp_mode="static",
        eps_clip=args.eps_clip,
        seed=args.seed,
        max_rows=_dataset_max_rows(dataset_name, args),
    )
    if prepared is None or not prepared.per_pubs:
        logger.warning(f"[exp E] nothing prepared for {dataset_name}; skip")
        return pd.DataFrame()

    sensor = next(
        (s for s in prepared.spec["sensors"]
         if s in prepared.per_pubs
         and s in prepared.spec["static_clamps"]
         and len(prepared.per_pubs[s][0]) >= 2),
        None,
    )
    if sensor is None:
        logger.warning(f"[exp E] no suitable sensor in {dataset_name}; skip")
        return pd.DataFrame()

    per_pub = prepared.per_pubs[sensor][0]
    lo, hi = prepared.spec["static_clamps"][sensor]
    B = float(hi - lo)

    strategies = ["uniform", "p_gated_ba", "n_weighted"]
    epsilons = [0.5, 1.0, 2.0]
    w = 8
    P = 2

    # Scenarios exercised end-to-end:
    #   pooled    — many publishers on one shared leaf (no walk-up; matches
    #               offline reference exactly).
    #   hierarchy — each publisher on its own leaf under the normative
    #               topic tree, P>=2 forces Algorithm 1 walk-up on every
    #               release.
    # The pooled scenario is always valid; the hierarchy scenario is skipped
    # only when the dataset lacks a ``publisher_topic`` factory.
    raw_scenarios = getattr(args, "live_scenarios", None)
    if isinstance(raw_scenarios, str):
        scenarios: list[str] = [
            s.strip() for s in raw_scenarios.split(",") if s.strip()
        ]
    else:
        scenarios = list(raw_scenarios) if raw_scenarios else []
    if not scenarios:
        scenarios = ["pooled", "hierarchy"]
    if "hierarchy" in scenarios and "publisher_topic" not in prepared.spec:
        logger.info(
            f"[exp E] {dataset_name}: dataset lacks 'publisher_topic' spec; "
            "skipping hierarchy scenario"
        )
        scenarios = [s for s in scenarios if s != "hierarchy"]

    n_subscribers: int = max(1, int(getattr(args, "live_n_subscribers", 3)))

    # Dataset-specific output path so running E across multiple datasets
    # (e.g. --experiment E --dataset all) does not clobber earlier results.
    exp_dir = os.path.join(output_dir, "experiments", "E_live_broker",
                           dataset_name)
    os.makedirs(exp_dir, exist_ok=True)

    rows = []
    all_messages: list[dict] = []
    log_messages = getattr(args, "log_messages", True)
    t_exp_start = time.time()
    for strategy in strategies:
        for epsilon in epsilons:
            for scenario in scenarios:
                t_cfg = time.time()
                logger.info(
                    f"[exp E] {dataset_name}/{sensor} strategy={strategy} "
                    f"eps={epsilon} w={w} P={P} scenario={scenario} "
                    f"subscribers={n_subscribers} "
                    f"(live broker {broker_host}:{broker_port})"
                )
                try:
                    out = _drive_live_config(
                        per_pub, sensor, prepared.spec,
                        strategy=strategy, epsilon=epsilon, w=w, P=P,
                        broker_host=broker_host, broker_port=broker_port,
                        live_dt=live_dt, n_steps=n_steps, seed=seed,
                        scenario=scenario,
                        n_subscribers=n_subscribers,
                    )
                except Exception as exc:
                    logger.exception(
                        f"[exp E] config failed (strategy={strategy}, "
                        f"eps={epsilon}, scenario={scenario}): {exc}"
                    )
                    rows.append({
                        "dataset": dataset_name, "sensor": sensor,
                        "strategy": strategy, "epsilon": epsilon,
                        "w": w, "P": P, "scenario": scenario,
                        "num_subscribers": n_subscribers,
                        "status": "error", "error": str(exc),
                    })
                    continue

                log = out["plugin_log"]
                released = [r for r in log if not r["deferred"]
                            and r["released_value"] is not None]
                plugin_published = [r for r in log
                                    if r["released_value"] is not None]

                # Canonical metrics from the plugin log.
                live_nmae, live_mae, live_kl, live_attr = _live_metrics(
                    released, out["tau_truth"], B,
                )

                # Offline reference with the same seed; used for comparison
                # metrics and to populate epsilon_tau / lambda_tau in the
                # per-release message log (plugin doesn't expose budget
                # metadata directly).
                agg = [e["true_clamped_mean"] for e in out["tau_truth"]]
                cnt = [e["n_active"] for e in out["tau_truth"]]
                offline = run_dp_on_stream(
                    agg, cnt, epsilon=epsilon, window_size=w,
                    min_publishers=P, payload_bound=B,
                    strategy=strategy, seed=seed,
                )
                off_m = offline["metrics"]

                walkup_count = sum(1 for r in log if r.get("walk_up"))
                walkup_rate = walkup_count / max(1, len(log))
                # P-gate violation: a released (non-deferred) record with
                # n_tau < plugin_P indicates the gate is broken.
                p_gate_violations = sum(
                    1 for r in released
                    if int(r["n_tau"]) < int(out.get("plugin_P", P))
                )

                # Per-release message log.  Canonical rows from the live
                # plugin path, augmented with broker-delivery and walk-up
                # audit columns.
                if log_messages:
                    subscriber_by_t = {
                        float(m.get("t_start", -1)): m
                        for m in out["subscriber_received"]
                    }
                    off_budgets = offline.get("budgets_spent") or []
                    for rec in log:
                        tau_idx = int(rec["tau"])  # 1-indexed
                        n_tau = int(rec["n_tau"])
                        true_v = rec["true_aggregate"]
                        noisy_v = rec["released_value"]
                        eps_tau = (float(off_budgets[tau_idx - 1])
                                   if 0 <= tau_idx - 1 < len(off_budgets)
                                   else 0.0)
                        deferred = bool(rec["deferred"]) or eps_tau <= 0
                        if (true_v is not None and noisy_v is not None
                                and not deferred):
                            noise = float(noisy_v) - float(true_v)
                        else:
                            noise = 0.0
                        lam = (float(B) / (n_tau * eps_tau)
                               if eps_tau > 0 and n_tau > 0
                               else float("inf"))
                        delta_f = (float(B) / n_tau
                                   if n_tau > 0 else float("inf"))
                        t_start = float(rec.get("t_start", 0.0))
                        all_messages.append({
                            "dataset": dataset_name,
                            "clamp_mode": "static",  # E runs under Option A
                            "sensor": sensor,
                            "strategy": strategy,
                            "P": int(P),
                            "epsilon": float(epsilon),
                            "w": int(w),
                            "payload_bound": float(B),
                            "seed": int(seed),
                            "experiment": f"E_live_broker/{scenario}",
                            "config_id": (
                                f"{dataset_name}|E|{scenario}|{sensor}"
                                f"|{strategy}|P={P}|eps={epsilon}|w={w}"
                                f"|run={out['run_id']}"
                            ),
                            "tau": tau_idx,
                            "t_start_logical": t_start,
                            "true_aggregate": (float(true_v)
                                               if true_v is not None else None),
                            "noisy_value": (float(noisy_v)
                                            if noisy_v is not None else None),
                            "n_tau": n_tau,
                            "epsilon_tau": eps_tau,
                            "lambda_tau": lam,
                            "noise_sample": noise,
                            "deferred": deferred,
                            "delta_f": delta_f,
                            # Live-broker-specific audit columns:
                            "scenario": scenario,
                            "plugin_P": int(out.get("plugin_P", P)),
                            "num_subscribers": n_subscribers,
                            "leaf_topic": rec.get("leaf_topic"),
                            "release_scope": rec.get("release_scope"),
                            "walk_up": bool(rec.get("walk_up", False)),
                            "broker_delivered": t_start in subscriber_by_t,
                            "run_id": out["run_id"],
                        })

                # Broker-path integrity: every protected-topic publish the
                # plugin made should have been delivered to every subscriber.
                per_sub = out.get("per_subscriber_counts", [])
                broker_deliveries = len(out["subscriber_received"])
                expected_deliveries = len(plugin_published) * n_subscribers
                broker_delivery_ok = (
                    broker_deliveries == expected_deliveries
                    and all(c == len(plugin_published) for c in per_sub)
                )

                rows.append({
                    "dataset": dataset_name,
                    "sensor": sensor,
                    "strategy": strategy,
                    "epsilon": epsilon,
                    "w": w,
                    "P": P,
                    "scenario": scenario,
                    "plugin_P": int(out.get("plugin_P", P)),
                    "num_subscribers": n_subscribers,
                    "per_subscriber_counts": ";".join(str(c) for c in per_sub),
                    "run_id": out["run_id"],
                    "num_taus": len(log),
                    "num_released": len(released),
                    "num_plugin_published": len(plugin_published),
                    "num_walkups": walkup_count,
                    "walkup_rate": walkup_rate,
                    "p_gate_violations": p_gate_violations,
                    "release_rate_live": len(released) / max(1, len(log)),
                    "release_rate_offline": off_m.get("release_rate",
                                                      float("nan")),
                    "nmae_live": live_nmae,
                    "nmae_offline": off_m.get("normalized_mae", float("nan")),
                    "nmae_abs_delta": abs(
                        live_nmae - off_m.get("normalized_mae", float("nan"))),
                    "mae_live": live_mae,
                    "kl_live": live_kl,
                    "kl_offline": off_m.get("kl_divergence", float("nan")),
                    "attribution_advantage_live": live_attr,
                    "attribution_advantage_offline":
                        off_m.get("attribution_advantage", float("nan")),
                    "broker_deliveries": broker_deliveries,
                    "expected_deliveries": expected_deliveries,
                    "broker_delivery_ok": broker_delivery_ok,
                    "num_input_published": out["num_published"],
                    "wall_clock_seconds": round(time.time() - t_cfg, 2),
                    "status": "ok",
                })
                logger.info(
                    f"[exp E]   scenario={scenario} "
                    f"released={len(released)}/{len(log)} "
                    f"walkups={walkup_count} "
                    f"p_violations={p_gate_violations} "
                    f"broker_delivered={broker_deliveries}/{expected_deliveries} "
                    f"nmae_live={live_nmae:.4f} "
                    f"nmae_offline={off_m.get('normalized_mae', 0):.4f} "
                    f"(elapsed {time.time() - t_cfg:.1f}s)"
                )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(exp_dir, "experiment_E_live_broker.csv"), index=False)
    logger.info(
        f"  Experiment E wrote {len(df)} rows -> {exp_dir} "
        f"(total wall-clock {time.time() - t_exp_start:.1f}s)"
    )

    if log_messages and all_messages:
        from message_logger import write_messages_csv
        n = write_messages_csv(
            all_messages,
            os.path.join(exp_dir, "experiment_E_messages.csv"),
        )
        logger.info(f"  Experiment E wrote {n} per-release messages "
                    f"(live plugin path; broker_delivered flag included)")

    # Plot generation is delegated to generate_plots.py by default; inline
    # rendering only runs when the user passes --generate-plots (monkey-patch
    # of plt.savefig in main() no-ops this call otherwise).
    if getattr(args, "generate_plots", False):
        _plot_experiment_E(df,
                           os.path.join(exp_dir, "experiment_E_live_broker.png"),
                           dataset_name, sensor, live_dt,
                           broker_host, broker_port)
    return df


def _live_metrics(released_records, tau_truth, payload_bound):
    """Compute NMAE / MAE / KL / attribution advantage from the plugin's release log."""
    if not released_records:
        return float("nan"), float("nan"), float("nan"), float("nan")
    # Match by tau index (plugin's current_tau counts from 1; tau_truth from 0).
    truth_by_tau = {e["tau"]: e for e in tau_truth}
    abs_err = []
    attr = []
    fresh_true, fresh_noisy = [], []
    for r in released_records:
        t = truth_by_tau.get(r["tau"] - 1)
        if t is None:
            continue
        err = abs(r["released_value"] - t["true_clamped_mean"])
        abs_err.append(err)
        if r["n_tau"] > 0:
            attr.append(1.0 / r["n_tau"])
        fresh_true.append(t["true_clamped_mean"])
        fresh_noisy.append(r["released_value"])
    if not abs_err:
        return float("nan"), float("nan"), float("nan"), float("nan")
    mae = float(np.mean(abs_err))
    nmae = mae / max(payload_bound, 1e-9)
    # KL using the same binning rules as dp_engine (import-on-demand).
    try:
        from dp_engine import compute_kl_divergence
        kl = float(compute_kl_divergence(np.array(fresh_true),
                                         np.array(fresh_noisy)))
    except Exception:
        kl = float("nan")
    attr_adv = float(np.mean(attr)) if attr else float("nan")
    return nmae, mae, kl, attr_adv


def _plot_experiment_E(df, path, dataset_name, sensor, live_dt, broker_host, broker_port):
    """Bar + scatter plot comparing live-broker metrics to offline reference."""
    if df.empty:
        return
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    # (a) NMAE live vs offline per config
    labels = [f"{r.strategy}\neps={r.epsilon}" for r in ok.itertuples()]
    x = np.arange(len(labels))
    bw = 0.38
    axes[0].bar(x - bw / 2, ok["nmae_live"], bw, label="live broker", color="C0")
    axes[0].bar(x + bw / 2, ok["nmae_offline"], bw, label="offline (same seed)", color="C1")
    axes[0].set(xticks=x, ylabel="NMAE",
                title="(a) Utility: live broker vs offline engine")
    axes[0].set_xticklabels(labels, fontsize=8, rotation=0)
    axes[0].grid(True, alpha=0.3, axis="y")
    axes[0].legend(fontsize=8)

    # (b) KL live vs offline
    axes[1].bar(x - bw / 2, ok["kl_live"], bw, label="live broker", color="C0")
    axes[1].bar(x + bw / 2, ok["kl_offline"], bw, label="offline (same seed)", color="C1")
    axes[1].set(xticks=x, ylabel="KL divergence",
                title="(b) Distributional utility")
    axes[1].set_xticklabels(labels, fontsize=8, rotation=0)
    axes[1].grid(True, alpha=0.3, axis="y")
    axes[1].legend(fontsize=8)

    # (c) Broker delivery integrity
    axes[2].bar(x - bw / 2, ok["num_released"], bw, label="plugin released", color="C2")
    axes[2].bar(x + bw / 2, ok["broker_deliveries"], bw, label="subscriber received", color="C3")
    axes[2].set(xticks=x, ylabel="count",
                title="(c) MQTT path integrity")
    axes[2].set_xticklabels(labels, fontsize=8, rotation=0)
    axes[2].grid(True, alpha=0.3, axis="y")
    axes[2].legend(fontsize=8)

    fig.suptitle(
        f"Experiment E: live MQTT broker [{broker_host}:{broker_port}] "
        f"dataset={dataset_name} sensor={sensor} dt={live_dt}s",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


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
    datasets, clamp_modes, output_dir, args, strategies, which="ABCD",
):
    """Drive Experiments A, B, C, and D (subset selectable via `which`)."""
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
        if "D" in which:
            experiment_D_plugin_path(
                datasets, clamp_mode,
                os.path.join(output_dir, "cross_dataset", clamp_mode),
                args,
            )
        grid_config = getattr(args, "grid_config", None)
        if "F" in which:
            ablation_experiment(
                datasets, clamp_mode,
                os.path.join(output_dir, "cross_dataset", clamp_mode),
                args, epsilon=1.0, w=8, P=getattr(args, "ablation_P", 3),
                k_ext=getattr(args, "k_ext", 3),
                epsilon_count=getattr(args, "epsilon_count", 0.0),
                grid_config=grid_config,
            )
        if "G" in which:
            overhead_experiment(
                datasets, clamp_mode,
                os.path.join(output_dir, "cross_dataset", clamp_mode),
                args, epsilon=1.0, w=8, P=getattr(args, "ablation_P", 3),
                epsilon_count=getattr(args, "epsilon_count", 0.0),
                grid_config=grid_config,
            )
        if "H" in which:
            average_case_utility_experiment(
                datasets, clamp_mode,
                os.path.join(output_dir, "cross_dataset", clamp_mode),
                args, epsilon=1.0, w=8, P=getattr(args, "ablation_P", 3),
                epsilon_count=getattr(args, "epsilon_count", 0.5),
                grid_config=grid_config,
            )


# ═════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════

CLAMP_MODES = ["static", "dp_released"]


def _run_grid_search_block(args, targets, clamp_modes, eps_values, strategies):
    """Paper Sec. 7.5 grid search over (P_min x P_max x Delta_t x K_ext) scored
    by MAE, per dataset/clamp/strategy/epsilon.  Writes per-(dataset,eps) grids
    and the consolidated grid_canonical.json.  Returns the canonical config path.
    """
    p_min_grid = [1, 2, 3, 4, 6]
    p_max_grid = [None, 4, 8, 16]
    dt_grid = [1, 2, 4]
    k_ext_grid = [0, 2, 4]
    canonical_records: list[dict] = []
    for clamp_mode in clamp_modes:
        for name in targets:
            first_sensor = DATASETS[name]["sensors"][0]
            prepared = prepare_dataset(
                name, clamp_mode=clamp_mode, eps_clip=args.eps_clip,
                seed=args.seed, max_rows=_dataset_max_rows(name, args),
                sensors=[first_sensor],
            )
            if prepared is None or first_sensor not in prepared.per_pubs:
                continue
            pp, B = prepared.per_pubs[first_sensor]
            dirs = _dataset_dirs(args.output_dir, name, clamp_mode)
            grid_dir = os.path.join(dirs["tuning"], "grid_search")
            for eps in eps_values:
                grid_search_hyperparameters(
                    pp, B, name, f"{first_sensor}_eps{eps}", grid_dir,
                    epsilon=eps, w=8, strategies=strategies,
                    p_min_grid=p_min_grid, p_max_grid=p_max_grid,
                    dt_grid=dt_grid, k_ext_grid=k_ext_grid,
                    epsilon_count=args.epsilon_count,
                    workers=args.workers,
                )
                best_csv = os.path.join(
                    grid_dir, f"{name}_{first_sensor}_eps{eps}_gridsearch_best.csv")
                if os.path.exists(best_csv):
                    bdf = pd.read_csv(best_csv)
                    for _i, r in bdf.iterrows():
                        rec = r.to_dict()
                        rec["dataset"] = name
                        rec["clamp_mode"] = clamp_mode
                        canonical_records.append(rec)
    return _write_grid_canonical(args.output_dir, canonical_records)


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
    parser.add_argument("--epsilon-count", type=float, default=0.0,
                        help="eps_count: per-step budget spent to release a "
                             "differentially private publisher count |P_tau| "
                             "(sensitivity 1) when gating / walking the topic "
                             "hierarchy (paper Sec. 6.3 step 1, Table 3).  0 "
                             "uses the exact count (Kellaris baselines).")
    parser.add_argument("--max-publishers", type=int, default=None,
                        help="P_max: cap the multiplicity folded into the mean "
                             "so Delta_f = R/n changes by a bounded amount "
                             "across stream elements (Sec. 6.5).  None = no cap.")
    parser.add_argument("--ablation-P", type=int, default=3,
                        help="P_min used by the ablation / overhead / "
                             "average-case experiments (F/G/H).")
    parser.add_argument("--k-ext", type=int, default=3,
                        help="K_ext: max adaptive interval extensions used by "
                             "the ablation (F) interval-extension module and "
                             "the plugin path (Sec. 6.7).")
    parser.add_argument("--alpha", type=float, default=0.25,
                        help="Attribution-advantage target; Algorithm 2 seeds P_0 = ceil(1/alpha)")
    parser.add_argument("--I-max", type=int, default=20,
                        help="Algorithm 2 iteration cap for the greedy hill-climb over P")
    parser.add_argument("--n-restarts", type=int, default=3,
                        help="Algorithm 2 intelligent multi-start: the first "
                             "seed is P_0 = ceil(1/alpha); the remaining "
                             "n_restarts-1 seeds are drawn from quantiles of "
                             "the observed n_tau distribution.  1 disables "
                             "random restart and reverts to paper's "
                             "deterministic seed only.")
    parser.add_argument("--restart-rng-seed", type=int, default=12345,
                        help="Random seed used to pick the non-P_0 restart "
                             "seeds in Algorithm 2 (controls reproducibility "
                             "of the quantile fallback draw).")
    parser.add_argument("--log-messages", dest="log_messages",
                        action="store_true", default=True,
                        help="Write per-release message CSV (true aggregate, "
                             "noisy value, n_tau, eps_tau, lambda_tau, "
                             "deferred flag, ...) for every DP run.  Default: on.")
    parser.add_argument("--no-log-messages", dest="log_messages",
                        action="store_false",
                        help="Disable per-release message logging.")
    parser.add_argument("--generate-plots", dest="generate_plots",
                        action="store_true", default=False,
                        help="Generate PNG plots inline during the run.  "
                             "Default: off -- use generate_plots.py post-hoc.")
    parser.add_argument("--no-generate-plots", dest="generate_plots",
                        action="store_false",
                        help="Skip inline plot generation (the canonical "
                             "workflow: run experiments, then generate plots "
                             "separately from the CSVs via generate_plots.py).")
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
    parser.add_argument("--grid-search", action="store_true",
                        help="Run the Section 7.5 utility-hyperparameter grid "
                             "search (P_min x P_max x Delta_t x K_ext, scored "
                             "by MAE) per dataset/strategy/epsilon, write "
                             "grid_canonical.json, then exit.")
    parser.add_argument("--use-grid-config", default=None,
                        help="Path to a grid_canonical.json produced by "
                             "--grid-search.  Downstream experiments resolve "
                             "their (P_min, P_max, K_ext) from the MAE-optimal "
                             "grid config per (dataset, clamp, strategy, eps) "
                             "instead of CLI defaults (paper Sec. 7.5: grid "
                             "search fixes the params for the rest).  Defaults "
                             "to auto-detecting <output-dir>/grid_canonical.json.")
    parser.add_argument("--grid-first", action="store_true",
                        help="Run the Section 7.5 grid search FIRST, write the "
                             "canonical config, and have this same run consume "
                             "it for every downstream experiment.")
    parser.add_argument(
        "--experiment",
        choices=["full", "sweep", "tune", "A", "B", "C", "D", "E", "F", "G", "H",
                 "ABC", "ABCD", "ABCDFGH", "FGH", "none"],
        default="full",
        help="'full' runs sweep + intro + tuning + experiments A/B/C/D/F/G/H.  "
             "'sweep' is the main grid only.  'tune' is just Algorithm 2 / "
             "brute-force tuning.  A/B/C/D/F/G/H pick one single-axis "
             "experiment: A greedy-vs-brute P-tuning, B vary-w, C vary-eps, "
             "D plugin end-to-end, F incremental-module ablation (Sec. 7.8, "
             "offline, no broker), G overhead/privacy-utility comparison "
             "(Sec. 7.9), H average-case utility (Sec. 7.11).  'E' runs the "
             "live-broker sanity subset.  'ABC'/'ABCD'/'FGH'/'ABCDFGH' run "
             "those subsets without the main sweep.",
    )
    parser.add_argument(
        "--broker-host", default="localhost",
        help="MQTT broker host for Experiment E (live-broker sanity subset)",
    )
    parser.add_argument(
        "--broker-port", type=int, default=1883,
        help="MQTT broker port for Experiment E",
    )
    parser.add_argument(
        "--live-dt", type=float, default=0.1,
        help="Wall-clock seconds per logical timestamp for Experiment E "
             "(short values compress runtime; default 0.1s)",
    )
    parser.add_argument(
        "--live-n-steps", type=int, default=120,
        help="Cap on logical timestamps per config for Experiment E",
    )
    parser.add_argument(
        "--no-auto-broker", dest="auto_broker", action="store_false",
        default=True,
        help="Do not auto-start an embedded amqtt broker for Experiment E "
             "when none is listening; fail instead.  Default behaviour is to "
             "reuse any existing broker at --broker-host/--broker-port, "
             "otherwise spin one up for the run.",
    )
    parser.add_argument(
        "--include-energy-in-E", action="store_true", default=False,
        help="When running Experiment E with --dataset all, also include "
             "the 'energy' dataset.  Default: skip energy (~36k windows/"
             "sensor -> hours of broker traffic per config).",
    )
    parser.add_argument(
        "--live-scenarios", default="pooled,hierarchy",
        help="Comma-separated subset of {pooled,hierarchy} to run under "
             "Experiment E.  'pooled' exercises the many-publisher shared-"
             "leaf path; 'hierarchy' puts each publisher on its own leaf "
             "under the dataset's normative MQTT tree so P>=2 forces "
             "Algorithm 1 walk-ups on every release.",
    )
    parser.add_argument(
        "--live-n-subscribers", type=int, default=3,
        help="Number of concurrent subscriber clients attached to the "
             "protected prefix per Experiment E config.  The broker should "
             "fan each release out to every subscriber; a mismatch between "
             "broker_deliveries and expected_deliveries flags a fan-out bug.",
    )
    parser.add_argument(
        "--run-live-E-after-full", action="store_true", default=False,
        help="When --experiment full is used, additionally run Experiment E "
             "after the main pipeline completes (live MQTT broker on every "
             "non-energy dataset by default; respects --include-energy-in-E).",
    )
    parser.add_argument(
        "--workers", type=int, default=0,
        help="Parallel worker processes for the sweep and experiments "
             "A/B/C.  0 (default) uses os.cpu_count() - 1.  Set to 1 to "
             "run serially.",
    )
    args = parser.parse_args()
    args.workers = _default_workers(args.workers)
    logger.info(f"Using {args.workers} worker process(es) for parallel tasks")

    # When plot generation is disabled, short-circuit savefig so the
    # experiment functions keep writing their CSVs but produce no PNGs.
    # generate_plots.py reads those CSVs post-hoc and renders every figure.
    if not args.generate_plots:
        def _savefig_noop(*_a, **_kw):  # pragma: no cover - trivial
            return None
        plt.savefig = _savefig_noop
        logger.info("Inline plot generation DISABLED. Run `python "
                    "generate_plots.py --output-dir <results_dir>` "
                    "after the experiment completes to render every figure.")

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

    # Resolve the canonical grid config (paper Sec. 7.5): explicit path wins,
    # else auto-detect <output-dir>/grid_canonical.json.
    grid_cfg_path = args.use_grid_config or _grid_canonical_path(args.output_dir)
    args.grid_config = _load_grid_config(grid_cfg_path)

    if args.grid_first:
        # Run the grid search first, then consume its canonical config below.
        path = _run_grid_search_block(args, targets, clamp_modes,
                                      eps_values, strategies)
        args.grid_config = _load_grid_config(path)

    if args.grid_search:
        _run_grid_search_block(args, targets, clamp_modes, eps_values, strategies)
        logger.info("Section 7.5 grid search complete; canonical config written. "
                    "Re-run the experiments with --use-grid-config "
                    f"{_grid_canonical_path(args.output_dir)} to consume it "
                    "(or use --grid-first to do both in one run).")
        return

    if args.tune_only:
        for clamp_mode in clamp_modes:
            for name in targets:
                first_sensor = DATASETS[name]["sensors"][0]
                prepared = prepare_dataset(
                    name,
                    clamp_mode=clamp_mode,
                    eps_clip=args.eps_clip,
                    seed=args.seed,
                    max_rows=_dataset_max_rows(name, args),
                    sensors=[first_sensor],
                )
                if prepared is None:
                    continue
                if first_sensor not in prepared.raw_per_pubs:
                    logger.warning(f"Skipping {name} tuning: no per-pub stream"); continue
                if first_sensor not in prepared.per_pubs:
                    continue
                pp, B = prepared.per_pubs[first_sensor]
                dirs = _dataset_dirs(args.output_dir, name, clamp_mode)
                tune = tune_hyperparameters(
                    pp, B, name, first_sensor, dirs["tuning"],
                    epsilon=1.0, w=8,
                    strategies=strategies,
                    alpha=args.alpha, I_max=args.I_max,
                    workers=args.workers,
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
    run_single_axis = args.experiment in (
        "full", "ABC", "ABCD", "ABCDFGH", "FGH",
        "A", "B", "C", "D", "F", "G", "H",
    )
    run_live_E = args.experiment == "E"

    if run_live_E:
        # --dataset <name>   -> run E on that one dataset.
        # --dataset all      -> run E on every dataset EXCEPT energy
        #                       (energy has ~36k windows/sensor -> hours per
        #                       config on a live broker).  Set
        #                       --include-energy-in-E to force it.
        if args.dataset == "all":
            datasets_for_E = [ds for ds in DATASETS.keys()
                              if ds != "energy" or args.include_energy_in_E]
            if not args.include_energy_in_E:
                logger.info("[exp E] --dataset all: skipping 'energy' "
                            "(pass --include-energy-in-E to force)")
        else:
            datasets_for_E = [args.dataset]
            if args.dataset == "energy":
                logger.warning(
                    "[exp E] --dataset energy is strongly discouraged "
                    "(~36k windows/sensor). Recommend 'wearable' or "
                    "'manufacturing'."
                )
        for ds_for_E in datasets_for_E:
            logger.info(f"===== Experiment E: dataset={ds_for_E} =====")
            experiment_E_live_broker(
                ds_for_E, args.output_dir, args,
                broker_host=args.broker_host,
                broker_port=args.broker_port,
                live_dt=args.live_dt,
                n_steps=args.live_n_steps,
                seed=args.seed or 123,
                auto_start_broker=args.auto_broker,
            )
        logger.info(
            f"Experiment E complete ({len(datasets_for_E)} dataset(s)).")
        return

    if run_main_pipeline:
        for clamp_mode in clamp_modes:
            for name in targets:
                try:
                    results = run_dataset(
                        name, s_values, eps_values, w_values, strategies,
                        args.output_dir, args,
                        clamp_mode=clamp_mode,
                        quick=args.quick, skip_extras=args.skip_extras,
                        workers=args.workers,
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
        if args.experiment == "full":
            which = "ABCDFGH"   # full pipeline runs every single-axis experiment
        elif args.experiment in ("ABCD", "ABC", "ABCDFGH", "FGH"):
            which = args.experiment
        else:
            which = args.experiment  # single letter A/B/C/D/F/G/H
        run_single_axis_experiments(
            targets, clamp_modes, args.output_dir, args, strategies, which=which,
        )

    # Experiment E auto-runs after --experiment full when --run-live-E-after-full
    # is set.  Each dataset gets its own live-broker sub-run (pooled + hierarchy
    # scenarios, --live-n-subscribers concurrent subscribers); results land
    # under <output_dir>/experiments/E_live_broker/<dataset>/.
    if (args.experiment == "full"
            and getattr(args, "run_live_E_after_full", False)):
        e_targets = [ds for ds in (
            targets if args.dataset != "all" else list(DATASETS.keys())
        ) if ds != "energy" or args.include_energy_in_E]
        if "energy" in (targets if args.dataset != "all"
                        else list(DATASETS.keys())) \
                and not args.include_energy_in_E:
            logger.info("[exp E/auto] skipping 'energy' dataset (pass "
                        "--include-energy-in-E to force)")
        for ds_for_E in e_targets:
            logger.info(f"===== Experiment E (post-full): dataset={ds_for_E} =====")
            try:
                experiment_E_live_broker(
                    ds_for_E, args.output_dir, args,
                    broker_host=args.broker_host,
                    broker_port=args.broker_port,
                    live_dt=args.live_dt,
                    n_steps=args.live_n_steps,
                    seed=args.seed or 123,
                    auto_start_broker=args.auto_broker,
                )
            except Exception as exc:
                logger.exception(
                    f"[exp E/auto] failed on {ds_for_E}: {exc} "
                    "(continuing with next dataset)"
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
