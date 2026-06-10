#!/usr/bin/env python3
"""Shared experiment engine: the DP runners, parallel-pool infra, topic-hierarchy
helpers, dataset/grid resolution, the stubbed-MQTT plugin driver, plotting, and
the per-dataset / cross-dataset orchestration that every experiment module uses.

This module is the library; ``run_experiment.py`` is a thin CLI over it and each
``experiments/<name>.py`` imports it as ``core`` (``from experiments import
engine as core``).  Worker functions and the worker-global caches they read live
HERE (or in the experiment module that defines the worker), so ProcessPoolExecutor
pickling + the spawn initializer resolve the same module's globals consistently.
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

# Run-global split parameter rho_tau for the aggregate stream element
# (Definition: Aggregate Stream Element).  It is a single constant for a whole
# run (not a per-task sweep dimension), so the CLI publishes it via the
# DP_RHO_SPLIT environment variable -- which spawned ProcessPoolExecutor workers
# inherit -- instead of threading it through every task tuple / worker global.
# ``set_rho_split`` is called once after arg parsing; ``run_dp_on_stream`` (and
# the live-broker driver) read it back per call so workers pick up the same rho.
DP_RHO_SPLIT_ENV = "DP_RHO_SPLIT"


def set_rho_split(rho: float) -> None:
    """Publish the run-global rho_tau so spawned workers inherit it."""
    os.environ[DP_RHO_SPLIT_ENV] = repr(float(rho))


def resolve_rho_split(rho_split: float | None = None) -> float:
    """Return the rho_tau to use: an explicit override, else DP_RHO_SPLIT, else 0.2."""
    if rho_split is not None:
        return float(rho_split)
    return float(os.environ.get(DP_RHO_SPLIT_ENV, "0.2"))


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
    rho_split: float | None = None,
    max_publishers: int | None = None,
    sums: list[float] | None = None,
) -> dict:
    """Run the DP engine on a pre-aggregated stream (offline evaluation).

    For population-aware strategies (P-gated / n-weighted) the broker releases
    the aggregate stream element as a sum/count pair (Definition: Aggregate
    Stream Element): the per-step budget eps_tau = epsilon/window_size is split
    by ``rho_split`` into the noisy count n~_tau (rho share, Laplace scale
    1/(rho*eps_tau)) and the noisy sum S~_tau (1-rho share, scale R/((1-rho)*
    eps_tau)), post-processed into gamma_tau = S~_tau/max(n~_tau, 1).  rho <= 0
    disables the DP count (exact |P_tau|).  ``epsilon_count`` is deprecated and
    ignored (superseded by ``rho_split``).  ``max_publishers`` caps the
    multiplicity that enters the sensitivity (P_max, Sec. 6.5).
    """
    np.random.seed(seed)
    config = PrivacyConfig(
        epsilon=float(epsilon),
        window_size=int(window_size),
        min_publishers=int(min_publishers),
        payload_bound=float(payload_bound),
        strategy=BudgetStrategy(strategy),
        rho_split=resolve_rho_split(rho_split),
        epsilon_count=float(epsilon_count),
        # NaN-safe: a "no cap" P_max resolved from the grid optimum arrives as
        # NaN (pandas/JSON null), and NaN is truthy -- coerce it to None.
        # ``x == x`` is False only for NaN.
        max_publishers=(int(max_publishers)
                        if max_publishers and max_publishers == max_publishers
                        else None),
    )
    stream = StreamState(config=config)
    # The noisy-sum component S~_tau is built from the ACTUAL pooled sum
    # sum_{p in P_tau} x_{p,tau}, not from aggregate*n.  ``sums`` carries that
    # real per-timestamp data sum when the caller has it; otherwise we recover it
    # from the pooled mean and count -- which is exact because the upstream
    # rebuild emits one value per publisher, so aggregate*n == sum_p x_{p,tau}.
    if sums is None:
        sums = [a * n for a, n in zip(aggregates, pub_counts)]
    for agg, n, s in zip(aggregates, pub_counts, sums):
        stream.release(agg, n, true_sum=s)

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
        # NOTE: the per-window KL array is intentionally NOT returned here -- it
        # was an O(T) stride-1 pass computed on every config and consumed by
        # nobody (the scalar-metric workers discard the full arrays), so it
        # roughly doubled run_dp_on_stream's cost on the long streams.  The
        # window-averaged scalar lives in metrics["kl_global_utility"]; callers
        # that genuinely need the per-window series can call
        # compute_windowed_kl_divergence(true_values, noisy_values, w) directly.
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
        # run_dp_on_stream no longer returns the per-window KL array (it was an
        # O(T) pass consumed by nobody on the hot path); recompute it here only
        # for the inline-plot path that actually needs the series.
        "kl_windowed": np.asarray(
            compute_windowed_kl_divergence(
                res["true_values"], res["noisy_values"], int(w)),
            dtype=np.float32),
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

def _aggregate_over_trials(df, group_cols, metric_cols):
    """Collapse per-trial rows into per-configuration mean/std over trials.

    Returns a DataFrame with the group_cols plus, for every present metric,
    ``<metric>_mean`` and ``<metric>_std``, plus ``n_trials``.  Used by every
    experiment that supports ``--trials`` so the aggregate (mean +- std across
    the N noise realizations) is one CSV alongside the per-trial rows.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    group_cols = [c for c in group_cols if c in df.columns]
    present = [c for c in metric_cols if c in df.columns]
    if not group_cols or not present:
        return pd.DataFrame()
    g = df.groupby(group_cols, dropna=False)
    agg = g[present].agg(["mean", "std"]).reset_index()
    agg.columns = [
        col if isinstance(col, str)
        else (col[0] if col[1] == "" else f"{col[0]}_{col[1]}")
        for col in agg.columns
    ]
    counts = g.size().reset_index(name="n_trials")
    return agg.merge(counts, on=group_cols)


# Metric columns aggregated across trials (only those present are used).
_TRIAL_METRIC_COLS = [
    "normalized_mae", "mae", "rmse", "relative_error",
    "kl_divergence", "kl_global_utility", "release_rate",
    "attribution_advantage", "avg_n_tau", "eps_count_spent",
]


# The main parameter sweep lives in experiments/sweep.py; lazy-delegating stub
# keeps run_dataset's 'sweep' phase working.  The pool worker reads the
# core-owned _WORKER_STREAMS global filled by _init_streams_worker.
def sweep(*a, **k):
    from experiments.sweep import sweep as _f
    return _f(*a, **k)


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

# The extras experiments (n-weighted spotlight, K_ext induced-latency sweep
# Sec. 7.10, subscriber collusion) live in experiments/extras.py; lazy stubs.
def n_weighted_spotlight(*a, **k):
    from experiments.extras import n_weighted_spotlight as _f
    return _f(*a, **k)


def dynamic_interval_experiment(*a, **k):
    from experiments.extras import dynamic_interval_experiment as _f
    return _f(*a, **k)


def collusion_experiment(*a, **k):
    from experiments.extras import collusion_experiment as _f
    return _f(*a, **k)


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
    epsilon_count=0.0, max_publishers=None,
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
        rho_split=resolve_rho_split(), max_publishers=max_publishers,
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


# ═════════════════════════════════════════════════════════════════════════
#  Hyperparameter tuning (Algorithm 2 greedy + brute force) -- Sec. 6.9
# ═════════════════════════════════════════════════════════════════════════
# The implementation lives in experiments/tuning.py (its workers + worker-side
# cache must be importable under multiprocessing spawn from that module).  Core
# keeps lazy-delegating stubs so run_dataset's tuning phase calls into it.

def greedy_tune_P(*args, **kwargs):
    """Moved to experiments.tuning; thin lazy-delegating stub (avoids a circular import)."""
    from experiments.tuning import greedy_tune_P as _f
    return _f(*args, **kwargs)


def brute_force_tune_P(*args, **kwargs):
    """Moved to experiments.tuning; thin lazy-delegating stub (avoids a circular import)."""
    from experiments.tuning import brute_force_tune_P as _f
    return _f(*args, **kwargs)


def tune_hyperparameters(*args, **kwargs):
    """Moved to experiments.tuning; thin lazy-delegating stub (avoids a circular import)."""
    from experiments.tuning import tune_hyperparameters as _f
    return _f(*args, **kwargs)


# ═════════════════════════════════════════════════════════════════════════
#  Introduction figures (Sec. 1.3) + Figure 1 reproduction
# ═════════════════════════════════════════════════════════════════════════
# Implementations (extreme1_global / extreme1_1_per_type / extreme2_per_publisher
# / kl_extremes_vs_ours / u_shaped_curve / figure1_reproduction and the
# INTRO_* constants) live in experiments/intro.py.  Core keeps lazy-delegating
# stubs for the two entry points run_dataset calls; the shared topic-hierarchy
# helpers below (_topic_level_groups / level_subscription_streams /
# _group_true_stream) stay here because many experiments import them.

def figure1_reproduction(*args, **kwargs):
    """Moved to experiments.intro; thin lazy-delegating stub (avoids a circular import)."""
    from experiments.intro import figure1_reproduction as _f
    return _f(*args, **kwargs)


def run_intro_figures(*args, **kwargs):
    """Moved to experiments.intro; thin lazy-delegating stub (avoids a circular import)."""
    from experiments.intro import run_intro_figures as _f
    return _f(*args, **kwargs)


def _topic_level_groups(dataset_name, sensor, pub_ids):
    """Group a sensor's publishers by topic-hierarchy prefix at every level.

    Returns {level L (1..depth) -> {topic-prefix : [publisher ids]}} using the
    dataset's normative ``publisher_topic`` mapping.  Level 1 is the domain
    root (one group = the whole type); the deepest level is the per-publisher
    leaf.  Used to evaluate SUBSCRIPTIONS at each level of the hierarchy.
    Returns {} when the dataset has no topic mapping (e.g. synthetic tests).
    """
    spec = DATASETS.get(dataset_name)
    topic_of = spec.get("publisher_topic") if spec else None
    if topic_of is None:
        return {}
    paths = {}
    for p in pub_ids:
        try:
            paths[p] = topic_of(p, sensor).split("/")
        except Exception:
            continue
    if not paths:
        return {}
    maxdepth = max(len(v) for v in paths.values())
    levels: dict[int, dict[str, list]] = {}
    for L in range(1, maxdepth + 1):
        groups: dict[str, list] = {}
        for p, segs in paths.items():
            prefix = "/".join(segs[:L])
            groups.setdefault(prefix, []).append(p)
        levels[L] = groups
    return levels


def level_subscription_streams(dataset_name, sensor, per_pub, k_ext=0):
    """Enumerate the (aggregate, count) stream for a SUBSCRIPTION bound at every
    point in the PerCom topic hierarchy.

    Returns a list of ``(level, scope_label, agg, cnt, n_pubs)`` — one entry per
    (hierarchy level x subtree).  Level 1 is the domain root (one subscription
    pooling the whole type); the deepest level is the per-publisher leaf.  Each
    entry is the pooled stream a subscriber bound to that subtree receives, so
    an experiment can evaluate utility at EVERY level of the topic tree.

    When the dataset has no topic mapping (synthetic), falls back to a single
    whole-type subscription.
    """
    levels = _topic_level_groups(dataset_name, sensor, list(per_pub.keys()))
    if not levels:
        agg, cnt = _adaptive_interval_rebuild(per_pub, 1, k_ext)
        return [(1, sensor, agg, cnt, len(per_pub))]
    out = []
    for L in sorted(levels):
        for prefix, members in levels[L].items():
            agg, cnt = _adaptive_interval_rebuild(per_pub, 1, k_ext, subset=members)
            out.append((L, prefix, agg, cnt, len(members)))
    return out


def grid_level_streams(dataset_name, sensor, per_pub, clamp_mode, grid_config,
                       strategy, epsilon, default_k_ext=0, override_pmin=None):
    """Like ``level_subscription_streams`` but builds each level's stream from the
    GRID-OPTIMAL hyperparameters for (dataset, clamp, strategy, epsilon).

    The grid search (Sec. 7.5) fixes everything except the two free axes epsilon
    and w: P_min, P_max, Delta_t (base_dt) and K_ext come from the optimum, so the
    stream is rebuilt at the grid P_min / K_ext / Delta_t (which shape it via the
    adaptive-interval pooling), and the release-time P_max / rho are returned for
    the caller to pass to ``run_dp_on_stream``.  ``override_pmin`` forces P_min to
    a caller value (the intro figure sweeps every P_min instead of taking the
    grid's).  Returns ``[(L, scope, agg, cnt, n_pubs, prm)]`` where ``prm`` is the
    resolved {P_min, P_max, k_ext, delta_t, rho_split} for that (strategy, eps)."""
    prm = _resolve_params(
        grid_config, dataset_name, clamp_mode, strategy, epsilon,
        {"P_min": (override_pmin if override_pmin is not None else 1),
         "P_max": None, "k_ext": default_k_ext, "delta_t": 1})
    P = int(override_pmin) if override_pmin is not None else int(prm["P_min"])
    k_ext = int(prm.get("k_ext", default_k_ext) or 0)
    base_dt = int(prm.get("delta_t", 1) or 1)
    # The released stream's P_min is whatever actually shaped it (grid or override).
    prm = dict(prm); prm["P_min"] = P
    levels = _topic_level_groups(dataset_name, sensor, list(per_pub.keys()))
    if not levels:
        agg, cnt = _adaptive_interval_rebuild(per_pub, P, k_ext, base_dt=base_dt)
        return [(1, sensor, agg, cnt, len(per_pub), prm)]
    out = []
    for L in sorted(levels):
        for prefix, members in levels[L].items():
            agg, cnt = _adaptive_interval_rebuild(
                per_pub, P, k_ext, subset=members, base_dt=base_dt)
            out.append((L, prefix, agg, cnt, len(members), prm))
    return out


def _group_true_stream(per_pub, pub_ids, T):
    """Per-tau mean over a subset of publishers (the true signal a subscription
    bound to that subtree wants)."""
    out = []
    for tau in range(T):
        vals = [per_pub[p][tau] for p in pub_ids
                if tau < len(per_pub[p]) and per_pub[p][tau] is not None]
        out.append(float(np.mean(vals)) if vals else None)
    return out


def _kl_of(true_vals, noisy_vals):
    tc = [v for v in true_vals if v is not None]
    nc = [v for v in noisy_vals if v is not None]
    if len(tc) < 10 or len(nc) < 10:
        return float("nan")
    k = compute_kl_divergence(tc, nc)
    return k if np.isfinite(k) else float("nan")


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


def _shard_sensors(args, sensors):
    """Round-robin sensor sharding for extra cluster parallelism.

    When ``--sensor-shard 'i/k'`` is set, keep ``sensors[i::k]`` (deterministic
    over the dataset's fixed sensor order), so a heavy multi-sensor dataset's
    per-level sweep can run one shard per sensor-group on separate nodes -- the
    lever for the ~8 h ``energy`` sweep long pole.  Returns all sensors when
    unset.  Empty groups (a dataset with fewer sensors than k) are simply
    skipped by the caller.
    """
    sh = getattr(args, "_sensor_shard", None)
    sensors = list(sensors)
    if not sh:
        return sensors
    i, k = sh
    return [s for idx, s in enumerate(sensors) if idx % k == i]


def run_dataset(
    name: str,
    s_values, eps_values, w_values, strategies, output_dir, args,
    clamp_mode: str = "static",
    quick: bool = False, skip_extras: bool = False,
    workers: int = 1,
    phases: set[str] | None = None,
) -> dict:
    """Run the main-pipeline phases on one (dataset, clamp_mode) pair.

    ``phases`` selects which of {"sweep", "intro", "tuning", "extras"} to run
    (None = all).  Each phase is independently shardable on a cluster — the
    fine-grained launcher runs them as separate SLURM tasks so a heavy
    dataset's sweep / intro (Figure 1) / tuning / K_ext spread across nodes
    instead of running back-to-back on one.  ``skip_extras`` drops the extras
    and tuning phases (legacy flag).

    Returns a dict of DataFrames for cross-dataset aggregation.
    """
    ALL_PHASES = {"sweep", "intro", "tuning", "extras"}
    phases = set(phases) if phases else set(ALL_PHASES)
    if skip_extras:
        phases -= {"extras", "tuning"}
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

    # Optional per-sensor sharding (--sensor-shard i/k): restrict this shard to
    # its slice of the dataset's sensors so a heavy dataset's sweep spreads
    # across nodes one sensor-group per task.
    if getattr(args, "_sensor_shard", None):
        keep = set(_shard_sensors(args, list(streams.keys())))
        streams = {s: v for s, v in streams.items() if s in keep}
        per_pubs = {s: v for s, v in per_pubs.items() if s in keep}
        if not streams:
            logger.info(f"  [{name}/{clamp_mode}] sensor shard is empty; skipping")
            return {}

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
    trials = max(1, getattr(args, "trials", 1))
    w_mid = max(w_values) // 2
    results: dict = {}

    # ---- phase: sweep (the main P x eps x w x strategy grid) --------------
    if "sweep" in phases:
        logger.info(f"  [{name}/{clamp_mode}] --- phase: sweep ---")
        df = sweep(
            name, streams, s_values, eps_values, w_values, strategies,
            workers=workers,
            clamp_mode=clamp_mode,
            log_messages=log_messages,
            messages_csv_path=sweep_messages_csv if log_messages else None,
            trials=trials,
            per_pubs=per_pubs,          # per-level subscriptions (every topic level)
            k_ext=getattr(args, "k_ext", 0),
            max_publishers=getattr(args, "max_publishers", None),
            # The sweep keeps only eps & w free; P_min/P_max/Delta_t/K_ext/rho
            # come from the §7.5 grid optimum for each (dataset, strategy, eps).
            grid_config=getattr(args, "grid_config", None),
        )
        df["clamp_mode"] = clamp_mode
        df["eps_clip"] = args.eps_clip if clamp_mode == "dp_released" else 0.0
        # Per-trial individual numbers.
        df.to_csv(os.path.join(dirs["sweep"], "sweep_results.csv"), index=False)
        # Per-dataset aggregate across trials (mean/std over the N realizations),
        # grouped per subscription level/scope.
        if trials > 1:
            agg = _aggregate_over_trials(
                df, ["dataset", "clamp_mode", "sensor", "subscription_level",
                     "scope", "P", "epsilon", "w", "strategy"],
                _TRIAL_METRIC_COLS,
            )
            agg.to_csv(os.path.join(dirs["sweep"], "sweep_results_aggregate.csv"),
                       index=False)
            logger.info(f"  [{name}] sweep aggregate over {trials} trials -> "
                        f"sweep_results_aggregate.csv ({len(agg)} configs)")
        print_summary(df, f"{spec['label']} [clamp_mode={clamp_mode}]")
        if getattr(args, "generate_plots", False):
            plot_results(df, name, dirs["sweep"], streams, workers=workers)
        results["sweep"] = df

    # ---- phase: intro figures (incl. Figure 1 reproduction) ---------------
    if "intro" in phases:
        logger.info(f"  [{name}/{clamp_mode}] --- phase: intro figures ---")
        run_intro_figures(streams, per_pubs, name, dirs["intro"], P_our=4,
                          workers=workers)
        logger.info(f"  [{name}/{clamp_mode}] --- phase: figure1 reproduction ---")
        fig1_df = figure1_reproduction(per_pubs, name, dirs["intro"],
                                       epsilon=1.0, w=w_mid,
                                       workers=workers)
        fig1_df = fig1_df.copy()
        fig1_df["clamp_mode"] = clamp_mode
        results["figure1"] = fig1_df

    # ---- phase: extras (n-weighted spotlight, collusion, K_ext §7.10) ------
    if "extras" in phases:
        logger.info(f"  [{name}/{clamp_mode}] --- phase: n-weighted spotlight ---")
        n_weighted_spotlight(streams, name, dirs["extras"], epsilon=1.0, w=w_mid, P=2)
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

    # ---- phase: hyperparameter tuning (Algorithm 2 greedy + brute) --------
    if "tuning" in phases and per_pubs:
        sensor_name = next(iter(per_pubs))
        pp, B = per_pubs[sensor_name]
        logger.info(f"  [{name}/{clamp_mode}] --- phase: hyperparameter tuning ({sensor_name}) ---")
        tune = tune_hyperparameters(
            pp, B, name, sensor_name, dirs["tuning"],
            epsilon=1.0, w=w_mid,
            strategies=strategies,
            alpha=args.alpha, I_max=args.I_max,
            workers=workers,
            epsilon_count=getattr(args, "epsilon_count", 0.0),
            max_publishers=getattr(args, "max_publishers", None),
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
        keep = set(_shard_sensors(args, list(prepared.streams.keys())))
        entries = []
        for sensor, (agg, cnt, R) in prepared.streams.items():
            if sensor not in keep:
                continue
            pp = prepared.per_pubs[sensor][0]
            entries.append((sensor, pp, R, (agg, cnt)))
        if entries:
            yield ds_name, entries


def _experiment_single_axis_task(task):
    """Shared worker for Experiment B (vary w) and C (vary eps).

    The stream arrays are read from the worker-global cache by ``key`` (set via
    ``_init_streams_worker``) so the large (agg, cnt) arrays are shipped to each
    worker ONCE, not pickled into every task tuple.  The trailing
    (subscription_level, scope) identify which point of the PerCom topic
    hierarchy this subscription is bound to (so B/C test every level)."""
    (ds_name, sensor, key, strategy, P, eps, w,
     clamp_mode, log_messages, experiment_tag, trial, seed,
     subscription_level, scope, rho_ds, p_max) = task
    agg, cnt, R = _WORKER_STREAMS[key]
    # B (vary w) and C (vary eps) keep eps and w free and take every other
    # hyperparameter from the grid optimum: the stream cached at ``key`` was
    # already rebuilt with the grid P_min / K_ext / Delta_t, and the release uses
    # the grid P_min gate, P_max (p_max) and rho (rho_ds) for this (strategy,eps).
    res = run_dp_on_stream(
        agg, cnt, epsilon=eps, window_size=w,
        min_publishers=P, payload_bound=R,
        strategy=strategy, seed=seed,
        rho_split=rho_ds, max_publishers=p_max,
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
        "strategy": strategy, "P": P, "P_max": p_max, "rho_split": rho_ds,
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
        "trial": trial,
        "seed": seed,
        "subscription_level": subscription_level,
        "scope": scope,
    }
    # Log per-release messages only for the first trial (avoid trials x volume).
    if log_messages and trial == 0:
        from message_logger import build_message_rows
        out["_messages"] = build_message_rows(
            res,
            dataset=ds_name, clamp_mode=clamp_mode, sensor=sensor,
            strategy=strategy, P=P, epsilon=eps, w=w,
            payload_bound=R, seed=seed, experiment=experiment_tag,
        )
    return out


# Experiment A (greedy vs brute P-tuning) lives in experiments/greedy_vs_brute.py.
def experiment_A_greedy_vs_brute(*a, **k):
    from experiments.greedy_vs_brute import experiment_A_greedy_vs_brute as _f
    return _f(*a, **k)


def _split_messages_from_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Pop ``_messages`` off every returned task row; return (summary, msgs)."""
    messages: list[dict] = []
    summaries: list[dict] = []
    for r in rows:
        if isinstance(r, dict) and "_messages" in r:
            messages.extend(r.pop("_messages"))
        summaries.append(r)
    return summaries, messages


# Experiments B (vary w) and C (vary eps) live in experiments/window.py and
# experiments/epsilon.py; lazy-delegating stubs keep the core dispatch working.
def experiment_B_vary_w(*a, **k):
    from experiments.window import experiment_B_vary_w as _f
    return _f(*a, **k)


def experiment_C_vary_epsilon(*a, **k):
    from experiments.epsilon import experiment_C_vary_epsilon as _f
    return _f(*a, **k)


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
        rho_split=resolve_rho_split(),
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


# Experiment D (plugin end-to-end path) lives in experiments/plugin_path.py.
def experiment_D_plugin_path(*a, **k):
    from experiments.plugin_path import experiment_D_plugin_path as _f
    return _f(*a, **k)


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

    Aggregation is ONE value per publisher (x_{p,tau} = the mean of publisher
    p's readings inside the window), matching the DP model where the pool P_tau
    contributes one clamped value per publisher.  This keeps the emitted
    (mean, count) mutually consistent -- mean == (sum_p x_{p,tau}) / |P_tau| --
    so the pooled sum the DP engine privatizes is exactly mean * count (it is
    NOT distorted by publishers that emit several readings in a widened window).
    """
    pubs = subset if subset is not None else list(per_pub.keys())
    if not pubs:
        return [], []
    base_dt = max(1, int(base_dt))
    T = len(per_pub[pubs[0]])
    agg, cnt = [], []
    tau = 0
    while tau < T:
        pub_readings: dict[str, list[float]] = {}
        end = tau
        ext = 0
        while True:
            block_end = min(end + base_dt, T)
            for slot in range(end, block_end):
                for p in pubs:
                    v = per_pub[p][slot]
                    if v is not None:
                        pub_readings.setdefault(p, []).append(v)
            end = block_end
            if len(pub_readings) >= P or ext >= k_ext or end >= T:
                break
            ext += 1
        # One value per active publisher, then the pooled per-publisher mean.
        pub_values = [float(np.mean(vs)) for vs in pub_readings.values()]
        agg.append(float(np.mean(pub_values)) if pub_values else 0.0)
        cnt.append(len(pub_values))
        tau = end
    return agg, cnt


# Sec. 7.5 grid search lives in experiments/grid_search.py; lazy-delegating stub.
def grid_search_hyperparameters(*a, **k):
    from experiments.grid_search import grid_search_hyperparameters as _f
    return _f(*a, **k)


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

def _grid_canonical_path(output_dir: str, filename: str = "grid_canonical.json") -> str:
    return os.path.join(output_dir, filename)


def _write_grid_canonical(output_dir: str, best_records: list[dict],
                          filename: str = "grid_canonical.json") -> str:
    """Persist the per-(dataset,clamp,strategy,epsilon) MAE-optimal configs.

    ``filename`` lets a per-epsilon grid shard write a fragment
    (``grid_canonical_eps{eps}.json``) instead of the single canonical file, so
    the grid phase can be split across nodes per epsilon; ``_load_grid_config``
    merges every ``grid_canonical*.json`` fragment in the directory back into one
    lookup for the downstream experiments."""
    path = _grid_canonical_path(output_dir, filename)
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
            "rho_split": (float(r["rho_split"]) if r.get("rho_split") is not None
                          else None),
            "mae": float(r.get("mae")) if r.get("mae") is not None else None,
        })
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    logger.info(f"  wrote canonical grid config ({len(payload)} entries) -> {path}")
    return path


def _merge_grid_trial_fragments(frag_paths: list[str]) -> dict | None:
    """Average MAE across per-trial full-grid fragments and pick each
    (dataset, clamp, strategy, epsilon)'s argmin config.

    Each fragment (one per (epsilon, trial) grid shard) holds every scored
    config with its single-trial MAE; we group by the full config key, average
    MAE over the trials, then select the minimum-mean-MAE config per strategy.
    Returns the same lookup shape as ``_load_grid_config`` so it is a drop-in.
    """
    from collections import defaultdict
    # (dataset, clamp, strategy, eps, P_min, P_max, dt, k_ext) -> [mae, ...]
    buckets: dict = defaultdict(list)
    meta: dict = {}
    n = 0
    for fp in frag_paths:
        try:
            with open(fp) as fh:
                records = json.load(fh)
        except Exception as exc:  # pragma: no cover
            logger.warning(f"could not read grid trial fragment {fp}: {exc}")
            continue
        n += 1
        for r in records:
            mae = r.get("mae")
            if mae is None or not np.isfinite(mae):
                continue
            pmax = r.get("P_max")
            pmax = None if pmax is None or (isinstance(pmax, float) and np.isnan(pmax)) else int(pmax)
            r["P_max"] = pmax   # normalize NaN/null -> None so consumers don't int(NaN)
            rho = r.get("rho_split")
            key = (r.get("dataset"), r.get("clamp_mode"), r.get("strategy"),
                   round(float(r.get("epsilon", 0.0)), 4),
                   int(r.get("P_min")), pmax,
                   int(r.get("delta_t", 1)), int(r.get("k_ext", 0)),
                   round(float(rho), 4) if rho is not None else None)
            buckets[key].append(float(mae))
            meta[key] = r
    if not buckets:
        return None
    # Per (dataset, clamp, strategy, eps): the config with the lowest mean MAE.
    best: dict = {}
    for key, maes in buckets.items():
        ds, clamp, strat, eps = key[0], key[1], key[2], key[3]
        mean_mae = float(np.mean(maes))
        sel = (ds, clamp, strat, eps)
        if sel not in best or mean_mae < best[sel][0]:
            r = dict(meta[key])
            r["mae"] = mean_mae
            r["n_trials"] = len(maes)
            best[sel] = (mean_mae, r)
    cfg = {sel: rec for sel, (_m, rec) in best.items()}
    logger.info(f"merged {n} grid trial fragment(s) -> {len(cfg)} canonical "
                f"entries (MAE averaged over trials)")
    return cfg


def _load_grid_config(path: str | None) -> dict | None:
    """Load the canonical grid config into a lookup dict keyed by
    (dataset, clamp_mode, strategy, round(epsilon, 4)).

    Merges every ``grid_canonical*.json`` in ``path``'s directory, so a grid
    phase that was split across nodes per epsilon (each shard writing a
    ``grid_canonical_eps{eps}.json`` fragment) is recombined transparently — the
    downstream experiments still point ``--use-grid-config`` at the single
    ``grid_canonical.json`` path.

    When the grid was sharded per TRIAL (``grid_trial*.json`` full-grid
    fragments present), those take precedence: MAE is averaged across the trial
    fragments per config and each strategy's optimum is then chosen on the
    seed-averaged MAE (so the §7.5 optimum is robust to single-draw noise)."""
    if not path:
        return None
    import glob
    dirpath = os.path.dirname(path) or "."
    trial_frags = sorted(glob.glob(os.path.join(dirpath, "grid_trial*.json")))
    if trial_frags:
        merged = _merge_grid_trial_fragments(trial_frags)
        if merged:
            return merged
    candidates = sorted(glob.glob(os.path.join(dirpath, "grid_canonical*.json")))
    # Honor the exact path too (covers a non-fragmented single-file write).
    if os.path.exists(path) and path not in candidates:
        candidates.append(path)
    if not candidates:
        return None
    cfg: dict = {}
    n_files = 0
    for fp in candidates:
        try:
            with open(fp) as fh:
                records = json.load(fh)
        except Exception as exc:  # pragma: no cover
            logger.warning(f"could not read grid config {fp}: {exc}")
            continue
        n_files += 1
        for r in records:
            key = (r.get("dataset"), r.get("clamp_mode"), r.get("strategy"),
                   round(float(r.get("epsilon", 0.0)), 4))
            cfg[key] = r
    if not cfg:
        return None
    logger.info(f"loaded canonical grid config ({len(cfg)} entries from "
                f"{n_files} file(s)) in {dirpath}")
    return cfg


def _resolve_params(grid_config: dict | None, dataset: str, clamp_mode: str,
                    strategy: str, epsilon: float, defaults: dict) -> dict:
    """Return {P_min, P_max, k_ext, delta_t, rho_split} from the canonical grid
    config for this (dataset, clamp, strategy, epsilon), else the supplied
    defaults.  rho_split always resolves (grid optimum -> default rho) so every
    downstream experiment runs at the grid-selected split.

    Lookup order, so that EVERY (w, epsilon, strategy) the experiments test maps
    to a grid optimum (Thesis §7.6: the per-dataset optima are frozen as the
    canonical config for the downstream experiments):
      1. exact (dataset, clamp, strategy, epsilon);
      2. same (dataset, clamp, strategy) at the NEAREST grid epsilon -- so the
         epsilon-sweep (Exp C) values that lie between/outside the grid's epsilon
         choices still use that strategy's optimum;
      3. any (dataset, clamp) entry at the nearest epsilon (strategy-agnostic);
      4. the supplied defaults."""
    out = dict(defaults)
    out.setdefault("rho_split", resolve_rho_split())
    if grid_config is None:
        return out
    eps = round(float(epsilon), 4)
    rec = grid_config.get((dataset, clamp_mode, strategy, eps))
    if rec is None:
        # Nearest-epsilon for this strategy (the grid is sampled at a few epsilon
        # choices; the optimum is stable enough that the closest is the right
        # canonical to reuse for an off-grid epsilon).
        same_strat = [(e, r) for (d, c, s, e), r in grid_config.items()
                      if d == dataset and c == clamp_mode and s == strategy]
        if same_strat:
            rec = min(same_strat, key=lambda er: abs(er[0] - eps))[1]
    if rec is None:
        # Strategy-agnostic: nearest epsilon for any entry of this (dataset, clamp).
        any_strat = [(e, r) for (d, c, _s, e), r in grid_config.items()
                     if d == dataset and c == clamp_mode]
        if any_strat:
            rec = min(any_strat, key=lambda er: abs(er[0] - eps))[1]
    if rec is None:
        return out
    for k_cfg, k_out in (("P_min", "P_min"), ("P_max", "P_max"),
                         ("k_ext", "k_ext"), ("delta_t", "delta_t"),
                         ("rho_split", "rho_split")):
        if rec.get(k_cfg) is not None:
            out[k_out] = rec[k_cfg]
    return out


# Experiment F (incremental-module ablation) lives in experiments/ablation.py;
# lazy-delegating stub keeps it callable from the core dispatch.
def ablation_experiment(*a, **k):
    from experiments.ablation import ablation_experiment as _f
    return _f(*a, **k)


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

# Experiment G (overhead / privacy-utility) lives in experiments/overhead.py.
def overhead_experiment(*a, **k):
    from experiments.overhead import overhead_experiment as _f
    return _f(*a, **k)


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

# Experiment H (average-case utility) lives in experiments/average_case.py.
def average_case_utility_experiment(*a, **k):
    from experiments.average_case import average_case_utility_experiment as _f
    return _f(*a, **k)


# Experiment I (subscriptions at every topic-hierarchy level) lives in
# experiments/subscription_levels.py.  A lazy-delegating stub keeps it callable
# from this core module's API without a top-level circular import.
def subscription_levels_experiment(*a, **k):
    from experiments.subscription_levels import subscription_levels_experiment as _f
    return _f(*a, **k)


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
                args, trials=max(1, getattr(args, "trials", 1)),
            )
        grid_config = getattr(args, "grid_config", None)
        trials = max(1, getattr(args, "trials", 1))
        if "F" in which:
            ablation_experiment(
                datasets, clamp_mode,
                os.path.join(output_dir, "cross_dataset", clamp_mode),
                args, epsilon=1.0, w=8, P=getattr(args, "ablation_P", 3),
                k_ext=getattr(args, "k_ext", 3),
                epsilon_count=getattr(args, "epsilon_count", 0.0),
                grid_config=grid_config, trials=trials,
            )
        if "G" in which:
            overhead_experiment(
                datasets, clamp_mode,
                os.path.join(output_dir, "cross_dataset", clamp_mode),
                args, epsilon=1.0, w=8, P=getattr(args, "ablation_P", 3),
                epsilon_count=getattr(args, "epsilon_count", 0.0),
                grid_config=grid_config, trials=trials,
            )
        if "H" in which:
            average_case_utility_experiment(
                datasets, clamp_mode,
                os.path.join(output_dir, "cross_dataset", clamp_mode),
                args, epsilon=1.0, w=8, P=getattr(args, "ablation_P", 3),
                epsilon_count=getattr(args, "epsilon_count", 0.5),
                grid_config=grid_config, trials=trials,
            )


# ═════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════

CLAMP_MODES = ["static", "dp_released"]


def _run_grid_search_block(args, targets, clamp_modes, eps_values, strategies,
                           canonical_filename="grid_canonical.json",
                           trial=None, full_fragment=None):
    """Paper Sec. 7.5 grid search over (P_min x P_max x Delta_t x K_ext) scored
    by MAE, per dataset/clamp/strategy/epsilon.  Writes per-(dataset,eps) grids
    and the consolidated canonical config.  Returns the written path.

    ``canonical_filename`` is overridden to ``grid_canonical_eps{eps}.json`` by a
    per-epsilon grid shard (``--grid-eps``) so the grid phase can be split across
    nodes; ``_load_grid_config`` merges the fragments back together.

    ``trial`` + ``full_fragment``: when set (the per-trial grid shards), the
    block scores at the trial's own noise seed and writes the FULL per-config
    grid (every strategy x P_min x P_max x dt x k_ext, with its MAE) to
    ``full_fragment`` instead of a pre-selected canonical.  ``_load_grid_config``
    then averages MAE across the trial fragments per config and picks each
    strategy's optimum -- so the §7.5 optimum is chosen on seed-averaged MAE and
    the trials run as independent cluster tasks.
    """
    # Per-trial shards reseed so the trials are independent noise draws.
    seed = 77 if trial is None else 77 + 1000 * int(trial)
    p_min_grid = [1, 2, 3, 4, 6]
    p_max_grid = [None, 4, 8, 16]
    dt_grid = [1, 2, 4]
    k_ext_grid = [0, 2, 4]
    # rho_tau candidates for the aggregate stream element (Definition: Aggregate
    # Stream Element).  6 values anchored at the definition's error-equalizing
    # sqrt(R)/(5 sqrt(R)) = 0.2, plus a smaller 0.1 (more budget to the sum) and
    # a span toward sum-starving values: small rho starves the count (noisy
    # gate), large rho starves the sum (noisy magnitude), so the MAE optimum is
    # interior.  The selected best rho is written into grid_canonical.json and
    # consumed by downstream experiments.
    rho_grid = [0.1, 0.2, 0.4, 0.5, 0.6, 0.8]
    # --quick is meant to validate the whole path fast; the full §7.5 grid is
    # 5x4x3x3x6 = 1080 combos x strategies x eps (thousands of DP passes per
    # dataset) and is the slowest shard.  Shrink it under --quick so the smoke
    # test stays ~1 min; the FULL run still uses the complete grid.
    if getattr(args, "quick", False):
        p_min_grid = [1, 3, 6]
        p_max_grid = [None, 8]
        dt_grid = [1, 2]
        k_ext_grid = [0, 2]
        rho_grid = [0.1, 0.2, 0.5]
    # Per-rho cluster shard (--grid-rho): restrict this task to one rho candidate
    # so the rho sweep fans out one SLURM task per rho; the full-grid fragments
    # are min-merged across rho downstream (see grid_search.main).
    grid_rho = getattr(args, "grid_rho", None)
    if grid_rho is not None:
        rho_grid = [float(grid_rho)]
    canonical_records: list[dict] = []
    full_records: list[dict] = []
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
                # The per-shard diagnostic CSVs are written into the SHARED grid
                # dir, so the tag must be unique per (eps, trial, rho) shard --
                # otherwise concurrent rho shards on different nodes clobber each
                # other's *_gridsearch.csv.  (The canonical merge uses the
                # rho-tagged JSON fragments, not these CSVs.)
                tag = f"{first_sensor}_eps{eps}"
                if trial is not None:
                    tag += f"_t{trial}"
                if grid_rho is not None:
                    tag += f"_rho{grid_rho}"
                df = grid_search_hyperparameters(
                    pp, B, name, tag, grid_dir,
                    epsilon=eps, w=8, strategies=strategies,
                    p_min_grid=p_min_grid, p_max_grid=p_max_grid,
                    dt_grid=dt_grid, k_ext_grid=k_ext_grid, rho_grid=rho_grid,
                    seed=seed, workers=args.workers,
                )
                if full_fragment is not None:
                    # Per-trial mode: keep EVERY scored config (averaged + argmin
                    # downstream in _load_grid_config across the trial fragments).
                    if df is not None and not df.empty:
                        for _i, r in df.iterrows():
                            rec = r.to_dict()
                            rec["dataset"] = name
                            rec["clamp_mode"] = clamp_mode
                            rec["epsilon"] = eps
                            rec["trial"] = int(trial)
                            full_records.append(rec)
                    continue
                best_csv = os.path.join(grid_dir, f"{name}_{tag}_gridsearch_best.csv")
                if os.path.exists(best_csv):
                    bdf = pd.read_csv(best_csv)
                    for _i, r in bdf.iterrows():
                        rec = r.to_dict()
                        rec["dataset"] = name
                        rec["clamp_mode"] = clamp_mode
                        canonical_records.append(rec)
    if full_fragment is not None:
        path = _grid_canonical_path(args.output_dir, full_fragment)
        with open(path, "w") as fh:
            json.dump(full_records, fh)
        logger.info(f"  wrote per-trial grid fragment ({len(full_records)} configs) -> {path}")
        return path
    return _write_grid_canonical(args.output_dir, canonical_records,
                                 filename=canonical_filename)



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
