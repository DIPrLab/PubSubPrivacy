#!/usr/bin/env python3
"""Main parameter sweep: strategy x P x epsilon x w (paper Sec. 7), evaluated at
EVERY subscription level of the PerCom topic tree, per dataset.

Because per-level x strategy x P x eps x w x trials is large, this experiment is
sharded per (dataset x strategy) on the cluster (gen_jobs emits one sweep shard
per strategy) and fans every task out over the worker pool.  Pass --strategies
to restrict to one strategy (the cluster does this per shard).

Run: python -m experiments.sweep --dataset all --strategies p_gated_ba --trials 6
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from experiments import _common as C
from experiments import engine as core


def _sweep_task(task):
    """One sweep point (level-subscription, strategy, eps, w, trial).  Only eps
    and w are free; P_min/P_max/Delta_t/K_ext/rho come from the grid optimum for
    (dataset, strategy, eps) -- the stream at ``key`` was already rebuilt with the
    grid P_min/K_ext/Delta_t, and P_min/P_max/rho are passed through here."""
    (i, dataset_name, sensor, key, level, scope, P, eps, w, strat,
     clamp_mode, log_messages, trial, rho_ds, p_max) = task
    aggregates, pub_counts, B = core._WORKER_STREAMS[key]
    result = core.run_dp_on_stream(
        aggregates, pub_counts, epsilon=eps, window_size=w, min_publishers=P,
        payload_bound=B, strategy=strat, seed=i,
        rho_split=rho_ds, max_publishers=p_max)
    m = result["metrics"]
    avg_n = float(np.mean([n for n in pub_counts if n > 0])) if any(pub_counts) else 0.0
    out = {
        "dataset": dataset_name, "clamp_mode": clamp_mode, "sensor": sensor,
        "subscription_level": level, "scope": scope,
        "P": P, "P_max": p_max, "rho_split": rho_ds,
        "epsilon": eps, "w": w, "strategy": strat, "trial": trial, "seed": i,
        "mae": m["mae"], "rmse": m["rmse"], "relative_error": m["relative_error"],
        "normalized_mae": m["normalized_mae"], "kl_divergence": m["kl_divergence"],
        "kl_global_utility": m["kl_global_utility"], "release_rate": m["release_rate"],
        "deferrals": m["deferrals"], "attribution_advantage": m["attribution_advantage"],
        # Composed DP cost: eps_count spent on the DP publisher counts (charged
        # inside the window budget eps); 0 for the Kellaris baselines.
        "eps_count_spent": m.get("eps_count_spent", 0.0),
        "dp_count_releases": m.get("dp_count_releases", 0),
        "noise_scale_theoretical": B * w / (max(avg_n, 1.0) * eps),
        "payload_bound": B, "num_timestamps": len(aggregates),
        "avg_publishers": float(np.mean(pub_counts)),
    }
    # Log messages only for trial 0 at the root level (avoid level x trial blow-up).
    if log_messages and trial == 0 and level == 1:
        from message_logger import build_message_rows
        out["_messages"] = build_message_rows(
            result, dataset=dataset_name, clamp_mode=clamp_mode, sensor=sensor,
            strategy=strat, P=P, epsilon=eps, w=w, payload_bound=B, seed=i,
            experiment="sweep")
    return out


def sweep(dataset_name, streams, s_values, epsilon_values, w_values, strategies,
          workers=1, clamp_mode="static", log_messages=True,
          messages_csv_path=None, trials=1, per_pubs=None, k_ext=0,
          epsilon_count=0.0, max_publishers=None,
          grid_config=None) -> "pd.DataFrame":
    """Per-level sweep over the two FREE axes (epsilon, w) x strategy, at EVERY
    point of the PerCom topic hierarchy.  P_min, P_max, Delta_t, K_ext and rho
    come from the §7.5 grid optimum for each (dataset, strategy, eps) -- so the
    legacy ``s_values`` (P sweep) is no longer a sweep axis; each level's stream
    is rebuilt at the grid P_min/K_ext/Delta_t.  ``per_pubs`` (sensor ->
    (per_pub, B)) drives the levels; without it we fall back to the pre-pooled
    ``streams`` (single whole-type level, release-time grid params only)."""
    # Build the stream universe keyed by (sensor, level, scope, strategy, eps):
    # the grid P_min/K_ext/Delta_t shape the stream, so it differs per (strategy,
    # eps).  Each task then sweeps w (and trials) at the grid release params.
    streams_by_key, task_index = {}, []
    if per_pubs:
        for sensor, (per_pub, B) in per_pubs.items():
            for strat in strategies:
                for eps in epsilon_values:
                    for (L, scope, agg, cnt, _np, prm) in core.grid_level_streams(
                            dataset_name, sensor, per_pub, clamp_mode, grid_config,
                            strat, eps, default_k_ext=k_ext):
                        key = f"{sensor}|{L}|{scope}|{strat}|{eps}"
                        streams_by_key[key] = (agg, cnt, B)
                        task_index.append((sensor, key, L, scope, strat, eps, prm))
    else:
        for sensor, (agg, cnt, B) in streams.items():
            for strat in strategies:
                for eps in epsilon_values:
                    prm = core._resolve_params(
                        grid_config, dataset_name, clamp_mode, strat, eps,
                        {"P_min": 1, "P_max": max_publishers})
                    key = f"{sensor}|1|{sensor}|{strat}|{eps}"
                    streams_by_key[key] = (agg, cnt, B)
                    task_index.append((sensor, key, 1, sensor, strat, eps, prm))

    tasks, i = [], 0
    for (sensor, key, L, scope, strat, eps, prm) in task_index:
        for w in w_values:
            for trial in range(max(1, trials)):
                tasks.append((i, dataset_name, sensor, key, L, scope,
                              prm["P_min"], eps, w, strat, clamp_mode,
                              log_messages, trial, prm["rho_split"], prm["P_max"]))
                i += 1
    rows = core._run_parallel_tasks(
        tasks, _sweep_task, workers=workers,
        initializer=core._init_streams_worker, initargs=(streams_by_key,),
        progress_label=f"  [{dataset_name} sweep]", progress_every=200)
    all_messages, summary_rows = [], []
    for r in rows:
        if isinstance(r, dict) and "_messages" in r:
            all_messages.extend(r.pop("_messages"))
        summary_rows.append(r)
    if log_messages and messages_csv_path and all_messages:
        from message_logger import write_messages_csv
        n = write_messages_csv(all_messages, messages_csv_path, append=False)
        core.logger.info(f"  [{dataset_name}] sweep messages -> {messages_csv_path} ({n} rows)")
    core.logger.info(f"  [{dataset_name}] sweep: {len(task_index)} (level x strategy x eps) "
                     f"x w x {max(1, trials)} trial(s) = {len(tasks)} runs")
    return pd.DataFrame(summary_rows)


def main():
    args = C.resolve(C.make_parser("Per-level parameter sweep (Sec. 7).").parse_args())
    C.run_main_phase(args, {"sweep"})


if __name__ == "__main__":
    main()
