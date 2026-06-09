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

import itertools
import os

import numpy as np
import pandas as pd

from experiments import _common as C
from experiments import engine as core


def _sweep_task(task):
    """One sweep combo (level-subscription, config, trial)."""
    (i, dataset_name, sensor, key, level, scope, P, eps, w, strat,
     clamp_mode, log_messages, trial) = task
    aggregates, pub_counts, B = core._WORKER_STREAMS[key]
    result = core.run_dp_on_stream(
        aggregates, pub_counts, epsilon=eps, window_size=w, min_publishers=P,
        payload_bound=B, strategy=strat, seed=i)
    m = result["metrics"]
    avg_n = float(np.mean([n for n in pub_counts if n > 0])) if any(pub_counts) else 0.0
    out = {
        "dataset": dataset_name, "clamp_mode": clamp_mode, "sensor": sensor,
        "subscription_level": level, "scope": scope,
        "P": P, "epsilon": eps, "w": w, "strategy": strat, "trial": trial, "seed": i,
        "mae": m["mae"], "rmse": m["rmse"], "relative_error": m["relative_error"],
        "normalized_mae": m["normalized_mae"], "kl_divergence": m["kl_divergence"],
        "kl_global_utility": m["kl_global_utility"], "release_rate": m["release_rate"],
        "deferrals": m["deferrals"], "attribution_advantage": m["attribution_advantage"],
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
          messages_csv_path=None, trials=1, per_pubs=None, k_ext=0) -> "pd.DataFrame":
    """Per-level sweep.  ``per_pubs`` (sensor -> (per_pub, B)) drives the
    subscription levels; if None, falls back to the per-sensor pooled
    ``streams`` (single level = the whole type)."""
    # Build the stream universe keyed by (sensor, level, scope).
    streams_by_key, level_index = {}, []
    if per_pubs:
        for sensor, (per_pub, B) in per_pubs.items():
            for (L, scope, agg, cnt, _np) in core.level_subscription_streams(
                    dataset_name, sensor, per_pub, k_ext=k_ext):
                key = f"{sensor}|{L}|{scope}"
                streams_by_key[key] = (agg, cnt, B)
                level_index.append((sensor, key, L, scope))
    else:
        for sensor, (agg, cnt, B) in streams.items():
            key = f"{sensor}|1|{sensor}"
            streams_by_key[key] = (agg, cnt, B)
            level_index.append((sensor, key, 1, sensor))

    tasks, i = [], 0
    for (sensor, key, L, scope) in level_index:
        for (P, eps, w, strat) in itertools.product(
                s_values, epsilon_values, w_values, strategies):
            for trial in range(max(1, trials)):
                tasks.append((i, dataset_name, sensor, key, L, scope, P, eps, w,
                              strat, clamp_mode, log_messages, trial))
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
    core.logger.info(f"  [{dataset_name}] sweep: {len(level_index)} level-subscriptions "
                     f"x configs x {max(1, trials)} trial(s) = {len(tasks)} runs")
    return pd.DataFrame(summary_rows)


def main():
    args = C.resolve(C.make_parser("Per-level parameter sweep (Sec. 7).").parse_args())
    C.run_main_phase(args, {"sweep"})


if __name__ == "__main__":
    main()
