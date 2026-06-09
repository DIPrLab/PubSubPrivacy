#!/usr/bin/env python3
"""Experiment G: overhead / privacy-utility comparison (paper Sec. 7.9).

Compares, per dataset and at EVERY level of the PerCom topic hierarchy: classic
(no privacy), ldp (P_min=1 local/input DP, lambda=R*w/eps), per_type_wevent (one
stream per sensor type, Extreme 1.1), and ours (clamped w-event DP with
P-allocation).  Reports NMAE / KL / release rate / attribution advantage + a
compute-overhead proxy (ms/element, eps_count).  True broker throughput/latency
is the live Experiment E.  Every (sensor x level x approach x trial) task fans
out over the worker pool; cluster-sharded per dataset.

Run: python -m experiments.overhead --dataset all --trials 6
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from experiments import _common as C
from experiments import engine as core


def _level_entries(ds_name, sensor, per_pub, k_ext):
    """Per-level subscription entries for one sensor: a list of
    ``(level, scope, sub_per_pub, agg, cnt)`` over the PerCom topic tree.

    ``sub_per_pub`` is the per-publisher subset bound to that subtree (needed by
    the LDP baseline, which perturbs each publisher's own input), while
    ``(agg, cnt)`` is the pooled stream the output-DP approaches consume.
    """
    groups = core._topic_level_groups(ds_name, sensor, list(per_pub.keys()))
    if not groups:  # no topic mapping (synthetic): single whole-type scope
        agg, cnt = core._adaptive_interval_rebuild(per_pub, 1, k_ext)
        return [(1, sensor, dict(per_pub), agg, cnt)]
    out = []
    for L in sorted(groups):
        for prefix, members in groups[L].items():
            sub = {p: per_pub[p] for p in members}
            agg, cnt = core._adaptive_interval_rebuild(per_pub, 1, k_ext, subset=members)
            out.append((L, prefix, sub, agg, cnt))
    return out


def _overhead_task(task):
    """One (level-subscription, approach, trial) overhead measurement."""
    (key, approach, ds_name, sensor, level, scope, epsilon, w, eps_count,
     strategy, P_ds, P_max_ds, trial, seed_t) = task
    sub_per_pub, agg, cnt, B = core._WORKER_STREAMS[key]
    extra: dict = {}
    t0 = time.perf_counter()
    if approach == "classic":
        m = {"normalized_mae": 0.0, "mae": 0.0, "kl_divergence": 0.0,
             "release_rate": 1.0, "attribution_advantage": float("nan")}
        compute_s = 0.0
    elif approach == "ldp":
        m = core.run_ldp_on_per_pub(sub_per_pub, B, epsilon, w, seed=seed_t)["metrics"]
        compute_s = time.perf_counter() - t0
        extra = {"noise_scale": m.get("ldp_noise_scale")}
    elif approach == "per_type_wevent":
        m = core.run_dp_on_stream(agg, cnt, epsilon=epsilon, window_size=w,
                                  min_publishers=1, payload_bound=B,
                                  strategy="uniform", seed=seed_t)["metrics"]
        compute_s = time.perf_counter() - t0
    else:  # "ours"
        m = core.run_dp_on_stream(agg, cnt, epsilon=epsilon, window_size=w,
                                  min_publishers=P_ds, payload_bound=B,
                                  strategy=strategy, seed=seed_t,
                                  epsilon_count=eps_count,
                                  max_publishers=P_max_ds)["metrics"]
        compute_s = time.perf_counter() - t0
        extra = {"strategy": strategy, "P_min": P_ds, "P_max": P_max_ds}
    row = {
        "dataset": ds_name, "sensor": sensor, "clamp_mode": None,
        "subscription_level": level, "scope": scope, "approach": approach,
        "epsilon": epsilon, "w": w, "P": P_ds, "payload_bound": B,
        "trial": trial, "seed": seed_t,
        "normalized_mae": m.get("normalized_mae"), "mae": m.get("mae"),
        "kl_divergence": m.get("kl_divergence"),
        "release_rate": m.get("release_rate"),
        "attribution_advantage": m.get("attribution_advantage"),
        "eps_count_spent": m.get("eps_count_spent", 0.0),
        "compute_ms_per_element": 1000.0 * compute_s / max(1, len(agg)),
    }
    row.update(extra)
    return row


def overhead_experiment(
    datasets, clamp_mode, output_dir, args, *,
    epsilon: float = 1.0, w: int = 8, P: int = 3, epsilon_count: float = 0.0,
    our_strategy: str = "p_gated_uniform", seed: int = 77,
    grid_config: dict | None = None, trials: int = 1,
) -> "pd.DataFrame":
    """Sec. 7.9 overhead / privacy-utility comparison (offline), at every topic
    level.  'ours' uses the Sec. 7.5 grid-optimal (P_min, P_max) when
    ``grid_config`` is supplied; each noisy approach is repeated over ``trials``
    seeds (per-trial + per-(dataset, level, scope, approach) aggregate)."""
    workers = core._default_workers(getattr(args, "workers", None))
    k_ext = getattr(args, "k_ext", 0)
    approaches = ("classic", "ldp", "per_type_wevent", "ours")
    streams_by_key: dict = {}
    tasks = []
    for ds_name in datasets:
        prepared = core.prepare_dataset(
            ds_name, clamp_mode=clamp_mode, eps_clip=args.eps_clip,
            seed=args.seed, max_rows=core._dataset_max_rows(ds_name, args))
        if prepared is None or not prepared.per_pubs:
            continue
        prm = core._resolve_params(grid_config, ds_name, clamp_mode, our_strategy,
                                   epsilon, {"P_min": P, "P_max": None})
        P_ds, P_max_ds = prm["P_min"], prm["P_max"]
        for sensor in [s for s in prepared.spec["sensors"]
                       if s in prepared.per_pubs and s in prepared.streams]:
            per_pub, B = prepared.per_pubs[sensor]
            for (L, scope, sub_per_pub, agg, cnt) in _level_entries(
                    ds_name, sensor, per_pub, k_ext):
                key = f"{ds_name}|{sensor}|{L}|{scope}"
                streams_by_key[key] = (sub_per_pub, agg, cnt, B)
                for approach in approaches:
                    for trial in range(max(1, trials)):
                        tasks.append((key, approach, ds_name, sensor, L, scope,
                                      epsilon, w, epsilon_count, our_strategy,
                                      P_ds, P_max_ds, trial, seed + 1000 * trial))
    rows = core._run_parallel_tasks(
        tasks, _overhead_task, workers=workers,
        initializer=core._init_streams_worker, initargs=(streams_by_key,),
        progress_label=f"  [exp G {clamp_mode}]", progress_every=200)
    for r in rows:
        r["clamp_mode"] = clamp_mode
    exp_dir = os.path.join(output_dir, "experiments", "G_overhead")
    os.makedirs(exp_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(exp_dir, "experiment_G_overhead.csv"), index=False)
    if max(1, trials) > 1 and not df.empty:
        core._aggregate_over_trials(
            df, ["dataset", "sensor", "clamp_mode", "subscription_level",
                 "scope", "approach"],
            core._TRIAL_METRIC_COLS + ["compute_ms_per_element", "eps_count_spent"],
        ).to_csv(os.path.join(exp_dir, "experiment_G_overhead_aggregate.csv"),
                 index=False)
    core.logger.info(f"  Experiment G (overhead) wrote {len(df)} rows "
                     f"({len(streams_by_key)} level-subscriptions) -> {exp_dir}")
    return df


def main():
    args = C.resolve(C.make_parser(
        "Experiment G: overhead / privacy-utility comparison (Sec. 7.9)."
    ).parse_args())
    for clamp_mode in args._clamp_modes:
        overhead_experiment(
            args._targets, clamp_mode, C.cross_dir(args, clamp_mode), args,
            epsilon=1.0, w=8, P=args.ablation_P, epsilon_count=args.epsilon_count,
            grid_config=args.grid_config, trials=max(1, args.trials))


if __name__ == "__main__":
    main()
