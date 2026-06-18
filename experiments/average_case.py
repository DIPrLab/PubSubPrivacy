#!/usr/bin/env python3
"""Experiment H: average-case utility (paper Sec. 7.11 / 6.6).

Relates the range-compatible publisher fraction |P_R|/|P| (Definition 6.3) and
the topic-hierarchy depth h to realized utility, evaluated at EVERY level of the
PerCom topic tree, per dataset.  Uses the Sec. 7.5 grid-optimal (P_min, P_max)
when a grid_canonical.json is present; repeated over ``trials`` seeds (per-trial
rows + per-(dataset, sensor, level, scope) aggregate).  All datasets, every
(sensor x level x trial) task fanned out over the worker pool.

Run: python -m experiments.average_case --dataset all --trials 6
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from experiments import _common as C
from experiments import engine as core


def _topic_depth(spec, per_pubs) -> int:
    """Number of levels in the dataset's PerCom topic tree (max '/'-segment
    count over a sample of publisher topics)."""
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


def _range_compatible_fraction(spec, sensors, ref_sensor) -> float:
    """Fraction of the dataset's sensor types range-compatible with the
    reference sensor (hull width <= R), a proxy for |P_R|/|P| (Definition 6.3)."""
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
        if max(b_ref, b) - min(a_ref, a) <= R + 1e-9:
            compatible += 1
    return compatible / total if total else float("nan")


def _avg_case_task(task):
    """One (level-subscription, trial) average-case utility measurement."""
    (key, ds_name, sensor, level, scope, frac, depth_h, n_types,
     epsilon, w, rho_ds, strategy, P_ds, P_max_ds, trial, seed_t) = task
    agg, cnt, B = core._WORKER_STREAMS[key]
    m = core.run_dp_on_stream(
        agg, cnt, epsilon=epsilon, window_size=w, min_publishers=P_ds,
        payload_bound=B, strategy=strategy, seed=seed_t,
        rho_split=rho_ds, max_publishers=P_max_ds)["metrics"]
    return {
        "dataset": ds_name, "ref_sensor": sensor, "clamp_mode": None,
        "subscription_level": level, "scope": scope,
        "range_compatible_fraction": frac, "topic_hierarchy_depth_h": depth_h,
        "n_sensor_types": n_types, "avg_n_tau": float(np.mean(cnt)) if cnt else float("nan"),
        "trial": trial, "seed": seed_t,
        "normalized_mae": m.get("normalized_mae"),
        "kl_divergence": m.get("kl_divergence"),
        "release_rate": m.get("release_rate"),
        "eps_count_spent": m.get("eps_count_spent", 0.0),
        "dp_count_releases": m.get("dp_count_releases", 0),
        "epsilon": epsilon, "w": w, "P": P_ds, "P_max": P_max_ds,
        "rho_split": rho_ds,
    }


def average_case_utility_experiment(
    datasets, clamp_mode, output_dir, args, *,
    epsilon: float = 1.0, w: int = 8, P: int = 3, epsilon_count: float = 0.5,
    strategy: str = "p_gated_uniform", seed: int = 77,
    grid_config: dict | None = None, trials: int = 1,
) -> "pd.DataFrame":
    workers = core._default_workers(getattr(args, "workers", None))
    k_ext = getattr(args, "k_ext", 0)
    streams_by_key: dict = {}
    tasks = []
    for ds_name in datasets:
        prepared = core.prepare_dataset(
            ds_name, clamp_mode=clamp_mode, eps_clip=args.eps_clip,
            seed=args.seed, max_rows=core._dataset_max_rows(ds_name, args))
        if prepared is None or not prepared.per_pubs:
            continue
        sensors = [s for s in prepared.spec["sensors"] if s in prepared.streams]
        if not sensors:
            continue
        depth_h = _topic_depth(prepared.spec, prepared.per_pubs)
        prm = core._resolve_params(grid_config, ds_name, clamp_mode, strategy,
                                   epsilon, {"P_min": P, "P_max": None})
        P_ds, P_max_ds = prm["P_min"], prm["P_max"]
        rho_ds = prm["rho_split"]   # grid-selected split for this (ds, strategy, eps)
        # Every sensor as the subscription reference, at every topic level.
        for sensor in [s for s in sensors if s in prepared.per_pubs]:
            frac = _range_compatible_fraction(prepared.spec, sensors, sensor)
            per_pub, B = prepared.per_pubs[sensor]
            for (L, scope, agg, cnt, _np) in core.level_subscription_streams(
                    ds_name, sensor, per_pub, k_ext=k_ext):
                key = f"{ds_name}|{sensor}|{L}|{scope}"
                streams_by_key[key] = (agg, cnt, B)
                for trial in range(max(1, trials)):
                    tasks.append((key, ds_name, sensor, L, scope, frac, depth_h,
                                  len(sensors), epsilon, w, rho_ds, strategy,
                                  P_ds, P_max_ds, trial, seed + 1000 * trial))
    rows = core._run_parallel_tasks(
        tasks, _avg_case_task, workers=workers,
        initializer=core._init_streams_worker, initargs=(streams_by_key,),
        progress_label=f"  [exp H {clamp_mode}]", progress_every=200)
    for r in rows:
        r["clamp_mode"] = clamp_mode
    exp_dir = os.path.join(output_dir, "experiments", "H_average_case")
    os.makedirs(exp_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(exp_dir, "experiment_H_average_case.csv"), index=False)
    if max(1, trials) > 1 and not df.empty:
        core._aggregate_over_trials(
            df, ["dataset", "ref_sensor", "clamp_mode", "subscription_level",
                 "scope", "range_compatible_fraction", "topic_hierarchy_depth_h"],
            core._TRIAL_METRIC_COLS,
        ).to_csv(os.path.join(exp_dir, "experiment_H_average_case_aggregate.csv"),
                 index=False)
    core.logger.info(f"  Experiment H (average-case utility) wrote {len(df)} rows "
                     f"({len(streams_by_key)} level-subscriptions) -> {exp_dir}")
    return df


def main():
    args = C.resolve(C.make_parser(
        "Experiment H: average-case utility (Sec. 7.11)."
    ).parse_args())
    for clamp_mode in args._clamp_modes:
        average_case_utility_experiment(
            args._targets, clamp_mode, C.cross_dir(args, clamp_mode), args,
            epsilon=1.0, w=8, P=args.ablation_P,
            epsilon_count=(args.epsilon_count or 0.5),
            grid_config=args.grid_config, trials=max(1, args.trials))


if __name__ == "__main__":
    main()
