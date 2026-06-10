#!/usr/bin/env python3
"""Experiment I: subscriptions at EVERY topic-hierarchy level (paper Sec. 6.6).

For every dataset and sensor type, evaluates the mechanism's utility (clamped
w-event DP with P-allocation) for a subscription bound at each point of the
PerCom topic tree (domain root -> ... -> per-publisher leaf), in parallel.  Each
(dataset, sensor, level, subtree) is one subscription whose pooled stream is
scored.  Consumes the Sec. 7.5 grid-optimal params when a grid_canonical.json is
present.

The DP engine, dataset prep, parallel pool, and the topic-level primitive live
in the shared core (``run_experiment``); the experiment driver + its pool worker
live here.

Run: python -m experiments.subscription_levels --dataset all --trials 6
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from experiments import _common as C
from experiments import engine as core


# Worker-global stream cache: id -> (agg, cnt, B).  Filled per worker process
# via the pool initializer so task tuples stay small (just the stream id).
_WORKER_LEVEL_STREAMS: dict | None = None


def _init_level_streams_worker(streams_by_id):
    global _WORKER_LEVEL_STREAMS
    _WORKER_LEVEL_STREAMS = streams_by_id


def _level_sub_task(task):
    sid, epsilon, w, P, strategy, rho_ds, p_max, trial, seed, meta = task
    agg, cnt, B = _WORKER_LEVEL_STREAMS[sid]
    res = core.run_dp_on_stream(
        agg, cnt, epsilon=epsilon, window_size=w, min_publishers=P,
        payload_bound=B, strategy=strategy, seed=seed,
        rho_split=rho_ds, max_publishers=p_max,
    )
    m = res["metrics"]
    return {
        **meta, "epsilon": epsilon, "w": w, "P": P, "strategy": strategy,
        "trial": trial, "seed": seed,
        "normalized_mae": m["normalized_mae"], "mae": m["mae"],
        "kl_divergence": m["kl_divergence"], "release_rate": m["release_rate"],
        "attribution_advantage": m["attribution_advantage"],
        "avg_n_tau": float(np.mean(cnt)) if cnt else float("nan"),
        "eps_count_spent": m.get("eps_count_spent", 0.0),
    }


def subscription_levels_experiment(
    datasets, clamp_mode, output_dir, args, *,
    epsilon: float = 1.0, w: int = 8, strategy: str = "p_gated_uniform",
    seed: int = 77, grid_config: dict | None = None, trials: int = 1,
) -> "pd.DataFrame":
    """Evaluate utility for a subscription bound at EVERY topic-hierarchy level,
    for every dataset and sensor type, in parallel."""
    streams_by_id: dict = {}
    metas: list[dict] = []
    P_default = getattr(args, "ablation_P", 3)
    k_ext = getattr(args, "k_ext", 0)
    for ds in datasets:
        prepared = core.prepare_dataset(
            ds, clamp_mode=clamp_mode, eps_clip=args.eps_clip,
            seed=args.seed, max_rows=core._dataset_max_rows(ds, args))
        if prepared is None or not prepared.per_pubs:
            continue
        for sensor in [s for s in prepared.spec["sensors"]
                       if s in prepared.per_pubs]:
            per_pub, B = prepared.per_pubs[sensor]
            prm = core._resolve_params(grid_config, ds, clamp_mode, strategy,
                                       epsilon, {"P_min": P_default, "P_max": None})
            P_ds, P_max_ds = prm["P_min"], prm["P_max"]
            rho_ds = prm["rho_split"]   # grid-selected split for this (ds, strategy, eps)
            for (L, scope, agg, cnt, npub) in core.level_subscription_streams(
                    ds, sensor, per_pub, k_ext=k_ext):
                if len(agg) < w + 2:
                    continue
                sid = f"{ds}|{sensor}|{L}|{scope}"
                streams_by_id[sid] = (agg, cnt, B)
                metas.append({
                    "stream_id": sid, "dataset": ds, "sensor": sensor,
                    "clamp_mode": clamp_mode, "subscription_level": L,
                    "scope": scope, "n_pubs": npub, "payload_bound": B,
                    "P_min": P_ds, "P_max": P_max_ds, "rho_split": rho_ds,
                })
    tasks = [
        (m["stream_id"], epsilon, w, m["P_min"], strategy,
         m["rho_split"], m["P_max"],
         trial, seed + 1000 * trial, m)
        for m in metas for trial in range(max(1, trials))
    ]
    workers = core._default_workers(getattr(args, "workers", None))
    rows = core._run_parallel_tasks(
        tasks, _level_sub_task, workers=workers,
        initializer=_init_level_streams_worker, initargs=(streams_by_id,),
        progress_label=f"  [sub-levels {clamp_mode}]",
        progress_every=max(20, len(tasks) // 10),
    ) if tasks else []
    df = pd.DataFrame([r for r in rows if r])
    exp_dir = os.path.join(output_dir, "experiments", "I_subscription_levels")
    os.makedirs(exp_dir, exist_ok=True)
    df.to_csv(os.path.join(exp_dir, "experiment_I_subscription_levels.csv"),
              index=False)
    if max(1, trials) > 1 and not df.empty:
        core._aggregate_over_trials(
            df, ["dataset", "sensor", "clamp_mode", "subscription_level", "scope"],
            core._TRIAL_METRIC_COLS,
        ).to_csv(
            os.path.join(exp_dir, "experiment_I_subscription_levels_aggregate.csv"),
            index=False)
    core.logger.info(f"  Experiment I (subscription levels) wrote {len(df)} rows "
                     f"({len(streams_by_id)} subscriptions) -> {exp_dir}")
    return df


def main():
    args = C.resolve(C.make_parser(
        "Subscription utility at every topic-hierarchy level (Experiment I)."
    ).parse_args())
    for clamp_mode in args._clamp_modes:
        subscription_levels_experiment(
            args._targets, clamp_mode, C.cross_dir(args, clamp_mode), args,
            epsilon=1.0, w=8, grid_config=args.grid_config,
            trials=max(1, args.trials),
        )


if __name__ == "__main__":
    main()
