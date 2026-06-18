#!/usr/bin/env python3
"""Experiment / Section 7.5: utility-hyperparameter grid search.

Sweeps (P_min x P_max x Delta_t x K_ext) per dataset/strategy/epsilon, scored by
MAE, and writes grid_canonical.json (the MAE-optimal config the other
experiments consume via --use-grid-config).  Parallel over the grid cells.

Run: python -m experiments.grid_search --dataset all
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from experiments import _common as C
from experiments import engine as core


# Worker context: precomputed (agg, cnt) streams keyed by (base_dt,k_ext,p_min).
_WORKER_GRID_CTX = None


def _init_grid_worker(streams_by_key, payload_bound, epsilon, w, seed):
    global _WORKER_GRID_CTX
    _WORKER_GRID_CTX = (streams_by_key, payload_bound, epsilon, w, seed)


def _grid_eval_task(task):
    """Score one (strategy, base_dt, k_ext, p_min, p_max, rho) grid cell."""
    strat, base_dt, k_ext, p_min, p_max, rho = task
    streams_by_key, B, eps, w, seed = _WORKER_GRID_CTX
    agg, cnt = streams_by_key[(base_dt, k_ext, p_min)]
    m = core.run_dp_on_stream(
        agg, cnt, epsilon=eps, window_size=w, min_publishers=p_min,
        payload_bound=B, strategy=strat, seed=seed,
        rho_split=rho, max_publishers=p_max)["metrics"]
    return {
        "strategy": strat, "epsilon": eps, "w": w,
        "P_min": p_min, "P_max": p_max, "delta_t": base_dt, "k_ext": k_ext,
        "rho_split": rho,
        "mae": m.get("mae"), "normalized_mae": m.get("normalized_mae"),
        "kl_divergence": m.get("kl_divergence"), "release_rate": m.get("release_rate"),
        "avg_n_tau": float(np.mean(cnt)) if cnt else float("nan"),
        "n_logical_timestamps": len(agg),
    }


def grid_search_hyperparameters(
    per_pub, payload_bound, dataset_name, sensor_name, output_dir, *,
    epsilon, w, strategies, p_min_grid, p_max_grid, dt_grid, k_ext_grid,
    rho_grid, seed: int = 77, workers: int = 1,
) -> "pd.DataFrame":
    """Sec. 7.5 grid search: the (agg,cnt) stream depends only on
    (base_dt,k_ext,p_min) -- NOT on rho_split, which only recalibrates the DP
    noise -- so each stream is rebuilt once per such key and the DP scoring
    (including the rho sweep) fans out over ``workers``.  Writes the full grid +
    per-strategy MAE optimum (whose chosen config now carries the best rho)."""
    streams_by_key: dict = {}
    for base_dt in dt_grid:
        for k_ext in k_ext_grid:
            for p_min in p_min_grid:
                agg, cnt = core._adaptive_interval_rebuild(
                    per_pub, p_min, k_ext, base_dt=base_dt)
                if len(agg) >= w + 2:
                    streams_by_key[(base_dt, k_ext, p_min)] = (agg, cnt)
    tasks = [
        (strat, base_dt, k_ext, p_min, p_max, rho)
        for strat in strategies
        for (base_dt, k_ext, p_min) in streams_by_key
        for p_max in p_max_grid
        for rho in rho_grid
        if not (p_max is not None and p_max < p_min)
    ]
    if workers and workers > 1 and len(tasks) > 1:
        scored = core._run_parallel_tasks(
            tasks, _grid_eval_task, workers=workers, initializer=_init_grid_worker,
            initargs=(streams_by_key, payload_bound, epsilon, w, seed),
            progress_label=f"  [{dataset_name}/{sensor_name}] grid",
            progress_every=max(20, len(tasks) // 10))
    else:
        _init_grid_worker(streams_by_key, payload_bound, epsilon, w, seed)
        scored = [_grid_eval_task(t) for t in tasks]
    rows = [{"dataset": dataset_name, "sensor": sensor_name, **r} for r in scored]
    df = pd.DataFrame(rows)
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, f"{dataset_name}_{sensor_name}_gridsearch.csv"),
              index=False)
    best_rows = []
    if not df.empty:
        finite = df[df["mae"].notna() & np.isfinite(df["mae"])]
        for strat in strategies:
            sub = finite[finite["strategy"] == strat]
            if not sub.empty:
                best_rows.append(sub.loc[sub["mae"].idxmin()].to_dict())
    pd.DataFrame(best_rows).to_csv(
        os.path.join(output_dir, f"{dataset_name}_{sensor_name}_gridsearch_best.csv"),
        index=False)
    core.logger.info(
        f"  [{dataset_name}/{sensor_name}] grid search: {len(df)} configs, "
        f"{len(best_rows)} per-strategy optima -> {output_dir}")
    return df


def main():
    args = C.resolve(C.make_parser(
        "Sec. 7.5 utility-hyperparameter grid search (writes grid_canonical.json)."
    ).parse_args())
    # Per-epsilon sharding: when --grid-eps is given, run only that epsilon and
    # write a grid_canonical_eps<eps>.json fragment, so the grid phase can be
    # split one-shard-per-epsilon across nodes (the heavy energy/traffic grids
    # then run their epsilons in parallel instead of serially).  The downstream
    # experiments still point --use-grid-config at grid_canonical.json;
    # _load_grid_config merges every fragment in that directory.
    grid_eps = getattr(args, "grid_eps", None)
    grid_trial = getattr(args, "grid_trial", None)
    grid_rho = getattr(args, "grid_rho", None)
    eps_values = [grid_eps] if grid_eps is not None else args.eps_values
    # Per-rho sharding REQUIRES the full-fragment merge path: per-rho canonicals
    # share the same (dataset,clamp,strategy,eps) keys, so the canonical-file
    # merge (_load_grid_config) would take last-wins instead of the cross-rho
    # MAE minimum.  So whenever --grid-rho is set we force the trial-fragment
    # path (synthesizing trial 0 if none was given) -- _merge_grid_trial_fragments
    # then picks the global optimum across all rho.  The cluster always pairs
    # --grid-rho with --grid-trial, so this only hardens manual single-shard use.
    if grid_trial is not None or grid_rho is not None:
        t = grid_trial if grid_trial is not None else 0
        # Per-trial (and optionally per-eps / per-rho) shard: write a FULL-grid
        # fragment at this trial's seed; _merge_grid_trial_fragments averages MAE
        # across trials and picks each (dataset,clamp,strategy,eps)'s global
        # optimum across ALL fragments -- including across rho shards, since rho
        # is part of each record and of the merge key.  The filename carries the
        # eps/rho tags so the shards never overwrite one another.  One SLURM task
        # per (eps, trial, rho).
        suffix = (f"_eps{grid_eps}" if grid_eps is not None else "") + \
                 (f"_rho{grid_rho}" if grid_rho is not None else "")
        core._run_grid_search_block(
            args, args._targets, args._clamp_modes, eps_values, args.strategies,
            trial=t, full_fragment=f"grid_trial{suffix}_t{t}.json")
    else:
        canonical = (f"grid_canonical_eps{grid_eps}.json"
                     if grid_eps is not None else "grid_canonical.json")
        core._run_grid_search_block(
            args, args._targets, args._clamp_modes, eps_values, args.strategies,
            canonical_filename=canonical)


if __name__ == "__main__":
    main()
