#!/usr/bin/env python3
"""Experiment / Section 6.1: utility-hyperparameter grid search.

Sweeps (P_min x P_max x Delta_t x K_ext x rho) per dataset/strategy/epsilon
and writes grid_canonical.json (the optimal config the other experiments
consume via --use-grid-config).  Every cell runs the full PAWS pipeline
(experiments.pipeline: early-stop adaptive interval + cached-count hierarchy
walk) on one subscription per distinct depth of the busiest publisher's topic
path (leaf -> ... -> root), and is scored by the coverage-penalized MAE
RELATIVE TO THE BEST CONSTANT PREDICTOR (1 - skill), averaged over those
depths.  A cell therefore wins only by telling the subscriber more than a fixed
number would: shrinking releases toward 0 (or any constant) scores >= 1, and
rarely releasing is penalized too.  NMAE and correlation are reported
alongside.

rho (the count share of each timestamp's budget) only re-splits the budget, so
it is swept as part of the same grid; ``*_rho_profile.csv`` reports the best
objective reached at each rho (the rho-selection curve).

Run: python -m experiments.grid_search --dataset all
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from dp_engine import BudgetStrategy, is_p_gated
from experiments import _common as C
from experiments import engine as core
from experiments import pipeline as pl


# Worker context: (PubMatrix, [chain, ...], R, epsilon, w, seed, wait policy).
_WORKER_GRID_CTX = None


def _init_grid_worker(mat, chains, payload_bound, epsilon, w, seed, wait_policy,
                      center=None):
    global _WORKER_GRID_CTX
    _WORKER_GRID_CTX = (mat, chains, payload_bound, epsilon, w, seed, wait_policy,
                        center)


def _nanmean(values) -> float:
    a = np.array([v for v in values if v is not None], dtype=float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")


def _grid_eval_task(task):
    """Score one (strategy, base_dt, k_ext, p_min, p_max, rho) grid cell."""
    strat, base_dt, k_ext, p_min, p_max, rho = task
    mat, chains, B, eps, w, seed, wait_policy, center = _WORKER_GRID_CTX
    per_level = [
        pl.simulate_subscription(
            mat, chain, approach="paws", epsilon=eps, w=w, payload_bound=B,
            P_min=p_min, P_max=p_max, strategy=strat, rho=rho,
            base_dt=base_dt, k_ext=k_ext, wait=wait_policy, walk=True,
            count_cache=True, seed=seed + 17 * k, center=center)["metrics"]
        for k, chain in enumerate(chains)
    ]
    return {
        "strategy": strat, "epsilon": eps, "w": w,
        "P_min": p_min, "P_max": p_max, "delta_t": base_dt, "k_ext": k_ext,
        "rho_split": rho,
        "objective": _nanmean(m["rel_mae_penalized"] for m in per_level),
        "skill": _nanmean(m["skill"] for m in per_level),
        "corr": _nanmean(m["corr"] for m in per_level),
        "skill_prior": _nanmean(m["skill_prior"] for m in per_level),
        "nmae_penalized": _nanmean(m["nmae_penalized"] for m in per_level),
        "normalized_mae": _nanmean(m["normalized_mae"] for m in per_level),
        "mae": _nanmean(m["mae"] for m in per_level),
        "coverage": _nanmean(m["coverage"] for m in per_level),
        "release_rate": _nanmean(m["release_rate"] for m in per_level),
        "mean_wait_dt": _nanmean(m["mean_wait_dt"] for m in per_level),
        "walk_rate": _nanmean(m["walk_rate"] for m in per_level),
        "n_levels_scored": len(per_level),
        "n_logical_timestamps": int(np.mean([m["n_timestamps"] for m in per_level])),
    }


def _grid_tasks(strategies, n_pubs, p_min_grid, p_max_grid, dt_grid,
                k_ext_grid, rho_grid, wait_policy):
    """The grid cells worth scoring.  Population-aware strategies sweep the
    whole grid minus cells that are equivalent or infeasible: P_max < P_min,
    P_max >= the publisher count (same as no cap), and rho so large that the
    count shares (2 per timestamp when the early-stop rule runs) leave nothing
    for the sum.  The Kellaris strategies never gate, wait or walk, so only
    Delta_t matters to them."""
    tasks = []
    for strat in strategies:
        if is_p_gated(BudgetStrategy(strat)):
            for base_dt in dt_grid:
                for k_ext in k_ext_grid:
                    units = 2 if (k_ext > 0 and wait_policy == "early_stop") else 1
                    for p_min in p_min_grid:
                        for p_max in p_max_grid:
                            if p_max is not None and (p_max < p_min or p_max >= n_pubs):
                                continue
                            for rho in rho_grid:
                                if rho * units >= 1.0:
                                    continue
                                tasks.append((strat, base_dt, k_ext, p_min, p_max, rho))
        else:
            for base_dt in dt_grid:
                tasks.append((strat, base_dt, 0, 1, None, rho_grid[0]))
    return tasks


def grid_search_hyperparameters(
    per_pub, payload_bound, dataset_name, sensor_name, output_dir, *,
    epsilon, w, strategies, p_min_grid, p_max_grid, dt_grid, k_ext_grid,
    rho_grid, seed: int = 77, workers: int = 1, sensor: str | None = None,
    wait_policy: str = "predictive",
) -> "pd.DataFrame":
    """Sec. 6.1 grid search over the full PAWS pipeline (see module docstring).

    ``sensor_name`` tags the output files; ``sensor`` is the sensor whose topic
    tree defines the scored subscriptions (defaults to ``sensor_name``).
    Writes the full grid, the per-strategy optimum (lowest objective) and the
    rho profile."""
    sensor = sensor or sensor_name
    mat = pl.PubMatrix.from_per_pub(per_pub)
    chains = [s.chain for s in pl.busiest_chain(dataset_name, sensor, per_pub)]
    tasks = _grid_tasks(strategies, len(mat.pubs), p_min_grid, p_max_grid,
                        dt_grid, k_ext_grid, rho_grid, wait_policy)
    clamp = pl.DATASETS.get(dataset_name, {}).get("static_clamps", {}).get(sensor)
    center = (float(clamp[0]) + float(clamp[1])) / 2.0 if clamp else None
    initargs = (mat, chains, payload_bound, epsilon, w, seed, wait_policy, center)
    if workers and workers > 1 and len(tasks) > 1:
        scored = core._run_parallel_tasks(
            tasks, _grid_eval_task, workers=workers, initializer=_init_grid_worker,
            initargs=initargs,
            progress_label=f"  [{dataset_name}/{sensor_name}] grid",
            progress_every=max(20, len(tasks) // 10))
    else:
        _init_grid_worker(*initargs)
        scored = [_grid_eval_task(t) for t in tasks]
    rows = [{"dataset": dataset_name, "sensor": sensor_name, **r} for r in scored]
    df = pd.DataFrame(rows)
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, f"{dataset_name}_{sensor_name}_gridsearch.csv"),
              index=False)
    best_rows = []
    if not df.empty:
        finite = df[df["objective"].notna() & np.isfinite(df["objective"])]
        for strat in strategies:
            sub = finite[finite["strategy"] == strat]
            if not sub.empty:
                best_rows.append(sub.loc[sub["objective"].idxmin()].to_dict())
        gated = finite[finite["strategy"].map(lambda s: is_p_gated(BudgetStrategy(s)))]
        if not gated.empty:
            profile = (gated.loc[gated.groupby(["strategy", "rho_split"])["objective"].idxmin()]
                       [["strategy", "rho_split", "objective", "skill", "corr",
                         "normalized_mae", "coverage", "P_min", "P_max",
                         "delta_t", "k_ext"]]
                       .sort_values(["strategy", "rho_split"]))
            profile.insert(0, "epsilon", epsilon)
            profile.insert(0, "dataset", dataset_name)
            profile.to_csv(os.path.join(
                output_dir, f"{dataset_name}_{sensor_name}_rho_profile.csv"), index=False)
    pd.DataFrame(best_rows).to_csv(
        os.path.join(output_dir, f"{dataset_name}_{sensor_name}_gridsearch_best.csv"),
        index=False)
    core.logger.info(
        f"  [{dataset_name}/{sensor_name}] grid search: {len(df)} configs over "
        f"{len(chains)} subscription depth(s), {len(best_rows)} per-strategy "
        f"optima -> {output_dir}")
    return df


def main():
    args = C.resolve(C.make_parser(
        "Sec. 6.1 utility-hyperparameter grid search (writes grid_canonical.json)."
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
