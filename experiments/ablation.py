#!/usr/bin/env python3
"""Experiment F: incremental-module ablation (paper Sec. 7.8).

Adds the mechanism's modules one at a time and measures the utility impact:
  M1 P-gated allocation only -> M2 + subscription rewriting (interval extension,
  Sec. 6.7) -> M3 + walking up the topic tree (Algorithm 1, Sec. 6.5).
Reported at two subscription scopes (leaf isolates the walk-up; pooled isolates
the interval extension).  Fully offline (no broker), parallel, all datasets.

Run: python -m experiments.ablation --dataset all --trials 6
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from experiments import _common as C
from experiments import engine as core

MODULES = ["M1_pgate", "M2_interval_ext", "M3_walk_up"]


def _ablation_task(task):
    """One (scope, module, trial) ablation point, run from the worker cache."""
    (key, ds_name, sensor, clamp_mode, scope, module, m_idx,
     epsilon, w, P_ds, P_max_ds, k_ext_ds, rho_ds, strategy,
     trial, seed_t) = task
    agg, cnt, B = core._WORKER_STREAMS[key]
    if len(agg) < w + 2:
        m = {"normalized_mae": float("nan"), "kl_divergence": float("nan"),
             "release_rate": float("nan"), "mae": float("nan")}
    else:
        m = core.run_dp_on_stream(
            agg, cnt, epsilon=epsilon, window_size=w, min_publishers=P_ds,
            payload_bound=B, strategy=strategy, seed=seed_t,
            rho_split=rho_ds, max_publishers=P_max_ds)["metrics"]
    return {
        "dataset": ds_name, "sensor": sensor, "clamp_mode": clamp_mode,
        "scope": scope, "module": module, "module_idx": m_idx,
        "epsilon": epsilon, "w": w, "P": P_ds, "P_max": P_max_ds,
        "k_ext": k_ext_ds, "rho_split": rho_ds,
        "strategy": strategy, "payload_bound": B,
        "trial": trial, "seed": seed_t,
        "normalized_mae": m.get("normalized_mae"), "mae": m.get("mae"),
        "kl_divergence": m.get("kl_divergence"),
        "release_rate": m.get("release_rate"),
        "avg_n_tau": float(np.mean(cnt)) if cnt else float("nan"),
        "n_logical_timestamps": len(agg),
    }


def _ablation_module_streams(per_pub, P, k_ext, scope):
    """Build the (aggregate, count) stream each cumulative module produces for
    one subscription scope.  Returns {module_name -> (agg, cnt)}."""
    pubs = list(per_pub.keys())
    if scope == "leaf":
        # Subscriber binds to the single busiest publisher's leaf.
        leaf_pub = max(pubs, key=lambda p: sum(v is not None for v in per_pub[p]))
        leaf_subset = [leaf_pub]
        m1 = core._adaptive_interval_rebuild(per_pub, P, 0, subset=leaf_subset)
        m2 = core._adaptive_interval_rebuild(per_pub, P, k_ext, subset=leaf_subset)
        m3 = core._adaptive_interval_rebuild(per_pub, P, k_ext, subset=pubs)
        return {"M1_pgate": m1, "M2_interval_ext": m2, "M3_walk_up": m3}
    # scope == "pooled": gate active on the whole-sensor scope.
    m1 = core._adaptive_interval_rebuild(per_pub, P, 0, subset=pubs)
    m2 = core._adaptive_interval_rebuild(per_pub, P, k_ext, subset=pubs)
    m3 = m2  # no ancestor above the pooled root, so walk-up adds nothing here
    return {"M1_pgate": m1, "M2_interval_ext": m2, "M3_walk_up": m3}


def ablation_experiment(
    datasets, clamp_mode, output_dir, args, *,
    epsilon: float = 1.0, w: int = 8, P: int = 3, k_ext: int = 3,
    epsilon_count: float = 0.0, strategy: str = "p_gated_uniform",
    seed: int = 77, grid_config: dict | None = None, trials: int = 1,
) -> "pd.DataFrame":
    """Sec. 7.8 incremental-module ablation; per-trial rows + per-dataset
    aggregate.  Uses the Sec. 7.5 grid-optimal (P_min, P_max, K_ext) when
    ``grid_config`` is supplied."""
    workers = core._default_workers(getattr(args, "workers", None))
    streams_by_key: dict = {}
    tasks = []
    for ds_name in datasets:
        prepared = core.prepare_dataset(
            ds_name, clamp_mode=clamp_mode, eps_clip=args.eps_clip,
            seed=args.seed, max_rows=core._dataset_max_rows(ds_name, args))
        if prepared is None or not prepared.per_pubs:
            continue
        sensor = next(
            (s for s in prepared.spec["sensors"]
             if s in prepared.per_pubs and s in prepared.spec["static_clamps"]
             and len(prepared.per_pubs[s][0]) >= 2), None)
        if sensor is None:
            core.logger.warning(f"[exp F] no suitable sensor in {ds_name}; skip")
            continue
        prm = core._resolve_params(grid_config, ds_name, clamp_mode, strategy,
                                   epsilon, {"P_min": P, "P_max": None, "k_ext": k_ext})
        P_ds, P_max_ds, k_ext_ds = prm["P_min"], prm["P_max"], prm["k_ext"]
        rho_ds = prm["rho_split"]   # grid-selected split for this (ds, strategy, eps)
        per_pub, B = prepared.per_pubs[sensor]
        for scope in ("leaf", "pooled"):
            streams = _ablation_module_streams(per_pub, P_ds, k_ext_ds, scope)
            for m_idx, module in enumerate(MODULES, start=1):
                agg, cnt = streams[module]
                key = f"{ds_name}|{sensor}|{scope}|{module}"
                streams_by_key[key] = (agg, cnt, B)
                for trial in range(max(1, trials)):
                    tasks.append((key, ds_name, sensor, clamp_mode, scope, module,
                                  m_idx, epsilon, w, P_ds, P_max_ds, k_ext_ds,
                                  rho_ds, strategy, trial, seed + 1000 * trial))
    rows = core._run_parallel_tasks(
        tasks, _ablation_task, workers=workers,
        initializer=core._init_streams_worker, initargs=(streams_by_key,),
        progress_label=f"  [exp F {clamp_mode}]", progress_every=100)
    exp_dir = os.path.join(output_dir, "experiments", "F_ablation")
    os.makedirs(exp_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(exp_dir, "experiment_F_ablation.csv"), index=False)
    if max(1, trials) > 1:
        core._aggregate_over_trials(
            df, ["dataset", "sensor", "clamp_mode", "scope", "module", "module_idx"],
            core._TRIAL_METRIC_COLS,
        ).to_csv(os.path.join(exp_dir, "experiment_F_ablation_aggregate.csv"),
                 index=False)
    core.logger.info(f"  Experiment F (ablation) wrote {len(df)} rows -> {exp_dir}")
    if not df.empty and getattr(args, "generate_plots", False):
        core._plot_single_axis_experiment(
            df[df["scope"] == "leaf"], "module_idx", "module (cumulative)",
            os.path.join(exp_dir, "experiment_F_ablation_leaf.png"),
            f"Ablation (leaf scope) [clamp={clamp_mode}]")
    return df


def main():
    args = C.resolve(C.make_parser(
        "Experiment F: incremental-module ablation (Sec. 7.8)."
    ).parse_args())
    for clamp_mode in args._clamp_modes:
        ablation_experiment(
            args._targets, clamp_mode, C.cross_dir(args, clamp_mode), args,
            epsilon=1.0, w=8, P=args.ablation_P, k_ext=args.k_ext,
            epsilon_count=args.epsilon_count, grid_config=args.grid_config,
            trials=max(1, args.trials))


if __name__ == "__main__":
    main()
