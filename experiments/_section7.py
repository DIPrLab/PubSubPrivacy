"""Shared driver for the Section 6 experiments built on ``experiments.pipeline``
(utility U, downstream ML M, induced wait K, ablation F).

``run_points`` evaluates every subscription of every sensor's topic tree, for
each requested point (an epsilon / window / wait-policy / module setting) and
approach, over ``trials`` noise seeds.  Subscriptions whose walk sees exactly
the same publisher pools (e.g. a device node and its single metric leaf) are
simulated once and reported under each label.  One worker pool per dataset
keeps memory bounded, as in the older experiments.
"""
from __future__ import annotations

import os

import pandas as pd

from experiments import engine as core
from experiments import pipeline as pl

SIM_METRIC_COLS = [
    "skill", "skill_penalized", "rel_mae_penalized", "corr", "skill_prior",
    "normalized_mae", "mae", "coverage", "nmae_penalized", "nmae_clip_penalized", "nmae_release_scope",
    "release_rate", "attribution_advantage", "avg_pooled_n", "walk_rate",
    "avg_walk_levels", "walk_kl", "mean_wait_dt", "p95_wait_dt", "max_wait_dt",
    "wait_rate", "count_units_per_ts", "compute_ms_per_ts",
]

SUB_COLS = ["subscription_level", "scope", "scope_kind", "n_pubs"]

ML_METRIC_COLS = ["ml_forecast_nmae", "ml_class_bal_acc", "ml_class_f1",
                  "ml_n_train", "ml_n_test"]

_PER_SUB = ["dataset", "sensor", "clamp_mode", "subscription_level", "scope",
            "scope_kind"]

# Every row carries ``strategy``: the budget strategy for PAWS rows, the
# baseline's name (ldp / wevent / classic) otherwise; ``ours`` marks the
# paper's strategy (--our-strategy, n_weighted).

# Output layout of every Section 6 experiment: its directory under
# experiments/, the per-configuration grouping its trials are averaged over,
# its metrics, and the columns its summary averages sensors/subscriptions to.
# The experiment modules write through it, and cluster/aggregate.py uses it to
# REBUILD the aggregate / summary / by-level CSVs from the merged per-trial
# rows -- the shards split work by sensor group, so a shard's own summary only
# covers its sensors and cannot simply be concatenated.
OUTPUTS = {
    "experiment_U_epsilon": dict(
        subdir="U_utility",
        group=_PER_SUB + ["approach", "strategy", "ours", "epsilon", "w"],
        metrics=SIM_METRIC_COLS,
        summary=["dataset", "clamp_mode", "approach", "strategy", "ours",
                 "epsilon", "w"]),
    "experiment_U_window": dict(
        subdir="U_utility",
        group=_PER_SUB + ["approach", "strategy", "ours", "epsilon", "w"],
        metrics=SIM_METRIC_COLS,
        summary=["dataset", "clamp_mode", "approach", "strategy", "ours",
                 "epsilon", "w"]),
    "experiment_M_ml": dict(
        subdir="M_ml_utility",
        group=_PER_SUB + ["approach", "strategy", "ours", "epsilon", "w"],
        metrics=ML_METRIC_COLS + SIM_METRIC_COLS,
        summary=["dataset", "clamp_mode", "approach", "strategy", "ours",
                 "epsilon", "w"]),
    "experiment_K_induced_wait": dict(
        subdir="K_induced_wait",
        group=_PER_SUB + ["strategy", "ours", "wait_policy", "k_ext", "epsilon", "w"],
        metrics=SIM_METRIC_COLS + ["mean_wait_s", "p95_wait_s", "max_wait_s",
                                   "delta_t_seconds"],
        summary=["dataset", "clamp_mode", "strategy", "ours", "wait_policy", "k_ext"]),
    "experiment_F_ablation": dict(
        subdir="F_ablation",
        group=_PER_SUB + ["strategy", "ours", "module", "module_idx", "epsilon", "w"],
        metrics=SIM_METRIC_COLS,
        summary=["dataset", "clamp_mode", "strategy", "ours", "scope_kind", "module",
                 "module_idx", "epsilon"]),
}


def our_strategy(args) -> str:
    return getattr(args, "our_strategy", None) or "n_weighted"


def paws_approaches(args, population_aware_only: bool = False) -> tuple:
    """One ``"paws:<strategy>"`` approach per budget strategy to evaluate
    (``--strategies``, default all nine).  ``population_aware_only`` keeps the
    P-gated strategies and n-weighted -- the only ones the gate, the adaptive
    interval and the hierarchy walk act on."""
    from dp_engine import BudgetStrategy, is_p_gated
    strategies = list(getattr(args, "strategies", None) or core.ALL_STRATEGIES)
    if population_aware_only:
        strategies = [s for s in strategies if is_p_gated(BudgetStrategy(s))]
    ours = our_strategy(args)
    if ours not in strategies and (not population_aware_only
                                   or is_p_gated(BudgetStrategy(ours))):
        strategies.append(ours)
    return tuple(f"paws:{s}" for s in strategies)


def approach_kwargs(args, dataset, clamp_mode, approach, epsilon, w, *,
                    strategy=None, overrides=None) -> tuple[dict, dict]:
    """Simulator kwargs + row metadata for one approach at (epsilon, w).

    ``approach`` is a baseline (ldp / wevent / classic), ``"paws"`` (the paper
    strategy, or ``strategy``) or ``"paws:<strategy>"``.  Every PAWS strategy
    takes its OWN Sec. 6.1 grid optimum for (dataset, strategy, epsilon).  The
    baselines run on the base interval Delta_t of our strategy's optimum, so
    every approach answers the same per-interval query."""
    if approach.startswith("paws:"):
        approach, strategy = "paws", approach.split(":", 1)[1]
    ours = our_strategy(args)
    strategy = (strategy or ours) if approach == "paws" else ours
    prm = pl.resolve_paws_params(core, getattr(args, "grid_config", None),
                                 dataset, clamp_mode, strategy, epsilon, args)
    kw = {"approach": approach, "epsilon": float(epsilon), "w": int(w),
          "base_dt": prm["base_dt"]}
    meta = {"approach": approach, "epsilon": float(epsilon), "w": int(w),
            "delta_t": prm["base_dt"],
            "strategy": strategy if approach == "paws" else approach,
            "ours": bool(approach == "paws" and strategy == ours)}
    if approach == "paws":
        kw.update(P_min=prm["P_min"], P_max=prm["P_max"], strategy=strategy,
                  rho=prm["rho"], k_ext=prm["k_ext"],
                  wait=getattr(args, "wait_policy", "predictive"), walk=True,
                  count_cache=getattr(args, "count_cache", True))
        if overrides:
            kw.update(overrides)
        meta.update(strategy=strategy, P_min=kw["P_min"], P_max=kw["P_max"],
                    rho_split=kw["rho"], k_ext=kw["k_ext"], wait_policy=kw["wait"],
                    walk=kw["walk"], count_cache=kw["count_cache"])
    return kw, meta


def run_points(datasets, clamp_mode, args, *, points, approaches, kwargs_fn,
               label, trials=1, seed=77, first_sensor_only=False,
               task_fn=None) -> pd.DataFrame:
    """Simulate every (sensor, subscription, point, approach, trial).

    ``kwargs_fn(dataset, sensor, point, approach) -> (sim kwargs, meta)`` or
    None to skip.  ``task_fn`` replaces ``pipeline.sim_task`` (same task tuple;
    it may read ``pipeline._WORKER_CTX``)."""
    task_fn = task_fn or pl.sim_task
    workers = core._default_workers(getattr(args, "workers", None))
    rows = []
    for ds_name in datasets:
        prepared = core.prepare_dataset(
            ds_name, clamp_mode=clamp_mode, eps_clip=args.eps_clip,
            seed=args.seed, max_rows=core._dataset_max_rows(ds_name, args))
        if prepared is None or not prepared.per_pubs:
            continue
        sensors = [s for s in prepared.spec["sensors"] if s in prepared.per_pubs]
        sensors = core._shard_sensors(args, sensors)
        if first_sensor_only:
            sensors = sensors[:1]
        mats, chains, tasks, expand = {}, {}, [], {}
        for sensor in sensors:
            per_pub, B = prepared.per_pubs[sensor]
            mat = pl.PubMatrix.from_per_pub(per_pub)
            if mat.T < 2:
                continue
            # Public center of the declared clamp for PAWS' centered sum.
            lo_hi = prepared.spec.get("static_clamps", {}).get(sensor)
            center = (float(lo_hi[0]) + float(lo_hi[1])) / 2.0 if lo_hi else None
            mk = f"{ds_name}|{sensor}"
            mats[mk] = mat
            by_chain: dict = {}
            for sub in pl.subscriptions_for(ds_name, sensor, mat.pubs):
                ck = (mk, sub.chain_key)
                chains[ck] = sub.chain
                by_chain.setdefault(ck, []).append(sub)
            for point in points:
                for approach in approaches:
                    spec = kwargs_fn(ds_name, sensor, point, approach)
                    if spec is None:
                        continue
                    kw, meta = spec
                    for ck, subs in by_chain.items():
                        for trial in range(max(1, trials)):
                            tid = len(tasks)
                            tasks.append(({"_tid": tid}, mk, ck,
                                          {**kw, "payload_bound": B,
                                           "center": center,
                                           "seed": seed + 1000 * trial}))
                            expand[tid] = (subs, {
                                "dataset": ds_name, "sensor": sensor,
                                "clamp_mode": clamp_mode, "payload_bound": B,
                                **point, **meta, "trial": trial,
                                "seed": seed + 1000 * trial})
        if not tasks:
            continue
        core.logger.info(f"  [{label} {ds_name}/{clamp_mode}] {len(tasks)} simulations "
                         f"over {len(chains)} distinct subscription pools")
        results = core._run_parallel_tasks(
            tasks, task_fn, workers=workers,
            initializer=pl.init_sim_worker, initargs=(mats, chains),
            progress_label=f"  [{label} {ds_name}]",
            progress_every=max(50, len(tasks) // 10))
        for r in results:
            subs, meta = expand[r.pop("_tid")]
            for sub in subs:
                rows.append({**meta, "subscription_level": sub.level,
                             "scope": sub.scope, "scope_kind": sub.scope_kind,
                             "n_pubs": sub.n_pubs, **r})
    return pd.DataFrame(rows)


def write_outputs(df: pd.DataFrame, exp_dir: str, stem: str, *, group_cols,
                  metric_cols, summary_cols, trials: int) -> None:
    """Per-trial rows, the per-configuration mean/std over trials, and a
    summary averaged over sensors and subscriptions (per ``summary_cols``) and
    per subscription level (``summary_cols`` + subscription_level)."""
    if df.empty:
        # An empty shard (e.g. a sensor group beyond a dataset's sensor count)
        # writes nothing, so the aggregator never meets a header-less CSV.
        core.logger.info(f"  {stem}: no rows (empty shard) -- nothing written")
        return
    os.makedirs(exp_dir, exist_ok=True)
    df.to_csv(os.path.join(exp_dir, f"{stem}.csv"), index=False)
    metric_cols = [c for c in metric_cols if c in df.columns]
    agg = core._aggregate_over_trials(df, group_cols, metric_cols)
    agg.to_csv(os.path.join(exp_dir, f"{stem}_aggregate.csv"), index=False)
    means = [f"{c}_mean" for c in metric_cols if f"{c}_mean" in agg.columns]
    summary_cols = [c for c in summary_cols if c in agg.columns]
    (agg.groupby(summary_cols, dropna=False)[means].mean().reset_index()
        .to_csv(os.path.join(exp_dir, f"{stem}_summary.csv"), index=False))
    if "subscription_level" in agg.columns:
        (agg.groupby(summary_cols + ["subscription_level"], dropna=False)[means]
            .mean().reset_index()
            .to_csv(os.path.join(exp_dir, f"{stem}_by_level.csv"), index=False))
    core.logger.info(f"  {stem}: {len(df)} rows ({trials} trial(s)) -> {exp_dir}")



def write_experiment(df: pd.DataFrame, output_dir: str, stem: str,
                     trials: int = 1) -> None:
    """Write a Section 6 experiment's CSVs per its ``OUTPUTS`` layout under
    ``<output_dir>/experiments/<subdir>/``."""
    spec = OUTPUTS[stem]
    write_outputs(df, os.path.join(output_dir, "experiments", spec["subdir"]), stem,
                  group_cols=spec["group"], metric_cols=spec["metrics"],
                  summary_cols=spec["summary"], trials=trials)


def rebuild_summaries(root: str) -> int:
    """Recompute every Section 6 aggregate / summary / by-level CSV under
    ``root`` from its merged per-trial CSV.  Returns how many were rebuilt."""
    n = 0
    for dirpath, _dirs, files in os.walk(root):
        for stem, spec in OUTPUTS.items():
            if f"{stem}.csv" not in files:
                continue
            df = pd.read_csv(os.path.join(dirpath, f"{stem}.csv"))
            trials = int(df["trial"].nunique()) if "trial" in df.columns else 1
            write_outputs(df, dirpath, stem, group_cols=spec["group"],
                          metric_cols=spec["metrics"], summary_cols=spec["summary"],
                          trials=trials)
            n += 1
    return n
