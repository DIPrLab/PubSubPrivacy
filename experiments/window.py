#!/usr/bin/env python3
"""Experiment B: effect of the window size w (paper Sec. 7.6).

Fixes (P, epsilon, strategy) at the paper's five canonical combos and sweeps
w in {4,6,8,10,12,16}, over every dataset on the PerCom topic hierarchy, in
parallel.  ``--trials`` repeats each point over N seeds (per-trial + aggregate).

Run: python -m experiments.window --dataset all --trials 6
"""
from __future__ import annotations

import os

import pandas as pd

from experiments import _common as C
from experiments import engine as core


def experiment_B_vary_w(datasets, clamp_mode, output_dir, args,
                        fixed_combos=None, w_values=None) -> "pd.DataFrame":
    fixed_combos = fixed_combos or core.EXPERIMENT_FIXED_COMBOS_B
    w_values = w_values or core.EXPERIMENT_B_W_VALUES
    workers = core._default_workers(getattr(args, "workers", None))
    log_messages = getattr(args, "log_messages", True)
    trials = max(1, getattr(args, "trials", 1))
    k_ext = getattr(args, "k_ext", 0)
    grid_config = getattr(args, "grid_config", None)
    rows = []
    # Only w stays free; P_min, P_max, Delta_t, K_ext and rho come from the §7.5
    # grid optimum for each (dataset, strategy, eps).  The combos' fixed P is
    # therefore dropped -- distinct (strategy, eps) pairs drive the grid lookup --
    # and each level's stream is rebuilt at the grid P_min/K_ext/Delta_t.  Test
    # EVERY point of the PerCom topic hierarchy; all tasks fan out over the pool.
    combos = sorted({(c["strategy"], c["epsilon"]) for c in fixed_combos})
    for ds_name, entries in core._iter_clamped_by_dataset(
            datasets, clamp_mode, args.eps_clip, args.seed, args):
        tasks, streams_by_key = [], {}
        for sensor, per_pub, R, _ in entries:
            for (strategy, eps) in combos:
                for (L, scope, agg, cnt, _np, prm) in core.grid_level_streams(
                        ds_name, sensor, per_pub, clamp_mode, grid_config,
                        strategy, eps, default_k_ext=k_ext):
                    key = f"{sensor}|{L}|{scope}|{strategy}|{eps}"
                    streams_by_key[key] = (agg, cnt, R)  # shipped to workers ONCE
                    for w in w_values:
                        for trial in range(trials):
                            tasks.append((
                                ds_name, sensor, key,
                                strategy, prm["P_min"], eps, w,
                                clamp_mode, log_messages, "B_vary_w",
                                trial, 77 + 1000 * trial, L, scope,
                                prm["rho_split"], prm["P_max"],
                            ))
        rows.extend(core._run_parallel_tasks(
            tasks, core._experiment_single_axis_task, workers=workers,
            initializer=core._init_streams_worker, initargs=(streams_by_key,),
            progress_label=f"  [exp B {ds_name}/{clamp_mode}]", progress_every=100))
    summaries, messages = core._split_messages_from_rows(rows)
    df = pd.DataFrame(summaries)
    exp_dir = os.path.join(output_dir, "experiments", "B_vary_w")
    os.makedirs(exp_dir, exist_ok=True)
    df.to_csv(os.path.join(exp_dir, "experiment_B_vary_w.csv"), index=False)
    if trials > 1:
        core._aggregate_over_trials(
            df, ["dataset", "sensor", "clamp_mode", "subscription_level", "scope",
                 "strategy", "P", "epsilon", "w"],
            core._TRIAL_METRIC_COLS,
        ).to_csv(os.path.join(exp_dir, "experiment_B_vary_w_aggregate.csv"),
                 index=False)
    if log_messages and messages:
        from message_logger import write_messages_csv
        n = write_messages_csv(messages, os.path.join(exp_dir, "experiment_B_messages.csv"))
        core.logger.info(f"  Experiment B wrote {n} per-release messages")
    core.logger.info(f"  Experiment B wrote {len(df)} rows -> {exp_dir}")
    core._plot_single_axis_experiment(
        df, x_col="w", x_label="window size  w",
        path=os.path.join(exp_dir, "experiment_B_vary_w.png"),
        title=f"Experiment B [clamp={clamp_mode}]: NMAE and KL vs w")
    return df


def main():
    args = C.resolve(C.make_parser(
        "Experiment B: vary window size w (Sec. 7.6)."
    ).parse_args())
    for clamp_mode in args._clamp_modes:
        experiment_B_vary_w(args._targets, clamp_mode,
                            C.cross_dir(args, clamp_mode), args)


if __name__ == "__main__":
    main()
