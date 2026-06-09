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
    rows = []
    # Test EVERY point of the PerCom topic hierarchy: for each (dataset, sensor)
    # iterate every subscription level/subtree (core.level_subscription_streams),
    # then sweep w x combo x trial.  All tasks fan out over the worker pool.
    for ds_name, entries in core._iter_clamped_by_dataset(
            datasets, clamp_mode, args.eps_clip, args.seed, args):
        tasks, streams_by_key = [], {}
        for sensor, per_pub, R, _ in entries:
            for (L, scope, agg, cnt, _np) in core.level_subscription_streams(
                    ds_name, sensor, per_pub, k_ext=k_ext):
                key = f"{sensor}|{L}|{scope}"
                streams_by_key[key] = (agg, cnt, R)  # shipped to workers ONCE
                for combo in fixed_combos:
                    for w in w_values:
                        for trial in range(trials):
                            tasks.append((
                                ds_name, sensor, key,
                                combo["strategy"], combo["P"], combo["epsilon"], w,
                                clamp_mode, log_messages, "B_vary_w",
                                trial, 77 + 1000 * trial, L, scope,
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
