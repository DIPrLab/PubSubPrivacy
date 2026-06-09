#!/usr/bin/env python3
"""Experiment C: effect of the privacy budget epsilon (paper Sec. 7.7).

Fixes (P, w, strategy) at the paper's five canonical combos and sweeps epsilon
in {0.1,0.25,0.5,1,2,4,8}, over every dataset on the PerCom topic hierarchy, in
parallel.  ``--trials`` repeats each point over N seeds (per-trial + aggregate).

Run: python -m experiments.epsilon --dataset all --trials 6
"""
from __future__ import annotations

import os

import pandas as pd

from experiments import _common as C
from experiments import engine as core


def experiment_C_vary_epsilon(datasets, clamp_mode, output_dir, args,
                              fixed_combos=None, eps_values=None) -> "pd.DataFrame":
    fixed_combos = fixed_combos or core.EXPERIMENT_FIXED_COMBOS_C
    eps_values = eps_values or core.EXPERIMENT_C_EPS_VALUES
    workers = core._default_workers(getattr(args, "workers", None))
    log_messages = getattr(args, "log_messages", True)
    trials = max(1, getattr(args, "trials", 1))
    k_ext = getattr(args, "k_ext", 0)
    rows = []
    # Test EVERY point of the PerCom topic hierarchy per (dataset, sensor).
    for ds_name, entries in core._iter_clamped_by_dataset(
            datasets, clamp_mode, args.eps_clip, args.seed, args):
        tasks, streams_by_key = [], {}
        for sensor, per_pub, R, _ in entries:
            for (L, scope, agg, cnt, _np) in core.level_subscription_streams(
                    ds_name, sensor, per_pub, k_ext=k_ext):
                key = f"{sensor}|{L}|{scope}"
                streams_by_key[key] = (agg, cnt, R)  # shipped to workers ONCE
                for combo in fixed_combos:
                    for eps in eps_values:
                        for trial in range(trials):
                            tasks.append((
                                ds_name, sensor, key,
                                combo["strategy"], combo["P"], eps, combo["w"],
                                clamp_mode, log_messages, "C_vary_epsilon",
                                trial, 77 + 1000 * trial, L, scope,
                            ))
        rows.extend(core._run_parallel_tasks(
            tasks, core._experiment_single_axis_task, workers=workers,
            initializer=core._init_streams_worker, initargs=(streams_by_key,),
            progress_label=f"  [exp C {ds_name}/{clamp_mode}]", progress_every=100))
    summaries, messages = core._split_messages_from_rows(rows)
    df = pd.DataFrame(summaries)
    exp_dir = os.path.join(output_dir, "experiments", "C_vary_epsilon")
    os.makedirs(exp_dir, exist_ok=True)
    df.to_csv(os.path.join(exp_dir, "experiment_C_vary_epsilon.csv"), index=False)
    if trials > 1:
        core._aggregate_over_trials(
            df, ["dataset", "sensor", "clamp_mode", "subscription_level", "scope",
                 "strategy", "P", "epsilon", "w"],
            core._TRIAL_METRIC_COLS,
        ).to_csv(os.path.join(exp_dir, "experiment_C_vary_epsilon_aggregate.csv"),
                 index=False)
    if log_messages and messages:
        from message_logger import write_messages_csv
        n = write_messages_csv(messages, os.path.join(exp_dir, "experiment_C_messages.csv"))
        core.logger.info(f"  Experiment C wrote {n} per-release messages")
    core.logger.info(f"  Experiment C wrote {len(df)} rows -> {exp_dir}")
    core._plot_single_axis_experiment(
        df, x_col="epsilon", x_label="privacy budget  ε",
        path=os.path.join(exp_dir, "experiment_C_vary_epsilon.png"),
        title=f"Experiment C [clamp={clamp_mode}]: NMAE and KL vs ε", logx=True)
    return df


def main():
    args = C.resolve(C.make_parser(
        "Experiment C: vary privacy budget epsilon (Sec. 7.7)."
    ).parse_args())
    for clamp_mode in args._clamp_modes:
        experiment_C_vary_epsilon(args._targets, clamp_mode,
                                  C.cross_dir(args, clamp_mode), args)


if __name__ == "__main__":
    main()
