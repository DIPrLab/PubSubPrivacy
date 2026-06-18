#!/usr/bin/env python3
"""Experiment D: end-to-end broker plugin path (paper Sec. 6 mechanism).

Drives the broker-side PrivacyPlugin on a stubbed MQTT client (no broker) over
two topologies per dataset: 'pooled' (all publishers on one leaf; verifies the
plugin's DP math matches the offline engine byte-for-byte) and 'hierarchy' (each
publisher on its own PerCom leaf so Algorithm 1 walks up on every release).
The stubbed-MQTT driver (_drive_plugin_scenario) is shared infra in the core.

Run: python -m experiments.plugin_path --dataset all
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from experiments import _common as C
from experiments import engine as core


def experiment_D_plugin_path(
    datasets, clamp_mode, output_dir, args, *,
    epsilon: float = 1.0, w: int = 8, P: int = 2, delta_t: float = 1.0,
    n_steps: int = 200, seed: int = 123, trials: int = 1,
) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """Plugin end-to-end vs offline DP engine on a stubbed MQTT path; per-release
    rows (trial 0) + per-(run, trial) summary (release/walk-up/defer counts,
    t_start sanity, P-gate audit, byte-for-byte offline match for pooled), with a
    per-(dataset, scenario) aggregate over ``trials`` seeds."""
    trials = max(1, trials)
    rows, summary_rows = [], []
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
            core.logger.warning(f"[exp D] no suitable sensor in {ds_name}; skip")
            continue
        per_pub = prepared.per_pubs[sensor][0]
        for scenario in ("pooled", "hierarchy"):
            for trial in range(trials):
                seed_t = seed + 1000 * trial
                result = core._drive_plugin_scenario(
                    per_pub, sensor, prepared.spec, scenario,
                    epsilon=epsilon, w=w, P=P, delta_t=delta_t,
                    n_steps=n_steps, seed=seed_t)
                plugin, mock, tau_log = result["plugin"], result["mock"], result["tau_log"]
                log = plugin.release_log
                released = [r for r in log if not r["deferred"] and r["released_value"] is not None]
                deferred = [r for r in log if r["deferred"]]
                walkups = [r for r in log if r["walk_up"]]
                p_violations = sum(1 for r in released if r["n_tau"] < result["plugin_P"])
                published_t = [m["t_start"] for m in mock.published]
                monotonic = all(published_t[i+1] >= published_t[i]
                                for i in range(len(published_t) - 1))
                spacings = [published_t[i+1] - published_t[i]
                            for i in range(len(published_t) - 1)]
                max_diff_offline = float("nan")
                if scenario == "pooled" and released:
                    agg = [e["true_clamped_mean"] for e in tau_log]
                    cnt = [e["n_active"] for e in tau_log]
                    lo, hi = prepared.spec["static_clamps"][sensor]
                    B = float(hi - lo)
                    offline = core.run_dp_on_stream(
                        agg, cnt, epsilon=epsilon, window_size=w,
                        min_publishers=result["plugin_P"], payload_bound=B,
                        strategy="uniform", seed=seed_t)
                    offline_by_tau = {i: v for i, v in enumerate(offline["noisy_values"])
                                      if v is not None}
                    diffs = [abs(r["released_value"] - offline_by_tau[r["tau"] - 1])
                             for r in released
                             if (r["tau"] - 1) in offline_by_tau
                             and offline_by_tau[r["tau"] - 1] is not None]
                    max_diff_offline = max(diffs) if diffs else float("nan")
                # Per-release rows only for the first trial (avoid trials x volume).
                if trial == 0:
                    for r in log:
                        rows.append({
                            "dataset": ds_name, "scenario": scenario, "sensor": sensor,
                            "clamp_mode": clamp_mode, "tau": r["tau"], "t_start": r["t_start"],
                            "leaf_topic": r["leaf_topic"], "release_scope": r["release_scope"],
                            "true_aggregate": r["true_aggregate"],
                            "released_value": r["released_value"], "n_tau": r["n_tau"],
                            "walk_up": r["walk_up"], "deferred": r["deferred"],
                        })
                summary_rows.append({
                    "dataset": ds_name, "scenario": scenario, "sensor": sensor,
                    "clamp_mode": clamp_mode, "plugin_P": result["plugin_P"],
                    "trial": trial, "seed": seed_t,
                    "num_taus": len(log), "num_releases": len(released),
                    "num_deferrals": len(deferred), "num_walkups": len(walkups),
                    "release_rate": len(released) / max(1, len(log)),
                    "walkup_rate": len(walkups) / max(1, len(log)),
                    "t_start_monotonic": monotonic,
                    "t_start_spacing_mean": float(np.mean(spacings)) if spacings else float("nan"),
                    "t_start_spacing_std": float(np.std(spacings)) if spacings else float("nan"),
                    "p_gate_violations": p_violations,
                    "max_abs_diff_vs_offline": max_diff_offline,
                    "num_published": len(mock.published),
                })
            core.logger.info(
                f"[exp D] {ds_name}/{sensor} scenario={scenario}: "
                f"{trials} trial(s) (per-release rows from trial 0)")
    exp_dir = os.path.join(output_dir, "experiments", "D_plugin_path")
    os.makedirs(exp_dir, exist_ok=True)
    df_rel = pd.DataFrame(rows)
    df_sum = pd.DataFrame(summary_rows)
    df_rel.to_csv(os.path.join(exp_dir, "experiment_D_plugin_releases.csv"), index=False)
    df_sum.to_csv(os.path.join(exp_dir, "experiment_D_plugin_summary.csv"), index=False)
    if trials > 1 and not df_sum.empty:
        core._aggregate_over_trials(
            df_sum, ["dataset", "scenario", "sensor", "clamp_mode", "plugin_P"],
            ["release_rate", "walkup_rate", "num_releases", "num_walkups",
             "num_deferrals", "p_gate_violations", "max_abs_diff_vs_offline"],
        ).to_csv(os.path.join(exp_dir, "experiment_D_plugin_summary_aggregate.csv"),
                 index=False)
    core.logger.info(f"  Experiment D wrote {len(df_rel)} release rows, "
                     f"{len(df_sum)} summary rows -> {exp_dir}")
    if not df_sum.empty and getattr(args, "generate_plots", False):
        plt = core.plt
        datasets_present = sorted(df_sum["dataset"].unique())
        scenarios = ["pooled", "hierarchy"]
        x = np.arange(len(datasets_present)); bw = 0.38
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for i, (metric, ylabel) in enumerate([
                ("release_rate", "release rate"),
                ("walkup_rate", "walk-up rate (Algorithm 1 fires)"),
                ("max_abs_diff_vs_offline", "|plugin - offline|  (pooled only)")]):
            ax = axes[i]
            for j, scen in enumerate(scenarios):
                vals = [float(df_sum[(df_sum["dataset"] == ds) & (df_sum["scenario"] == scen)][metric].iloc[0])
                        if not df_sum[(df_sum["dataset"] == ds) & (df_sum["scenario"] == scen)].empty
                        else float("nan") for ds in datasets_present]
                ax.bar(x + (j - 0.5) * bw, vals, bw, label=scen, alpha=0.85)
            ax.set(xticks=x, xlabel="dataset", ylabel=ylabel, title=ylabel)
            ax.set_xticklabels(datasets_present, rotation=30, fontsize=8)
            ax.grid(True, alpha=0.3, axis="y"); ax.legend(fontsize=8)
        fig.suptitle(f"Experiment D: plugin end-to-end [clamp={clamp_mode}]", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, "experiment_D_plugin_path.png"), dpi=150)
        plt.close()
    return df_rel, df_sum


def main():
    args = C.resolve(C.make_parser(
        "Experiment D: end-to-end broker plugin path (stubbed MQTT)."
    ).parse_args())
    for clamp_mode in args._clamp_modes:
        experiment_D_plugin_path(args._targets, clamp_mode,
                                 C.cross_dir(args, clamp_mode), args,
                                 trials=max(1, args.trials))


if __name__ == "__main__":
    main()
