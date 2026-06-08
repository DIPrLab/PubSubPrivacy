#!/usr/bin/env python3
"""
Post-hoc plot generator for clamped w-event DP / P-allocation experiments.

Reads every CSV under a results directory produced by ``run_experiment.py``
and re-renders every figure.  ``run_experiment.py --no-generate-plots`` (the
default) writes only CSVs; this script renders the PNGs separately so the
experimental pipeline and the plotting pipeline are cleanly decoupled.

Expected input layout (produced by ``run_experiment.py``):

    <output_dir>/
      <dataset>/
        <clamp_mode>/
          sweep/sweep_results.csv
          intro/{figure_extreme1_global,figure_extreme2_per_publisher,
                 figure_extremes_vs_ours,figure_u_shaped_P_vs_KL,
                 figure1_kl_vs_P}.csv        (varies per dataset)
          tuning/*.csv
          extras/{collusion,k_ext_sweep,n_weighted_spotlight}.csv
          messages/sweep_messages.csv
      cross_dataset/
        combined_sweep_results.csv
        figure1_all_datasets.csv
        <clamp_mode>/
          experiments/{A_greedy_vs_brute,B_vary_w,C_vary_epsilon,
                       D_plugin_path}/*.csv

Usage:

    python generate_plots.py --output-dir results_3
    python generate_plots.py --output-dir results_3 --only sweep,experiments
"""

from __future__ import annotations

import argparse
import logging
import os
from glob import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════
#  Generic helpers
# ═════════════════════════════════════════════════════════════════════════

def _safe_read_csv(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        logger.warning(f"Failed to read {path}: {exc}")
        return None


def _savefig(path: str, dpi: int = 150):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()
    logger.info(f"  -> {path}")


def _hide_unused(axes_grid, n_used, nrows, ncols):
    for idx in range(n_used, nrows * ncols):
        axes_grid[idx // ncols][idx % ncols].set_visible(False)


# ═════════════════════════════════════════════════════════════════════════
#  Per-dataset / per-clamp_mode plots (sweep, intro, tuning, extras)
# ═════════════════════════════════════════════════════════════════════════

def plot_sweep(sweep_dir: str, dataset_name: str):
    """Render every figure driven by sweep_results.csv."""
    df = _safe_read_csv(os.path.join(sweep_dir, "sweep_results.csv"))
    if df is None or df.empty:
        return
    sensors = sorted(df["sensor"].unique())
    strategies = sorted(df["strategy"].unique())
    first = sensors[0]
    w_mid = int(sorted(df["w"].unique())[len(df["w"].unique()) // 2])
    eps_mid = float(sorted(df["epsilon"].unique())[len(df["epsilon"].unique()) // 2])
    p_mid = int(sorted(df["P"].unique())[len(df["P"].unique()) // 2])
    ncols = min(3, len(sensors))
    nrows = (len(sensors) + ncols - 1) // ncols

    # 1 -- MAE vs epsilon, one panel per strategy.
    fig, axes = plt.subplots(1, len(strategies),
                             figsize=(4 * len(strategies), 4.5), sharey=True)
    if len(strategies) == 1:
        axes = [axes]
    sub = df[(df["sensor"] == first) & (df["w"] == w_mid)]
    for ax, strat in zip(axes, strategies):
        sd = sub[sub["strategy"] == strat]
        for P in sorted(sd["P"].unique()):
            d = sd[sd["P"] == P].sort_values("epsilon")
            ax.plot(d["epsilon"], d["mae"], marker="o", label=f"P={P}")
        ax.set(xlabel="eps", ylabel="MAE", title=strat)
        ax.legend(fontsize=7); ax.set_xscale("log"); ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"{dataset_name}: MAE vs eps [{first}, w={w_mid}]",
                 fontsize=13)
    _savefig(os.path.join(sweep_dir, f"{dataset_name}_mae_vs_epsilon.png"))

    # 2 -- MAE vs P (bar per sensor, uniform strategy).
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)
    sub = df[(df["epsilon"] == eps_mid) & (df["w"] == w_mid)
             & (df["strategy"] == "uniform")]
    for idx, sensor in enumerate(sensors):
        ax = axes[idx // ncols][idx % ncols]
        sd = sub[sub["sensor"] == sensor].sort_values("P")
        ax.bar(sd["P"].astype(str), sd["mae"], color="steelblue", alpha=0.8)
        ax.set(xlabel="P", ylabel="MAE", title=sensor)
        ax.grid(True, alpha=0.3, axis="y")
    _hide_unused(axes, len(sensors), nrows, ncols)
    fig.suptitle(
        f"{dataset_name}: MAE vs P (eps={eps_mid}, w={w_mid}, uniform)",
        fontsize=13)
    _savefig(os.path.join(sweep_dir, f"{dataset_name}_mae_vs_P.png"))

    # 3 -- Strategy comparison (grouped bar per sensor).
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)
    sub = df[(df["epsilon"] == eps_mid) & (df["w"] == w_mid)
             & (df["P"] == p_mid)]
    palette = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b2", "#937860"]
    for idx, sensor in enumerate(sensors):
        ax = axes[idx // ncols][idx % ncols]
        sd = sub[sub["sensor"] == sensor]
        strats = sorted(sd["strategy"].unique())
        vals = [sd[sd["strategy"] == s]["mae"].mean() for s in strats]
        ax.bar(strats, vals, color=palette[:len(strats)], alpha=0.85)
        ax.set(ylabel="MAE", title=sensor)
        ax.tick_params(axis="x", rotation=30, labelsize=7)
        ax.grid(True, alpha=0.3, axis="y")
    _hide_unused(axes, len(sensors), nrows, ncols)
    fig.suptitle(
        f"{dataset_name}: Strategy Comparison "
        f"(eps={eps_mid}, w={w_mid}, P={p_mid})",
        fontsize=13)
    _savefig(os.path.join(sweep_dir, f"{dataset_name}_strategy_comparison.png"))

    # 4 -- Time series from sweep_messages.csv if available.
    msg_path = os.path.join(os.path.dirname(sweep_dir), "messages",
                            "sweep_messages.csv")
    if os.path.exists(msg_path):
        msgs = _safe_read_csv(msg_path)
        if msgs is not None and not msgs.empty:
            fig, axes = plt.subplots(nrows, ncols,
                                     figsize=(6 * ncols, 4 * nrows),
                                     squeeze=False)
            for idx, sensor in enumerate(sensors):
                ax = axes[idx // ncols][idx % ncols]
                shown = False
                for strat, color in (("uniform", "tab:red"),
                                     ("n_weighted", "tab:green")):
                    sub_msg = msgs[
                        (msgs["sensor"] == sensor)
                        & (msgs["strategy"] == strat)
                        & (msgs["P"] == p_mid)
                        & (msgs["epsilon"] == eps_mid)
                        & (msgs["w"] == w_mid)
                    ].sort_values("tau")
                    if sub_msg.empty:
                        continue
                    shown = True
                    clip = sub_msg.head(200)
                    ax.plot(clip["tau"], clip["true_aggregate"],
                            "k-", alpha=0.7, lw=1.2,
                            label="True" if strat == "uniform" else None)
                    ax.plot(clip["tau"], clip["noisy_value"],
                            color=color, alpha=0.55, lw=1, label=strat)
                if shown:
                    ax.legend(fontsize=7)
                ax.set(xlabel="tau", ylabel=sensor,
                       title=f"{sensor} (eps={eps_mid}, P={p_mid})")
                ax.grid(True, alpha=0.3)
            _hide_unused(axes, len(sensors), nrows, ncols)
            fig.suptitle(f"{dataset_name}: True vs DP-Protected Streams",
                         fontsize=13)
            _savefig(os.path.join(sweep_dir, f"{dataset_name}_timeseries.png"))

    # 5 -- KL vs epsilon per strategy.
    fig, axes = plt.subplots(1, len(strategies),
                             figsize=(4 * len(strategies), 4.5), sharey=True)
    if len(strategies) == 1:
        axes = [axes]
    sub = df[(df["sensor"] == first) & (df["w"] == w_mid)]
    for ax, strat in zip(axes, strategies):
        sd = sub[sub["strategy"] == strat]
        for P in sorted(sd["P"].unique()):
            d = sd[sd["P"] == P].sort_values("epsilon")
            ax.plot(d["epsilon"], d["kl_divergence"], marker="o", label=f"P={P}")
        ax.set(xlabel="eps", ylabel="KL divergence", title=strat)
        ax.legend(fontsize=7); ax.set_xscale("log"); ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
    fig.suptitle(
        f"{dataset_name}: KL Divergence vs eps [{first}, w={w_mid}]",
        fontsize=13)
    _savefig(os.path.join(sweep_dir, f"{dataset_name}_kl_vs_epsilon.png"))

    # 6 -- Release rate vs P (one panel per strategy).
    fig, axes = plt.subplots(1, len(strategies),
                             figsize=(4 * len(strategies), 4.5), sharey=True)
    if len(strategies) == 1:
        axes = [axes]
    sub = df[(df["sensor"] == first) & (df["w"] == w_mid)
             & (df["epsilon"] == eps_mid)]
    for ax, strat in zip(axes, strategies):
        sd = sub[sub["strategy"] == strat].sort_values("P")
        ax.plot(sd["P"], sd["release_rate"], "o-", color="tab:purple")
        ax.set(xlabel="P", ylabel="release rate", title=strat, ylim=(0, 1.05))
        ax.grid(True, alpha=0.3)
    fig.suptitle(
        f"{dataset_name}: Release rate vs P [{first}, eps={eps_mid}, w={w_mid}]",
        fontsize=13)
    _savefig(os.path.join(sweep_dir, f"{dataset_name}_release_rate_vs_P.png"))


def plot_intro_figures(intro_dir: str, dataset_name: str):
    """Render the four intro figures from their CSVs."""
    # Extreme 1: global true vs noisy timeseries
    #   (columns: t, global_true, global_noisy, num_pubs_total).
    p = os.path.join(intro_dir, f"{dataset_name}_figure_extreme1_global.csv")
    df = _safe_read_csv(p)
    if df is not None and not df.empty and "global_true" in df.columns:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        n = min(300, len(df))
        t_axis = df["t"].values[:n] if "t" in df.columns else df.index[:n]
        ax.plot(t_axis, df["global_true"].values[:n],
                "k-", lw=1.2, label="True global")
        if "global_noisy" in df.columns:
            ax.plot(t_axis, df["global_noisy"].values[:n],
                    "tab:red", alpha=0.6, lw=1, label="Extreme 1 (global DP)")
        ax.set(xlabel="tau", ylabel="global mean",
               title=f"{dataset_name}: Extreme 1 -- one stream per system")
        ax.legend(); ax.grid(True, alpha=0.3)
        _savefig(os.path.join(intro_dir,
                              f"{dataset_name}_figure_extreme1_global.png"))

    # Extreme 2: per-publisher DP
    #   (columns: publisher_id, t, true, noisy).
    p = os.path.join(intro_dir,
                     f"{dataset_name}_figure_extreme2_per_publisher.csv")
    df = _safe_read_csv(p)
    if df is not None and not df.empty and "publisher_id" in df.columns:
        pubs = sorted(df["publisher_id"].unique())
        ncols = min(2, max(1, len(pubs)))
        nrows = (len(pubs) + ncols - 1) // ncols
        if pubs:
            fig, axes = plt.subplots(nrows, ncols,
                                     figsize=(6 * ncols, 3.5 * nrows),
                                     squeeze=False)
            for idx, pub in enumerate(pubs):
                ax = axes[idx // ncols][idx % ncols]
                sd = df[df["publisher_id"] == pub].sort_values("t").head(200)
                t_axis = sd["t"].values if "t" in sd.columns else sd.index
                if "true" in sd.columns:
                    ax.plot(t_axis, sd["true"].values,
                            "k-", lw=1.2, label="True")
                if "noisy" in sd.columns:
                    ax.plot(t_axis, sd["noisy"].values,
                            "tab:red", alpha=0.6, label="Per-pub DP (n=1)")
                ax.set(title=pub); ax.legend(fontsize=7)
                ax.grid(True, alpha=0.3)
            _hide_unused(axes, len(pubs), nrows, ncols)
            fig.suptitle(f"{dataset_name}: Extreme 2 -- one stream per publisher",
                         fontsize=13)
            _savefig(os.path.join(
                intro_dir, f"{dataset_name}_figure_extreme2_per_publisher.png"))

    # Extremes vs Ours KL bar.
    p = os.path.join(intro_dir, f"{dataset_name}_figure_extremes_vs_ours.csv")
    df = _safe_read_csv(p)
    if df is not None and not df.empty and "regime" in df.columns:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        agg = df.groupby("regime")["kl_divergence"].mean()
        colors = ["#e74c3c" if r != "ours" else "#2ecc71" for r in agg.index]
        ax.bar(agg.index, agg.values, color=colors, alpha=0.9)
        ax.set(ylabel="Average KL divergence",
               title=f"{dataset_name}: Extreme regimes vs our P-allocation")
        ax.grid(True, alpha=0.3, axis="y")
        _savefig(os.path.join(
            intro_dir, f"{dataset_name}_figure_extremes_vs_ours.png"))

    # U-shape / Figure 1.
    for candidate in (f"{dataset_name}_figure_u_shaped_P_vs_KL.csv",
                      f"{dataset_name}_figure1_ushape.csv",
                      f"{dataset_name}_figure1_kl_vs_P.csv"):
        p = os.path.join(intro_dir, candidate)
        df = _safe_read_csv(p)
        if df is None or df.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x_col = "P_label" if "P_label" in df.columns else "P_scope"
        if x_col not in df.columns:
            continue
        agg = df.groupby(x_col)["kl_divergence"].mean()
        ax.plot(agg.index.astype(str), agg.values, "o-", color="#3498db", lw=1.5)
        ax.set(xlabel="aggregation scope P",
               ylabel="average KL divergence",
               title=f"{dataset_name}: Figure 1 (U-shape across P)")
        ax.grid(True, alpha=0.3)
        _savefig(os.path.join(intro_dir, candidate.replace(".csv", ".png")))
        break


def plot_tuning(tuning_dir: str, dataset_name: str):
    """Render per-sensor Algorithm 2 greedy vs brute-force curves."""
    brute_paths = sorted(
        glob(os.path.join(tuning_dir, f"{dataset_name}_*_tuning_brute_force.csv"))
    )
    for brute_p in brute_paths:
        base = brute_p.replace("_tuning_brute_force.csv", "")
        greedy_p = base + "_tuning_greedy.csv"
        summary_p = base + "_tuning_strategy_summary.csv"
        brute = _safe_read_csv(brute_p)
        greedy = _safe_read_csv(greedy_p)
        summary = _safe_read_csv(summary_p)
        if brute is None or brute.empty:
            continue
        strategies = sorted(brute["strategy"].unique())
        ncols = min(3, len(strategies)) or 1
        nrows = (len(strategies) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(5.2 * ncols, 4.0 * nrows),
                                 squeeze=False)
        palette = plt.cm.tab10(np.linspace(0, 1, max(len(strategies), 1)))
        for idx, (color, strat) in enumerate(zip(palette, strategies)):
            ax = axes[idx // ncols][idx % ncols]
            bsub = brute[brute["strategy"] == strat].sort_values("P")
            if bsub.empty:
                continue
            ax.plot(bsub["P"], bsub["tuning_loss"], "o-", color=color, lw=1.4,
                    alpha=0.85, label="brute force (every P)")
            if greedy is not None and not greedy.empty:
                gsub = greedy[
                    (greedy["strategy"] == strat)
                    & greedy["action"].str.startswith(("seed", "probe", "step",
                                                       "r1_", "r2_", "r3_",
                                                       "r4_", "r5_"))
                ]
                ax.scatter(gsub["P"], gsub["tuning_loss"], marker="x", s=70,
                           color="black", zorder=5, label="greedy probe")
            if summary is not None and not summary.empty:
                srow = summary[summary["strategy"] == strat]
                if not srow.empty:
                    gP = int(srow.iloc[0]["greedy_P"])
                    gL = float(srow.iloc[0]["greedy_loss"])
                    bP = int(srow.iloc[0]["brute_P"])
                    bL = float(srow.iloc[0]["brute_loss"])
                    ax.scatter([gP], [gL], marker="*", s=300, color="gold",
                               edgecolor="black", zorder=6,
                               label=f"greedy best (P={gP})")
                    ax.scatter([bP], [bL], marker="D", s=110, color="red",
                               edgecolor="white", zorder=6,
                               label=f"brute-force best (P={bP})")
            ax.set(xlabel="P", ylabel="tuning loss", title=strat)
            ax.legend(fontsize=6); ax.grid(True, alpha=0.3)
        for idx in range(len(strategies), nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)
        fig.suptitle(
            f"{os.path.basename(base)}: Algorithm 2 greedy vs brute-force",
            fontsize=12)
        _savefig(base + "_tuning.png")


def plot_extras(extras_dir: str, dataset_name: str):
    """Render n_weighted / collusion / K_ext extras."""
    # n_weighted spotlight.
    p = os.path.join(extras_dir, f"{dataset_name}_n_weighted_spotlight.csv")
    df = _safe_read_csv(p)
    if df is not None and not df.empty and "cv_n_tau" in df.columns:
        adv_col = next(
            (c for c in ("kl_advantage",
                         "kl_advantage_uniform_minus_nweighted")
             if c in df.columns),
            None,
        )
        if adv_col is None and {"kl_uniform", "kl_n_weighted"} <= set(df.columns):
            df = df.copy()
            df["kl_advantage"] = df["kl_uniform"] - df["kl_n_weighted"]
            adv_col = "kl_advantage"
        if adv_col is not None:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.scatter(df["cv_n_tau"], df[adv_col],
                       s=60, alpha=0.8, c="tab:blue")
            ax.axhline(0, color="red", ls="--", alpha=0.6)
            ax.set(xlabel="CV of n_tau",
                   ylabel="KL(uniform) - KL(n-weighted)",
                   title=f"{dataset_name}: n-weighted advantage vs n_tau variance")
            ax.grid(True, alpha=0.3)
            _savefig(os.path.join(extras_dir,
                                  f"{dataset_name}_n_weighted_spotlight.png"))

    # Collusion.
    p = os.path.join(extras_dir, f"{dataset_name}_collusion.csv")
    df = _safe_read_csv(p)
    if df is not None and not df.empty and "num_colluders_c" in df.columns:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(df["num_colluders_c"], df["empirical_mae_ratio"],
                "o-", color="tab:red", label="Empirical MAE ratio")
        ax.plot(df["num_colluders_c"], df["predicted_mae_ratio"],
                "s--", color="tab:blue", label="Predicted 1/√c")
        ax.set(xlabel="Number of colluding subscribers c",
               ylabel="MAE ratio vs c=1",
               title=f"{dataset_name}: Collusion shrinks noise by 1/√c")
        ax.legend(); ax.grid(True, alpha=0.3)
        ax.set_xscale("log"); ax.set_yscale("log")
        _savefig(os.path.join(extras_dir, f"{dataset_name}_collusion.png"))

    # K_ext sweep.
    for p in glob(os.path.join(extras_dir, f"{dataset_name}_*_k_ext_sweep.csv")):
        df = _safe_read_csv(p)
        if df is None or df.empty:
            continue
        base = p.replace("_k_ext_sweep.csv", "")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(df["K_ext"], df["mean_wait_dt"], "o-", color="tab:blue",
                     label="mean")
        axes[0].plot(df["K_ext"], df["max_wait_dt"], "s--", color="tab:purple",
                     label="max")
        axes[0].set(xlabel="K_ext", ylabel="Wait (× dt)",
                    title=f"{os.path.basename(base)}: Latency vs K_ext")
        axes[0].legend(); axes[0].grid(True, alpha=0.3)
        ax2 = axes[1]; ax3 = ax2.twinx()
        ax2.plot(df["K_ext"], df["kl_divergence"], "o-", color="tab:red",
                 label="KL")
        ax3.plot(df["K_ext"], df["normalized_mae"], "s--", color="tab:green",
                 label="NMAE")
        ax2.set(xlabel="K_ext", ylabel="KL divergence")
        ax3.set_ylabel("Normalized MAE")
        ax2.set_title("Utility vs K_ext"); ax2.grid(True, alpha=0.3)
        ax2.legend(loc="upper left"); ax3.legend(loc="upper right")
        _savefig(base + "_k_ext.png")


# ═════════════════════════════════════════════════════════════════════════
#  Cross-dataset / single-axis experiment plots
# ═════════════════════════════════════════════════════════════════════════

def plot_experiment_A(exp_dir: str):
    df = _safe_read_csv(
        os.path.join(exp_dir, "experiment_A_greedy_vs_brute.csv"))
    if df is None or df.empty:
        return
    datasets_present = sorted(df["dataset"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    mean_speedup = df.groupby("dataset")["speedup"].mean().reindex(datasets_present)
    axes[0].bar(mean_speedup.index, mean_speedup.values,
                color="#3498db", alpha=0.85, edgecolor="white")
    for i, v in enumerate(mean_speedup.values):
        if np.isfinite(v):
            axes[0].text(i, v + 0.05, f"{v:.2f}x", ha="center",
                         fontsize=10, fontweight="bold")
    axes[0].set(ylabel="mean speedup (brute / greedy evals)",
                title="Experiment A: Algorithm 2 greedy vs naive enumeration",
                xlabel="dataset")
    axes[0].grid(True, alpha=0.3, axis="y")
    axes[0].tick_params(axis="x", rotation=30, labelsize=8)

    axes[1].scatter(df["brute_loss"], df["greedy_loss"], s=40, alpha=0.7,
                    c=df["dataset"].astype("category").cat.codes, cmap="tab10")
    mn = min(df["brute_loss"].min(), df["greedy_loss"].min())
    mx = max(df["brute_loss"].max(), df["greedy_loss"].max())
    axes[1].plot([mn, mx], [mn, mx], "k--", lw=0.7, alpha=0.5)
    axes[1].set(xlabel="brute-force optimum loss",
                ylabel="greedy result loss",
                title="greedy vs brute-force loss (identity = optimal)")
    axes[1].grid(True, alpha=0.3)
    _savefig(os.path.join(exp_dir, "experiment_A_speedup_and_gap.png"))


def _plot_single_axis(csv_path: str, out_png: str, x_col: str, x_label: str,
                      title: str, logx: bool = False):
    df = _safe_read_csv(csv_path)
    if df is None or df.empty:
        return
    datasets = sorted(df["dataset"].unique())
    ncols = min(3, len(datasets)) or 1
    nrows = (len(datasets) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols * 2,
        figsize=(5.0 * ncols * 2, 3.6 * nrows),
        squeeze=False,
    )
    palette = plt.cm.tab10(np.linspace(0, 1, 10))
    for idx, ds_name in enumerate(datasets):
        ax_nmae = axes[idx // ncols][2 * (idx % ncols)]
        ax_kl = axes[idx // ncols][2 * (idx % ncols) + 1]
        sub = df[df["dataset"] == ds_name]
        fixed_cols = [c for c in ["strategy", "P", "epsilon", "w"]
                      if c != x_col and c in sub.columns]
        combos = sub[fixed_cols + ["sensor"]].drop_duplicates().to_dict("records")
        for j, combo in enumerate(combos):
            mask = pd.Series(True, index=sub.index)
            for k, v in combo.items():
                mask &= (sub[k] == v)
            line = sub[mask].sort_values(x_col)
            if line.empty:
                continue
            label = (f"{combo.get('sensor','')} / "
                     f"{combo.get('strategy','')} P={combo.get('P','')}")
            c = palette[j % 10]
            ax_nmae.plot(line[x_col], line["normalized_mae"], "o-",
                         color=c, lw=1.2, alpha=0.85, label=label)
            ax_kl.plot(line[x_col], line["kl_divergence"], "o-",
                       color=c, lw=1.2, alpha=0.85, label=label)
        for ax, ylabel in [(ax_nmae, "NMAE"), (ax_kl, "KL div.")]:
            ax.set(xlabel=x_label, ylabel=ylabel, title=f"{ds_name} / {ylabel}")
            if logx:
                ax.set_xscale("log")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=5, ncol=2)
    for idx in range(len(datasets), nrows * ncols):
        axes[idx // ncols][2 * (idx % ncols)].set_visible(False)
        axes[idx // ncols][2 * (idx % ncols) + 1].set_visible(False)
    fig.suptitle(title, fontsize=12)
    _savefig(out_png)


def plot_experiment_B(exp_dir: str, clamp_mode: str):
    _plot_single_axis(
        os.path.join(exp_dir, "experiment_B_vary_w.csv"),
        os.path.join(exp_dir, "experiment_B_vary_w.png"),
        x_col="w", x_label="window size  w",
        title=f"Experiment B [clamp={clamp_mode}]: NMAE and KL vs w",
    )


def plot_experiment_C(exp_dir: str, clamp_mode: str):
    _plot_single_axis(
        os.path.join(exp_dir, "experiment_C_vary_epsilon.csv"),
        os.path.join(exp_dir, "experiment_C_vary_epsilon.png"),
        x_col="epsilon", x_label="privacy budget  ε",
        title=f"Experiment C [clamp={clamp_mode}]: NMAE and KL vs ε",
        logx=True,
    )


def plot_experiment_E(dataset_exp_dir: str):
    """Render the Experiment E live-broker PNG from its CSV.

    ``dataset_exp_dir`` is the per-dataset sub-directory under
    ``<output_dir>/experiments/E_live_broker/<dataset>/``.
    """
    df = _safe_read_csv(
        os.path.join(dataset_exp_dir, "experiment_E_live_broker.csv"))
    if df is None or df.empty:
        return
    ok = df[df["status"] == "ok"].copy() if "status" in df.columns else df
    if ok.empty:
        return

    # Build config labels (strategy+eps+scenario when scenario is present).
    def _label(row):
        parts = [str(row.get("strategy", "?")),
                 f"eps={row.get('epsilon', '?')}"]
        if "scenario" in row and pd.notna(row.get("scenario")):
            parts.append(row["scenario"])
        return "\n".join(parts)

    labels = [_label(r) for r in ok.to_dict("records")]
    x = np.arange(len(labels))
    bw = 0.38
    n_panels = 4 if "walkup_rate" in ok.columns else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4.5))

    # (a) NMAE live vs offline
    axes[0].bar(x - bw / 2, ok["nmae_live"], bw,
                label="live broker", color="C0")
    axes[0].bar(x + bw / 2, ok["nmae_offline"], bw,
                label="offline (same seed)", color="C1")
    axes[0].set(xticks=x, ylabel="NMAE",
                title="(a) Utility: live broker vs offline engine")
    axes[0].set_xticklabels(labels, fontsize=7, rotation=0)
    axes[0].grid(True, alpha=0.3, axis="y")
    axes[0].legend(fontsize=8)

    # (b) KL live vs offline
    axes[1].bar(x - bw / 2, ok["kl_live"], bw,
                label="live broker", color="C0")
    axes[1].bar(x + bw / 2, ok["kl_offline"], bw,
                label="offline (same seed)", color="C1")
    axes[1].set(xticks=x, ylabel="KL divergence",
                title="(b) Distributional utility")
    axes[1].set_xticklabels(labels, fontsize=7, rotation=0)
    axes[1].grid(True, alpha=0.3, axis="y")
    axes[1].legend(fontsize=8)

    # (c) Broker fan-out integrity
    axes[2].bar(x - bw / 2, ok["num_plugin_published"], bw,
                label="plugin published", color="C2")
    axes[2].bar(x + bw / 2, ok["broker_deliveries"], bw,
                label="subscribers received", color="C3")
    if "expected_deliveries" in ok.columns:
        axes[2].plot(x, ok["expected_deliveries"],
                     "k--", lw=0.7, alpha=0.6, label="expected (pub × subs)")
    axes[2].set(xticks=x, ylabel="count",
                title="(c) Broker fan-out (multi-subscriber)")
    axes[2].set_xticklabels(labels, fontsize=7, rotation=0)
    axes[2].grid(True, alpha=0.3, axis="y")
    axes[2].legend(fontsize=8)

    # (d) Walk-up + P-gate violations
    if n_panels == 4:
        axes[3].bar(x - bw / 2, ok["walkup_rate"], bw,
                    label="walk-up rate", color="C4")
        if "p_gate_violations" in ok.columns:
            axes[3].bar(x + bw / 2, ok["p_gate_violations"], bw,
                        label="P-gate violations (must be 0)", color="C5")
        axes[3].set(xticks=x,
                    title="(d) P-enforced walk-up + gate integrity")
        axes[3].set_xticklabels(labels, fontsize=7, rotation=0)
        axes[3].grid(True, alpha=0.3, axis="y")
        axes[3].legend(fontsize=8)

    title_parts = [f"Experiment E: live MQTT broker path"]
    if "dataset" in ok.columns:
        title_parts.append(f"dataset={ok['dataset'].iloc[0]}")
    if "sensor" in ok.columns:
        title_parts.append(f"sensor={ok['sensor'].iloc[0]}")
    fig.suptitle(" · ".join(title_parts), fontsize=11)
    _savefig(os.path.join(dataset_exp_dir, "experiment_E_live_broker.png"))


def plot_experiment_D(exp_dir: str, clamp_mode: str):
    df = _safe_read_csv(os.path.join(exp_dir, "experiment_D_plugin_summary.csv"))
    if df is None or df.empty:
        return
    datasets_present = sorted(df["dataset"].unique())
    scenarios = ["pooled", "hierarchy"]
    x = np.arange(len(datasets_present))
    bw = 0.38
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metrics = [
        ("release_rate", "release rate"),
        ("walkup_rate", "walk-up rate (Algorithm 1 fires)"),
        ("max_abs_diff_vs_offline", "|plugin - offline|  (pooled only)"),
    ]
    for i, (metric, ylabel) in enumerate(metrics):
        ax = axes[i]
        for j, scen in enumerate(scenarios):
            vals = []
            for ds in datasets_present:
                sub = df[(df["dataset"] == ds) & (df["scenario"] == scen)]
                vals.append(float(sub[metric].iloc[0])
                            if not sub.empty else float("nan"))
            ax.bar(x + (j - 0.5) * bw, vals, bw, label=scen, alpha=0.85)
        ax.set(xticks=x, xlabel="dataset", ylabel=ylabel, title=ylabel)
        ax.set_xticklabels(datasets_present, rotation=30, fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(fontsize=8)
    fig.suptitle(f"Experiment D: plugin end-to-end [clamp={clamp_mode}]",
                 fontsize=12)
    _savefig(os.path.join(exp_dir, "experiment_D_plugin_path.png"))


def plot_experiment_F(exp_dir: str, clamp_mode: str):
    """Ablation (Sec. 7.8): per-dataset AND aggregate module-utility curves.

    Writes one figure per dataset (experiment_F_ablation_<dataset>.png) plus an
    aggregate facet (experiment_F_ablation_all.png) showing release-rate and
    NMAE as the M1->M2->M3 modules are added, for the leaf and pooled scopes.
    """
    df = _safe_read_csv(os.path.join(exp_dir, "experiment_F_ablation.csv"))
    if df is None or df.empty or "module_idx" not in df.columns:
        return
    datasets_present = sorted(df["dataset"].unique())
    scopes = ["leaf", "pooled"]

    def _one(ax, sub, metric, ylabel):
        for scope in scopes:
            s = sub[sub["scope"] == scope].sort_values("module_idx")
            if s.empty:
                continue
            ax.plot(s["module_idx"], s[metric], "o-", label=scope, alpha=0.85)
        ax.set(xlabel="module (1=P-gate, 2=+rewrite, 3=+walk-up)",
               ylabel=ylabel, title=ylabel)
        ax.set_xticks([1, 2, 3])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    # Per-dataset figures.
    for ds in datasets_present:
        sub = df[df["dataset"] == ds]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        _one(axes[0], sub, "release_rate", "release rate")
        _one(axes[1], sub, "normalized_mae", "NMAE")
        fig.suptitle(f"Ablation [{ds}, clamp={clamp_mode}]: utility vs module",
                     fontsize=12)
        _savefig(os.path.join(exp_dir, f"experiment_F_ablation_{ds}.png"))

    # Aggregate: release-rate gain of each module, averaged across datasets.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for scope in scopes:
        s = (df[df["scope"] == scope]
             .groupby("module_idx")[["release_rate", "normalized_mae"]]
             .mean().reset_index())
        if s.empty:
            continue
        axes[0].plot(s["module_idx"], s["release_rate"], "o-", label=scope)
        axes[1].plot(s["module_idx"], s["normalized_mae"], "o-", label=scope)
    for ax, yl in ((axes[0], "mean release rate"), (axes[1], "mean NMAE")):
        ax.set(xlabel="module (cumulative)", ylabel=yl, title=yl)
        ax.set_xticks([1, 2, 3]); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle(f"Ablation (all datasets) [clamp={clamp_mode}]", fontsize=12)
    _savefig(os.path.join(exp_dir, "experiment_F_ablation_all.png"))


def plot_experiment_G(exp_dir: str, clamp_mode: str):
    """Overhead / privacy-utility (Sec. 7.9): per-dataset AND aggregate bars
    over {classic, ldp, per_type_wevent, ours} for NMAE, KL, and attribution
    advantage (identity protection; lower = better)."""
    df = _safe_read_csv(os.path.join(exp_dir, "experiment_G_overhead.csv"))
    if df is None or df.empty or "approach" not in df.columns:
        return
    order = ["classic", "ldp", "per_type_wevent", "ours"]
    metrics = [("normalized_mae", "NMAE (lower=better utility)"),
               ("kl_divergence", "KL divergence"),
               ("attribution_advantage", "attribution adv. (lower=better privacy)")]

    def _bars(sub, title, path):
        appr = [a for a in order if a in set(sub["approach"])]
        x = np.arange(len(appr))
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
        for i, (m, yl) in enumerate(metrics):
            vals = [float(sub[sub["approach"] == a][m].iloc[0]) for a in appr]
            axes[i].bar(x, vals, color=["C7", "C3", "C1", "C2"][:len(appr)])
            axes[i].set(xticks=x, ylabel=yl, title=yl)
            axes[i].set_xticklabels(appr, rotation=20, fontsize=8)
            axes[i].grid(True, alpha=0.3, axis="y")
        fig.suptitle(title, fontsize=12)
        _savefig(path)

    for ds in sorted(df["dataset"].unique()):
        _bars(df[df["dataset"] == ds],
              f"Overhead [{ds}, clamp={clamp_mode}]",
              os.path.join(exp_dir, f"experiment_G_overhead_{ds}.png"))
    # Aggregate: mean across datasets per approach.
    agg = df.groupby("approach", as_index=False)[
        ["normalized_mae", "kl_divergence", "attribution_advantage"]].mean()
    _bars(agg, f"Overhead (all datasets) [clamp={clamp_mode}]",
          os.path.join(exp_dir, "experiment_G_overhead_all.png"))


def plot_experiment_H(exp_dir: str, clamp_mode: str):
    """Average-case utility (Sec. 7.11): aggregate scatter of range-compatible
    fraction vs utility (one point per dataset) plus per-dataset bars."""
    df = _safe_read_csv(os.path.join(exp_dir, "experiment_H_average_case.csv"))
    if df is None or df.empty or "range_compatible_fraction" not in df.columns:
        return
    # Aggregate scatter: range-compat fraction (and depth h) vs release_rate / NMAE.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].scatter(df["range_compatible_fraction"], df["release_rate"], s=60)
    axes[1].scatter(df["range_compatible_fraction"], df["normalized_mae"], s=60,
                    c=df.get("topic_hierarchy_depth_h", None), cmap="viridis")
    for _i, r in df.iterrows():
        axes[0].annotate(str(r["dataset"]),
                         (r["range_compatible_fraction"], r["release_rate"]),
                         fontsize=7)
        axes[1].annotate(str(r["dataset"]),
                         (r["range_compatible_fraction"], r["normalized_mae"]),
                         fontsize=7)
    axes[0].set(xlabel="range-compatible fraction |P_R|/|P|",
                ylabel="release rate", title="coverage vs range-compatibility")
    axes[1].set(xlabel="range-compatible fraction |P_R|/|P|",
                ylabel="NMAE", title="utility vs range-compatibility (color=depth h)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Average-case utility (all datasets) [clamp={clamp_mode}]",
                 fontsize=12)
    _savefig(os.path.join(exp_dir, "experiment_H_average_case_all.png"))


def plot_cross_dataset(cross_dir: str):
    """Render cross_dataset/*.csv into their plots."""
    combined = _safe_read_csv(
        os.path.join(cross_dir, "combined_sweep_results.csv"))
    if combined is not None and not combined.empty:
        clamp_modes = sorted(combined["clamp_mode"].unique()) \
            if "clamp_mode" in combined.columns else ["static"]
        for clamp_mode in clamp_modes:
            sub = combined[combined["clamp_mode"] == clamp_mode] \
                if "clamp_mode" in combined.columns else combined
            if sub.empty:
                continue
            datasets_present = sorted(sub["dataset"].unique())
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            for ax, metric, ylabel in [
                (axes[0], "normalized_mae", "Normalized MAE"),
                (axes[1], "kl_divergence", "KL divergence"),
            ]:
                for ds in datasets_present:
                    g = (sub[sub["dataset"] == ds]
                         .groupby("epsilon")[metric].mean().sort_index())
                    ax.plot(g.index, g.values, marker="o", label=ds)
                ax.set(xlabel="eps", ylabel=ylabel, title=f"{ylabel} vs eps")
                ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
            fig.suptitle(f"Cross-Dataset [clamp={clamp_mode}]", fontsize=13)
            _savefig(os.path.join(cross_dir,
                                  f"cross_dataset_{clamp_mode}.png"))

    f1 = _safe_read_csv(os.path.join(cross_dir, "figure1_all_datasets.csv"))
    if f1 is not None and not f1.empty and "P_label" in f1.columns:
        order = ["per-pub", "P=2", "P=3", "P=4", "P=6", "P=8", "global"]
        clamp_modes = sorted(f1["clamp_mode"].unique()) \
            if "clamp_mode" in f1.columns else [None]
        for clamp_mode in clamp_modes:
            sub = f1[f1["clamp_mode"] == clamp_mode] \
                if clamp_mode is not None else f1
            if sub.empty:
                continue
            def _bucket(row):
                if row["P_label"] == "per-pub":
                    return "per-pub"
                if row["P_label"] == "global":
                    return "global"
                return f"P={int(row['P_scope'])}"
            sub = sub.copy()
            sub["bucket"] = sub.apply(_bucket, axis=1)
            buckets = [b for b in order if b in sub["bucket"].unique()]
            per_bucket = (sub.groupby("bucket")["kl_divergence"]
                          .agg(["mean", "std"]).reindex(buckets))
            out_dir = os.path.join(cross_dir, clamp_mode) \
                if clamp_mode else cross_dir
            os.makedirs(out_dir, exist_ok=True)
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            datasets_present = sorted(sub["dataset"].unique())
            palette = plt.cm.tab10(
                np.linspace(0, 1, max(len(datasets_present), 1)))
            for color, ds in zip(palette, datasets_present):
                ds_sub = sub[sub["dataset"] == ds]\
                    .set_index("bucket").reindex(buckets)
                axes[0].plot(ds_sub.index, ds_sub["kl_divergence"],
                             marker="o", label=ds, color=color, alpha=0.85)
            axes[0].set(xlabel="Aggregation scope P",
                        ylabel="KL divergence",
                        title="Per-dataset U-shape")
            axes[0].grid(True, alpha=0.3); axes[0].legend(fontsize=8)

            means = per_bucket["mean"]
            stds = per_bucket["std"].fillna(0.0)
            colors = ["#e74c3c"] + ["#2ecc71"] * (len(buckets) - 2) + ["#e74c3c"]
            bars = axes[1].bar(means.index, means.values, yerr=stds.values,
                               color=colors, alpha=0.85, edgecolor="white",
                               capsize=4)
            for bar, v in zip(bars, means.values):
                if np.isfinite(v):
                    axes[1].text(bar.get_x() + bar.get_width() / 2,
                                 bar.get_height() + 0.02, f"{v:.2f}",
                                 ha="center", va="bottom", fontsize=9,
                                 fontweight="bold")
            axes[1].set(xlabel="Aggregation scope P",
                        ylabel="Avg KL divergence",
                        title=f"Figure 1 avg ({len(datasets_present)} datasets)")
            axes[1].grid(True, alpha=0.3, axis="y")
            _savefig(os.path.join(out_dir, "figure1_all_datasets.png"))

    gap = _safe_read_csv(
        os.path.join(cross_dir, "combined_tuning_gap_summary.csv"))
    if gap is not None and not gap.empty:
        clamp_modes = sorted(gap["clamp_mode"].unique()) \
            if "clamp_mode" in gap.columns else ["static"]
        fig, axes = plt.subplots(
            1, len(clamp_modes),
            figsize=(6.5 * len(clamp_modes), 4.5),
            sharey=True, squeeze=False,
        )
        for idx, clamp_mode in enumerate(clamp_modes):
            ax = axes[0][idx]
            sub = gap[gap["clamp_mode"] == clamp_mode] \
                if "clamp_mode" in gap.columns else gap
            if sub.empty:
                ax.set_visible(False); continue
            pivot = sub.pivot_table(index="dataset", columns="strategy",
                                    values="gap_loss", aggfunc="mean")
            pivot.plot(kind="bar", ax=ax, width=0.85, alpha=0.85)
            ax.set(title=f"greedy - brute-force loss gap  [{clamp_mode}]",
                   ylabel="loss gap (greedy - brute)")
            ax.axhline(0, color="black", lw=0.8, alpha=0.6)
            ax.grid(True, alpha=0.3, axis="y")
            ax.legend(fontsize=7, loc="best")
        _savefig(os.path.join(cross_dir, "tuning_greedy_vs_brute_gap.png"))


# ═════════════════════════════════════════════════════════════════════════
#  Entry point
# ═════════════════════════════════════════════════════════════════════════

def run_all(output_dir: str, only: set[str] | None = None):
    only = only or {"sweep", "intro", "tuning", "extras", "experiments",
                    "cross_dataset"}
    if not os.path.isdir(output_dir):
        raise SystemExit(f"--output-dir {output_dir} does not exist")

    # Per-dataset / per-clamp_mode
    for name in sorted(os.listdir(output_dir)):
        ds_dir = os.path.join(output_dir, name)
        if not os.path.isdir(ds_dir) or name == "cross_dataset":
            continue
        for clamp_mode in sorted(os.listdir(ds_dir)):
            mode_dir = os.path.join(ds_dir, clamp_mode)
            if not os.path.isdir(mode_dir):
                continue
            if "sweep" in only:
                plot_sweep(os.path.join(mode_dir, "sweep"), name)
            if "intro" in only:
                plot_intro_figures(os.path.join(mode_dir, "intro"), name)
            if "tuning" in only:
                plot_tuning(os.path.join(mode_dir, "tuning"), name)
            if "extras" in only:
                plot_extras(os.path.join(mode_dir, "extras"), name)

    # Cross-dataset + single-axis experiments
    cross_dir = os.path.join(output_dir, "cross_dataset")
    if os.path.isdir(cross_dir):
        if "cross_dataset" in only:
            plot_cross_dataset(cross_dir)
        if "experiments" in only:
            for clamp_mode in sorted(os.listdir(cross_dir)):
                cm_dir = os.path.join(cross_dir, clamp_mode)
                exp_root = os.path.join(cm_dir, "experiments")
                if not os.path.isdir(exp_root):
                    continue
                plot_experiment_A(os.path.join(exp_root, "A_greedy_vs_brute"))
                plot_experiment_B(os.path.join(exp_root, "B_vary_w"), clamp_mode)
                plot_experiment_C(os.path.join(exp_root, "C_vary_epsilon"),
                                  clamp_mode)
                plot_experiment_D(os.path.join(exp_root, "D_plugin_path"),
                                  clamp_mode)
                plot_experiment_F(os.path.join(exp_root, "F_ablation"),
                                  clamp_mode)
                plot_experiment_G(os.path.join(exp_root, "G_overhead"),
                                  clamp_mode)
                plot_experiment_H(os.path.join(exp_root, "H_average_case"),
                                  clamp_mode)

    # Experiment E lives at the top level (not per-clamp-mode), with one
    # sub-directory per dataset written by --run-live-E-after-full or
    # --experiment E.
    if "experiments" in only:
        e_root = os.path.join(output_dir, "experiments", "E_live_broker")
        if os.path.isdir(e_root):
            for dataset_name in sorted(os.listdir(e_root)):
                ds_dir = os.path.join(e_root, dataset_name)
                if os.path.isdir(ds_dir):
                    plot_experiment_E(ds_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Render every PNG referenced by the paper from the "
                    "CSVs under --output-dir.  Expects the layout written "
                    "by run_experiment.py."
    )
    parser.add_argument("--output-dir", default="results_3",
                        help="Directory that run_experiment.py wrote to.")
    parser.add_argument(
        "--only", default=None,
        help="Comma-separated subset of {sweep,intro,tuning,extras,"
             "experiments,cross_dataset}.  Default: render every category.",
    )
    args = parser.parse_args()
    only = None
    if args.only:
        only = {s.strip() for s in args.only.split(",") if s.strip()}
    run_all(args.output_dir, only=only)
    logger.info("generate_plots.py: all requested figures rendered.")


if __name__ == "__main__":
    main()
