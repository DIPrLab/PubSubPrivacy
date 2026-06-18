#!/usr/bin/env python3
"""Render the publication figures for experiment.tex from the per-trial
aggregate CSVs (paper_bundle / results_combined).  Outputs PDF + PNG into
figures/.  Run: python make_paper_figures.py
"""
from __future__ import annotations
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "results_combined/cross_dataset/static/experiments"
OUT = "figures"
os.makedirs(OUT, exist_ok=True)
DS = ["energy", "traffic", "wearable", "pune", "mobility", "manufacturing"]
COL = dict(zip(DS, plt.cm.tab10(np.linspace(0, 1, 10))))
MK = dict(zip(DS, ["o", "s", "^", "D", "v", "P"]))
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig)
    print("wrote", name)


def fig_B():
    """NMAE vs w (linear) + KL vs w (flat), uniform/P=2/level-1."""
    d = pd.read_csv(f"{BASE}/B_vary_w/experiment_B_vary_w_aggregate.csv")
    d = d[(d.strategy == "uniform") & (d.P == 2) & (d.subscription_level == 1)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for ds in DS:
        g = d[d.dataset == ds].groupby("w", as_index=False)[
            ["normalized_mae_mean", "normalized_mae_std", "kl_divergence_mean"]].mean()
        a1.errorbar(g.w, g.normalized_mae_mean, yerr=g.normalized_mae_std,
                    marker=MK[ds], color=COL[ds], lw=1.6, capsize=2, label=ds)
        a2.plot(g.w, g.kl_divergence_mean, marker=MK[ds], color=COL[ds], lw=1.6, label=ds)
    # ideal linear reference (normalized to energy@w=4)
    w = np.array([4, 6, 8, 10, 12, 16])
    base = d[(d.dataset == "energy") & (d.w == 4)].normalized_mae_mean.mean()
    a1.plot(w, base * w / 4, "k--", lw=1.0, alpha=0.6, label=r"linear $\propto w$")
    a1.set(xlabel="window size $w$", ylabel="NMAE", title="(a) NMAE grows linearly in $w$")
    a2.set(xlabel="window size $w$", ylabel="KL divergence", title="(b) KL is ~flat in $w$")
    a1.legend(fontsize=8, ncol=2)
    _save(fig, "exp_B_window")


def fig_C():
    """NMAE vs eps on log-log (slope -1), uniform/P=2/level-1."""
    d = pd.read_csv(f"{BASE}/C_vary_epsilon/experiment_C_vary_epsilon_aggregate.csv")
    d = d[(d.strategy == "uniform") & (d.P == 2) & (d.subscription_level == 1)]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for ds in DS:
        g = d[d.dataset == ds].groupby("epsilon", as_index=False)["normalized_mae_mean"].mean()
        ax.plot(g.epsilon, g.normalized_mae_mean, marker=MK[ds], color=COL[ds], lw=1.6, label=ds)
    eps = np.array(sorted(d.epsilon.unique()))
    base = d[(d.dataset == "energy") & (d.epsilon == 1.0)].normalized_mae_mean.mean()
    ax.plot(eps, base / eps, "k--", lw=1.0, alpha=0.6, label=r"$\propto 1/\epsilon$")
    ax.set(xscale="log", yscale="log", xlabel=r"privacy budget $\epsilon$",
           ylabel="NMAE", title=r"NMAE $\propto 1/\epsilon$ (slope $-1$ on log-log)")
    ax.legend(fontsize=8, ncol=2)
    _save(fig, "exp_C_epsilon")


def fig_F():
    """Ablation: release rate + NMAE by cumulative module (leaf scope)."""
    d = pd.read_csv(f"{BASE}/F_ablation/experiment_F_ablation_aggregate.csv")
    d = d[d.scope == "leaf"]
    mods = ["M1_pgate", "M2_interval_ext", "M3_walk_up"]
    lab = ["M1\nP-gate", "M2\n+interval", "M3\n+walk-up"]
    x = np.arange(len(mods)); bw = 0.13
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for j, ds in enumerate(DS):
        g = d[d.dataset == ds].set_index("module")
        rr = [g.loc[m, "release_rate_mean"] if m in g.index else np.nan for m in mods]
        nm = [g.loc[m, "normalized_mae_mean"] if m in g.index else np.nan for m in mods]
        off = (j - len(DS) / 2 + 0.5) * bw
        a1.bar(x + off, rr, bw, color=COL[ds], label=ds)
        a2.bar(x + off, nm, bw, color=COL[ds], label=ds)
    a1.set(xticks=x, ylabel="release rate", title="(a) Walk-up restores delivery at the leaf")
    a1.set_xticklabels(lab, fontsize=8)
    a2.set(xticks=x, ylabel="NMAE", title="(b) Walk-up collapses leaf NMAE")
    a2.set_xticklabels(lab, fontsize=8)
    a1.legend(fontsize=8, ncol=2)
    _save(fig, "exp_F_ablation")


def fig_G():
    """Overhead: NMAE (utility) + attribution advantage (privacy) by approach."""
    d = pd.read_csv(f"{BASE}/G_overhead/experiment_G_overhead_aggregate.csv")
    d = d[d.subscription_level == 1]
    appr = ["ldp", "per_type_wevent", "ours"]
    lab = {"ldp": "LDP (P=1)", "per_type_wevent": "per-type\n$w$-event", "ours": "ours"}
    x = np.arange(len(DS)); bw = 0.25
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.3))
    for k, ap in enumerate(appr):
        nm = [d[(d.dataset == ds) & (d.approach == ap)].normalized_mae_mean.mean() for ds in DS]
        aa = [d[(d.dataset == ds) & (d.approach == ap)].attribution_advantage_mean.mean() for ds in DS]
        a1.bar(x + (k - 1) * bw, nm, bw, label=lab[ap])
        a2.bar(x + (k - 1) * bw, aa, bw, label=lab[ap])
    a1.set(xticks=x, ylabel="NMAE", title="(a) Utility: ours $\\approx$ per-type, both $\\ll$ LDP")
    a1.set_xticklabels(DS, rotation=30, fontsize=8)
    a2.axhline(1.0, color="r", ls=":", alpha=0.6)
    a2.set(xticks=x, ylabel="attribution advantage  $1/n_\\tau$",
           title="(b) Privacy: ours $\\ll$ LDP's full exposure (=1)")
    a2.set_xticklabels(DS, rotation=30, fontsize=8)
    a1.legend(fontsize=9); a2.legend(fontsize=9)
    _save(fig, "exp_G_overhead")


def fig_L():
    """NMAE vs subscription level (root->leaf) per dataset."""
    d = pd.read_csv(f"{BASE}/I_subscription_levels/experiment_I_subscription_levels_aggregate.csv")
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for ds in DS:
        g = d[d.dataset == ds].groupby("subscription_level", as_index=False)["normalized_mae_mean"].mean()
        ax.plot(g.subscription_level, g.normalized_mae_mean, marker=MK[ds], color=COL[ds], lw=1.6, label=ds)
    ax.set(xlabel="subscription level (1 = type root $\\to$ deepest = leaf)",
           ylabel="NMAE", title="Utility vs. subscription depth")
    ax.legend(fontsize=8, ncol=2)
    _save(fig, "exp_L_levels")


def fig_latency():
    """K_ext induced latency: suppression-vs-wait tradeoff (the sparse datasets)."""
    import glob
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for f in sorted(glob.glob("paper_bundle/08_extras_kext_sec7.10/*_k_ext_sweep.csv")):
        ds = os.path.basename(f).split("__")[0]
        d = pd.read_csv(f).sort_values("K_ext")
        a1.plot(d.K_ext, d.suppression_rate, marker=MK[ds], color=COL[ds], lw=1.6, label=ds)
        a2.plot(d.K_ext, d.mean_wait_dt, marker=MK[ds], color=COL[ds], lw=1.6, label=ds)
    a1.set(xlabel="$K_{ext}$", ylabel="suppression rate",
           title="(a) $K_{ext}$ cuts suppression (sparse pools)")
    a2.set(xlabel="$K_{ext}$", ylabel="mean wait ($\\Delta t$ units)",
           title="(b) at a small added delivery delay")
    a1.legend(fontsize=8, ncol=2)
    _save(fig, "exp_latency_kext")


if __name__ == "__main__":
    fig_B(); fig_C(); fig_F(); fig_G(); fig_L(); fig_latency()
    print("all figures ->", OUT)
