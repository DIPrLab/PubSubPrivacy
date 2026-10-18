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
import matplotlib.ticker

BASE = os.path.join(os.environ.get("PAPER_RESULTS", "results_combined"),
                    "cross_dataset", "static", "experiments")
OUT = os.environ.get("PAPER_FIGURES", "figures")
os.makedirs(OUT, exist_ok=True)
DS = ["energy", "traffic", "wearable", "pune", "mobility", "manufacturing"]
# Validated categorical palette (fixed slot order; adjacent CVD dE >= 9.1).
COL = dict(zip(DS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
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


# ── Section 7 experiments (full-pipeline simulator) ─────────────────────────
# Approaches keep fixed categorical slots (validated palette, adjacent CVD
# dE >= 9.1); the no-privacy reference is neutral gray.  Every series also
# carries its own marker and a legend, since three slots sit below 3:1
# contrast on white.
APPR = ["paws", "wevent", "ldp", "classic"]
APPR_LAB = {"paws": "PAWS (n-weighted, ours)", "wevent": "$w$-event DP", "ldp": "local DP",
            "classic": "no privacy"}
APPR_COL = {"paws": "#2a78d6", "ldp": "#eb6834", "wevent": "#eda100",
            "classic": "#7a7a75"}
APPR_MK = {"paws": "o", "wevent": "s", "ldp": "^", "classic": "x"}
MODS = ["M1_pgate", "M2_interval_ext", "M3_walk_up"]
MOD_LAB = ["M1\nP-gate", "M2\n+predictive $\\Delta t$", "M3\n+walk-up"]


def _read(path):
    if not os.path.exists(path):
        print("skip (missing)", path)
        return None
    return pd.read_csv(path)


def _panels(d):
    """One panel per dataset present in ``d``; yields (ax, dataset, rows) and
    hides the panels of datasets with no rows (partial results)."""
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.8))
    axes = axes.ravel()
    present = [(ax, ds, d[d.dataset == ds]) for ax, ds in zip(axes, DS)]
    for ax, _ds, g in present:
        if g.empty:
            ax.set_visible(False)
    return fig, axes, [p for p in present if not p[2].empty]


def _eps_ticks(ax, values):
    """Log-scale epsilon axis labelled with the budgets actually run."""
    vals = sorted(set(float(v) for v in values))
    ax.set_xscale("log")
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xticks(vals)
    ax.set_xticklabels([f"{v:g}" for v in vals])


def _ours(d):
    """Baselines + PAWS under our strategy (n_weighted) only."""
    if "ours" not in d:
        return d
    ours = d["ours"].astype(str).str.lower() == "true"
    return d[(d.approach != "paws") | ours]


def _approach_lines(ax, d, x, y, approaches=APPR):
    d = _ours(d)
    for ap in approaches:
        g = d[d.approach == ap].sort_values(x)
        if g.empty or y not in g:
            continue
        ax.plot(g[x], g[y], marker=APPR_MK[ap], color=APPR_COL[ap], lw=2,
                ms=6, label=APPR_LAB[ap])


def fig_U_epsilon():
    """NMAE (fresh releases) vs epsilon per dataset, averaged over sensors and
    every subscription level; PAWS' coverage-penalized NMAE dashed."""
    d = _read(f"{BASE}/U_utility/experiment_U_epsilon_summary.csv")
    if d is None:
        return
    fig, axes, panels = _panels(d)
    for ax, ds, g in panels:
        _approach_lines(ax, g, "epsilon", "normalized_mae_mean",
                        [a for a in APPR if a != "classic"])
        p = g[g.approach == "paws"].sort_values("epsilon")
        ax.plot(p.epsilon, p.nmae_penalized_mean, ls="--", color=APPR_COL["paws"],
                lw=1.5, label="PAWS, missing = 1")
        ax.set(yscale="log", title=ds, xlabel=r"$\epsilon$", ylabel="NMAE")
        _eps_ticks(ax, g.epsilon)
    panels[0][0].legend(fontsize=8)
    fig.suptitle("Utility vs privacy budget (all subscription levels)")
    fig.tight_layout()
    _save(fig, "exp_U_epsilon")


def fig_U_skill():
    """Skill vs epsilon per dataset: against the clamp midpoint (does the
    stream carry the level?) solid, against the true median (does it track the
    changes?) dashed.  Above 0 = more informative than that constant."""
    d = _read(f"{BASE}/U_utility/experiment_U_epsilon_summary.csv")
    if d is None or "skill_prior_mean" not in d:
        return
    d = _ours(d)
    fig, axes, panels = _panels(d)
    for ax, ds, g in panels:
        for ap in [a for a in APPR if a != "classic"]:
            h = g[g.approach == ap].sort_values("epsilon")
            if h.empty:
                continue
            ax.plot(h.epsilon, h.skill_prior_mean, marker=APPR_MK[ap], color=APPR_COL[ap],
                    lw=2, ms=6, label=APPR_LAB[ap])
            ax.plot(h.epsilon, h.skill_mean, ls="--", color=APPR_COL[ap], lw=1.2)
        ax.axhline(0.0, color="#7a7a75", lw=1)
        lo = min(-1.0, float(g.skill_prior_mean.min()))
        ax.set(ylim=(max(lo, -3.0), 1.0), title=ds, xlabel=r"$\epsilon$", ylabel="skill")
        _eps_ticks(ax, g.epsilon)
    panels[0][0].legend(fontsize=8)
    fig.suptitle("Skill vs privacy budget (solid: vs range midpoint, dashed: vs true median)")
    fig.tight_layout()
    _save(fig, "exp_U_skill")


def fig_U_strategies():
    """Every budget strategy at its own grid optimum: skill vs the range
    midpoint against epsilon, median over datasets."""
    d = _read(f"{BASE}/U_utility/experiment_U_epsilon_summary.csv")
    if d is None or "strategy" not in d or "skill_prior_mean" not in d:
        return
    g = d.groupby(["strategy", "epsilon"], as_index=False)["skill_prior_mean"].median()
    order = [s for s in ["n_weighted", "p_gated_uniform", "p_gated_sample", "p_gated_bd",
                         "p_gated_ba", "uniform", "sample", "budget_distribution",
                         "budget_absorption", "wevent", "ldp"] if s in set(g.strategy)]
    pal = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
           "#4a3aa7", "#e34948"]
    markers = "os^Dv<>Px*h"
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for i, st in enumerate(order):
        h = g[g.strategy == st].sort_values("epsilon")
        base = st in ("wevent", "ldp")
        ax.plot(h.epsilon, h.skill_prior_mean, lw=2.6 if st == "n_weighted" else 1.6,
                ls=":" if base else "-", marker=markers[i % len(markers)],
                color="#7a7a75" if base else pal[i % len(pal)],
                label=st + (" (ours)" if st == "n_weighted" else ""))
    ax.axhline(0.0, color="#7a7a75", lw=1)
    ax.set(ylim=(max(-6.0, float(g.skill_prior_mean.min())), 1.0),
           xlabel=r"$\epsilon$", ylabel="skill vs range midpoint",
           title="Every strategy at its grid optimum (median over datasets)")
    _eps_ticks(ax, g.epsilon)
    ax.legend(fontsize=7, ncol=2)
    _save(fig, "exp_U_strategies")


def fig_U_window():
    d = _read(f"{BASE}/U_utility/experiment_U_window_summary.csv")
    if d is None:
        return
    fig, axes, panels = _panels(d)
    for ax, ds, g in panels:
        _approach_lines(ax, g, "w", "normalized_mae_mean",
                        [a for a in APPR if a != "classic"])
        ax.set(yscale="log", title=ds, xlabel="window length $w$", ylabel="NMAE")
    panels[0][0].legend(fontsize=8)
    fig.suptitle("Utility vs window length")
    fig.tight_layout()
    _save(fig, "exp_U_window")


def fig_U_depth(eps=0.8):
    """NMAE vs subscription level at one budget (level 1 = domain root)."""
    d = _read(f"{BASE}/U_utility/experiment_U_epsilon_by_level.csv")
    if d is None:
        return
    eps = min(d.epsilon.unique(), key=lambda e: abs(e - eps))
    d = d[d.epsilon == eps]
    fig, axes, panels = _panels(d)
    for ax, ds, g in panels:
        _approach_lines(ax, g, "subscription_level",
                        "normalized_mae_mean", [a for a in APPR if a != "classic"])
        ax.set(yscale="log", title=ds, xlabel="subscription level (1 = root)",
               ylabel="NMAE")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Utility vs subscription depth ($\\epsilon$={eps:g})")
    fig.tight_layout()
    _save(fig, "exp_U_depth")


def fig_M():
    """Downstream ML: forecast NMAE and classification balanced accuracy."""
    d = _read(f"{BASE}/M_ml_utility/experiment_M_ml_summary.csv")
    if d is None:
        return
    for metric, ylab, name, logy in (
            ("ml_forecast_nmae_mean", "forecast NMAE", "exp_M_forecast", True),
            ("ml_class_bal_acc_mean", "balanced accuracy", "exp_M_classify", False)):
        fig, axes, panels = _panels(d)
        for ax, ds, g in panels:
            _approach_lines(ax, g, "epsilon", metric)
            ax.set(title=ds, xlabel=r"$\epsilon$", ylabel=ylab)
            _eps_ticks(ax, g.epsilon)
            if logy:
                ax.set_yscale("log")
        panels[0][0].legend(fontsize=8)
        fig.suptitle(f"Downstream ML on the received stream: {ylab}")
        fig.tight_layout()
        _save(fig, name)


def fig_K():
    """Induced wait vs K_ext per wait policy, and the utility it buys."""
    d = _read(f"{BASE}/K_induced_wait/experiment_K_induced_wait_summary.csv")
    if d is None:
        return
    if "ours" in d:
        d = d[d["ours"].astype(str).str.lower() == "true"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for ds in DS:
        for pol, ls in (("predictive", "-"), ("early_stop", "--"), ("reactive", ":")):
            g = d[(d.dataset == ds) & (d.wait_policy == pol)].sort_values("k_ext")
            if g.empty:
                continue
            lab = ds if pol == "predictive" else None
            a1.plot(g.k_ext, g.mean_wait_dt_mean, ls=ls, marker=MK[ds], color=COL[ds],
                    lw=2, label=lab)
            a2.plot(g.k_ext, g.nmae_penalized_mean, ls=ls, marker=MK[ds],
                    color=COL[ds], lw=2)
    a1.set(xlabel="$K_{ext}$", ylabel=r"mean extra wait ($\Delta t$)",
           title="(a) induced wait\n(solid: predictive, dashed: early stop, dotted: reactive)")
    a2.set(xlabel="$K_{ext}$", ylabel="NMAE (missing = 1)",
           title="(b) utility")
    for a in (a1, a2):
        a.set_xticks(sorted(d.k_ext.unique()))
    a1.legend(fontsize=8, ncol=2)
    _save(fig, "exp_K_induced_wait")


def fig_J():
    """Broker CPU per Delta_t tick by configuration, per dataset."""
    d = _read(f"{BASE}/J_latency/experiment_J_latency_inprocess.csv")
    if d is None:
        return
    if "ours" in d:   # component variants under our strategy, plus the baselines
        d = d[(d["ours"].astype(str).str.lower() == "true")
              | d.config.isin(["passthrough", "wevent"])]
    cfgs = ["passthrough", "wevent", "paws_nowait", "paws_nocache", "paws"]
    lab = ["forward\nonly", "$w$-event", "PAWS\nno wait", "PAWS\nno cache", "PAWS"]
    g = d.groupby(["dataset", "config"], as_index=False)[
        ["broker_cpu_ms_per_dt", "ingest_us_per_msg"]].mean()
    x = np.arange(len(cfgs)); bw = 0.13
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.3))
    present = [ds for ds in DS if ds in set(g.dataset)]
    for j, ds in enumerate(present):
        s = g[g.dataset == ds].set_index("config").reindex(cfgs)
        off = (j - len(present) / 2 + 0.5) * bw
        a1.bar(x + off, s.broker_cpu_ms_per_dt, bw * 0.9, color=COL[ds], label=ds)
        a2.bar(x + off, s.ingest_us_per_msg, bw * 0.9, color=COL[ds], label=ds)
    a1.set(xticks=x, ylabel=r"broker CPU per $\Delta t$ (ms)", title="(a) release work")
    a2.set(xticks=x, ylabel=r"ingest per message ($\mu$s)", title="(b) per-message work")
    for a in (a1, a2):
        a.set_xticklabels(lab, fontsize=8)
    a1.legend(fontsize=8, ncol=2, loc="upper left")
    _save(fig, "exp_J_latency")


def fig_F(eps=0.8):
    """Ablation at the leaf: coverage and fresh-release NMAE per module."""
    d = _read(f"{BASE}/F_ablation/experiment_F_ablation_summary.csv")
    if d is None:
        return
    if "ours" in d:
        d = d[d["ours"].astype(str).str.lower() == "true"]
    eps = min(d.epsilon.unique(), key=lambda e: abs(e - eps))
    d = d[(d.scope_kind == "leaf") & (d.epsilon == eps)]
    x = np.arange(len(MODS)); bw = 0.13
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    present = [ds for ds in DS if ds in set(d.dataset)]
    for j, ds in enumerate(present):
        g = d[d.dataset == ds].set_index("module").reindex(MODS)
        off = (j - len(present) / 2 + 0.5) * bw
        for ax, col in zip(axes, ("coverage_mean", "normalized_mae_mean",
                                  "nmae_penalized_mean")):
            ax.bar(x + off, g[col], bw * 0.9, color=COL[ds], label=ds)
    for ax, t in zip(axes, ("(a) coverage", "(b) NMAE, fresh releases",
                            "(c) NMAE, missing = 1")):
        ax.set(xticks=x, title=t)
        ax.set_xticklabels(MOD_LAB, fontsize=8)
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle(f"Ablation at the leaf subscription ($\\epsilon$={eps:g})")
    fig.tight_layout()
    _save(fig, "exp_F_ablation")


if __name__ == "__main__":
    for f in (fig_B, fig_C, fig_G, fig_L, fig_U_epsilon, fig_U_skill, fig_U_strategies, fig_U_window, fig_U_depth,
              fig_M, fig_K, fig_J, fig_F):
        try:
            f()
        except (FileNotFoundError, KeyError, IndexError, ValueError) as exc:
            print(f"skip {f.__name__}: {exc}")
    print("all figures ->", OUT)
