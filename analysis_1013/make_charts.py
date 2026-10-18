#!/usr/bin/env python3
"""Render the charts and summary tables for analysis_1013/ANALYSIS.md from the
10/13 cluster run (results_combined_1013/ + paper_bundle_1013/shards/).

Run from the repo root:  python analysis_1013/make_charts.py
"""
from __future__ import annotations

import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

logging.disable(logging.INFO)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from experiments import engine as core  # noqa: E402

E = os.path.join(REPO, "results_combined_1013", "cross_dataset", "static", "experiments")
SHARDS = os.path.join(REPO, "paper_bundle_1013", "shards")
OUT = os.path.join(REPO, "analysis_1013", "figures")
os.makedirs(OUT, exist_ok=True)
DS = ["energy", "traffic", "wearable", "pune", "mobility", "manufacturing"]

# Validated categorical palette (fixed slot order; adjacent CVD dE >= 9.1).
# Every series also gets its own marker and a legend entry.
COL = {"n_weighted": "#2a78d6", "wevent": "#eda100", "ldp": "#eb6834",
       "p_gated_sample": "#1baf7a", "p_gated_uniform": "#e87ba4",
       "classic": "#7a7a75"}
MK = {"n_weighted": "o", "wevent": "s", "ldp": "^", "p_gated_sample": "D",
      "p_gated_uniform": "v", "classic": "x"}
LAB = {"n_weighted": "PAWS n-weighted (ours)", "wevent": "w-event DP",
       "ldp": "local DP", "p_gated_sample": "PAWS p-gated sample",
       "p_gated_uniform": "PAWS p-gated uniform", "classic": "no privacy"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5df"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
    "lines.linewidth": 2, "lines.markersize": 6, "axes.axisbelow": True,
})


def rd(rel):
    return pd.read_csv(os.path.join(E, rel))


def save(fig, name):
    fig.savefig(os.path.join(OUT, f"{name}.png"))
    plt.close(fig)
    print("wrote", name)


def eps_axis(ax, values):
    vals = sorted(set(float(v) for v in values))
    ax.set_xscale("log")
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xticks(vals)
    ax.set_xticklabels([f"{v:g}" for v in vals])


def lines(ax, table, x, series):
    for s in series:
        if s not in table.index:
            continue
        y = table.loc[s]
        ax.plot(table.columns, y.values, marker=MK[s], color=COL[s], label=LAB[s],
                ls=":" if s in ("wevent", "ldp", "classic") else "-")


# ---------------------------------------------------------------- data
u = rd("U_utility/experiment_U_epsilon_summary.csv")
ul = rd("U_utility/experiment_U_epsilon_by_level.csv")
uw = rd("U_utility/experiment_U_window_summary.csv")
m = rd("M_ml_utility/experiment_M_ml_summary.csv")
k = rd("K_induced_wait/experiment_K_induced_wait_summary.csv")
f = rd("F_ablation/experiment_F_ablation_summary.csv")
j = rd("J_latency/experiment_J_latency_inprocess.csv")
d = rd("D_plugin_path/experiment_D_plugin_summary.csv")
MAIN = ["n_weighted", "wevent", "ldp"]
PAWS = ["n_weighted", "p_gated_uniform", "p_gated_sample", "wevent", "ldp"]

# 1. NMAE vs epsilon
t = u.pivot_table(index="strategy", columns="epsilon", values="normalized_mae_mean",
                  aggfunc="median")
fig, ax = plt.subplots(figsize=(6.4, 4.2))
lines(ax, t, "epsilon", MAIN)
ax.set_yscale("log")
eps_axis(ax, t.columns)
ax.set(xlabel="privacy budget ε  (w = 8)", ylabel="NMAE (log scale; lower is better)",
       title="Error vs budget, median over the six datasets")
ax.legend()
save(fig, "01_nmae_vs_eps")

# 2. NMAE ratio per dataset at eps 4
r = u[u.epsilon == 4.0].pivot_table(index="dataset", columns="strategy",
                                     values="normalized_mae_mean").reindex(DS)
fig, ax = plt.subplots(figsize=(6.4, 3.8))
x = np.arange(len(DS))
bw = 0.38
for i, s in enumerate(["wevent", "ldp"]):
    ax.bar(x + (i - 0.5) * bw, (r[s] / r["n_weighted"]).values, bw * 0.92,
           color=COL[s], label=f"{LAB[s]} / ours")
ax.axhline(1.0, color=MUTED, lw=1)
ax.set(xticks=x, ylabel="error ratio (× ours' NMAE)",
       title="How many times larger the baselines' error is (ε = 4)")
ax.set_xticklabels(DS, fontsize=9)
ax.legend()
save(fig, "02_nmae_ratio_eps4")

# 3. skill vs prior: root vs deeper
ul["kind"] = np.where(ul.subscription_level == 1, "root", "deeper")
lv = ul.groupby(["strategy", "dataset", "epsilon", "kind"], as_index=False)["skill_prior_mean"].mean()
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, kind, title in [(axes[0], "root", "Root subscription (whole sensor type)"),
                        (axes[1], "deeper", "Deeper subscriptions (subtree / leaf)")]:
    tt = lv[lv.kind == kind].pivot_table(index="strategy", columns="epsilon",
                                         values="skill_prior_mean", aggfunc="median")
    lines(ax, tt, "epsilon", PAWS)
    ax.axhline(0.0, color=INK, lw=1)
    ax.set_ylim(-6, 1)
    eps_axis(ax, tt.columns)
    ax.set(xlabel="privacy budget ε  (w = 8)", title=title)
axes[0].set_ylabel("skill vs range midpoint\n(> 0: more informative than the declared range)")
h, lb = axes[0].get_legend_handles_labels()
fig.legend(h, lb, loc="lower center", ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.06))
fig.suptitle("Skill against the data-independent constant, median over datasets "
             "(values below −6 clipped)", fontsize=10, color=MUTED)
save(fig, "03_skill_root_vs_deeper")

# 4. all strategies at eps 4: skill at root
order = ["p_gated_sample", "sample", "p_gated_uniform", "n_weighted", "p_gated_ba",
         "p_gated_bd", "uniform", "budget_absorption", "budget_distribution", "wevent", "ldp"]
tr = lv[(lv.kind == "root") & (lv.epsilon == 4.0)].groupby("strategy")["skill_prior_mean"].median()
tr = tr.reindex([s for s in order if s in tr.index]).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(6.8, 4.4))
colors = [COL["n_weighted"] if s == "n_weighted" else
          (COL["wevent"] if s in ("wevent", "ldp") else "#9ec5f4") for s in tr.index]
ax.barh(np.arange(len(tr)), tr.values, color=colors, height=0.72)
ax.axvline(0, color=INK, lw=1)
ax.set_yticks(np.arange(len(tr)))
ax.set_yticklabels([s + ("  (ours)" if s == "n_weighted" else "") for s in tr.index], fontsize=9)
ax.invert_yaxis()
for i, v in enumerate(tr.values):
    ax.text(v + (0.04 if v >= 0 else -0.04), i, f"{v:+.2f}", va="center",
            ha="left" if v >= 0 else "right", fontsize=8, color=INK)
ax.set(xlabel="skill vs range midpoint (root, ε = 4, median over datasets)",
       title="Every budget strategy at its own grid optimum")
ax.set_xlim(min(-3.0, tr.min() - 0.5), 1.0)
save(fig, "04_strategies_eps4_root")

# 5. window sweep
tw = uw.pivot_table(index="strategy", columns="w", values="normalized_mae_mean", aggfunc="median")
fig, ax = plt.subplots(figsize=(6.4, 4.2))
lines(ax, tw, "w", ["n_weighted", "p_gated_sample", "wevent", "ldp"])
ax.set(xlabel="window length w  (ε = 1)", ylabel="NMAE (lower is better)",
       title="Error vs window length, median over datasets")
ax.set_xticks(tw.columns)
ax.legend()
save(fig, "05_nmae_vs_w")

# 6. ML forecast
tm = m.pivot_table(index="strategy", columns="epsilon", values="ml_forecast_nmae_mean",
                   aggfunc="median")
fig, ax = plt.subplots(figsize=(6.4, 4.2))
lines(ax, tm, "epsilon", ["n_weighted", "p_gated_sample", "wevent", "ldp", "classic"])
ax.set_yscale("log")
eps_axis(ax, tm.columns)
ax.set(xlabel="privacy budget ε", ylabel="forecast NMAE vs truth (log scale)",
       title="Downstream next-value forecast trained on the received stream")
ax.legend(fontsize=8)
save(fig, "06_ml_forecast")

# 7. induced wait (ours)
ko = k[k.strategy == "n_weighted"].groupby(["wait_policy", "k_ext"], as_index=False)[
    ["mean_wait_dt_mean", "skill_prior_mean"]].mean()
fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
pol = {"predictive": ("#2a78d6", "o", "-"), "early_stop": ("#1baf7a", "D", "--"),
       "reactive": ("#eb6834", "^", ":")}
base = ko[ko.wait_policy == "none"]
for p, (c, mk, ls) in pol.items():
    g = ko[ko.wait_policy == p].sort_values("k_ext")
    axes[0].plot(g.k_ext, g.mean_wait_dt_mean, color=c, marker=mk, ls=ls, label=p)
    axes[1].plot(g.k_ext, g.skill_prior_mean, color=c, marker=mk, ls=ls, label=p)
axes[1].axhline(float(base.skill_prior_mean.iloc[0]), color=INK, lw=1, ls="-")
axes[1].text(8, float(base.skill_prior_mean.iloc[0]) - 0.25, "no waiting", ha="right",
             fontsize=8, color=INK)
for ax in axes:
    ax.set_xticks([1, 2, 4, 8])
    ax.set_xlabel("K_ext")
axes[0].set(ylabel="mean extra wait (Δt units)", title="Wait added per interval (ours)")
axes[1].set(ylabel="skill vs range midpoint (all levels)",
            title="Utility it buys (higher is better)")
axes[0].legend()
save(fig, "07_induced_wait")

# 8. ablation (ours, leaf)
fl = f[(f.strategy == "n_weighted") & (f.scope_kind == "leaf")]
mods = ["M1_pgate", "M2_interval_ext", "M3_walk_up"]
mlab = ["M1 P-gate", "M2 + adaptive Δt", "M3 + walk-up"]
epss = [1.0, 4.0, 8.0]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
x = np.arange(len(epss))
bw = 0.26
shades = ["#9ec5f4", "#5598e7", "#184f95"]
for i, (mo, lb) in enumerate(zip(mods, mlab)):
    g = fl[fl.module == mo].groupby("epsilon")[["coverage_mean", "skill_prior_mean"]].mean().reindex(epss)
    axes[0].bar(x + (i - 1) * bw, g.coverage_mean.values, bw * 0.92, color=shades[i], label=lb)
    axes[1].bar(x + (i - 1) * bw, g.skill_prior_mean.values, bw * 0.92, color=shades[i], label=lb)
for ax in axes:
    ax.set_xticks(x)
    ax.set_xticklabels([f"ε = {e:g}" for e in epss])
axes[1].axhline(0, color=INK, lw=1)
axes[0].set(ylabel="coverage (share of timestamps delivered)",
            title="Leaf subscription: delivery (ours)", ylim=(0, 1))
axes[1].set(ylabel="skill vs range midpoint",
            title="Leaf subscription: informativeness (ours)")
h, lb = axes[0].get_legend_handles_labels()
fig.legend(h, lb, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.06))
save(fig, "08_ablation_leaf")

# 9. broker latency
jm = j.groupby(["config", "strategy"], as_index=False)[
    ["broker_cpu_ms_per_dt", "ingest_us_per_msg", "close_ms_p95"]].median()
pick = [("passthrough", "-", "forward only"), ("wevent", "uniform", "w-event DP"),
        ("paws", "n_weighted", "PAWS (ours)"), ("paws_nocache", "n_weighted", "PAWS, no count cache"),
        ("paws_nowait", "n_weighted", "PAWS, no adaptive Δt")]
vals, labs = [], []
for c, s, lb in pick:
    row = jm[(jm.config == c) & (jm.strategy == s)]
    if not row.empty:
        vals.append(float(row.broker_cpu_ms_per_dt.iloc[0]))
        labs.append(lb)
fig, ax = plt.subplots(figsize=(6.4, 3.6))
cols = [COL["classic"], COL["wevent"], COL["n_weighted"], "#5598e7", "#5598e7"][:len(vals)]
ax.barh(np.arange(len(vals)), vals, color=cols, height=0.6)
ax.set_yticks(np.arange(len(vals)))
ax.set_yticklabels(labs)
ax.invert_yaxis()
for i, v in enumerate(vals):
    ax.text(v + 0.03, i, f"{v:.2f} ms", va="center", fontsize=8, color=INK)
ax.set(xlabel="broker CPU per Δt tick (ms, median over datasets and trials)",
       title="Broker processing cost")
ax.set_xlim(0, max(vals) * 1.25)
save(fig, "09_broker_cpu")

# 10. grid choices (boundary check)
rows = []
for ds in DS:
    cfg = core._load_grid_config(os.path.join(SHARDS, f"{ds}__static__grid", "grid_canonical.json"))
    for (_d, _c, st, eps), rec in cfg.items():
        rows.append(dict(dataset=_d, strategy=st, epsilon=eps, P_min=rec.get("P_min"),
                         rho=rec.get("rho_split"), k_ext=rec.get("k_ext"),
                         P_max=rec.get("P_max"), delta_t=rec.get("delta_t")))
g = pd.DataFrame(rows)
pa = g[g.strategy.isin(["n_weighted", "p_gated_uniform", "p_gated_sample", "p_gated_bd", "p_gated_ba"])]
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
for ax, col, grid, title in [
        (axes[0], "rho", [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8], "ρ (count share)"),
        (axes[1], "P_min", [1, 2, 3, 4, 6], "P_min"),
        (axes[2], "k_ext", [0, 2, 4], "K_ext")]:
    cnt = pa[col].value_counts().reindex(grid, fill_value=0)
    edge = [grid[0], grid[-1]]
    ax.bar([str(v) for v in grid], cnt.values,
           color=["#2a78d6" if v in edge else "#9ec5f4" for v in grid])
    ax.set(title=f"{title}: chosen value", xlabel="grid value (darker = edge of range)")
axes[0].set_ylabel("population-aware optima (of 180)")
save(fig, "10_grid_choices")

# ---------------------------------------------------------------- tables
tab = {}
tab["nmae"] = t.reindex(PAWS).round(2)
tab["skill_root"] = lv[lv.kind == "root"].pivot_table(index="strategy", columns="epsilon",
                                                       values="skill_prior_mean", aggfunc="median").reindex(PAWS).round(2)
tab["skill_deeper"] = lv[lv.kind == "deeper"].pivot_table(index="strategy", columns="epsilon",
                                                           values="skill_prior_mean", aggfunc="median").reindex(PAWS).round(2)
tab["corr"] = u.pivot_table(index="strategy", columns="epsilon", values="corr_mean",
                            aggfunc="median").reindex(PAWS).round(3)
tab["ratio_eps4"] = (r[["wevent", "ldp"]].div(r["n_weighted"], axis=0)).round(2)
tab["ml_bal_acc_eps4"] = m[m.epsilon == 4.0].pivot_table(index="dataset", columns="strategy",
                                                          values="ml_class_bal_acc_mean").reindex(DS)[
    ["classic", "n_weighted", "p_gated_sample", "wevent", "ldp"]].round(2)
tab["wait_s"] = k[(k.strategy == "n_weighted") & (k.wait_policy == "predictive") & (k.k_ext == 4)
                  ].set_index("dataset")["mean_wait_s_mean"].reindex(DS).round(0)
tab["latency"] = jm.round(3)
tab["d"] = d.groupby("scenario")[["max_abs_diff_vs_offline", "p_gate_violations",
                                  "releases_below_P_exact", "num_releases"]].mean().round(4)
for name, df in tab.items():
    print(f"\n### {name}\n{df.to_string()}")
