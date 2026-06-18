#!/usr/bin/env python3
"""Emit native pgfplots (TikZ) figure macros for experiment.tex from the
per-trial aggregate CSVs.  Every figure has ONE series per individual dataset
(not a cross-dataset average) and is drawn by LaTeX in Overleaf with inline
coordinates -- no external image binaries.  Writes experiment_figures.tex,
which experiment.tex \\input's; the main document must load pgfplots.

Run: python make_pgfplots.py
"""
from __future__ import annotations
import os, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

BASE = "results_combined/cross_dataset/static/experiments"
DS = ["energy", "traffic", "wearable", "pune", "mobility", "manufacturing"]
# distinct color + mark per dataset (defined once in the preamble macro below)
STYLE = {
    "energy":        ("c0", "*"),
    "traffic":       ("c1", "square*"),
    "wearable":      ("c2", "triangle*"),
    "pune":          ("c3", "diamond*"),
    "mobility":      ("c4", "pentagon*"),
    "manufacturing": ("c5", "x"),
}
out = []


def coords(xs, ys):
    pts = " ".join(f"({x},{y:.5g})" for x, y in zip(xs, ys)
                   if y is not None and np.isfinite(y))
    return pts


def line_axis(series, xlabel, ylabel, title, opts=""):
    """series: list of (dataset, xs, ys).  Returns an axis env with one
    \\addplot per dataset."""
    s = [f"\\begin{{axis}}[width=\\linewidth,height=5.2cm,xlabel={{{xlabel}}},"
         f"ylabel={{{ylabel}}},title={{{title}}},grid=both,grid style={{gray!20}},"
         f"legend style={{font=\\tiny,at={{(0.5,-0.28)}},anchor=north,legend columns=3}},"
         f"label style={{font=\\small}},title style={{font=\\small}},{opts}]"]
    for ds, xs, ys in series:
        c, m = STYLE[ds]
        s.append(f"\\addplot[{c},mark={m},line width=0.9pt] coordinates {{{coords(xs,ys)}}};")
        s.append(f"\\addlegendentry{{\\texttt{{{ds}}}}}")
    s.append("\\end{axis}")
    return "\n".join(s)


# ---- B: NMAE vs w (linear) + KL vs w (flat) -------------------------------
B = pd.read_csv(f"{BASE}/B_vary_w/experiment_B_vary_w_aggregate.csv")
B = B[(B.strategy == "uniform") & (B.P == 2) & (B.subscription_level == 1)]
serN, serK = [], []
for ds in DS:
    g = B[B.dataset == ds].groupby("w", as_index=False)[["normalized_mae_mean", "kl_divergence_mean"]].mean().sort_values("w")
    serN.append((ds, g.w.tolist(), g.normalized_mae_mean.tolist()))
    serK.append((ds, g.w.tolist(), g.kl_divergence_mean.tolist()))
out.append(r"\newcommand{\figExpWindow}{%" + "\n"
           r"\begin{figure}[t]\centering" + "\n"
           r"\begin{subfigure}{0.49\linewidth}\centering" + "\n"
           r"\begin{tikzpicture}" + "\n" + line_axis(serN, "window size $w$", "NMAE", "(a) NMAE $\\propto w$") + "\n"
           r"\end{tikzpicture}\end{subfigure}\hfill" + "\n"
           r"\begin{subfigure}{0.49\linewidth}\centering" + "\n"
           r"\begin{tikzpicture}" + "\n" + line_axis(serK, "window size $w$", "KL divergence", "(b) KL $\\approx$ flat") + "\n"
           r"\end{tikzpicture}\end{subfigure}" + "\n"
           r"\caption{Effect of the window size $w$ ($P$-gated Uniform, $P=2$, $\epsilon=1$, type-root subscription), per dataset, 6 seeds. (a) NMAE rises linearly in $w$ on every dataset (the $w{=}16$/$w{=}4$ ratio is $\approx\!4$ on all six); (b) KL is essentially flat in $w$.}\label{fig:exp-w}" + "\n"
           r"\end{figure}}")

# ---- C: NMAE vs eps (log-log) ---------------------------------------------
C = pd.read_csv(f"{BASE}/C_vary_epsilon/experiment_C_vary_epsilon_aggregate.csv")
C = C[(C.strategy == "uniform") & (C.P == 2) & (C.subscription_level == 1)]
serC = []
for ds in DS:
    g = C[C.dataset == ds].groupby("epsilon", as_index=False)["normalized_mae_mean"].mean().sort_values("epsilon")
    serC.append((ds, g.epsilon.tolist(), g.normalized_mae_mean.tolist()))
out.append(r"\newcommand{\figExpEps}{%" + "\n"
           r"\begin{figure}[t]\centering" + "\n"
           r"\begin{tikzpicture}" + "\n"
           + line_axis(serC, r"privacy budget $\epsilon$", "NMAE",
                       r"NMAE $\propto 1/\epsilon$", opts="xmode=log,ymode=log") + "\n"
           r"\end{tikzpicture}" + "\n"
           r"\caption{Effect of the privacy budget $\epsilon$ ($P$-gated Uniform, $P=2$, $w=8$, type-root subscription), per dataset, 6 seeds. On log--log axes every dataset follows a slope-$-1$ line: NMAE $\propto 1/\epsilon$ (the $\epsilon{=}0.5$/$\epsilon{=}4$ ratio is $\approx\!8$ on all six).}\label{fig:exp-eps}" + "\n"
           r"\end{figure}}")

# ---- L: NMAE vs subscription level ----------------------------------------
L = pd.read_csv(f"{BASE}/I_subscription_levels/experiment_I_subscription_levels_aggregate.csv")
serL = []
for ds in DS:
    g = L[L.dataset == ds].groupby("subscription_level", as_index=False)["normalized_mae_mean"].mean().sort_values("subscription_level")
    serL.append((ds, g.subscription_level.tolist(), g.normalized_mae_mean.tolist()))
out.append(r"\newcommand{\figExpLevels}{%" + "\n"
           r"\begin{figure}[t]\centering" + "\n"
           r"\begin{tikzpicture}" + "\n"
           + line_axis(serL, "subscription level (1 = type root)", "NMAE",
                       "Utility vs.\\ subscription depth") + "\n"
           r"\end{tikzpicture}" + "\n"
           r"\caption{Subscriber utility at \emph{every} level of the constructed topic hierarchy (level 1 = type root; deeper = finer subtree), per dataset, 6 seeds. Finer subscriptions pool fewer publishers, so noise (NMAE) grows with depth --- the per-level cost the walk-up trades against.}\label{fig:exp-levels}" + "\n"
           r"\end{figure}}")

# ---- latency: suppression vs K_ext ----------------------------------------
serS, serW = [], []
for f in sorted(glob.glob("paper_bundle/08_extras_kext_sec7.10/*_k_ext_sweep.csv")):
    ds = os.path.basename(f).split("__")[0]
    d = pd.read_csv(f).sort_values("K_ext")
    serS.append((ds, d.K_ext.tolist(), d.suppression_rate.tolist()))
    serW.append((ds, d.K_ext.tolist(), d.mean_wait_dt.tolist()))
serS = [s for s in (next((x for x in serS if x[0] == ds), None) for ds in DS) if s]
serW = [s for s in (next((x for x in serW if x[0] == ds), None) for ds in DS) if s]
out.append(r"\newcommand{\figExpLatency}{%" + "\n"
           r"\begin{figure}[t]\centering" + "\n"
           r"\begin{subfigure}{0.49\linewidth}\centering" + "\n"
           r"\begin{tikzpicture}" + "\n" + line_axis(serS, "$K_{ext}$", "suppression rate", "(a) suppression vs.\\ $K_{ext}$") + "\n"
           r"\end{tikzpicture}\end{subfigure}\hfill" + "\n"
           r"\begin{subfigure}{0.49\linewidth}\centering" + "\n"
           r"\begin{tikzpicture}" + "\n" + line_axis(serW, "$K_{ext}$", "mean wait ($\\Delta t$)", "(b) added delivery delay") + "\n"
           r"\end{tikzpicture}\end{subfigure}" + "\n"
           r"\caption{Induced latency of the $K_{ext}$ interval extension, per dataset. (a) For the sparsest feed (\texttt{wearable}) $K_{ext}$ cuts the suppression rate from $0.15$ to $0.02$; dense feeds rarely suppress. (b) The cost is a small increase in mean delivery delay (wait $\le 1.16\,\Delta t$; $p_{95}\le 2\,\Delta t$).}\label{fig:exp-latency}" + "\n"
           r"\end{figure}}")

# ---- F: ablation -- release rate + NMAE by cumulative module (leaf) -------
F = pd.read_csv(f"{BASE}/F_ablation/experiment_F_ablation_aggregate.csv")
F = F[F.scope == "leaf"]
mods = ["M1_pgate", "M2_interval_ext", "M3_walk_up"]


def bar_axis(metric, ylabel, title, datasets):
    # Plain symbolic coords (M1,M2,M3) + explicit xticklabels -> Overleaf-safe.
    s = [f"\\begin{{axis}}[ybar,width=\\linewidth,height=5.2cm,"
         f"symbolic x coords={{M1,M2,M3}},xtick=data,"
         f"xticklabels={{M1 P-gate,M2 +interval,M3 +walk-up}},"
         f"x tick label style={{font=\\tiny,rotate=20,anchor=east}},"
         f"ylabel={{{ylabel}}},title={{{title}}},title style={{font=\\small}},label style={{font=\\small}},"
         f"bar width=2.2pt,enlarge x limits=0.25,ymin=0,"
         f"legend style={{font=\\tiny,at={{(0.5,-0.30)}},anchor=north,legend columns=3}}]"]
    for ds in datasets:
        c, _ = STYLE[ds]
        g = F[F.dataset == ds].set_index("module")
        pts = []
        for short, m in zip(["M1", "M2", "M3"], mods):
            if m in g.index and np.isfinite(g.loc[m, metric]):
                pts.append(f"({short},{g.loc[m, metric]:.4g})")
        s.append(f"\\addplot[{c},fill={c}] coordinates {{{' '.join(pts)}}};")
        s.append(f"\\addlegendentry{{\\texttt{{{ds}}}}}")
    s.append("\\end{axis}")
    return "\n".join(s)


out.append(r"\newcommand{\figExpAblation}{%" + "\n"
           r"\begin{figure}[t]\centering" + "\n"
           r"\begin{subfigure}{0.49\linewidth}\centering" + "\n"
           r"\begin{tikzpicture}" + "\n" + bar_axis("release_rate_mean", "release rate", "(a) delivery at the leaf", DS) + "\n"
           r"\end{tikzpicture}\end{subfigure}\hfill" + "\n"
           r"\begin{subfigure}{0.49\linewidth}\centering" + "\n"
           r"\begin{tikzpicture}" + "\n" + bar_axis("normalized_mae_mean", "NMAE", "(b) leaf NMAE", ["energy", "manufacturing", "mobility"]) + "\n"
           r"\end{tikzpicture}\end{subfigure}" + "\n"
           r"\caption{Incremental-module ablation at the leaf subscription, per dataset, 6 seeds. (a) With only the P-gate (M1) or the interval extension (M2), the sparse leaf ($n_\tau{=}1$) either releases at huge noise or cannot release at all (\texttt{pune}/\texttt{traffic}/\texttt{wearable} sit at release rate $0$); adding the walk-up (M3) restores delivery to $0.84$--$1.0$. (b) Where M1/M2 do release (\texttt{energy}/\texttt{manufacturing}/\texttt{mobility}) the walk-up collapses NMAE by $\sim\!8\times$ (e.g.\ $7.98\!\to\!1.02$ on \texttt{energy}).}\label{fig:exp-ablation}" + "\n"
           r"\end{figure}}")

# ---- G: overhead -- NMAE + attribution by approach, per dataset -----------
G = pd.read_csv(f"{BASE}/G_overhead/experiment_G_overhead_aggregate.csv")
G = G[G.subscription_level == 1]
appr = [("ldp", "LDP ($P{=}1$)"), ("per_type_wevent", "per-type"), ("ours", "ours")]
acol = {"ldp": "c1", "per_type_wevent": "c4", "ours": "c0"}


def overhead_axis(metric, ylabel, title, extra=""):
    sx = "{" + ",".join(DS) + "}"   # plain symbolic coords; tick labels styled below
    s = [f"\\begin{{axis}}[ybar,width=\\linewidth,height=5.2cm,symbolic x coords={sx},"
         f"xtick=data,x tick label style={{font=\\tiny,rotate=25,anchor=east}},ylabel={{{ylabel}}},"
         f"title={{{title}}},title style={{font=\\small}},label style={{font=\\small}},bar width=4pt,"
         f"enlarge x limits=0.12,ymin=0,legend style={{font=\\tiny,at={{(0.5,-0.32)}},anchor=north,legend columns=3}},{extra}]"]
    for ap, lab in appr:
        pts = []
        for ds in DS:
            sub = G[(G.dataset == ds) & (G.approach == ap)]
            if len(sub):
                pts.append(f"({ds},{sub[metric].mean():.4g})")
        s.append(f"\\addplot[{acol[ap]},fill={acol[ap]}] coordinates {{{' '.join(pts)}}};")
        s.append(f"\\addlegendentry{{{lab}}}")
    s.append("\\end{axis}")
    return "\n".join(s)


out.append(r"\newcommand{\figExpOverhead}{%" + "\n"
           r"\begin{figure}[t]\centering" + "\n"
           r"\begin{subfigure}{0.49\linewidth}\centering" + "\n"
           r"\begin{tikzpicture}" + "\n" + overhead_axis("normalized_mae_mean", "NMAE", "(a) utility (lower better)") + "\n"
           r"\end{tikzpicture}\end{subfigure}\hfill" + "\n"
           r"\begin{subfigure}{0.49\linewidth}\centering" + "\n"
           r"\begin{tikzpicture}" + "\n" + overhead_axis("attribution_advantage_mean", "attribution adv.\\ $1/n_\\tau$", "(b) exposure (lower better)") + "\n"
           r"\end{tikzpicture}\end{subfigure}" + "\n"
           r"\caption{Overhead / privacy--utility comparison at the type-root subscription, per dataset, 6 seeds. (a) Our $P$-allocation matches per-type $w$-event utility and beats local DP by $\sim\!3\times$ NMAE (e.g.\ \texttt{energy} $1.06$ vs.\ LDP $3.18$). (b) Local DP exposes every contributor (attribution advantage $=1$); ours keeps it at $1/n_\tau\!\approx\!0.10$--$0.23$ --- lower noise \emph{and} far stronger identity protection. (classic/no-privacy is the NMAE${=}0$ reference, omitted.)}\label{fig:exp-overhead}" + "\n"
           r"\end{figure}}")

with open("experiment_figures.tex", "w", encoding="utf-8") as fh:
    fh.write("% Auto-generated by make_pgfplots.py -- native pgfplots figures,\n"
             "% one series per individual dataset.  \\input from experiment.tex.\n"
             "% Requires in the MAIN preamble:\n"
             "%   \\usepackage{pgfplots}\\usepackage{subcaption}\n"
             "%   \\pgfplotsset{compat=1.17}\n"
             "% Dataset colors c0..c5:\n"
             r"\definecolor{c0}{HTML}{1F77B4}\definecolor{c1}{HTML}{FF7F0E}"
             r"\definecolor{c2}{HTML}{2CA02C}\definecolor{c3}{HTML}{D62728}"
             r"\definecolor{c4}{HTML}{9467BD}\definecolor{c5}{HTML}{8C564B}" + "\n\n")
    fh.write("\n\n".join(out) + "\n")
print("wrote experiment_figures.tex with",
      len(out), "figure macros (one series per dataset)")
