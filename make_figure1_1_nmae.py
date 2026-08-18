#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_streams import prepare_dataset, _stable_hash
from experiments.engine import (
    _run_parallel_tasks,
    _dp_named_task,
    _init_named_streams_worker,
    _default_workers,
    logger,
)


def figure1_1_nmae(per_pub_all, dataset_name, output_dir,
                   epsilon=1.0, w=6, n_trials=20, workers=1):
    os.makedirs(output_dir, exist_ok=True)

    # rebuild the per-sensor aggregate streams (as figure1 does)
    sensor_streams: dict[str, tuple[list[float], list[int], float]] = {}
    for s, (pp, B) in per_pub_all.items():
        T_s = len(next(iter(pp.values())))
        agg, cnt = [], []
        for tau in range(T_s):
            vals = [pp[p][tau] for p in pp if pp[p][tau] is not None]
            agg.append(float(np.mean(vals)) if vals else 0.0)
            cnt.append(len(vals))
        sensor_streams[s] = (agg, cnt, B)

    sensor_names = list(sensor_streams.keys())
    T = min(len(v[0]) for v in sensor_streams.values())
    N_pubs = max(len(pp) for pp, _ in per_pub_all.values())
    P_values = [p for p in [2, 3, 4, 6, 8] if p <= N_pubs]

    # Extreme 1 stream: one cross-metric stream per system
    all_B = max(B for _, B in per_pub_all.values())
    cross_metric_true: list[float] = []
    num_pubs_total: list[int] = []
    for tau in range(T):
        vals = [sensor_streams[s][0][tau] for s in sensor_names
                if sensor_streams[s][1][tau] > 0]
        cross_metric_true.append(float(np.mean(vals)) if vals else 0.0)
        num_pubs_total.append(sum(sensor_streams[s][1][tau] for s in sensor_names))

    named: dict = {}
    for s, (pp, B) in per_pub_all.items():
        for pub_id, series in pp.items():
            pa = [v if v is not None else 0.0 for v in series]
            pc = [1 if v is not None else 0 for v in series]
            named[("pub", s, pub_id)] = (pa, pc, B)
        named[("agg", s)] = sensor_streams[s]
    named[("global",)] = (cross_metric_true, num_pubs_total, all_B)

    # task list (same seeds as figure1 so the two figures are paired)
    tasks, ops, idx = [], [], 0
    for s in sensor_names:
        pp, _ = per_pub_all[s]
        for i, pub_id in enumerate(pp.keys()):
            for trial in range(n_trials):
                tasks.append((idx, ("pub", s, pub_id), "uniform", 1,
                              epsilon, w, 500_000 + i * 1000 + trial, "metrics"))
                ops.append(("p1",))
                idx += 1
    for P in P_values:
        for s in sensor_names:
            if max(sensor_streams[s][1]) < P:
                continue
            for trial in range(n_trials):
                tasks.append((idx, ("agg", s), "uniform", P,
                              epsilon, w, 600_000 + P * 1000 + trial, "metrics"))
                ops.append(("mid", P))
                idx += 1
    for s in sensor_names:
        for trial in range(n_trials):
            tasks.append((idx, ("global",), "uniform", 1, epsilon, w,
                          700_000 + _stable_hash(s) + trial * 7919, "noisy"))
            ops.append(("global", s))
            idx += 1

    logger.info(f"  [{dataset_name}] figure1_1_nmae: dispatching {len(tasks)} "
                f"DP runs across {workers} worker(s)")
    results = _run_parallel_tasks(
        tasks, _dp_named_task, workers=workers,
        initializer=_init_named_streams_worker, initargs=(named,),
        progress_label=f"  [{dataset_name}] figure1.1",
        progress_every=max(50, len(tasks) // 20),
    )

    # bucket the NMAEs
    p1: list[float] = []
    mid: dict[int, list[float]] = {P: [] for P in P_values}
    glob: list[float] = []
    for op, res in zip(ops, results):
        if op[0] == "p1":
            v = res["nmae"]
            if np.isfinite(v):
                p1.append(v)
        elif op[0] == "mid":
            v = res["nmae"]
            if np.isfinite(v):
                mid[op[1]].append(v)
        else:
            s = op[1]
            R_s = sensor_streams[s][2]
            true_stream = sensor_streams[s][0][:T]
            nvals = res["noisy_arr"]
            errs = [abs(t - float(n)) for t, n in zip(true_stream, nvals)
                    if t is not None and np.isfinite(n)]
            if errs and R_s > 0:
                v = float(np.mean(errs)) / R_s
                if np.isfinite(v):
                    glob.append(v)

    def _m(v):
        return float(np.mean(v)) if v else float("nan")

    def _sd(v):
        return float(np.std(v)) if v else float("nan")

    rows = [{"dataset": dataset_name, "P_scope": 1, "P_label": "per-pub",
             "nmae": _m(p1), "nmae_std": _sd(p1), "n_runs": len(p1),
             "epsilon": epsilon, "w": w, "N": N_pubs}]
    for P in P_values:
        rows.append({"dataset": dataset_name, "P_scope": P, "P_label": f"P={P}",
                     "nmae": _m(mid[P]), "nmae_std": _sd(mid[P]),
                     "n_runs": len(mid[P]),
                     "epsilon": epsilon, "w": w, "N": N_pubs})
    rows.append({"dataset": dataset_name, "P_scope": N_pubs + 1, "P_label": "global",
                 "nmae": _m(glob), "nmae_std": _sd(glob), "n_runs": len(glob),
                 "epsilon": epsilon, "w": w, "N": N_pubs})
    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, f"{dataset_name}_figure1_1_nmae_vs_P.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"  Figure-1.1 CSV saved: {csv_path}")

    # plot (log y: Extreme 1 is orders of magnitude worse)
    labels = df["P_label"].tolist()
    values = df["nmae"].tolist()
    finite = [v for v in values if np.isfinite(v)]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#e74c3c"] + ["#2ecc71"] * len(P_values) + ["#e74c3c"]
    bars = ax.bar(labels, values, color=colors, alpha=0.85, edgecolor="white")
    ax.set_yscale("log")
    for bar, v in zip(bars, values):
        if np.isfinite(v):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.08,
                    f"{v:.3g}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
    ax.set(xlabel="Aggregation scope P",
           ylabel="Average NMAE (log scale, lower = better)",
           title=f"{dataset_name}: Point-wise Error vs. Aggregation Scope"
                 f"  (eps={epsilon}, w={w})")
    ax.grid(True, alpha=0.3, axis="y", which="both")
   
    ax.axhline(1.0, color="#34495e", ls="--", lw=1.2, alpha=0.8)
    ax.text(len(labels) - 0.4, 1.05, "error = payload bound R",
            ha="right", va="bottom", fontsize=8, color="#34495e")
    if finite:
        ax.set_ylim(top=max(finite) * 4)
    plt.tight_layout()
    png_path = os.path.join(output_dir, f"{dataset_name}_figure1_1_nmae_vs_P.png")
    plt.savefig(png_path, dpi=150)
    plt.savefig(png_path.replace(".png", ".pdf"))
    plt.close()
    logger.info(f"  Figure-1.1 plot saved: {png_path}")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="manufacturing")
    ap.add_argument("--clamp-mode", default="static")
    ap.add_argument("--epsilon", type=float, default=1.0)
    ap.add_argument("--w", type=int, default=6)     # w_mid = max(w_values)//2
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--eps-clip", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--output-dir", default=None,
                    help="default: results_combined/<dataset>/<clamp_mode>/intro")
    args = ap.parse_args()

    out = args.output_dir or os.path.join(
        "results_combined", args.dataset, args.clamp_mode, "intro")
    prepared = prepare_dataset(args.dataset, clamp_mode=args.clamp_mode,
                               eps_clip=args.eps_clip, seed=args.seed,
                               max_rows=args.max_rows)
    if prepared is None or prepared.is_empty:
        raise SystemExit(f"could not prepare dataset {args.dataset}")
    df = figure1_1_nmae(prepared.per_pubs, args.dataset, out,
                        epsilon=args.epsilon, w=args.w, n_trials=args.trials,
                        workers=_default_workers(args.workers))
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
