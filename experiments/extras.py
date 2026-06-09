#!/usr/bin/env python3
"""Auxiliary probes (the 'extras' phase): n-weighted spotlight, subscriber
collusion (sqrt(c) noise reduction), and the K_ext induced-latency sweep
(paper Sec. 7.10).  Runs over every dataset on the PerCom topic hierarchy, in
parallel.  Shared DP workers (_dp_named_task) live in the core.

Run: python -m experiments.extras --dataset all
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from experiments import _common as C
from experiments import engine as core


def n_weighted_spotlight(streams, dataset_name, output_dir,
                         epsilon=1.0, w=8, P=2):
    """KL_uniform - KL_n_weighted vs CV(n_tau) (paper Sec. 6.4)."""
    rows = []
    for sensor, (agg, cnt, B) in streams.items():
        cv = float(np.std(cnt) / max(np.mean(cnt), 1e-9))
        r_u = core.run_dp_on_stream(agg, cnt, epsilon, w, P, B, "uniform", seed=7)
        r_n = core.run_dp_on_stream(agg, cnt, epsilon, w, P, B, "n_weighted", seed=7)
        rows.append({
            "sensor": sensor, "cv_n_tau": cv, "mean_n_tau": float(np.mean(cnt)),
            "kl_uniform": r_u["metrics"]["kl_divergence"],
            "kl_n_weighted": r_n["metrics"]["kl_divergence"],
            "mae_uniform": r_u["metrics"]["mae"],
            "mae_n_weighted": r_n["metrics"]["mae"],
            "nmae_uniform": r_u["metrics"]["normalized_mae"],
            "nmae_n_weighted": r_n["metrics"]["normalized_mae"],
        })
    out = pd.DataFrame(rows)
    out["kl_advantage"] = out["kl_uniform"] - out["kl_n_weighted"]
    out.to_csv(os.path.join(output_dir, f"{dataset_name}_n_weighted_spotlight.csv"),
               index=False)
    core.logger.info(f"  n-weighted spotlight saved "
                     f"(CV range {out['cv_n_tau'].min():.2f}-{out['cv_n_tau'].max():.2f}).")
    return out


# K_ext induced-latency sweep (Sec. 7.10) -----------------------------------
_WORKER_PER_PUB = None


def _init_per_pub_worker(per_pub):
    global _WORKER_PER_PUB
    _WORKER_PER_PUB = per_pub


def _k_ext_task(task):
    """One k_ext merge + DP run; per_pub cached in the worker global."""
    k_ext, epsilon, w, P, payload_bound, seed = task
    per_pub = _WORKER_PER_PUB
    T = len(next(iter(per_pub.values())))
    merged_agg, merged_cnt, merged_wait = [], [], []
    pending_vals, pending_pubs, pending_waits = [], set(), 0
    for tau in range(T):
        for p, series in per_pub.items():
            v = series[tau]
            if v is not None:
                pending_vals.append(v)
                pending_pubs.add(p)
        pending_waits += 1
        if len(pending_pubs) >= P or pending_waits > k_ext:
            merged_agg.append(float(np.mean(pending_vals)) if pending_vals else 0.0)
            merged_cnt.append(len(pending_pubs))
            merged_wait.append(pending_waits)
            pending_vals, pending_pubs, pending_waits = [], set(), 0
    if pending_vals:
        merged_agg.append(float(np.mean(pending_vals)))
        merged_cnt.append(len(pending_pubs))
        merged_wait.append(pending_waits)
    m = core.run_dp_on_stream(
        merged_agg, merged_cnt, epsilon=epsilon, window_size=w, min_publishers=P,
        payload_bound=payload_bound, strategy="p_gated_ba", seed=seed)["metrics"]
    waits = np.array(merged_wait, dtype=float) if merged_wait else np.array([])
    n_rel = len(merged_agg)
    k_ext_firings = int(np.sum(waits > 1)) if waits.size else 0
    total_extensions = int(np.sum(np.maximum(waits - 1, 0))) if waits.size else 0
    under_P = int(np.sum(np.array(merged_cnt) < P)) if merged_cnt else 0
    return {
        "K_ext": k_ext, "T_max_over_dt": k_ext + 1, "num_releases": n_rel,
        "release_rate": m["release_rate"],
        "mean_wait_dt": float(np.mean(waits)) if waits.size else float("nan"),
        "p95_wait_dt": float(np.percentile(waits, 95)) if waits.size else float("nan"),
        "max_wait_dt": int(waits.max()) if waits.size else 0,
        "k_ext_firings": k_ext_firings,
        "k_ext_firing_rate": (k_ext_firings / n_rel) if n_rel else float("nan"),
        "total_extensions": total_extensions,
        "extensions_per_release": (total_extensions / n_rel) if n_rel else float("nan"),
        "suppression_rate": (under_P / n_rel) if n_rel else float("nan"),
        "mae": m["mae"], "kl_divergence": m["kl_divergence"],
        "normalized_mae": m["normalized_mae"],
    }


def dynamic_interval_experiment(per_pub, payload_bound, dataset_name, sensor_name,
                                output_dir, epsilon=1.0, w=8, P=3,
                                k_ext_values=(0, 1, 2, 4), workers=1) -> "pd.DataFrame":
    """Sec. 7.10 induced-latency: K_ext sweep reporting wait (mean/p95/max),
    K_ext firings, suppression rate, and utility."""
    tasks = [(int(k_ext), float(epsilon), int(w), int(P), float(payload_bound),
              13 + int(k_ext)) for k_ext in k_ext_values]
    rows = core._run_parallel_tasks(
        tasks, _k_ext_task, workers=min(workers, len(tasks)),
        initializer=_init_per_pub_worker, initargs=(per_pub,),
        progress_label=f"  [{dataset_name}/{sensor_name}] k_ext", progress_every=1)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(output_dir, f"{dataset_name}_{sensor_name}_k_ext_sweep.csv"),
               index=False)
    core.logger.info(f"  K_ext sweep done ({sensor_name}).")
    return out


# Subscriber collusion (paper Sec. 4.7): sqrt(c) noise reduction -------------
def collusion_experiment(streams, dataset_name, output_dir, epsilon=1.0, w=8,
                         P=2, c_values=(1, 2, 4, 8, 16), trials_per_c=32,
                         workers=1) -> "pd.DataFrame":
    sensor = next(iter(streams))
    agg, cnt, B = streams[sensor]
    max_c = max(c_values)
    named = {("collusion",): (agg, cnt, B)}
    tasks, idx = [], 0
    for trial in range(trials_per_c):
        for k in range(max_c):
            tasks.append((idx, ("collusion",), "uniform", P, epsilon, w,
                          1_000 * trial + k, "noisy"))
            idx += 1
    results = core._run_parallel_tasks(
        tasks, core._dp_named_task, workers=workers,
        initializer=core._init_named_streams_worker, initargs=(named,),
        progress_label=f"  [{dataset_name}] collusion",
        progress_every=max(50, len(tasks) // 20))
    noisy_by_tk, t_idx = {}, 0
    for trial in range(trials_per_c):
        for k in range(max_c):
            noisy_by_tk[(trial, k)] = results[t_idx]["noisy_arr"]
            t_idx += 1
    true_arr = np.array([t if t is not None else np.nan for t in agg], dtype=float)
    rows = []
    for c in c_values:
        maes = []
        for trial in range(trials_per_c):
            avg_noisy = np.zeros_like(true_arr, dtype=float)
            for k in range(c):
                avg_noisy += noisy_by_tk[(trial, k)]
            avg_noisy /= c
            mask = np.isfinite(avg_noisy) & np.isfinite(true_arr)
            maes.append(float(np.mean(np.abs(avg_noisy[mask] - true_arr[mask]))))
        rows.append({"num_colluders_c": c, "mean_mae": float(np.mean(maes)),
                     "std_mae": float(np.std(maes)),
                     "predicted_mae_ratio": 1.0 / np.sqrt(c)})
    out = pd.DataFrame(rows)
    out["empirical_mae_ratio"] = out["mean_mae"] / out.iloc[0]["mean_mae"]
    out.to_csv(os.path.join(output_dir, f"{dataset_name}_collusion.csv"), index=False)
    core.logger.info("  Collusion experiment done.")
    return out


def main():
    args = C.resolve(C.make_parser(
        "Extras: n-weighted spotlight, collusion, K_ext induced-latency (Sec. 7.10)."
    ).parse_args())
    C.run_main_phase(args, {"extras"})


if __name__ == "__main__":
    main()
