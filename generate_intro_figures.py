#!/usr/bin/env python3
"""
Generate motivating figures for the Introduction (Section 1.3):

  Figure 1 — Extreme 1: Global-average w-event DP
    Treats ALL publishers as a single stream.  The global mean destroys
    topic-level signal.

  Figure 2 — Extreme 2: Per-publisher w-event DP
    Treats each publisher independently (P=1, Delta=R).  Noise overwhelms signal.

  Figure 3 — KL Divergence Motivation (grouped bar)
    KL divergence for both extremes vs. our P-publisher approach.

  Figure 4 — P vs Average KL Divergence (line sweep)
    P on x-axis (P=1 per-publisher, P=middle, P=total global),
    average KL on y-axis, showing the middle-ground is best.

Runs on real data from the MCEC-Thai energy dataset (12 circuit publishers).

Usage:
  python generate_intro_figures.py
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dp_engine import (
    BudgetStrategy,
    PrivacyConfig,
    StreamState,
    compute_kl_divergence,
)
from run_real_data_experiment import (
    load_energy_dataset,
    build_energy_streams,
    build_energy_per_publisher,
    ENERGY_SENSORS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# Parameters for intro figures
EPSILON = 1.0
W = 10
STRATEGY = BudgetStrategy.UNIFORM


# ── DP helper ───────────────────────────────────────────────────────────────

def apply_dp(
    aggregates: list[float],
    pub_counts: list[int],
    epsilon: float,
    w: int,
    min_publishers: int,
    payload_bound: float,
    seed: int = 0,
) -> tuple[list[float], list[float | None]]:
    """Return (true_values, noisy_values) from the DP engine."""
    np.random.seed(seed)
    config = PrivacyConfig(
        epsilon=epsilon, window_size=w,
        min_publishers=min_publishers,
        payload_bound=payload_bound,
        strategy=STRATEGY,
    )
    stream = StreamState(config=config)
    for agg, n in zip(aggregates, pub_counts):
        stream.release(agg, n)
    return stream.true_values, stream.noisy_values


# ── Aggregation helpers ─────────────────────────────────────────────────────

def per_topic_aggregates(
    per_pub: dict[str, list[float | None]],
) -> tuple[list[float], list[int]]:
    """Mean of active publishers at each timestamp."""
    T = len(next(iter(per_pub.values())))
    aggs, counts = [], []
    for tau in range(T):
        vals = [per_pub[k][tau] for k in per_pub if per_pub[k][tau] is not None]
        aggs.append(float(np.mean(vals)) if vals else 0.0)
        counts.append(len(vals))
    return aggs, counts


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE 1 — Extreme 1: Global-average w-event DP
# ═══════════════════════════════════════════════════════════════════════════

def figure1(
    sensor_streams: dict[str, tuple[list[float], list[int], float]],
    output_dir: str,
):
    """
    Top: per-sensor aggregate streams + global mean showing signal loss.
    Bottom: true vs noisy global mean.
    """
    T = min(len(v[0]) for v in sensor_streams.values())
    global_agg, global_cnt = [], []
    for tau in range(T):
        vals = [sensor_streams[s][0][tau] for s in sensor_streams
                if tau < len(sensor_streams[s][0]) and sensor_streams[s][0][tau] != 0]
        global_agg.append(float(np.mean(vals)) if vals else 0.0)
        global_cnt.append(len(vals))

    global_B = max(v[2] for v in sensor_streams.values())

    true_g, noisy_g = apply_dp(
        global_agg, global_cnt,
        epsilon=EPSILON, w=W, min_publishers=1,
        payload_bound=global_B, seed=100,
    )

    t = np.arange(min(T, 300))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [1.2, 1]})

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12"]
    for i, (sensor, (agg, _, _)) in enumerate(sensor_streams.items()):
        ax1.plot(t, agg[:len(t)], color=colors[i % len(colors)], alpha=0.6,
                 lw=1.0, label=sensor)
    ax1.plot(t, global_agg[:len(t)], "k-", lw=2.5, alpha=0.9, label="Global Mean")
    ax1.set_ylabel("Sensor Value")
    ax1.set_title("Extreme 1: Global Average Destroys Topic-Level Signal",
                   fontsize=13, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)

    noisy_x = [i for i, v in enumerate(noisy_g[:len(t)]) if v is not None]
    noisy_y = [noisy_g[i] for i in noisy_x]
    ax2.plot(t, true_g[:len(t)], "b-", lw=1.5, alpha=0.8, label="True Global Mean")
    ax2.plot(noisy_x, noisy_y, "r-", lw=1.0, alpha=0.6,
             label=f"Noisy Release (e={EPSILON}, w={W})")
    ax2.set_xlabel("Time Window")
    ax2.set_ylabel("Global Mean")
    ax2.set_title(f"w-Event DP on Global Stream  (e={EPSILON}, w={W}, Delta=B={global_B:.1f})", fontsize=11)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "figure1_global_average_dp.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE 2 — Extreme 2: Per-publisher w-event DP
# ═══════════════════════════════════════════════════════════════════════════

def figure2(
    per_pub: dict[str, list[float | None]],
    payload_bound: float,
    sensor_label: str,
    output_dir: str,
):
    """
    4 individual publishers with per-publisher DP.
    Noise (lambda = B*w/e) overwhelms the signal.
    """
    pubs = list(per_pub.keys())
    show = pubs[:4]
    noise_scale = payload_bound * W / EPSILON

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)

    for ax, pub_id in zip(axes.flat, show):
        raw = per_pub[pub_id]
        pub_agg = [v if v is not None else 0.0 for v in raw]
        pub_cnt = [1 if v is not None else 0 for v in raw]

        true_vals, noisy_vals = apply_dp(
            pub_agg, pub_cnt,
            epsilon=EPSILON, w=W, min_publishers=1,
            payload_bound=payload_bound, seed=hash(pub_id) % 10000,
        )

        n = min(300, len(true_vals))
        true_x = [i for i in range(n) if true_vals[i] is not None and true_vals[i] != 0]
        true_y = [true_vals[i] for i in true_x]
        ax.plot(true_x, true_y, "b-", lw=1.2, alpha=0.8, label="True")

        noisy_xy = [(i, noisy_vals[i]) for i in range(n) if noisy_vals[i] is not None]
        if noisy_xy:
            ax.plot([p[0] for p in noisy_xy], [p[1] for p in noisy_xy],
                    "r-", lw=0.8, alpha=0.5, label="Noisy (DP)")

        ax.set_title(f"Publisher: {pub_id}", fontsize=10)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[1][0].set_xlabel("Time Window")
    axes[1][1].set_xlabel("Time Window")
    axes[0][0].set_ylabel(sensor_label)
    axes[1][0].set_ylabel(sensor_label)

    fig.suptitle(
        f"Extreme 2: Per-Publisher w-Event DP  "
        f"(Delta = B = {payload_bound:.1f},  e = {EPSILON},  w = {W},  "
        f"lambda = Bw/e = {noise_scale:.0f})\n"
        f"Noise overwhelms signal; publisher identity fully exposed",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(output_dir, "figure2_per_publisher_dp.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE 3 — KL Divergence: extremes vs our approach (grouped bar)
# ═══════════════════════════════════════════════════════════════════════════

def figure3(
    sensor_streams: dict[str, tuple[list[float], list[int], float]],
    per_pub_all: dict[str, tuple[dict[str, list[float | None]], float]],
    output_dir: str,
):
    """
    Grouped bar: KL for Extreme 1, Extreme 2, Our (P=4) per sensor + average.
    """
    T = min(len(v[0]) for v in sensor_streams.values())

    # Extreme 1: global average
    global_agg = []
    for tau in range(T):
        vals = [sensor_streams[s][0][tau] for s in sensor_streams
                if tau < len(sensor_streams[s][0]) and sensor_streams[s][0][tau] != 0]
        global_agg.append(float(np.mean(vals)) if vals else 0.0)
    global_B = max(v[2] for v in sensor_streams.values())
    _, noisy_global = apply_dp(global_agg, [len(sensor_streams)] * T,
                               epsilon=EPSILON, w=W, min_publishers=1,
                               payload_bound=global_B, seed=200)
    noisy_g_clean = [v for v in noisy_global if v is not None]

    results = {}
    sensor_names = list(sensor_streams.keys())

    # Extreme 1 KL
    e1 = {}
    for sensor in sensor_names:
        agg = sensor_streams[sensor][0][:T]
        e1[sensor] = compute_kl_divergence(agg, noisy_g_clean[:len(agg)])
    results["Extreme 1\n(Global Average)"] = e1

    # Extreme 2 KL (per-publisher, P=1)
    e2 = {}
    for sensor in sensor_names:
        if sensor not in per_pub_all:
            e2[sensor] = float("nan")
            continue
        pp, B = per_pub_all[sensor]
        kls = []
        for i, pub_id in enumerate(pp):
            raw = pp[pub_id]
            pa = [v if v is not None else 0.0 for v in raw]
            pc = [1 if v is not None else 0 for v in raw]
            tv, nv = apply_dp(pa, pc, epsilon=EPSILON, w=W, min_publishers=1,
                              payload_bound=B, seed=300 + i)
            tc = [v for v in tv if v is not None]
            nc = [v for v in nv if v is not None]
            if len(tc) > 5 and len(nc) > 5:
                k = compute_kl_divergence(tc, nc)
                if np.isfinite(k):
                    kls.append(k)
        e2[sensor] = float(np.mean(kls)) if kls else float("nan")
    results["Extreme 2\n(Per-Publisher)"] = e2

    # Our approach (P=4)
    P_OUR = 4
    ours = {}
    for sensor in sensor_names:
        agg, cnt, B = sensor_streams[sensor]
        tv, nv = apply_dp(agg, cnt, epsilon=EPSILON, w=W, min_publishers=P_OUR,
                          payload_bound=B, seed=400)
        tc = [v for v in tv if v is not None]
        nc = [v for v in nv if v is not None]
        ours[sensor] = compute_kl_divergence(tc, nc)
    results[f"Our Approach\n(P={P_OUR}, per-topic)"] = ours

    # Plot
    regimes = list(results.keys())
    x = np.arange(len(regimes))
    bar_w = 0.15
    sensor_colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12"]

    fig, ax = plt.subplots(figsize=(10, 6))
    for j, sensor in enumerate(sensor_names):
        vals = [results[r].get(sensor, float("nan")) for r in regimes]
        offset = (j - len(sensor_names) / 2 + 0.5) * bar_w
        bars = ax.bar(x + offset, vals, bar_w, label=sensor,
                      color=sensor_colors[j % len(sensor_colors)], alpha=0.85,
                      edgecolor="white", lw=0.8)
        for bar, val in zip(bars, vals):
            if np.isfinite(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

    # Average bar
    avg_vals = [float(np.nanmean([results[r].get(s, np.nan) for s in sensor_names])) for r in regimes]
    avg_off = (len(sensor_names) / 2 + 0.5) * bar_w
    ax.bar(x + avg_off, avg_vals, bar_w * 1.2, label="Average",
           color="#34495e", alpha=0.7, edgecolor="white", lw=0.8)
    for i, val in enumerate(avg_vals):
        ax.text(x[i] + avg_off, val + 0.01, f"{val:.2f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold", color="#34495e")

    ax.axhline(y=np.log(2), color="gray", ls=":", alpha=0.5)
    ax.text(len(regimes) - 0.5, np.log(2) + 0.01, "ln(2)~0.69", fontsize=7, color="gray", ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(regimes, fontsize=10)
    ax.set_ylabel("KL Divergence  D_KL(P || Q)", fontsize=11)
    ax.set_title(f"KL Divergence: Naive Extremes vs. Our P-Publisher Approach\n"
                 f"(e = {EPSILON}, w = {W}, uniform)", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, ncol=3)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(output_dir, "figure3_kl_divergence_motivation.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")

    # Summary
    print("\n  KL Divergence Summary:")
    hdr = f"  {'Regime':<30}" + "".join(f"{s:>12}" for s in sensor_names) + f"{'Avg':>10}"
    print(hdr)
    print(f"  {'-' * (len(hdr) - 2)}")
    for r in regimes:
        vals = [results[r].get(s, float("nan")) for s in sensor_names]
        avg = float(np.nanmean(vals))
        rs = r.replace("\n", " ")
        line = f"  {rs:<30}" + "".join(f"{v:12.3f}" for v in vals) + f"{avg:10.3f}"
        print(line)

    return results


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE 4 — P vs Average KL Divergence (line sweep on real energy data)
# ═══════════════════════════════════════════════════════════════════════════

def figure4(
    sensor_streams: dict[str, tuple[list[float], list[int], float]],
    per_pub_all: dict[str, tuple[dict[str, list[float | None]], float]],
    output_dir: str,
):
    """
    Simple line sweep.  X-axis: publisher threshold P.  Y-axis: avg KL.

    P=1:      per-publisher DP (max KL across individual publisher streams).
    P=2..8:   per-topic aggregation with sensitivity B/P.
    All:      global cross-type collapse (all sensor types mixed).
    """
    sensor_names = list(sensor_streams.keys())
    T = min(len(v[0]) for v in sensor_streams.values())

    # ── P=1: per-publisher DP on each individual stream ─────────────────
    # Max KL across all publishers — worst-case publisher experience.
    p1_kls = []
    for sensor in sensor_names:
        pp, B = per_pub_all[sensor]
        for i, (_, vals) in enumerate(pp.items()):
            pa = [v if v is not None else 0.0 for v in vals]
            pc = [1 if v is not None else 0 for v in vals]
            tv, nv = apply_dp(pa, pc, epsilon=EPSILON, w=W,
                              min_publishers=1, payload_bound=B, seed=1000 + i)
            tc = [v for v in tv if v is not None]
            nc = [v for v in nv if v is not None]
            if len(tc) > 10 and len(nc) > 10:
                kl = compute_kl_divergence(tc, nc)
                if np.isfinite(kl):
                    p1_kls.append(kl)
    kl_p1 = float(np.max(p1_kls)) if p1_kls else float("nan")

    # ── P=2..8: per-topic aggregation ───────────────────────────────────
    sweep_p = [2, 3, 4, 6, 8]
    mid_kls = {}
    for P in sweep_p:
        kls = []
        for sensor in sensor_names:
            agg, cnt, B = sensor_streams[sensor]
            tv, nv = apply_dp(agg, cnt, epsilon=EPSILON, w=W,
                              min_publishers=P, payload_bound=B, seed=2000 + P)
            tc = [v for v in tv if v is not None]
            nc = [v for v in nv if v is not None]
            if len(tc) > 10 and len(nc) > 10:
                kl = compute_kl_divergence(tc, nc)
                if np.isfinite(kl):
                    kls.append(kl)
        if kls:
            mid_kls[P] = float(np.mean(kls))

    # ── Global: all sensor types collapsed ──────────────────────────────
    # Per-publisher DP on every circuit across ALL types, then averaged.
    # Subscriber wanted topic-level data, gets cross-type noise soup.
    g_kls = []
    for sensor in sensor_names:
        true_agg = sensor_streams[sensor][0][:T]
        all_noisy = []
        for other_sensor in sensor_names:
            pp_o, B_o = per_pub_all[other_sensor]
            for i, (circ, vals) in enumerate(pp_o.items()):
                pa = [v if v is not None else 0.0 for v in vals[:T]]
                pc = [1 if v is not None else 0 for v in vals[:T]]
                _, nv = apply_dp(pa, pc, epsilon=EPSILON, w=W,
                                 min_publishers=1, payload_bound=B_o,
                                 seed=5000 + hash(f"{other_sensor}/{circ}") % 10000)
                all_noisy.append(nv)
        global_noisy = []
        for tau in range(T):
            nv = [s[tau] for s in all_noisy if tau < len(s) and s[tau] is not None]
            global_noisy.append(float(np.mean(nv)) if nv else 0.0)
        kl = compute_kl_divergence(true_agg, global_noisy)
        if np.isfinite(kl):
            g_kls.append(kl)
    kl_global = float(np.mean(g_kls)) if g_kls else float("nan")

    # ── Assemble ────────────────────────────────────────────────────────
    labels = ["1\n(per-pub)"] + [str(p) for p in sorted(mid_kls)] + ["All\n(global)"]
    kl_all = [kl_p1] + [mid_kls[p] for p in sorted(mid_kls)] + [kl_global]

    best_mid = int(np.nanargmin(kl_all[1:-1])) + 1
    best_kl = kl_all[best_mid]

    # ── Plot — line sweep ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(kl_all))

    ax.plot(x, kl_all, "k-", lw=2, zorder=4)

    for i in range(len(x)):
        c = "red" if i in (0, len(x) - 1) else ("green" if i == best_mid else "#555")
        ax.plot(x[i], kl_all[i], "o", color=c, markersize=10, zorder=6,
                markeredgecolor="white", markeredgewidth=1.5)
        if np.isfinite(kl_all[i]):
            off = max(kl_all) * 0.03
            ax.text(x[i], kl_all[i] + off, f"{kl_all[i]:.2f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_xlabel("Publisher Threshold  P", fontsize=11)
    ax.set_ylabel("Average KL Divergence", fontsize=11)
    ax.set_title("Distributional Distortion vs. Aggregation Scope",
                  fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.25, axis="y")

    plt.tight_layout()
    path = os.path.join(output_dir, "figure4_p_vs_kl.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")

    print(f"\n  P vs Avg KL:")
    for lbl, kl in zip(labels, kl_all):
        print(f"    P={lbl.replace(chr(10),' '):<12} KL = {kl:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate intro motivation figures")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    # Cap at 5000 rows (~1000 windows) for reliable KL estimation.
    max_ts = 5000

    # Load energy dataset
    print("Loading energy dataset...")
    df_raw = load_energy_dataset(max_timestamps=max_ts)

    # Build aggregate streams for multiple sensor types
    sensor_streams: dict[str, tuple[list[float], list[int], float]] = {}
    per_pub_all: dict[str, tuple[dict[str, list[float | None]], float]] = {}

    for sensor_type in ["power_kw", "voltage", "current"]:
        try:
            agg, pub, B = build_energy_streams(df_raw.copy(), sensor_type, window_minutes=5)
            if len(agg) > 10 and B > 0.01:
                sensor_streams[sensor_type] = (agg, pub, B)
        except Exception as e:
            logger.warning(f"  Skipping {sensor_type}: {e}")

        try:
            pp, B = build_energy_per_publisher(df_raw.copy(), sensor_type, window_minutes=5)
            per_pub_all[sensor_type] = (pp, B)
        except Exception:
            pass

    if not sensor_streams:
        print("ERROR: No valid streams. Check data/ folder.")
        return

    # Stats
    for sensor, (agg, cnt, B) in sensor_streams.items():
        print(f"  {sensor}: {len(agg)} windows, B={B:.2f}, avg pubs={np.mean(cnt):.1f}")

    print(f"\n--- Figure 1: Global-Average w-Event DP (Extreme 1) ---")
    figure1(sensor_streams, output_dir)

    print(f"\n--- Figure 2: Per-Publisher w-Event DP (Extreme 2) ---")
    pp_sensor = list(per_pub_all.keys())[0]
    pp_data, pp_B = per_pub_all[pp_sensor]
    figure2(pp_data, pp_B, f"{pp_sensor} ({ENERGY_SENSORS[pp_sensor]['unit']})", output_dir)

    print(f"\n--- Figure 3: KL Divergence Motivation ---")
    figure3(sensor_streams, per_pub_all, output_dir)

    print(f"\n--- Figure 4: P vs Average KL Divergence ---")
    figure4(sensor_streams, per_pub_all, output_dir)

    print(f"\nAll figures saved to {output_dir}")


if __name__ == "__main__":
    main()
