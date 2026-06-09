#!/usr/bin/env python3
"""Introduction figures: Extreme 1 (global), Extreme 1.1 (per-type), Extreme 2 / LDP (per-publisher), the KL U-shape and Figure 1.

Runs the 'intro' phase of the main pipeline over every target dataset on the
PerCom topic hierarchy, in parallel (--workers), writing the same CSVs the
core pipeline does for this phase.  Run: python -m experiments.intro

The actual figure code lives HERE; ``run_experiment`` keeps thin
lazy-delegating stubs so the per-dataset orchestration (clamp, dirs, manifest)
in ``run_dataset`` stays in one place.  The intro phase produces, per dataset:

  * extreme1_global         -- one-stream-per-system extreme (paper Sec. 1.3)
  * extreme1_1_per_type     -- ONE STREAM PER PUBLISHER TYPE; subscriptions at
                               every topic level are broken by type-only agg
  * extreme2_per_publisher  -- per-publisher DP (n_tau=1, lambda=R*w/eps = LDP)
  * kl_extremes_vs_ours     -- grouped-bar KL across both extremes + our approach
  * u_shaped_curve          -- the KL-vs-P U-shape (paper Figure 1 on real data)
  * figure1_reproduction    -- KL vs aggregation scope P (cross-dataset input)
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments import _common as C
from experiments import engine as core
from data_streams import _stable_hash   # process-stable seed digest (not salted hash())
# Shared helpers/types kept in core; imported by name so the moved bodies run
# verbatim.  The worker functions (_dp_named_task / _init_named_streams_worker)
# stay in core, so ProcessPoolExecutor pickles them by their original
# run_experiment qualified name regardless of where they're referenced.
from experiments.engine import (
    compute_kl_divergence,
    _run_parallel_tasks,
    _dp_named_task,
    _init_named_streams_worker,
    _topic_level_groups,
    _group_true_stream,
    PrivacyConfig,
    StreamState,
    BudgetStrategy,
    logger,
)

INTRO_EPSILON = 1.0
INTRO_W = 8
INTRO_N_TRIALS = 20   # seed average for Figure 1 (Laplace noise is high-variance)


# ═════════════════════════════════════════════════════════════════════════
#  Figure 1 reproduction: KL divergence vs aggregation scope P
# ═════════════════════════════════════════════════════════════════════════

def figure1_reproduction(
    per_pub_all: dict[str, tuple[dict[str, list[float | None]], float]],
    dataset_name: str,
    output_dir: str,
    epsilon: float = 1.0,
    w: int = 8,
    n_trials: int = 20,
    workers: int = 1,
) -> pd.DataFrame:
    """
    Reproduce paper Figure 1 on real data (cross-dataset aggregation input).

    All regimes use **Uniform** budget allocation (paper §1.3: Figure 1 is
    about the effect of AGGREGATION SCOPE, not budget-allocation strategy);
    KL is averaged over `n_trials` seeds per sensor to suppress single-draw
    Laplace variance.  The "global" point is Paper Extreme 1 (one stream per
    system, cross-metric, one Laplace per tau).

    Returns a DataFrame with columns
    (dataset, P_scope, P_label, kl_divergence, epsilon, w, N).
    """
    logger.info(f"  [{dataset_name}] figure1_reproduction: building tasks...")
    # Reconstruct the per-sensor aggregate streams from per_pub_all so we
    # don't need the original `streams` dict.
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

    # Global (Extreme 1) stream: one stream per system, single noise per tau
    # with R = sup(R) and n_tau = total publishers across metrics.
    all_B = max(B for _, B in per_pub_all.values())
    cross_metric_true: list[float] = []
    num_pubs_total: list[int] = []
    for tau in range(T):
        vals = [sensor_streams[s][0][tau] for s in sensor_names
                if sensor_streams[s][1][tau] > 0]
        cross_metric_true.append(float(np.mean(vals)) if vals else 0.0)
        num_pubs_total.append(sum(sensor_streams[s][1][tau] for s in sensor_names))

    # Pack every stream the workers might need into one keyed dict.
    named: dict = {}
    for s, (pp, B) in per_pub_all.items():
        for pub_id, series in pp.items():
            pa = [v if v is not None else 0.0 for v in series]
            pc = [1 if v is not None else 0 for v in series]
            named[("pub", s, pub_id)] = (pa, pc, B)
        named[("agg", s)] = sensor_streams[s]
    named[("global",)] = (cross_metric_true, num_pubs_total, all_B)

    # Build the task list.  Each task carries an "op" tag so we can bucket the
    # results (p1 / mid[P] / global[sensor]) when everything finishes.
    tasks: list = []
    ops: list = []  # parallel to `tasks`
    idx = 0
    # Per-publisher extreme (P=1)
    for s in sensor_names:
        pp, _ = per_pub_all[s]
        for i, pub_id in enumerate(pp.keys()):
            for trial in range(n_trials):
                seed = 500_000 + i * 1000 + trial
                tasks.append((idx, ("pub", s, pub_id), "uniform", 1,
                              epsilon, w, seed, "metrics"))
                ops.append(("p1",))
                idx += 1
    # Intermediate P on clamped aggregate
    for P in P_values:
        for s in sensor_names:
            agg, cnt, _ = sensor_streams[s]
            if max(cnt) < P:
                continue
            for trial in range(n_trials):
                seed = 600_000 + P * 1000 + trial
                tasks.append((idx, ("agg", s), "uniform", P,
                              epsilon, w, seed, "metrics"))
                ops.append(("mid", P))
                idx += 1
    # Global (Extreme 1) — need noisy_values back for custom KL vs per-metric truth
    for s in sensor_names:
        for trial in range(n_trials):
            seed = 700_000 + _stable_hash(s) + trial * 7919
            tasks.append((idx, ("global",), "uniform", 1,
                          epsilon, w, seed, "noisy"))
            ops.append(("global", s))
            idx += 1

    logger.info(f"  [{dataset_name}] figure1_reproduction: dispatching "
                f"{len(tasks)} DP runs across {workers} worker(s)")
    results = _run_parallel_tasks(
        tasks, _dp_named_task,
        workers=workers,
        initializer=_init_named_streams_worker,
        initargs=(named,),
        progress_label=f"  [{dataset_name}] figure1",
        progress_every=max(50, len(tasks) // 20),
    )

    # Aggregate by bucket.
    p1_kls: list[float] = []
    mid_kls_buf: dict[int, list[float]] = {P: [] for P in P_values}
    g_kls: list[float] = []
    for op, res in zip(ops, results):
        kind = op[0]
        if kind == "p1":
            k = res["kl"]
            if np.isfinite(k):
                p1_kls.append(k)
        elif kind == "mid":
            _, P = op
            k = res["kl"]
            if np.isfinite(k):
                mid_kls_buf[P].append(k)
        else:  # "global"
            _, s = op
            true_stream = sensor_streams[s][0][:T]
            nvals = res["noisy_arr"]
            true_c = [v for v in true_stream if v is not None]
            nc = [float(v) for v in nvals if np.isfinite(v)]
            if len(true_c) >= 10 and len(nc) >= 10:
                k = compute_kl_divergence(true_c, nc)
                if np.isfinite(k):
                    g_kls.append(k)

    kl_p1 = float(np.mean(p1_kls)) if p1_kls else float("nan")
    mid_kls: dict[int, float] = {
        P: (float(np.mean(v)) if v else float("nan"))
        for P, v in mid_kls_buf.items()
    }
    kl_global = float(np.mean(g_kls)) if g_kls else float("nan")

    rows = [{"dataset": dataset_name, "P_scope": 1, "P_label": "per-pub",
             "kl_divergence": kl_p1, "epsilon": epsilon, "w": w, "N": N_pubs}]
    for P in P_values:
        rows.append({"dataset": dataset_name, "P_scope": P, "P_label": f"P={P}",
                     "kl_divergence": mid_kls[P], "epsilon": epsilon, "w": w, "N": N_pubs})
    rows.append({"dataset": dataset_name, "P_scope": N_pubs + 1, "P_label": "global",
                 "kl_divergence": kl_global, "epsilon": epsilon, "w": w, "N": N_pubs})
    df_fig1 = pd.DataFrame(rows)
    df_fig1.to_csv(os.path.join(output_dir, f"{dataset_name}_figure1_kl_vs_P.csv"),
                   index=False)

    labels = df_fig1["P_label"].tolist()
    values = df_fig1["kl_divergence"].tolist()
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#e74c3c"] + ["#2ecc71"] * len(P_values) + ["#e74c3c"]
    bars = ax.bar(labels, values, color=colors, alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, values):
        if np.isfinite(v):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set(xlabel="Aggregation scope P",
           ylabel="Average KL divergence",
           title=f"{dataset_name}: Distributional Distortion vs. Aggregation Scope"
                 f"  (eps={epsilon}, w={w})")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(output_dir, f"{dataset_name}_figure1_kl_vs_P.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"  Figure-1 reproduction saved: {path}")
    return df_fig1


# ═════════════════════════════════════════════════════════════════════════
#  Paper's motivating intro figures (Section 1.3)
# ═════════════════════════════════════════════════════════════════════════
#
# Produces, per dataset:
#   extreme1_global: one-stream-per-system extreme (paper Section 1.3)
#   extreme2_per_publisher: one-stream-per-publisher extreme
#   kl_extremes_vs_ours: grouped-bar KL across both extremes + our approach
#   u_shaped_curve: the KL-vs-P U-shape (paper Figure 1 on real data)
# All functions write both CSV (where data-like) and PNG (always).

def _apply_dp(aggregates, pub_counts, epsilon, w, min_publishers,
              payload_bound, seed=0, strategy=BudgetStrategy.UNIFORM):
    np.random.seed(seed)
    cfg = PrivacyConfig(
        epsilon=epsilon, window_size=w, min_publishers=min_publishers,
        payload_bound=payload_bound, strategy=strategy,
    )
    state = StreamState(config=cfg)
    for a, n in zip(aggregates, pub_counts):
        state.release(float(a), int(n))
    return state.true_values, state.noisy_values


def extreme1_global(sensor_streams, output_dir, dataset_name):
    """Global mean across every publisher/topic.  Destroys topic signal."""
    T = min(len(v[0]) for v in sensor_streams.values())
    global_B = max(v[2] for v in sensor_streams.values())
    num_pubs_per_tau = [sum(v[1][tau] for v in sensor_streams.values()) for tau in range(T)]
    global_agg = []
    for tau in range(T):
        vals = [agg[tau] for (agg, cnt, _) in sensor_streams.values() if cnt[tau] > 0]
        global_agg.append(float(np.mean(vals)) if vals else 0.0)

    _, noisy_global = _apply_dp(global_agg, num_pubs_per_tau,
                                epsilon=INTRO_EPSILON, w=INTRO_W,
                                min_publishers=1, payload_bound=global_B, seed=100)

    pd.DataFrame({
        "t": range(T),
        "global_true": global_agg,
        "global_noisy": [v if v is not None else np.nan for v in noisy_global[:T]],
        "num_pubs_total": num_pubs_per_tau,
    }).to_csv(os.path.join(output_dir, f"{dataset_name}_figure_extreme1_global.csv"),
              index=False)

    t = np.arange(min(T, 300))
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1]},
    )
    palette = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12"]
    for i, (sensor, (agg, _, _)) in enumerate(sensor_streams.items()):
        ax1.plot(t, agg[:len(t)], color=palette[i % len(palette)],
                 alpha=0.6, lw=1.0, label=sensor)
    ax1.plot(t, global_agg[:len(t)], "k-", lw=2.5, alpha=0.9, label="global mean")
    ax1.set_ylabel("Sensor value")
    ax1.set_title("Extreme 1: Global Average Destroys Topic-Level Signal",
                  fontsize=13, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8, ncol=2); ax1.grid(True, alpha=0.3)

    noisy_x = [i for i, v in enumerate(noisy_global[:len(t)]) if v is not None]
    noisy_y = [noisy_global[i] for i in noisy_x]
    ax2.plot(t, global_agg[:len(t)], "b-", lw=1.5, alpha=0.8, label="true global mean")
    ax2.plot(noisy_x, noisy_y, "r-", lw=1.0, alpha=0.6,
             label=f"noisy DP release (eps={INTRO_EPSILON}, w={INTRO_W})")
    ax2.set(xlabel="Time window", ylabel="Global mean",
            title=f"w-event DP on global stream  (Delta_f = R/n_tau, R={global_B:.1f})")
    ax2.legend(loc="upper right", fontsize=8); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, f"{dataset_name}_figure_extreme1_global.png")
    plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()
    logger.info(f"  saved {path}")


def extreme1_1_per_type(sensor_streams, per_pub_all, dataset_name, output_dir,
                        workers: int = 1):
    """Extreme 1.1 (paper Introduction): ONE STREAM PER PUBLISHER TYPE.

    Aggregate all publishers of the same sensor type into one stream per type
    (here the per-type mean = sensor_streams[s]), apply w-event DP with Uniform
    allocation and no P-gate, and deliver the SAME type-level value to every
    subscriber of that type.  The paper's critique is that this destroys
    topic-level (sub-type) semantics: a subscriber interested in one publisher's
    topic receives the type average instead of its own signal.

    We quantify that semantic loss exactly as Extreme 1 does one level up
    (sensor-true vs global-release): here it is publisher-true vs type-release,
    averaged over INTRO_N_TRIALS noise seeds and over the publishers of the type.
    Writes a per-(dataset, sensor-type) CSV with the per-type KL/NMAE distortion.
    """
    rows = []
    for s, (agg, cnt, B) in sensor_streams.items():
        if s not in per_pub_all:
            continue
        per_pub, _Bpp = per_pub_all[s]
        T = len(agg)
        # Topic-hierarchy levels for this type's publishers (level 1 = domain
        # root / whole type; deepest = per-publisher leaf).  A SUBSCRIPTION at
        # level L wants the mean over its subtree, but Extreme 1.1 delivers the
        # same pure-type release to all of them -- so finer subscriptions break.
        levels = _topic_level_groups(dataset_name, s, list(per_pub.keys()))
        if not levels:  # no topic mapping (e.g. synthetic): one pseudo-leaf level
            levels = {1: {p: [p] for p in per_pub}}
        # Per (level, subtree) accumulators of KL/NMAE vs the type release.
        level_kls: dict[int, list] = {L: [] for L in levels}
        level_nmaes: dict[int, list] = {L: [] for L in levels}
        for trial in range(INTRO_N_TRIALS):
            # PURE-TYPE aggregation: DP on the type-level mean over EVERY
            # publisher of the type, ignoring topic position (Extreme 1.1).
            _t, type_release = _apply_dp(
                agg, cnt, epsilon=INTRO_EPSILON, w=INTRO_W, min_publishers=1,
                payload_bound=B, seed=410_000 + trial * 17,
            )
            for L, groups in levels.items():
                for _prefix, members in groups.items():
                    sub_true = _group_true_stream(per_pub, members, T)
                    t_vals, r_vals = [], []
                    for i in range(min(T, len(type_release))):
                        if sub_true[i] is not None and type_release[i] is not None:
                            t_vals.append(sub_true[i]); r_vals.append(type_release[i])
                    if len(t_vals) >= 10:
                        k = compute_kl_divergence(t_vals, r_vals)
                        if np.isfinite(k):
                            level_kls[L].append(k)
                        mae = float(np.mean([abs(a - b) for a, b in zip(t_vals, r_vals)]))
                        level_nmaes[L].append(mae / B if B > 0 else float("nan"))
        maxL = max(levels)
        for L in sorted(levels):
            kls, nmaes = level_kls[L], level_nmaes[L]
            sizes = [len(m) for m in levels[L].values()]
            rows.append({
                "dataset": dataset_name, "sensor_type": s, "payload_bound": B,
                "n_publishers": len(per_pub), "epsilon": INTRO_EPSILON, "w": INTRO_W,
                "subscription_level": L,
                "is_type_root": (L == 1), "is_leaf": (L == maxL),
                "n_subscriptions_at_level": len(levels[L]),
                "avg_pubs_per_subscription": float(np.mean(sizes)) if sizes else float("nan"),
                # Distortion a level-L subscriber suffers from the pure-type release.
                "kl_vs_type_release": float(np.mean(kls)) if kls else float("nan"),
                "nmae_vs_type_release": float(np.nanmean(nmaes)) if nmaes else float("nan"),
                "n_trials": INTRO_N_TRIALS,
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir,
                           f"{dataset_name}_figure_extreme1_1_per_type.csv"),
              index=False)
    if not df.empty:
        leaf = df[df["is_leaf"]]
        root = df[df["is_type_root"]]
        logger.info(
            f"  [{dataset_name}] Extreme 1.1 (pure-type): KL vs subscription "
            f"grows root={root['kl_vs_type_release'].mean():.3f} -> "
            f"leaf={leaf['kl_vs_type_release'].mean():.3f} "
            f"(finer subscriptions broken by type-only aggregation)")
    return df


def extreme2_per_publisher(per_pub, payload_bound, sensor_label,
                           dataset_name, output_dir):
    """Per-publisher DP: n_tau=1 everywhere, lambda = R*w/eps."""
    pubs = list(per_pub.keys())
    show = pubs[:4]
    noise_scale = payload_bound * INTRO_W / INTRO_EPSILON

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    csv_rows = []
    for ax, pub_id in zip(axes.flat, show):
        raw = per_pub[pub_id]
        pub_agg = [v if v is not None else 0.0 for v in raw]
        pub_cnt = [1 if v is not None else 0 for v in raw]
        true_vals, noisy_vals = _apply_dp(
            pub_agg, pub_cnt,
            epsilon=INTRO_EPSILON, w=INTRO_W, min_publishers=1,
            payload_bound=payload_bound, seed=_stable_hash(pub_id),
        )
        n = min(300, len(true_vals))
        true_x = [i for i in range(n) if true_vals[i] is not None and true_vals[i] != 0]
        true_y = [true_vals[i] for i in true_x]
        ax.plot(true_x, true_y, "b-", lw=1.2, alpha=0.85, label="true")
        noisy_xy = [(i, noisy_vals[i]) for i in range(n) if noisy_vals[i] is not None]
        if noisy_xy:
            ax.plot([p[0] for p in noisy_xy], [p[1] for p in noisy_xy],
                    "r-", lw=0.8, alpha=0.5, label="DP release")
        ax.set_title(f"publisher: {pub_id}", fontsize=10)
        ax.legend(fontsize=7, loc="upper right"); ax.grid(True, alpha=0.3)
        for i in range(n):
            csv_rows.append({
                "publisher_id": pub_id, "t": i,
                "true": true_vals[i] if i < len(true_vals) else None,
                "noisy": noisy_vals[i] if i < len(noisy_vals) else None,
            })

    pd.DataFrame(csv_rows).to_csv(
        os.path.join(output_dir, f"{dataset_name}_figure_extreme2_per_publisher.csv"),
        index=False,
    )

    axes[1][0].set_xlabel("Time window"); axes[1][1].set_xlabel("Time window")
    axes[0][0].set_ylabel(sensor_label); axes[1][0].set_ylabel(sensor_label)
    fig.suptitle(
        f"Extreme 2: Per-publisher w-event DP  "
        f"(n_tau=1, Delta_f=R={payload_bound:.1f}, lambda=R*w/eps={noise_scale:.0f})\n"
        f"Noise overwhelms the signal; publisher identity is fully exposed",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(output_dir, f"{dataset_name}_figure_extreme2_per_publisher.png")
    plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()
    logger.info(f"  saved {path}")


def kl_extremes_vs_ours(sensor_streams, per_pub_all, P_our,
                        dataset_name, output_dir, workers: int = 1):
    """Grouped bar of KL for Extreme 1 / Extreme 2 / Our-approach per sensor.

    All three regimes are averaged over `INTRO_N_TRIALS` noise seeds so the
    bars reflect the expected distortion, not a single-draw artefact.
    """
    logger.info(f"  [{dataset_name}] kl_extremes_vs_ours: building tasks...")
    T = min(len(v[0]) for v in sensor_streams.values())
    sensor_names = list(sensor_streams.keys())
    global_B = max(v[2] for v in sensor_streams.values())

    # Extreme 1 (paper §1.3): one stream per system.
    global_true: list[float] = []
    for tau in range(T):
        vals = [sensor_streams[s][0][tau] for s in sensor_names
                if sensor_streams[s][1][tau] > 0]
        global_true.append(float(np.mean(vals)) if vals else 0.0)
    num_pubs_global = [sum(sensor_streams[s][1][tau] for s in sensor_names)
                       for tau in range(T)]

    # Pack every keyed stream into the worker pool.
    named: dict = {}
    for s in sensor_names:
        if s in per_pub_all:
            pp, B = per_pub_all[s]
            for pub_id, series in pp.items():
                pa = [v if v is not None else 0.0 for v in series]
                pc = [1 if v is not None else 0 for v in series]
                named[("pub", s, pub_id)] = (pa, pc, B)
        named[("agg", s)] = sensor_streams[s]
    named[("global",)] = (global_true, num_pubs_global, global_B)

    tasks: list = []
    ops: list = []
    idx = 0
    # Extreme 1: INTRO_N_TRIALS global DP runs; KL per-sensor computed in caller.
    for trial in range(INTRO_N_TRIALS):
        seed = 200_000 + trial * 13
        tasks.append((idx, ("global",), "uniform", 1,
                      INTRO_EPSILON, INTRO_W, seed, "noisy"))
        ops.append(("e1",))
        idx += 1
    # Extreme 2: per-publisher DP
    for s in sensor_names:
        if s not in per_pub_all:
            continue
        pp, _ = per_pub_all[s]
        for i, pub_id in enumerate(pp.keys()):
            for trial in range(INTRO_N_TRIALS):
                seed = 300_000 + i * 1000 + trial
                tasks.append((idx, ("pub", s, pub_id), "uniform", 1,
                              INTRO_EPSILON, INTRO_W, seed, "self_kl"))
                ops.append(("e2", s))
                idx += 1
    # Our approach: P-gated BA at P_our on each sensor's clamped aggregate
    for s in sensor_names:
        for trial in range(INTRO_N_TRIALS):
            seed = 400_000 + trial
            tasks.append((idx, ("agg", s), "p_gated_ba", P_our,
                          INTRO_EPSILON, INTRO_W, seed, "self_kl"))
            ops.append(("ours", s))
            idx += 1

    logger.info(f"  [{dataset_name}] kl_extremes_vs_ours: dispatching "
                f"{len(tasks)} DP runs across {workers} worker(s)")
    run_results = _run_parallel_tasks(
        tasks, _dp_named_task,
        workers=workers,
        initializer=_init_named_streams_worker,
        initargs=(named,),
        progress_label=f"  [{dataset_name}] extremes",
        progress_every=max(50, len(tasks) // 20),
    )

    e1: dict[str, list[float]] = {s: [] for s in sensor_names}
    e2: dict[str, list[float]] = {s: [] for s in sensor_names}
    ours: dict[str, list[float]] = {s: [] for s in sensor_names}
    for op, res in zip(ops, run_results):
        kind = op[0]
        if kind == "e1":
            noisy_arr = res["noisy_arr"]
            noisy_list = [float(v) if np.isfinite(v) else None for v in noisy_arr]
            for s in sensor_names:
                k = compute_kl_divergence(
                    list(sensor_streams[s][0][:T]), noisy_list,
                )
                if np.isfinite(k):
                    e1[s].append(k)
        elif kind == "e2":
            _, s = op
            k = res["kl_self"]
            if np.isfinite(k):
                e2[s].append(k)
        else:  # "ours"
            _, s = op
            k = res["kl_self"]
            if np.isfinite(k):
                ours[s].append(k)

    results = {
        "Extreme 1\n(global average)":
            {s: float(np.mean(v)) if v else float("nan") for s, v in e1.items()},
        "Extreme 2\n(per-publisher)":
            {s: float(np.mean(v)) if v else float("nan") for s, v in e2.items()},
        f"Our approach\n(P={P_our} topic pool)":
            {s: float(np.mean(v)) if v else float("nan") for s, v in ours.items()},
    }

    csv_rows = []
    for regime, per_sensor in results.items():
        for sensor_name, kl in per_sensor.items():
            csv_rows.append({
                "regime": regime.replace("\n", " "),
                "sensor": sensor_name,
                "kl_divergence": kl,
                "P_our": P_our,
            })
    pd.DataFrame(csv_rows).to_csv(
        os.path.join(output_dir, f"{dataset_name}_figure_extremes_vs_ours.csv"),
        index=False,
    )

    regimes = list(results.keys())
    x = np.arange(len(regimes))
    bar_w = 0.15
    palette = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for j, s in enumerate(sensor_names):
        vals = [results[r].get(s, float("nan")) for r in regimes]
        offset = (j - len(sensor_names) / 2 + 0.5) * bar_w
        bars = ax.bar(x + offset, vals, bar_w, label=s,
                      color=palette[j % len(palette)], alpha=0.85,
                      edgecolor="white", lw=0.8)
        for bar, val in zip(bars, vals):
            if np.isfinite(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom",
                        fontsize=7, fontweight="bold")
    avg_vals = [float(np.nanmean([results[r].get(s, np.nan) for s in sensor_names]))
                for r in regimes]
    avg_off = (len(sensor_names) / 2 + 0.5) * bar_w
    ax.bar(x + avg_off, avg_vals, bar_w * 1.2, label="average",
           color="#34495e", alpha=0.7, edgecolor="white", lw=0.8)
    for i, val in enumerate(avg_vals):
        ax.text(x[i] + avg_off, val + 0.01, f"{val:.2f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold", color="#34495e")
    ax.axhline(y=np.log(2), color="gray", ls=":", alpha=0.5)
    ax.text(len(regimes) - 0.5, np.log(2) + 0.01, "ln(2) ~ 0.69",
            fontsize=7, color="gray", ha="right")
    ax.set_xticks(x); ax.set_xticklabels(regimes, fontsize=10)
    ax.set_ylabel("KL divergence  D_KL(P || Q)", fontsize=11)
    ax.set_title(
        f"{dataset_name}: naive extremes vs. our approach  "
        f"(eps={INTRO_EPSILON}, w={INTRO_W})",
        fontsize=13, fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=9, ncol=3); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(output_dir, f"{dataset_name}_figure_extremes_vs_ours.png")
    plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()
    logger.info(f"  saved {path}")
    return results


def u_shaped_curve(sensor_streams, per_pub_all, dataset_name, output_dir,
                   workers: int = 1):
    """Reproduce paper Figure 1 on real data: KL vs aggregation scope P.

    All three regimes use Uniform budget allocation (paper §1.3: Figure 1 is
    about the effect of AGGREGATION SCOPE, not budget-allocation strategy).
    Each (P, sensor) is averaged over INTRO_N_TRIALS noise seeds to suppress
    single-draw Laplace variance.
    """
    logger.info(f"  [{dataset_name}] u_shaped_curve: building tasks...")
    sensor_names = list(sensor_streams.keys())
    T = min(len(v[0]) for v in sensor_streams.values())
    sweep_p = [2, 3, 4, 6, 8]

    # Cross-metric global stream (Extreme 1 / "one stream per system").
    all_B = max(v[2] for v in sensor_streams.values())
    num_pubs_total = [sum(v[1][tau] for v in sensor_streams.values())
                      for tau in range(T)]
    cross_metric_true: list[float] = []
    for tau in range(T):
        vals = [sensor_streams[s][0][tau] for s in sensor_names
                if sensor_streams[s][1][tau] > 0]
        cross_metric_true.append(float(np.mean(vals)) if vals else 0.0)

    # Pack every stream keyed for the worker pool.
    named: dict = {}
    for s in sensor_names:
        if s in per_pub_all:
            pp, B = per_pub_all[s]
            for pub_id, series in pp.items():
                pa = [v if v is not None else 0.0 for v in series]
                pc = [1 if v is not None else 0 for v in series]
                named[("pub", s, pub_id)] = (pa, pc, B)
        named[("agg", s)] = sensor_streams[s]
    named[("global",)] = (cross_metric_true, num_pubs_total, all_B)

    tasks: list = []
    ops: list = []
    idx = 0
    # P=1 per-publisher uses kl_self (_kl_of semantics)
    for s in sensor_names:
        if s not in per_pub_all:
            continue
        pp, _ = per_pub_all[s]
        for i, pub_id in enumerate(pp.keys()):
            for trial in range(INTRO_N_TRIALS):
                seed = 1_000_000 + i * 1000 + trial
                tasks.append((idx, ("pub", s, pub_id), "uniform", 1,
                              INTRO_EPSILON, INTRO_W, seed, "self_kl"))
                ops.append(("p1",))
                idx += 1
    # Intermediate P on clamped aggregate
    for P in sweep_p:
        for s in sensor_names:
            agg, cnt, _ = sensor_streams[s]
            if max(cnt) < P:
                continue
            for trial in range(INTRO_N_TRIALS):
                seed = 2_000_000 + P * 1000 + trial
                tasks.append((idx, ("agg", s), "uniform", P,
                              INTRO_EPSILON, INTRO_W, seed, "self_kl"))
                ops.append(("mid", P))
                idx += 1
    # Global (Extreme 1) — compare sensor's true vs global-DP noisy
    for s in sensor_names:
        for trial in range(INTRO_N_TRIALS):
            seed = 5_000_000 + _stable_hash(s) + trial * 7919
            tasks.append((idx, ("global",), "uniform", 1,
                          INTRO_EPSILON, INTRO_W, seed, "noisy"))
            ops.append(("global", s))
            idx += 1

    logger.info(f"  [{dataset_name}] u_shaped_curve: dispatching "
                f"{len(tasks)} DP runs across {workers} worker(s)")
    results = _run_parallel_tasks(
        tasks, _dp_named_task,
        workers=workers,
        initializer=_init_named_streams_worker,
        initargs=(named,),
        progress_label=f"  [{dataset_name}] u_shape",
        progress_every=max(50, len(tasks) // 20),
    )

    p1_kls: list[float] = []
    mid_kls_buf: dict[int, list[float]] = {P: [] for P in sweep_p}
    g_kls: list[float] = []
    for op, res in zip(ops, results):
        kind = op[0]
        if kind == "p1":
            k = res["kl_self"]
            if np.isfinite(k):
                p1_kls.append(k)
        elif kind == "mid":
            _, P = op
            k = res["kl_self"]
            if np.isfinite(k):
                mid_kls_buf[P].append(k)
        else:  # "global": compare sensor's true vs cross-metric noisy
            _, s = op
            true_stream = sensor_streams[s][0][:T]
            noisy_arr = res["noisy_arr"]
            # Replicate _kl_of: compute_kl_divergence handles None/NaN filtering.
            k = compute_kl_divergence(list(true_stream),
                                      [float(v) if np.isfinite(v) else None
                                       for v in noisy_arr])
            if np.isfinite(k):
                g_kls.append(k)

    kl_p1 = float(np.mean(p1_kls)) if p1_kls else float("nan")
    mid_kls: dict[int, float] = {
        P: (float(np.mean(v)) if v else float("nan"))
        for P, v in mid_kls_buf.items()
    }
    kl_global = float(np.mean(g_kls)) if g_kls else float("nan")

    P_labels = ["1\n(per-pub)"] + [str(p) for p in sweep_p] + ["all\n(global)"]
    kl_values = [kl_p1] + [mid_kls[p] for p in sweep_p] + [kl_global]
    pd.DataFrame({
        "P_label": [l.replace("\n", " ") for l in P_labels],
        "kl_divergence": kl_values,
    }).to_csv(
        os.path.join(output_dir, f"{dataset_name}_figure_u_shaped_P_vs_KL.csv"),
        index=False,
    )

    best_mid = int(np.nanargmin(kl_values[1:-1])) + 1
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(kl_values))
    ax.plot(x, kl_values, "k-", lw=2, zorder=4)
    for i in range(len(x)):
        if i in (0, len(x) - 1):
            color = "#e74c3c"
        elif i == best_mid:
            color = "#2ecc71"
        else:
            color = "#555"
        ax.plot(x[i], kl_values[i], "o", color=color, markersize=12, zorder=6,
                markeredgecolor="white", markeredgewidth=1.5)
        if np.isfinite(kl_values[i]):
            off = max(kl_values) * 0.04
            ax.text(x[i], kl_values[i] + off, f"{kl_values[i]:.2f}",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.axvspan(-0.5, 0.5, color="#e74c3c", alpha=0.08, zorder=1)
    ax.axvspan(0.5, len(x) - 1.5, color="#2ecc71", alpha=0.08, zorder=1)
    ax.axvspan(len(x) - 1.5, len(x) - 0.5, color="#e74c3c", alpha=0.08, zorder=1)
    ax.set_xticks(x); ax.set_xticklabels(P_labels, fontsize=10)
    ax.set_xlabel("Aggregation scope  P", fontsize=12)
    ax.set_ylabel("Average KL divergence", fontsize=12)
    ax.set_title(
        f"{dataset_name}: Distributional distortion vs. aggregation scope  "
        f"(eps={INTRO_EPSILON}, w={INTRO_W})",
        fontsize=12, fontweight="bold",
    )
    ax.grid(True, alpha=0.25, axis="y")
    plt.tight_layout()
    path = os.path.join(output_dir, f"{dataset_name}_figure_u_shaped_P_vs_KL.png")
    plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()
    logger.info(f"  saved {path}")


def run_intro_figures(sensor_streams, per_pub_all, dataset_name, output_dir,
                      P_our=4, workers: int = 1):
    """Produce the four paper-style intro figures for one dataset."""
    if not sensor_streams:
        return
    extreme1_global(sensor_streams, output_dir, dataset_name)
    # Extreme 1.1 (one stream per publisher type) sits between Extreme 1
    # (global) and Extreme 2 (per-publisher) on the aggregation-scope continuum.
    extreme1_1_per_type(sensor_streams, per_pub_all, dataset_name, output_dir,
                        workers=workers)
    if per_pub_all:
        pp_sensor = next(iter(per_pub_all))
        pp_data, pp_B = per_pub_all[pp_sensor]
        extreme2_per_publisher(pp_data, pp_B, pp_sensor, dataset_name, output_dir)
    kl_extremes_vs_ours(sensor_streams, per_pub_all, P_our, dataset_name, output_dir,
                        workers=workers)
    u_shaped_curve(sensor_streams, per_pub_all, dataset_name, output_dir,
                   workers=workers)


def main():
    args = C.resolve(C.make_parser("Introduction figures: Extreme 1 (global), Extreme 1.1 (per-type), Extreme 2 / LDP (per-publisher), the KL U-shape and Figure 1.").parse_args())
    C.run_main_phase(args, {"intro"})


if __name__ == "__main__":
    main()
