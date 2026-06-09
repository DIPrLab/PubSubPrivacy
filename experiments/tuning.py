#!/usr/bin/env python3
"""Algorithm 2 greedy hill-climb vs naive brute-force P tuning (Sec. 6.9).

Runs the 'tuning' phase of the main pipeline over every target dataset on the
PerCom topic hierarchy, in parallel (--workers), writing the same CSVs the
core pipeline does for this phase.  Run: python -m experiments.tuning

The actual tuning code lives HERE; ``run_experiment`` keeps thin
lazy-delegating stubs so the per-dataset orchestration in ``run_dataset`` stays
in one place.  The greedy/brute worker functions and the worker-side cached
context (``_WORKER_TUNE_CTX``) live in THIS module, so ProcessPoolExecutor
(created by ``core._run_parallel_tasks``) pickles them by their
``experiments.tuning`` qualified name and the spawned child sets/reads the same
module global consistently.
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
# Shared per-stream evaluators kept in core; imported by name so the moved
# bodies run verbatim.
from experiments.engine import (
    _evaluate_stream,
    _rebuild_stream_with_dt,
    _run_parallel_tasks,
    logger,
)


# Worker-side state for hyperparameter tuning: cached (agg, cnt, B, eps, w,
# utility_weight, latency_weight).  Shared by brute-force and per-strategy
# greedy walks so the rebuild runs exactly once.
_WORKER_TUNE_CTX: tuple | None = None


def _init_tune_worker(agg, cnt, payload_bound, epsilon, w,
                      utility_weight, latency_weight):
    global _WORKER_TUNE_CTX
    _WORKER_TUNE_CTX = (agg, cnt, payload_bound, epsilon, w,
                        utility_weight, latency_weight)


def _tune_eval_task(task):
    """Brute-force evaluator: (strategy, P, seed) -> row dict."""
    strategy, P, seed = task
    agg, cnt, B, eps, w, uw, lw = _WORKER_TUNE_CTX
    return _evaluate_stream(agg, cnt, B, eps, w, P, strategy, uw, lw, seed)


def _greedy_walk_task(task):
    """One strategy's Algorithm-2 greedy walk, end-to-end in the worker."""
    (strategy, p_max, alpha, I_max, loss_tie_tol, seed,
     n_restarts, restart_rng_seed) = task
    agg, cnt, B, eps, w, uw, lw = _WORKER_TUNE_CTX
    return _greedy_walk_on_stream(
        agg, cnt, B, eps, w, strategy, p_max,
        alpha=alpha, I_max=I_max,
        utility_weight=uw, latency_weight=lw,
        loss_tie_tol=loss_tie_tol, seed=seed,
        n_restarts=n_restarts, restart_rng_seed=restart_rng_seed,
    )


def _intelligent_restart_seeds(
    cnt: list[int], alpha: float, p_max: int,
    n_restarts: int, rng_seed: int,
) -> list[int]:
    """Pick P values for Algorithm 2 multi-start.

    The first seed is always P_0 = ceil(1/alpha), the paper's
    identity-protection seed (Section 6.7).  The remaining n_restarts-1 seeds
    are sampled *intelligently* rather than uniformly: we draw from the
    empirical distribution of n_tau so the hill-climb is seeded at plausible
    publisher-count levels the stream actually exhibits.  Concretely, we take
    quantiles of the non-zero n_tau distribution (p25, median, p75, ...) so
    the restarts cover the pool-density regimes that matter for the loss.

    All returned values are clamped to [1, p_max] and de-duplicated while
    preserving order.
    """
    P_0 = max(1, int(np.ceil(1.0 / max(alpha, 1e-9))))
    P_0 = max(1, min(P_0, max(1, p_max)))
    seeds: list[int] = [P_0]
    if n_restarts <= 1:
        return seeds

    nonzero = [int(c) for c in cnt if c is not None and c > 0]
    k_extra = max(0, n_restarts - 1)
    extras: list[int] = []
    rng = np.random.default_rng(rng_seed)

    if nonzero:
        # Symmetric quantile grid avoiding 0 and 1 (equivalent to equally spaced
        # probabilities strictly inside the distribution's support).
        qs = np.linspace(1.0 / (k_extra + 1), k_extra / (k_extra + 1), k_extra)
        quantile_candidates = [int(np.clip(round(q), 1, p_max))
                               for q in np.quantile(nonzero, qs)]
        extras.extend(quantile_candidates)

    # Fill any remaining slots with rng-sampled draws from the observed n_tau
    # so the restart surface is not degenerate when the stream's n_tau support
    # is thin (common for traffic/manufacturing sub-pubs).
    while len(extras) < k_extra:
        if nonzero:
            extras.append(int(np.clip(rng.choice(nonzero), 1, p_max)))
        else:
            extras.append(int(rng.integers(1, max(2, p_max) + 1)))

    for s in extras:
        s = int(np.clip(s, 1, max(1, p_max)))
        if s not in seeds:
            seeds.append(s)
    return seeds


def _greedy_hillclimb_from(
    agg, cnt, payload_bound, epsilon, w, strategy, p_max,
    P_start: int, utility_weight, latency_weight,
    I_max: int, loss_tie_tol: float, seed: int,
    restart_idx: int, action_prefix: str = "",
) -> tuple[list[dict], dict, int]:
    """One hill-climb trajectory from ``P_start``.  Returns (trajectory rows,
    best evaluated row, evaluation count for this run).
    """
    evaluations = 0
    trajectory: list[dict] = []
    P = max(1, min(P_start, p_max))
    best = _evaluate_stream(agg, cnt, payload_bound, epsilon, w, P, strategy,
                            utility_weight, latency_weight, seed)
    evaluations += 1
    trajectory.append({
        **best, "iter": 0,
        "action": f"{action_prefix}seed",
        "restart_idx": restart_idx,
    })
    seen = {P: best["tuning_loss"]}
    for i in range(1, I_max + 1):
        neighbors = []
        for P_nbr in (P - 1, P + 1):
            if P_nbr < 1 or P_nbr > p_max:
                continue
            if P_nbr in seen:
                neighbors.append({"P": P_nbr, "tuning_loss": seen[P_nbr],
                                  "_cached": True})
                continue
            r = _evaluate_stream(agg, cnt, payload_bound, epsilon, w, P_nbr, strategy,
                                 utility_weight, latency_weight, seed)
            evaluations += 1
            seen[P_nbr] = r["tuning_loss"]
            neighbors.append({**r, "_cached": False})
            trajectory.append({
                **r, "iter": i,
                "action": f"{action_prefix}probe_P={P_nbr}",
                "restart_idx": restart_idx,
            })
        if not neighbors:
            break
        # Tie-break: on equal loss prefer the higher P (better identity protection).
        best_nbr = min(neighbors, key=lambda x: (x["tuning_loss"], -x["P"]))
        strict_improvement = best_nbr["tuning_loss"] < best["tuning_loss"] - loss_tie_tol
        if strict_improvement:
            P = best_nbr["P"]
            if "evaluated" in best_nbr:
                best = {k: v for k, v in best_nbr.items() if not k.startswith("_")}
            else:
                best = _evaluate_stream(agg, cnt, payload_bound, epsilon, w, P, strategy,
                                        utility_weight, latency_weight, seed)
                evaluations += 1
            trajectory.append({
                **best, "iter": i,
                "action": f"{action_prefix}step_to_P={P}",
                "restart_idx": restart_idx,
            })
        else:
            trajectory.append({
                **best, "iter": i,
                "action": f"{action_prefix}stop_local_optimum",
                "restart_idx": restart_idx,
            })
            break
    return trajectory, best, evaluations


def _greedy_walk_on_stream(
    agg, cnt, payload_bound, epsilon, w, strategy, p_max,
    alpha=0.25, I_max=20, utility_weight=1.0, latency_weight=0.2,
    loss_tie_tol: float = 1e-9, seed: int = 77,
    n_restarts: int = 3, restart_rng_seed: int = 12345,
) -> dict:
    """Stream-level Algorithm 2 walk with intelligent multi-start.

    The first restart seeds at P_0 = ceil(1/alpha) (paper Section 6.7).
    Additional restarts are sampled from quantiles of the observed n_tau
    distribution, so the hill-climb explores the pool-density regimes the
    stream actually exhibits instead of always starting at the same point.
    Each restart runs an independent greedy walk; we return the best local
    optimum across restarts.
    """
    P_0 = max(1, int(np.ceil(1.0 / max(alpha, 1e-9))))
    P_seeds = _intelligent_restart_seeds(
        cnt, alpha=alpha, p_max=max(1, p_max),
        n_restarts=max(1, n_restarts), rng_seed=restart_rng_seed,
    )

    all_traj: list[dict] = []
    best_overall: dict | None = None
    total_evals = 0
    for r_idx, P_start in enumerate(P_seeds):
        prefix = "" if r_idx == 0 else f"r{r_idx}_"
        traj, best_here, evals = _greedy_hillclimb_from(
            agg, cnt, payload_bound, epsilon, w, strategy, p_max,
            P_start=P_start, utility_weight=utility_weight,
            latency_weight=latency_weight, I_max=I_max,
            loss_tie_tol=loss_tie_tol, seed=seed,
            restart_idx=r_idx, action_prefix=prefix,
        )
        all_traj.extend(traj)
        total_evals += evals
        if best_overall is None or (
            best_here["tuning_loss"]
            < best_overall["tuning_loss"] - loss_tie_tol
        ) or (
            abs(best_here["tuning_loss"] - best_overall["tuning_loss"])
            <= loss_tie_tol
            and best_here["P"] > best_overall["P"]
        ):
            best_overall = best_here

    return {
        "strategy": strategy,
        "best": best_overall,
        "seed_P": P_0,
        "restart_seeds_P": P_seeds,
        "trajectory": pd.DataFrame(all_traj),
        "evaluations": total_evals,
        "n_restarts": len(P_seeds),
    }


def greedy_tune_P(
    per_pub, payload_bound, epsilon, w, strategy, p_max,
    alpha=0.25, I_max=20, utility_weight=1.0, latency_weight=0.2,
    loss_tie_tol: float = 1e-9,
    n_restarts: int = 3, restart_rng_seed: int = 12345,
) -> dict:
    """Algorithm 2 from the paper: greedy hill-climb over P with intelligent
    multi-start (seeded at P_0 = ceil(1/alpha) plus additional restarts drawn
    from quantiles of the observed n_tau distribution).

    Compatibility shim: rebuilds the stream from ``per_pub`` and delegates
    to ``_greedy_walk_on_stream``.  Callers with a cached stream should use
    that helper directly.
    """
    agg, cnt = _rebuild_stream_with_dt(per_pub, 1)
    return _greedy_walk_on_stream(
        agg, cnt, payload_bound, epsilon, w, strategy, p_max,
        alpha=alpha, I_max=I_max,
        utility_weight=utility_weight, latency_weight=latency_weight,
        loss_tie_tol=loss_tie_tol,
        n_restarts=n_restarts, restart_rng_seed=restart_rng_seed,
    )


def brute_force_tune_P(
    per_pub, payload_bound, epsilon, w, strategy, p_max,
    utility_weight=1.0, latency_weight=0.2,
) -> pd.DataFrame:
    """Naive full enumeration: evaluate every P in [1, p_max] for one strategy.

    Compatibility shim: rebuilds the stream once and delegates per-P to
    ``_evaluate_stream``.  Callers with a cached stream should use the
    parallel brute-force path in ``tune_hyperparameters`` instead.
    """
    agg, cnt = _rebuild_stream_with_dt(per_pub, 1)
    rows = []
    for P in range(1, int(p_max) + 1):
        rows.append(_evaluate_stream(agg, cnt, payload_bound, epsilon, w, P, strategy,
                                     utility_weight, latency_weight))
    return pd.DataFrame(rows)


def tune_hyperparameters(
    per_pub: dict[str, list[float | None]],
    payload_bound: float,
    dataset_name: str,
    sensor_name: str,
    output_dir: str,
    epsilon: float,
    w: int,
    strategies: list[str] | str = "p_gated_ba",
    alpha: float = 0.25,
    I_max: int = 20,
    utility_weight: float = 1.0,
    latency_weight: float = 0.2,
    p_max: int | None = None,
    workers: int = 1,
    n_restarts: int = 3,
    restart_rng_seed: int = 12345,
) -> dict:
    """
    Stage 1 of Section 5.7: tune P per strategy using the paper's Algorithm 2
    greedy hill-climb, and compare against a naive full enumeration over every
    integer P in [1, p_max].  Delta_t adapts online via the extension mechanism
    of Section 5.6, so it is not tuned here.  Strategies are evaluated in
    parallel (one run each); the paper treats A as selected by subscription
    requirements rather than jointly optimized.

    Writes:
      {ds}_{sensor}_tuning_greedy.csv          -- every evaluated (P, strategy)
      {ds}_{sensor}_tuning_brute_force.csv     -- every P in [1, p_max]
      {ds}_{sensor}_tuning_strategy_summary.csv -- greedy vs brute-force best
      {ds}_{sensor}_tuning.png                 -- per-strategy loss curves
    """
    if isinstance(strategies, str):
        strategies = [strategies]

    # Rebuild the stream exactly once; every greedy probe + brute-force eval
    # reuses the same cached (agg, cnt).  Worker processes see the same cached
    # tuple via the initializer, so each task is a pure run_dp_on_stream call.
    logger.info(f"  [{dataset_name}/{sensor_name}] tune_hyperparameters: rebuilding stream...")
    agg, cnt = _rebuild_stream_with_dt(per_pub, 1)
    obs_max = max(cnt) if cnt else 1
    if p_max is None:
        p_max = max(2, min(obs_max, len(per_pub)))
    p_max = int(p_max)

    tune_init_args = (agg, cnt, payload_bound, epsilon, w,
                      utility_weight, latency_weight)

    # Brute-force: one task per (strategy, P).
    brute_tasks = [(strat, P, 77)
                   for strat in strategies
                   for P in range(1, p_max + 1)]
    logger.info(f"  [{dataset_name}/{sensor_name}] brute-force: dispatching "
                f"{len(brute_tasks)} (strategy, P) tasks across {workers} worker(s)")
    brute_rows_flat = _run_parallel_tasks(
        brute_tasks, _tune_eval_task,
        workers=workers,
        initializer=_init_tune_worker,
        initargs=tune_init_args,
        progress_label=f"  [{dataset_name}/{sensor_name}] brute",
        progress_every=max(10, len(brute_tasks) // 10),
    )
    brute_df = pd.DataFrame(brute_rows_flat)

    # Greedy: one task per strategy (walks are serial within, but across
    # strategies they're independent).  Each task carries its n_restarts and
    # restart-rng seed so workers can reproduce the intelligent multi-start
    # seed schedule.
    greedy_tasks = [(strat, p_max, alpha, I_max, 1e-9, 77,
                     n_restarts, restart_rng_seed + i)
                    for i, strat in enumerate(strategies)]
    logger.info(f"  [{dataset_name}/{sensor_name}] greedy: dispatching "
                f"{len(greedy_tasks)} walks across "
                f"{min(workers, len(greedy_tasks))} worker(s)")
    greedy_out = _run_parallel_tasks(
        greedy_tasks, _greedy_walk_task,
        workers=min(workers, len(greedy_tasks)),
        initializer=_init_tune_worker,
        initargs=tune_init_args,
        progress_label=f"  [{dataset_name}/{sensor_name}] greedy",
        progress_every=1,
    )

    greedy_rows: list[pd.DataFrame] = []
    greedy_results: dict[str, dict] = {}
    for strat, g in zip(strategies, greedy_out):
        g["trajectory"]["strategy"] = strat
        greedy_rows.append(g["trajectory"])
        greedy_results[strat] = g

    greedy_df = pd.concat(greedy_rows, ignore_index=True) if greedy_rows else pd.DataFrame()

    greedy_df.to_csv(
        os.path.join(output_dir, f"{dataset_name}_{sensor_name}_tuning_greedy.csv"),
        index=False,
    )
    brute_df.to_csv(
        os.path.join(output_dir, f"{dataset_name}_{sensor_name}_tuning_brute_force.csv"),
        index=False,
    )

    # Per-strategy greedy vs brute-force comparison.
    summary_rows = []
    for strat in strategies:
        g_res = greedy_results[strat]
        g_best = g_res["best"]
        g_evals = g_res["evaluations"]
        b_sub = brute_df[brute_df["strategy"] == strat]
        if b_sub.empty:
            continue
        # Tie-break: on equal loss prefer the higher P (better identity protection).
        b_sub = b_sub.assign(_neg_P=-b_sub["P"]).sort_values(
            ["tuning_loss", "_neg_P"]).drop(columns=["_neg_P"])
        b_best = b_sub.iloc[0]
        gap_loss = float(g_best["tuning_loss"] - b_best["tuning_loss"])
        gap_P = int(g_best["P"] - b_best["P"])
        restart_seeds = g_res.get("restart_seeds_P", [g_res.get("seed_P", 1)])
        summary_rows.append({
            "strategy": strat,
            "greedy_P": int(g_best["P"]),
            "greedy_loss": float(g_best["tuning_loss"]),
            "greedy_nmae": g_best["normalized_mae"],
            "greedy_kl": g_best["kl_divergence"],
            "greedy_release_rate": g_best["release_rate"],
            "greedy_evaluations": g_evals,
            "greedy_seed_P": g_res["seed_P"],
            "greedy_n_restarts": int(g_res.get("n_restarts", 1)),
            "greedy_restart_seeds_P": ";".join(str(p) for p in restart_seeds),
            "brute_P": int(b_best["P"]),
            "brute_loss": float(b_best["tuning_loss"]),
            "brute_nmae": b_best["normalized_mae"],
            "brute_kl": b_best["kl_divergence"],
            "brute_release_rate": b_best["release_rate"],
            "brute_evaluations": len(b_sub),
            "gap_loss": gap_loss,
            "gap_P": gap_P,
            "speedup": float(len(b_sub) / max(g_evals, 1)),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        os.path.join(output_dir, f"{dataset_name}_{sensor_name}_tuning_strategy_summary.csv"),
        index=False,
    )

    logger.info(f"  [tuning] {dataset_name}/{sensor_name}  p_max={p_max}, alpha={alpha} -> seed P_0={max(1, int(np.ceil(1.0 / max(alpha, 1e-9))))}")
    logger.info(f"    {'strategy':<20} {'greedy_P':>9} {'brute_P':>8} "
                f"{'greedy_L':>9} {'brute_L':>8} {'gap_L':>8} "
                f"{'gr_evals':>9} {'br_evals':>9} {'speedup':>8}")
    for _, r in summary_df.iterrows():
        logger.info(
            f"    {r['strategy']:<20} {int(r['greedy_P']):>9} {int(r['brute_P']):>8} "
            f"{r['greedy_loss']:>9.4f} {r['brute_loss']:>8.4f} "
            f"{r['gap_loss']:>8.4f} {int(r['greedy_evaluations']):>9} "
            f"{int(r['brute_evaluations']):>9} {r['speedup']:>8.2f}"
        )

    # Per-strategy loss-vs-P plot with greedy trajectory overlay.
    palette = plt.cm.tab10(np.linspace(0, 1, max(len(strategies), 1)))
    ncols = min(3, len(strategies)) or 1
    nrows = (len(strategies) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.0 * nrows),
                             squeeze=False)
    for idx, (color, strat) in enumerate(zip(palette, strategies)):
        ax = axes[idx // ncols][idx % ncols]
        bsub = brute_df[brute_df["strategy"] == strat].sort_values("P")
        if bsub.empty:
            continue
        ax.plot(bsub["P"], bsub["tuning_loss"], "o-", color=color, lw=1.4,
                alpha=0.85, label="brute force (every P)")
        g_traj = greedy_df[(greedy_df["strategy"] == strat)
                           & (greedy_df["action"].str.startswith(("seed", "probe", "step")))]
        ax.scatter(g_traj["P"], g_traj["tuning_loss"], marker="x", s=70,
                   color="black", zorder=5, label="greedy probe")
        # Final greedy best
        g_best = greedy_results[strat]["best"]
        ax.scatter([g_best["P"]], [g_best["tuning_loss"]], marker="*", s=300,
                   color="gold", edgecolor="black", zorder=6,
                   label=f"greedy best (P={int(g_best['P'])})")
        b_best = bsub.sort_values("tuning_loss").iloc[0]
        ax.scatter([b_best["P"]], [b_best["tuning_loss"]], marker="D", s=110,
                   color="red", edgecolor="white", zorder=6,
                   label=f"brute-force best (P={int(b_best['P'])})")
        ax.set(xlabel="P", ylabel="tuning loss", title=strat)
        ax.legend(fontsize=6); ax.grid(True, alpha=0.3)
    for idx in range(len(strategies), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)
    fig.suptitle(f"{dataset_name}/{sensor_name}: Algorithm 2 greedy vs brute-force over P "
                 f"(alpha={alpha}, p_max={p_max})", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_{sensor_name}_tuning.png"), dpi=150)
    plt.close()

    return {"greedy": greedy_df, "brute_force": brute_df, "gap_summary": summary_df}


def main():
    args = C.resolve(C.make_parser("Algorithm 2 greedy hill-climb vs naive brute-force P tuning (Sec. 6.9).").parse_args())
    C.run_main_phase(args, {"tuning"})


if __name__ == "__main__":
    main()
