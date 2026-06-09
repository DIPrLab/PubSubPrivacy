#!/usr/bin/env python3
"""Experiment A: greedy subscription WALK-UP vs brute-force over rewrite depths
(paper Sec. 6.5 Algorithm 1 + the Sec. 6.9 NP-hard MCKP reduction).

For a subscription bound at a leaf, the broker must pick a rewrite depth in the
PerCom topic tree that pools >= P range-compatible publishers while spending as
little eps_count on discovery as possible.  This experiment compares:

  * GREEDY (Algorithm 1): walk up from the leaf, spending one eps_count per
    probed level, stopping at the first ancestor whose DP publisher count meets
    P (or the root).  Records the chosen level, its utility, and the eps_count
    spent (= levels probed x eps_count).
  * BRUTE: probe EVERY level (rewrite depth), spend eps_count at each, and pick
    the depth with the best utility.  eps_count cost = (#levels) x eps_count.

Reports the utility gap, the eps_count SAVED by greedy, and the probe speedup,
per (dataset, sensor) over every dataset.  Parallel-free per dataset (few
subscriptions) but cluster-sharded per dataset.

Run: python -m experiments.greedy_vs_brute --dataset all --epsilon-count 0.5
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from experiments import _common as C
from experiments import engine as core


def _leaf_chain(ds_name, sensor, per_pub, k_ext=0):
    """Return the leaf->root chain of (level, scope, agg, cnt, n_pubs) for the
    busiest leaf publisher's subscription (the candidate rewrite depths)."""
    streams = core.level_subscription_streams(ds_name, sensor, per_pub, k_ext=k_ext)
    by_prefix = {scope: (L, agg, cnt, npub) for (L, scope, agg, cnt, npub) in streams}
    spec = core.DATASETS.get(ds_name)
    topic_of = spec.get("publisher_topic") if spec else None
    leaf_pub = max(per_pub, key=lambda p: sum(v is not None for v in per_pub[p]))
    if topic_of is None:
        # No topic tree: single whole-type scope.
        agg, cnt = core._adaptive_interval_rebuild(per_pub, 1, k_ext)
        return leaf_pub, [(1, sensor, agg, cnt, len(per_pub))]
    path = topic_of(leaf_pub, sensor).split("/")
    maxL = max(L for (L, _, _, _, _) in streams)
    chain = []
    for L in range(maxL, 0, -1):  # leaf (deepest) -> root
        prefix = "/".join(path[:L])
        if prefix in by_prefix:
            L2, agg, cnt, npub = by_prefix[prefix]
            chain.append((L, prefix, agg, cnt, npub))
    return leaf_pub, chain


def experiment_A_greedy_vs_brute(datasets, clamp_mode, output_dir, args,
                                 fixed_combos=None, strategies=None) -> "pd.DataFrame":
    epsilon, w = 1.0, 8
    strategy = "p_gated_uniform"
    eps_count = getattr(args, "epsilon_count", 0.0) or 0.5
    seed = 77
    trials = max(1, getattr(args, "trials", 1))
    grid_config = getattr(args, "grid_config", None)
    rows = []
    for ds_name in datasets:
        prepared = core.prepare_dataset(
            ds_name, clamp_mode=clamp_mode, eps_clip=args.eps_clip,
            seed=args.seed, max_rows=core._dataset_max_rows(ds_name, args))
        if prepared is None or not prepared.per_pubs:
            continue
        for sensor in [s for s in prepared.spec["sensors"] if s in prepared.per_pubs]:
            per_pub, B = prepared.per_pubs[sensor]
            prm = core._resolve_params(grid_config, ds_name, clamp_mode, strategy,
                                       epsilon, {"P_min": getattr(args, "ablation_P", 3),
                                                 "P_max": None})
            P_ds, P_max_ds = prm["P_min"], prm["P_max"]
            leaf_pub, chain = _leaf_chain(ds_name, sensor, per_pub,
                                          k_ext=getattr(args, "k_ext", 0))
            if not chain:
                continue
            # Repeat over trials: the greedy DP count draw AND the Laplace
            # release noise are stochastic, so the walk decision + utilities
            # vary by seed (per-trial rows + per-(dataset,sensor) aggregate).
            for trial in range(trials):
                seed_t = seed + 1000 * trial
                rng = np.random.default_rng(seed_t)

                # GREEDY walk-up: probe leaf->root, spend eps_count/level, stop
                # at first ancestor whose DP count meets P (root always releases).
                greedy_probed, greedy = 0, chain[-1]  # default = root
                for (L, scope, agg, cnt, npub) in chain:
                    greedy_probed += 1
                    avg_active = float(np.mean(cnt)) if cnt else 0.0
                    dp_count = (max(0.0, avg_active + rng.laplace(scale=1.0 / eps_count))
                                if eps_count > 0 else avg_active)
                    if dp_count >= P_ds or L == 1:
                        greedy = (L, scope, agg, cnt, npub)
                        break
                gL, gscope, gagg, gcnt, gnp = greedy
                gm = core.run_dp_on_stream(
                    gagg, gcnt, epsilon=epsilon, window_size=w, min_publishers=P_ds,
                    payload_bound=B, strategy=strategy, seed=seed_t,
                    epsilon_count=eps_count, max_publishers=P_max_ds)["metrics"]
                greedy_eps_count = greedy_probed * eps_count

                # BRUTE: evaluate every rewrite depth, pick best NMAE.
                best = None
                for (L, scope, agg, cnt, npub) in chain:
                    m = core.run_dp_on_stream(
                        agg, cnt, epsilon=epsilon, window_size=w, min_publishers=P_ds,
                        payload_bound=B, strategy=strategy, seed=seed_t,
                        epsilon_count=eps_count, max_publishers=P_max_ds)["metrics"]
                    nm = m.get("normalized_mae")
                    if best is None or (nm is not None and np.isfinite(nm)
                                        and nm < best[2]):
                        best = (L, scope, nm if nm is not None else float("inf"),
                                m.get("release_rate"))
                brute_eps_count = len(chain) * eps_count

                rows.append({
                    "dataset": ds_name, "sensor": sensor, "clamp_mode": clamp_mode,
                    "leaf_publisher": str(leaf_pub), "P_min": P_ds, "P_max": P_max_ds,
                    "epsilon": epsilon, "w": w, "epsilon_count": eps_count,
                    "n_levels": len(chain), "trial": trial, "seed": seed_t,
                    # greedy walk-up (Algorithm 1)
                    "greedy_level": gL, "greedy_scope": gscope,
                    "greedy_n_pubs": gnp,
                    "greedy_normalized_mae": gm.get("normalized_mae"),
                    "greedy_release_rate": gm.get("release_rate"),
                    "greedy_levels_probed": greedy_probed,
                    "greedy_eps_count_spent": greedy_eps_count,
                    # brute over rewrite depths
                    "brute_level": best[0], "brute_scope": best[1],
                    "brute_normalized_mae": best[2], "brute_release_rate": best[3],
                    "brute_levels_probed": len(chain),
                    "brute_eps_count_spent": brute_eps_count,
                    # comparison
                    "nmae_gap": (gm.get("normalized_mae") - best[2]
                                 if gm.get("normalized_mae") is not None else float("nan")),
                    "eps_count_saved": brute_eps_count - greedy_eps_count,
                    "probe_speedup": len(chain) / max(1, greedy_probed),
                })
    exp_dir = os.path.join(output_dir, "experiments", "A_greedy_vs_brute")
    os.makedirs(exp_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(exp_dir, "experiment_A_greedy_vs_brute.csv"), index=False)
    if trials > 1 and not df.empty:
        core._aggregate_over_trials(
            df, ["dataset", "sensor", "clamp_mode"],
            ["greedy_normalized_mae", "greedy_release_rate", "greedy_eps_count_spent",
             "brute_normalized_mae", "brute_release_rate", "brute_eps_count_spent",
             "nmae_gap", "eps_count_saved", "probe_speedup", "greedy_levels_probed"],
        ).to_csv(os.path.join(exp_dir, "experiment_A_greedy_vs_brute_aggregate.csv"),
                 index=False)
    core.logger.info(f"  Experiment A (walk-up greedy vs brute) wrote {len(df)} "
                     f"rows -> {exp_dir}")
    return df


def main():
    args = C.resolve(C.make_parser(
        "Experiment A: greedy subscription walk-up vs brute over rewrite depths "
        "(Sec. 6.5/6.9), accounting for eps_count spend."
    ).parse_args())
    for clamp_mode in args._clamp_modes:
        experiment_A_greedy_vs_brute(args._targets, clamp_mode,
                                     C.cross_dir(args, clamp_mode), args)


if __name__ == "__main__":
    main()
