#!/usr/bin/env python3
"""Experiment K: wait induced by the adaptive interval (paper Sec. 6, RQ2).

PAWS widens an interval past the base Delta_t when its subscribed scope is
expected to hold fewer than P_min publishers (cap K_ext * Delta_t for the
predictive policy, (1 + K_ext) * Delta_t for the others).  This experiment
measures that extra wait -- the delivery delay the approach adds on top of the
interval itself -- for every subscription level of every dataset, under

  none        never wait (release or walk at Delta_t)
  predictive  the paper's sizing (Sec. 4.1): width fixed up front at
              clip(P_aim / lambda, Delta_t, K_ext * Delta_t) from past noisy
              counts, plus at most one corrective Delta_t extension
  reactive    the original policy: extend by whole Delta_t, one DP count per
              check
  early_stop  stop as soon as P_min is met: the Sparse Vector stopping rule,
              checked every raw slot, one count share per interval

for K_ext in ``--k-ext-values``.  Wait is reported in Delta_t units
(mean / p95 / max, and the fraction of intervals that waited at all) and in
seconds where the dataset's slot length is known, next to the utility it buys
(NMAE, coverage, penalized NMAE) and the count shares it costs.

Run: python -m experiments.induced_wait --dataset all --trials 6
"""
from __future__ import annotations

from experiments import _common as C
from experiments import _section7 as S7
from experiments import pipeline as pl


POLICIES = ("none", "predictive", "reactive", "early_stop")


def induced_wait_experiment(datasets, clamp_mode, output_dir, args, *,
                            k_ext_values=(1, 2, 4, 8), trials: int = 1,
                            policies=POLICIES):
    points = [{"wait_policy": "none", "k_ext": 0}] if "none" in policies else []
    points += [{"wait_policy": p, "k_ext": int(k)}
               for p in POLICIES[1:] if p in policies
               for k in k_ext_values]
    eps, w = float(args.fixed_epsilon), int(args.fixed_w)

    def kwargs_fn(ds, sensor, point, approach):
        return S7.approach_kwargs(
            args, ds, clamp_mode, approach, eps, w,
            overrides={"wait": point["wait_policy"], "k_ext": point["k_ext"]})

    # Every population-aware strategy (the only ones that wait), each at its
    # own grid optimum apart from the swept wait policy and K_ext.
    df = S7.run_points(datasets, clamp_mode, args, points=points,
                       approaches=S7.paws_approaches(args, population_aware_only=True),
                       kwargs_fn=kwargs_fn,
                       label="exp K", trials=trials)
    if not df.empty:
        slot_s = df["dataset"].map(pl.SLOT_SECONDS).astype(float)
        dt_s = slot_s * df["delta_t"]
        df["delta_t_seconds"] = dt_s
        for stat in ("mean", "p95", "max"):
            df[f"{stat}_wait_s"] = df[f"{stat}_wait_dt"] * dt_s
    S7.write_experiment(df, output_dir, "experiment_K_induced_wait", trials)
    return df


def main():
    p = C.make_parser("Experiment K: wait induced by the adaptive interval.")
    p.add_argument("--k-ext-values", type=int, nargs="+", default=None,
                   help="K_ext caps to sweep (default 1 2 4 8; --quick 1 2).")
    p.add_argument("--wait-policies", nargs="+", choices=list(POLICIES),
                   default=list(POLICIES),
                   help="Wait policies to compare (the cluster runs one per "
                        "shard).")
    args = C.resolve(p.parse_args())
    k_vals = args.k_ext_values or ([1, 2] if args.quick else [1, 2, 4, 8])
    for clamp_mode in args._clamp_modes:
        induced_wait_experiment(args._targets, clamp_mode,
                                C.cross_dir(args, clamp_mode), args,
                                k_ext_values=k_vals, trials=max(1, args.trials),
                                policies=args.wait_policies)


if __name__ == "__main__":
    main()
