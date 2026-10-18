#!/usr/bin/env python3
"""Experiment U: subscriber utility (paper Sec. 6, RQ1).

Compares PAWS -- under EVERY budget strategy (``--strategies``, default all
nine), each at its own Sec. 6.1 grid optimum for (dataset, strategy, epsilon),
with n-weighted P-allocation marked as ours -- against the two baselines --
local DP (each publisher perturbs
its own reading, P_min = 1) and w-event DP (Kellaris et al. uniform allocation
on the subscribed scope's aggregate) -- on all six datasets, for a
subscription bound at EVERY level of each sensor's topic tree, sweeping

  * the privacy budget epsilon (window fixed, ``--fixed-w``), and
  * the window length w (budget fixed, ``--fixed-epsilon``).

Utility is NMAE of the delivered values against the subscribed scope's true
aggregate (a deferred timestamp repeats the previous output, as in the
paper), reported with coverage and the coverage-penalized NMAE.  PAWS runs the
full pipeline (predictive adaptive interval + cached-count hierarchy walk) at
the Sec. 6.1 grid optimum; all approaches share the same base interval Delta_t.

Outputs (under <output-dir>/cross_dataset/<clamp>/experiments/U_utility/):
  experiment_U_<sweep>.csv            per-trial rows
  experiment_U_<sweep>_aggregate.csv  mean/std over trials per subscription
  experiment_U_<sweep>_summary.csv    averaged over sensors and subscriptions
  experiment_U_<sweep>_by_level.csv   ... per subscription level

Run: python -m experiments.utility --dataset all --sweep both --trials 6
"""
from __future__ import annotations

from experiments import _common as C
from experiments import _section7 as S7

BASELINES = ("ldp", "wevent")


def utility_experiment(datasets, clamp_mode, output_dir, args, *,
                       sweep: str = "epsilon", trials: int = 1):
    if sweep == "epsilon":
        points = [{"epsilon": float(e), "w": int(args.fixed_w)}
                  for e in args.eps_values]
    elif sweep == "window":
        points = [{"epsilon": float(args.fixed_epsilon), "w": int(w)}
                  for w in args.w_values]
    else:
        raise ValueError(f"unknown sweep {sweep!r}")

    def kwargs_fn(ds, sensor, point, approach):
        kw, meta = S7.approach_kwargs(args, ds, clamp_mode, approach,
                                      point["epsilon"], point["w"])
        return kw, meta

    approaches = BASELINES + S7.paws_approaches(args)
    df = S7.run_points(datasets, clamp_mode, args, points=points,
                       approaches=approaches, kwargs_fn=kwargs_fn,
                       label=f"exp U/{sweep}", trials=trials)
    S7.write_experiment(df, output_dir, f"experiment_U_{sweep}", trials)
    return df


def main():
    p = C.make_parser("Experiment U: utility vs epsilon / window / depth "
                      "against LDP and w-event DP.")
    p.add_argument("--sweep", choices=["epsilon", "window", "both"],
                   default="both")
    args = C.resolve(p.parse_args())
    sweeps = ["epsilon", "window"] if args.sweep == "both" else [args.sweep]
    for clamp_mode in args._clamp_modes:
        for sweep in sweeps:
            utility_experiment(args._targets, clamp_mode,
                               C.cross_dir(args, clamp_mode), args,
                               sweep=sweep, trials=max(1, args.trials))


if __name__ == "__main__":
    main()
