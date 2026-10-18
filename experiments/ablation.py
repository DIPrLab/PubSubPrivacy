#!/usr/bin/env python3
"""Experiment F: incremental-module ablation (paper Sec. 6.4, RQ3).

Switches the framework's three components on one at a time and measures the
subscriber utility each adds, for a subscription at EVERY level of every
sensor's topic tree on every dataset:

  M1_pgate         P-allocation gate only: release when the DP count of the
                   subscribed scope reaches P_min, otherwise defer
  M2_interval_ext  + adaptive interval: the paper's predictive sizing
                   (Sec. 4.1) -- each width fixed up front from past noisy
                   counts, up to K_ext * Delta_t, plus at most one corrective
                   extension (``--wait-policy``, default predictive)
  M3_walk_up       + clamp-compatible hierarchy walk with cached delta counts

Each module runs the full broker pipeline offline (experiments.pipeline).
Unlike every other experiment, the ablation does NOT repeat the previous output
on a deferral: scoring stale repeats would credit a module that rarely releases
with the accuracy of one that does, hiding exactly the difference the ablation
measures.  A deferral delivers nothing; utility is NMAE over fresh releases,
reported with coverage and the coverage-penalized NMAE (an undelivered
timestamp counts as the no-signal error 1.0).

P_min, P_max, rho and Delta_t come from the Sec. 6.1 grid optimum; K_ext is the
grid's, or ``--k-ext`` when the grid chose not to wait (so M2 is never a no-op).

Run: python -m experiments.ablation --dataset all --trials 6
"""
from __future__ import annotations

from experiments import _common as C
from experiments import _section7 as S7

MODULES = ["M1_pgate", "M2_interval_ext", "M3_walk_up"]


def ablation_experiment(
    datasets, clamp_mode, output_dir, args, *,
    epsilon: float = 1.0, w: int = 8, P: int = 3, k_ext: int = 3,
    epsilon_count: float = 0.0, strategy: str | None = None,
    seed: int = 77, grid_config: dict | None = None, trials: int = 1,
    eps_values=None,
):
    """Sec. 6.4 incremental-module ablation; per-trial rows + aggregates.

    ``P`` / ``epsilon_count`` / ``grid_config`` are accepted for the original
    call signature; the parameters are resolved from ``args`` like every other
    Section 6 experiment."""
    eps_values = list(eps_values) if eps_values else [float(epsilon)]
    points = [{"module": m, "module_idx": i, "epsilon": float(e)}
              for e in eps_values for i, m in enumerate(MODULES, start=1)]
    wait_policy = getattr(args, "wait_policy", "predictive")
    if wait_policy == "none":
        wait_policy = "predictive"

    def kwargs_fn(ds, sensor, point, approach):
        kw, meta = S7.approach_kwargs(args, ds, clamp_mode, approach,
                                      point["epsilon"], w)
        k = kw["k_ext"] if kw["k_ext"] > 0 else int(k_ext)
        overrides = {
            "M1_pgate": {"wait": "none", "walk": False, "k_ext": k},
            "M2_interval_ext": {"wait": wait_policy, "walk": False, "k_ext": k},
            "M3_walk_up": {"wait": wait_policy, "walk": True, "k_ext": k},
        }[point["module"]]
        kw.update(overrides)
        kw["repeat_on_defer"] = False    # ablation only: score fresh releases
        meta.update(wait_policy=kw["wait"], walk=kw["walk"], k_ext=k)
        return kw, meta

    # Every population-aware strategy (the components act only on those), each
    # at its own grid optimum; ``strategy`` restricts to one.
    approaches = ((f"paws:{strategy}",) if strategy
                  else S7.paws_approaches(args, population_aware_only=True))
    df = S7.run_points(datasets, clamp_mode, args, points=points,
                       approaches=approaches, kwargs_fn=kwargs_fn,
                       label="exp F", trials=trials, seed=seed)
    S7.write_experiment(df, output_dir, "experiment_F_ablation", trials)
    return df


def main():
    p = C.make_parser("Experiment F: incremental-module ablation (Sec. 6.4).")
    p.add_argument("--ablation-eps", type=float, nargs="+", default=None,
                   help="Budgets to run the ablation at (default: the paper "
                        "epsilon grid).")
    args = C.resolve(p.parse_args())
    for clamp_mode in args._clamp_modes:
        ablation_experiment(
            args._targets, clamp_mode, C.cross_dir(args, clamp_mode), args,
            w=args.fixed_w, k_ext=args.k_ext, trials=max(1, args.trials),
            eps_values=args.ablation_eps or args.eps_values)


if __name__ == "__main__":
    main()
