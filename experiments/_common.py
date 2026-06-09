"""Shared scaffolding for the per-experiment modules.

``experiments/engine.py`` is the shared CORE LIBRARY (DP engine wrappers,
dataset ingestion, metrics, the parallel task pool, the topic-hierarchy /
subscription primitives, and the per-dataset / cross-dataset orchestration).
Each ``experiments/<name>.py`` module is a thin, focused, independently-runnable
entry point for ONE paper experiment that imports the engine (``from
experiments import engine as core``) and drives it over every dataset on the
PerCom topic hierarchy.  ``run_experiment.py`` is just a CLI over the engine.
This module provides the common CLI + argument resolution so the per-experiment
files stay tiny and consistent.

Run any experiment standalone, e.g.::

    python -m experiments.sweep        --dataset all --clamp-mode static
    python -m experiments.window       --dataset energy
    python -m experiments.subscription_levels --dataset all --trials 6
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

# Make the repo root importable when run as a script or as `-m experiments.x`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments import engine as core  # the shared core library  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CLAMP_MODES = ["static", "dp_released"]


def make_parser(description: str) -> argparse.ArgumentParser:
    """Build the CLI shared by every experiment module."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--dataset", choices=list(core.DATASETS.keys()) + ["all"],
                   default="all", help="Dataset key, or 'all'.")
    p.add_argument("--clamp-mode", choices=CLAMP_MODES + ["both"], default="static",
                   help="Definition 5.4 clamp: static (Option A) / dp_released "
                        "(Option B) / both.  Default static.")
    p.add_argument("--output-dir", default="results")
    p.add_argument("--workers", type=int, default=0,
                   help="Parallel worker processes (0 = cpu-1).")
    p.add_argument("--trials", type=int, default=1,
                   help="Independent noise-seed repetitions per config "
                        "(per-trial rows + mean/std aggregate).")
    p.add_argument("--epsilon-count", type=float, default=0.05,
                   help="eps_count for the DP publisher count (Sec. 6.3 step 1).")
    p.add_argument("--max-publishers", type=int, default=None,
                   help="P_max sensitivity-binding cap (Sec. 6.5).")
    p.add_argument("--ablation-P", type=int, default=3,
                   help="P_min for the ablation/overhead/average-case experiments.")
    p.add_argument("--k-ext", type=int, default=3, help="K_ext (Sec. 6.7).")
    p.add_argument("--eps-clip", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alpha", type=float, default=0.25)
    p.add_argument("--I-max", type=int, default=20)
    p.add_argument("--n-restarts", type=int, default=3)
    p.add_argument("--restart-rng-seed", type=int, default=12345)
    p.add_argument("--max-energy-timestamps", type=int, default=None)
    p.add_argument("--max-traffic-rows", type=int, default=None)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--strategies", nargs="+", default=None)
    p.add_argument("--use-grid-config", default=None,
                   help="grid_canonical.json to source MAE-optimal params from "
                        "(Sec. 7.5).  Defaults to <output-dir>/grid_canonical.json.")
    p.add_argument("--grid-eps", type=float, default=None,
                   help="Restrict the §7.5 grid search to this single epsilon and "
                        "write a grid_canonical_eps<eps>.json fragment, so the "
                        "grid phase can be split across nodes per epsilon "
                        "(experiments.grid_search only).")
    p.add_argument("--sensor-shard", default=None,
                   help="Round-robin sensor sharding 'i/k': process only "
                        "sensors[i::k] of each dataset, so a heavy dataset's "
                        "per-level sweep can run one task per sensor-group on "
                        "separate nodes (shortens the energy sweep long pole).")
    p.add_argument("--grid-trial", type=int, default=None,
                   help="Run ONE trial (noise seed) of the §7.5 grid and write a "
                        "per-trial full-grid fragment, so the grid's --trials "
                        "repetitions can be split across nodes as separate tasks. "
                        "_load_grid_config averages MAE across the trial fragments "
                        "before picking each strategy's optimum "
                        "(experiments.grid_search only).")
    p.add_argument("--quick", action="store_true", help="Reduced grids.")
    p.add_argument("--no-log-messages", dest="log_messages", action="store_false",
                   default=True, help="Skip the per-release messages CSV.")
    p.add_argument("--generate-plots", dest="generate_plots", action="store_true",
                   default=False, help="Render PNGs inline (default: off).")
    return p


def resolve(args) -> argparse.Namespace:
    """Fill every attribute the core functions read via getattr, and derive the
    target dataset / clamp-mode lists and the parameter grids."""
    args.workers = core._default_workers(args.workers)
    args.skip_extras = False
    args.tune_only = False
    args.grid_search = False
    args.grid_first = False
    # Targets / clamp modes.
    args._targets = (list(core.DATASETS.keys()) if args.dataset == "all"
                     else [args.dataset])
    args._clamp_modes = (CLAMP_MODES if args.clamp_mode == "both"
                         else [args.clamp_mode])
    # Strategies + parameter grids (mirror run_experiment.main defaults).
    args.strategies = args.strategies or core.ALL_STRATEGIES
    if args.quick:
        args.s_values = [1, 2, 4]
        args.eps_values = [0.5, 1.0, 2.0]
        args.w_values = [6, 10]
    else:
        args.s_values = [1, 2, 4, 6]
        args.eps_values = [0.5, 1.0, 2.0, 4.0]
        args.w_values = [4, 8, 10, 12]
    # Optional round-robin sensor shard 'i/k' -> args._sensor_shard = (i, k).
    args._sensor_shard = None
    if getattr(args, "sensor_shard", None):
        try:
            i, k = (int(x) for x in str(args.sensor_shard).split("/"))
            if k > 0 and 0 <= i < k:
                args._sensor_shard = (i, k)
        except Exception:
            raise SystemExit(f"--sensor-shard must be 'i/k' (got {args.sensor_shard!r})")
    # Canonical grid config (Sec. 7.5): explicit path, else auto-detect.
    path = args.use_grid_config or core._grid_canonical_path(args.output_dir)
    args.grid_config = core._load_grid_config(path)
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"workers={args.workers}  datasets={args._targets}  "
                f"clamp={args._clamp_modes}  trials={args.trials}")
    return args


def cross_dir(args, clamp_mode: str) -> str:
    return os.path.join(args.output_dir, "cross_dataset", clamp_mode)


def run_main_phase(args, phases: set[str]) -> None:
    """Run one main-pipeline phase ({sweep|intro|tuning|extras}) over every
    target dataset x clamp mode, then write the cross-dataset aggregates."""
    sweep_frames, fig1_frames = [], []
    greedy, brute, gap = [], [], []
    for clamp_mode in args._clamp_modes:
        for name in args._targets:
            try:
                res = core.run_dataset(
                    name, args.s_values, args.eps_values, args.w_values,
                    args.strategies, args.output_dir, args,
                    clamp_mode=clamp_mode, quick=args.quick,
                    skip_extras=args.skip_extras, workers=args.workers,
                    phases=phases,
                )
            except FileNotFoundError as e:
                logger.warning(f"Skipping {name}: {e}")
                continue
            if "sweep" in res:
                sweep_frames.append(res["sweep"])
            if "figure1" in res:
                fig1_frames.append(res["figure1"])
            if "tuning_greedy" in res:
                greedy.append(res["tuning_greedy"])
                brute.append(res["tuning_brute"])
                gap.append(res["tuning_gap"])
    if {"sweep", "intro"} & phases:
        core._write_cross_dataset_sweep_and_fig1(
            args.output_dir, args._clamp_modes, sweep_frames, fig1_frames)
    if "tuning" in phases:
        core._write_cross_dataset_tuning(
            args.output_dir, args._clamp_modes, greedy, brute, gap)


def run_single_axis(args, which: str) -> None:
    """Run one single-axis experiment letter (A/B/C/D/F/G/H) over all targets."""
    core.run_single_axis_experiments(
        args._targets, args._clamp_modes, args.output_dir, args,
        args.strategies, which=which)

