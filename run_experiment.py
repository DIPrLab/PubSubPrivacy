#!/usr/bin/env python3
"""Command-line entry point for the PubSubPrivacy experiment suite.

This is a THIN WRAPPER: all functional code lives in ``experiments/engine.py``
(the offline engine + orchestration) and ``experiments/live_broker.py``
(Experiment E).  This file only builds the argument parser and dispatches into
those modules.  Each ``experiments/<name>.py`` can also be run directly as
``python -m experiments.<name>``.
"""
from __future__ import annotations

import argparse
import os
import sys

from experiments.engine import *          # noqa: F401,F403  (engine surface for main)
from experiments.engine import (          # explicit: underscore names import * skips
    _dataset_dirs, _dataset_max_rows, _default_workers, _grid_canonical_path,
    _load_grid_config, _run_grid_search_block,
    _write_cross_dataset_sweep_and_fig1, _write_cross_dataset_tuning,
)
from experiments.live_broker import experiment_E_live_broker


def main():
    parser = argparse.ArgumentParser(
        description="Run clamped w-event DP with P-allocation on real-world datasets"
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()) + ["all"],
        default="all",
        help="Dataset to run; 'all' runs every entry in DATASETS",
    )
    parser.add_argument(
        "--clamp-mode",
        choices=CLAMP_MODES + ["both"],
        default="both",
        help="Definition 3.2: 'static' (Option A operator-declared), "
             "'dp_released' (Option B DP-released min/max), 'both' runs each "
             "as a side-by-side experiment.",
    )
    parser.add_argument("--eps-clip", type=float, default=0.1,
                        help="Option B calibration budget epsilon_clip (Def 3.2)")
    parser.add_argument("--epsilon-count", type=float, default=0.0,
                        help="eps_count: per-step budget spent to release a "
                             "differentially private publisher count |P_tau| "
                             "(sensitivity 1) when gating / walking the topic "
                             "hierarchy (paper Sec. 6.3 step 1, Table 3).  0 "
                             "uses the exact count (Kellaris baselines).")
    parser.add_argument("--max-publishers", type=int, default=None,
                        help="P_max: cap the multiplicity folded into the mean "
                             "so Delta_f = R/n changes by a bounded amount "
                             "across stream elements (Sec. 6.5).  None = no cap.")
    parser.add_argument("--ablation-P", type=int, default=3,
                        help="P_min used by the ablation / overhead / "
                             "average-case experiments (F/G/H).")
    parser.add_argument("--k-ext", type=int, default=3,
                        help="K_ext: max adaptive interval extensions used by "
                             "the ablation (F) interval-extension module and "
                             "the plugin path (Sec. 6.7).")
    parser.add_argument("--alpha", type=float, default=0.25,
                        help="Attribution-advantage target; Algorithm 2 seeds P_0 = ceil(1/alpha)")
    parser.add_argument("--I-max", type=int, default=20,
                        help="Algorithm 2 iteration cap for the greedy hill-climb over P")
    parser.add_argument("--n-restarts", type=int, default=3,
                        help="Algorithm 2 intelligent multi-start: the first "
                             "seed is P_0 = ceil(1/alpha); the remaining "
                             "n_restarts-1 seeds are drawn from quantiles of "
                             "the observed n_tau distribution.  1 disables "
                             "random restart and reverts to paper's "
                             "deterministic seed only.")
    parser.add_argument("--restart-rng-seed", type=int, default=12345,
                        help="Random seed used to pick the non-P_0 restart "
                             "seeds in Algorithm 2 (controls reproducibility "
                             "of the quantile fallback draw).")
    parser.add_argument("--log-messages", dest="log_messages",
                        action="store_true", default=True,
                        help="Write per-release message CSV (true aggregate, "
                             "noisy value, n_tau, eps_tau, lambda_tau, "
                             "deferred flag, ...) for every DP run.  Default: on.")
    parser.add_argument("--no-log-messages", dest="log_messages",
                        action="store_false",
                        help="Disable per-release message logging.")
    parser.add_argument("--generate-plots", dest="generate_plots",
                        action="store_true", default=False,
                        help="Generate PNG plots inline during the run.  "
                             "Default: off -- use generate_plots.py post-hoc.")
    parser.add_argument("--no-generate-plots", dest="generate_plots",
                        action="store_false",
                        help="Skip inline plot generation (the canonical "
                             "workflow: run experiments, then generate plots "
                             "separately from the CSVs via generate_plots.py).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for Option B calibration noise")
    parser.add_argument("--trials", type=int, default=1,
                        help="Number of independent noise-seed repetitions per "
                             "configuration for the sweep and Experiments "
                             "B/C/F/G/H.  Each trial is one Laplace realization "
                             "with a distinct seed; the runner writes BOTH the "
                             "per-trial rows (column 'trial') AND a per-dataset "
                             "aggregate CSV (mean/std over trials).  Default 1 "
                             "(single realization).")
    parser.add_argument("--quick", action="store_true", help="Reduced sweep for testing")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--max-energy-timestamps", type=int, default=None,
                        help="Cap rows for the energy dataset (MCEC-Thai)")
    parser.add_argument("--max-traffic-rows", type=int, default=None,
                        help="Cap rows per file for the traffic dataset")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Cap rows for wearable/pune/mobility/manufacturing")
    parser.add_argument("--strategies", nargs="+", default=None,
                        help=f"Subset of {ALL_STRATEGIES} to run (defaults to all)")
    parser.add_argument("--skip-extras", action="store_true",
                        help="Skip the n-weighted / collusion / K_ext / tuning experiments")
    parser.add_argument("--tune-only", action="store_true",
                        help="Only run the Section 5.7 hyperparameter tuning")
    parser.add_argument("--grid-search", action="store_true",
                        help="Run the Section 7.5 utility-hyperparameter grid "
                             "search (P_min x P_max x Delta_t x K_ext, scored "
                             "by MAE) per dataset/strategy/epsilon, write "
                             "grid_canonical.json, then exit.")
    parser.add_argument("--use-grid-config", default=None,
                        help="Path to a grid_canonical.json produced by "
                             "--grid-search.  Downstream experiments resolve "
                             "their (P_min, P_max, K_ext) from the MAE-optimal "
                             "grid config per (dataset, clamp, strategy, eps) "
                             "instead of CLI defaults (paper Sec. 7.5: grid "
                             "search fixes the params for the rest).  Defaults "
                             "to auto-detecting <output-dir>/grid_canonical.json.")
    parser.add_argument("--grid-first", action="store_true",
                        help="Run the Section 7.5 grid search FIRST, write the "
                             "canonical config, and have this same run consume "
                             "it for every downstream experiment.")
    parser.add_argument(
        "--experiment",
        choices=["full", "sweep", "intro", "tuning", "extras", "tune",
                 "A", "B", "C", "D", "E", "F", "G", "H",
                 "ABC", "ABCD", "ABCDFGH", "FGH", "none"],
        default="full",
        help="'full' runs every main-pipeline phase (sweep + intro/Fig.1 + "
             "tuning + extras incl. K_ext) + single-axis A/B/C/D/F/G/H.  The "
             "main-pipeline phases are individually selectable as SEPARATE "
             "shards for cluster parallelism: 'sweep' (P x eps x w grid), "
             "'intro' (Figure 1 / extremes / U-shape), 'tuning' (Algorithm 2 "
             "greedy+brute), 'extras' (n-weighted spotlight, collusion, K_ext "
             "induced-latency Sec. 7.10).  A/B/C/D/F/G/H pick one single-axis "
             "experiment: A greedy-vs-brute P-tuning, B vary-w, C vary-eps, "
             "D plugin end-to-end, F incremental-module ablation (Sec. 7.8, "
             "offline, no broker), G overhead/privacy-utility comparison "
             "(Sec. 7.9), H average-case utility (Sec. 7.11).  'E' runs the "
             "live-broker sanity subset.  'ABC'/'ABCD'/'FGH'/'ABCDFGH' run "
             "those subsets without the main pipeline.",
    )
    parser.add_argument(
        "--broker-host", default="localhost",
        help="MQTT broker host for Experiment E (live-broker sanity subset)",
    )
    parser.add_argument(
        "--broker-port", type=int, default=1883,
        help="MQTT broker port for Experiment E",
    )
    parser.add_argument(
        "--live-dt", type=float, default=0.1,
        help="Wall-clock seconds per logical timestamp for Experiment E "
             "(short values compress runtime; default 0.1s)",
    )
    parser.add_argument(
        "--live-n-steps", type=int, default=120,
        help="Cap on logical timestamps per config for Experiment E",
    )
    parser.add_argument(
        "--no-auto-broker", dest="auto_broker", action="store_false",
        default=True,
        help="Do not auto-start an embedded amqtt broker for Experiment E "
             "when none is listening; fail instead.  Default behaviour is to "
             "reuse any existing broker at --broker-host/--broker-port, "
             "otherwise spin one up for the run.",
    )
    parser.add_argument(
        "--include-energy-in-E", action="store_true", default=False,
        help="When running Experiment E with --dataset all, also include "
             "the 'energy' dataset.  Default: skip energy (~36k windows/"
             "sensor -> hours of broker traffic per config).",
    )
    parser.add_argument(
        "--live-scenarios", default="pooled,hierarchy",
        help="Comma-separated subset of {pooled,hierarchy} to run under "
             "Experiment E.  'pooled' exercises the many-publisher shared-"
             "leaf path; 'hierarchy' puts each publisher on its own leaf "
             "under the dataset's normative MQTT tree so P>=2 forces "
             "Algorithm 1 walk-ups on every release.",
    )
    parser.add_argument(
        "--live-n-subscribers", type=int, default=3,
        help="Number of concurrent subscriber clients attached to the "
             "protected prefix per Experiment E config.  The broker should "
             "fan each release out to every subscriber; a mismatch between "
             "broker_deliveries and expected_deliveries flags a fan-out bug.",
    )
    parser.add_argument(
        "--run-live-E-after-full", action="store_true", default=False,
        help="When --experiment full is used, additionally run Experiment E "
             "after the main pipeline completes (live MQTT broker on every "
             "non-energy dataset by default; respects --include-energy-in-E).",
    )
    parser.add_argument(
        "--workers", type=int, default=0,
        help="Parallel worker processes for the sweep and experiments "
             "A/B/C.  0 (default) uses os.cpu_count() - 1.  Set to 1 to "
             "run serially.",
    )
    args = parser.parse_args()
    args.workers = _default_workers(args.workers)
    logger.info(f"Using {args.workers} worker process(es) for parallel tasks")

    # When plot generation is disabled, short-circuit savefig so the
    # experiment functions keep writing their CSVs but produce no PNGs.
    # generate_plots.py reads those CSVs post-hoc and renders every figure.
    if not args.generate_plots:
        def _savefig_noop(*_a, **_kw):  # pragma: no cover - trivial
            return None
        plt.savefig = _savefig_noop
        logger.info("Inline plot generation DISABLED. Run `python "
                    "generate_plots.py --output-dir <results_dir>` "
                    "after the experiment completes to render every figure.")

    if args.quick:
        s_values = [1, 2, 4]
        eps_values = [0.5, 1.0, 2.0]
        w_values = [6, 10]
    else:
        s_values = [1, 2, 4, 6]
        eps_values = [0.5, 1.0, 2.0, 4.0]
        w_values = [4, 8, 10, 12]

    strategies = args.strategies or ALL_STRATEGIES
    invalid = [s for s in strategies if s not in ALL_STRATEGIES]
    if invalid:
        raise SystemExit(f"Unknown strategies: {invalid}.  Valid: {ALL_STRATEGIES}")

    os.makedirs(args.output_dir, exist_ok=True)
    targets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    clamp_modes = CLAMP_MODES if args.clamp_mode == "both" else [args.clamp_mode]

    sweep_frames: list[pd.DataFrame] = []
    figure1_frames: list[pd.DataFrame] = []
    greedy_frames: list[pd.DataFrame] = []
    brute_frames: list[pd.DataFrame] = []
    gap_frames: list[pd.DataFrame] = []

    # Resolve the canonical grid config (paper Sec. 7.5): explicit path wins,
    # else auto-detect <output-dir>/grid_canonical.json.
    grid_cfg_path = args.use_grid_config or _grid_canonical_path(args.output_dir)
    args.grid_config = _load_grid_config(grid_cfg_path)

    if args.grid_first:
        # Run the grid search first, then consume its canonical config below.
        path = _run_grid_search_block(args, targets, clamp_modes,
                                      eps_values, strategies)
        args.grid_config = _load_grid_config(path)

    if args.grid_search:
        _run_grid_search_block(args, targets, clamp_modes, eps_values, strategies)
        logger.info("Section 7.5 grid search complete; canonical config written. "
                    "Re-run the experiments with --use-grid-config "
                    f"{_grid_canonical_path(args.output_dir)} to consume it "
                    "(or use --grid-first to do both in one run).")
        return

    if args.tune_only:
        for clamp_mode in clamp_modes:
            for name in targets:
                first_sensor = DATASETS[name]["sensors"][0]
                prepared = prepare_dataset(
                    name,
                    clamp_mode=clamp_mode,
                    eps_clip=args.eps_clip,
                    seed=args.seed,
                    max_rows=_dataset_max_rows(name, args),
                    sensors=[first_sensor],
                )
                if prepared is None:
                    continue
                if first_sensor not in prepared.raw_per_pubs:
                    logger.warning(f"Skipping {name} tuning: no per-pub stream"); continue
                if first_sensor not in prepared.per_pubs:
                    continue
                pp, B = prepared.per_pubs[first_sensor]
                dirs = _dataset_dirs(args.output_dir, name, clamp_mode)
                tune = tune_hyperparameters(
                    pp, B, name, first_sensor, dirs["tuning"],
                    epsilon=1.0, w=8,
                    strategies=strategies,
                    alpha=args.alpha, I_max=args.I_max,
                    workers=args.workers,
                )
                for key in ("greedy", "brute_force", "gap_summary"):
                    tune[key]["dataset"] = name
                    tune[key]["sensor"] = first_sensor
                    tune[key]["clamp_mode"] = clamp_mode
                greedy_frames.append(tune["greedy"])
                brute_frames.append(tune["brute_force"])
                gap_frames.append(tune["gap_summary"])
        _write_cross_dataset_tuning(args.output_dir, clamp_modes,
                                    greedy_frames, brute_frames, gap_frames)
        return

    run_main_pipeline = args.experiment in (
        "full", "sweep", "intro", "tuning", "extras")
    # Which run_dataset phases this invocation runs (None = all, for 'full').
    _PHASE_OF = {"sweep": {"sweep"}, "intro": {"intro"},
                 "tuning": {"tuning"}, "extras": {"extras"}}
    main_phases = _PHASE_OF.get(args.experiment)  # None for 'full'
    run_single_axis = args.experiment in (
        "full", "ABC", "ABCD", "ABCDFGH", "FGH",
        "A", "B", "C", "D", "F", "G", "H",
    )
    run_live_E = args.experiment == "E"

    if run_live_E:
        # --dataset <name>   -> run E on that one dataset.
        # --dataset all      -> run E on every dataset EXCEPT energy
        #                       (energy has ~36k windows/sensor -> hours per
        #                       config on a live broker).  Set
        #                       --include-energy-in-E to force it.
        if args.dataset == "all":
            datasets_for_E = [ds for ds in DATASETS.keys()
                              if ds != "energy" or args.include_energy_in_E]
            if not args.include_energy_in_E:
                logger.info("[exp E] --dataset all: skipping 'energy' "
                            "(pass --include-energy-in-E to force)")
        else:
            datasets_for_E = [args.dataset]
            if args.dataset == "energy":
                logger.warning(
                    "[exp E] --dataset energy is strongly discouraged "
                    "(~36k windows/sensor). Recommend 'wearable' or "
                    "'manufacturing'."
                )
        for ds_for_E in datasets_for_E:
            logger.info(f"===== Experiment E: dataset={ds_for_E} =====")
            experiment_E_live_broker(
                ds_for_E, args.output_dir, args,
                broker_host=args.broker_host,
                broker_port=args.broker_port,
                live_dt=args.live_dt,
                n_steps=args.live_n_steps,
                seed=args.seed or 123,
                auto_start_broker=args.auto_broker,
            )
        logger.info(
            f"Experiment E complete ({len(datasets_for_E)} dataset(s)).")
        return

    if run_main_pipeline:
        for clamp_mode in clamp_modes:
            for name in targets:
                try:
                    results = run_dataset(
                        name, s_values, eps_values, w_values, strategies,
                        args.output_dir, args,
                        clamp_mode=clamp_mode,
                        quick=args.quick, skip_extras=args.skip_extras,
                        workers=args.workers,
                        phases=main_phases,
                    )
                except FileNotFoundError as e:
                    logger.warning(f"Skipping {name}: {e}")
                    continue
                if "sweep" in results:
                    sweep_frames.append(results["sweep"])
                if "figure1" in results:
                    figure1_frames.append(results["figure1"])
                if "tuning_greedy" in results:
                    greedy_frames.append(results["tuning_greedy"])
                    brute_frames.append(results["tuning_brute"])
                    gap_frames.append(results["tuning_gap"])

        _write_cross_dataset_sweep_and_fig1(args.output_dir, clamp_modes,
                                            sweep_frames, figure1_frames)
        _write_cross_dataset_tuning(args.output_dir, clamp_modes,
                                    greedy_frames, brute_frames, gap_frames)

    if run_single_axis:
        if args.experiment == "full":
            which = "ABCDFGH"   # full pipeline runs every single-axis experiment
        elif args.experiment in ("ABCD", "ABC", "ABCDFGH", "FGH"):
            which = args.experiment
        else:
            which = args.experiment  # single letter A/B/C/D/F/G/H
        run_single_axis_experiments(
            targets, clamp_modes, args.output_dir, args, strategies, which=which,
        )

    # Experiment E auto-runs after --experiment full when --run-live-E-after-full
    # is set.  Each dataset gets its own live-broker sub-run (pooled + hierarchy
    # scenarios, --live-n-subscribers concurrent subscribers); results land
    # under <output_dir>/experiments/E_live_broker/<dataset>/.
    if (args.experiment == "full"
            and getattr(args, "run_live_E_after_full", False)):
        e_targets = [ds for ds in (
            targets if args.dataset != "all" else list(DATASETS.keys())
        ) if ds != "energy" or args.include_energy_in_E]
        if "energy" in (targets if args.dataset != "all"
                        else list(DATASETS.keys())) \
                and not args.include_energy_in_E:
            logger.info("[exp E/auto] skipping 'energy' dataset (pass "
                        "--include-energy-in-E to force)")
        for ds_for_E in e_targets:
            logger.info(f"===== Experiment E (post-full): dataset={ds_for_E} =====")
            try:
                experiment_E_live_broker(
                    ds_for_E, args.output_dir, args,
                    broker_host=args.broker_host,
                    broker_port=args.broker_port,
                    live_dt=args.live_dt,
                    n_steps=args.live_n_steps,
                    seed=args.seed or 123,
                    auto_start_broker=args.auto_broker,
                )
            except Exception as exc:
                logger.exception(
                    f"[exp E/auto] failed on {ds_for_E}: {exc} "
                    "(continuing with next dataset)"
                )

    logger.info("All experiments complete.")




if __name__ == "__main__":
    main()
