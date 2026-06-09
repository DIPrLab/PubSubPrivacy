#!/usr/bin/env python3
"""Aggregate per-shard cluster outputs into one combined result tree.

Each shard produced by ``run_cluster.sh`` writes a self-contained
``run_experiment.py`` output directory (per-dataset dirs + a partial
``cross_dataset/`` tree).  This script:

  1. Copies every shard's per-dataset directory into ``--out`` (datasets are
     disjoint across dataset-sharded runs, so there are no collisions).
  2. Concatenates every same-named CSV found anywhere under the shards
     (sweep, tuning, experiments A-H, cross-dataset combined tables, ...) so
     the combined tree has one CSV per logical artifact spanning all shards.
  3. Optionally re-renders every figure from the combined CSVs via
     ``generate_plots.py``.

It is intentionally dependency-light (pandas + stdlib) and idempotent.
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import shutil
import subprocess
import sys
from collections import defaultdict

import pandas as pd


# ── Curated "paper bundle" manifest ──────────────────────────────────────
# The KEY individual (per-dataset) + combined (cross-dataset) artifacts the
# paper actually references, grouped by paper section.  The aggregation job
# copies just these into one flat directory (paper_bundle/) so the whole set
# the paper needs downloads in a single small scp -- without the multi-GB
# per-release message dumps or shard intermediates.  Patterns match file
# BASENAMES (fnmatch); {dataset}/{clamp} context is preserved in the copied
# filename.  Anything matching EXCLUDE_GLOBS is dropped even if it matched.
PAPER_BUNDLE = {
    "01_grid_search_sec7.5": [
        "grid_canonical.json", "*_gridsearch.csv", "*_gridsearch_best.csv",
    ],
    "02_main_sweep": [
        "sweep_results.csv", "sweep_results_aggregate.csv",
        "combined_sweep_results.csv",
    ],
    "03_intro_figures_sec1.3": [
        "*_figure1_kl_vs_P.csv", "*_figure1_kl_vs_P.png",
        "*_figure_extreme1_global.csv", "*_figure_extreme1_global.png",
        "*_figure_extreme1_1_per_type.csv", "*_figure_extreme1_1_per_type.png",
        "*_figure_extreme2_per_publisher.csv", "*_figure_extreme2_per_publisher.png",
        "*_figure_extremes_vs_ours.csv", "*_figure_extremes_vs_ours.png",
        "*_figure_u_shaped_P_vs_KL.csv", "*_figure_u_shaped_P_vs_KL.png",
        "figure1_all_datasets.csv", "figure1_all_datasets.png",
        "figure1_avg_across_datasets.csv",
    ],
    "04_tuning_sec6.9": [
        "*_tuning_strategy_summary.csv", "*_tuning_greedy.csv",
        "*_tuning_brute_force.csv", "*_tuning.png",
        "combined_tuning_greedy.csv", "combined_tuning_brute_force.csv",
        "combined_tuning_gap_summary.csv", "tuning_best_per_dataset.csv",
        "tuning_greedy_vs_brute_gap.png",
    ],
    "05_experiments_A-D": [
        "experiment_A_greedy_vs_brute.csv",
        "experiment_B_vary_w.csv", "experiment_B_vary_w_aggregate.csv", "experiment_B_vary_w.png",
        "experiment_C_vary_epsilon.csv", "experiment_C_vary_epsilon_aggregate.csv", "experiment_C_vary_epsilon.png",
        "experiment_D_plugin_summary.csv", "experiment_D_plugin_path.png",
    ],
    "06_ablation_overhead_avgcase_sec7.8-7.11": [
        "experiment_F_ablation.csv", "experiment_F_ablation_aggregate.csv", "experiment_F_ablation*.png",
        "experiment_G_overhead.csv", "experiment_G_overhead_aggregate.csv", "experiment_G_overhead*.png",
        "experiment_H_average_case.csv", "experiment_H_average_case_aggregate.csv", "experiment_H_average_case*.png",
    ],
    "07_subscription_levels": [
        "experiment_I_subscription_levels.csv",
        "experiment_I_subscription_levels_aggregate.csv",
        "experiment_I_subscription_levels*.png",
    ],
    "08_extras_kext_sec7.10": [
        "*_k_ext_sweep.csv", "*_k_ext_sweep.png",
        "*_n_weighted_spotlight.csv", "*_n_weighted_spotlight.png",
        "*_collusion.csv", "*_collusion.png",
    ],
    "09_provenance": [
        "*_clamps.csv", "*_topics.csv", "*_subscriber_filters.csv",
    ],
}
# Bulky per-release dumps and shard intermediates never belong in the bundle.
EXCLUDE_GLOBS = ["*_messages.csv", "*_plugin_releases.csv"]


def _find_csvs(root: str) -> dict[str, list[str]]:
    """Map relative-path-within-shard -> [absolute paths across shards]."""
    by_relpath: dict[str, list[str]] = defaultdict(list)
    for shard in sorted(os.listdir(root)):
        shard_dir = os.path.join(root, shard)
        if not os.path.isdir(shard_dir):
            continue
        for dirpath, _dirs, files in os.walk(shard_dir):
            for f in files:
                if not f.endswith(".csv"):
                    continue
                ap = os.path.join(dirpath, f)
                rel = os.path.relpath(ap, shard_dir)
                by_relpath[rel].append(ap)
    return by_relpath


def _excluded(basename: str) -> bool:
    return any(fnmatch.fnmatch(basename, g) for g in EXCLUDE_GLOBS)


def build_paper_bundle(combined_dir: str, bundle_dir: str) -> None:
    """Copy the curated key paper artifacts from the combined tree into one
    flat, section-organized directory for a single small download.

    Each match is copied to ``bundle_dir/<section>/<flattened-relpath>`` where
    the relpath (which carries dataset/clamp_mode) has its separators replaced
    by ``__`` so every section is one flat folder.  Writes an INDEX.md listing
    exactly what was collected (and what was intentionally left out).
    """
    if os.path.isdir(bundle_dir):
        shutil.rmtree(bundle_dir)
    os.makedirs(bundle_dir, exist_ok=True)

    # Index every file in the combined tree by basename for fast pattern match.
    all_files: list[str] = []
    for dirpath, _dirs, files in os.walk(combined_dir):
        # Don't descend into a bundle dir nested under combined.
        if os.path.abspath(dirpath).startswith(os.path.abspath(bundle_dir)):
            continue
        for f in files:
            all_files.append(os.path.join(dirpath, f))

    index_lines = ["# Paper result bundle", "",
                   f"Curated from `{combined_dir}` by the aggregation job — the key",
                   "individual (per-dataset) + combined (cross-dataset) artifacts the",
                   "paper references, in one directory for a single small download.",
                   "Bulky per-release message dumps are intentionally excluded.", ""]
    total = 0
    for section, patterns in PAPER_BUNDLE.items():
        sec_dir = os.path.join(bundle_dir, section)
        picked: list[str] = []
        for ap in all_files:
            base = os.path.basename(ap)
            if _excluded(base):
                continue
            if not any(fnmatch.fnmatch(base, pat) for pat in patterns):
                continue
            rel = os.path.relpath(ap, combined_dir)
            flat = rel.replace(os.sep, "__").replace("/", "__")
            os.makedirs(sec_dir, exist_ok=True)
            shutil.copy2(ap, os.path.join(sec_dir, flat))
            picked.append(flat)
        if picked:
            total += len(picked)
            index_lines.append(f"## {section}  ({len(picked)} files)")
            index_lines += [f"- {p}" for p in sorted(picked)]
            index_lines.append("")
    with open(os.path.join(bundle_dir, "INDEX.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(index_lines))
    print(f"Paper bundle -> {bundle_dir}  ({total} key artifacts across "
          f"{len(PAPER_BUNDLE)} sections; per-release message dumps excluded)")


def aggregate(shards_dir: str, out_dir: str, plots: bool,
              bundle_dir: str | None = None) -> None:
    os.makedirs(out_dir, exist_ok=True)
    by_relpath = _find_csvs(shards_dir)
    n_concat = 0
    for rel, paths in sorted(by_relpath.items()):
        dst = os.path.join(out_dir, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if len(paths) == 1:
            shutil.copy2(paths[0], dst)
            continue
        # Same artifact emitted by multiple shards -> concatenate, dedupe.
        frames = []
        for p in paths:
            try:
                frames.append(pd.read_csv(p))
            except Exception as exc:  # pragma: no cover - corrupt/empty shard CSV
                print(f"  WARN: skip {p}: {exc}", file=sys.stderr)
        if not frames:
            continue
        combined = pd.concat(frames, ignore_index=True)
        combined = combined.drop_duplicates()
        combined.to_csv(dst, index=False)
        n_concat += 1
    print(f"Aggregated CSVs -> {out_dir}  "
          f"({len(by_relpath)} artifacts, {n_concat} concatenated across shards)")

    # Also copy non-CSV artifacts (clamp manifests, topic CSVs already covered;
    # PNGs are regenerated below if --plots).
    if plots:
        runner_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        gen = os.path.join(runner_dir, "generate_plots.py")
        if os.path.exists(gen):
            print(f"Rendering figures from combined CSVs via {gen} ...")
            subprocess.run([sys.executable, gen, "--output-dir", out_dir],
                           check=False)
        else:
            print("generate_plots.py not found; skipping figure render.",
                  file=sys.stderr)

    # Curated single-directory download of the key paper artifacts (after any
    # plot render, so the bundle picks up freshly-rendered PNGs too).
    if bundle_dir:
        build_paper_bundle(out_dir, bundle_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards-dir", required=True,
                    help="Directory containing one sub-dir per shard.")
    ap.add_argument("--out", required=True,
                    help="Combined output directory to write.")
    ap.add_argument("--plots", action="store_true",
                    help="Re-render figures from the combined CSVs.")
    ap.add_argument("--paper-bundle", default=None,
                    help="Also collect the key paper artifacts into this one "
                         "flat directory for easy single-scp download.")
    args = ap.parse_args()
    aggregate(args.shards_dir, args.out, args.plots, bundle_dir=args.paper_bundle)


if __name__ == "__main__":
    main()
