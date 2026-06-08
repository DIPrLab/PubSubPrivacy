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
import os
import shutil
import subprocess
import sys
from collections import defaultdict

import pandas as pd


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


def aggregate(shards_dir: str, out_dir: str, plots: bool) -> None:
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards-dir", required=True,
                    help="Directory containing one sub-dir per shard.")
    ap.add_argument("--out", required=True,
                    help="Combined output directory to write.")
    ap.add_argument("--plots", action="store_true",
                    help="Re-render figures from the combined CSVs.")
    args = ap.parse_args()
    aggregate(args.shards_dir, args.out, args.plots)


if __name__ == "__main__":
    main()
