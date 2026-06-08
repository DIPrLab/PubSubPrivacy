#!/usr/bin/env bash
# ============================================================================
# run_cluster.sh -- fan the experiment shards out across cores/nodes with
# GNU parallel, then (optionally) aggregate every shard into one result tree.
#
# This is scheduler-agnostic: it needs only bash + GNU parallel + ssh (for the
# multi-node case).  Each shard is one `run_experiment.py` invocation that uses
# --workers for intra-node parallelism, while GNU parallel distributes the
# shards themselves.  So you get two levels of parallelism: shards across
# nodes, cores within a shard.
#
# Usage:
#   cluster/run_cluster.sh                       # local, all cores
#   JOBS=8 cluster/run_cluster.sh                # local, 8 shards at a time
#   SSHLOGINFILE=nodes.txt cluster/run_cluster.sh   # multi-node over ssh
#   SHARD_BY=dataset-clamp-exp JOBS=32 cluster/run_cluster.sh   # finer shards
#   DRY_RUN=1 cluster/run_cluster.sh             # print the plan, run nothing
#
# Knobs (in addition to every gen_jobs.sh knob, which is honored):
#   JOBS          parallel --jobs value             (default: 100% of cores)
#   SSHLOGINFILE  GNU parallel --sshloginfile path  (default: unset -> local)
#   AGGREGATE     run aggregate.py at the end (1/0)  (default: 1)
#   PLOTS         render PNGs after aggregation (1/0)(default: 0)
#   DRY_RUN       print joblist + plan, execute none (default: 0)
#   OUT           base output dir                    (default: results_cluster)
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${OUT:-results_cluster}"
JOBS="${JOBS:-100%}"
AGGREGATE="${AGGREGATE:-1}"
PLOTS="${PLOTS:-0}"
DRY_RUN="${DRY_RUN:-0}"
PYTHON="${PYTHON:-python}"

export OUT  # so gen_jobs.sh writes shards under the same base dir

if ! command -v parallel >/dev/null 2>&1; then
  echo "ERROR: GNU parallel not found on PATH." >&2
  echo "  Debian/Ubuntu: sudo apt-get install parallel" >&2
  echo "  macOS:         brew install parallel" >&2
  echo "  (fallback) pipe gen_jobs.sh into 'xargs -P N -I{} bash -c {}')" >&2
  exit 3
fi

mkdir -p "$OUT/logs" "$OUT/shards"
JOBLIST="$OUT/joblist.txt"
bash "$HERE/gen_jobs.sh" > "$JOBLIST"
N=$(wc -l < "$JOBLIST")
echo "Generated $N shard(s) -> $JOBLIST  (SHARD_BY=${SHARD_BY:-dataset-clamp})"

# Assemble GNU parallel arguments.
PARGS=(--jobs "$JOBS" --bar --joblog "$OUT/logs/parallel.joblog"
       --results "$OUT/logs/runs" --halt soon,fail=10)
if [ -n "${SSHLOGINFILE:-}" ]; then
  # Multi-node: ship the working dir to each node and run there.  Nodes must
  # share the dataset 'data/' dir (NFS) or have it staged identically.
  PARGS+=(--sshloginfile "$SSHLOGINFILE" --workdir "$(pwd)")
  echo "Multi-node mode: distributing over hosts in $SSHLOGINFILE"
fi

if [ "$DRY_RUN" = "1" ]; then
  echo "---- DRY RUN: jobs that would execute ----"
  cat "$JOBLIST"
  echo "---- parallel ${PARGS[*]} ----"
  exit 0
fi

# Two-phase when fine-grained sharding emitted separate grid shards: PHASE 1
# runs the grid shards (which write grid_canonical.json), PHASE 2 runs the rest
# (the F/G/H shards consume the grid via --use-grid-config).  Coarse 'gridfull'
# shards have no '--grid-search' line, so this collapses to a single phase.
GRIDLIST="$OUT/joblist.grid.txt"
RESTLIST="$OUT/joblist.rest.txt"
grep -- '--grid-search' "$JOBLIST" > "$GRIDLIST" || true
grep -v -- '--grid-search' "$JOBLIST" > "$RESTLIST" || true
NG=$(wc -l < "$GRIDLIST"); NR=$(wc -l < "$RESTLIST")

if [ "$NG" -gt 0 ] && [ "$NR" -gt 0 ]; then
  echo "PHASE 1: $NG grid shard(s) (Sec. 7.5 canonical config)"
  parallel "${PARGS[@]}" bash -c {} :::: "$GRIDLIST"
  echo "PHASE 2: $NR experiment shard(s)"
  parallel "${PARGS[@]}" bash -c {} :::: "$RESTLIST"
else
  echo "Dispatching $N shard(s) with: parallel --jobs $JOBS ..."
  # Each line of JOBLIST is a complete shell command; run it via bash -c.
  parallel "${PARGS[@]}" bash -c {} :::: "$JOBLIST"
fi
echo "All shards complete."

if [ "$AGGREGATE" = "1" ]; then
  echo "Aggregating shards -> $OUT/combined ..."
  AGG_ARGS=(--shards-dir "$OUT/shards" --out "$OUT/combined")
  [ "$PLOTS" = "1" ] && AGG_ARGS+=(--plots)
  "$PYTHON" "$HERE/aggregate.py" "${AGG_ARGS[@]}"
fi
