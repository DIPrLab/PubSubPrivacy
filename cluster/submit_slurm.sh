#!/usr/bin/env bash
# ============================================================================
# submit_slurm.sh -- submit the full experiment suite to a SLURM
# cluster as a JOB ARRAY (the standard SLURM pattern
# for "the same calculation on different data"), then chain an aggregation job.
#
#   1. gen_jobs.sh produces a joblist (one shard per line).
#   2. We submit a job array sized 1..N (N = number of shards), one array task
#      per shard, each on its own node with --cpus-per-task cores that
#      run_experiment.py uses as --workers (set via WORKERS at gen time).
#   3. A dependent aggregation job (afterok) merges every shard into
#      $OUT/combined once the array finishes.
#
# Usage (from the repo root on the cluster's login node):
#   cluster/submit_slurm.sh
#   CPUS=16 PARTITION=medium TIME=2-00:00:00 cluster/submit_slurm.sh
#   SHARD_BY=dataset-clamp-exp THROTTLE=20 cluster/submit_slurm.sh
#   DRY_RUN=1 cluster/submit_slurm.sh        # print plan + joblist, submit nothing
#
# Knobs (plus every gen_jobs.sh knob: SHARD_BY, DATASETS, CLAMP_MODES,
# EXPERIMENTS, EXTRA_ARGS, OUT):
#   PARTITION   SLURM partition           (default: medium)
#   TIME        per-task walltime         (default: 24:00:00)
#   CPUS        --cpus-per-task per shard (default: 8; also becomes --workers)
#   THROTTLE    max concurrent tasks per array chunk (default: unset = unbounded)
#   MAX_ARRAY   largest array to submit (default: 1000, under the cluster's
#               MaxArraySize); longer joblists are split into chained chunks
#   PLOTS       render figures at the end (default: 0)
#   PUBSUB_ENV_SETUP  shell to activate a modern-Python env on the node
#                     (e.g. 'source ~/pubsub/.venv/bin/activate' or
#                      'conda activate pubsub').  >= Python 3.10 required.
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
OUT="${OUT:-results_cluster}"
PARTITION="${PARTITION:-medium}"
TIME="${TIME:-24:00:00}"
CPUS="${CPUS:-8}"
THROTTLE="${THROTTLE:-}"
PLOTS="${PLOTS:-0}"
DRY_RUN="${DRY_RUN:-0}"
MAX_ARRAY="${MAX_ARRAY:-1000}"

# Each shard's intra-node fan-out should match the SLURM core allocation.
export WORKERS="$CPUS"
export OUT

mkdir -p "$OUT"
JOBLIST="$OUT/joblist.txt"
bash "$HERE/gen_jobs.sh" > "$JOBLIST"
N=$(wc -l < "$JOBLIST")
[ "$N" -gt 0 ] || { echo "gen_jobs.sh produced no shards" >&2; exit 1; }


# Split into grid shards (phase 1, write canonical config) and the rest
# (phase 2, F/G/H consume the grid via --use-grid-config).  Coarse 'gridfull'
# sharding has no '--grid-search' line, so NG=0 -> single array.  Fine
# 'dataset-clamp-exp' sharding maximizes node usage (heavy datasets spread
# their experiments across many nodes instead of one straggler shard).
GRIDLIST="$OUT/joblist.grid.txt"; RESTLIST="$OUT/joblist.rest.txt"
grep -- '-m experiments.grid_search' "$JOBLIST" > "$GRIDLIST" || true
grep -v -- '-m experiments.grid_search' "$JOBLIST" > "$RESTLIST" || true
NG=$(wc -l < "$GRIDLIST"); NR=$(wc -l < "$RESTLIST")
TWO_PHASE=0; [ "$NG" -gt 0 ] && [ "$NR" -gt 0 ] && TWO_PHASE=1

nchunks() { echo $(( ($1 + MAX_ARRAY - 1) / MAX_ARRAY )); }

echo "SLURM submission plan:"
echo "  shards (array tasks) : $N   (SHARD_BY=${SHARD_BY:-dataset-clamp})"
if [ "$TWO_PHASE" = "1" ]; then
  echo "  scheduling           : TWO-PHASE  (phase1 grid=$NG in $(nchunks "$NG") array(s)  ->  phase2 experiments=$NR in $(nchunks "$NR") array(s), afterok)"
else
  echo "  scheduling           : $N tasks in $(nchunks "$N") array(s) of <= $MAX_ARRAY"
fi
echo "  partition / time     : $PARTITION / $TIME"
echo "  cpus-per-task        : $CPUS  (= run_experiment.py --workers)"
echo "  output               : $OUT   (combined -> $OUT/combined)"
echo "  joblist              : $JOBLIST"

if [ "$DRY_RUN" = "1" ]; then
  echo "---- DRY RUN: joblist ----"; cat "$JOBLIST"
  if [ "$TWO_PHASE" = "1" ]; then
    echo "---- would submit PHASE 1: $(nchunks "$NG") array(s) over $NG grid shards ----"
    echo "---- would submit PHASE 2: $(nchunks "$NR") array(s) over $NR shards, --dependency=afterok:<all phase-1 arrays> ----"
  else
    echo "---- would submit: $(nchunks "$N") array(s) over $N shards ----"
  fi
  echo "---- then aggregation: sbatch --dependency=afterany:<all experiment arrays> slurm_aggregate.sbatch ----"
  exit 0
fi

submit_array() {  # $1 joblist  $2 spec  [$3 dependency-jid] -> echoes job id
  local jl="$1" spec="$2" dep="${3:-}" deparg=()
  [ -n "$dep" ] && deparg=(--dependency=afterok:"$dep")
  sbatch --parsable --array="$spec" --partition="$PARTITION" --time="$TIME" \
    --cpus-per-task="$CPUS" \
    --export="ALL,JOBLIST=$jl,REPO=$REPO,OUT=$OUT,PUBSUB_ENV_SETUP=${PUBSUB_ENV_SETUP:-}" \
    "${deparg[@]}" "$HERE/slurm_array.sbatch"
}

submit_chunked() {  # $1 joblist  [$2 colon-joined dependency jids] -> echoes colon-joined jids
  # SLURM rejects arrays past MaxArraySize, so a long joblist is submitted as
  # consecutive chunk files of <= MAX_ARRAY lines, each its own array; callers
  # depend on all of them (afterok:jid1:jid2:...).
  local jl="$1" dep="${2:-}" n start=1 end chunk spec jid jids=""
  n=$(wc -l < "$jl")
  while [ "$start" -le "$n" ]; do
    end=$(( start + MAX_ARRAY - 1 )); [ "$end" -gt "$n" ] && end=$n
    chunk="${jl%.txt}.part$(( (start - 1) / MAX_ARRAY + 1 )).txt"
    sed -n "${start},${end}p" "$jl" > "$chunk"
    spec="1-$(( end - start + 1 ))"
    [ -n "$THROTTLE" ] && spec="${spec}%${THROTTLE}"
    jid=$(submit_array "$chunk" "$spec" "$dep")
    echo "  submitted $chunk ($spec): $jid" >&2
    jids="${jids:+$jids:}$jid"
    start=$(( end + 1 ))
  done
  echo "$jids"
}

if [ "$TWO_PHASE" = "1" ]; then
  GRID_JIDS=$(submit_chunked "$GRIDLIST")
  echo "Submitted PHASE 1 grid ($NG tasks): $GRID_JIDS"
  ARRAY_JID=$(submit_chunked "$RESTLIST" "$GRID_JIDS")
  echo "Submitted PHASE 2 experiments ($NR tasks, afterok:$GRID_JIDS): $ARRAY_JID"
else
  ARRAY_JID=$(submit_chunked "$JOBLIST")
  echo "Submitted ($N tasks): $ARRAY_JID"
fi

# afterany: merge whatever finished even if a few shards failed, so one bad
# shard never blocks the whole result tree (re-run it and re-aggregate).
AGG_JID=$(sbatch --parsable \
  --dependency=afterany:"$ARRAY_JID" --partition="$PARTITION" \
  --export="ALL,REPO=$REPO,OUT=$OUT,PLOTS=$PLOTS,PUBSUB_ENV_SETUP=${PUBSUB_ENV_SETUP:-}" \
  "$HERE/slurm_aggregate.sbatch")
echo "Submitted aggregation (afterany:$ARRAY_JID): $AGG_JID"
echo "Track with: squeue -u \$USER"
