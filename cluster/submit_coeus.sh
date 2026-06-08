#!/usr/bin/env bash
# ============================================================================
# submit_coeus.sh -- submit the full experiment suite to PSU Research
# Computing's Coeus cluster as a SLURM JOB ARRAY (the method Coeus recommends
# for "the same calculation on different data"), then chain an aggregation job.
#
#   1. gen_jobs.sh produces a joblist (one shard per line).
#   2. We submit a job array sized 1..N (N = number of shards), one array task
#      per shard, each on its own node with --cpus-per-task cores that
#      run_experiment.py uses as --workers (set via WORKERS at gen time).
#   3. A dependent aggregation job (afterok) merges every shard into
#      $OUT/combined once the array finishes.
#
# Usage (from the repo root on Coeus):
#   cluster/submit_coeus.sh
#   CPUS=16 PARTITION=medium TIME=2-00:00:00 cluster/submit_coeus.sh
#   SHARD_BY=dataset-clamp-exp THROTTLE=20 cluster/submit_coeus.sh
#   DRY_RUN=1 cluster/submit_coeus.sh        # print plan + joblist, submit nothing
#
# Knobs (plus every gen_jobs.sh knob: SHARD_BY, DATASETS, CLAMP_MODES,
# EXPERIMENTS, EXTRA_ARGS, OUT):
#   PARTITION   SLURM partition           (default: medium -- Coeus default)
#   TIME        per-task walltime         (default: 24:00:00)
#   CPUS        --cpus-per-task per shard (default: 8; also becomes --workers)
#   THROTTLE    max concurrent array tasks(default: unset = unbounded)
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

# Each shard's intra-node fan-out should match the SLURM core allocation.
export WORKERS="$CPUS"
export OUT

mkdir -p "$OUT"
JOBLIST="$OUT/joblist.txt"
bash "$HERE/gen_jobs.sh" > "$JOBLIST"
N=$(wc -l < "$JOBLIST")
[ "$N" -gt 0 ] || { echo "gen_jobs.sh produced no shards" >&2; exit 1; }

ARRAY_SPEC="1-$N"
[ -n "$THROTTLE" ] && ARRAY_SPEC="${ARRAY_SPEC}%${THROTTLE}"

# Split into grid shards (phase 1, write canonical config) and the rest
# (phase 2, F/G/H consume the grid via --use-grid-config).  Coarse 'gridfull'
# sharding has no '--grid-search' line, so NG=0 -> single array.  Fine
# 'dataset-clamp-exp' sharding maximizes node usage (heavy datasets spread
# their experiments across many nodes instead of one straggler shard).
GRIDLIST="$OUT/joblist.grid.txt"; RESTLIST="$OUT/joblist.rest.txt"
grep -- '--grid-search' "$JOBLIST" > "$GRIDLIST" || true
grep -v -- '--grid-search' "$JOBLIST" > "$RESTLIST" || true
NG=$(wc -l < "$GRIDLIST"); NR=$(wc -l < "$RESTLIST")
TWO_PHASE=0; [ "$NG" -gt 0 ] && [ "$NR" -gt 0 ] && TWO_PHASE=1

echo "Coeus submission plan:"
echo "  shards (array tasks) : $N   (SHARD_BY=${SHARD_BY:-dataset-clamp})"
if [ "$TWO_PHASE" = "1" ]; then
  echo "  scheduling           : TWO-PHASE  (phase1 grid=$NG  ->  phase2 experiments=$NR, afterok)"
else
  echo "  scheduling           : single array  $ARRAY_SPEC"
fi
echo "  partition / time     : $PARTITION / $TIME"
echo "  cpus-per-task        : $CPUS  (= run_experiment.py --workers)"
echo "  output               : $OUT   (combined -> $OUT/combined)"
echo "  joblist              : $JOBLIST"

if [ "$DRY_RUN" = "1" ]; then
  echo "---- DRY RUN: joblist ----"; cat "$JOBLIST"
  if [ "$TWO_PHASE" = "1" ]; then
    echo "---- would submit PHASE 1: sbatch --array=1-$NG ... coeus_array.sbatch  (grid) ----"
    echo "---- would submit PHASE 2: sbatch --array=1-$NR --dependency=afterok:<phase1> ... coeus_array.sbatch ----"
  else
    echo "---- would submit: sbatch --array=$ARRAY_SPEC --partition=$PARTITION --time=$TIME --cpus-per-task=$CPUS coeus_array.sbatch ----"
  fi
  echo "---- then aggregation: sbatch --dependency=afterok:<experiments> coeus_aggregate.sbatch ----"
  exit 0
fi

submit_array() {  # $1 joblist  $2 spec  [$3 dependency-jid] -> echoes job id
  local jl="$1" spec="$2" dep="${3:-}" deparg=()
  [ -n "$dep" ] && deparg=(--dependency=afterok:"$dep")
  sbatch --parsable --array="$spec" --partition="$PARTITION" --time="$TIME" \
    --cpus-per-task="$CPUS" \
    --export="ALL,JOBLIST=$jl,REPO=$REPO,OUT=$OUT,PUBSUB_ENV_SETUP=${PUBSUB_ENV_SETUP:-}" \
    "${deparg[@]}" "$HERE/coeus_array.sbatch"
}

if [ "$TWO_PHASE" = "1" ]; then
  GSPEC="1-$NG"; RSPEC="1-$NR"
  [ -n "$THROTTLE" ] && { GSPEC="${GSPEC}%${THROTTLE}"; RSPEC="${RSPEC}%${THROTTLE}"; }
  GRID_JID=$(JOBLIST="$GRIDLIST" submit_array "$GRIDLIST" "$GSPEC")
  echo "Submitted PHASE 1 grid array ($NG tasks): $GRID_JID"
  ARRAY_JID=$(JOBLIST="$RESTLIST" submit_array "$RESTLIST" "$RSPEC" "$GRID_JID")
  echo "Submitted PHASE 2 experiment array ($NR tasks, afterok:$GRID_JID): $ARRAY_JID"
else
  ARRAY_JID=$(submit_array "$JOBLIST" "$ARRAY_SPEC")
  echo "Submitted job array ($N tasks): $ARRAY_JID"
fi

AGG_JID=$(sbatch --parsable \
  --dependency=afterok:"$ARRAY_JID" --partition="$PARTITION" \
  --export="ALL,REPO=$REPO,OUT=$OUT,PLOTS=$PLOTS,PUBSUB_ENV_SETUP=${PUBSUB_ENV_SETUP:-}" \
  "$HERE/coeus_aggregate.sbatch")
echo "Submitted aggregation (afterok:$ARRAY_JID): $AGG_JID"
echo "Track with: squeue -u \$USER"
