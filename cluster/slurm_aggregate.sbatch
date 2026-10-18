#!/bin/bash
# ============================================================================
# coeus_aggregate.sbatch -- merge every shard's output into one combined tree.
# Submitted by submit_coeus.sh with --dependency=afterok on the job array, so
# it runs only after all shards complete.
# ============================================================================
#SBATCH --job-name=pubsubdp_agg
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail
REPO="${REPO:-$SLURM_SUBMIT_DIR}"
OUT="${OUT:-results_cluster}"
cd "$REPO"

if [ -n "${PUBSUB_ENV_SETUP:-}" ]; then
  eval "$PUBSUB_ENV_SETUP"
elif [ -f "$REPO/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$REPO/.venv/bin/activate"
else
  module load Python/gcc/3.7.5/gcc-6.3.0 2>/dev/null || true
fi

PLOTS_FLAG=""
[ "${PLOTS:-0}" = "1" ] && PLOTS_FLAG="--plots"
# Always also build the curated single-directory paper bundle so the key
# individual + combined results download in one small scp (see $OUT/paper_bundle).
python cluster/aggregate.py --shards-dir "$OUT/shards" --out "$OUT/combined" \
  --paper-bundle "$OUT/paper_bundle" $PLOTS_FLAG
echo "[$(date)] aggregation complete -> $OUT/combined"
echo "[$(date)] key paper results bundled -> $OUT/paper_bundle  (scp this one dir)"
