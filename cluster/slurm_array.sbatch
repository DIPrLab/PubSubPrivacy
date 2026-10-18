#!/bin/bash
# ============================================================================
# coeus_array.sbatch -- run the experiment shards as a SLURM JOB ARRAY on PSU
# Research Computing's Coeus cluster (https://sites.google.com/pdx.edu/research-computing).
#
# Coeus uses SLURM and explicitly recommends job arrays for "the same
# calculation on different data" -- which is exactly our sharded experiments.
# GNU parallel is NOT installed on Coeus, so on the cluster we use a job array
# instead: each array task runs ONE shard line from the joblist that
# gen_jobs.sh produced; intra-shard parallelism uses --cpus-per-task cores via
# run_experiment.py --workers.  (GNU parallel remains the local / SSH option;
# see run_cluster.sh.)
#
# Do not sbatch this directly -- use submit_coeus.sh, which generates the
# joblist, sizes --array, and chains the aggregation job.  The directives below
# are overridable on the submit_coeus.sh command line / environment.
# ============================================================================
#SBATCH --job-name=pubsubdp
#SBATCH --partition=medium            # Coeus default partition (7-day limit)
#SBATCH --nodes=1                     # one node per shard (shards are independent)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8             # intra-shard --workers; raise for big sweeps
#SBATCH --time=24:00:00
#SBATCH --output=%x_%A_%a.out         # <jobname>_<arrayjobid>_<taskid>.out
#SBATCH --error=%x_%A_%a.err

set -euo pipefail

# --- environment ------------------------------------------------------------
# Coeus uses Linux environment modules.  The stock Python is old
# (Python/gcc/3.7.5) but this codebase needs >= 3.10 (PEP 604 'int | None' and
# 'dict[str, ...]' syntax), so prefer a user venv/conda with a modern Python.
# Set PUBSUB_ENV_SETUP to whatever activates yours; the default tries a venv at
# repo-root/.venv and falls back to a module load.
JOBLIST="${JOBLIST:?set JOBLIST to the gen_jobs.sh output file}"
REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"

if [ -n "${PUBSUB_ENV_SETUP:-}" ]; then
  eval "$PUBSUB_ENV_SETUP"
elif [ -f "$REPO/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$REPO/.venv/bin/activate"
else
  module load Python/gcc/3.7.5/gcc-6.3.0 2>/dev/null || true
fi

# --- run this task's shard --------------------------------------------------
LINE="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$JOBLIST")"
if [ -z "$LINE" ]; then
  echo "No joblist line for array index ${SLURM_ARRAY_TASK_ID}" >&2
  exit 1
fi
echo "[$(date)] node=$(hostname) task=${SLURM_ARRAY_TASK_ID} cpus=${SLURM_CPUS_PER_TASK:-?}"
echo "  CMD: $LINE"
# Honor the SLURM core allocation as the intra-shard worker count.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
eval "$LINE"
echo "[$(date)] task ${SLURM_ARRAY_TASK_ID} done"
