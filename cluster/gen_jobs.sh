#!/usr/bin/env bash
# ============================================================================
# gen_jobs.sh -- emit one fully-formed `run_experiment.py` command per line.
#
# Each emitted line is an INDEPENDENT shard that writes to its own output
# sub-directory, so the whole job list is embarrassingly parallel and can be
# fanned out across cores/nodes by GNU parallel (see run_cluster.sh).
#
# Sharding granularity is controlled by SHARD_BY:
#   dataset             one shard per dataset            (12 datasets-ish ->  6 shards)
#   dataset-clamp       per (dataset x clamp_mode)       (default;  6 x 2 = 12 shards)
#   dataset-clamp-exp   per (dataset x clamp x experiment) (finest; ~6x2x8 shards)
#
# Environment knobs (all optional; sensible defaults below):
#   OUT          base output dir                        (default: results_cluster)
#   SHARD_BY     granularity                            (default: dataset-clamp)
#   DATASETS     space-separated dataset keys           (default: all six)
#   CLAMP_MODES  space-separated clamp modes            (default: static dp_released)
#   EXPERIMENTS  experiments when SHARD_BY=...-exp       (default: sweep A B C D F G H grid)
#                ('grid' maps to --grid-search = paper Sec. 7.5)
#   WORKERS      --workers per shard (intra-node fan-out)(default: 0 = cpu-1)
#   EXTRA_ARGS   appended verbatim to every command      (e.g. "--max-rows 5000")
#   PYTHON       python interpreter                      (default: python)
#   RUNNER       path to run_experiment.py               (default: ../run_experiment.py
#                                                          relative to this script)
#
# Output:  job command lines on stdout.  Pipe to GNU parallel, xargs, or a
#          scheduler array.  Nothing is executed here.
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
RUNNER="${RUNNER:-$HERE/../run_experiment.py}"
OUT="${OUT:-results_cluster}"
SHARD_BY="${SHARD_BY:-dataset-clamp}"
DATASETS="${DATASETS:-energy traffic wearable pune mobility manufacturing}"
CLAMP_MODES="${CLAMP_MODES:-static dp_released}"
EXPERIMENTS="${EXPERIMENTS:-grid sweep A B C D F G H}"
WORKERS="${WORKERS:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

emit() {
  # $1 dataset  $2 clamp  $3 experiment  $4 shard-tag
  #   exp='grid'      => --grid-search             (Sec. 7.5, write canonical)
  #   exp='gridfull'  => --grid-first --experiment full
  #                       (grid search first, then full pipeline CONSUMES the
  #                        canonical config in the same process -- paper Sec.
  #                        7.5: grid fixes the params for the rest)
  #   else            => --experiment <exp>.  In fine mode F/G/H additionally
  #                       get --use-grid-config pointing at the (ds,clamp) grid
  #                       shard's canonical config so they consume the grid
  #                       optimum produced by phase 1.
  local ds="$1" clamp="$2" exp="$3" tag="$4" mode extra="$EXTRA_ARGS"
  case "$exp" in
    grid)     mode="--grid-search" ;;
    gridfull) mode="--grid-first --experiment full" ;;
    F|G|H)
      mode="--experiment $exp"
      extra="$extra --use-grid-config \"$OUT/shards/${ds}__${clamp}__grid/grid_canonical.json\"" ;;
    *)        mode="--experiment $exp" ;;
  esac
  echo "$PYTHON \"$RUNNER\" $mode --dataset $ds --clamp-mode $clamp" \
       "--workers $WORKERS --output-dir \"$OUT/shards/$tag\" $extra"
}

# Coarse shards run a SELF-CONTAINED 'gridfull' job: --grid-first runs the
# Sec. 7.5 grid search, writes grid_canonical.json, and the full pipeline
# (sweep + intro/Fig.1 + tuning + extras incl. the K_ext induced-latency sweep
# Sec. 7.10 + single-axis A/B/C/D/F/G/H) then CONSUMES it -- so the whole of
# Sec. 7 is covered by one shard with the grid-optimal params, no cross-shard
# coordination.  The fine 'dataset-clamp-exp' granularity MAXIMIZES parallelism
# (~6x2x9 shards instead of 12): it emits a separate fast 'grid' shard per
# (ds,clamp) plus one shard per experiment, and the F/G/H shards reference that
# grid shard's grid_canonical.json via --use-grid-config.  run_cluster.sh /
# submit_coeus.sh then run all 'grid' shards in PHASE 1 and the experiment
# shards in PHASE 2 (depends on phase 1), so the grid still fixes the params for
# F/G/H -- but the heavy datasets (e.g. energy) spread their experiments across
# many nodes instead of one straggler shard.  'grid' is listed FIRST in
# EXPERIMENTS so the phase split is unambiguous.
for ds in $DATASETS; do
  for clamp in $CLAMP_MODES; do
    case "$SHARD_BY" in
      dataset)
        # one shard per dataset (run both clamp modes inside the shard)
        [ "$clamp" = "static" ] || continue
        emit "$ds" both gridfull "${ds}" ;;
      dataset-clamp)
        emit "$ds" "$clamp" gridfull "${ds}__${clamp}" ;;
      dataset-clamp-exp)
        for exp in $EXPERIMENTS; do
          emit "$ds" "$clamp" "$exp" "${ds}__${clamp}__${exp}"
        done ;;
      *)
        echo "gen_jobs.sh: unknown SHARD_BY='$SHARD_BY'" >&2; exit 2 ;;
    esac
  done
done
