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
#   EXPERIMENTS  experiments when SHARD_BY=...-exp
#                (default: grid sweep intro tuning extras A B C D F G H L;
#                 each token maps to `python -m experiments.<module>`;
#                 'grid' = paper Sec. 7.5; sweep is split per SWEEP_STRATEGIES)
#   SWEEP_STRATEGIES  one sweep shard per strategy in fine mode (default: 8)
#   GRID_EPS          one grid shard per epsilon in fine mode (default: 0.5 1 2 4)
#   LOG_MESSAGES      1 to re-enable per-release message CSVs (default: 0 = off)
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
REPO="$(cd "$HERE/.." && pwd)"
PYTHON="${PYTHON:-python}"
RUNNER="${RUNNER:-$REPO/run_experiment.py}"
OUT="${OUT:-results_cluster}"
SHARD_BY="${SHARD_BY:-dataset-clamp}"
DATASETS="${DATASETS:-energy traffic wearable pune mobility manufacturing}"
# Static (Option A) clamp only for now; set CLAMP_MODES="static dp_released"
# to also run the Option B DP-released-clamp configuration.
CLAMP_MODES="${CLAMP_MODES:-static}"
# Main-pipeline phases (sweep/intro/tuning/extras) are now separate shards so a
# heavy dataset spreads them across nodes; grid first (phase 1), the rest phase 2.
# Each token maps to a split-out experiment MODULE (experiments/<name>.py),
# run as `python -m experiments.<name>`.  'L' = subscriptions at every topic-
# hierarchy level.  'gridfull' (coarse only) runs the self-contained
# --grid-first --experiment full via the core run_experiment.py.
EXPERIMENTS="${EXPERIMENTS:-grid sweep intro tuning extras A B C D F G H L}"
WORKERS="${WORKERS:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
# Per-release message logging is OFF by default on the cluster: the per-release
# message buffer is what OOMs the energy sweep, and ALL paper metrics are still
# computed without it (the workers return scalar metric dicts).  Set
# LOG_MESSAGES=1 to re-enable the per-release message CSVs.  We only inject the
# flag when the caller hasn't already named it in EXTRA_ARGS.
LOG_MESSAGES="${LOG_MESSAGES:-0}"
if [ "$LOG_MESSAGES" != "1" ] && [[ "$EXTRA_ARGS" != *"--no-log-messages"* ]] \
   && [[ "$EXTRA_ARGS" != *"--log-messages"* ]]; then
  EXTRA_ARGS="$EXTRA_ARGS --no-log-messages"
fi
# The per-level sweep is the heaviest experiment (every topic level x strategy x
# P x eps x w x trials), so in fine mode we SPLIT it into one shard per budget
# strategy (each --strategies <s>) to spread the cost across cluster nodes.
SWEEP_STRATEGIES="${SWEEP_STRATEGIES:-uniform sample budget_distribution budget_absorption p_gated_uniform p_gated_sample p_gated_ba n_weighted}"
# The §7.5 grid (phase 1) is the other long pole: per dataset it sweeps
# P_min x P_max x Delta_t x K_ext x strategies for EACH epsilon serially, so the
# big datasets (energy/traffic) take ~30 min on one node.  Split it one shard
# per epsilon (each --grid-eps <e>), all sharing the ${ds}__${clamp}__grid
# output dir so the per-eps grid_canonical_eps<e>.json fragments collect there;
# _load_grid_config merges them for the F/G/H/L consumers.  Must match the
# epsilon set the experiments use (experiments/_common.py eps_values).
GRID_EPS="${GRID_EPS:-0.5 1.0 2.0 4.0}"

# token -> experiments.<module>
_module_of() {
  case "$1" in
    grid)   echo experiments.grid_search ;;
    sweep)  echo experiments.sweep ;;
    intro)  echo experiments.intro ;;
    tuning) echo experiments.tuning ;;
    extras) echo experiments.extras ;;
    A)      echo experiments.greedy_vs_brute ;;
    B)      echo experiments.window ;;
    C)      echo experiments.epsilon ;;
    D)      echo experiments.plugin_path ;;
    F)      echo experiments.ablation ;;
    G)      echo experiments.overhead ;;
    H)      echo experiments.average_case ;;
    L)      echo experiments.subscription_levels ;;
    *)      echo "" ;;
  esac
}

emit() {
  # $1 dataset  $2 clamp  $3 experiment-token  $4 shard-tag
  #   'gridfull' (coarse) => core run_experiment.py --grid-first --experiment full
  #   otherwise           => python -m experiments.<module>.  The F/G/H/L
  #                          modules also get --use-grid-config pointing at the
  #                          (ds,clamp) grid shard so they consume the Sec. 7.5
  #                          optimum produced by phase 1.
  local ds="$1" clamp="$2" exp="$3" tag="$4" extra="$EXTRA_ARGS"
  if [ "$exp" = "gridfull" ]; then
    echo "$PYTHON \"$RUNNER\" --grid-first --experiment full --dataset $ds" \
         "--clamp-mode $clamp --workers $WORKERS --output-dir \"$OUT/shards/$tag\" $extra"
    return
  fi
  local M; M="$(_module_of "$exp")"
  if [ -z "$M" ]; then echo "gen_jobs.sh: unknown experiment '$exp'" >&2; exit 2; fi
  case "$exp" in
    F|G|H|L)
      extra="$extra --use-grid-config \"$OUT/shards/${ds}__${clamp}__grid/grid_canonical.json\"" ;;
  esac
  echo "PYTHONPATH=\"$REPO\" $PYTHON -m $M --dataset $ds --clamp-mode $clamp" \
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
# For the coarse 'dataset' granularity we run all requested clamp modes inside
# ONE shard; map the CLAMP_MODES list onto run_experiment's --clamp-mode value
# ('both' iff >= 2 modes, else the single mode -> honors CLAMP_MODES=static).
_NCLAMP=$(echo $CLAMP_MODES | wc -w)
if [ "$_NCLAMP" -ge 2 ]; then CLAMP_ARG="both"; else CLAMP_ARG="$CLAMP_MODES"; fi

if [ "$SHARD_BY" = "dataset" ]; then
  for ds in $DATASETS; do
    emit "$ds" "$CLAMP_ARG" gridfull "${ds}"
  done
  exit 0
fi

for ds in $DATASETS; do
  for clamp in $CLAMP_MODES; do
    case "$SHARD_BY" in
      dataset-clamp)
        emit "$ds" "$clamp" gridfull "${ds}__${clamp}" ;;
      dataset-clamp-exp)
        for exp in $EXPERIMENTS; do
          if [ "$exp" = "sweep" ]; then
            # Split the heavy per-level sweep into one shard per strategy.
            for strat in $SWEEP_STRATEGIES; do
              EXTRA_ARGS="$EXTRA_ARGS --strategies $strat" \
                emit "$ds" "$clamp" sweep "${ds}__${clamp}__sweep_${strat}"
            done
          elif [ "$exp" = "grid" ]; then
            # Split the §7.5 grid into one shard per epsilon (across nodes).  All
            # share the ${ds}__${clamp}__grid dir, so the per-eps canonical
            # fragments land together for the merge the F/G/H/L consumers read.
            for geps in $GRID_EPS; do
              EXTRA_ARGS="$EXTRA_ARGS --grid-eps $geps" \
                emit "$ds" "$clamp" grid "${ds}__${clamp}__grid"
            done
          else
            emit "$ds" "$clamp" "$exp" "${ds}__${clamp}__${exp}"
          fi
        done ;;
      *)
        echo "gen_jobs.sh: unknown SHARD_BY='$SHARD_BY'" >&2; exit 2 ;;
    esac
  done
done
