# Cluster execution (generic shell + GNU parallel)

Scheduler-agnostic harness for running the full experiment suite across the
cores of one machine or across many nodes over SSH. No SLURM/Kubernetes
dependency — just `bash`, [GNU parallel](https://www.gnu.org/software/parallel/),
and (for multi-node) `ssh`.

Two levels of parallelism:

* **shards across nodes/cores** — GNU parallel distributes independent shards,
* **cores within a shard** — each shard is a `run_experiment.py` call that fans
  its own task list out over `--workers` cores.

Each shard writes to its own `OUT/shards/<tag>/` directory, so shards never
collide and the run is embarrassingly parallel.

## Files

| File | Role |
|------|------|
| `gen_jobs.sh` | Emits one fully-formed `run_experiment.py` command per line (the shard list). Executes nothing. |
| `run_cluster.sh` | Generates the shard list and fans it out with GNU parallel; optionally aggregates + plots at the end. |
| `aggregate.py` | Merges every shard's CSVs into one combined result tree (concatenating same-named artifacts) and can re-render all figures. |

## Quick start

```bash
# Local, use all cores, default sharding (dataset x clamp = 6 shards, static),
# aggregate into results_cluster/combined/ at the end.
cluster/run_cluster.sh

# Local, cap to 8 shards at a time.
JOBS=8 cluster/run_cluster.sh

# See the exact plan without running anything.
DRY_RUN=1 cluster/run_cluster.sh

# MAXIMALLY parallel: finer shards (dataset x clamp x experiment, ~354 shards
# static clamp) for a big node pool, 32 concurrent.  run_cluster.sh /
# submit_coeus.sh run the grid shards in PHASE 1 and the experiment shards in
# PHASE 2 automatically (F/G/H/L consume the grid via --use-grid-config; the
# heavy sweep is split into 8 strategies x 3 sensor-groups/dataset and the grid into 24
# per-eps shards/dataset); no manual two-step.
SHARD_BY=dataset-clamp-exp JOBS=32 cluster/run_cluster.sh

# Multi-node over SSH (nodes share the repo + data/ via NFS or identical stage).
SSHLOGINFILE=nodes.txt JOBS=4 cluster/run_cluster.sh   # 4 shards per node

# Smoke test on a laptop: smaller grids + capped rows passed through to every shard.
EXTRA_ARGS="--quick --max-rows 5000 --max-energy-timestamps 3000" \
  JOBS=4 cluster/run_cluster.sh
```

`nodes.txt` is a GNU parallel `--sshloginfile`: one `[ncores/]user@host` per
line, e.g.

```
8/alice@node01
8/alice@node02
4/alice@node03
```

## Knobs

All `gen_jobs.sh` knobs are honored by `run_cluster.sh`:

| Var | Default | Meaning |
|-----|---------|---------|
| `OUT` | `results_cluster` | base output dir |
| `SHARD_BY` | `dataset-clamp` | `dataset` \| `dataset-clamp` \| `dataset-clamp-exp` |
| `DATASETS` | all six | space-separated dataset keys |
| `CLAMP_MODES` | `static` | clamp modes to run (`static dp_released` for both) |
| `EXPERIMENTS` | `grid sweep intro tuning extras A B C D F G H L` | experiment modules for `dataset-clamp-exp` (each `python -m experiments.<module>`) |
| `SWEEP_STRATEGIES` | 8 strategies | one sweep shard emitted per strategy (fine mode) |
| `GRID_EPS` | `0.5 1.0 2.0 4.0` | one §7.5 grid shard emitted per epsilon (fine mode); fragments merged on consume |
| `LOG_MESSAGES` | `0` (off) | `1` re-enables the per-release message CSVs (off avoids the energy OOM) |
| `WORKERS` | `0` (cpu−1) | intra-shard `--workers` |
| `EXTRA_ARGS` | (empty) | appended verbatim to every shard command |
| `JOBS` | `100%` | GNU parallel `--jobs` |
| `SSHLOGINFILE` | (unset) | enable multi-node |
| `AGGREGATE` | `1` | run `aggregate.py` at the end |
| `PLOTS` | `0` | render PNGs after aggregation |
| `DRY_RUN` | `0` | print plan, run nothing |

## Output

```
results_cluster/
  joblist.txt              # the exact shard commands that ran
  logs/parallel.joblog     # GNU parallel per-shard exit codes + timings
  logs/runs/               # per-shard stdout/stderr (parallel --results)
  shards/<tag>/            # each shard's full run_experiment.py output tree
  combined/                # aggregate.py: one CSV per artifact across shards
                           #   (+ figures if PLOTS=1)
```

## No GNU parallel?

`gen_jobs.sh` is just a command emitter, so any launcher works:

```bash
# xargs fallback (N concurrent shards)
OUT=results_cluster cluster/gen_jobs.sh | xargs -P 8 -I{} bash -c '{}'
python cluster/aggregate.py --shards-dir results_cluster/shards --out results_cluster/combined
```
