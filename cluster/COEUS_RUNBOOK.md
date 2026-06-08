# Running the full paper experiments on PSU Coeus

Step-by-step runbook for reproducing **every** experiment in *Privacy in
Publish Subscribe Systems* (§7) on Portland State University's
[Coeus HPC cluster](https://sites.google.com/pdx.edu/research-computing).
Coeus uses **SLURM**, and we submit the suite as a **job array** (the method
Coeus recommends for "the same calculation on different data"), with a
dependent aggregation job.

**Maximize parallelism with the fine `dataset-clamp-exp` granularity** (the
recommended mode on a real cluster — see Step 2). It splits the work into
**~108 shards** (6 datasets × 2 clamp modes × {grid, sweep, A, B, C, D, F, G, H})
scheduled in **two SLURM phases**:

* **Phase 1** — one fast `grid` shard per `(dataset, clamp)` runs the §7.5
  grid search and writes `grid_canonical.json` (12 tasks, run in parallel).
* **Phase 2** — one shard per experiment (96 tasks, `--dependency=afterok` on
  phase 1). The F/G/H shards consume their `(dataset, clamp)` grid optimum via
  `--use-grid-config`; the sweep/A/B/C/D shards are grid-independent.

This is what makes it *maximally* parallel: the heavy `energy` dataset spreads
its 9 experiments across 9 nodes instead of stranding everything on one
straggler shard. The two-level parallelism is **shards across nodes** ×
**`--workers` cores within a shard**.

> The coarser default (`dataset-clamp`, 12 shards, each a self-contained
> `--grid-first --experiment full`) is simpler and needs no phases, but caps
> you at 12 nodes and is bounded by the slowest single shard (energy). Use it
> only for small runs or a busy queue.

Either way **all of §7 is covered** — main sweep, intro/Figure 1, tuning,
extras (incl. the §7.10 K_ext induced-latency sweep), and single-axis
A/B/C/D + F (ablation §7.8), G (overhead §7.9), H (average-case §7.11) — with
the grid search fixing the parameters for the rest.

---

## 0. One-time prerequisites (login node)

```bash
ssh <odin>@coeus.rc.pdx.edu          # your PSU/ODIN account
git clone <this-repo-url> ~/PubSubPrivacy
cd ~/PubSubPrivacy
```

### 0a. A modern Python (>= 3.10)

The code uses `int | None` / `dict[str, ...]` syntax → **Python ≥ 3.10**.
Coeus' stock `Python/gcc/3.7.5` is too old, so create a venv from a newer
Python (via a module if available, otherwise miniconda in your home dir):

```bash
# Option 1: a newer Python module, if one is listed
module avail Python                  # look for a 3.10+ build
module load Python/<3.10+ build>
python3 -m venv ~/PubSubPrivacy/.venv

# Option 2 (robust): miniconda in $HOME
#   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
#   bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
#   ~/miniconda3/bin/conda create -y -n pubsub python=3.11
#   then PUBSUB_ENV_SETUP='source ~/miniconda3/bin/activate pubsub'

source ~/PubSubPrivacy/.venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt      # numpy, pandas, matplotlib, paho-mqtt, ...
python -c "import numpy, pandas, matplotlib; print('deps OK')"
```

### 0b. Stage the datasets

Place the six dataset files under `~/PubSubPrivacy/data/` exactly as the main
[README "Datasets" table](../README.md#datasets) lists (the dir/CSV names
matter). On a shared filesystem (`/home` on Coeus is shared across nodes) every
node sees them automatically. Verify:

```bash
ls data/   # 2024_12_20/  Manufacturing_dataset.csv  Multi-Circuit...csv
           # Pune_SmartCity_Test_Dataset.csv  smart_mobility_dataset.csv
           # Wearable IoT Health Dataset.csv
```

### 0c. Make `submit_coeus.sh` aware of your env

Set `PUBSUB_ENV_SETUP` to whatever activates your Python on a compute node.
The array/aggregation jobs `eval` it before running:

```bash
export PUBSUB_ENV_SETUP='source ~/PubSubPrivacy/.venv/bin/activate'
# (or 'source ~/miniconda3/bin/activate pubsub')
```

---

## 1. Dry run — inspect the plan, submit nothing

```bash
cd ~/PubSubPrivacy
# Maximally-parallel fine mode (recommended): ~108 shards, two-phase.
DRY_RUN=1 SHARD_BY=dataset-clamp-exp cluster/submit_coeus.sh
```

You should see `scheduling : TWO-PHASE (phase1 grid=12 -> phase2 experiments=96,
afterok)` and the 108 shard commands (grid shards have `--grid-search`; F/G/H
shards carry `--use-grid-config`). Confirm it looks right.

---

## 2. Submit the full suite (maximally parallel)

```bash
# ~108 shards (dataset x clamp x experiment), 8 cores each, two SLURM phases.
SHARD_BY=dataset-clamp-exp CPUS=8 PARTITION=medium TIME=1-00:00:00 \
  THROTTLE=32 \
  PUBSUB_ENV_SETUP='source ~/PubSubPrivacy/.venv/bin/activate' \
  cluster/submit_coeus.sh
```

What it does:
1. `gen_jobs.sh` writes `results_cluster/joblist.txt` (108 lines) and
   `submit_coeus.sh` splits it into a grid joblist (12) and an experiment
   joblist (96).
2. submits **PHASE 1** job array `1-12` — the §7.5 grid shards.
3. submits **PHASE 2** job array `1-96` with `--dependency=afterok` on phase 1
   — the sweep/A/B/C/D/F/G/H shards (F/G/H read their grid optimum).
4. submits a **dependent** aggregation job (`afterok` on phase 2,
   `coeus_aggregate.sbatch`) merging every shard into `results_cluster/combined/`.

It prints all three job ids. `THROTTLE=32` caps concurrent tasks at 32 (raise it
to use more nodes; drop it to be gentler on the queue). Each shard still uses
`--cpus-per-task=8` cores as `run_experiment.py --workers 8`, so peak core usage
≈ `min(32, 96) × 8`.

### Simpler coarse mode (12 self-contained shards, no phases)

```bash
# Each shard = --grid-first --experiment full (grid then full in one process).
CPUS=16 PARTITION=medium TIME=2-00:00:00 \
  PUBSUB_ENV_SETUP='source ~/PubSubPrivacy/.venv/bin/activate' \
  cluster/submit_coeus.sh
```

Use this when the queue is busy or for a quick run; it caps at 12 nodes and is
bounded by the energy shard.

### Render figures at the end of aggregation

```bash
PLOTS=1 SHARD_BY=dataset-clamp-exp CPUS=8 cluster/submit_coeus.sh
```

---

## 3. Monitor

```bash
squeue -u $USER                       # queued / running array tasks
sacct -j <ARRAY_JOB_ID> --format=JobID,State,Elapsed,MaxRSS   # per-task status
tail -f pubsubdp_<ARRAY_JOB_ID>_*.out # live shard logs (one per array task)
```

If a task fails, its `.err` file has the traceback; re-run just that index with
`sbatch --array=<idx> ... coeus_array.sbatch` after fixing.

---

## 4. Results

```
results_cluster/
  joblist.txt                          # exact shard commands submitted
  joblist.grid.txt / joblist.rest.txt  # phase-1 / phase-2 split (fine mode)
  shards/<dataset>__<clamp>__<exp>/    # each shard's run_experiment.py tree
      <dataset>__<clamp>__grid/grid_canonical.json   # Sec. 7.5 optimum (phase 1)
      <dataset>__<clamp>__sweep/<dataset>/<clamp>/sweep/...
      <dataset>__<clamp>__F/cross_dataset/<clamp>/experiments/F_ablation/...
      ... (one shard dir per experiment; coarse mode uses <dataset>__<clamp>/)
  combined/                            # aggregation job: one CSV per artifact,
                                       #   concatenated across all shards
      cross_dataset/<clamp>/experiments/F_ablation/experiment_F_ablation.csv  (all datasets)
      ...
```

Per-dataset data lives in each shard dir (and as `dataset`-keyed rows in the
combined CSVs); the aggregate lives in `combined/`.

### Render figures (if you didn't pass `PLOTS=1`)

The aggregation job can render them, or do it on the login node:

```bash
source ~/PubSubPrivacy/.venv/bin/activate
python generate_plots.py --output-dir results_cluster/combined
# -> per-dataset + aggregate PNGs, including:
#    .../F_ablation/experiment_F_ablation_<dataset>.png + _all.png
#    .../G_overhead/experiment_G_overhead_<dataset>.png + _all.png
#    .../H_average_case/experiment_H_average_case_all.png
```

---

## 5. Quick smoke test first (recommended)

Before committing a multi-day run, validate the whole path on capped data:

```bash
EXTRA_ARGS="--quick --max-rows 5000 --max-energy-timestamps 3000 --max-traffic-rows 30000" \
  CPUS=8 TIME=02:00:00 \
  PUBSUB_ENV_SETUP='source ~/PubSubPrivacy/.venv/bin/activate' \
  cluster/submit_coeus.sh
```

---

## Notes

* **No GNU parallel needed on Coeus** — the SLURM array indexes the same
  joblist directly. (GNU parallel is the local / multi-node-SSH path; see
  `run_cluster.sh`.)
* **Walltime**: the `energy` dataset is the largest (~36k windows/sensor). In
  the fine mode its work is split across 9 experiment shards, so a 1-day `TIME`
  per shard is ample; in coarse mode one shard does all of energy, so give it
  2 days (`medium` caps at 7).
* **Reproducibility**: every DP run reseeds (`run_experiment.py --seed`), so
  array tasks are order-independent and results are deterministic per seed.
* **Live-broker Experiment E** is intentionally NOT part of the array (it needs
  a running MQTT broker). Run it separately on an `interactive` allocation if
  you want the live throughput/latency numbers (`--experiment E`).
