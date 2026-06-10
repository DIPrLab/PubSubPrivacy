# Running the full paper experiments on PSU Coeus

Step-by-step runbook for reproducing **every** experiment in *Privacy in
Publish Subscribe Systems* (§7) on Portland State University's
[Coeus HPC cluster](https://sites.google.com/pdx.edu/research-computing).
Coeus uses **SLURM**, and we submit the suite as a **job array** (the method
Coeus recommends for "the same calculation on different data"), with a
dependent aggregation job.

> **Clamp mode:** the cluster runs the **static (Option A) clamp only** by
> default for now. To also run the Option B DP-released clamp, prepend
> `CLAMP_MODES="static dp_released"` to any command below (doubles the shards).

**Maximize parallelism with the fine `dataset-clamp-exp` granularity** (the
recommended mode on a real cluster — see Step 2). It splits the work into
**~1074 shards** (6 datasets × {grid×144 per-(ε×trial×ρ), sweep×24 per-(strategy×
sensor-group), intro, tuning, extras, A, B, C, D, F, G, H, L}, static clamp)
scheduled in **two SLURM phases**:

* **Phase 1** — the §7.5 grid search, **split one shard per (dataset × ε ×
  trial × ρ)** (864 tasks) so the grid runs its **6 noise-seed trials AND its 6
  rho_tau candidates as separate tasks** across nodes. Each shard writes a
  `grid_trial_eps<ε>_rho<ρ>_t<trial>.json` full-grid fragment into the shared
  `<ds>__<clamp>__grid` dir; the downstream consumers **average MAE across the
  trial fragments and pick each strategy's global optimum across all ρ
  fragments** automatically (`_merge_grid_trial_fragments`), so the best ρ is
  selected alongside P_min/P_max/Δt/K_ext.
* **Phase 2** — the experiment shards (210 tasks, `--dependency=afterok` on
  phase 1). The per-level **sweep is split into 8 strategies × 3 sensor-groups =
  24 shards/dataset** (`--strategies <s> --sensor-shard i/3`) so the heavy
  `energy` sweep fans across nodes one sensor-group per task (its ~8 h long pole
  drops to ~⅓).  **Every phase-2 shard carries `--use-grid-config`** and
  consumes its dataset's merged §7.6 grid optimum: only **epsilon and w stay
  free**, while P_min, P_max, Δt, K_ext and ρ are taken from the optimum for
  each (dataset, strategy, epsilon) — an off-grid epsilon (Exp C) snaps to the
  nearest grid epsilon.  Exceptions: the **intro** figure deliberately sweeps
  every P_min per dataset; **tuning** is itself the greedy/brute tuner; **A**
  (greedy-vs-brute) and **D** (plugin validation) drive their own walk-up
  scenarios.  Every experiment tests subscriptions at **each topic level** and
  logs `subscription_level`/`scope` (plus the resolved `P`, `P_max`, `rho_split`)
  to its CSV.

This is what makes it *maximally* parallel: the heavy `energy` dataset spreads
its ~35 experiment shards (incl. 24 sweep shards = 8 strategies × 3
sensor-groups) across dozens of nodes instead of stranding everything on one straggler shard. The two-level
parallelism is **shards across nodes** × **`--workers` cores within a shard**.

> The coarser default (`dataset-clamp`, 6 shards with the static clamp, each a
> self-contained `--grid-first --experiment full`) is simpler and needs no
> phases, but caps you at 6 nodes and is bounded by the slowest single shard
> (energy). Use it only for small runs or a busy queue.

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
# Maximally-parallel fine mode (recommended): ~1074 shards (static clamp), two-phase.
DRY_RUN=1 SHARD_BY=dataset-clamp-exp cluster/submit_coeus.sh
```

You should see `scheduling : TWO-PHASE (phase1 grid=864 -> phase2 experiments=210,
afterok)` and the 1074 shard commands (grid shards invoke `-m experiments.grid_search`
with `--grid-eps`/`--grid-trial`;
F/G/H/L shards carry `--use-grid-config`; the sweep appears as 24 shards/dataset
(8 strategies × 3 sensor-groups)). Confirm it looks right.

---

## 2. Submit the full suite (maximally parallel)

```bash
# ~1074 shards (dataset x experiment, static clamp), 10 cores each (2 shards/
# 20-core node), two SLURM phases.  --trials 6 repeats every config over 6 noise
# seeds (per-trial rows + mean/std aggregate per dataset); --no-log-messages
# avoids the per-release message buffer that OOMs the energy sweep.
# Aggregate stream element (Definition: Aggregate Stream Element): for the
# population-aware strategies the broker releases a noisy COUNT and a noisy SUM,
# post-processed into the mean gamma_tau = S~_tau / max(n~_tau, 1).  The per-step
# budget eps_tau = eps/w is split by rho_tau: the count n~_tau gets rho*eps_tau
# (Laplace scale 1/(rho*eps_tau)) and the sum S~_tau gets (1-rho)*eps_tau (scale
# R/((1-rho)*eps_tau)), so the element spends exactly eps_tau inside the one
# w-event budget.  The §7.5 grid SWEEPS rho over GRID_RHO (default
# 0.1 0.2 0.4 0.5 0.6 0.8) as its own shard dimension and the best rho per
# (dataset,strategy,eps) is written into grid_canonical.json and consumed by the
# downstream F/G/H/L experiments (just like P_min/P_max/dt/k_ext).  The default
# release split (when not grid-driven) is rho=0.2, the definition's error-
# minimizing sqrt(R)/(5 sqrt(R)).  Override the release split directly with
# EXTRA_ARGS="... --rho-split <v>" (0 = exact count / Kellaris baselines), or pin
# the grid to one rho with GRID_RHO=0.2 (skips the rho sweep / 6x smaller grid).
# NOTE: --epsilon-count is DEPRECATED and ignored (superseded by --rho-split).
SHARD_BY=dataset-clamp-exp CPUS=10 PARTITION=medium TIME=2-00:00:00 \
  THROTTLE=80 \
  EXTRA_ARGS="--trials 6 --no-log-messages" \
  PUBSUB_ENV_SETUP='source ~/PubSubPrivacy/.venv/bin/activate' \
  cluster/submit_coeus.sh
```

What it does:
1. `gen_jobs.sh` writes `results_cluster/joblist.txt` (~1074 lines) and
   `submit_coeus.sh` splits it into a grid joblist (864 = 6 datasets × 4 ε ×
   6 trials × 6 rho) and an experiment joblist (210).
2. submits **PHASE 1** job array `1-864` — the §7.5 grid shards (per dataset ×
   ε × trial × ρ); the per-(eps,trial,ρ) full-grid fragments are min-merged so
   each strategy's optimum is chosen across all ρ.
3. submits **PHASE 2** job array `1-210` with `--dependency=afterok` on phase 1
   — the sweep(×24 = 8 strategies × 3 sensor-groups)/intro/tuning/extras/A/B/C/D/F/G/H/L shards
   (F/G/H/L read their grid optimum, including the best ρ).
4. submits a **dependent** aggregation job (`afterok` on phase 2,
   `coeus_aggregate.sbatch`) merging every shard into `results_cluster/combined/`.

It prints all three job ids. `THROTTLE=80` caps concurrent tasks at 80; with
`CPUS=10` that fits the 210 experiment shards across the 86×20-core nodes. Raise/drop `THROTTLE` to use more nodes / be
gentler on the queue. `--trials 6` makes each shard do 6× the DP runs, but they
fan out over the shard's 10 workers, so per-shard wall time barely moves.

### Simpler coarse mode (6 self-contained shards, no phases)

```bash
# Each shard = --grid-first --experiment full (grid then full in one process).
CPUS=16 PARTITION=medium TIME=2-00:00:00 \
  EXTRA_ARGS="--trials 6 --no-log-messages" \
  PUBSUB_ENV_SETUP='source ~/PubSubPrivacy/.venv/bin/activate' \
  cluster/submit_coeus.sh
```

Use this when the queue is busy or for a quick run; it caps at 6 nodes (static
clamp) and is bounded by the energy shard.

### Render figures at the end of aggregation

```bash
PLOTS=1 SHARD_BY=dataset-clamp-exp CPUS=10 THROTTLE=80 \
  EXTRA_ARGS="--trials 6 --no-log-messages" cluster/submit_coeus.sh
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
  shards/<dataset>__<clamp>__<exp>/    # each shard's output tree
      <dataset>__<clamp>__grid/grid_canonical.json   # Sec. 7.5 optimum (phase 1)
      <dataset>__<clamp>__sweep_<strategy>/<dataset>/<clamp>/sweep/...  (8/dataset)
      <dataset>__<clamp>__F/cross_dataset/<clamp>/experiments/F_ablation/...
      ... (one shard dir per experiment; coarse mode uses <dataset>__<clamp>/)
  combined/                            # aggregation job: one CSV per artifact,
                                       #   concatenated across all shards
      cross_dataset/<clamp>/experiments/F_ablation/experiment_F_ablation.csv  (all datasets)
      ...
```

Per-dataset data lives in each shard dir (and as `dataset`-keyed rows in the
combined CSVs); the aggregate lives in `combined/`.

### Copy results back to your machine

The aggregation job also builds **`results_cluster/paper_bundle/`** — the
curated key individual (per-dataset) + combined (cross-dataset) results the
paper references, in one small section-organized directory (per-release message
dumps excluded, see its `INDEX.md`). For most purposes you only need to scp
that one directory. Run these **from your local machine**, not the login node:

```bash
# RECOMMENDED: just the curated paper results (small, single download):
scp -r jacoboli@login1.coeus.rc.pdx.edu:~/PubSubPrivacy/results_cluster/paper_bundle \
       ./paper_bundle

# The full merged tree (one CSV per artifact, all datasets — larger):
scp -r jacoboli@login1.coeus.rc.pdx.edu:~/PubSubPrivacy/results_cluster/combined \
       ./results_cluster_combined

# Everything (combined + paper_bundle + every per-shard tree + joblists):
scp -r jacoboli@login1.coeus.rc.pdx.edu:~/PubSubPrivacy/results_cluster \
       ./results_cluster

# rsync alternative — resumable, skips unchanged files, good for multi-GB pulls:
rsync -avz --progress \
  jacoboli@login1.coeus.rc.pdx.edu:~/PubSubPrivacy/results_cluster/paper_bundle/ \
  ./paper_bundle/

# Also grab the SLURM logs if you need to debug a failed shard:
scp 'jacoboli@login1.coeus.rc.pdx.edu:~/PubSubPrivacy/pubsubdp_*.{out,err}' ./logs/
```

Then render figures locally from the full tree if you didn't pass `PLOTS=1`:
`python generate_plots.py --output-dir ./results_cluster_combined`.

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
  the fine mode its work is split across ~35 experiment shards — and each sweep
  shard now covers ONE sensor (`--sensor-shard i/3`) and each grid shard ONE
  (ε, trial, ρ) — so a 1-day `TIME` per shard is ample (the per-ρ grid shards
  are 1/6 the work of the old per-(ε,trial) shards); in coarse mode one shard
  does all of energy, so give it 2 days (`medium` caps at 7).
* **Array size**: phase 1 is `1-864` and phase 2 `1-210`; both are under SLURM's
  default `MaxArraySize` (1001+ on Coeus), so no array-size tuning is needed for
  the default static-clamp run. NOTE: running BOTH clamp modes doubles the grid
  to 1728 (> 1001) — submit the clamp modes as separate invocations, or pin
  `GRID_RHO=0.2` to shrink the grid 6x. `THROTTLE` still caps how many run at once.
* **Reproducibility**: every DP run reseeds (`run_experiment.py --seed`), so
  array tasks are order-independent and results are deterministic per seed.
* **Live-broker Experiment E** is intentionally NOT part of the array (it needs
  a running MQTT broker). Run it separately on an `interactive` allocation if
  you want the live throughput/latency numbers (`--experiment E`).
