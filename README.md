# PubSubPrivacy

**Clamped w-event Differential Privacy with P-allocation for MQTT Publish-Subscribe Systems**

Reference implementation of the paper *Privacy in Publish Subscribe Systems*
(Olinger and Pappachan, 2026). Extends the w-event DP framework of Kellaris
et al. (2014) to the pub/sub setting with:

  * **Clamped aggregate streams** (Section 3.3): each release is the mean of
    the clamped payloads of the publishers active during the interval.
  * **Per-element sensitivity `Δf = R / n_τ`** (Section 4.3) — keyed on the
    observed multiplicity, not on the scheduling threshold P, so the ε
    guarantee never depends on a private payload value.
  * **P-allocation** (Section 5.3), a population-aware budget-allocation
    strategy family joining Kellaris et al.'s Uniform, Sample, BD, and BA.
  * **n-weighted P-allocation** (Section 5.4) — distributes ε across a window
    in proportion to `n_τ`, shrinking the Laplace scale quadratically in
    publisher density.
  * **Clamp-compatible topic-hierarchy walk** (Algorithm 1), operating on a
    normative MQTT topic tree per dataset (e.g. `energy/building01/{circuit}/{metric}`,
    `factory/line1/{machine}/{sensor}`, `health/clinic01/{device}/{metric}`).
  * **Dynamic timestamp-interval extension** up to `T_max = K_ext · Δt`
    (Section 5.6).
  * **Two-stage hyperparameter tuning** (Section 5.7): Algorithm 2 greedy
    hill-climb over the publisher threshold `P`, seeded at `P_0 = ⌈1/α⌉`,
    with Δt adapting online via the extension mechanism and the inner
    strategy `(A, θ)` selected by subscription requirements. Paired with a
    naive full-brute-force enumeration of every `P ∈ [1, max n_τ]` per
    strategy so the **empirical gap** between greedy and optimal is
    measurable directly.

The experimental pipeline runs end-to-end on **six real-world public
datasets** — no synthetic data anywhere.

## Architecture

```
Publishers (IoT sensors)
       |
       v
  [raw MQTT topics]
       |
       v
  PrivacyPlugin (broker middleware)
    - Clamps each payload to [a_p, b_p]                          (Def. 3.2)
    - Buffers messages per logical timestamp (interval Δt)        (Def. 3.1)
    - Adaptive extension up to K_ext · Δt if n_τ < P              (Sec. 5.6)
    - Clamp-compatible hierarchy walk when n_τ still < P          (Alg. 1)
    - Computes clamped mean aggregate e_τ                         (Def. 3.3)
    - Releases  ê_τ = e_τ + Lap(R / (n_τ · ε_τ))                  (Thm. 5.1)
    - Enforces sliding window budget  Σ ε_j ≤ ε                   (Eq. 4)
       |
       v
  [protected MQTT topics]
       |
       v
  Subscribers (receive only (ê_τ, t_start_τ) pairs)
```

## Components

| File | Description |
|------|-------------|
| `dp_engine.py` | Core DP engine. Laplace mechanism, sliding-window budget, all eight budget strategies exercised by the paper (Uniform, Sample, BD with the Algorithm 4 forward buffer, BA, P-gated Uniform/Sample/BA, n-weighted), utility metrics (MAE/NMAE, windowed + global KL), attribution advantage. |
| `plugin.py` | MQTT privacy plugin. Per-publisher clamping (Def. 3.2), Algorithm 1 hierarchy walk with clamp-compatibility, dynamic interval extension (Section 5.6), delivery of only `(ê_τ, t_start_τ)` pairs per Def. 3.4. |
| `data_streams.py` | Dataset ingestion, stream construction, `DATASETS` registry (six real-world datasets), and the Definition 3.2 clamp options (static / DP-released). |
| `run_experiment.py` | Single entry-point for the full pipeline: per-dataset parameter sweep, strategy comparison, paper's four intro figures, n-weighted spotlight, collusion experiment, K_ext sweep, Section 5.7 multi-strategy hyperparameter tuning, Experiments A/B/C/D (Section 6.5–6.8), and cross-dataset aggregates. |
| `config.yaml` | Configuration mirroring the paper's DP / scheduling parameter taxonomy. |

## Parameter taxonomy (paper §3.1)

### DP parameters (enter the privacy analysis)

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| `epsilon` | ε | Global privacy budget per sliding window |
| `window_size` | w | Number of logical timestamps per sliding window |
| `payload_bound` | R | Global clamp range = sup_p(b_p − a_p) |

Per-element sensitivity for the default mean aggregation is
`Δf = R / n_τ`; Laplace scale is `λ_τ = R / (n_τ · ε_τ)`. Under uniform
allocation `ε_τ = ε/w`, this becomes `λ_τ = R·w / (n_τ · ε)`.

### Scheduling hyperparameters (do NOT enter the DP calculation)

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| `min_publishers` | P | Publisher threshold for the P-allocation release gate |
| `timestamp_interval` | Δt | Base wall-clock interval per logical timestamp |
| `k_ext` | K_ext | Max number of Δt extensions when `n_τ < P`; `T_max = K_ext·Δt` |
| `strategy` | A | Inner budget-allocation strategy |
| `ba_threshold` | θ | BA similarity threshold (fraction of R) |

P bounds the publisher-identity attribution advantage at 1/P. Tuning P
does **not** change the ε guarantee; it trades subscriber accuracy +
identity protection against latency and scope coarsening.

## Budget-allocation strategies

| Name | Description |
|------|-------------|
| `uniform` | `ε_τ = ε/w` for every release (Kellaris et al., Section 5.2). |
| `sample` | Spend full ε once per window; repeat previous output otherwise. |
| `budget_distribution` (BD) | Kellaris et al. Algorithm 4 lines 11–18: on a value-similar skip, forward the base share `(ε/w)/(w−1)` into each of the next `w−1` slots via a `fwd` buffer; on a release spend `ε/w + fwd[τ]` and reset that slot. |
| `budget_absorption` (BA) | Kellaris et al. Algorithm 4 lines 20–28: skip value-similar timestamps, absorb the unused share into a pot, drain the pot on the next value-different release. |
| `p_gated_uniform` / `p_gated_sample` / `p_gated_ba` | P-allocation wrappers (Section 5.3) — only spend budget when `n_τ ≥ P`, otherwise repeat the last output (deferred). |
| `n_weighted` | **Paper Section 5.4 contribution**: `ε_τ = ε · n_τ / Σ_j n_j`; dense-pool timestamps get more budget so Laplace scale shrinks quadratically in `n_τ`. |

## Quick start

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the real-data experiment

```bash
# FULL paper pipeline on every dataset in both clamp modes (Option A + B).
# Produces every artifact referenced in the paper: the main sweep,
# intro figures (Section 1.3), Section 5.7 greedy-vs-brute tuning, the
# Option A vs B clamp comparison, the n-weighted / collusion / K_ext
# auxiliary probes, and Experiments A/B/C/D (Section 6.5–6.8).
python run_experiment.py --experiment full --clamp-mode both \
                        --alpha 0.25 --eps-clip 0.1 \
                        --output-dir results

# Same pipeline, sequential over every single-axis experiment explicitly.
# `--experiment full` already covers this; use ABCD when you want to skip
# the main sweep and only run the four single-axis experiments.
python run_experiment.py --experiment ABCD --clamp-mode both \
                        --alpha 0.25 --eps-clip 0.1 \
                        --output-dir results

# Quick demo on every dataset (reduced grid) in both clamp modes.
python run_experiment.py --quick --clamp-mode both

# Single clamp mode only.
python run_experiment.py --clamp-mode static        # Option A only
python run_experiment.py --clamp-mode dp_released   # Option B only

# One dataset at a time.
python run_experiment.py --dataset energy
python run_experiment.py --dataset traffic
python run_experiment.py --dataset wearable
python run_experiment.py --dataset pune
python run_experiment.py --dataset mobility
python run_experiment.py --dataset manufacturing

# Just the Section 5.7 hyperparameter tuning (Algorithm 2 greedy + naive
# brute-force) across every strategy, every dataset, every clamp mode.
python run_experiment.py --tune-only --clamp-mode both

# Subset of strategies.
python run_experiment.py --quick --strategies uniform p_gated_ba n_weighted

# Truncate the larger datasets for faster iteration.
python run_experiment.py --quick --max-energy-timestamps 3000 \
                        --max-traffic-rows 50000 --max-rows 5000

# Tuning-sensitivity dials (paper Algorithm 2 hyperparameters).
python run_experiment.py --alpha 0.10          # P_0 = ceil(1/0.10) = 10
python run_experiment.py --I-max 40            # More hill-climb iterations
python run_experiment.py --eps-clip 0.05       # Tighter Option B calibration

# Single-axis experiments (hold all hyperparameters fixed except one).
python run_experiment.py --experiment A        # greedy vs naive P-tuning
python run_experiment.py --experiment B        # vary w, fix (P, eps, strategy)
python run_experiment.py --experiment C        # vary eps, fix (P, w, strategy)
python run_experiment.py --experiment D        # plugin end-to-end vs offline
python run_experiment.py --experiment ABC      # all three single-axis, no main sweep
python run_experiment.py --experiment ABCD     # ABC plus plugin end-to-end
```

Every `--experiment full` invocation produces, for each dataset × clamp mode:

* the full parameter sweep (all strategies × P × ε × w),
* the four paper intro figures (two extremes + U-shape + KL bar),
* Section 5.7 hyperparameter tuning across **all** strategies,
* the n-weighted spotlight, collusion experiment, and K_ext sweep,
* Experiments A/B/C/D (Section 6.5–6.8, single-axis + plugin end-to-end),

plus cross-dataset aggregates: combined sweep CSV, combined tuning CSV, the
best `(strategy, P, Δt)` per dataset, an aggregated Figure 1 averaging KL
across every evaluated dataset, and combined CSVs per single-axis experiment
(A/B/C/D) under `cross_dataset/<clamp_mode>/experiments/`.

## Output layout

Every run of `run_experiment.py` produces the same standardized tree under
`--output-dir` (default `results/`). The `<clamp_mode>` sub-folder is
`static` (Option A, Section 3.2) or `dp_released` (Option B); running
`--clamp-mode both` emits both side-by-side for comparison.

```
results/
  <dataset>/
    <clamp_mode>/
      <dataset>_<clamp_mode>_clamps.csv    # per-publisher [a_p, b_p] actually used
      sweep/
        sweep_results.csv                   # every (strategy, P, eps, w, sensor) row
        <dataset>_mae_vs_epsilon.png
        <dataset>_mae_vs_P.png
        <dataset>_strategy_comparison.png
        <dataset>_timeseries.png
        <dataset>_budget_utilization.png
        <dataset>_kl_vs_epsilon.png
        <dataset>_kl_heatmap.png
        <dataset>_kl_windowed.png
        <dataset>_release_rate_vs_P.png
      intro/
        <dataset>_figure_extreme1_global.csv/.png
        <dataset>_figure_extreme2_per_publisher.csv/.png
        <dataset>_figure_extremes_vs_ours.csv/.png
        <dataset>_figure_u_shaped_P_vs_KL.csv/.png
        <dataset>_figure1_kl_vs_P.csv/.png
      tuning/
        <dataset>_<sensor>_tuning_greedy.csv        # Algorithm 2 trajectory (every probe)
        <dataset>_<sensor>_tuning_brute_force.csv   # every P in [1, p_max]
        <dataset>_<sensor>_tuning_strategy_summary.csv  # greedy vs brute-force
        <dataset>_<sensor>_tuning.png               # loss-vs-P curve with greedy overlay
      extras/
        <dataset>_n_weighted_spotlight.csv/.png
        <dataset>_collusion.csv/.png
        <dataset>_<sensor>_k_ext_sweep.csv
        <dataset>_<sensor>_k_ext.png
  cross_dataset/
    combined_sweep_results.csv              # every dataset x clamp_mode x config
    combined_tuning_greedy.csv              # every greedy probe across datasets
    combined_tuning_brute_force.csv         # every brute-force evaluation
    combined_tuning_gap_summary.csv         # greedy vs brute per (dataset, strategy, clamp_mode)
    tuning_best_per_dataset.csv             # headline best per (dataset, clamp_mode)
    tuning_greedy_vs_brute_gap.png          # bar chart of |greedy − brute| loss gap
    figure1_all_datasets.csv                # per-dataset U-shape points (all clamp modes)
    <clamp_mode>/
      figure1_all_datasets.csv              # per-clamp-mode slice
      figure1_avg_across_datasets.csv       # KL averaged across datasets per P
      figure1_all_datasets.png              # cross-dataset Figure 1 for this clamp mode
    cross_dataset_<clamp_mode>.png          # NMAE / KL vs eps per clamp mode
```

Every experiment writes its raw data to CSV alongside its PNG so downstream
analysis (papers, notebooks, regression tests) works off the artifacts
without re-running the sweep.

## Datasets

All evaluation is on real-world public CSVs. Place each file (or directory)
at the path listed below, then run `python run_experiment.py`.

Our scenario-to-dataset mapping (which smart-IoT domain each CSV covers) is
aligned with PSMark's public device/test-suite catalogue —
[DAMSlabUMBC/PSMark `devices/README.txt`](https://github.com/DAMSlabUMBC/PSMark/blob/main/psmark/configs/builtin-test-suites/devices/README.txt) —
four of our six datasets (`wearable`, `pune`, `mobility`, `manufacturing`)
are taken directly from the Kaggle sources listed there.

| Key | Dataset | Path under `data/` | Source |
|-----|---------|--------------------|--------|
| `energy` | MCEC-Thai multi-circuit electric consumption | `Multi-Circuit Electric Consumption Data for Application of Energy Disaggregation.csv` | [IEEE 10949848](https://ieeexplore.ieee.org/document/10949848) · [Mendeley](https://data.mendeley.com/datasets/nmnk58bgtb/1) |
| `traffic` | Colorado Springs multi-sensor intersection (radar + lidar) | `2024_12_20/` | [IEEE 11134445](https://ieeexplore.ieee.org/document/11134445) · [NLR 287](https://data.nlr.gov/submissions/287) |
| `wearable` | Wearable IoT Healthcare (10 devices, multi-metric) | `Wearable IoT Health Dataset.csv` | [Kaggle dcsavinod](https://www.kaggle.com/datasets/dcsavinod/iot-in-healthcare-and-well-being) (PSMark scenario 5) |
| `pune` | Pune Smart City air-quality sensor network (10 stations) | `Pune_SmartCity_Test_Dataset.csv` | [Kaggle akshman](https://www.kaggle.com/datasets/akshman/pune-smartcity-test-dataset) (PSMark scenario 4) |
| `mobility` | Smart Mobility Traffic telemetry | `smart_mobility_dataset.csv` | [Kaggle ziya07](https://www.kaggle.com/datasets/ziya07/smart-mobility-traffic-dataset) (PSMark scenario 3) |
| `manufacturing` | Smart Manufacturing Process Data | `Manufacturing_dataset.csv` | [Kaggle programmer3](https://www.kaggle.com/datasets/programmer3/smart-manufacturing-process-data) (PSMark scenario 6) |

### How publishers and subscribers are constructed per dataset

Each dataset is coerced into the paper's pub/sub model: publishers are the
finite set of message-emitting entities, each one emitting a single clamped
value per logical timestamp τ; a subscription binds to a topic and receives
`(ê_τ, t_start_τ)` pairs after the broker applies clamped-aggregate DP. Two
datasets (`mobility`, `manufacturing`) ship as time-indexed tables without
an explicit per-row publisher column; for those we synthesize publisher IDs
from structure that is genuinely present in the data (lat/lon quantile bins
for mobility; the raw minute-index within a Δt bucket for manufacturing).
Those two are **acknowledged limitations** — the synthesis matches the
paper's Δt discretization but is less faithful than the four datasets with
real per-row publisher IDs. The four explicitly-modeled datasets (`energy`,
`traffic`, `wearable`, `pune`) preserve the original source semantics.

| Dataset | Publisher ID (message sender) | Logical timestamp (Δt) | Realism |
|---------|-------------------------------|------------------------|---------|
| **energy** | CT circuit column `CT1…CT17` (12 physical circuits in a Bangkok residential unit, each metered separately). | 5-minute wall-clock window over the shared `Time` column. | ★★★★★ every publisher is a real circuit. |
| **traffic** | Each radar/lidar file (`EVO_RADAR_1…4`, `OS1_LIDAR_1…2`) = one physical sensor at an intersection. | 10-second window; inside a window every detection from a sensor is collapsed into one mean value per metric. | ★★★★★ every publisher is a real sensor; detections per-sensor are averaged. |
| **wearable** | `Device_ID` (ten wearables, `Device_1…Device_10`). | Per-device sequence index via `cumcount`. The raw `Timestamp` column is cyclic (12 distinct `mm:ss.f` labels) and non-monotonic, so we replace it with a logical τ ordering. | ★★★★☆ publishers are real, Δt is synthetic because the source dataset's wall-clock labels aren't usable. |
| **pune** | `NAME` column (ten fixed sensor stations like `BopadiSquare_65`, `Karve Statue Square_5`, ...). | 60-minute wall-clock bin over `LASTUPDATEDATETIME`. | ★★★★★ station = publisher, genuinely longitudinal. |
| **mobility** | No per-row publisher in the raw CSV. We quantile-bin `(Latitude, Longitude)` into a 4×4 grid and use the cell label (`cell_0_0` … `cell_3_3`) as `publisher_id`. Each cell acts as a virtual intersection sensor. | 30-minute bin over the parsed `Timestamp`. | ★★☆☆☆ publishers are synthetic groupings; different 4×4 binning would produce different results. |
| **manufacturing** | No per-row publisher. Each 10-minute bucket contains 10 minute-rows, which we label `sub_0 … sub_9` as virtual sub-publishers (matching the paper's "Δt holds n_τ records" discretization). | 10-minute bin over the parsed `Timestamp`. | ★★☆☆☆ sub-pubs are synthetic (single-machine dataset); evaluates the Δt discretization more than true per-publisher structure. |

### Normative MQTT topic hierarchies + realistic subscriber filters

Each dataset is placed under a paper-aligned topic tree (§3, §5.1). The
runner emits two CSVs per run so the MQTT topology is fully transparent:

* `<dataset>/<clamp_mode>/<dataset>_topics.csv` — every
  `(publisher_id, sensor) → topic` mapping.
* `<dataset>/<clamp_mode>/<dataset>_subscriber_filters.csv` — the wildcard
  filters the realistic subscriber scenarios listed below would bind to.

| Dataset | Topic hierarchy | Realistic subscriber role | Subscriber filters it binds to | Paper scenario |
|---------|-----------------|---------------------------|--------------------------------|----------------|
| **energy** | `energy/building01/{circuit}/{metric}` (e.g. `energy/building01/CT5/power_kw`) | Utility analytics / demand-response provider | `energy/building01/#`, `energy/building01/+/power_kw`, `energy/building01/CT1/#` | Smart-meter energy monitoring (§1.1.2) |
| **traffic** | `traffic/intersection01/{radar\|lidar}/{sensor_id}/{metric}` (e.g. `traffic/intersection01/radar/EVO_RADAR_1/speed`) | City traffic-management / navigation service | `traffic/intersection01/#`, `traffic/intersection01/+/+/speed`, `traffic/intersection01/radar/#` | Traffic monitoring (§1.1.1) |
| **wearable** | `health/clinic01/{device_id}/{metric}` (e.g. `health/clinic01/Device_5/heart_rate`) | Clinical RPM dashboard / cardiac alerting | `health/clinic01/#`, `health/clinic01/+/heart_rate`, `health/clinic01/Device_5/#` | Wearable health telemetry (§1.1.3) |
| **pune** | `air_quality/pune/{station_slug}/{pollutant}` (e.g. `air_quality/pune/Hadapsar_Gadital_01/pm10`) | Municipal air-quality dashboard / pollution alerts | `air_quality/pune/#`, `air_quality/pune/+/pm10`, `air_quality/pune/Hadapsar_Gadital_01/#` | Environmental monitoring (extension of §1.1) |
| **mobility** | `mobility/nyc/{NE\|NW\|SE\|SW}/{grid_cell}/{metric}` (e.g. `mobility/nyc/NE/cell_3_2/traffic_speed`) | Congestion / ride-sharing optimizer | `mobility/nyc/#`, `mobility/nyc/NE/#`, `mobility/nyc/+/+/traffic_speed` | Smart-city mobility (§1.1.1 extension) |
| **manufacturing** | `factory/line1/{machine_id}/{sensor}` (e.g. `factory/line1/machine01/vibration`) | Plant-floor dashboard / predictive-maintenance | `factory/line1/#`, `factory/line1/+/vibration`, `factory/line1/machine01/#` | Factory IoT (§1.3, §5.1 motivating topology) |

The topic hierarchies match the paper's motivating conventions (`factory/line/machine/sensor`,
`health/device_id/metric`, `traffic/segment_id/metric`) so Algorithm 1's
clamp-compatible scope walk operates over a realistic tree structure. Even
though the offline evaluator runs one subscription at a time (the leaf),
all the wildcard filters above are semantically valid against the emitted
topic set — the `<dataset>_topics.csv` manifest makes the subscriber-side
attachment point explicit for downstream regression / attack tests.

### Clamp ranges actually used (Option A from Definition 3.2)

The static `[a_p, b_p]` we declare per sensor type. These are chosen from
datasheet / physiological / regulatory ranges, **not** from the data. Every
payload is clamped into this interval before any aggregation, so the global
clamp range `R = sup_p (b_p − a_p)` used in the DP sensitivity calculation
is a data-independent constant. They are wired in `DATASETS[<name>]["static_clamps"]`.

| Dataset | Sensor | `[a_p, b_p]` | Fallback `M` (Option B) |
|---------|--------|--------------|-------------------------|
| energy | `power_kw` | `[0, 50]` kW | 100 |
| energy | `voltage` | `[180, 260]` V | 300 |
| energy | `current` | `[0, 100]` A | 200 |
| traffic | `speed` | `[0, 50]` m/s | 100 |
| traffic | `object_count` | `[0, 50]` | 100 |
| wearable | `heart_rate` | `[30, 220]` bpm | 250 |
| wearable | `steps` | `[0, 5000]` per 5 min | 10000 |
| wearable | `temperature` | `[15, 45]` C | 60 |
| wearable | `calories_burned` | `[0, 50]` kcal / 5 min | 100 |
| pune | `pm10` | `[0, 1000]` µg/m³ | 2000 |
| pune | `pm2` | `[0, 500]` µg/m³ | 1000 |
| pune | `humidity` | `[0, 100]` % | 100 |
| pune | `sound` | `[20, 140]` dB | 200 |
| pune | `ozone` | `[0, 500]` ppb | 1000 |
| mobility | `traffic_speed` | `[0, 120]` km/h | 200 |
| mobility | `vehicle_count` | `[0, 500]` | 1000 |
| mobility | `road_occupancy` | `[0, 100]` % | 100 |
| mobility | `emission` | `[0, 800]` g/km | 2000 |
| manufacturing | `temperature` | `[0, 200]` C | 500 |
| manufacturing | `machine_speed` | `[0, 5000]` RPM | 10000 |
| manufacturing | `quality` | `[0, 10]` | 20 |
| manufacturing | `vibration` | `[0, 1]` mm/s | 5 |
| manufacturing | `energy` | `[0, 10]` kWh / 10 min | 50 |

### Clamp modes: Option A vs Option B (paper §3.2)

`run_experiment.py --clamp-mode {static,dp_released,both}` runs two distinct
experiments per dataset, side-by-side:

**Option A — `static` (operator-declared).** Uses the `[a_p, b_p]` above,
fixed before the mechanism runs. No privacy budget is charged for clamping.
Clamp is loose by design (domain-knowledge ceilings), so `R` is larger than
the data actually spans, which inflates Laplace noise.

**Option B — `dp_released` (DP-released min/max, §3.2).** For each publisher
the broker reserves a separate budget `ε_clip` (default `0.1`, flag
`--eps-clip`) and releases noisy per-publisher min/max via
`â_p = min x̃_{p,τ} + Lap(M/ε_clip)` and `b̂_p = max x̃_{p,τ} + Lap(M/ε_clip)`,
clipped to the public fallback `[-M, M]` from the table above. The noisy
`(â_p, b̂_p)` are treated as public constants for subsequent clamping (DP
post-processing). Total DP cost composes to `ε + ε_clip`. This tightens `R`
to the empirical support (plus calibration noise), shrinking `λ_τ` —
worthwhile when the tightening outweighs the `ε_clip` outlay.

Both modes emit their own sub-tree under `results/<dataset>/<clamp_mode>/`
with its own sweep, intro figures, and tuning, so operators can read off the
empirical accuracy penalty of Option A's loose clamp versus Option B's
calibration cost.

### Section 5.7 tuning: Algorithm 2 greedy + naive brute-force enumeration

Per the paper's updated Tuning architecture, `tune_hyperparameters` now runs
two-stage scope-first tuning with `P` as the primary knob (Δt adapts online
via Section 5.6; `(A, θ)` is selected by subscription requirements rather
than jointly optimized):

* **Greedy hill-climb (Algorithm 2).** For each strategy, seed
  `P_0 = ⌈1/α⌉` (default `α = 0.25` → `P_0 = 4`; flag `--alpha`). Probe
  neighbors `P_0 − 1` and `P_0 + 1`; step to the better if it lowers the
  scalarized loss `L = NMAE + 0.2 · (1 − release_rate)`; stop at a local
  optimum or after `--I-max` iterations (default 20).
* **Naive brute-force enumeration.** For the same strategy, evaluate every
  integer `P ∈ [1, p_max]` where `p_max = max n_τ` observed on the trace.

Per-strategy outputs land in `tuning/` with both the full greedy trajectory
and the complete brute-force curve, so the **empirical gap**
`greedy_loss − brute_loss` is directly measurable — this is what the paper
claims is small on the 1-D `P`-loss surface. Cross-dataset gap bars land in
`cross_dataset/tuning_greedy_vs_brute_gap.png`.

### Three single-axis experiments (`--experiment {A|B|C|ABC|full}`)

On top of the main sweep, the runner exposes three focused experiments that
hold every hyperparameter fixed except one. Each produces its own CSV + PNG
per clamp mode, in addition to the main sweep:

**Experiment A — Greedy vs naive P-tuning (`--experiment A`).** For each
`(dataset, sensor, strategy)` and several `(ε, w)` points, runs Algorithm 2
greedy hill-climb and brute-force enumeration over every integer
`P ∈ [1, max n_τ]`. Records `greedy_P`, `brute_P`, `gap_loss`, and
`speedup = brute_evals / greedy_evals`. The cross-dataset plot shows mean
speedup per dataset and a `greedy_loss` vs `brute_loss` scatter so the
empirical optimality gap is visible at a glance.

**Experiment B — Vary w (`--experiment B`).** Fixes `(P, ε, strategy)` at
five canonical combinations (e.g. `P=2, ε=1.0, uniform`) and sweeps
`w ∈ {4, 6, 8, 10, 12, 16}`. Each row logs the observed NMAE and KL, the
payload bound `R`, the theoretical `λ = R·w / (n·ε)`, and the predicted
NMAE `w / (n·ε)` so the paper's linear-in-w hypothesis (Section 6.6) is
mechanically verifiable from the CSV alone.

**Experiment C — Vary ε (`--experiment C`).** Fixes `(P, w, strategy)` at
five combinations and sweeps `ε ∈ {0.1, 0.25, 0.5, 1, 2, 4, 8}`. Same
logging as B (R, predicted λ, predicted NMAE) to mechanically verify the
inverse-in-ε hypothesis (Section 6.7). Plotted on a log-x axis.

**Experiment D — Plugin path end-to-end (`--experiment D`).** Exercises
the broker-side `PrivacyPlugin` on a stubbed MQTT client (no external
broker needed), driving every captured trace through `_on_message` and
`_flush_and_release`. For each dataset it runs two scenarios:

* `pooled` — all publishers on a single shared leaf topic; no scope walk,
  one DP stream. Verifies the plugin's full pipeline (clamp → buffer →
  `StreamState` → Laplace) matches `run_dp_on_stream` byte-for-byte under
  the same seed (`max_abs_diff_vs_offline` column).
* `hierarchy` — each publisher on its own leaf topic under the dataset's
  normative MQTT tree; `plugin_P > 1` forces Algorithm 1 walk-ups every
  release. Verifies the walk fires, walked releases carry the true
  wall-clock `t_start` (Def. 3.1), and no release slips past the gate
  (`p_gate_violations` column must be 0).

Outputs a per-release CSV, a per-run summary (release/walk-up/deferral
counts, `t_start` monotonicity + spacing stats, P-gate audit), and a 3-
panel comparison PNG. Landing under
`results/cross_dataset/<clamp_mode>/experiments/D_plugin_path/`.

All four write per-dataset CSVs and a cross-dataset aggregate under
`results/cross_dataset/<clamp_mode>/experiments/{A,B,C,D}_*/`.
`--experiment ABCD` runs all four without the main sweep; `--experiment full`
(default) runs the main sweep plus A/B/C/D.

## Adding a new dataset

Append a loader + `build_streams` + `build_per_publisher` triplet and a
`DATASETS` entry (with `static_clamps` and `fallback_M`) inside
`data_streams.py`. Every downstream step — sweep, intro figures, tuning,
cross-dataset aggregates — picks it up automatically.

## Message format

**Publisher (raw):**

```json
{"publisher_id": "sensor_42", "value": 72.45, "ts": 1711720800.0}
```

**Subscriber (protected) — paper Definition 3.4:**

```json
{"t_start": 1711720800.0, "value": 73.12}
```

`t_start` is the **wall-clock start** of the construction interval (seconds
since Unix epoch), as required by Def. 3.1 — not the logical index `τ`.
Adaptive `K_ext` extensions move the closing boundary of an interval but
never shift its `t_start`, per Def. 3.1. Only `(ê_τ, t_start_τ)` is
delivered; the pool size `n_τ`, release scope, logical `τ`, and all other
runtime quantities remain broker-internal.

## Reproducing the paper end-to-end

The canonical command that runs every artifact the paper references, on
every dataset, in both clamp modes, sequentially, and saves every CSV /
PNG under `results/`:

```bash
python run_experiment.py --experiment full --clamp-mode both \
                        --alpha 0.25 --eps-clip 0.1 \
                        --output-dir results
```

For a faster smoke-test of the pipeline (reduced grids, capped row counts,
separate output tree) use:

```bash
python run_experiment.py --experiment full --quick --clamp-mode both \
                        --max-energy-timestamps 3000 \
                        --max-traffic-rows 30000 --max-rows 5000 \
                        --output-dir results_test
```

```python run_experiment.py --experiment E --dataset wearable --output-dir results_e```
