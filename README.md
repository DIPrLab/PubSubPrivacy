# PubSubPrivacy

**PAWS: Differentially Private Aggregation for IoT Publish/Subscribe Streams**

Reference implementation of the paper *PAWS: Differentially Private
Aggregation for IoT Publish/Subscribe Streams* (under submission). PAWS
realizes w-event differential privacy (Kellaris et al., 2014) for aggregate
(mean) queries on MQTT publish/subscribe streams, as a broker-side plugin. Its
components, with the paper section each implements:

  * **Aggregate stream construction** (paper §3–§4): messages are bucketed
    into logical timestamps of width Δt, each payload is clamped to its
    publisher's declared range, and each release is a noisy centered sum
    (Laplace scale `R/((1−ρ)ε_τ)`, sensitivity `R` under Definition 3.1)
    divided by a noisy publisher count (scale `1/(ρ ε_τ)`), released only when
    at least `P_min` publishers contribute (the **P_min gate**).
  * **Timestamp widening** (§4.1, Eqs. 2–3): the interval is sized up front
    from the discounted arrival rate of past noisy counts, aiming at
    `1.5·P_min` publishers, capped at `K_ext·Δt`. Pure post-processing.
  * **Subscription rewriting** (§4.2, Definition 4.1, Algorithm 1): when the
    subscribed scope still has fewer than `P_min` publishers, the release
    walks up the topic tree to the nearest ancestor whose **range-compatible**
    pool meets the threshold.
  * **Budget weighting** (§5, Eqs. 8–9, Algorithm 2; `n_weighted` in the
    code): the window budget is split in inverse proportion to each
    timestamp's noisy publisher count, `ε_τ = ε·(1/n_τ)/Σ_j(1/n_j)`, clipped
    to the remaining window budget.
  * **Evaluation** (§6): utility vs. ε, w and subscription depth, downstream
    forecasting, induced wait and broker overhead, and the incremental
    ablation, on six real IoT datasets (see "Section 6 experiments" below).

### Code ↔ paper terminology

| Paper | Code / CLI |
|---|---|
| Budget weighting (publisher weighted) | `n_weighted` (`--our-strategy n_weighted`) |
| P_min gate | `P_min`, the `p_gated_*` strategies |
| PAWS Sample | `p_gated_sample` |
| Timestamp widening | adaptive interval, `--wait-policy predictive`, `K_ext` |
| Subscription rewriting | hierarchy walk / walk-up, `walk=True` |
| Global w-event DP baseline | `wevent` |
| Per-publisher (local) DP baseline | `ldp` |
| Count split ρ | `rho_split`, `--rho-split` |

**Implementation details beyond the paper text** (all preserve the
guarantee): the walk-up adds a noisy count of only the publishers new at each
level to the cached count below, each charged one count share as in
Algorithm 1 (`--no-count-cache` counts the whole ancestor pool afresh at each
level instead); the released mean divides by the average of the last `w` noisy
counts at the same scope, clipped to `[1, N]` (post-processing); and an
optional `P_max` cap truncates very large pools.

Experiments A–L, `intro`, `tuning` and `sweep` (and their section references
further down this README) come from earlier drafts of the paper and are kept
for reference; the current paper's evaluation is the "Section 6 experiments"
set (U, M, K, J, F and the grid).

The experimental pipeline runs end-to-end on **six real-world public
datasets** — no synthetic data anywhere.

## Mechanism and evaluation updates (SIGMOD 2027 revision)

| Change | Where | What it does |
|--------|-------|--------------|
| **Cached counts on the hierarchy walk** | `dp_engine.walk_population_counts`, `plugin._walk` | Walking up adds a noisy count of only the *delta* (publishers new at that level) to the cached count below: `ñ_k = ñ_{k-1} + (n_k − n_{k-1}) + Lap(1/(ρε_τ))`. Each delta is a fresh DP count, so every probed level is charged one count share, as in Algorithm 1 (m probed levels cost m·ε_count); caching avoids re-counting the publishers already counted below. Levels whose (public) publisher set equals the one below are skipped with no draw and no charge. `--no-count-cache` counts the whole ancestor pool afresh at each level instead. |
| **Timestamp widening / predictive Δt (paper §4.1)** | `dp_engine.PredictiveInterval`, `plugin._predictive_wait` | Each interval's width is fixed up front from the discounted arrival rate of past noisy counts, `λ̂_τ = max(ñ_{τ−1},1)/(Δt_{τ−1}·w) + (1−1/w)·λ̂_{τ−1}` (seeded so `Δt_1 = Δt`), as `Δt_τ = clip(1.5·P_min/λ̂_τ, Δt, K_ext·Δt)` (paper Eqs. 2–3). Sizing is post-processing and costs no budget. An interval whose closing count misses P_min gets at most one corrective Δt extension when the window can pay for the recount, then walks, then defers. Alternatives: `--wait-policy early_stop` (Sparse Vector rule that closes as soon as P_min is met, one count share per interval), `reactive` (the original whole-Δt extension, one count per check), `none`. |
| **Grid search for ρ** | `experiments/grid_search.py` | The §7.5 grid sweeps ρ ∈ {0.05 … 0.8} jointly with P_min, P_max, Δt, K_ext over the full pipeline at one subscription per depth, selecting on the coverage-penalized NMAE; `*_rho_profile.csv` gives the best objective at each ρ. |
| **No repeated outputs in the ablation** | `experiments/ablation.py`, `PrivacyConfig.repeat_on_defer` | Everywhere else a deferred timestamp repeats the previous output, as in the paper. The ablation turns this off (`repeat_on_defer=False`): a deferral delivers nothing, so a module that rarely releases is not credited with stale repeats. |
| **Utility metric** | `dp_engine.release_utility` | NMAE of the delivered values against the subscribed scope's true aggregate, plus `coverage` and `nmae_penalized = coverage·NMAE + (1 − coverage)` (an undelivered timestamp counts as the paper's no-signal error 1.0). Timestamps with no active publisher have no true aggregate and are not scored. In the ablation the delivered values are fresh releases only. |
| **Unbiased release estimator** | `dp_engine.StreamState.release`, `PrivacyConfig.center` / `count_smoothing` | The released mean was a noisy sum divided by a fresh noisy count; with the small ρ the grid chose, large count draws collapsed outputs toward 0 (10/12 run: correlation with the truth ≈ 0, outputs 13–25% of the true mean). Now: (1) the sum is centered on the public clamp midpoint c, privatizing Σ(x−c) with sensitivity R (a substitution neighbour can move one centered payload across the whole range) and releasing c + S̃/ñ, so count error pulls toward the middle of the range, not 0; (2) ñ is the average of the last w noisy counts at the same scope (post-processing, no budget; variance ÷ w); (3) ñ is clipped to [1, N], N = publishers registered under the scope (public). The budget-weighting allocator (ours, Eq. 8) weights by this same smoothed count rather than one raw noisy draw. |
| **Skill metric** | `dp_engine.release_utility` | `skill = 1 − MAE / MAE(best constant)`, the constant being the median truth, plus `corr` (released vs true). A stream no more informative than a fixed number scores ≤ 0. `skill_prior` compares against the data-independent clamp midpoint instead (does the stream carry the level?), while `skill` asks whether it tracks the changes. The grid selects on `rel_mae_penalized = 1 − skill_penalized` (undelivered = the constant's error). |
| **Full-pipeline simulator** | `experiments/pipeline.py` | Replays the raw per-publisher trace through the broker's decisions (stopping rule, cached walk, P_max, the engine's budget accounting) per subscription, with LDP / w-event / no-privacy baselines on the same intervals. The older offline experiments decided interval extension on the exact count and never walked the tree. |

Accounting fixes made along the way: the plugin's interval-extension counts are
now charged to the stream's window (they were drawn but dropped), and BD/BA's
dissimilarity share can no longer push a window over ε when count shares have
used most of it (`test_pipeline.py` covers both).

### Section 6 experiments

| Exp. | Module | Measures |
|------|--------|----------|
| U | `experiments/utility.py` | NMAE / coverage of PAWS vs **local DP** and **w-event DP** on all 6 datasets, at every subscription depth, over ε ∈ {0.5, 1, 2, 4, 6, 8} (`--sweep epsilon`) and w ∈ {4, 8, 12, 16, 20, 24} (`--sweep window`), scored by skill against the best constant predictor as well as NMAE. |
| M | `experiments/ml_utility.py` | Downstream ML on the received stream: Ridge next-value forecast (NMAE) and logistic threshold classification (balanced accuracy, F1), vs no privacy / LDP / w-event. |
| J | `experiments/latency.py` | Broker latency: per-message ingest and per-release work (in-process) for forward-only, w-event and PAWS variants; `--live` adds publish→delivery latency through a real MQTT broker. |
| K | `experiments/induced_wait.py` | Extra wait induced by the adaptive interval (Δt units and seconds; mean / p95 / max) for the predictive, early-stop and reactive policies over K_ext, with the utility it buys and `walk_kl`, the semantic shift of the walk-ups that waiting avoids: D_KL(P ‖ Q) between the subscribed scope's readings (P) and the walked-up pool's readings (Q) over every fresh release that walked, binned on the public clamp interval (NaN when nothing walked). Every Section 7 simulation reports it. |
| F | `experiments/ablation.py` | Three cumulative components — M1 P-gate, M2 + predictive adaptive interval, M3 + cached walk-up — at every depth, on fresh releases only (no repeated outputs). |

```bash
python -m experiments.grid_search  --dataset all              # writes grid_canonical.json first
python -m experiments.utility      --dataset all --sweep both --trials 6
python -m experiments.ml_utility   --dataset all --trials 6
python -m experiments.latency      --dataset all --trials 3 --live
python -m experiments.induced_wait --dataset all --trials 6
python -m experiments.ablation     --dataset all --trials 6
PAPER_RESULTS=results python make_paper_figures.py           # exp_U_*, exp_M_*, exp_J/K/F figures
```

`--our-strategy` picks the budget strategy used for PAWS (default
`n_weighted`, the paper's n-weighted P-allocation; e.g. `p_gated_uniform`), `--eps-values` / `--w-values` override
the sweeps. On the cluster these run as phase-2 shards `U M K J F`
(`cluster/gen_jobs.sh`) consuming the phase-1 grid.

### Section 6 tables (10/13 cluster run)

Reference tables for the paper's Section 6, from the 10/13 run
(`results_combined_1013/`, grid trials in `paper_bundle_1013/shards/*__static__grid/`):
static clamp, 6 noise seeds per configuration.

> The runs behind them calibrated
> PAWS's centered sum to sensitivity R/2; under the paper's neighbouring
> relation (Definition 3.1) the sensitivity is R, which the code now uses.
> PAWS's sum noise in these tables is therefore half what the stated ε
> requires, and they also predate the 1.5·P_min interval target (Eq. 3) and
> the per-level charging of cached walk-up counts (Algorithm 1).
> Re-run the grid and the Section 6 experiments before relying on them.

**Datasets.** All six are real, public IoT feeds. |P| is the number of
distinct publishers and Subs. the number of wildcard subscriber bindings a
realistic domain subscriber registers. For energy, traffic, wearable and pune
the publisher is a genuine per-row entity; for mobility and manufacturing it is
synthesized from structure in the source (a 4×4 latitude/longitude grid of
virtual sensors, and minute-of-interval sub-publishers of a single machine).

| Dataset | Source | \|P\| | Subs. | Sensors evaluated |
|---|---|---:|---:|---|
| energy | IEEE Data Descriptions, 2025 | 12 | 3 | power (kW), voltage, current |
| traffic | IEEE Data Descriptions, 2025 | 4–8 | 4 | speed, object count |
| wearable | Kaggle, 2024 | 10 | 5 | heart rate, steps, temperature, calories |
| pune | Kaggle, 2023 | 10 | 3 | PM10, PM2.5, humidity, sound, ozone |
| mobility | Kaggle, 2024 | 16 | 4 | speed, vehicle count, occupancy, emission |
| manufacturing | Kaggle, 2024 | 10 | 5 | temperature, speed, vibration, quality |

**Constructed topic hierarchies.** h is the number of levels; braces are the
per-dataset entities populating each level.

| Dataset | h | Topic hierarchy |
|---|---:|---|
| energy | 4 | `energy/building01/{CT}/{metric}` |
| traffic | 5 | `traffic/intersection01/{kind}/{id}/{metric}` |
| wearable | 4 | `health/{hospital}/{device}/{metric}` |
| pune | 4 | `air_quality/pune/{station}/{pollutant}` |
| mobility | 5 | `mobility/nyc/{quadrant}/{cell}/{metric}` |
| manufacturing | 5 | `factory/line1/{station}/{machine}/{sensor}` |

**Grid-optimal parameters for ours (n-weighted).** Recomputed from the
per-trial grid JSONs (the grid's selection score, coverage-penalized MAE
relative to the best constant predictor, averaged over seeds). P_min\* = 6 and P_max\* = ∞
throughout, except energy (P_max\* = 8 at ε ≥ 4). Δt is the multiplier on each
dataset's base slot.

| Dataset | ε=1: Δt | K_ext | ρ | ε=4: Δt | K_ext | ρ | ε=8: Δt | K_ext | ρ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| energy | 4 | 0 | .05 | 4 | 0 | .05 | 4 | 4 | .20 |
| traffic | 1 | 0 | .05 | 1 | 0 | .05 | 1 | 0 | .20 |
| wearable | 1 | 0 | .05 | 4 | 0 | .05 | 1 | 0 | .40 |
| pune | 4 | 0 | .05 | 4 | 0 | .05 | 4 | 4 | .50 |
| mobility | 1 | 0 | .05 | 1 | 0 | .05 | 4 | 0 | .30 |
| manufacturing | 4 | 0 | .05 | 4 | 0 | .05 | 2 | 0 | .30 |

**Skill against the range midpoint at the type-root subscription** (> 0: more
informative than the declared clamp range). PG-S = PAWS P-gated Sample; best
per row and budget in bold.

| Dataset | ε=4: ours | PG-S | w-event | LDP | ε=8: ours | PG-S | w-event | LDP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| energy | −0.03 | **0.72** | 0.03 | −2.26 | 0.56 | **0.81** | 0.52 | −0.63 |
| traffic | −0.03 | **0.42** | −0.79 | −3.74 | 0.29 | **0.32** | 0.10 | −1.37 |
| wearable | 0.40 | **0.62** | −1.22 | −2.53 | 0.50 | **0.58** | −0.00 | −0.96 |
| pune | 0.39 | **0.74** | −0.25 | −2.75 | 0.39 | **0.78** | 0.38 | −0.87 |
| mobility | 0.08 | **0.35** | −1.90 | −5.95 | **0.45** | 0.43 | 0.32 | −1.58 |
| manufacturing | 0.52 | **0.88** | 0.15 | −1.96 | 0.67 | **0.91** | 0.58 | −0.46 |

**NMAE of every budget strategy** at ε = 4, w = 8, each at its own grid
optimum. Best per column in bold.

*Type-root subscription:*

| Group | Strategy | energy | traffic | wearable | pune | mobility | manuf. |
|---|---|---:|---:|---:|---:|---:|---:|
| PAWS | n-weighted (ours) | .201 | .196 | .142 | .140 | .131 | .116 |
| PAWS | P-gated Uniform | .198 | .195 | .142 | .139 | .132 | .114 |
| PAWS | P-gated Sample | .072 | .105 | .087 | .064 | .090 | .031 |
| PAWS | P-gated BA | .199 | .231 | .144 | .237 | .134 | .111 |
| PAWS | P-gated BD | .183 | .274 | .185 | .288 | .146 | .149 |
| Kellaris | Uniform | .261 | .343 | .466 | .258 | .166 | .199 |
| Kellaris | Sample | **.036** | **.103** | **.079** | **.046** | **.057** | **.027** |
| Kellaris | BD | .349 | .498 | .677 | .362 | .228 | .280 |
| Kellaris | BA | .273 | .426 | .568 | .282 | .172 | .209 |
| Baseline | w-event DP | .261 | .343 | .524 | .258 | .405 | .203 |
| Baseline | local DP | .783 | .908 | .834 | .781 | .970 | .705 |

*All subscription levels:*

| Group | Strategy | energy | traffic | wearable | pune | mobility | manuf. |
|---|---|---:|---:|---:|---:|---:|---:|
| PAWS | n-weighted (ours) | .827 | .669 | .696 | .757 | .711 | .694 |
| PAWS | P-gated Uniform | .811 | .658 | .682 | .725 | .692 | .684 |
| PAWS | P-gated Sample | **.197** | **.152** | **.183** | **.173** | **.201** | **.088** |
| PAWS | P-gated BA | .842 | .758 | .922 | .648 | .740 | .727 |
| PAWS | P-gated BD | .945 | .848 | 1.095 | .750 | .862 | .871 |
| Kellaris | Uniform | 1.871 | 1.629 | 1.815 | 1.834 | 1.797 | 1.675 |
| Kellaris | Sample | .236 | .229 | .259 | .235 | .286 | .216 |
| Kellaris | BD | 2.823 | 2.477 | 2.592 | 2.773 | 2.545 | 2.518 |
| Kellaris | BA | 2.249 | 2.045 | 2.284 | 2.215 | 2.305 | 2.043 |
| Baseline | w-event DP | 1.871 | 1.629 | 1.775 | 1.834 | 1.876 | 1.704 |
| Baseline | local DP | 1.912 | 1.768 | 1.874 | 1.882 | 1.931 | 1.824 |

**Walk-up semantic shift** (Experiment K re-run with `walk_kl`, ours, ε = 1,
w = 8, 6 seeds, all subscription levels; `results_K_walkkl/`). Walk KL is
D_KL(P ‖ Q) in nats between the subscribed scope's readings and the walked-up
pool's readings, over releases that walked; walk rate is the share of releases
that walked. The re-run reproduces the 10/13 K numbers exactly otherwise.

| Dataset | No waiting: KL | walk rate | Predictive K_ext=4: KL | walk rate | Reactive K_ext=4: KL | walk rate |
|---|---:|---:|---:|---:|---:|---:|
| energy | 0.079 | 0.20 | 0.149 | 0.08 | 0.198 | 0.008 |
| traffic | 0.105 | 0.25 | 0.107 | 0.10 | 0.136 | 0.014 |
| wearable | 0.066 | 0.27 | 0.124 | 0.12 | 0.200 | 0.028 |
| pune | 0.208 | 0.20 | 0.208 | 0.08 | 0.297 | 0.008 |
| mobility | 0.046 | 0.28 | 0.068 | 0.11 | 0.175 | 0.013 |
| manufacturing | 0.041 | 0.26 | 0.090 | 0.10 | 0.192 | 0.019 |

**Broker cost per dataset** (Experiment J, in-process; median over sensors and
6 trials). PAWS runs ours (n-weighted); "no cache" recomputes per-level walk
counts and "no wait" disables the adaptive interval.

| Metric | Configuration | energy | traffic | wearable | pune | mobility | manuf. |
|---|---|---:|---:|---:|---:|---:|---:|
| Broker CPU per Δt tick (ms) | forward only | 0.07 | 0.06 | 0.07 | 0.08 | 0.05 | 0.08 |
| | w-event DP | 0.43 | 0.27 | 0.44 | 0.45 | 0.57 | 0.47 |
| | PAWS (ours) | 0.95 | 0.57 | 0.97 | 0.92 | 1.54 | 1.01 |
| | PAWS, no cache | 0.97 | 0.58 | 0.97 | 0.92 | 1.56 | 1.02 |
| | PAWS, no wait | 1.28 | 0.79 | 1.37 | 16.89 | 2.16 | 1.41 |
| Ingest per message (µs) | forward only | 8.6 | 9.4 | 8.3 | 9.7 | 9.3 | 8.2 |
| | PAWS (ours) | 8.8 | 9.3 | 8.5 | 9.8 | 10.1 | 8.4 |
| Interval close, p95 (ms) | w-event DP | 0.05 | 0.04 | 0.05 | 0.05 | 0.05 | 0.05 |
| | PAWS (ours) | 0.16 | 0.16 | 0.16 | 0.17 | 0.17 | 0.17 |

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
    - Buffers messages per logical timestamp (interval Δt)        (Def. 5.1)
    - Sizes each interval up front, clip(P_aim/λ̂, Δt, K_ext·Δt),
      from past noisy counts (free); ≤ 1 corrective extension    (Sec. 5.1.1)
    - Clamp-compatible hierarchy walk on cached delta counts
      when the DP count is still < P_min                          (Alg. 1)
    - Releases noisy sum / noisy count  S̃_τ / max(ñ_τ, 1)         (Def. 5.3)
    - Charges every count share inside the window  Σ ε_j ≤ ε      (Def. 2.3)
    - Repeats the previous output on a deferred timestamp
       |
       v
  [protected MQTT topics]
       |
       v
  Subscribers (receive only (ê_τ, t_start_τ) pairs)
```

## Components

The code is organized as a **shared engine library + thin per-experiment
modules + a CLI wrapper**, so each paper experiment is independently runnable
and cluster-shardable:

| File | Description |
|------|-------------|
| `dp_engine.py` | Core DP engine. Laplace mechanism, sliding-window budget invariant, all nine budget strategies (Uniform, Sample, BD with the Algorithm 4 forward buffer, BA with absorb/nullify, P-gated Uniform/Sample/BD/BA, n-weighted), the differentially-private publisher count (`ε_count`), the `P_max` sensitivity cap, utility metrics (MAE/NMAE, windowed + global KL), attribution advantage. The P-gate accepts a caller-supplied count (`release(…, n_gate=…)`) so the broker's Algorithm-1 decision is the single authority on release-vs-defer (no double-gate). |
| `plugin.py` | MQTT privacy plugin (broker middleware). Per-publisher clamping (Def. 3.2), Algorithm 1 clamp-compatible hierarchy walk, dynamic interval extension (§6.7), `ε_count` DP counts charged per probed level, `P_max` truncation, delivery of only `(ê_τ, t_start_τ)` pairs (Def. 3.4). Hands its DP gate count to `StreamState.release` so the engine doesn't re-gate on the exact count. |
| `data_streams.py` | Dataset ingestion, stream construction, the `DATASETS` registry (six real-world datasets) + their normative PerCom topic hierarchies, and the Definition 3.2 clamp options (static Option A / DP-released Option B, with a process-stable per-sensor seed). |
| `message_logger.py` | Per-release message serialization → append-friendly CSV rows. Built only when message logging is on; **all metrics are computed without it**, so `--no-log-messages` (the cluster default) avoids the per-release buffer that otherwise OOMs the largest dataset. |
| **`experiments/engine.py`** | **The shared core library.** The DP runners (`run_dp_on_stream`, `run_ldp_on_per_pub`), the `ProcessPoolExecutor` pool + worker-global stream caches, the topic-hierarchy / per-level subscription primitives (`_topic_level_groups`, `level_subscription_streams`), dataset/grid resolution (`_resolve_params`, grid-canonical I/O), the stubbed-MQTT plugin driver, plotting, trial aggregation, and the per-dataset / cross-dataset orchestration (`run_dataset`, `run_single_axis_experiments`, `_run_grid_search_block`). Every module imports it as `core`. |
| **`experiments/<name>.py`** | One module per paper experiment, each runnable as `python -m experiments.<name>`: `sweep`, `grid_search`, `intro` (Extreme 1 / 1.1 / 2, U-shape, Figure 1), `tuning`, `extras` (n-weighted spotlight, collusion, K_ext §7.10), `greedy_vs_brute` (A), `window` (B), `epsilon` (C), `plugin_path` (D), `ablation` (F), `overhead` (G), `average_case` (H), `subscription_levels` (L). `_common.py` provides the shared CLI/arg resolution. |
| **`experiments/live_broker.py`** | Experiment E: live MQTT broker end-to-end (auto-started Mosquitto by default, or embedded amqtt with `--broker-impl amqtt`, + real paho publishers/subscribers through the plugin). Kept out of the offline engine because it needs a running broker. |
| `run_experiment.py` | **Thin CLI wrapper** over `experiments/engine.py` — builds the argument parser and dispatches into the engine + experiment modules. Carries no functional code itself. |
| `generate_plots.py` | Post-hoc plot generator. Reads the CSVs (including the per-trial `*_aggregate.csv` files) and renders the full PNG catalogue. Run after the experiments, or pass `--generate-plots` for the inline workflow. |
| `cluster/` | Cluster launchers: `gen_jobs.sh` (emits one shard per `dataset×clamp×experiment`), `submit_slurm.sh` (SLURM two-phase array), `run_cluster.sh` (GNU parallel), `aggregate.py` (merge shards + build the `paper_bundle/`), and the SLURM `.sbatch` files. See `cluster/SLURM_RUNBOOK.md`. |
| `experiments/pipeline.py` | Offline full-pipeline simulator used by the Section 7 experiments (U/M/K/F and the grid): stopping rule, cached walk, P_max, budget accounting, and the LDP / w-event / no-privacy baselines. `experiments/_section7.py` is their shared driver. |
| `test_pipeline.py` | Cached-walk charging, predictive interval sizing, the stopping rule, repeat-on-defer (off in the ablation), the utility metric, and the window invariant through the simulator (every strategy × wait × walk × cache) and the plugin. |
| `test_engine.py` / `test_extremes.py` | DP correctness tests (w-event budget invariant for all strategies, single-sourced gate, BA cap-rollback) and the intro extremes (LDP scale, Extreme 1.1 per-level, Extreme 1/2 figures). |

## Parameter taxonomy (paper §3.2)

### DP parameters (enter the privacy analysis)

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| `epsilon` | ε | Global privacy budget per sliding window |
| `window_size` | w | Number of logical timestamps per sliding window |
| `payload_bound` | R | Global clamp range = sup_p(b_p − a_p) |
| `epsilon_count` | ε_count | Budget spent per step to release a **differentially private publisher count** `\|P_τ\|` (sensitivity 1) when gating / walking the topic hierarchy (§6.3 step 1, Alg. 1 step 3, Table 3). Composes additively with ε (like `ε_clip`). `0` = exact count (Kellaris baselines). CLI: `--epsilon-count`. |

Per-element sensitivity for the default mean aggregation is
`Δf = R / n_τ`; Laplace scale is `λ_τ = R / (n_τ · ε_τ)`. Under uniform
allocation `ε_τ = ε/w`, this becomes `λ_τ = R·w / (n_τ · ε)`. The publisher
count that drives the **release gate and the hierarchy walk** is released under
`ε_count` (`StreamState._dp_count` / `plugin._dp_count`); the noise
**calibration** still divides by the actual pooled `n_τ`, which is public under
the neighboring relation (Def. 5.1 holds `P_τ` fixed across DP neighbors).

### Scheduling hyperparameters (do NOT enter the DP calculation)

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| `min_publishers` | P_min | Publisher threshold for the P-allocation release gate |
| `max_publishers` | P_max | Caps the multiplicity folded into the mean so `Δf = R/n_τ` changes by a bounded amount across stream elements (§6.5); `n_τ > P_max` aggregates only the first P_max publishers. `None` = no cap. CLI: `--max-publishers`. |
| `timestamp_interval` | Δt | Base wall-clock interval per logical timestamp |
| `k_ext` | K_ext | Max number of Δt extensions when `n_τ < P_min`; `T_max = K_ext·Δt` |
| `strategy` | A | Inner budget-allocation strategy |
| `ba_threshold` | θ | BA similarity threshold (fraction of R) |

`P_min` bounds the publisher-identity attribution advantage at 1/P_min; the
`[P_min, P_max]` band bounds the per-element sensitivity *change*. Tuning P
does **not** change the ε guarantee; it trades subscriber accuracy +
identity protection against latency and scope coarsening. **`P_min = 1`
dissolves to local differential privacy** (per-publisher input perturbation,
`λ = R·w/ε`; paper §1 Extreme 2, §6.6) — `run_ldp_on_per_pub` makes this LDP
regime explicit and it is the LDP baseline in the §7.9 overhead comparison and
the rightmost (per-publisher) point of Figure 1.

## Budget-allocation strategies

| Name | Description |
|------|-------------|
| `uniform` | `ε_τ = ε/w` for every release (Kellaris et al., Section 5.2). |
| `sample` | Spend full ε once per window; repeat previous output otherwise. |
| `budget_distribution` (BD) | Kellaris et al. Algorithm 4 lines 11–18: on a value-similar skip, forward the base share `(ε/w)/(w−1)` into each of the next `w−1` slots via a `fwd` buffer; on a release spend `ε/w + fwd[τ]` and reset that slot. |
| `budget_absorption` (BA) | Kellaris et al. Algorithm 4 lines 20–28: skip value-similar timestamps, absorb the unused share into a pot, drain the pot on the next value-different release. |
| `p_gated_uniform` / `p_gated_sample` / `p_gated_ba` | P-allocation wrappers (Section 5.3) — only spend budget when `n_τ ≥ P`, otherwise repeat the last output (deferred). |
| `n_weighted` | **Paper Section 6.4 contribution**: `ε_τ = ε · (1/n_τ) / Σ_j(1/n_j)` — budget **inversely** proportional to pool size, which equalizes the released Laplace scale `λ_τ = R·Σ_j(1/n_j)/ε` across the window (independent of `n_τ`). Implicitly P-gated (uses the population count). |

## Quick start

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the real-data experiment (two-step workflow)

```bash
# 1.  FULL paper pipeline on every dataset in both clamp modes (Option A + B).
#     Writes every CSV the paper references: the main sweep, intro figures
#     (Section 1.3), Section 5.7 greedy-vs-brute tuning with intelligent
#     multi-start, the Option A vs B clamp comparison, the n-weighted /
#     collusion / K_ext auxiliary probes, and Experiments A/B/C/D
#     (Section 6.5–6.8).  Per-release messages (true aggregate + noisy
#     delivery + all metadata) land under each experiment's messages/ dir.
python run_experiment.py --experiment full --clamp-mode both \
                        --alpha 0.25 --eps-clip 0.1 \
                        --n-restarts 3 \
                        --output-dir results_3

# 2.  Render every PNG from the CSVs in results_3/.
python generate_plots.py --output-dir results_3

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
python run_experiment.py --n-restarts 5        # More intelligent restart starts
python run_experiment.py --n-restarts 1        # Paper's original deterministic seed only
python run_experiment.py --restart-rng-seed 42 # Reproducibility of quantile draws
python run_experiment.py --eps-clip 0.05       # Tighter Option B calibration

# Message / plot logging dials.
python run_experiment.py --no-log-messages     # Skip the messages/*.csv outputs
python run_experiment.py --generate-plots      # Render PNGs inline (legacy mode)

# Single-axis experiments (hold all hyperparameters fixed except one).
python run_experiment.py --experiment A        # greedy vs naive P-tuning
python run_experiment.py --experiment B        # vary w, fix (P, eps, strategy)
python run_experiment.py --experiment C        # vary eps, fix (P, w, strategy)
python run_experiment.py --experiment D        # plugin end-to-end vs offline
python run_experiment.py --experiment ABC      # all three single-axis, no main sweep
python run_experiment.py --experiment ABCD     # ABC plus plugin end-to-end
```

Every `--experiment full` invocation produces, for each dataset × clamp mode:

* the full parameter sweep (all strategies × P × ε × w), at every topic level,
* the paper intro figures — **Extreme 1** (global), **Extreme 1.1** (one stream
  per publisher type, with the per-level distortion it inflicts on finer
  subscriptions), **Extreme 2 / LDP** (per-publisher), the **U-shape**, the
  **KL bar**, and the **Figure 1** reproduction,
* Section 5.7 hyperparameter tuning across **all** strategies,
* the n-weighted spotlight, collusion experiment, and K_ext sweep,
* Experiments A/B/C/D (single-axis + plugin end-to-end) and F/G/H/L
  (ablation, overhead, average-case, per-level subscriptions),

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
results_3/
  <dataset>/
    <clamp_mode>/
      <dataset>_<clamp_mode>_clamps.csv    # per-publisher [a_p, b_p] actually used
      <dataset>_topics.csv                 # (publisher_id, sensor) -> MQTT topic
      <dataset>_subscriber_filters.csv     # wildcard filters realistic subscribers bind to
      messages/
        sweep_messages.csv                 # per-release log: true_aggregate,
                                           # noisy_value, n_tau, eps_tau,
                                           # lambda_tau, deferred, noise_sample,
                                           # delta_f, for every (config, tau).
      sweep/
        sweep_results.csv                   # every (strategy, P, eps, w, sensor) row
        # PNGs land here only when --generate-plots is set on run_experiment.py
        # or after running: python generate_plots.py --output-dir results_3
      intro/
        <dataset>_figure_extreme1_global.csv  (+ .png from generate_plots.py)
        <dataset>_figure_extreme2_per_publisher.csv  (+ .png)
        <dataset>_figure_extremes_vs_ours.csv  (+ .png)
        <dataset>_figure_u_shaped_P_vs_KL.csv  (+ .png)
        <dataset>_figure1_kl_vs_P.csv  (+ .png)
      tuning/
        <dataset>_<sensor>_tuning_greedy.csv        # Algorithm 2 trajectory
                                                    # (every probe across every
                                                    # restart, with restart_idx
                                                    # column)
        <dataset>_<sensor>_tuning_brute_force.csv   # every P in [1, p_max]
        <dataset>_<sensor>_tuning_strategy_summary.csv  # greedy vs brute-force,
                                                    # plus greedy_n_restarts +
                                                    # greedy_restart_seeds_P
        <dataset>_<sensor>_tuning.png               # loss-vs-P curve (generate_plots)
      extras/
        <dataset>_n_weighted_spotlight.csv  (+ .png)
        <dataset>_collusion.csv  (+ .png)
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
      experiments/
        A_greedy_vs_brute/
          experiment_A_greedy_vs_brute.csv   # greedy vs brute for every (ds,strat,eps,w)
          experiment_A_speedup_and_gap.png
        B_vary_w/
          experiment_B_vary_w.csv            # NMAE/KL vs w for 5 fixed combos
          experiment_B_messages.csv          # per-release messages (NEW)
          experiment_B_vary_w.png
        C_vary_epsilon/
          experiment_C_vary_epsilon.csv      # NMAE/KL vs eps for 5 fixed combos
          experiment_C_messages.csv          # per-release messages (NEW)
          experiment_C_vary_epsilon.png
        D_plugin_path/
          experiment_D_plugin_releases.csv   # already per-release (plugin log)
          experiment_D_plugin_summary.csv
          experiment_D_plugin_path.png
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
| **wearable** | `health/{hospital}/{device_id}/{metric}` (e.g. `health/hospital03/Device_5/heart_rate`) | Clinical RPM dashboard / cardiac alerting | `health/#`, `health/+/+/heart_rate`, `health/hospital01/#`, `health/+/Device_5/#` | Wearable health telemetry (§1.1.3); PSMark-HC 5-hospital deployment |
| **pune** | `air_quality/pune/{station_slug}/{pollutant}` (e.g. `air_quality/pune/Hadapsar_Gadital_01/pm10`) | Municipal air-quality dashboard / pollution alerts | `air_quality/pune/#`, `air_quality/pune/+/pm10`, `air_quality/pune/Hadapsar_Gadital_01/#` | Environmental monitoring (extension of §1.1) |
| **mobility** | `mobility/nyc/{NE\|NW\|SE\|SW}/{grid_cell}/{metric}` (e.g. `mobility/nyc/NE/cell_3_2/traffic_speed`) | Congestion / ride-sharing optimizer | `mobility/nyc/#`, `mobility/nyc/NE/#`, `mobility/nyc/+/+/traffic_speed` | Smart-city mobility (§1.1.1 extension) |
| **manufacturing** | `factory/line1/{station}/{machine_id}/{sensor}` (e.g. `factory/line1/processing/machine01/vibration`) | Plant-floor dashboard / predictive-maintenance | `factory/line1/#`, `factory/line1/+/+/vibration`, `factory/line1/processing/#`, `factory/line1/+/machine01/#` | Factory IoT (§1.3, §5.1); PSMark-F station/machine deployment |

The topic hierarchies match the paper's motivating conventions (`factory/line/machine/sensor`,
`health/device_id/metric`, `traffic/segment_id/metric`) so Algorithm 1's
clamp-compatible scope walk operates over a realistic tree structure. The
`<dataset>_topics.csv` manifest makes every subscriber-side attachment point
explicit for downstream regression / attack tests.

### Subscriptions at every topic level (per-level evaluation)

The evaluator does **not** test a single leaf subscription — every experiment
binds subscriptions at **every level of the PerCom topic hierarchy** and reports
utility per level. A subscription at level `L` pools the publishers under one
subtree prefix (level 1 = the domain root / whole type, the deepest level = a
single-publisher leaf); the helper `core.level_subscription_streams(dataset,
sensor, per_pub)` enumerates `(level, scope, aggregate, count, n_pubs)` for
every `(level × subtree)` and the experiments fan those out over the worker
pool. Result rows carry `subscription_level` and `scope` columns so you can read
utility vs. aggregation depth directly. Concretely:

| Experiment | Per-level coverage |
|---|---|
| **sweep** (main grid) | every level × subtree, for every sensor, × strategy × P × ε × w |
| **B** (vary w), **C** (vary ε) | every level × subtree × canonical combo |
| **L** (`subscription_levels`) | the dedicated per-level utility experiment |
| **G** (overhead), **H** (average-case) | every level × subtree (LDP baseline uses the per-publisher subset of each subtree) |
| **intro / Extreme 1.1** | every level — quantifies the distortion a level-`L` subscriber suffers when a single per-*type* release is delivered to all of them |
| **A** (greedy-vs-brute walk-up) | the full leaf→root level chain (the candidate rewrite depths) |
| **F** (ablation) | the `leaf` and `pooled` scopes (by design — they isolate the walk-up vs. interval-extension modules) |

### Trials and per-dataset aggregates

Every config is repeated over `--trials N` independent noise seeds (the paper
runs **6**). Each experiment writes **both** the per-trial rows (a `trial`
column) **and** a `*_aggregate.csv` with the mean/std of every metric grouped
per dataset (and per `subscription_level`/`scope` where applicable), so the
reported numbers are seed-averaged with visible variance. `generate_plots.py`
renders per-dataset + cross-dataset (`_all`) figures from those aggregates.

#### How the trees follow PSMark (PerCom)

The paper builds the topic hierarchies "based on standard publish–subscribe
benchmarking on the same datasets" — i.e. **PSMark** (*PSMark: A Distributed
IoT Benchmark for Publish/Subscribe Under Domain-Based Workloads*, PerCom).
PSMark itself does **not** prescribe literal topic strings; its
[device specification](https://github.com/DAMSlabUMBC/PSMark) defines, per
domain, the **device types**, the **per-device metrics**, and an
**edge-server / deployment grouping** (e.g. *3 factory floors*, *5 regional
hospitals*, smart-city sub-domains). Our trees realize exactly that structure —
`<domain>/<edge-server-or-grouping>/<device>/<metric>` — and our sensor lists
match PSMark's metric lists (smart-meter kWh/voltage/current; factory
temperature/speed/quality/vibration/energy; wearable heart_rate/steps/
temperature/calories; Pune humidity/PM/ozone/CO2/sound):

| PSMark domain | PSMark grouping (PerCom) | Our dataset(s) | Realized topic tree |
|---------------|--------------------------|----------------|---------------------|
| Smart City | smart meters + smart mobility + Pune air quality, per edge server | `energy`, `traffic`, `mobility`, `pune` | `energy/building01/{circuit}/{metric}`, `traffic/intersection01/{radar\|lidar}/{id}/{metric}`, `mobility/nyc/{quadrant}/{cell}/{metric}`, `air_quality/pune/{station}/{pollutant}` |
| Smart Factory | assembly line, 6 machines grouped into stations across floors | `manufacturing` | `factory/line1/{station}/{machine}/{sensor}` (stations: sorting / processing / warehouse / robotics) |
| Smart Healthcare | ICU beds across **5 regional hospitals**, one edge server each | `wearable` | `health/{hospital}/{device}/{metric}` (devices distributed over hospital01..hospital05) |
| Smart Home | UNSW device deployment | — (not in our six datasets) | — |

The **manufacturing** and **wearable** trees were deepened in this pass to add
PSMark's station and hospital grouping levels, which (a) makes Algorithm 1's
walk-up ladder realistic (machine → station → line → factory; device →
hospital → region) and (b) gives the §7.11 average-case experiment a
meaningful topic-hierarchy depth `h`. The four smart-city datasets keep their
existing roots (each is one facet of PSMark-C) and already carry multi-level
grouping (radar/lidar class, NE/NW/SE/SW quadrant, circuit, station).

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

### Per-release message logging

Every DP run records every logical-timestamp release as one row in an
append-friendly CSV.  The main sweep drains to
`results_3/<dataset>/<clamp_mode>/messages/sweep_messages.csv`; Experiments
B and C drain to `experiment_B_messages.csv` / `experiment_C_messages.csv`
under `cross_dataset/<clamp_mode>/experiments/{B,C}_*/`; Experiment D
already writes its own per-release log
(`experiment_D_plugin_releases.csv`); Experiment E drains to
`experiments/E_live_broker/<dataset>/experiment_E_messages.csv` with extra
live-broker audit columns (`leaf_topic`, `release_scope`, `walk_up`,
`broker_delivered`, `run_id`).  The schema
(see `message_logger.MESSAGE_LOG_COLUMNS`) is:

| Column | Meaning |
|--------|---------|
| `dataset`, `clamp_mode`, `sensor`, `strategy`, `P`, `epsilon`, `w`, `payload_bound`, `seed`, `experiment`, `config_id` | Configuration context |
| `tau`, `t_start_logical` | Logical timestamp and `(τ−1)·Δt` offline proxy for Def. 3.1's wall-clock start |
| `true_aggregate` | Clamped-mean `e_τ` BEFORE noise — the *original message* the broker would have released without DP |
| `noisy_value` | `ê_τ` delivered to the subscriber (or repeat-of-last when deferred) |
| `n_tau` | Multiplicity `|P_τ|` (broker-internal) |
| `epsilon_tau` | Per-element budget spent at τ |
| `lambda_tau` | Laplace scale `Δf / ε_τ = R / (n_τ ε_τ)` |
| `noise_sample` | `noisy − true` at τ (0.0 when deferred) |
| `deferred` | `True` when the release gate or skip fired |
| `delta_f` | Per-element sensitivity `R / n_τ` for the default mean |

These rows are the canonical audit trail for "every output message and its
original" — pass `--no-log-messages` to disable.

### Section 5.7 tuning: Algorithm 2 greedy + naive brute-force enumeration

Per the paper's updated Tuning architecture, `tune_hyperparameters` now runs
two-stage scope-first tuning with `P` as the primary knob (Δt adapts online
via Section 5.6; `(A, θ)` is selected by subscription requirements rather
than jointly optimized):

* **Greedy hill-climb with intelligent multi-start (Algorithm 2).** For each
  strategy, run `--n-restarts` independent hill-climbs (default 3) and keep
  the best local optimum.  The first restart always seeds at the paper's
  `P_0 = ⌈1/α⌉` identity-protection seed (default `α = 0.25` → `P_0 = 4`;
  flags `--alpha` / `--n-restarts` / `--restart-rng-seed`).  The remaining
  `--n-restarts − 1` starts are drawn from quantiles of the observed
  `n_τ` distribution on the trace, so the walks probe the pool-density
  regimes the stream actually exhibits instead of always starting at the
  same point — setting `--n-restarts 1` reverts to the paper's original
  deterministic seed.  Each restart probes neighbors `P − 1` and `P + 1`,
  steps to the better if it lowers the scalarized loss
  `L = NMAE + 0.2 · (1 − release_rate)`, and stops at a local optimum or
  after `--I-max` iterations (default 20).  The full greedy CSV records the
  restart index per probe; the summary CSV records `greedy_n_restarts` and
  `greedy_restart_seeds_P` for auditability.
* **Naive brute-force enumeration.** For the same strategy, evaluate every
  integer `P ∈ [1, p_max]` where `p_max = max n_τ` observed on the trace.

Per-strategy outputs land in `tuning/` with both the full greedy trajectory
and the complete brute-force curve, so the **empirical gap**
`greedy_loss − brute_loss` is directly measurable — this is what the paper
claims is small on the 1-D `P`-loss surface. Cross-dataset gap bars land in
`cross_dataset/tuning_greedy_vs_brute_gap.png`.

### Single-axis experiments (`--experiment {A|B|C|ABC|full}`, or `python -m experiments.<name>`)

On top of the main sweep, the runner exposes focused experiments that hold
every hyperparameter fixed except one. Each produces its own CSV + PNG per
clamp mode (per-trial rows + a `*_aggregate.csv`), runnable either via the CLI
`--experiment` flag or directly as a cluster-shardable module
(`python -m experiments.<name>`).

**Experiment A — greedy subscription walk-up vs brute over rewrite depths
(`--experiment A`, `experiments/greedy_vs_brute`).** This is about **Algorithm 1
(the subscription walk-up), not P-tuning.** For a subscription bound at a leaf,
the broker must pick a rewrite depth in the topic tree that pools ≥ P
range-compatible publishers while spending as little `ε_count` on discovery as
possible. **Greedy** walks the leaf→root chain spending one `ε_count` per probed
level and stops at the first ancestor whose **differentially-private** count
meets P; **brute** probes every level. Records the chosen level, utility,
`greedy_eps_count_spent` / `brute_eps_count_spent`, the `eps_count_saved`, and
the `probe_speedup` — so the cost of greedy discovery (and what it saves over
exhaustive probing) is explicit, accounting for the DP-count spend. (P-tuning
proper — Algorithm 2 greedy vs brute over `P` — lives separately in
`experiments/tuning`.)

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

### Ablation, overhead, and average-case experiments (`--experiment {F|G|H}`)

Three additional paper experiments, all **fully offline** (no MQTT broker, no
plugin object required):

**Experiment F — incremental-module ablation (§7.8, `--experiment F`).** Adds
the mechanism's modules one at a time and measures the utility impact of each:

1. **M1 — P-gated allocation only**: gate on `n_τ ≥ P_min`, defer otherwise.
2. **M2 — + subscription rewriting**: + adaptive interval extension (§6.7) —
   hold an under-P scope open up to `K_ext·Δt` to pool more publishers.
3. **M3 — + walking up the tree**: + Algorithm 1 hierarchy walk (§6.5) — rewrite
   scope to the nearest ancestor whose range-compatible pool meets P.

It reports two subscription scopes because the two rewriting modules dominate in
different sparsity regimes: at the `leaf` scope (one publisher) only **M3**'s
walk-up restores utility (spatial sparsity); at the `pooled` scope (gate active
on the whole-sensor topic) **M2**'s interval extension recovers the
temporally-sparse buckets. Together they give the complete incremental picture
§6.6 predicts. CSV: `cross_dataset/<clamp>/experiments/F_ablation/`.

**Experiment G — overhead / privacy-utility comparison (§7.9, `--experiment G`).**
At **every topic level × subtree** (per-level, fanned out over the worker pool),
compares **classic** (no privacy), **ldp** (`P_min=1` local DP, per-publisher
input perturbation `λ=R·w/ε`, using the per-publisher subset of each subtree),
**per_type_wevent** (one stream per sensor type), and **ours** (clamped w-event
DP with P-allocation). Reports NMAE, KL, release rate, attribution advantage
(identity protection), and a compute-overhead proxy (`compute_ms_per_element`,
`eps_count` surcharge). Rows carry `subscription_level`/`scope`/`approach`. True
broker throughput/latency is measured by the live Experiment E. CSV:
`.../G_overhead/`.

**Experiment H — average-case utility (§7.11, `--experiment H`).** At **every
topic level × subtree**, relates the **range-compatible publisher fraction**
(`|P_R|/|P|`) and the **topic-hierarchy depth `h`** to realized utility per
dataset, per §6.6's continuum. Rows carry `subscription_level`/`scope`. CSV:
`.../H_average_case/`.

**Experiment L — subscriptions at every topic level (`python -m experiments.subscription_levels`).**
The dedicated per-level experiment: evaluates the utility a subscriber receives
bound at each level of the PerCom tree (root/type → … → leaf) for every dataset,
making the utility-vs-aggregation-depth curve explicit. CSV:
`.../I_subscription_levels/`.

`--experiment FGH` runs F/G/H; `--experiment full` and `ABCDFGH` include them
alongside A/B/C/D. F/G/H/L consume the §7.5 grid optimum via `--use-grid-config`.

### The grid search (paper §6.1) fixes the params for every other experiment

```bash
# Run the Section 7.5 grid search (P_min x P_max x Δt x K_ext, scored by MAE)
# per dataset/strategy/epsilon, write the canonical config, and exit.
python run_experiment.py --grid-search --output-dir results_3
# -> results_3/grid_canonical.json + per-(dataset,ε) grid CSVs under tuning/grid_search/

# Run the grid search FIRST, then have the SAME run consume the MAE-optimal
# (P_min, P_max, K_ext) for every downstream experiment (paper §7.5: "we
# utilize these optimized values as the canonical fixed values for the
# following experiments").
python run_experiment.py --grid-first --experiment full --output-dir results_3

# Or consume a previously-written canonical config:
python run_experiment.py --experiment FGH \
    --use-grid-config results_3/grid_canonical.json --output-dir results_3
```

When a canonical config is present (`--use-grid-config`, or auto-detected at
`<output-dir>/grid_canonical.json`), Experiments F/G/H and the "ours" overhead
baseline resolve their `(P_min, P_max, K_ext)` per `(dataset, clamp, strategy,
ε)` from the grid optimum instead of the CLI defaults.

## Running on a cluster

Every paper experiment is now its own module under
[`experiments/`](experiments/) (`sweep`, `grid_search`, `intro`, `tuning`,
`extras`, `greedy_vs_brute`=A, `window`=B, `epsilon`=C, `plugin_path`=D,
`ablation`=F, `overhead`=G, `average_case`=H, `subscription_levels`=L), each run
as `python -m experiments.<name> --dataset … --clamp-mode … --workers …`.
`run_experiment.py` is a thin wrapper that dispatches to these modules and holds
the shared infrastructure they import. Every module runs on **all six datasets**,
builds the **PerCom/PSMark topic hierarchy**, tests **subscriptions at every
level of that hierarchy**, and repeats each config **≥6 trials** (per-trial rows
+ mean/std aggregate per dataset). Two launchers are provided under
[`cluster/`](cluster/):

**Generic shell + GNU parallel** (laptop, workstation, or any SSH node pool):

```bash
cluster/run_cluster.sh                         # local, all cores, aggregate at end
JOBS=8 cluster/run_cluster.sh                  # cap concurrent shards
SSHLOGINFILE=nodes.txt JOBS=4 cluster/run_cluster.sh   # multi-node over ssh
SHARD_BY=dataset-clamp-exp JOBS=32 cluster/run_cluster.sh   # finest shards
DRY_RUN=1 cluster/run_cluster.sh               # print the plan, run nothing
```

**SLURM job array** — job arrays are the standard SLURM pattern for exactly
this "same computation on different data" workload (no GNU parallel needed:
the array indexes the same joblist directly):

```bash
# From the repo root on the cluster (activate a Python >= 3.10 env first).
# Maximally-parallel fine mode (recommended): ~354 shards, static clamp.
SHARD_BY=dataset-clamp-exp CPUS=10 PARTITION=medium TIME=1-00:00:00 THROTTLE=80 \
  EXTRA_ARGS="--trials 6 --no-log-messages" \
  PUBSUB_ENV_SETUP='source ~/PubSubPrivacy/.venv/bin/activate' \
  cluster/submit_slurm.sh
DRY_RUN=1 SHARD_BY=dataset-clamp-exp cluster/submit_slurm.sh   # print plan, submit nothing
```

**Two sharding granularities:**

* **`dataset-clamp-exp` (fine, recommended)** — one shard per
  `dataset × clamp × experiment`, ~354 shards (static clamp). Submitted in
  **two SLURM phases**: **phase 1** runs the 144 `grid` shards (the §7.5 grid
  search split one-per-`(dataset × ε)`, writing `grid_canonical_eps<ε>.json`
  fragments into a shared `<ds>__<clamp>__grid` dir); **phase 2** runs the 114
  experiment shards with `--dependency=afterok` on phase 1, where F/G/H/L
  consume their dataset's merged grid optimum via `--use-grid-config`. The heavy
  per-level **sweep is split into 8 strategies × 3 sensor-groups per dataset**
  (`--strategies <s> --sensor-shard i/3`) and the **grid into 24 per-(ε×trial)
  shards** (`--grid-eps <ε> --grid-trial <t>`, its 6 trials as separate tasks)
  so both long poles spread across nodes — the heavy `energy` dataset fans its
  work across ~23 nodes instead of one straggler.
* **`dataset-clamp` (coarse, default)** — one self-contained
  `--grid-first --experiment full` shard per `dataset × clamp` (6 shards, static
  clamp, no phases). The grid search runs first and the full in-process pipeline
  consumes the grid-optimal params — all of §7 per shard, capped at 6 nodes.

See [`cluster/README.md`](cluster/README.md) for every knob, and
[`cluster/SLURM_RUNBOOK.md`](cluster/SLURM_RUNBOOK.md) for the **step-by-step
SLURM runbook** (env setup → submit → monitor → **copy results back** →
figures).

### Copying cluster results back to your machine

The aggregation job builds **`results_cluster/paper_bundle/`** — the curated
key individual (per-dataset) + combined (cross-dataset) results the paper
references, in one small section-organized directory (bulky per-release message
dumps excluded; see its `INDEX.md`). For most purposes scp just that one dir.
Run these **from your local machine**:

```bash
# RECOMMENDED: just the curated paper results (small, single download):
scp -r <user>@<cluster-login>:~/PubSubPrivacy/results_cluster/paper_bundle \
       ./paper_bundle

# The full merged tree (one CSV per artifact, all datasets — larger):
scp -r <user>@<cluster-login>:~/PubSubPrivacy/results_cluster/combined \
       ./results_cluster_combined

# Everything (combined + paper_bundle + every per-shard dir + joblists + logs):
scp -r <user>@<cluster-login>:~/PubSubPrivacy/results_cluster \
       ./results_cluster

# rsync is better for large/resumable transfers (skips unchanged files):
rsync -avz --progress \
  <user>@<cluster-login>:~/PubSubPrivacy/results_cluster/paper_bundle/ \
  ./paper_bundle/

# Render figures locally from the full tree if you didn't pass PLOTS=1:
python generate_plots.py --output-dir ./results_cluster_combined
```

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

Each experiment runs **concurrently within** (ProcessPoolExecutor fans out
the task list over `--workers` cores) but **sequentially across**
experiments — the sweep finishes before Experiment A starts, A before B,
and so on, so the DP budget accounting for each experiment is isolated.
Inside each experiment, independent tasks (sensor × strategy × P × ε × w,
or per-config greedy walks) run in parallel.

The canonical command — runs every artifact the paper references, on every
dataset, in both clamp modes, writes every CSV under `results_3/`, and then
renders every PNG from those CSVs — is:

```bash
# 1.  Experiments (CSV-only; plots are a separate pass).
#     --run-live-E-after-full chains Experiment E after the main pipeline,
#     iterating over every non-energy dataset (live MQTT broker,
#     pooled + hierarchy scenarios, 3 concurrent subscribers per config).
python run_experiment.py --experiment full --clamp-mode both \
                        --alpha 0.25 --eps-clip 0.1 \
                        --n-restarts 3 \
                        --run-live-E-after-full \
                        --live-scenarios pooled,hierarchy \
                        --live-n-subscribers 3 \
                        --output-dir results_3

# 2.  Plots — reads every CSV under results_3/ and renders the full PNG set
#     (includes Experiment E figures under
#     experiments/E_live_broker/<dataset>/).
python generate_plots.py --output-dir results_3
```

Experiment E is optional — drop `--run-live-E-after-full` to skip it, or
run it separately afterwards:

```bash
python run_experiment.py --experiment E --dataset all \
                        --live-scenarios pooled,hierarchy \
                        --live-n-subscribers 3 \
                        --output-dir results_3
```

For a faster smoke-test of the pipeline (reduced grids, capped row counts,
separate output tree) use:

```bash
python run_experiment.py --experiment full --quick --clamp-mode both \
                        --max-energy-timestamps 3000 \
                        --max-traffic-rows 30000 --max-rows 5000 \
                        --output-dir results_test
python generate_plots.py --output-dir results_test
```

### Experiment E: what it actually tests

Experiment E exercises the broker-side plugin end-to-end against a real
MQTT broker. If nothing is listening on `--broker-host:--broker-port`, it
starts a local **Eclipse Mosquitto** process (the default, `--broker-impl
mosquitto`; the binary is taken from `$MOSQUITTO_BIN`, then `PATH`, then the
installer's default location) or, with `--broker-impl amqtt`, the embedded
pure-Python `amqtt` broker (`pip install amqtt`). An already-running broker is
always reused as-is. Mosquitto is not a pip package: install it from
https://mosquitto.org/download/ before running Experiment E or `latency --live`.  Per dataset, per
`(strategy × epsilon × scenario)` config, it spins up:

* a `PrivacyPlugin` running the full DP pipeline,
* all dataset publishers emitting their recorded traces on unique
  per-`run_id` raw topics (the **many-publishers** path — every publisher
  registered by that dataset's loader is emitted concurrently),
* `--live-n-subscribers` independent subscriber clients (default 3) all
  listening on the protected prefix, so the broker must fan every release
  out to every subscriber.

Scenarios (run both by default):

| Scenario | Topic layout | What it exercises |
|----------|--------------|-------------------|
| `pooled` | All publishers emit on ONE shared leaf; the plugin pools them into a single aggregate per τ. | Many-publishers path under a shared-leaf subscription; also byte-for-byte comparison against `run_dp_on_stream` under the same seed. |
| `hierarchy` | Each publisher emits on its OWN leaf under the dataset's normative MQTT tree (`factory/line1/<machine>/<sensor>`, etc.). The plugin sees `n_τ=1` at every leaf, so Algorithm 1 walks up to a clamp-compatible ancestor on every release. | **P-enforced walk-up** — every released record should have `walk_up=True` and `release_scope != leaf`.  The summary row tracks `walkup_rate` and `p_gate_violations` (must be 0). |

The `experiment_E_live_broker.csv` summary row records for every config:
`release_rate_live` vs `release_rate_offline`, `nmae_live` vs `nmae_offline`,
`kl_live` vs `kl_offline`, `broker_deliveries` vs `expected_deliveries`,
`per_subscriber_counts`, `num_walkups`, `walkup_rate`, and
`p_gate_violations`.  The `experiment_E_messages.csv` per-release log
carries the full DP-engine schema plus live-broker audit columns
(`scenario`, `plugin_P`, `num_subscribers`, `leaf_topic`, `release_scope`,
`walk_up`, `broker_delivered`, `run_id`).


