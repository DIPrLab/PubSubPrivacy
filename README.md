# PubSubPrivacy

**S-Sensitive w-Event Differential Privacy for MQTT Publish-Subscribe Systems**

An MQTT broker plugin that enforces differential privacy on IoT data streams, protecting both publisher identity and message content from honest-but-curious subscribers. Based on extending Kellaris et al.'s w-event DP framework to the pub/sub aggregate stream model.

## Architecture

```
Publishers (factory sensors)
       |
       v
  [raw MQTT topics]        factory/raw/{line}/{machine}/{sensor}
       |
       v
  PrivacyPlugin (broker middleware)
    - Buffers messages per timestamp interval
    - Aggregates payloads (mean over contributing publishers)
    - Enforces |P_tau| >= S (minimum publisher count)
    - Applies Laplace noise: e_hat = e_tau + Lap(B / (S * epsilon_tau))
    - Enforces sliding window budget constraint
       |
       v
  [protected MQTT topics]  factory/protected/{line}/{machine}/{sensor}
       |
       v
  Subscribers (only see noisy aggregates)
```

## Components

| File | Description |
|------|-------------|
| `dp_engine.py` | Core DP engine: Laplace mechanism, sliding window budget, Uniform/Sample/Budget Absorption strategies |
| `plugin.py` | MQTT privacy plugin middleware that intercepts, aggregates, and protects messages |
| `factory_iot.py` | Realistic factory IoT simulator with production lines, machines, and sensors |
| `run_experiment.py` | Synthetic data experimental evaluation with parameter sweeps and plotting |
| `run_real_data_experiment.py` | Real-data experimental evaluation using IEEE research datasets |
| `config.yaml` | Configuration file for all parameters |

## Key Parameters

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| `epsilon` | epsilon | Global privacy budget per sliding window |
| `window_size` | w | Number of timestamps in the sliding window |
| `min_publishers` | S | Minimum publishers per aggregate (sensitivity hyperparameter) |
| `strategy` | -- | Budget allocation: `uniform`, `sample`, or `budget_absorption` |

The per-element sensitivity is Delta_S = B / S, where B is the payload domain range. Noise scale is lambda = B * w / (S * epsilon) under uniform allocation.

## Quick Start

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run offline experiment (no MQTT broker needed)

```bash
# Quick demo
python run_experiment.py --offline --quick

# Full parameter sweep
python run_experiment.py --offline

# Custom parameters
python run_experiment.py --offline --num-publishers 12 --num-timestamps 200
```

### Run live experiment (requires Mosquitto or another MQTT broker)

```bash
# Start Mosquitto broker first
mosquitto -c mosquitto.conf

# Then run live experiment
python run_experiment.py --live --duration 120
```

### Run real-data experiment (IEEE research datasets)

Uses two real IoT datasets stored in `data/`:

1. **MCEC-Thai** (IEEE 10949848) - Multi-circuit electric consumption from a Thailand home (178K readings, 12 circuits, 1-min intervals)
2. **Colorado Springs Traffic** (IEEE 11134445) - Multi-sensor object detection at a traffic intersection (4 radars + 2 lidars, ~1.3M detections)

```bash
# Full sweep on both datasets
python run_real_data_experiment.py

# Quick demo with reduced parameter sweep
python run_real_data_experiment.py --quick

# Run only one dataset
python run_real_data_experiment.py --dataset energy
python run_real_data_experiment.py --dataset traffic

# Limit data size for faster testing
python run_real_data_experiment.py --quick --max-energy-timestamps 10000 --max-traffic-rows 50000
```

**Energy scenario (smart building):** Each circuit breaker is a publisher. Topics are `power_kw`, `voltage`, and `current`, aggregated across circuits per 5-minute window. 6-12 active publishers per window from real household data.

**Traffic scenario (smart city):** Each radar/lidar sensor is a publisher. Topics are `speed` (mean vehicle speed) and `object_count` (unique detected vehicles), aggregated per 10-second window. 2-6 active publishers per window from real intersection data.

### Output

**Synthetic experiments** are saved to `results/`:
- `sweep_results.csv` - Raw data from parameter sweep
- `mae_vs_epsilon.png` - MAE vs privacy budget by strategy
- `mae_vs_S.png` - MAE vs sensitivity parameter by sensor type
- `strategy_comparison.png` - Budget allocation strategy comparison
- `noise_scale_vs_S.png` - Theoretical noise scale analysis
- `timeseries_comparison.png` - True vs. DP-protected time series
- `budget_utilization.png` - Sliding window budget usage over time

**Real-data experiments** are saved to `results_real_data/`:
- `energy/` and `traffic/` - Per-dataset results:
  - `sweep_results.csv` - Raw sweep data with all metrics
  - `*_mae_vs_epsilon.png` - MAE vs privacy budget by strategy
  - `*_mae_vs_S.png` - MAE vs sensitivity parameter by sensor
  - `*_strategy_comparison.png` - Strategy comparison
  - `*_timeseries.png` - True vs DP-protected real data streams
  - `*_budget_utilization.png` - Sliding window budget usage
  - `*_kl_vs_epsilon.png` - KL divergence vs privacy budget
  - `*_kl_vs_mae_tradeoff.png` - Privacy-utility tradeoff scatter
  - `*_kl_heatmap.png` - KL divergence across all (w, epsilon, S, strategy) configs
  - `*_kl_windowed.png` - Per-window KL divergence over time
- `combined_sweep_results.csv` - Merged results from both datasets
- `cross_dataset_comparison.png` - Energy vs traffic privacy-utility comparison

## Factory IoT Schema

The simulator models a factory with:
- **2 production lines**, each with **4 machines**
- **4 sensor types** per machine:

| Sensor | Unit | Range [min, max] | Topic Pattern |
|--------|------|-------------------|---------------|
| Temperature | celsius | [15, 120] | `factory/raw/lineXX/machineYY/temperature` |
| Vibration | mm/s | [0, 50] | `factory/raw/lineXX/machineYY/vibration` |
| Power Draw | kW | [0, 500] | `factory/raw/lineXX/machineYY/power_draw` |
| Humidity | percent | [10, 95] | `factory/raw/lineXX/machineYY/humidity` |

Machines transition stochastically between **idle**, **running**, and **maintenance** states, producing realistic time-series with warm-up ramps, sinusoidal drift, and measurement noise.

## Message Format

**Publisher (raw):**
```json
{
  "publisher_id": "line01/machine02/temperature",
  "value": 72.4531,
  "unit": "celsius",
  "ts": 1711720800.0
}
```

**Subscriber (protected):**
```json
{
  "timestamp": 15,
  "value": 73.1204,
  "num_publishers": 7,
  "suppressed": false
}
```
## Real-Data Sources

| Dataset | IEEE DOI | Source | Description |
|---------|----------|--------|-------------|
| MCEC-Thai | [10949848](https://ieeexplore.ieee.org/document/10949848) | [Mendeley](https://data.mendeley.com/datasets/nmnk58bgtb/1) | Multi-circuit electric consumption, Bangkok home, 12 circuits, 1-min sampling |
| Colorado Springs Traffic | [11134445](https://ieeexplore.ieee.org/document/11134445) | [NLR](https://data.nlr.gov/submissions/287) | Multi-sensor object detection (4 radar + 2 lidar) at traffic intersection |
| UNSW-IoTraffic | [11137365](https://ieeexplore.ieee.org/document/11137365) | [Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.w0vt4b94b) | IoT network traffic data with packets, flows, and protocols |
