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
| `run_experiment.py` | Experimental evaluation script with parameter sweeps and plotting |
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

### Output

Results are saved to `results/`:
- `sweep_results.csv` - Raw data from parameter sweep
- `mae_vs_epsilon.png` - MAE vs privacy budget by strategy
- `mae_vs_S.png` - MAE vs sensitivity parameter by sensor type
- `strategy_comparison.png` - Budget allocation strategy comparison
- `noise_scale_vs_S.png` - Theoretical noise scale analysis
- `timeseries_comparison.png` - True vs. DP-protected time series
- `budget_utilization.png` - Sliding window budget usage over time

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
