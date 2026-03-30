# PubSub Privacy Plugin for MQTT

An MQTT middleware plugin implementing the two-layer privacy architecture from *"Privacy in Publish Subscribe Systems"* (Olinger & Pappachan, 2026).

- **Layer 1 - Content Privacy (Input Side):** w-event epsilon-differential privacy via Laplace noise injection at broker ingress. Protects *what* a message contains.
- **Layer 2 - Publisher Privacy (Output Side):** k-anonymity, l-diversity, and t-closeness via topic generalization at broker egress. Protects *who* sent a message.

Both layers use rolling windows of size `w` and produce per-message output (no batching).

## Project Structure

```text
pubsub_privacy/
  __init__.py              # Package exports
  content_privacy.py       # Layer 1: w-event DP (Uniform, Sample, BA strategies)
  publisher_privacy.py     # Layer 2: k-anonymity + l-diversity + t-closeness
  broker.py                # Two-layer privacy broker combining both layers
  mqtt_plugin.py           # MQTT middleware (subscribes raw/#, publishes private/#)
run_plugin.py              # CLI entry point for the MQTT plugin
smart_factory_benchmark.py # Self-contained benchmark (publisher + subscriber + graphs)
mosquitto.conf             # Minimal config for local Mosquitto broker
```

## Prerequisites

- **Python 3.10+**
- **Mosquitto MQTT broker** (for live MQTT mode only; benchmark runs without it)

### Install Mosquitto

**Windows (winget):**

```bash
winget install EclipseFoundation.Mosquitto
```

**Windows (Chocolatey):**

```bash
choco install mosquitto
```

**macOS:**

```bash
brew install mosquitto
```

**Ubuntu/Debian:**

```bash
sudo apt install mosquitto mosquitto-clients
```

**Docker (any platform):**

```bash
docker run -d --name mosquitto -p 1883:1883 eclipse-mosquitto:2 mosquitto -c /mosquitto-no-auth.conf
```

## Quick Start

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the benchmark (no MQTT broker needed)

```bash
python smart_factory_benchmark.py
```

This simulates a smart factory with temperature, seismic/vibration, power, air quality, and pressure sensors across 10 publisher companies. It runs a baseline (no privacy) and a privacy-enabled pass, then saves results to `results/`.

Customize the run:

```bash
python smart_factory_benchmark.py --messages 1000 --publishers 15 --epsilon 0.5 --window 20 --k 3 --l 2
```

### Output

CSV files:

- `results/factory_baseline.csv` - Per-message baseline latencies
- `results/factory_plugin.csv` - Per-message plugin latencies
- `results/factory_comparison.csv` - Side-by-side with overhead, payload distortion, topic generalization

Graphs:

- `results/factory_latency_comparison.png` - Mean + P95 bar chart
- `results/factory_latency_timeseries.png` - Per-message latency over time
- `results/factory_latency_distribution.png` - Latency histogram
- `results/factory_payload_distortion.png` - Original vs perturbed by sensor type
- `results/factory_k_anonymity.png` - k-anonymity achieved per message
- `results/factory_budget_remaining.png` - Rolling window budget over time
- `results/factory_latency_by_sensor.png` - Latency box plot by sensor type

### 3. Run the live MQTT plugin (optional)

Start the MQTT broker:

```bash
mosquitto -c mosquitto.conf -v
```

Start the privacy plugin:

```bash
python run_plugin.py --epsilon 1.0 --window 4 --sensitivity 30 --k 2 --l 2 -v
```

The plugin subscribes to `raw/#`, applies both privacy layers, and republishes sanitized messages to `private/#`.

## Plugin CLI Options

```text
python run_plugin.py --help

Content Privacy (Input Side):
  --epsilon FLOAT       DP budget (default: 1.0)
  --window INT          Rolling window size w (default: 10)
  --sensitivity FLOAT   Payload sensitivity Delta (default: 100.0)
  --strategy            uniform | sample | budget_absorption (default: uniform)
  --ba-threshold FLOAT  BA skip threshold (default: 1.0)

Publisher Privacy (Output Side):
  --k INT               k-anonymity (default: 2)
  --l INT               l-diversity, 0=disabled (default: 0)
  --t FLOAT             t-closeness, inf=disabled (default: inf)

MQTT:
  --host HOST           Broker host (default: localhost)
  --port PORT           Broker port (default: 1883)
  --source PATTERN      Source topic (default: raw/#)
  --dest PREFIX         Destination prefix (default: private)
```

## How It Works

### Message Flow

```text
Publisher                    MQTT Broker                 Privacy Plugin               Subscriber
    |                            |                            |                          |
    |-- publish raw/topic ------>|                            |                          |
    |                            |-- forward to plugin ------>|                          |
    |                            |                     [Layer 1: Laplace noise]          |
    |                            |                     [Layer 2: Topic generalize]       |
    |                            |<-- publish private/topic* --|                          |
    |                            |                            |                          |
    |                            |----------- deliver ------->|------------------------->|
```

### Content Privacy (Section 4.2)

Each publisher's payload stream is independently perturbed using the Laplace mechanism. The budget constraint ensures that in any rolling window of `w` consecutive messages from the same publisher, the total epsilon spent is at most the configured `epsilon`:

```text
For all i: sum(epsilon_k for k in [i-w+1, i]) <= epsilon
```

Three budget allocation strategies from Kellaris et al. (2014):

- **Uniform:** `epsilon_i = epsilon/w` per message. Constant noise.
- **Sample:** Every w-th message gets full budget; others repeat the last output.
- **BA (Budget Absorption):** Skip messages whose payload barely changed from the last published value, absorbing their budget for future high-change messages.

### Publisher Privacy (Section 4.3)

At delivery time, the broker generalizes each message's MQTT topic by walking up the topic hierarchy until at least `k` distinct publishers share the rewritten topic within the rolling window. Optional `l`-diversity ensures at least `l` distinct sensitive attribute values, and `t`-closeness bounds the distributional distance.

Example from the paper: `traffic/elm/1st` generalizes to `traffic/elm/*` when two publishers (p1, p2) share that subtree in the window.
