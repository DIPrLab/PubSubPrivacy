#!/usr/bin/env python3
"""
Smart Factory Benchmark

A self-contained benchmark that acts as both publisher and subscriber.
Simulates a realistic smart factory floor with multiple sensor types,
many publishers, and multiple subscribers. Measures per-message latency
with and without the privacy plugin, saves results to CSV, and generates
comparison graphs.

Sensor types:
  - Temperature sensors (assembly lines, ovens, coolant systems)
  - Seismic/vibration sensors (CNC machines, presses, foundations)
  - Power draw meters (motors, welding stations, HVAC)
  - Air quality sensors (paint booths, clean rooms)
  - Pressure sensors (hydraulic lines, pneumatic systems)

No external MQTT broker needed - runs the privacy broker in-process.

Usage:
    python smart_factory_benchmark.py
    python smart_factory_benchmark.py --messages 1000 --publishers 20 --subscribers 5
    python smart_factory_benchmark.py --epsilon 0.5 --window 20 --k 3 --l 2
"""

import argparse
import csv
import os
import statistics
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from scipy.special import rel_entr

from pubsub_privacy.broker import PrivacyBroker, PubSubMessage
from pubsub_privacy.content_privacy import BudgetStrategy

RESULTS_DIR = "results"


# ---------------------------------------------------------------------------
# Smart factory sensor simulation
# ---------------------------------------------------------------------------

SENSOR_TYPES = {
    "temperature": {
        "topics": [
            "factory/floor1/assembly/line_a/temp",
            "factory/floor1/assembly/line_b/temp",
            "factory/floor1/oven/zone1/temp",
            "factory/floor1/oven/zone2/temp",
            "factory/floor2/coolant/loop1/temp",
            "factory/floor2/coolant/loop2/temp",
            "factory/floor2/cleanroom/temp",
            "factory/exterior/loading_dock/temp",
        ],
        "range": (15.0, 300.0),
        "sensitivity": 285.0,
        "unit": "C",
    },
    "vibration": {
        "topics": [
            "factory/floor1/cnc/mill_01/vibration",
            "factory/floor1/cnc/mill_02/vibration",
            "factory/floor1/cnc/lathe_01/vibration",
            "factory/floor1/press/hydraulic_01/vibration",
            "factory/floor1/press/hydraulic_02/vibration",
            "factory/floor2/foundation/north/seismic",
            "factory/floor2/foundation/south/seismic",
            "factory/floor2/foundation/east/seismic",
        ],
        "range": (0.0, 50.0),
        "sensitivity": 50.0,
        "unit": "mm/s",
    },
    "power": {
        "topics": [
            "factory/floor1/motor/conveyor_a/power",
            "factory/floor1/motor/conveyor_b/power",
            "factory/floor1/welding/station_01/power",
            "factory/floor1/welding/station_02/power",
            "factory/floor2/hvac/ahu_01/power",
            "factory/floor2/hvac/ahu_02/power",
            "factory/floor2/compressor/main/power",
        ],
        "range": (0.0, 500.0),
        "sensitivity": 500.0,
        "unit": "kW",
    },
    "air_quality": {
        "topics": [
            "factory/floor1/paint/booth_a/aqi",
            "factory/floor1/paint/booth_b/aqi",
            "factory/floor2/cleanroom/aqi",
            "factory/floor2/warehouse/aqi",
        ],
        "range": (0.0, 500.0),
        "sensitivity": 500.0,
        "unit": "AQI",
    },
    "pressure": {
        "topics": [
            "factory/floor1/hydraulic/main_line/pressure",
            "factory/floor1/hydraulic/return_line/pressure",
            "factory/floor1/pneumatic/supply/pressure",
            "factory/floor2/pneumatic/tool_air/pressure",
        ],
        "range": (0.0, 350.0),
        "sensitivity": 350.0,
        "unit": "bar",
    },
}

PUBLISHER_COMPANIES = [
    {"id": "siemens_iot",       "attr": "multinational"},
    {"id": "bosch_sensors",     "attr": "multinational"},
    {"id": "honeywell_ind",     "attr": "multinational"},
    {"id": "local_integrator",  "attr": "small_business"},
    {"id": "acme_monitoring",   "attr": "small_business"},
    {"id": "abb_automation",    "attr": "multinational"},
    {"id": "rockwell_ctrl",     "attr": "multinational"},
    {"id": "startupX_sensors",  "attr": "startup"},
    {"id": "diy_arduino_lab",   "attr": "startup"},
    {"id": "midwest_gauge_co",  "attr": "small_business"},
    {"id": "omega_instruments", "attr": "multinational"},
    {"id": "keyence_precision", "attr": "multinational"},
    {"id": "fluke_industrial",  "attr": "multinational"},
    {"id": "banner_eng",       "attr": "small_business"},
    {"id": "turck_sensors",    "attr": "multinational"},
    {"id": "ifm_electronic",   "attr": "multinational"},
    {"id": "pepperl_fuchs",    "attr": "multinational"},
    {"id": "endress_hauser",   "attr": "multinational"},
    {"id": "yokogawa_ctrl",    "attr": "multinational"},
    {"id": "emerson_process",  "attr": "multinational"},
]

SUBSCRIBER_PROFILES = [
    {"id": "floor_manager",      "filter": "factory/floor1/#"},
    {"id": "hvac_controller",    "filter": "factory/+/+/+/temp"},
    {"id": "safety_system",      "filter": "factory/#"},
    {"id": "vibration_analyst",  "filter": "factory/+/+/+/vibration"},
    {"id": "energy_dashboard",   "filter": "factory/+/+/+/power"},
    {"id": "ehs_compliance",     "filter": "factory/+/+/+/aqi"},
    {"id": "maintenance_ai",     "filter": "factory/#"},
    {"id": "corporate_analytics","filter": "factory/#"},
]


@dataclass
class SensorMessage:
    publisher_id: str
    sensitive_attr: str
    topic: str
    value: float
    sensor_type: str
    timestamp: float


def generate_factory_stream(n_messages: int, n_publishers: int, rng: np.random.Generator) -> list[SensorMessage]:
    """Generate a realistic stream of factory sensor messages."""
    publishers = PUBLISHER_COMPANIES[:n_publishers]
    all_topics = []
    for stype, config in SENSOR_TYPES.items():
        for topic in config["topics"]:
            all_topics.append((stype, topic, config["range"]))

    messages = []
    base_time = time.time()

    for i in range(n_messages):
        pub = publishers[i % len(publishers)]
        stype, topic, (lo, hi) = all_topics[i % len(all_topics)]

        # Simulate correlated sensor readings (slow drift + noise)
        drift = 0.5 * (hi - lo) + 0.3 * (hi - lo) * np.sin(2 * np.pi * i / 200)
        noise = rng.normal(0, 0.05 * (hi - lo))
        value = np.clip(drift + noise, lo, hi)

        messages.append(SensorMessage(
            publisher_id=pub["id"],
            sensitive_attr=pub["attr"],
            topic=topic,
            value=round(float(value), 2),
            sensor_type=stype,
            timestamp=base_time + i * 0.1,
        ))

    return messages


# ---------------------------------------------------------------------------
# Benchmark engine
# ---------------------------------------------------------------------------


def run_benchmark(
    messages: list[SensorMessage],
    subscribers: list[dict],
    epsilon: float,
    w: int,
    strategy: str,
    k: int,
    l_div: int,
    t_close: float,
) -> tuple[list[dict], list[dict]]:
    """
    Run baseline and plugin benchmarks.

    Returns (baseline_results, plugin_results).
    Each result row includes per-message timing and metadata.
    """
    # --- Baseline: pass-through, no privacy ---
    baseline_results = []
    for i, msg in enumerate(messages):
        t0 = time.perf_counter_ns()
        output_value = msg.value
        output_topic = msg.topic
        t1 = time.perf_counter_ns()

        baseline_results.append({
            "message_index": i,
            "latency_us": (t1 - t0) / 1000.0,
            "publisher_id": msg.publisher_id,
            "sensor_type": msg.sensor_type,
            "topic_original": msg.topic,
            "topic_output": output_topic,
            "payload_original": msg.value,
            "payload_output": output_value,
        })

    # --- With privacy broker ---
    # Use the max sensitivity across sensor types
    max_sensitivity = max(cfg["sensitivity"] for cfg in SENSOR_TYPES.values())

    broker = PrivacyBroker(
        epsilon=epsilon,
        w=w,
        sensitivity=max_sensitivity,
        strategy=BudgetStrategy(strategy),
        ba_threshold=max_sensitivity * 0.02,
        k=k,
        l=l_div,
        t=t_close,
    )

    plugin_results = []
    for i, msg in enumerate(messages):
        t0 = time.perf_counter_ns()
        result = broker.process_raw(
            publisher_id=msg.publisher_id,
            payload=msg.value,
            topic=msg.topic,
            sensitive_attr=msg.sensitive_attr,
            timestamp=msg.timestamp,
        )
        t1 = time.perf_counter_ns()

        plugin_results.append({
            "message_index": i,
            "latency_us": (t1 - t0) / 1000.0,
            "publisher_id": msg.publisher_id,
            "sensor_type": msg.sensor_type,
            "topic_original": msg.topic,
            "topic_output": result.generalized_topic,
            "payload_original": msg.value,
            "payload_output": round(result.perturbed_payload, 4),
            "consistent_publishers": len(result.consistent_publishers),
            "budget_remaining": round(result.budget_info.get("window_budget_remaining", 0), 4),
            "window_id": result.budget_info.get("window_id", 0),
            "window_position": result.budget_info.get("window_position", 0),
            "budget_used": round(result.budget_info.get("window_budget_used", 0), 4),
        })

    return baseline_results, plugin_results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def save_csv(data: list[dict], filename: str) -> str:
    ensure_results_dir()
    path = os.path.join(RESULTS_DIR, filename)
    if not data:
        return path
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
    print(f"    Saved: {path}")
    return path


def print_stats(label: str, results: list[dict]):
    latencies = [r["latency_us"] for r in results if r["latency_us"] >= 0]
    if not latencies:
        print(f"  {label}: No results")
        return
    print(f"\n  {label} ({len(latencies)} messages):")
    print(f"    Mean latency:   {statistics.mean(latencies):>12.2f} us")
    print(f"    Median latency: {statistics.median(latencies):>12.2f} us")
    print(f"    P95 latency:    {np.percentile(latencies, 95):>12.2f} us")
    print(f"    P99 latency:    {np.percentile(latencies, 99):>12.2f} us")
    print(f"    Std dev:        {statistics.stdev(latencies) if len(latencies) > 1 else 0:>12.2f} us")
    print(f"    Min:            {min(latencies):>12.2f} us")
    print(f"    Max:            {max(latencies):>12.2f} us")


def compute_kl_divergence(original: list[float], perturbed: list[float], n_bins: int = 50) -> float:
    """Compute KL divergence D_KL(P_original || Q_perturbed) using histograms."""
    lo = min(min(original), min(perturbed))
    hi = max(max(original), max(perturbed))
    if lo == hi:
        return 0.0
    bins = np.linspace(lo, hi, n_bins + 1)
    p_counts, _ = np.histogram(original, bins=bins)
    q_counts, _ = np.histogram(perturbed, bins=bins)
    # Convert to probabilities with Laplace smoothing to avoid division by zero
    p = (p_counts + 1e-10) / (p_counts + 1e-10).sum()
    q = (q_counts + 1e-10) / (q_counts + 1e-10).sum()
    return float(np.sum(rel_entr(p, q)))


def print_privacy_stats(results: list[dict]):
    k_values = [r.get("consistent_publishers", 0) for r in results]
    if not k_values:
        return
    print(f"\n  Publisher Privacy (k-anonymity achieved):")
    print(f"    Mean k:   {statistics.mean(k_values):.2f}")
    print(f"    Min k:    {min(k_values)}")
    print(f"    Max k:    {max(k_values)}")

    # Topic generalization stats
    generalized = sum(1 for r in results if r["topic_original"] != r["topic_output"])
    print(f"    Topics generalized: {generalized}/{len(results)} ({100*generalized/len(results):.1f}%)")

    # Window stats
    window_ids = set()
    for r in results:
        pub = r.get("publisher_id", "")
        wid = r.get("window_id", 0)
        window_ids.add((pub, wid))
    print(f"\n  Window Statistics:")
    print(f"    Total windows completed: {len(window_ids)}")
    max_wid = max(r.get("window_id", 0) for r in results)
    print(f"    Max window ID:          {max_wid}")

    # Payload distortion
    errors = [abs(r["payload_original"] - r["payload_output"]) for r in results]
    print(f"\n  Content Privacy (payload distortion):")
    print(f"    Mean |error|:   {statistics.mean(errors):.4f}")
    print(f"    Median |error|: {statistics.median(errors):.4f}")
    print(f"    Max |error|:    {max(errors):.4f}")

    # KL divergence per sensor type
    sensor_types = sorted(set(r["sensor_type"] for r in results))
    print(f"\n  KL Divergence (original vs. plugin output):")
    all_orig = [r["payload_original"] for r in results]
    all_pert = [r["payload_output"] for r in results]
    overall_kl = compute_kl_divergence(all_orig, all_pert)
    print(f"    Overall:        {overall_kl:.4f}")
    for stype in sensor_types:
        subset = [r for r in results if r["sensor_type"] == stype]
        orig = [r["payload_original"] for r in subset]
        pert = [r["payload_output"] for r in subset]
        kl = compute_kl_divergence(orig, pert)
        print(f"    {stype:<15} {kl:.4f}")


def _add_window_boundaries(ax, plugin: list[dict], w: int, color="#888888", alpha=0.3):
    """Add vertical lines at window boundaries where budget resets."""
    boundaries = []
    for r in plugin:
        if r.get("window_position") == 1 and r["message_index"] > 0:
            boundaries.append(r["message_index"])
    boundaries = sorted(set(boundaries))
    for i, b in enumerate(boundaries):
        label = f"Window boundary (w={w})" if i == 0 else None
        ax.axvline(x=b, color=color, linestyle=":", alpha=alpha, linewidth=0.7, label=label)


def generate_graphs(baseline: list[dict], plugin: list[dict], w: int = 10, epsilon: float = 1.0):
    ensure_results_dir()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("    matplotlib not installed - skipping graphs (pip install matplotlib)")
        return

    b_lat = [r["latency_us"] for r in baseline]
    p_lat = [r["latency_us"] for r in plugin]

    # --- 1. Latency comparison bar chart ---
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["Baseline\n(no privacy)", "With Privacy\nPlugin"]
    means = [statistics.mean(b_lat), statistics.mean(p_lat)]
    p95s = [np.percentile(b_lat, 95), np.percentile(p_lat, 95)]
    x = range(len(labels))
    bw = 0.35
    bars1 = ax.bar([i - bw/2 for i in x], means, bw, label="Mean", color=["#2196F3", "#FF5722"])
    bars2 = ax.bar([i + bw/2 for i in x], p95s, bw, label="P95", color=["#90CAF9", "#FFAB91"], alpha=0.8)
    ax.set_ylabel("Latency (microseconds)")
    ax.set_title(f"Smart Factory: Message Processing Latency (w={w}, e={epsilon})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    for bar, val in zip(list(bars1) + list(bars2), means + p95s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "factory_latency_comparison.png"), dpi=150)
    plt.close()
    print(f"    Saved: results/factory_latency_comparison.png")

    # --- 2. Per-message latency time series ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax1.plot(range(len(b_lat)), b_lat, alpha=0.6, linewidth=0.5, color="#2196F3")
    ax1.set_ylabel("Latency (us)")
    ax1.set_title("Baseline - Per-Message Latency")
    ax1.axhline(y=statistics.mean(b_lat), color="red", linestyle="--", alpha=0.5, label=f"mean={statistics.mean(b_lat):.1f}us")
    ax1.legend()

    ax2.plot(range(len(p_lat)), p_lat, alpha=0.6, linewidth=0.5, color="#FF5722")
    ax2.set_ylabel("Latency (us)")
    ax2.set_xlabel("Message Index")
    ax2.set_title(f"With Privacy Plugin - Per-Message Latency (w={w})")
    ax2.axhline(y=statistics.mean(p_lat), color="red", linestyle="--", alpha=0.5, label=f"mean={statistics.mean(p_lat):.1f}us")
    _add_window_boundaries(ax2, plugin, w)
    ax2.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "factory_latency_timeseries.png"), dpi=150)
    plt.close()
    print(f"    Saved: results/factory_latency_timeseries.png")

    # --- 3. Latency distribution ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(b_lat, bins=60, alpha=0.6, label="Baseline", color="#2196F3", density=True)
    ax.hist(p_lat, bins=60, alpha=0.6, label="With Privacy Plugin", color="#FF5722", density=True)
    ax.set_xlabel("Latency (microseconds)")
    ax.set_ylabel("Density")
    ax.set_title("Latency Distribution")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "factory_latency_distribution.png"), dpi=150)
    plt.close()
    print(f"    Saved: results/factory_latency_distribution.png")

    # --- 4. Payload distortion per sensor type ---
    sensor_types = sorted(set(r["sensor_type"] for r in plugin))
    fig, axes = plt.subplots(1, len(sensor_types), figsize=(4 * len(sensor_types), 4), sharey=False)
    if len(sensor_types) == 1:
        axes = [axes]
    colors = ["#E91E63", "#9C27B0", "#3F51B5", "#009688", "#FF9800"]
    for ax, stype, color in zip(axes, sensor_types, colors):
        subset = [r for r in plugin if r["sensor_type"] == stype]
        orig = [r["payload_original"] for r in subset]
        pert = [r["payload_output"] for r in subset]
        ax.scatter(orig, pert, alpha=0.3, s=8, color=color)
        lims = [min(orig + pert), max(orig + pert)]
        ax.plot(lims, lims, "k--", alpha=0.3, linewidth=0.8)
        ax.set_xlabel("Original")
        ax.set_ylabel("Perturbed")
        ax.set_title(stype.replace("_", " ").title(), fontsize=10)
    plt.suptitle("Payload Distortion by Sensor Type", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "factory_payload_distortion.png"), dpi=150)
    plt.close()
    print(f"    Saved: results/factory_payload_distortion.png")

    # --- 5. k-anonymity achieved over time ---
    k_vals = [r.get("consistent_publishers", 0) for r in plugin]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(len(k_vals)), k_vals, alpha=0.7, linewidth=0.8, color="#4CAF50")
    ax.set_xlabel("Message Index")
    ax.set_ylabel("k (consistent publishers)")
    ax.set_title(f"Publisher Privacy: k-Anonymity Achieved per Message (w={w})")
    ax.axhline(y=statistics.mean(k_vals), color="red", linestyle="--", alpha=0.5,
               label=f"mean k={statistics.mean(k_vals):.1f}")
    _add_window_boundaries(ax, plugin, w)
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "factory_k_anonymity.png"), dpi=150)
    plt.close()
    print(f"    Saved: results/factory_k_anonymity.png")

    # --- 6. Budget remaining & used over time ---
    budget_remaining = [r.get("budget_remaining", 0) for r in plugin]
    budget_used = [r.get("budget_used", 0) for r in plugin]
    win_pos = [r.get("window_position", 0) for r in plugin]

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    ax_top.plot(range(len(budget_remaining)), budget_remaining, alpha=0.7, linewidth=0.8, color="#FF9800",
                label="Budget remaining")
    ax_top.plot(range(len(budget_used)), budget_used, alpha=0.7, linewidth=0.8, color="#E91E63",
                label="Budget used")
    ax_top.axhline(y=epsilon, color="green", linestyle="--", alpha=0.5, label=f"epsilon={epsilon}")
    _add_window_boundaries(ax_top, plugin, w)
    ax_top.set_ylabel("Budget (epsilon)")
    ax_top.set_title(f"Content Privacy: Per-Publisher Budget over Windows (w={w}, e={epsilon})")
    ax_top.legend(fontsize=7)

    ax_bot.bar(range(len(win_pos)), win_pos, color="#42A5F5", alpha=0.6, width=1.0)
    _add_window_boundaries(ax_bot, plugin, w)
    ax_bot.set_xlabel("Message Index")
    ax_bot.set_ylabel("Position in Window")
    ax_bot.set_title(f"Window Position per Message (resets every {w} messages per publisher)")
    ax_bot.set_ylim(0, w + 1)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "factory_budget_remaining.png"), dpi=150)
    plt.close()
    print(f"    Saved: results/factory_budget_remaining.png")

    # --- 7. KL divergence per sensor type ---
    sensor_types = sorted(set(r["sensor_type"] for r in plugin))
    kl_values = {}
    for stype in sensor_types:
        b_sub = [r["payload_output"] for r in baseline if r["sensor_type"] == stype]
        p_sub = [r["payload_output"] for r in plugin if r["sensor_type"] == stype]
        kl_values[stype] = compute_kl_divergence(b_sub, p_sub)

    overall_orig = [r["payload_output"] for r in baseline]
    overall_pert = [r["payload_output"] for r in plugin]
    overall_kl = compute_kl_divergence(overall_orig, overall_pert)

    fig, ax = plt.subplots(figsize=(9, 5))
    labels_kl = [s.replace("_", " ").title() for s in sensor_types] + ["Overall"]
    vals_kl = [kl_values[s] for s in sensor_types] + [overall_kl]
    bar_colors = colors[:len(sensor_types)] + ["#607D8B"]
    bars = ax.bar(labels_kl, vals_kl, color=bar_colors, alpha=0.8)
    for bar, val in zip(bars, vals_kl):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("KL Divergence (bits)")
    ax.set_title(f"KL Divergence: Original vs. Plugin Output (ε={epsilon}, w={w})")
    ax.set_xlabel("Sensor Type")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "factory_kl_divergence.png"), dpi=150)
    plt.close()
    print(f"    Saved: results/factory_kl_divergence.png")

    # --- 8. Latency by sensor type box plot ---
    fig, ax = plt.subplots(figsize=(10, 5))
    sensor_data = {}
    for r in plugin:
        sensor_data.setdefault(r["sensor_type"], []).append(r["latency_us"])
    labels_sorted = sorted(sensor_data.keys())
    box_data = [sensor_data[s] for s in labels_sorted]
    bp = ax.boxplot(box_data, tick_labels=[s.replace("_", "\n") for s in labels_sorted], patch_artist=True)
    for patch, color in zip(bp["boxes"], colors[:len(labels_sorted)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("Latency (microseconds)")
    ax.set_title("Plugin Latency by Sensor Type")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "factory_latency_by_sensor.png"), dpi=150)
    plt.close()
    print(f"    Saved: results/factory_latency_by_sensor.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Smart Factory Privacy Benchmark (self-contained, no MQTT broker needed)"
    )
    parser.add_argument("-n", "--messages", type=int, default=500, help="Total messages to simulate (default: 500)")
    parser.add_argument("--publishers", type=int, default=10, help="Number of publisher companies (max 20, default: 10)")
    parser.add_argument("--subscribers", type=int, default=5, help="Number of subscriber profiles (default: 5)")

    parser.add_argument("--epsilon", type=float, default=1.0, help="DP budget epsilon (default: 1.0)")
    parser.add_argument("--window", type=int, default=10, help="Rolling window w (default: 10)")
    parser.add_argument("--strategy", default="uniform", choices=["uniform", "sample", "budget_absorption"])
    parser.add_argument("--k", type=int, default=2, help="k-anonymity (default: 2)")
    parser.add_argument("--l", type=int, default=2, help="l-diversity (default: 2)")
    parser.add_argument("--t", type=float, default=float("inf"), help="t-closeness (default: inf)")

    args = parser.parse_args()
    n_pub = min(args.publishers, len(PUBLISHER_COMPANIES))
    n_sub = min(args.subscribers, len(SUBSCRIBER_PROFILES))

    print(f"\n{'='*70}")
    print(f"  Smart Factory Privacy Benchmark")
    print(f"{'='*70}")
    print(f"  Messages:    {args.messages}")
    print(f"  Publishers:  {n_pub} companies")
    print(f"  Subscribers: {n_sub} profiles")
    print(f"  Sensors:     {sum(len(c['topics']) for c in SENSOR_TYPES.values())} topics across {len(SENSOR_TYPES)} types")
    print(f"  Privacy:     epsilon={args.epsilon}, w={args.window}, k={args.k}, l={args.l}, strategy={args.strategy}")
    print(f"{'='*70}")

    # Generate sensor stream
    print("\n  Generating factory sensor stream...")
    rng = np.random.default_rng(42)
    messages = generate_factory_stream(args.messages, n_pub, rng)
    subscribers = SUBSCRIBER_PROFILES[:n_sub]

    # Print stream summary
    by_type = {}
    by_pub = {}
    for m in messages:
        by_type[m.sensor_type] = by_type.get(m.sensor_type, 0) + 1
        by_pub[m.publisher_id] = by_pub.get(m.publisher_id, 0) + 1

    print(f"\n  Stream composition:")
    for stype, count in sorted(by_type.items()):
        print(f"    {stype:<15} {count:>5} messages")
    print(f"\n  Publisher distribution:")
    for pub, count in sorted(by_pub.items(), key=lambda x: -x[1])[:5]:
        print(f"    {pub:<20} {count:>5} messages")
    if len(by_pub) > 5:
        print(f"    ... and {len(by_pub) - 5} more publishers")

    # Run benchmarks
    print(f"\n  Running benchmarks...")
    baseline, plugin = run_benchmark(
        messages=messages,
        subscribers=subscribers,
        epsilon=args.epsilon,
        w=args.window,
        strategy=args.strategy,
        k=args.k,
        l_div=args.l,
        t_close=args.t,
    )

    # Stats
    print_stats("Baseline (no privacy)", baseline)
    print_stats("With Privacy Plugin", plugin)
    print_privacy_stats(plugin)

    # Overhead
    b_mean = statistics.mean([r["latency_us"] for r in baseline])
    p_mean = statistics.mean([r["latency_us"] for r in plugin])
    print(f"\n  Overhead: {p_mean - b_mean:.2f} us/message ({p_mean/b_mean:.1f}x)")

    # Save CSVs
    print(f"\n  Saving results...")
    save_csv(baseline, "factory_baseline.csv")
    save_csv(plugin, "factory_plugin.csv")

    combined = []
    for b, p in zip(baseline, plugin):
        combined.append({
            "message_index": b["message_index"],
            "publisher_id": b["publisher_id"],
            "sensor_type": b["sensor_type"],
            "baseline_latency_us": b["latency_us"],
            "plugin_latency_us": p["latency_us"],
            "overhead_us": p["latency_us"] - b["latency_us"],
            "payload_original": b["payload_original"],
            "payload_baseline": b["payload_output"],
            "payload_plugin": p["payload_output"],
            "topic_original": p["topic_original"],
            "topic_generalized": p["topic_output"],
            "k_achieved": p.get("consistent_publishers", 0),
            "budget_remaining": p.get("budget_remaining", 0),
            "window_id": p.get("window_id", 0),
            "window_position": p.get("window_position", 0),
            "budget_used": p.get("budget_used", 0),
        })
    save_csv(combined, "factory_comparison.csv")

    # Graphs
    print(f"\n  Generating graphs...")
    generate_graphs(baseline, plugin, w=args.window, epsilon=args.epsilon)

    print(f"\n{'='*70}")
    print(f"  Done! All results in {RESULTS_DIR}/")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
