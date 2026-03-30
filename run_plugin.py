#!/usr/bin/env python3
"""
Entry point to run the MQTT Privacy Plugin.

Usage:
    python run_plugin.py [options]

This connects to an MQTT broker, subscribes to raw/# topics, applies the
two-layer privacy architecture (content DP + publisher k-anonymity), and
republishes sanitized messages under private/#.
"""

import argparse
import logging
import sys

from pubsub_privacy.content_privacy import BudgetStrategy
from pubsub_privacy.mqtt_plugin import MQTTPrivacyPlugin


def main():
    parser = argparse.ArgumentParser(
        description="MQTT Privacy Plugin - Two-layer privacy for pub/sub systems"
    )

    # MQTT connection
    parser.add_argument("--host", default="localhost", help="MQTT broker host (default: localhost)")
    parser.add_argument("--port", type=int, default=1883, help="MQTT broker port (default: 1883)")
    parser.add_argument("--source", default="raw/#", help="Source topic pattern (default: raw/#)")
    parser.add_argument("--dest", default="private", help="Destination topic prefix (default: private)")

    # Content privacy (Layer 1 - Input Side)
    parser.add_argument("--epsilon", type=float, default=1.0, help="DP privacy budget epsilon (default: 1.0)")
    parser.add_argument("--window", type=int, default=10, help="Rolling window size w (default: 10)")
    parser.add_argument("--sensitivity", type=float, default=100.0, help="Payload sensitivity Delta (default: 100.0)")
    parser.add_argument(
        "--strategy",
        choices=["uniform", "sample", "budget_absorption"],
        default="uniform",
        help="Budget allocation strategy (default: uniform)",
    )
    parser.add_argument("--ba-threshold", type=float, default=1.0, help="BA skip threshold (default: 1.0)")

    # Publisher privacy (Layer 2 - Output Side)
    parser.add_argument("--k", type=int, default=2, help="k-anonymity parameter (default: 2)")
    parser.add_argument("--l", type=int, default=0, help="l-diversity parameter, 0=disabled (default: 0)")
    parser.add_argument("--t", type=float, default=float("inf"), help="t-closeness threshold, inf=disabled")

    # General
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    strategy = BudgetStrategy(args.strategy)

    plugin = MQTTPrivacyPlugin(
        broker_host=args.host,
        broker_port=args.port,
        source_topic=args.source,
        dest_prefix=args.dest,
        epsilon=args.epsilon,
        w=args.window,
        sensitivity=args.sensitivity,
        strategy=strategy,
        ba_threshold=args.ba_threshold,
        k=args.k,
        l=args.l,
        t=args.t,
    )

    plugin.start()


if __name__ == "__main__":
    main()
