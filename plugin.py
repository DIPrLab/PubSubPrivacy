"""
MQTT Privacy Plugin — broker-side middleware that intercepts raw publisher
messages, constructs per-subscription aggregate streams, applies S-sensitive
w-event differential privacy, and republishes protected outputs.

Architecture:
  Publishers --> [raw topics] --> PrivacyPlugin --> [protected topics] --> Subscribers

The plugin subscribes to all raw topics, buffers messages per timestamp
interval, computes the mean aggregate, injects calibrated Laplace noise,
and publishes the noisy aggregate on the corresponding protected topic.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field

import paho.mqtt.client as mqtt

from dp_engine import BudgetStrategy, PrivacyConfig, StreamState

logger = logging.getLogger(__name__)


@dataclass
class TopicBuffer:
    payloads: dict[str, float] = field(default_factory=dict)  # publisher_id -> value

    @property
    def num_publishers(self) -> int:
        return len(self.payloads)

    @property
    def mean(self) -> float:
        if not self.payloads:
            return 0.0
        return sum(self.payloads.values()) / len(self.payloads)

    def add(self, publisher_id: str, value: float):
        self.payloads[publisher_id] = value

    def clear(self):
        self.payloads.clear()


class PrivacyPlugin:
    def __init__(
        self,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        raw_prefix: str = "factory/raw",
        protected_prefix: str = "factory/protected",
        epsilon: float = 1.0,
        window_size: int = 10,
        min_publishers: int = 3,
        strategy: str = "uniform",
        timestamp_interval: float = 5.0,
        sensor_bounds: dict[str, tuple[float, float]] | None = None,
    ):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.raw_prefix = raw_prefix
        self.protected_prefix = protected_prefix
        self.epsilon = epsilon
        self.window_size = window_size
        self.min_publishers = min_publishers
        self.strategy = BudgetStrategy(strategy)
        self.timestamp_interval = timestamp_interval
        self.sensor_bounds = sensor_bounds or {}

        # Per-topic state
        self._buffers: dict[str, TopicBuffer] = defaultdict(TopicBuffer)
        self._streams: dict[str, StreamState] = {}
        self._lock = threading.Lock()

        # MQTT client
        self._client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id="privacy-plugin",
        )
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message

        self._running = False
        self._timer_thread: threading.Thread | None = None

        # Collected release log for experiment analysis
        self.release_log: list[dict] = []

    def _get_payload_bound(self, topic: str) -> float:
        """Determine B (payload range) for a topic based on sensor type."""
        for sensor_type, (lo, hi) in self.sensor_bounds.items():
            if sensor_type in topic:
                return hi - lo
        return 100.0

    def _get_or_create_stream(self, topic: str) -> StreamState:
        if topic not in self._streams:
            payload_bound = self._get_payload_bound(topic)
            config = PrivacyConfig(
                epsilon=self.epsilon,
                window_size=self.window_size,
                min_publishers=self.min_publishers,
                payload_bound=payload_bound,
                strategy=self.strategy,
            )
            self._streams[topic] = StreamState(config=config)
        return self._streams[topic]

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        subscribe_topic = f"{self.raw_prefix}/#"
        client.subscribe(subscribe_topic)
        logger.info(f"Privacy plugin subscribed to {subscribe_topic}")

    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            publisher_id = payload.get("publisher_id", "unknown")
            value = float(payload.get("value", 0.0))
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Malformed message on {msg.topic}: {e}")
            return

        logical_topic = msg.topic
        if logical_topic.startswith(self.raw_prefix + "/"):
            logical_topic = logical_topic[len(self.raw_prefix) + 1:]

        with self._lock:
            self._buffers[logical_topic].add(publisher_id, value)

    def _flush_and_release(self):
        with self._lock:
            topics_to_process = list(self._buffers.keys())

        for logical_topic in topics_to_process:
            with self._lock:
                buf = self._buffers[logical_topic]
                num_pub = buf.num_publishers
                aggregate = buf.mean
                buf.clear()

            stream = self._get_or_create_stream(logical_topic)
            released = stream.release(aggregate, num_pub)

            if released is not None:
                protected_topic = f"{self.protected_prefix}/{logical_topic}"
                out = {
                    "timestamp": stream.current_tau,
                    "value": round(released, 4),
                    "num_publishers": num_pub,
                    "suppressed": num_pub < self.min_publishers,
                }
                self._client.publish(protected_topic, json.dumps(out))
                logger.debug(
                    f"[tau={stream.current_tau}] {logical_topic}: "
                    f"true={aggregate:.4f} noisy={released:.4f} "
                    f"pubs={num_pub}"
                )

            self.release_log.append({
                "topic": logical_topic,
                "tau": stream.current_tau,
                "true_aggregate": aggregate,
                "released_value": released,
                "num_publishers": num_pub,
                "suppressed": num_pub < self.min_publishers,
            })

    def _timer_loop(self):
        while self._running:
            time.sleep(self.timestamp_interval)
            if self._running:
                self._flush_and_release()

    def start(self):
        self._client.connect(self.broker_host, self.broker_port)
        self._client.loop_start()
        self._running = True
        self._timer_thread = threading.Thread(target=self._timer_loop, daemon=True)
        self._timer_thread.start()
        logger.info(
            f"Privacy plugin started (epsilon={self.epsilon}, w={self.window_size}, "
            f"S={self.min_publishers}, strategy={self.strategy.value})"
        )

    def stop(self):
        self._running = False
        if self._timer_thread:
            self._timer_thread.join(timeout=self.timestamp_interval * 2)
        self._client.loop_stop()
        self._client.disconnect()
        logger.info("Privacy plugin stopped")

    def get_stream_states(self) -> dict[str, StreamState]:
        return dict(self._streams)
