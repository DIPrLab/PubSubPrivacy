"""
MQTT Privacy Plugin

Wraps the two-layer PrivacyBroker as an MQTT middleware. Connects to an
MQTT broker (e.g., Mosquitto), subscribes to raw publisher topics, applies
content privacy + publisher privacy transformations, and republishes the
sanitized messages on a privacy-prefixed topic.

Architecture:
  Publishers -> raw topics (e.g., traffic/downtown/elm/1st)
  Plugin subscribes to raw/#, applies T(m), republishes to private/<generalized_topic>
  Subscribers -> subscribe to private/# to receive sanitized messages
"""

from __future__ import annotations

import json
import logging
import time
from typing import Optional

import paho.mqtt.client as mqtt

from pubsub_privacy.broker import PrivacyBroker, PubSubMessage
from pubsub_privacy.content_privacy import BudgetStrategy

logger = logging.getLogger(__name__)


class MQTTPrivacyPlugin:
    """
    MQTT middleware plugin implementing the paper's two-layer privacy architecture.

    Subscribes to a source topic tree, applies content privacy (w-event DP)
    and publisher privacy (k-anonymity via topic generalization), then
    republishes sanitized messages under a destination prefix.

    Expected MQTT payload format (JSON):
    {
        "publisher_id": "p1",
        "value": 42.0,
        "sensitive_attr": "small"  // optional
    }

    Parameters:
        broker_host: MQTT broker hostname.
        broker_port: MQTT broker port.
        source_topic: Topic pattern to subscribe to (e.g., "raw/#").
        dest_prefix: Prefix for republished sanitized messages.
        epsilon, w, sensitivity, strategy, ba_threshold: Content privacy params.
        k, l, t: Publisher privacy params.
    """

    def __init__(
        self,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        source_topic: str = "raw/#",
        dest_prefix: str = "private",
        epsilon: float = 1.0,
        w: int = 10,
        sensitivity: float = 100.0,
        strategy: BudgetStrategy = BudgetStrategy.UNIFORM,
        ba_threshold: float = 1.0,
        k: int = 2,
        l: int = 0,
        t: float = float("inf"),
        client_id: str = "privacy_plugin",
    ):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.source_topic = source_topic
        self.dest_prefix = dest_prefix
        self.client_id = client_id

        self.privacy_broker = PrivacyBroker(
            epsilon=epsilon,
            w=w,
            sensitivity=sensitivity,
            strategy=strategy,
            ba_threshold=ba_threshold,
            k=k,
            l=l,
            t=t,
        )

        self._client: Optional[mqtt.Client] = None
        self._running = False

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            logger.info(f"Connected to MQTT broker at {self.broker_host}:{self.broker_port}")
            client.subscribe(self.source_topic)
            logger.info(f"Subscribed to: {self.source_topic}")
        else:
            logger.error(f"Connection failed with code {rc}")

    def _on_message(self, client, userdata, msg):
        try:
            raw_topic = msg.topic
            payload_str = msg.payload.decode("utf-8")

            try:
                data = json.loads(payload_str)
                publisher_id = data.get("publisher_id", "unknown")
                value = float(data.get("value", 0))
                sensitive_attr = data.get("sensitive_attr")
            except (json.JSONDecodeError, ValueError):
                # Try plain numeric payload
                publisher_id = "unknown"
                value = float(payload_str)
                sensitive_attr = None

            # Strip source prefix to get the actual topic hierarchy
            if raw_topic.startswith("raw/"):
                actual_topic = raw_topic[4:]
            else:
                actual_topic = raw_topic

            # Apply two-layer privacy transformation
            result = self.privacy_broker.process_raw(
                publisher_id=publisher_id,
                payload=value,
                topic=actual_topic,
                sensitive_attr=sensitive_attr,
                timestamp=time.time(),
            )

            # Build sanitized output
            output = {
                "value": round(result.perturbed_payload, 4),
                "generalized_topic": result.generalized_topic,
                "consistent_publishers": len(result.consistent_publishers),
                "timestamp": result.timestamp,
                "budget_remaining": round(
                    result.budget_info.get("window_budget_remaining", 0), 4
                ),
            }

            # Republish on privacy-prefixed generalized topic
            dest_topic = f"{self.dest_prefix}/{result.generalized_topic}"
            client.publish(dest_topic, json.dumps(output))

            logger.debug(
                f"[{raw_topic}] payload={value} -> "
                f"[{dest_topic}] perturbed={output['value']} "
                f"(k={output['consistent_publishers']})"
            )

        except Exception as e:
            logger.error(f"Error processing message on {msg.topic}: {e}")

    def start(self):
        """Start the MQTT privacy plugin (blocking)."""
        self._client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=self.client_id,
        )
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message

        logger.info(
            f"Starting MQTT Privacy Plugin\n"
            f"  Source: {self.source_topic}\n"
            f"  Destination prefix: {self.dest_prefix}\n"
            f"  Content Privacy: epsilon={self.privacy_broker.content_engine.epsilon}, "
            f"w={self.privacy_broker.w}, "
            f"strategy={self.privacy_broker.content_engine.strategy.value}\n"
            f"  Publisher Privacy: k={self.privacy_broker.publisher_engine.k}, "
            f"l={self.privacy_broker.publisher_engine.l}"
        )

        self._client.connect(self.broker_host, self.broker_port, keepalive=60)
        self._running = True
        try:
            self._client.loop_forever()
        except KeyboardInterrupt:
            logger.info("Shutting down privacy plugin...")
            self.stop()

    def stop(self):
        """Stop the plugin."""
        self._running = False
        if self._client:
            self._client.disconnect()
            self._client.loop_stop()
