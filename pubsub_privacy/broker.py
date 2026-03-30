"""
Privacy Broker - Two-Layer Privacy Architecture

Combines content privacy (input side) and publisher privacy (output side)
into a single broker transformation T(m) as described in Section 4.6.

Pipeline:
  1. Message arrives -> Content privacy perturbs payload (input side, per-publisher)
  2. Message stored in rolling window
  3. At delivery -> Publisher privacy generalizes topic (output side, per-subscriber)

The two layers compose without interference: payload perturbation does not
alter publisher identities or topics, and topic generalization does not
alter the already-perturbed payloads.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from pubsub_privacy.content_privacy import BudgetStrategy, ContentPrivacyEngine
from pubsub_privacy.publisher_privacy import PublisherPrivacyEngine, WindowMessage


@dataclass
class PubSubMessage:
    """A message tuple m = (p, x, s, tau) per Section 3."""
    publisher_id: str       # p: publisher identity
    payload: float          # x: numeric payload
    topic: str              # s: topic in hierarchy
    timestamp: float        # tau: message timestamp
    sensitive_attr: Optional[str] = None  # A(m): sensitive publisher attribute


@dataclass
class RewrittenMessage:
    """Output y = T(m) after both privacy layers."""
    original_topic: str
    generalized_topic: str
    original_payload: float
    perturbed_payload: float
    timestamp: float
    consistent_publishers: set[str] = field(default_factory=set)
    budget_info: dict = field(default_factory=dict)


class PrivacyBroker:
    """
    Two-layer privacy broker for publish-subscribe systems.

    Layer 1 (Input/Content Privacy): w-event epsilon-DP via Laplace noise.
    Layer 2 (Output/Publisher Privacy): k-anonymity via topic generalization.

    Parameters:
        epsilon: DP budget for content privacy.
        w: Rolling window size (shared by both layers).
        sensitivity: Payload sensitivity Delta.
        strategy: Budget allocation strategy for content privacy.
        ba_threshold: BA strategy skip threshold.
        k: k-anonymity parameter for publisher privacy.
        l: l-diversity parameter (0 to disable).
        t: t-closeness threshold (inf to disable).
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        w: int = 10,
        sensitivity: float = 1.0,
        strategy: BudgetStrategy = BudgetStrategy.UNIFORM,
        ba_threshold: float = 1.0,
        k: int = 2,
        l: int = 0,
        t: float = float("inf"),
    ):
        self.content_engine = ContentPrivacyEngine(
            epsilon=epsilon,
            w=w,
            sensitivity=sensitivity,
            strategy=strategy,
            ba_threshold=ba_threshold,
        )
        self.publisher_engine = PublisherPrivacyEngine(k=k, l=l, t=t, w=w)
        self.w = w
        self._message_count = 0

    def process_message(self, msg: PubSubMessage) -> RewrittenMessage:
        """
        Apply the full two-layer privacy transformation T(m).

        Step 1 (Ingress): Perturb payload via w-event DP (content privacy).
        Step 2: Add to rolling window for publisher privacy tracking.
        Step 3 (Egress): Generalize topic for k-anonymity (publisher privacy).

        Returns:
            RewrittenMessage with perturbed payload and generalized topic.
        """
        self._message_count += 1

        # Layer 1: Content privacy (input side)
        perturbed_payload = self.content_engine.perturb(
            msg.publisher_id, msg.payload
        )

        # Add to publisher privacy window
        window_msg = WindowMessage(
            position=self._message_count,
            publisher_id=msg.publisher_id,
            topic=msg.topic,
            sensitive_attr=msg.sensitive_attr,
            payload=perturbed_payload,
            timestamp=str(msg.timestamp),
        )
        self.publisher_engine.add_message(window_msg)

        # Layer 2: Publisher privacy (output side)
        generalized_topic = self.publisher_engine.generalize_topic(msg.topic)
        consistent_pubs = self.publisher_engine.get_consistent_publishers(
            generalized_topic
        )

        budget_info = self.content_engine.get_budget_usage(msg.publisher_id)

        return RewrittenMessage(
            original_topic=msg.topic,
            generalized_topic=generalized_topic,
            original_payload=msg.payload,
            perturbed_payload=perturbed_payload,
            timestamp=msg.timestamp,
            consistent_publishers=consistent_pubs,
            budget_info=budget_info,
        )

    def process_raw(
        self,
        publisher_id: str,
        payload: float,
        topic: str,
        sensitive_attr: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> RewrittenMessage:
        """Convenience method to process a message from raw fields."""
        msg = PubSubMessage(
            publisher_id=publisher_id,
            payload=payload,
            topic=topic,
            timestamp=timestamp or time.time(),
            sensitive_attr=sensitive_attr,
        )
        return self.process_message(msg)

    def get_state(self) -> dict[str, Any]:
        """Return broker state for debugging/monitoring."""
        return {
            "messages_processed": self._message_count,
            "window_state": self.publisher_engine.get_window_state(),
        }
