"""
Publisher Privacy Layer (Output Side)

Implements rolling k-anonymity, l-diversity, and t-closeness via topic
generalization at broker egress, per Section 4.3 of the paper.

The broker maintains a rolling window of recent messages. At delivery time,
it generalizes each message's topic by walking up the topic hierarchy until
at least k distinct publishers share the rewritten topic within the window.

The topic hierarchy is modeled as a rooted tree where '/' separates levels:
  traffic/downtown/elm/1st  ->  traffic/downtown/elm/*  ->  traffic/downtown/*  ->  traffic/*  ->  *
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WindowMessage:
    """A message stored in the rolling window."""
    position: int
    publisher_id: str
    topic: str
    sensitive_attr: Optional[str] = None
    payload: float = 0.0
    timestamp: str = ""


class TopicHierarchy:
    """Utilities for MQTT-style topic hierarchies."""

    @staticmethod
    def generalize(topic: str, levels: int = 1) -> str:
        """Walk up the topic tree by replacing trailing segments with '*'."""
        parts = topic.split("/")
        if levels >= len(parts):
            return "*"
        return "/".join(parts[: len(parts) - levels]) + "/*"

    @staticmethod
    def all_generalizations(topic: str) -> list[str]:
        """Return all possible generalizations from most specific to root."""
        parts = topic.split("/")
        results = [topic]
        for i in range(1, len(parts)):
            results.append("/".join(parts[: len(parts) - i]) + "/*")
        results.append("*")
        return results

    @staticmethod
    def matches(topic: str, pattern: str) -> bool:
        """Check if a topic matches a subscription pattern (with wildcards)."""
        if pattern == "*" or pattern == "#":
            return True
        t_parts = topic.split("/")
        p_parts = pattern.split("/")

        for i, p in enumerate(p_parts):
            if p == "#":
                return True
            if p == "*" or p == "+":
                if i >= len(t_parts):
                    return False
                continue
            if i >= len(t_parts) or t_parts[i] != p:
                return False
        return len(t_parts) == len(p_parts) or (
            len(p_parts) > 0 and p_parts[-1] in ("*", "#")
        )


class PublisherPrivacyEngine:
    """
    Rolling k-publisher privacy with optional l-diversity and t-closeness.

    At egress, the broker generalizes each message's topic until at least k
    distinct publishers share the rewritten topic within the current rolling
    window. If l-diversity is enabled, the generalization continues until at
    least l distinct sensitive attribute values appear. If t-closeness is
    enabled, it checks that the distribution of sensitive attributes in the
    equivalence class is within distance t of the global distribution.

    Parameters:
        k: Minimum number of distinct publishers per output topic (k-anonymity).
        l: Minimum number of distinct sensitive attribute values (l-diversity). 0 to disable.
        t: Maximum distance for t-closeness. float('inf') to disable.
        w: Rolling window size.
    """

    def __init__(
        self,
        k: int = 2,
        l: int = 0,
        t: float = float("inf"),
        w: int = 10,
    ):
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k
        self.l = l
        self.t_closeness = t
        self.w = w
        self._window: list[WindowMessage] = []
        self._position: int = 0

    def add_message(self, msg: WindowMessage) -> None:
        """Add a message to the rolling window."""
        self._position += 1
        msg.position = self._position
        self._window.append(msg)
        # Trim to window size
        if len(self._window) > self.w:
            self._window = self._window[-self.w:]

    def _publishers_for_generalized_topic(self, gen_topic: str) -> set[str]:
        """Find all distinct publishers in the window whose topic matches."""
        publishers = set()
        for msg in self._window:
            if TopicHierarchy.matches(msg.topic, gen_topic):
                publishers.add(msg.publisher_id)
        return publishers

    def _sensitive_attrs_for_topic(self, gen_topic: str) -> list[Optional[str]]:
        """Collect sensitive attribute values for messages matching the generalized topic."""
        return [
            msg.sensitive_attr
            for msg in self._window
            if TopicHierarchy.matches(msg.topic, gen_topic)
        ]

    def _check_t_closeness(self, gen_topic: str) -> bool:
        """Check t-closeness constraint for a generalized topic."""
        if self.t_closeness == float("inf"):
            return True

        # Global distribution of sensitive attrs in window
        all_attrs = [msg.sensitive_attr for msg in self._window if msg.sensitive_attr is not None]
        if not all_attrs:
            return True

        group_attrs = [
            msg.sensitive_attr
            for msg in self._window
            if TopicHierarchy.matches(msg.topic, gen_topic) and msg.sensitive_attr is not None
        ]
        if not group_attrs:
            return True

        # Compute variational distance (EMD for categorical)
        global_counts: dict[str, float] = defaultdict(float)
        for a in all_attrs:
            global_counts[a] += 1.0 / len(all_attrs)

        group_counts: dict[str, float] = defaultdict(float)
        for a in group_attrs:
            group_counts[a] += 1.0 / len(group_attrs)

        all_keys = set(global_counts.keys()) | set(group_counts.keys())
        distance = 0.5 * sum(abs(global_counts[k] - group_counts[k]) for k in all_keys)

        return distance <= self.t_closeness

    def generalize_topic(self, original_topic: str) -> str:
        """
        Find the minimal generalization of the topic that satisfies
        k-anonymity (and optionally l-diversity, t-closeness) within
        the current rolling window.

        Returns the generalized topic string.
        """
        candidates = TopicHierarchy.all_generalizations(original_topic)

        for gen_topic in candidates:
            publishers = self._publishers_for_generalized_topic(gen_topic)
            if len(publishers) < self.k:
                continue

            # Check l-diversity
            if self.l > 0:
                attrs = set(self._sensitive_attrs_for_topic(gen_topic))
                attrs.discard(None)
                if len(attrs) < self.l:
                    continue

            # Check t-closeness
            if not self._check_t_closeness(gen_topic):
                continue

            return gen_topic

        # Fallback: return root wildcard (maximum generalization)
        return "*"

    def get_window_state(self) -> list[dict]:
        """Return current window contents for inspection."""
        return [
            {
                "position": m.position,
                "publisher": m.publisher_id,
                "topic": m.topic,
                "sensitive_attr": m.sensitive_attr,
            }
            for m in self._window
        ]

    def get_consistent_publishers(self, generalized_topic: str) -> set[str]:
        """Return the set of publishers consistent with a generalized topic in the window."""
        return self._publishers_for_generalized_topic(generalized_topic)
