"""
Content Privacy Layer (Input Side)

Implements w-event epsilon-differential privacy for publisher message streams.
Applied at broker ingress, independently per publisher stream.

Based on Kellaris et al. (2014) "Differentially Private Event Sequences
over Infinite Streams", adapted to per-message pub-sub delivery per
Section 4.2 of the paper.

Budget allocation strategies:
  - Uniform: epsilon_i = epsilon / w for every message
  - Sample: every w-th message gets full budget; others repeat last output
  - BA (Budget Absorption): skip messages whose payload barely changed,
    absorb saved budget for future high-change messages
"""

from __future__ import annotations

import enum
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


class BudgetStrategy(enum.Enum):
    UNIFORM = "uniform"
    SAMPLE = "sample"
    BA = "budget_absorption"


@dataclass
class PublisherState:
    """Tracks per-publisher rolling window budget state."""
    budgets: list[float] = field(default_factory=list)
    last_output: Optional[float] = None
    message_count: int = 0
    last_published_payload: Optional[float] = None
    window_id: int = 0
    window_position: int = 0


class ContentPrivacyEngine:
    """
    w-event epsilon-DP content privacy via Laplace noise injection.

    Each publisher's stream is independently sanitized. The budget
    constraint sum(epsilon_k for k in [i-w+1, i]) <= epsilon holds
    for every rolling window of w consecutive messages.

    Parameters:
        epsilon: Total privacy budget for any window of w messages.
        w: Rolling window size.
        sensitivity: Payload sensitivity Delta (max |x - x'|).
        strategy: Budget allocation strategy.
        ba_threshold: For BA strategy, skip if |payload - last| <= threshold.
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        w: int = 10,
        sensitivity: float = 1.0,
        strategy: BudgetStrategy = BudgetStrategy.UNIFORM,
        ba_threshold: float = 1.0,
    ):
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if w < 1:
            raise ValueError("w must be >= 1")
        if sensitivity <= 0:
            raise ValueError("sensitivity must be positive")

        self.epsilon = epsilon
        self.w = w
        self.sensitivity = sensitivity
        self.strategy = strategy
        self.ba_threshold = ba_threshold
        self._rng = np.random.default_rng()
        self._publisher_states: dict[str, PublisherState] = defaultdict(PublisherState)

    def _remaining_budget(self, state: PublisherState) -> float:
        """Budget remaining in the current rolling window."""
        recent = state.budgets[-(self.w - 1):] if len(state.budgets) > 0 else []
        return self.epsilon - sum(recent)

    def _allocate_uniform(self, state: PublisherState) -> float:
        return self.epsilon / self.w

    def _allocate_sample(self, state: PublisherState) -> tuple[float, bool]:
        """Returns (budget, should_publish). Publishes every w-th message."""
        state.message_count += 1
        if state.message_count % self.w == 1 or self.w == 1:
            return self.epsilon, True
        return 0.0, False

    def _allocate_ba(self, state: PublisherState, payload: float) -> tuple[float, bool]:
        """Budget Absorption: skip if payload barely changed from last published."""
        if state.last_published_payload is not None:
            if abs(payload - state.last_published_payload) <= self.ba_threshold:
                return 0.0, False

        remaining = self._remaining_budget(state)
        budget = min(remaining, self.epsilon / self.w)
        budget = max(budget, 1e-10)
        return budget, True

    def perturb(self, publisher_id: str, payload: float) -> float:
        """
        Apply w-event DP perturbation to a single message payload.

        Args:
            publisher_id: Unique publisher identifier.
            payload: Numeric payload value.

        Returns:
            Perturbed payload value with Laplace noise.
        """
        state = self._publisher_states[publisher_id]

        # Track window position; reset budget at window boundary
        state.window_position += 1
        if state.window_position > self.w:
            state.window_id += 1
            state.window_position = 1
            state.budgets.clear()
            state.last_output = None
            state.last_published_payload = None

        if self.strategy == BudgetStrategy.UNIFORM:
            eps_i = self._allocate_uniform(state)
            eps_i = min(eps_i, self._remaining_budget(state))
            if eps_i <= 0:
                eps_i = 1e-10
            scale = self.sensitivity / eps_i
            noise = self._rng.laplace(0, scale)
            output = payload + noise
            state.budgets.append(eps_i)
            state.last_output = output
            state.last_published_payload = payload

        elif self.strategy == BudgetStrategy.SAMPLE:
            eps_i, should_publish = self._allocate_sample(state)
            if should_publish:
                remaining = self._remaining_budget(state)
                eps_i = min(eps_i, remaining)
                if eps_i <= 0:
                    eps_i = 1e-10
                scale = self.sensitivity / eps_i
                noise = self._rng.laplace(0, scale)
                output = payload + noise
                state.last_output = output
                state.last_published_payload = payload
            else:
                output = state.last_output if state.last_output is not None else payload
            state.budgets.append(eps_i)

        elif self.strategy == BudgetStrategy.BA:
            eps_i, should_publish = self._allocate_ba(state, payload)
            if should_publish:
                remaining = self._remaining_budget(state)
                eps_i = min(eps_i, remaining)
                if eps_i <= 0:
                    eps_i = 1e-10
                scale = self.sensitivity / eps_i
                noise = self._rng.laplace(0, scale)
                output = payload + noise
                state.last_output = output
                state.last_published_payload = payload
            else:
                output = state.last_output if state.last_output is not None else payload
                eps_i = 0.0
            state.budgets.append(eps_i)

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        # Trim budget history to window size
        if len(state.budgets) > self.w:
            state.budgets = state.budgets[-self.w:]

        return output

    def get_budget_usage(self, publisher_id: str) -> dict:
        """Return current budget state for a publisher."""
        state = self._publisher_states[publisher_id]
        recent = state.budgets[-(self.w):] if state.budgets else []
        return {
            "publisher_id": publisher_id,
            "window_budget_used": sum(recent),
            "window_budget_remaining": self.epsilon - sum(recent),
            "epsilon": self.epsilon,
            "w": self.w,
            "strategy": self.strategy.value,
            "messages_in_window": len(recent),
            "window_id": state.window_id,
            "window_position": state.window_position,
        }
