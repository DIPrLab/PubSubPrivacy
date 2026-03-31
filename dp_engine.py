"""
Core differential privacy engine implementing S-sensitive w-event DP.

Implements the Laplace mechanism with sliding window budget constraint
and three budget allocation strategies: Uniform, Sample, Budget Absorption.
Based on Kellaris et al. (2014) extended to the pub/sub aggregate stream model.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class BudgetStrategy(Enum):
    UNIFORM = "uniform"
    SAMPLE = "sample"
    BUDGET_ABSORPTION = "budget_absorption"


@dataclass
class PrivacyConfig:
    epsilon: float  # Global privacy budget for each window of w timestamps
    window_size: int  # w: number of timestamps in the sliding window
    min_publishers: int  # S: minimum publishers required per aggregate element
    payload_bound: float  # B: range of the payload domain (max - min)
    strategy: BudgetStrategy = BudgetStrategy.UNIFORM
    # Budget Absorption: threshold for skipping an element (fraction of sensitivity)
    ba_threshold: float = 0.1

    @property
    def sensitivity(self) -> float:
        return self.payload_bound / self.min_publishers

    def noise_scale(self, epsilon_tau: float) -> float:
        if epsilon_tau <= 0:
            return float("inf")
        return self.sensitivity / epsilon_tau


@dataclass
class StreamState:
    config: PrivacyConfig
    # Rolling window of per-element budgets spent
    budget_window: deque = field(default_factory=deque)
    # Timestamp counter
    current_tau: int = 0
    # Last released (noisy) value — used for Sample/BA skip logic
    last_released: Optional[float] = None
    # Last true aggregate — used for BA change detection
    last_true_aggregate: Optional[float] = None
    # Budget carried forward
    absorbed_budget: float = 0.0
    # Running history for utility tracking
    true_values: list = field(default_factory=list)
    noisy_values: list = field(default_factory=list)
    budgets_spent: list = field(default_factory=list)

    def __post_init__(self):
        self.budget_window = deque(maxlen=self.config.window_size)

    def _budget_spent_in_window(self) -> float:
        return sum(self.budget_window)

    def _budget_remaining(self) -> float:
        return self.config.epsilon - self._budget_spent_in_window()

    def _allocate_budget_uniform(self) -> float:
        return self.config.epsilon / self.config.window_size

    def _allocate_budget_sample(self) -> tuple[float, bool]:
        if self.current_tau % self.config.window_size == 0:
            return self.config.epsilon, False  # release with full budget
        return 0.0, True  # skip

    def _allocate_budget_absorption(self, aggregate: float) -> tuple[float, bool]:
        if self.last_true_aggregate is not None:
            change = abs(aggregate - self.last_true_aggregate)
            threshold = self.config.ba_threshold * self.config.sensitivity
            if change < threshold:
                budget_uniform = self.config.epsilon / self.config.window_size
                self.absorbed_budget += budget_uniform
                return 0.0, True

        # Release: use uniform share plus any absorbed budget
        budget = self.config.epsilon / self.config.window_size + self.absorbed_budget
        # Cap at remaining window budget
        budget = min(budget, self._budget_remaining())
        self.absorbed_budget = 0.0
        return budget, False

    def release(self, aggregate: float, num_publishers: int) -> Optional[float]:
        self.current_tau += 1

        # Enforce minimum publisher count
        if num_publishers < self.config.min_publishers:
            # Buffer / suppress — append zero budget to maintain window alignment
            self.budget_window.append(0.0)
            self.true_values.append(aggregate)
            self.noisy_values.append(self.last_released)
            self.budgets_spent.append(0.0)
            return self.last_released

        # Determine budget allocation and whether to skip
        skip = False
        if self.config.strategy == BudgetStrategy.UNIFORM:
            epsilon_tau = self._allocate_budget_uniform()
        elif self.config.strategy == BudgetStrategy.SAMPLE:
            epsilon_tau, skip = self._allocate_budget_sample()
        elif self.config.strategy == BudgetStrategy.BUDGET_ABSORPTION:
            epsilon_tau, skip = self._allocate_budget_absorption(aggregate)
        else:
            raise ValueError(f"Unknown strategy: {self.config.strategy}")

        # Enforce window budget constraint
        remaining = self._budget_remaining()
        epsilon_tau = min(epsilon_tau, remaining)

        if skip or epsilon_tau <= 0:
            self.budget_window.append(0.0)
            self.true_values.append(aggregate)
            self.noisy_values.append(self.last_released)
            self.budgets_spent.append(0.0)
            return self.last_released

        # Apply Laplace mechanism
        noise_scale = self.config.noise_scale(epsilon_tau)
        noise = np.random.laplace(loc=0.0, scale=noise_scale)
        released = aggregate + noise

        # Track state
        self.budget_window.append(epsilon_tau)
        self.last_released = released
        self.last_true_aggregate = aggregate
        self.true_values.append(aggregate)
        self.noisy_values.append(released)
        self.budgets_spent.append(epsilon_tau)

        return released


def compute_utility_metrics(
    true_values: list[float],
    noisy_values: list[float],
) -> dict:
    pairs = [
        (t, n)
        for t, n in zip(true_values, noisy_values)
        if t is not None and n is not None
    ]
    if not pairs:
        return {"mae": float("nan"), "rmse": float("nan"), "relative_error": float("nan")}

    true_arr = np.array([p[0] for p in pairs])
    noisy_arr = np.array([p[1] for p in pairs])
    errors = noisy_arr - true_arr

    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    denom = np.mean(np.abs(true_arr))
    relative_error = mae / denom if denom > 0 else float("nan")

    return {"mae": float(mae), "rmse": float(rmse), "relative_error": float(relative_error)}


def _adaptive_bin_count(n_samples: int) -> int:
    """Choose histogram bins using Sturges' rule, clamped to [5, 50]."""
    if n_samples < 2:
        return 5
    return max(5, min(50, int(np.ceil(np.log2(n_samples)) + 1)))


def compute_kl_divergence(
    true_values: list[float],
    noisy_values: list[float],
    num_bins: int | None = None,
) -> float:
    true_clean = np.array([v for v in true_values if v is not None and np.isfinite(v)])
    noisy_clean = np.array([v for v in noisy_values if v is not None and np.isfinite(v)])

    if len(true_clean) < 2 or len(noisy_clean) < 2:
        return float("nan")

    if num_bins is None:
        num_bins = _adaptive_bin_count(min(len(true_clean), len(noisy_clean)))

    all_vals = np.concatenate([true_clean, noisy_clean])
    bin_edges = np.linspace(all_vals.min(), all_vals.max(), num_bins + 1)

    p_counts, _ = np.histogram(true_clean, bins=bin_edges)
    q_counts, _ = np.histogram(noisy_clean, bins=bin_edges)

    # Jeffreys prior smoothing (alpha=0.5) — principled for KL estimation
    alpha = 0.5
    p = p_counts.astype(float) + alpha
    q = q_counts.astype(float) + alpha
    p = p / p.sum()
    q = q / q.sum()

    # D_KL(P || Q) = sum P(i) * log(P(i) / Q(i))
    kl = float(np.sum(p * np.log(p / q)))
    return kl


def compute_windowed_kl_divergence(
    true_values: list[float],
    noisy_values: list[float],
    window_size: int,
    num_bins: int | None = None,
    stride: int = 1,
) -> list[float]:
    kl_per_window = []
    n = len(true_values)

    for tau in range(window_size, n + 1, stride):
        if window_size < 20:
            # Expanding window: use all data up to tau for more reliable estimation
            start = max(0, tau - max(window_size, 30))
        else:
            start = tau - window_size
        true_window = true_values[start:tau]
        noisy_window = noisy_values[start:tau]
        kl = compute_kl_divergence(true_window, noisy_window, num_bins=num_bins)
        kl_per_window.append(kl)

    return kl_per_window


def compute_global_utility(
    true_values: list[float],
    noisy_values: list[float],
    window_size: int,
    num_bins: int | None = None,
) -> float:
    kl_windows = compute_windowed_kl_divergence(
        true_values, noisy_values, window_size, num_bins
    )
    valid = [k for k in kl_windows if np.isfinite(k)]
    if not valid:
        return float("nan")
    return float(np.mean(valid))
