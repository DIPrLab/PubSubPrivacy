"""
Core differential privacy engine for clamped w-event DP with P-allocation.

Implements the paper:
  - Clamped aggregate stream (Definition 3.3): e_tau = mean(tilde{x}_{p,tau} | p in P_tau)
  - Per-element sensitivity Delta_f = R / n_tau for the default mean (Table 2).
    IMPORTANT: sensitivity is keyed on the *observed* publisher count n_tau
    (held fixed across DP neighbors), NOT on the scheduling threshold P.
    Using Delta_f = R/P would make the guarantee contingent on a lower bound
    on a data-dependent quantity and is therefore NOT a valid DP construction
    (see paper Section 4.3).
  - Sliding-window budget constraint: sum_{j=tau-w+1}^{tau} eps_j <= eps.
  - Budget-allocation strategies from Kellaris et al. (Section 5.2):
      Uniform, Sample, Budget Distribution (BD), Budget Absorption (BA).
  - P-allocation (Section 5.3): a population-aware release gate that requires
    n_tau >= P to spend budget; otherwise the element is deferred (last
    release repeated).  Composes with any inner strategy.
  - n-weighted P-allocation (Section 5.4, new contribution):
        eps_tau = eps * n_tau / sum_{j=tau-w+1}^tau n_j,
    which gives timestamps with more publishers a larger share of the budget,
    shrinking the Laplace scale quadratically with n_tau.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class BudgetStrategy(Enum):
    # ----- Kellaris et al. strategies (value-driven) -----
    UNIFORM = "uniform"
    SAMPLE = "sample"
    BUDGET_DISTRIBUTION = "budget_distribution"
    BUDGET_ABSORPTION = "budget_absorption"
    # ----- P-allocation wrappers (population-aware) -----
    P_GATED_UNIFORM = "p_gated_uniform"
    P_GATED_SAMPLE = "p_gated_sample"
    P_GATED_BD = "p_gated_bd"
    P_GATED_BA = "p_gated_ba"
    # n-weighted P-allocation (paper Section 5.4): couples eps_tau to n_tau.
    N_WEIGHTED = "n_weighted"


# Which inner allocator to use for each P-gated / n-weighted strategy.
_INNER_OF = {
    BudgetStrategy.P_GATED_UNIFORM: BudgetStrategy.UNIFORM,
    BudgetStrategy.P_GATED_SAMPLE: BudgetStrategy.SAMPLE,
    BudgetStrategy.P_GATED_BD: BudgetStrategy.BUDGET_DISTRIBUTION,
    BudgetStrategy.P_GATED_BA: BudgetStrategy.BUDGET_ABSORPTION,
    BudgetStrategy.N_WEIGHTED: BudgetStrategy.UNIFORM,  # fallback during warm-up
}


def is_p_gated(strategy: BudgetStrategy) -> bool:
    """P-allocation wrappers enforce the n_tau >= P release gate."""
    return strategy in _INNER_OF


@dataclass
class PrivacyConfig:
    """Operator-declared parameters.

    DP parameters (enter the privacy analysis, cannot be data-adapted):
        epsilon       : sliding-window privacy budget
        window_size   : w (logical timestamps)
        payload_bound : global clamp range R = sup_p (b_p - a_p)

    Scheduling hyperparameters (do NOT enter the DP calculation):
        min_publishers     : P, the publisher threshold for P-allocation
        strategy           : budget-allocation strategy (see BudgetStrategy)
        ba_threshold       : BA similarity threshold theta, as a fraction of R
    """
    epsilon: float
    window_size: int
    min_publishers: int  # P (scheduling knob only; not in sensitivity)
    payload_bound: float  # R
    strategy: BudgetStrategy = BudgetStrategy.UNIFORM
    ba_threshold: float = 0.1

    def sensitivity(self, n_tau: int) -> float:
        """Delta_f = R / n_tau for the default mean aggregation (Table 2).

        Called only when n_tau > 0 (gate/window-size enforcement upstream).
        """
        if n_tau <= 0:
            return float("inf")
        return self.payload_bound / n_tau

    def noise_scale(self, epsilon_tau: float, n_tau: int) -> float:
        """Laplace scale lambda_tau = Delta_f / eps_tau = R / (n_tau * eps_tau)."""
        if epsilon_tau <= 0 or n_tau <= 0:
            return float("inf")
        return self.sensitivity(n_tau) / epsilon_tau


@dataclass
class StreamState:
    """Runtime state for one per-subscription DP stream."""
    config: PrivacyConfig

    budget_window: deque = field(default_factory=deque)  # last w epsilons spent
    n_window: deque = field(default_factory=deque)       # last w multiplicities n_tau (eligible only)
    current_tau: int = 0
    last_released: Optional[float] = None
    last_true_aggregate: Optional[float] = None
    absorbed_budget: float = 0.0
    # Budget Distribution forward buffer (paper Algorithm 4 lines 11-18).
    # Index 0 is the share landing at the current tau; when we skip at tau
    # we add (eps/w)/(w-1) to indices 1..w-1; when we release at tau we
    # consume index 0. Maintained at length w: popleft() + append(0.0) per
    # tick keeps the indexing aligned with the logical timestamp.
    bd_forward: deque = field(default_factory=deque)
    # Set by _record on every release() call so callers (e.g. plugin.py) can
    # distinguish a fresh Laplace release from a repeat-of-previous deferral
    # without relying on numerical-equality heuristics.
    last_was_deferred: bool = False
    # History used for offline analysis / plotting
    true_values: list = field(default_factory=list)
    noisy_values: list = field(default_factory=list)
    budgets_spent: list = field(default_factory=list)
    pub_counts: list = field(default_factory=list)
    deferrals: int = 0     # count of P-gate deferrals
    releases: int = 0      # count of actual Laplace releases

    def __post_init__(self):
        # Budget/n windows store the past entries that share the current w-event
        # window with the timestamp being allocated, i.e. w-1 entries.  Using
        # maxlen=w here would carry one already-expired slot and over-subtract
        # from the remaining budget / inflate the n-weighted denominator
        # (still sound, just wastes utility and drifts from the paper's
        # exact formula in Sec. 5.4 eq. 5).
        past_window = max(0, self.config.window_size - 1)
        self.budget_window = deque(maxlen=past_window)
        self.n_window = deque(maxlen=past_window)
        # BD forward buffer: length w, zero-initialized, no maxlen so we can
        # safely popleft + append per tick.
        self.bd_forward = deque([0.0] * self.config.window_size)

    # ---- budget bookkeeping ---------------------------------------------

    def _budget_spent_in_window(self) -> float:
        return sum(self.budget_window)

    def _budget_remaining(self) -> float:
        return self.config.epsilon - self._budget_spent_in_window()

    # ---- inner value-driven allocators ----------------------------------

    def _alloc_uniform(self) -> tuple[float, bool]:
        return self.config.epsilon / self.config.window_size, False

    def _alloc_sample(self) -> tuple[float, bool]:
        # Release full budget every w-th eligible timestamp; else skip.
        if self.current_tau % self.config.window_size == 1:
            return self.config.epsilon, False
        return 0.0, True

    def _alloc_budget_distribution(self, aggregate: float) -> tuple[float, bool]:
        """
        BD (Kellaris et al., paper Algorithm 4 lines 11-18).

        The caller ticks ``bd_forward`` exactly once per tau in ``release()``
        (setting ``_bd_pending_now`` to the share landing at this tau), so
        this method never shifts the deque itself.

          - Skip (|e_tau - last_rel| < theta*R): forward the base share
            eps/w equally across fwd[tau+1..tau+w-1]; return (0, skip).
            The ``pending_now`` share for this tau is lost (paper: fwd is
            only drained on release).
          - Release: spend eps/w + pending_now; return that, false.
        """
        w = self.config.window_size
        pending_now = getattr(self, "_bd_pending_now", 0.0)

        if self.last_true_aggregate is not None:
            change = abs(aggregate - self.last_true_aggregate)
            threshold = self.config.ba_threshold * self.config.payload_bound
            if change < threshold:
                # Distribute eps/w equally across fwd[tau+1..tau+w-1].
                if w > 1 and len(self.bd_forward) >= w - 1:
                    per_slot = (self.config.epsilon / w) / (w - 1)
                    for i in range(w - 1):
                        self.bd_forward[i] += per_slot
                return 0.0, True

        share = self.config.epsilon / w + pending_now
        # The sliding-window budget check downstream in release() still caps
        # at remaining budget, so this cannot violate (Eq. 4).
        share = min(share, self.config.epsilon)
        return max(share, 0.0), False

    def _alloc_budget_absorption(self, aggregate: float) -> tuple[float, bool]:
        """
        BA (Kellaris et al.): If the aggregate has not changed much, skip and
        ABSORB the would-be share into a running pot that is added to the next
        release with a large change.
        """
        if self.last_true_aggregate is not None:
            change = abs(aggregate - self.last_true_aggregate)
            threshold = self.config.ba_threshold * self.config.payload_bound
            if change < threshold:
                self.absorbed_budget += self.config.epsilon / self.config.window_size
                # Cap absorbed budget at epsilon - any more cannot actually be
                # spent under the sliding-window constraint.
                self.absorbed_budget = min(self.absorbed_budget, self.config.epsilon)
                return 0.0, True

        budget = self.config.epsilon / self.config.window_size + self.absorbed_budget
        self.absorbed_budget = 0.0
        return budget, False

    def _alloc_n_weighted(self, n_tau: int) -> tuple[float, bool]:
        """n-weighted P-allocation (paper Section 5.4).

          eps_tau = eps * n_tau / sum_{j=tau-w+1}^{tau} n_j.

        During warm-up (fewer than w eligible timestamps observed) we fall
        back to the Uniform share eps/w, as recommended in the paper.
        n_window has maxlen = w-1 and holds prior eligible timestamps, so
        when it is at capacity we have a full w-long window once n_tau is
        added to the denominator.
        """
        if len(self.n_window) < self.n_window.maxlen:
            return self._alloc_uniform()
        total_n = sum(self.n_window) + n_tau
        if total_n <= 0:
            return 0.0, True
        return self.config.epsilon * n_tau / total_n, False

    # ---- dispatcher -----------------------------------------------------

    def _allocate(self, aggregate: float, n_tau: int) -> tuple[float, bool]:
        strat = self.config.strategy

        if strat == BudgetStrategy.UNIFORM:
            return self._alloc_uniform()
        if strat == BudgetStrategy.SAMPLE:
            return self._alloc_sample()
        if strat == BudgetStrategy.BUDGET_DISTRIBUTION:
            return self._alloc_budget_distribution(aggregate)
        if strat == BudgetStrategy.BUDGET_ABSORPTION:
            return self._alloc_budget_absorption(aggregate)

        if strat == BudgetStrategy.N_WEIGHTED:
            # N_WEIGHTED is implicitly P-gated (uses population counts).
            return self._alloc_n_weighted(n_tau)

        # P-gated wrappers: delegate to inner strategy; gate enforced upstream.
        inner = _INNER_OF[strat]
        saved = self.config.strategy
        self.config.strategy = inner
        try:
            result = self._allocate(aggregate, n_tau)
        finally:
            self.config.strategy = saved
        return result

    # ---- public release entry point -------------------------------------

    def release(self, aggregate: float, n_tau: int) -> Optional[float]:
        """
        Process one aggregate stream element with multiplicity n_tau.

        Returns the delivered (noisy or repeated) value.  The P-allocation gate
        is enforced here for P-gated / n-weighted strategies: if n_tau < P the
        element is deferred (last release is repeated) and no budget is spent.
        """
        self.current_tau += 1

        # Tick the BD forward-distribution buffer on every tau (whether or not
        # BD runs at this release), so its indexing stays aligned with the
        # logical timestamp under any P-gate or inner-strategy composition.
        if self.bd_forward:
            self._bd_pending_now = self.bd_forward.popleft()
            self.bd_forward.append(0.0)
        else:
            self._bd_pending_now = 0.0

        # --- P-allocation release gate ---------------------------------
        if is_p_gated(self.config.strategy) and n_tau < self.config.min_publishers:
            self._record(aggregate, self.last_released, 0.0, n_tau, deferred=True)
            return self.last_released

        # Must have at least one publisher for the mean to be defined.
        if n_tau <= 0:
            self._record(aggregate, self.last_released, 0.0, n_tau, deferred=True)
            return self.last_released

        epsilon_tau, skip = self._allocate(aggregate, n_tau)

        # Enforce window budget constraint (cap at remaining budget).
        remaining = self._budget_remaining()
        epsilon_tau = min(epsilon_tau, remaining)

        # Floor on epsilon_tau: releasing with a near-zero share would give a
        # Laplace scale that explodes numerically.  Treat anything below
        # 1e-6 * (eps / w) as a skip (equivalent to the window being full).
        min_eps_tau = 1e-6 * (self.config.epsilon / self.config.window_size)
        if skip or epsilon_tau <= min_eps_tau:
            self._record(aggregate, self.last_released, 0.0, n_tau, deferred=True)
            return self.last_released

        lam = self.config.noise_scale(epsilon_tau, n_tau)
        noise = np.random.laplace(loc=0.0, scale=lam)
        released = aggregate + noise

        self.last_released = released
        self.last_true_aggregate = aggregate
        self._record(aggregate, released, epsilon_tau, n_tau, deferred=False)
        return released

    def _record(self, aggregate, released, eps_tau, n_tau, deferred):
        self.last_was_deferred = bool(deferred)
        self.budget_window.append(eps_tau)
        # Only eligible (budget-spending) timestamps enter n_window, matching
        # the paper's causal definition for n-weighted allocation.
        if eps_tau > 0:
            self.n_window.append(n_tau)
            self.releases += 1
        else:
            self.deferrals += 1
        self.true_values.append(aggregate)
        self.noisy_values.append(released)
        self.budgets_spent.append(eps_tau)
        self.pub_counts.append(n_tau)


# ═════════════════════════════════════════════════════════════════════════
#  Utility metrics
# ═════════════════════════════════════════════════════════════════════════

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

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    denom = float(np.mean(np.abs(true_arr)))
    relative_error = mae / denom if denom > 0 else float("nan")

    return {"mae": mae, "rmse": rmse, "relative_error": relative_error}


def _adaptive_bin_count(n_samples: int) -> int:
    if n_samples < 2:
        return 5
    return max(5, min(50, int(np.ceil(np.log2(n_samples)) + 1)))


def compute_kl_divergence(
    true_values: list[float],
    noisy_values: list[float],
    num_bins: int | None = None,
) -> float:
    """D_KL(P || Q) with Jeffreys smoothing (alpha=0.5)."""
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

    alpha = 0.5
    p = p_counts.astype(float) + alpha
    q = q_counts.astype(float) + alpha
    p = p / p.sum()
    q = q / q.sum()

    return float(np.sum(p * np.log(p / q)))


def compute_windowed_kl_divergence(
    true_values: list[float],
    noisy_values: list[float],
    window_size: int,
    num_bins: int | None = None,
    stride: int = 1,
) -> list[float]:
    """Per-window KL divergence U_tau (paper Section 5.8)."""
    kl_per_window = []
    n = len(true_values)

    for tau in range(window_size, n + 1, stride):
        if window_size < 20:
            start = max(0, tau - max(window_size, 30))
        else:
            start = tau - window_size
        kl = compute_kl_divergence(true_values[start:tau], noisy_values[start:tau], num_bins)
        kl_per_window.append(kl)

    return kl_per_window


def compute_global_utility(
    true_values: list[float],
    noisy_values: list[float],
    window_size: int,
    num_bins: int | None = None,
) -> float:
    """U_global (paper Section 5.8): mean of per-window KL divergences."""
    kl_windows = compute_windowed_kl_divergence(
        true_values, noisy_values, window_size, num_bins
    )
    valid = [k for k in kl_windows if np.isfinite(k)]
    if not valid:
        return float("nan")
    return float(np.mean(valid))


# ═════════════════════════════════════════════════════════════════════════
#  Publisher-identity attribution advantage
# ═════════════════════════════════════════════════════════════════════════

def attribution_advantage(pub_counts: list[int], deferred_mask: list[bool]) -> float:
    """Mean 1/n_tau over released timestamps (paper Section 5.2)."""
    advs = [1.0 / n for n, d in zip(pub_counts, deferred_mask)
            if not d and n and n > 0]
    if not advs:
        return float("nan")
    return float(np.mean(advs))
