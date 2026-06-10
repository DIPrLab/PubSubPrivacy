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
  - BD and BA follow Kellaris et al. Section 4: each tau runs a private
    dissimilarity sub-mechanism M_{i,1} (spending eps/(2w), Laplace scale
    R / (n_tau * eps_{i,1})) whose noisy output drives the skip/publish
    decision, plus a publication sub-mechanism M_{i,2} that is bounded to a
    share of eps/2 per window so total(M_{i,1}+M_{i,2}) fits under eps.
  - BA additionally nullifies the alpha timestamps following a release that
    absorbed alpha prior skipped budgets (Kellaris et al. Figure 4 lines 5-6)
    so the sliding-window constraint holds by construction, not by truncation.
  - P-allocation (Section 5.3): a population-aware release gate that requires
    n_tau >= P to spend budget; otherwise the element is deferred (last
    release repeated).  Composes with any inner strategy.
  - n-weighted P-allocation (Section 6.4, new contribution):
        eps_tau = eps * (1/n_tau) / sum_{j=tau-w+1}^tau (1/n_j),
    i.e. eps_tau is INVERSELY proportional to n_tau, so dense-pool timestamps
    receive a SMALLER budget share (more noise) and the released Laplace scale
    is equalized across the window (lambda_tau = R*sum_j(1/n_j)/eps, independent
    of n_tau), offsetting the 1/n_tau mean-sensitivity term.  See
    _alloc_n_weighted and AUDIT.md (A3) for why the paper's printed Eq. (5)
    denominator `sum_j n_j` is a typo for `sum_j (1/n_j)`.
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
        epsilon_count : eps_count, the per-step budget spent to release a
                        differentially private publisher count |P_tau| with
                        sensitivity 1 (paper Sec. 6.3 step 1, Sec. 6.5 step 3,
                        Table 3).  n_tau is DP-private: ONE noisy count is drawn
                        per timestamp/scope and reused for the P_min gate, the
                        Laplace-scale denominator, and the n-weighted
                        denominator (reuse is free by post-processing), so the
                        exact n_tau never enters the released noise scale.
                        eps_count is charged INSIDE the single w-event budget
                        eps: the count spent at each timestamp takes from the
                        usable eps, so Sum_{window} (eps_count draws + releases)
                        <= eps (the value-driven allocators run against
                        eps_inner = eps - w*eps_count).  Default 0.05.  0 disables
                        the DP count and falls back to the exact n_tau
                        (Kellaris-style baselines that never gate on a count).

    Scheduling hyperparameters (do NOT enter the DP calculation):
        min_publishers     : P_min, the publisher threshold for P-allocation
        max_publishers     : P_max, caps the multiplicity that enters the mean
                             sensitivity so Delta_f = R / n_tau can change by at
                             most over [P_min, P_max] across stream elements
                             (paper Sec. 6.5; None disables the cap).
        strategy           : budget-allocation strategy (see BudgetStrategy)
        ba_threshold       : BA similarity threshold theta, as a fraction of R
    """
    epsilon: float
    window_size: int
    min_publishers: int  # P_min (scheduling knob only; not in sensitivity)
    payload_bound: float  # R
    strategy: BudgetStrategy = BudgetStrategy.UNIFORM
    ba_threshold: float = 0.1
    # Split parameter rho_tau in (0, 1) for the aggregate-stream-element
    # release (Definition: Aggregate Stream Element).  The per-step budget
    # eps_tau = epsilon / window_size is split between the two privatized
    # components the broker releases at tau: the noisy publisher count
    #   n~_tau = n_tau + Lap(1 / (rho * eps_tau))                 (count: rho share)
    # and the noisy sum
    #   S~_tau = sum_p x~_{p,tau} + Lap(Delta_S / ((1 - rho) * eps_tau))  (sum: 1-rho share),
    # post-processed into gamma_tau = S~_tau / max(n~_tau, 1) for f = mean.
    # rho is data-independent at release time, so by sequential composition the
    # element spends exactly eps_tau = (1-rho) eps_tau + rho eps_tau.  The count
    # share is reserved across the window (see StreamState.release), so the
    # allocator's pub_eps already equals (1-rho) * eps_tau -- the sum's share.
    # Default 0.2 = the definition's error-equalizing sqrt(R)/(5 sqrt(R)); it
    # empirically minimizes released error across datasets (see change notes).
    rho_split: float = 0.2
    # DEPRECATED: superseded by ``rho_split``.  Retained so existing callers
    # that pass ``epsilon_count`` keep constructing; it no longer sets the count
    # noise scale (that is now rho * eps_tau).  See ``count_epsilon``.
    epsilon_count: float = 0.05
    # P_max sensitivity-binding cap (Sec. 6.5).  None => no upper cap.
    max_publishers: Optional[int] = None

    def step_epsilon(self) -> float:
        """Per-step budget eps_tau = epsilon / window_size that rho splits."""
        if self.window_size <= 0:
            return 0.0
        return self.epsilon / self.window_size

    def count_epsilon(self) -> float:
        """Count share rho * eps_tau spent on the noisy publisher count n~_tau.

        The Laplace scale of the count (sensitivity 1) is 1 / count_epsilon() =
        1 / (rho * eps_tau).  rho <= 0 disables the DP count (exact n_tau).
        """
        if self.rho_split <= 0:
            return 0.0
        return self.rho_split * self.step_epsilon()

    def sum_sensitivity(self) -> float:
        """Data-independent sum sensitivity Delta_S = R (one clamped payload).

        Adding/removing one publisher shifts the pooled sum sum_p x~_{p,tau} by
        at most the clamp range R, so the noisy sum S~_tau uses Delta_S = R --
        independent of the multiplicity, unlike the mean's Delta_f = R/n_tau.
        """
        return self.payload_bound

    def effective_n(self, n_tau: int) -> int:
        """Multiplicity actually folded into the mean after the P_max cap.

        When n_tau > P_max the broker aggregates only the first P_max
        publishers and discards the rest (paper Sec. 6.5), so both the mean
        and its sensitivity are computed at min(n_tau, P_max).  This bounds the
        per-element sensitivity change for f = mean to the [P_min, P_max] band.
        """
        if self.max_publishers is not None and n_tau > self.max_publishers > 0:
            return int(self.max_publishers)
        return int(n_tau)

    def sensitivity(self, n_tau: int) -> float:
        """Delta_f = R / n_tau for the default mean aggregation (Table 2).

        n_tau is the post-P_max effective multiplicity; callers pass the value
        returned by ``effective_n``.  Called only when n_tau > 0 (gate /
        window-size enforcement upstream).
        """
        if n_tau <= 0:
            return float("inf")
        return self.payload_bound / n_tau

    def noise_scale(self, epsilon_tau: float, n_tau: int) -> float:
        """Laplace scale lambda_tau = Delta_f / eps_tau = R / (n_tau * eps_tau)."""
        if epsilon_tau <= 0 or n_tau <= 0:
            return float("inf")
        return self.sensitivity(n_tau) / epsilon_tau

    def sum_noise_scale(self, sum_epsilon_tau: float) -> float:
        """Laplace scale for the noisy sum S~_tau = Delta_S / ((1-rho) * eps_tau).

        ``sum_epsilon_tau`` is the (1-rho) share of eps_tau actually allocated
        to the publication (pub_eps), already net of the reserved count share,
        so this equals Delta_S / ((1-rho) * eps_tau).  Unlike ``noise_scale``
        there is no 1/n_tau factor: the sum sensitivity Delta_S = R is
        data-independent.
        """
        if sum_epsilon_tau <= 0:
            return float("inf")
        return self.sum_sensitivity() / sum_epsilon_tau


@dataclass
class StreamState:
    """Runtime state for one per-subscription DP stream."""
    config: PrivacyConfig

    budget_window: deque = field(default_factory=deque)  # last w epsilons spent
    n_window: deque = field(default_factory=deque)       # last w multiplicities n_tau (eligible only)
    current_tau: int = 0
    last_released: Optional[float] = None
    last_true_aggregate: Optional[float] = None
    # Budget Distribution forward buffer (Kellaris et al. BD, half-budget
    # variant).  Index 0 is the share landing at the current tau; when we
    # skip at tau we add (eps/(2w))/(w-1) to indices 0..w-2 (i.e. to taus
    # tau+1..tau+w-1); when we release at tau we consume index 0.
    # Maintained at length w: popleft() + append(0.0) per tick keeps the
    # indexing aligned with the logical timestamp.
    bd_forward: deque = field(default_factory=deque)
    # BA nullification state (Kellaris et al. Figure 4 lines 5-6).
    # ba_nullify_remaining counts how many future taus M_{i,2} must force
    # to null because the last publication absorbed that many prior skips.
    # ba_skipped_since_last_pub counts skipped publications waiting to be
    # absorbed at the next release.
    ba_nullify_remaining: int = 0
    ba_skipped_since_last_pub: int = 0
    # Set by _record on every release() call so callers (e.g. plugin.py) can
    # distinguish a fresh Laplace release from a repeat-of-previous deferral
    # without relying on numerical-equality heuristics.
    last_was_deferred: bool = False
    # Running total of eps_count spent on differentially private publisher
    # counts (paper Sec. 6.3 step 1).  Tracked separately from the w-event
    # budget, exactly as eps_clip is, and reported as composed DP cost.
    eps_count_spent: float = 0.0
    dp_counts: int = 0   # number of timestamps that paid eps_count
    # History used for offline analysis / plotting
    true_values: list = field(default_factory=list)
    noisy_values: list = field(default_factory=list)
    budgets_spent: list = field(default_factory=list)
    pub_counts: list = field(default_factory=list)
    # Authoritative per-timestamp deferred flag.  Needed because BD/BA spend a
    # nonzero dissimilarity budget on a *skipped* timestamp, so "budget == 0"
    # is NOT a reliable released/deferred test for those strategies.
    deferred_flags: list = field(default_factory=list)
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
        # Budget the value-driven allocators may spend per timestamp.  Equals
        # the full epsilon for the Kellaris baselines; for population-aware
        # strategies with a DP count it is reduced by the per-window count
        # reservation (w * eps_count) so that releases + count draws together
        # never breach Sum_{window} eps_j <= eps.  Set per call in release().
        self._alloc_eps = self.config.epsilon
        # BD forward buffer: length w, zero-initialized, no maxlen so we can
        # safely popleft + append per tick.
        self.bd_forward = deque([0.0] * self.config.window_size)

    # ---- budget bookkeeping ---------------------------------------------

    def _budget_spent_in_window(self) -> float:
        return sum(self.budget_window)

    def _budget_remaining(self) -> float:
        return self.config.epsilon - self._budget_spent_in_window()

    # ---- differentially private publisher count (eps_count) -------------

    def _dp_count(self, n_tau: int) -> int:
        """Release the noisy publisher count n~_tau = |P_tau| + Lap(1/(rho*eps_tau)).

        This is the count component of the aggregate stream element
        (Definition: Aggregate Stream Element): the count query has sensitivity
        1 (one publisher added/removed), so the Laplace scale is
        1 / count_epsilon() = 1 / (rho * eps_tau) -- the rho share of the
        per-step budget.  The noisy count is clamped to >= 0 and rounded to the
        nearest integer.  When count_epsilon() == 0 (rho <= 0) the DP count is
        disabled and the exact n_tau is used (Kellaris-style baselines that do
        not gate on a population threshold).

        The caller draws this ONCE per timestamp/scope and reuses the returned
        value n~_tau as the released count component AND for every downstream
        use of the multiplicity (the P_min gate, the released denominator
        max(n~_tau, 1), and the n-weighted denominator).  Reuse is free by
        post-processing, and because the exact n_tau is treated as DP-private it
        never enters the released noise scale -- only this noisy count does.
        """
        eps_n = self.config.count_epsilon()
        if eps_n <= 0:
            return int(n_tau)
        self.eps_count_spent += eps_n
        self.dp_counts += 1
        noisy = n_tau + float(np.random.laplace(loc=0.0, scale=1.0 / eps_n))
        return max(0, int(round(noisy)))

    # ---- private dissimilarity sub-mechanism M_{i,1} --------------------

    def _private_dissimilarity(
        self, aggregate: float, n_tau: int
    ) -> tuple[float, float]:
        """Kellaris et al. M_{i,1}: noisy dissimilarity vs. the last release.

        Returns ``(noisy_dis, eps_dissim)``.  The comparison is against
        ``last_released`` (o_l in the paper) which is already public, so
        only the sensitivity of the current aggregate enters the analysis:
        one added/removed row shifts ``|aggregate - last_released|`` by at
        most R / n_tau, so spending eps/(2w) budget requires Laplace noise
        of scale (R / n_tau) / (eps/(2w)) = 2 w R / (n_tau * eps).

        When no prior release exists (cold-start) dissim is unused and we
        spend zero budget; returning +inf forces the caller's publish branch.
        """
        if self.last_released is None or n_tau <= 0:
            return float("inf"), 0.0
        w = self.config.window_size
        eps_dissim = self._alloc_eps / (2 * w)
        sensitivity = self.config.payload_bound / n_tau
        scale = sensitivity / eps_dissim
        raw_dis = abs(aggregate - self.last_released)
        noisy = raw_dis + float(np.random.laplace(loc=0.0, scale=scale))
        return noisy, eps_dissim

    # ---- inner value-driven allocators ----------------------------------
    #
    # Each allocator returns ``(dissim_eps, pub_eps, skip)``.  Strategies
    # without a private dissimilarity step (Uniform, Sample, n-weighted)
    # return ``dissim_eps == 0``.  Total budget spent at tau is the sum
    # ``dissim_eps + pub_eps``; both participate in the sliding-window cap.

    def _alloc_uniform(self) -> tuple[float, float, bool]:
        return 0.0, self._alloc_eps / self.config.window_size, False

    def _alloc_sample(self) -> tuple[float, float, bool]:
        # Release full budget every w-th eligible timestamp; else skip.
        # (tau - 1) % w == 0 handles w == 1 (every tau publishes) and
        # matches the paper's "i mod w == 1" for w > 1.
        if (self.current_tau - 1) % self.config.window_size == 0:
            return 0.0, self._alloc_eps, False
        return 0.0, 0.0, True

    def _alloc_budget_distribution(
        self, aggregate: float, n_tau: int
    ) -> tuple[float, float, bool]:
        """BD with private dissimilarity (Kellaris et al. Algorithm 4).

        Per-tau layout in the w-event budget of eps:
          M_{i,1}: eps/(2w)           (noisy dissimilarity)
          M_{i,2}: eps/(2w) base share + accumulated forward shares from
                   earlier skips, capped at eps/2 per window.

        Window invariant: dissim totals exactly eps/2 over any w consecutive
        taus; publication totals at most eps/2 because every skip forwards
        only (eps/(2w))/(w-1) per future slot and at most w-1 slots land in
        any window.  Sum = eps.
        """
        w = self.config.window_size
        pending_now = getattr(self, "_bd_pending_now", 0.0)
        noisy_dis, eps_dissim = self._private_dissimilarity(aggregate, n_tau)

        if self.last_released is not None:
            threshold = self.config.ba_threshold * self.config.payload_bound
            if noisy_dis < threshold:
                # Forward the base share across the w-1 future slots.
                if w > 1 and len(self.bd_forward) >= w - 1:
                    per_slot = (self._alloc_eps / (2 * w)) / (w - 1)
                    for i in range(w - 1):
                        self.bd_forward[i] += per_slot
                return eps_dissim, 0.0, True

        pub_share = self._alloc_eps / (2 * w) + pending_now
        pub_share = max(pub_share, 0.0)
        pub_share = min(pub_share, self._alloc_eps / 2)
        return eps_dissim, pub_share, False

    def _alloc_budget_absorption(
        self, aggregate: float, n_tau: int
    ) -> tuple[float, float, bool]:
        """BA with private dissimilarity and nullification (Kellaris et al.
        Algorithm 5 / Figure 4).

        State:
          ba_skipped_since_last_pub: count of skipped publications whose
            base share eps/(2w) is waiting to be absorbed at the next release.
          ba_nullify_remaining: the next k taus' M_{i,2} must force null
            because the last release absorbed k prior skips (Figure 4 line 6).

        Per-tau layout:
          M_{i,1}: eps/(2w) always.
          M_{i,2}: nullified -> 0
                   skipped   -> 0 (and ba_skipped += 1)
                   released  -> eps/(2w) * min(1 + ba_skipped, w)
                                and ba_nullify_remaining = (that factor) - 1.

        Window invariant: dissim = eps/2; pub across any w consecutive taus
        is at most eps/2 because each skipped base share is absorbed by at
        most one later release and followed by exactly that many nullified
        slots, so the pub mass in the window never exceeds w * eps/(2w).
        """
        w = self.config.window_size
        base = self._alloc_eps / (2 * w)
        noisy_dis, eps_dissim = self._private_dissimilarity(aggregate, n_tau)

        # Nullification: skip by fiat, regardless of dissimilarity.
        if self.ba_nullify_remaining > 0:
            self.ba_nullify_remaining -= 1
            return eps_dissim, 0.0, True

        if self.last_released is not None:
            threshold = self.config.ba_threshold * self.config.payload_bound
            if noisy_dis < threshold:
                # Data is close to the last release; skip and save the share
                # for a future absorbing release.  Cap at w-1 so a single
                # release cannot absorb more than w-1 prior skips (Figure 4).
                if self.ba_skipped_since_last_pub < w - 1:
                    self.ba_skipped_since_last_pub += 1
                return eps_dissim, 0.0, True

        # Release: absorb prior skipped shares and nullify the same count
        # of following taus so the window constraint holds by construction.
        to_absorb = min(1 + self.ba_skipped_since_last_pub, w)
        pub_share = base * to_absorb
        self.ba_nullify_remaining = to_absorb - 1
        self.ba_skipped_since_last_pub = 0
        return eps_dissim, pub_share, False

    def _alloc_n_weighted(self, n_tau: int) -> tuple[float, float, bool]:
        """n-weighted P-allocation (paper Section 6.4, Eq. 5).

          eps^nom_tau = eps * (1/n_tau) / sum_{j=tau-w+1}^{tau} (1/n_j).

        The per-timestamp budget is *inversely* proportional to the
        multiplicity n_tau: dense-pool timestamps (large n_tau) receive a
        SMALLER share of eps and release with more Laplace noise, while sparse
        but eligible timestamps receive a larger share.  Because the mean
        sensitivity is Delta_f = R/n_tau, this exactly offsets the 1/n_tau
        sensitivity term and equalizes the released Laplace scale across the
        window:  lambda_tau = (R/n_tau)/eps^nom_tau = R * sum_j(1/n_j) / eps,
        independent of n_tau (paper Sec. 6.4).

        NOTE: the paper's printed Eq. (5) denominator reads ``sum_j n_j`` but
        that is inconsistent with both the stated "inversely proportional"
        property and the "shares sum to eps" claim; the only reading that
        satisfies both is the inverse-weight sum ``sum_j (1/n_j)`` used here.
        See AUDIT.md for the derivation.

        During warm-up (fewer than w eligible timestamps observed) we fall
        back to the Uniform share eps/w, as recommended in the paper.  The
        clipping step of Eq. (6) is applied downstream in ``release`` via the
        sliding-window budget cap.
        """
        if len(self.n_window) < self.n_window.maxlen:
            return self._alloc_uniform()
        if n_tau <= 0:
            return 0.0, 0.0, True
        inv_total = sum(1.0 / m for m in self.n_window if m > 0) + 1.0 / n_tau
        if inv_total <= 0:
            return 0.0, 0.0, True
        eps_nom = self._alloc_eps * (1.0 / n_tau) / inv_total
        return 0.0, eps_nom, False

    # ---- dispatcher -----------------------------------------------------

    def _allocate(
        self, aggregate: float, n_tau: int
    ) -> tuple[float, float, bool]:
        strat = self.config.strategy

        if strat == BudgetStrategy.UNIFORM:
            return self._alloc_uniform()
        if strat == BudgetStrategy.SAMPLE:
            return self._alloc_sample()
        if strat == BudgetStrategy.BUDGET_DISTRIBUTION:
            return self._alloc_budget_distribution(aggregate, n_tau)
        if strat == BudgetStrategy.BUDGET_ABSORPTION:
            return self._alloc_budget_absorption(aggregate, n_tau)

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

    def release(self, aggregate: float, n_tau: int,
                n_gate: Optional[int] = None,
                n_count_draws: int = 1,
                true_sum: Optional[float] = None) -> Optional[float]:
        """
        Process one aggregate stream element with multiplicity n_tau.

        Returns the delivered (noisy or repeated) value.  The P-allocation gate
        is enforced here for P-gated / n-weighted strategies: if the gate count
        < P the element is deferred (last release is repeated) and no budget is
        spent.

        ``true_sum`` is the actual pooled sum sum_{p in P_tau} x_{p,tau} that the
        noisy-sum component S~_tau is built from (Definition: Aggregate Stream
        Element).  Callers pass the real summed data; when omitted it falls back
        to ``aggregate * n_tau`` (exact when the upstream aggregates one value
        per publisher, e.g. the offline rebuild).  ``aggregate`` (the pooled
        mean) is retained only as that fallback and as the true value the utility
        metrics compare against -- it is NOT used to scale the released noise.

        ``n_gate`` lets the CALLER supply the (already differentially private)
        publisher count that the gate should compare against P_min — used by the
        broker plugin, which draws and charges the DP count(s) during its
        Algorithm-1 hierarchy walk and must not have the engine re-gate on the
        exact pooled count (that would be a second, inconsistent gate).  When
        ``n_gate`` is None (the offline path) the engine draws its own DP count
        from ``epsilon_count`` as before, so the gate is single-sourced either
        way.  ``n_count_draws`` is how many eps_count draws the caller made for
        THIS element (offline: 1; broker walk: leaf + one per probed ancestor),
        so the full per-timestamp count cost ``n_count_draws * eps_count`` is
        charged inside the window budget eps — the caller must ensure the window
        can afford it (the plugin pre-checks before each probe).
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

        # --- DP publisher count (rho * eps_tau) + P-allocation gate -------
        # For our population-aware strategies the multiplicity n_tau is
        # DP-PRIVATE.  We draw ONE noisy count n~_tau = |P_tau| + Lap(1/(rho*
        # eps_tau)) (sensitivity 1, the count component of the aggregate stream
        # element) and REUSE that single value for everything downstream -- the
        # P_min gate, the released denominator max(n~_tau, 1), and the
        # n-weighted denominator -- so the exact private n_tau never leaves the
        # broker and reuse is free by post-processing.  The count share rho *
        # eps_tau is charged INSIDE the w-event budget eps: it counts toward the
        # sliding-window sum, so Sum_{window} (count draws + sum releases) <= eps.
        # Each of the w timestamps spends rho * eps_tau on its count, so the
        # value-driven allocators run against eps_inner = eps - w*(rho*eps_tau)
        # = (1-rho)*eps (reserved below) and their pub_eps is exactly the sum's
        # (1-rho)*eps_tau share.  The Kellaris baselines (Uniform/Sample/BD/BA)
        # do not gate on a count and keep the exact n_tau (public in their
        # row-addition model).  A caller that already drew the count (``n_gate``,
        # the broker path) supplies it verbatim so the broker's Algorithm-1
        # decision is the single authority.
        eps_count_tau = 0.0
        population_aware = (is_p_gated(self.config.strategy)
                            or self.config.strategy == BudgetStrategy.N_WEIGHTED)
        if population_aware:
            ec = self.config.count_epsilon()  # rho * eps_tau
            # Pre-check: never draw the DP count if the window can't pay for it,
            # else the count spend would breach the w-event invariant.
            if n_gate is None and ec > 0 and self._budget_remaining() < ec:
                self._record(aggregate, self.last_released, 0.0, n_tau, deferred=True)
                return self.last_released
            n_dp = self._dp_count(n_tau) if n_gate is None else int(n_gate)
            # Offline draws ONE count here; the broker may have drawn several
            # (leaf + one per probed ancestor) -- or zero, if its budget-aware
            # pre-check could not afford even the leaf count -- and reports the
            # total so the full per-timestamp count cost is charged inside the
            # window budget.
            n_draws = 1 if n_gate is None else max(0, int(n_count_draws))
            eps_count_tau = ec * n_draws if ec > 0 else 0.0
            if n_dp < self.config.min_publishers:
                # Deferred -- the count draw already spent eps_count, so it still
                # consumes window budget.
                self._record(aggregate, self.last_released, eps_count_tau, n_tau, deferred=True)
                return self.last_released
        else:
            n_dp = n_tau

        # Must have at least one (effective) publisher for the mean to be defined.
        if n_dp <= 0:
            self._record(aggregate, self.last_released, eps_count_tau, n_tau, deferred=True)
            return self.last_released

        # Post-P_max effective multiplicity, derived from the DP count n_dp so
        # nothing here depends on the exact private count.  n_eff feeds the
        # budget allocator (the n-weighted share and the n_window it records) and
        # the Kellaris baselines' mean-sensitivity R/n_eff.  It does NOT enter the
        # population-aware released noise: that sum uses the data-independent
        # sensitivity Delta_S = R and is divided by the noisy count n_dp below.
        n_eff = self.config.effective_n(n_dp)

        # Reserve the per-window count share so sum releases + count draws
        # together stay under eps: eps_inner = eps - w*(rho*eps_tau) = (1-rho)*eps.
        # The allocator then hands back pub_eps = (1-rho)*eps_tau -- exactly the
        # sum's share -- so S~_tau is calibrated to Delta_S/((1-rho)*eps_tau)
        # below.  (rho <= 0 -> eps_count_tau == 0 -> eps_inner == eps.)
        self._alloc_eps = max(
            0.0, self.config.epsilon - self.config.window_size * eps_count_tau)

        # Snapshot the BA absorb/nullify counters BEFORE allocation so that, if
        # the defensive sliding-window cap below turns a planned BA *release*
        # into a skip, we can roll them back — otherwise the absorbed skips
        # would be lost and the following taus over-nullified (Kellaris Fig. 4
        # accounting would drift).  No-op for non-BA strategies (counters stay 0).
        _ba_null_before = self.ba_nullify_remaining
        _ba_skipped_before = self.ba_skipped_since_last_pub

        dissim_eps, pub_eps, skip = self._allocate(aggregate, n_eff)

        # Enforce the sliding-window budget constraint on the combined cost
        # of M_{i,1} and M_{i,2}.  The in-strategy accounting for BD/BA is
        # already window-safe by construction; this cap is a defensive floor
        # for Uniform/Sample/n-weighted and for strategy transitions, and it
        # GUARANTEES the w-event budget invariant holds regardless of any
        # sub-mechanism edge case (DP correctness never depends on the
        # sub-mechanism being perfectly tight).
        # eps_count is charged inside the one w-event budget, so the release
        # shares what is left of eps after this timestamp's count draw.
        remaining = self._budget_remaining() - eps_count_tau
        total_eps = dissim_eps + pub_eps
        capped = False
        if total_eps > remaining:
            # Publication is cheaper to sacrifice than the already-drawn
            # dissimilarity noise (which we cannot "un-spend").
            pub_eps = max(0.0, remaining - dissim_eps)
            total_eps = dissim_eps + pub_eps
            capped = True

        # Floor on pub_eps: releasing with a near-zero share would give a
        # Laplace scale that explodes numerically.  Treat anything below
        # 1e-6 * (eps / w) as a skip (equivalent to the window being full).
        min_pub_eps = 1e-6 * (self._alloc_eps / self.config.window_size)
        if skip or pub_eps <= min_pub_eps:
            # No publication at this tau; record dissim cost (may be 0 for
            # strategies without M_{i,1}) and repeat the last release.
            if capped and not skip:
                # The defensive cap (not the sub-mechanism) converted a planned
                # release into a skip; restore the BA counters so the saved
                # skips aren't lost and future taus aren't wrongly nullified.
                # This only moves counters, never budget, so the window
                # invariant is unaffected.
                self.ba_nullify_remaining = _ba_null_before
                self.ba_skipped_since_last_pub = _ba_skipped_before
            self._record(aggregate, self.last_released, eps_count_tau + total_eps,
                         n_eff, deferred=True)
            return self.last_released

        if population_aware:
            # Aggregate stream element via the sum/count split (Definition:
            # Aggregate Stream Element).  The broker drew the noisy count n~_tau
            # above (scale 1/(rho*eps_tau)) and gated on it (n_dp >= P_min);
            # here it releases the noisy sum
            #   S~_tau = (sum_{p in P_tau} x_{p,tau}) + Lap(Delta_S/((1-rho)*eps_tau))
            # with the data-independent sum sensitivity Delta_S = R and the
            # (1-rho)*eps_tau = pub_eps share, then post-processes the mean as
            #   gamma_tau = S~_tau / max(n~_tau, 1)
            # -- a NOISY SUM over the NOISY COUNT.  pooled_sum is the ACTUAL
            # summed data passed by the caller (NOT aggregate*n: that reconstructs
            # the sum from the mean and is only used as a fallback when the caller
            # cannot supply the real sum).  Dividing the true sum + Laplace noise
            # by the independent noisy count n_dp is what makes the count noise
            # propagate into gamma (it does not cancel).
            pooled_sum = true_sum if true_sum is not None else aggregate * n_tau
            sum_scale = self.config.sum_noise_scale(pub_eps)  # Delta_S/((1-rho)eps_tau)
            noisy_sum = pooled_sum + np.random.laplace(loc=0.0, scale=sum_scale)
            released = noisy_sum / max(n_dp, 1)   # n_dp = the noisy publisher count
        else:
            # Kellaris baselines: exact public count, direct noised mean with
            # sensitivity uses the post-P_max effective multiplicity (n_eff).
            lam = self.config.noise_scale(pub_eps, n_eff)
            released = aggregate + np.random.laplace(loc=0.0, scale=lam)

        self.last_released = released
        self.last_true_aggregate = aggregate
        self._record(aggregate, released, eps_count_tau + total_eps, n_eff, deferred=False)
        return released

    def _record(self, aggregate, released, eps_tau, n_tau, deferred):
        self.last_was_deferred = bool(deferred)
        self.budget_window.append(eps_tau)
        # Publications (not deferrals) define eligibility for n_window and
        # the release/deferral counters.  The dissim budget that BD/BA spend
        # on a skipped tau counts toward the sliding-window budget but not
        # toward the publication rate.
        if not deferred:
            self.n_window.append(n_tau)
            self.releases += 1
        else:
            self.deferrals += 1
        self.true_values.append(aggregate)
        self.noisy_values.append(released)
        self.budgets_spent.append(eps_tau)
        self.pub_counts.append(n_tau)
        self.deferred_flags.append(bool(deferred))


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
        # U_tau is defined over the w-event window of exactly `window_size`
        # logical timestamps (paper Sec. 6.10).  Use that window verbatim;
        # compute_kl_divergence already returns NaN when a window has too few
        # finite samples for a stable histogram, so no fixed-size override is
        # needed (and a fixed override would silently report a different window
        # length than the configured w).
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
