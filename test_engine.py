#!/usr/bin/env python3
"""DP-engine correctness tests — guard the w-event budget invariant + metrics.

These exist primarily to lock down the SLIDING-WINDOW BUDGET GUARANTEE
(paper Definition: sum_{j=tau-w+1}^{tau} eps_j <= eps for every window) across
ALL budget-allocation strategies, including the BD/BA Kellaris sub-mechanisms
whose forward/absorb accounting is subtle.  Any change to the budget logic in
dp_engine.py must keep these green.

Run:  python test_engine.py     (exits non-zero on any failure)
"""
from __future__ import annotations

import numpy as np

from dp_engine import (
    BudgetStrategy, PrivacyConfig, StreamState,
    attribution_advantage, compute_windowed_kl_divergence,
)

ALL_STRATEGIES = [
    "uniform", "sample", "budget_distribution", "budget_absorption",
    "p_gated_uniform", "p_gated_sample", "p_gated_bd", "p_gated_ba",
    "n_weighted",
]


def _synth_stream(T=400, n_pub=6, base=50.0, amp=10.0, seed=0):
    """A deterministic (aggregate, n_tau) stream with varying pool size."""
    rng = np.random.default_rng(seed)
    agg, cnt = [], []
    for t in range(T):
        # Pool size oscillates in [1, n_pub] so the P-gate sometimes fires.
        n = 1 + int((n_pub - 1) * (0.5 + 0.5 * np.sin(0.05 * t)))
        agg.append(float(base + amp * np.sin(0.1 * t) + rng.normal(0, 1.0)))
        cnt.append(n)
    return agg, cnt


def _run(strategy, w=8, eps=1.0, P=2, R=100.0, P_max=None, eps_count=0.0,
         seed=0, T=400):
    np.random.seed(seed)
    cfg = PrivacyConfig(
        epsilon=eps, window_size=w, min_publishers=P, payload_bound=R,
        strategy=BudgetStrategy(strategy), epsilon_count=eps_count,
        max_publishers=P_max,
    )
    st = StreamState(config=cfg)
    agg, cnt = _synth_stream(T=T, seed=seed)
    for a, n in zip(agg, cnt):
        st.release(a, n)
    return st


def test_window_budget_invariant():
    """For every strategy: the budget spent over ANY w consecutive timestamps
    must never exceed epsilon (the w-event DP guarantee)."""
    w, eps = 8, 1.0
    tol = 1e-9
    for strat in ALL_STRATEGIES:
        for seed in (0, 1, 2):
            st = _run(strat, w=w, eps=eps, seed=seed)
            spent = st.budgets_spent
            assert len(spent) > 0
            for i in range(len(spent)):
                window = spent[max(0, i - w + 1): i + 1]
                s = sum(window)
                assert s <= eps + tol, (
                    f"{strat} seed={seed}: window ending at {i} spent {s:.6f} > eps={eps}")
    print(f"  [ok] window budget invariant holds for all {len(ALL_STRATEGIES)} strategies")


def test_window_budget_invariant_pmax_and_count():
    """Invariant must also hold with P_max capping and a DP count budget."""
    w, eps = 6, 1.0
    tol = 1e-9
    for strat in ("p_gated_ba", "p_gated_bd", "n_weighted", "p_gated_uniform"):
        st = _run(strat, w=w, eps=eps, P=2, P_max=4, eps_count=0.1, seed=3)
        spent = st.budgets_spent
        for i in range(len(spent)):
            s = sum(spent[max(0, i - w + 1): i + 1])
            assert s <= eps + tol, f"{strat}: window at {i} spent {s:.6f} > {eps}"
        # eps_count is charged INSIDE the window budget (it is part of the
        # budgets_spent the invariant above checks) — it must be > 0 here since
        # the gate ran and drew DP counts.
        assert st.eps_count_spent > 0, f"{strat}: eps_count not charged"
    print("  [ok] invariant holds with P_max cap + eps_count charged inside eps")


def test_deferred_flags_consistent():
    """deferred_flags must be the authoritative released/deferred record, and
    releases + deferrals must partition the stream exactly."""
    for strat in ALL_STRATEGIES:
        st = _run(strat, seed=0)
        T = len(st.deferred_flags)
        assert st.releases + st.deferrals == T, (
            f"{strat}: releases({st.releases})+deferrals({st.deferrals}) != T({T})")
        assert sum(1 for d in st.deferred_flags if not d) == st.releases
    print("  [ok] deferred_flags partition stream into releases + deferrals for all strategies")


def test_pgate_defers_below_P():
    """A high P_min on a small pool must defer (repeat last release) and never
    emit a fresh release.  Each gated timestamp spends exactly the count share
    rho * eps_tau (the noisy-count draw n~_tau) -- or 0 once the sliding window
    can no longer afford another count -- and never a publication share, with
    the w-event window sum always <= eps."""
    np.random.seed(0)
    eps, w, tol = 1.0, 8, 1e-9
    cfg = PrivacyConfig(epsilon=eps, window_size=w, min_publishers=99,
                        payload_bound=100.0,
                        strategy=BudgetStrategy.P_GATED_UNIFORM,
                        rho_split=0.5)
    ec = cfg.count_epsilon()   # rho * eps_tau = 0.5 * (1.0 / 8)
    st = StreamState(config=cfg)
    for t in range(50):
        st.release(50.0 + t, 3)   # pool of 3 always < P_min=99 -> always gated
    assert st.releases == 0, "no release should pass a P_min=99 gate on pool=3"
    # Every per-tau spend is either a count draw (rho*eps_tau) or nothing --
    # never a publication share.
    assert all(abs(b) < tol or abs(b - ec) < tol for b in st.budgets_spent), \
        "gated taus spend only the count share (or 0), never a release share"
    # The count spend still obeys the w-event window invariant.
    for i in range(len(st.budgets_spent)):
        assert sum(st.budgets_spent[max(0, i - w + 1): i + 1]) <= eps + tol
    assert st.eps_count_spent > 0, "the DP count is still paid at the gate"
    print("  [ok] P-gate defers below P_min: 0 releases, only eps_count spent, window invariant holds")


def test_uniform_scale_formula():
    """Uniform: released noise scale must equal R/(n_eff * (eps/w))."""
    np.random.seed(0)
    R, w, eps, n = 80.0, 8, 1.0, 5
    cfg = PrivacyConfig(epsilon=eps, window_size=w, min_publishers=1,
                        payload_bound=R, strategy=BudgetStrategy.UNIFORM)
    st = StreamState(config=cfg)
    expected = R / (n * (eps / w))
    got = cfg.noise_scale(eps / w, n)
    assert abs(got - expected) < 1e-9, f"scale {got} != {expected}"
    print(f"  [ok] uniform Laplace scale = R/(n*eps/w) = {expected:.1f}")


def test_gate_override_single_sourced():
    """When the caller supplies n_gate (the broker's Algorithm-1 DP count), the
    engine gates on THAT, not on a freshly-drawn or exact count — so there is no
    second, inconsistent gate.  A high n_tau with a low n_gate must defer; a low
    (but >0) n_tau with a high n_gate must release."""
    np.random.seed(0)
    cfg = PrivacyConfig(epsilon=1.0, window_size=8, min_publishers=5,
                        payload_bound=80.0,
                        strategy=BudgetStrategy.P_GATED_UNIFORM,
                        epsilon_count=0.0)
    st = StreamState(config=cfg)
    # n_tau=10 (>=P) but caller's gate count says 2 (<P): must DEFER.
    out = st.release(50.0, 10, n_gate=2)
    assert st.deferred_flags[-1] is True and st.releases == 0, \
        "supplied n_gate < P_min must defer regardless of n_tau"
    # n_tau=2 but caller's gate count says 9 (>=P): must RELEASE.
    out = st.release(50.0, 2, n_gate=9)
    assert st.deferred_flags[-1] is False and st.releases == 1, \
        "supplied n_gate >= P_min must release (n_tau>0)"
    # n_gate must NOT charge eps_count (caller already paid it).
    assert st.eps_count_spent == 0.0, "engine must not charge eps_count when n_gate supplied"
    print("  [ok] gate is single-sourced via n_gate override; no engine-side recount")


def test_ba_invariant_holds_under_cap():
    """Budget Absorption with a tiny window and frequent releases still respects
    the w-event invariant AND keeps producing releases (the cap-rollback must
    not deadlock BA into permanent deferral)."""
    w, eps = 4, 1.0
    st = _run("budget_absorption", w=w, eps=eps, P=1, seed=7, T=300)
    spent = st.budgets_spent
    for i in range(len(spent)):
        s = sum(spent[max(0, i - w + 1): i + 1])
        assert s <= eps + 1e-9, f"BA window at {i} spent {s:.6f} > {eps}"
    assert st.releases > 0, "BA must still release sometimes (cap rollback didn't deadlock it)"
    print(f"  [ok] BA respects window invariant under cap; {st.releases} releases / {len(spent)} taus")


def test_attribution_advantage_released_only():
    """attribution_advantage averages 1/n over RELEASED timestamps only and
    ignores deferred ones (which leak nothing)."""
    pub = [2, 4, 0, 5]
    deferred = [False, False, True, False]
    adv = attribution_advantage(pub, deferred)
    expected = np.mean([1 / 2, 1 / 4, 1 / 5])
    assert abs(adv - expected) < 1e-9, f"{adv} != {expected}"
    print(f"  [ok] attribution advantage over released-only = {adv:.4f}")


def main():
    tests = [
        test_window_budget_invariant,
        test_window_budget_invariant_pmax_and_count,
        test_deferred_flags_consistent,
        test_pgate_defers_below_P,
        test_uniform_scale_formula,
        test_gate_override_single_sourced,
        test_ba_invariant_holds_under_cap,
        test_attribution_advantage_released_only,
    ]
    print(f"Running {len(tests)} DP-engine tests...")
    for t in tests:
        t()
    print("ALL DP-ENGINE TESTS PASSED")


if __name__ == "__main__":
    main()
