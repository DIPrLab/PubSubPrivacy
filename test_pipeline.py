#!/usr/bin/env python3
"""Tests for the pipeline components added for the Section 7 experiments:
cached hierarchy-walk counts, the predictive adaptive interval (Sec. 5.1.1) and
the Sparse Vector stopping rule, repeat-on-defer (off only in the ablation),
the delivered-value utility metric, and the w-event budget invariant through
the offline pipeline simulator and the broker plugin.

Run:  python test_pipeline.py     (exits non-zero on any failure)
"""
from __future__ import annotations

import numpy as np

from dp_engine import (
    BudgetStrategy, MonotoneAboveThreshold, PredictiveInterval, PrivacyConfig,
    StreamState,
    release_utility, walk_population_counts,
)
from experiments import engine as core
from experiments.pipeline import PubMatrix, simulate_subscription

TOL = 1e-9


def _window_max(budgets, w):
    return max((sum(budgets[max(0, i - w + 1):i + 1]) for i in range(len(budgets))),
               default=0.0)


def _synthetic_per_pub(n_pub=8, T=300, p_present=0.5, seed=0):
    rng = np.random.default_rng(seed)
    return {f"p{i}": [float(40 + 10 * np.sin(0.1 * t) + rng.normal(0, 2))
                      if rng.random() < p_present else None for t in range(T)]
            for i in range(n_pub)}


def test_cached_walk_charges_every_level():
    """Algorithm 1: every probed level is charged one count share, whether
    the level adds a cached delta count or counts the whole pool afresh."""
    counts = [0, 0, 1, 2, 9]          # only the root reaches P_min = 6
    np.random.seed(0)
    cached = walk_population_counts(lambda k: counts[k], len(counts), 6, 1e6,
                                    cache=True)
    np.random.seed(0)
    fresh = walk_population_counts(lambda k: counts[k], len(counts), 6, 1e6,
                                   cache=False)
    assert cached.met and cached.level == 4 and cached.units == 5, cached
    assert cached.draws == 5, "leaf + one delta per level, each charged"
    assert fresh.met and fresh.level == 4 and fresh.units == 5, fresh
    # With exact-ish noise the cached count telescopes to the root count.
    assert cached.noisy == 9 and fresh.noisy == 9
    print("  [ok] cached and per-level walks both charge 5 shares over 5 levels")


def test_cached_walk_reuses_paid_leaf_and_skips_same_levels():
    counts = [1, 1, 1, 7]
    calls = []

    def count_at(k):
        calls.append(k)
        return counts[k]

    np.random.seed(1)
    res = walk_population_counts(count_at, 4, 5, 1e6, cache=True, leaf_noisy=1.0,
                                 same_as_below=lambda k: k in (1, 2))
    assert res.met and res.level == 3 and res.units == 1, res
    assert res.draws == 1, "skipped single-child levels draw no noise and cost nothing"
    assert 1 not in calls and 2 not in calls
    print("  [ok] a caller-paid leaf is reused free; identical levels skipped; the probed level is charged")


def test_walk_respects_max_units():
    np.random.seed(2)
    res = walk_population_counts(lambda k: 0, 5, 3, 1e6, cache=False, max_units=2)
    assert res.units == 2 and not res.met
    res = walk_population_counts(lambda k: 0, 5, 3, 1e6, cache=True, max_units=0)
    assert res.units == 0 and not res.met
    print("  [ok] the walk never spends more shares than the window can afford")


def test_svt_stops_when_threshold_met():
    np.random.seed(3)
    svt = MonotoneAboveThreshold(threshold=4, epsilon=1e6)   # ~noiseless
    answers = [svt.test(c) for c in (1, 2, 3)]
    assert answers == [False, False, False]
    assert svt.test(4) is True and svt.halted and svt.queries == 4
    try:
        svt.test(5)
    except RuntimeError:
        pass
    else:
        raise AssertionError("AboveThreshold must halt after its one 'above'")
    print("  [ok] stopping rule stops at the first count >= P_min, then halts")


def test_deferral_repeats_unless_ablation():
    cfg = PrivacyConfig(epsilon=1.0, window_size=4, min_publishers=99,
                        payload_bound=10.0, strategy=BudgetStrategy.P_GATED_UNIFORM,
                        rho_split=0.0)
    st = StreamState(config=cfg)
    st.last_released = 5.0
    assert st.release(3.0, 2) == 5.0, "default: repeat the previous output"
    cfg.repeat_on_defer = False
    assert st.release(3.0, 2) is None and st.noisy_values[-1] is None
    print("  [ok] a deferral repeats the previous output; off (ablation) it delivers nothing")


def test_predictive_interval_sizing():
    """Paper Eqs. 2-3: seeded at Delta_t, widens toward 1.5 P_min / lambda when
    arrivals are sparse, never beyond K_ext * Delta_t, and narrows back when
    dense."""
    pi = PredictiveInterval(base=2.0, k_ext=4, p_min=4, p_max=8, w=4)
    assert pi.p_aim == 6.0 and pi.t_max == 8.0
    assert abs(pi.width() - 2.0) < TOL, "first interval is the base width"
    for _ in range(40):                       # 1 publisher per 2 time units
        pi.observe(1.0, 2.0)
    assert abs(pi.width() - 8.0) < TOL, "sparse arrivals widen to T_max"
    for _ in range(40):                       # 9 publishers per 2 time units
        pi.observe(9.0, 2.0)
    assert abs(pi.width() - 2.0) < TOL, "dense arrivals return to Delta_t"
    mid = PredictiveInterval(base=1.0, k_ext=8, p_min=6, p_max=None, w=1)
    mid.observe(2.0, 1.0)                     # w=1: lambda = last observation
    assert abs(mid.width() - 4.5) < TOL, "aim at 1.5 * P_min (9 / 2 per unit)"
    print("  [ok] predictive width: seeded at Delta_t, 1.5 P_min / lambda, clipped to [Delta_t, K_ext*Delta_t]")


def test_centered_sum_keeps_sensitivity_R():
    """Definition 3.1 neighbours change one payload anywhere in the clamp
    range, so centering the sum must not shrink its sensitivity below R."""
    for center in (None, 30.0):
        cfg = PrivacyConfig(epsilon=1.0, window_size=4, min_publishers=1, payload_bound=60.0,
                            center=center)
        assert cfg.sum_sensitivity() == 60.0, (center, cfg.sum_sensitivity())
    print("  [ok] sum sensitivity is R with and without centering")


def test_predictive_widens_sparse_scopes_for_free():
    """Offline pipeline: a sparse leaf gets much wider intervals than a dense
    root, and sizing itself spends no count share (at most the release count
    plus one corrective recount per timestamp)."""
    per_pub = _synthetic_per_pub(n_pub=8, T=400, p_present=0.3, seed=11)
    mat = PubMatrix.from_per_pub(per_pub)
    n = len(mat.pubs)
    leaf = np.zeros(n, bool); leaf[:2] = True
    kw = dict(approach="paws", epsilon=40.0, w=4, payload_bound=60.0, P_min=3,
              rho=0.4, k_ext=4, wait="predictive", walk=False, seed=12)
    sparse = simulate_subscription(mat, [leaf], **kw)["metrics"]
    dense = simulate_subscription(mat, [np.ones(n, bool)], **kw)["metrics"]
    # The 1.5 * P_min target (Eq. 3) widens every scope a little; a sparse
    # leaf must still wait much longer than a dense scope.
    assert sparse["mean_wait_dt"] > dense["mean_wait_dt"] + 1.0, (sparse, dense)
    assert sparse["count_units_per_ts"] <= 2.0 + TOL
    assert sparse["eps_window_max"] <= 40.0 + TOL
    print(f"  [ok] predictive wait: sparse leaf {sparse['mean_wait_dt']:.2f} dt, "
          f"dense root {dense['mean_wait_dt']:.2f} dt")


def test_release_utility_penalizes_missing():
    ru = release_utility([1.0, 2.0, 3.0, None], [1.5, None, 3.0, 9.0], 10.0)
    assert ru["n_scored"] == 3 and ru["n_fresh"] == 2
    assert abs(ru["nmae"] - 0.025) < TOL
    assert abs(ru["coverage"] - 2 / 3) < TOL
    assert abs(ru["nmae_penalized"] - (2 / 3 * 0.025 + 1 / 3)) < TOL
    print("  [ok] NMAE over fresh releases; undelivered timestamps cost 1.0 when penalized")


def test_simulator_window_invariant():
    """Every component combination keeps sum_{window} eps_j <= eps."""
    per_pub = _synthetic_per_pub()
    mat = PubMatrix.from_per_pub(per_pub)
    n = len(mat.pubs)
    leaf = np.zeros(n, bool); leaf[0] = True
    mid = np.zeros(n, bool); mid[:4] = True
    chain = [leaf, mid, np.ones(n, bool)]
    w, eps = 6, 1.0
    for strategy in ("p_gated_uniform", "n_weighted", "p_gated_ba"):
        for wait in ("none", "predictive", "reactive", "early_stop"):
            for walk in (False, True):
                for cache in (False, True):
                    for rho in (0.1, 0.45):
                        m = simulate_subscription(
                            mat, chain, approach="paws", epsilon=eps, w=w,
                            payload_bound=60.0, P_min=3, strategy=strategy, rho=rho,
                            base_dt=2, k_ext=2, wait=wait, walk=walk,
                            count_cache=cache, seed=4, center=30.0)["metrics"]
                        assert m["eps_window_max"] <= eps + TOL, (
                            strategy, wait, walk, cache, rho, m["eps_window_max"])
    for approach in ("wevent", "ldp", "classic"):
        m = simulate_subscription(mat, chain, approach=approach, epsilon=eps, w=w,
                                  payload_bound=60.0, seed=4)["metrics"]
        assert m["eps_window_max"] <= eps + TOL
    print("  [ok] simulator: window budget <= eps for every strategy x wait x walk x cache")


def test_no_crumb_releases():
    """When count shares leave only crumbs of the window, the engine must skip
    rather than release pure noise (regression: n-weighted released with
    ~1e-13 of the budget, NMAE ~1e12)."""
    per_pub = _synthetic_per_pub(p_present=0.9, seed=9)
    mat = PubMatrix.from_per_pub(per_pub)
    chain = [np.ones(len(mat.pubs), bool)]
    for strategy in ("n_weighted", "p_gated_uniform", "p_gated_ba"):
        for rho in (0.1, 0.3):
            res = simulate_subscription(
                mat, chain, approach="paws", epsilon=0.8, w=8, payload_bound=60.0,
                P_min=6, strategy=strategy, rho=rho, k_ext=2, seed=3,
                keep_series=True)
            s = res["series"]
            errs = [abs(r - t) / 60.0 for r, t in zip(s["released"], s["truth"])
                    if r is not None and t is not None]
            assert max(errs, default=0.0) < 1e4, (strategy, rho, max(errs))
    print("  [ok] no release on budget crumbs (bounded per-release error)")


def test_skill_metric():
    truth = [10.0, 12.0, 14.0, 16.0, 18.0]
    perfect = release_utility(truth, list(truth), 20.0)
    const = release_utility(truth, [14.0] * 5, 20.0)
    shrunk = release_utility(truth, [0.0] * 5, 20.0)
    assert abs(perfect["skill"] - 1.0) < TOL and abs(perfect["corr"] - 1.0) < TOL
    assert abs(const["skill"]) < TOL, "the best constant has skill 0"
    assert shrunk["skill"] < -4, "collapsing to 0 is far worse than the constant (-4.83)"
    half = release_utility(truth, [10.0, None, 14.0, None, 18.0], 20.0)
    assert half["skill"] == 1.0 and 0 < half["skill_penalized"] < 1.0
    print("  [ok] skill: perfect = 1, best constant = 0, shrink-to-0 << 0; misses cost the constant")


def test_smoothed_clipped_denominator():
    cfg = PrivacyConfig(epsilon=1.0, window_size=4, min_publishers=1,
                        payload_bound=10.0, strategy=BudgetStrategy.P_GATED_UNIFORM)
    st = StreamState(config=cfg)
    for c in (4, 8, 400, 4, 6):          # history keeps the last w = 4
        st._push_count("s", c)
    assert abs(st._denominator("s", 6, None) - 104.5) < TOL
    assert st._denominator("s", 6, 10) == 10.0, "clipped to the registered publishers"
    st._push_count("t", -30)
    assert st._denominator("t", -30, 10) == 1.0, "never below 1"
    assert PrivacyConfig(epsilon=1, window_size=4, min_publishers=1, payload_bound=80.0,
                         center=220.0).sum_sensitivity() == 80.0, "centered sum: still R"
    print("  [ok] denominator = mean of the last w noisy counts, clipped to [1, N]; centered sensitivity R")


def test_release_is_unbiased_far_from_zero():
    """Regression for the shrink-to-0 bias: values near 220 on a [180, 260]
    clamp, a tiny count share (rho = 0.05).  The released mean must stay near
    the truth and beat the best constant."""
    rng = np.random.default_rng(13)
    T, n_pub = 600, 10
    base = 220 + 15 * np.sin(np.arange(T) / 20.0)
    per_pub = {f"p{i}": [float(base[t] + rng.normal(0, 1)) for t in range(T)]
               for i in range(n_pub)}
    mat = PubMatrix.from_per_pub(per_pub)
    chain = [np.ones(n_pub, bool)]
    res = simulate_subscription(mat, chain, approach="paws", epsilon=40.0, w=4,
                                payload_bound=80.0, P_min=3, rho=0.05, k_ext=0,
                                wait="none", seed=14, center=220.0,
                                keep_series=True)
    s, m = res["series"], res["metrics"]
    pairs = [(t, r) for t, r in zip(s["truth"], s["released"])
             if t is not None and r is not None]
    ratio = np.mean([r for _, r in pairs]) / np.mean([t for t, _ in pairs])
    assert abs(ratio - 1.0) < 0.03, f"released/true mean = {ratio:.3f}"
    assert m["skill"] > 0.3 and m["corr"] > 0.6, m
    print(f"  [ok] unbiased far from 0: released/true mean = {ratio:.3f}, "
          f"skill = {m['skill']:.2f}, corr = {m['corr']:.2f}")


def test_latest_message_per_publisher():
    """Several messages from one publisher in an interval: the latest is used
    (broker buffer, pooled walk buffer, offline simulator, legacy rebuild)."""
    from plugin import TopicBuffer
    buf = TopicBuffer()
    buf.add("p1", 10.0, now=1.0)
    buf.add("p1", 30.0, now=2.0)
    buf.add("p1", 99.0, now=1.5)            # arrives late but is older
    buf.add("p2", 5.0, now=1.0)
    assert buf.payloads == {"p1": 30.0, "p2": 5.0} and buf.total() == 35.0
    per_pub = {"a": [1.0, None, 3.0, 4.0], "b": [None, 7.0, None, None]}
    mat = PubMatrix.from_per_pub(per_pub)
    active, vals = mat.window(0, 3)
    assert list(active) == [True, True] and list(vals) == [3.0, 7.0]
    active, vals = mat.window(2, 4)
    assert list(active) == [True, False] and vals[0] == 4.0
    agg, cnt = core._adaptive_interval_rebuild(per_pub, P=1, k_ext=0, base_dt=4)
    assert cnt == [2] and agg == [(4.0 + 7.0) / 2]
    print("  [ok] a publisher's latest message in the interval is the one used")


def test_plugin_pools_latest_across_topics():
    from plugin import PrivacyPlugin
    pl_ = PrivacyPlugin(sensor_bounds={"temp": (0.0, 100.0)})
    pl_._buffers["site/a/temp"].add("p1", 10.0, now=1.0)
    pl_._buffers["site/b/temp"].add("p1", 40.0, now=2.0)
    pooled = pl_._clamp_compatible_buffer("site", (0.0, 100.0), 100.0)
    assert pooled.payloads == {"p1": 40.0}, pooled.payloads
    print("  [ok] pooling across topics keeps a publisher's latest message")


def test_cached_walk_raises_coverage_at_leaf():
    """A one-publisher leaf can only release by walking up."""
    per_pub = _synthetic_per_pub(p_present=0.9, seed=5)
    mat = PubMatrix.from_per_pub(per_pub)
    n = len(mat.pubs)
    leaf = np.zeros(n, bool); leaf[0] = True
    chain = [leaf, np.ones(n, bool)]
    kw = dict(approach="paws", epsilon=50.0, w=4, payload_bound=60.0, P_min=4,
              rho=0.4, k_ext=0, wait="none", seed=6, repeat_on_defer=False)
    no_walk = simulate_subscription(mat, chain, walk=False, **kw)["metrics"]
    walk = simulate_subscription(mat, chain, walk=True, **kw)["metrics"]
    assert no_walk["coverage"] < 0.1 and walk["coverage"] > 0.8, (no_walk, walk)
    assert walk["walk_rate"] > 0.8
    print(f"  [ok] leaf coverage {no_walk['coverage']:.2f} -> {walk['coverage']:.2f} with the walk")


def test_clip_penalized_nmae_punishes_missing():
    """Clipped deliveries cost at most 1.0, a miss costs exactly 1.0, so
    delivering never scores worse than staying silent."""
    truth = [50.0, 50.0, 50.0, 50.0]
    wild = [500.0, None, -400.0, 55.0]        # two far outside [0, 100]
    m = release_utility(truth, wild, 100.0, prior=50.0)
    # clipped errors 0.5, (miss) 1.0, 0.5, 0.05 -> 2.05 / 4
    assert abs(m["nmae_clip_penalized"] - 2.05 / 4) < 1e-12, m
    assert m["nmae_penalized"] > 1.0 > m["nmae_clip_penalized"]
    silent = release_utility(truth, [None] * 4, 100.0, prior=50.0)
    assert silent["nmae_clip_penalized"] == 1.0
    assert np.isnan(release_utility(truth, wild, 100.0)["nmae_clip_penalized"])
    print("  [ok] clipped penalized NMAE: deliveries <= 1, misses = 1, silence scores worst")


def test_walk_kl_measures_semantic_shift():
    """walk_kl: ~0 when the walked-up pool reads like the subscribed leaf,
    large when the pool's readings sit elsewhere in the clamp range, and NaN
    when nothing walked."""
    base = _synthetic_per_pub(p_present=0.9, seed=9)
    shifted = {p: ([v + 60.0 if v is not None else None for v in vals]
                   if p != "p0" else vals) for p, vals in base.items()}
    n = len(base)
    leaf = np.zeros(n, bool); leaf[0] = True
    chain = [leaf, np.ones(n, bool)]
    kw = dict(approach="paws", epsilon=50.0, w=4, payload_bound=120.0, P_min=4,
              rho=0.4, k_ext=0, wait="none", seed=10, center=60.0)
    same = simulate_subscription(PubMatrix.from_per_pub(base), chain, **kw)["metrics"]
    far = simulate_subscription(PubMatrix.from_per_pub(shifted), chain, **kw)["metrics"]
    none = simulate_subscription(PubMatrix.from_per_pub(base), chain, walk=False,
                                 **kw)["metrics"]
    assert same["walk_rate"] > 0.8 and far["walk_rate"] > 0.8
    assert same["walk_kl"] < 0.1 < 1.0 < far["walk_kl"], (same["walk_kl"], far["walk_kl"])
    assert np.isnan(none["walk_kl"])
    print(f"  [ok] walk KL {same['walk_kl']:.3f} (similar pool) vs "
          f"{far['walk_kl']:.2f} (shifted pool); NaN without a walk")


def test_plugin_charges_wait_and_walk_counts():
    """Broker plugin: the window invariant holds with the adaptive interval and
    the walk, and the reactive extension checks are charged to the stream."""
    per_pub = _synthetic_per_pub(n_pub=6, T=120, p_present=0.4, seed=7)
    spec = {"static_clamps": {"temp": (0.0, 60.0)},
            "publisher_topic": lambda p, s: f"site/{'a' if p < 'p3' else 'b'}/{p}/{s}"}
    w = 6
    for wait, cache in (("predictive", True), ("early_stop", True),
                        ("reactive", False), ("none", True)):
        r = core._drive_plugin_scenario(
            per_pub, "temp", spec, "hierarchy", epsilon=1.0, w=w, P=3,
            delta_t=1.0, n_steps=120, seed=8, strategy="p_gated_uniform",
            k_ext=2, wait_policy=wait, count_cache=cache, rho_split=0.3)
        plugin = r["plugin"]
        for st in plugin.get_stream_states().values():
            assert _window_max(st.budgets_spent, w) <= 1.0 + TOL, wait
        charged = sum(sum(st.budgets_spent) for st in plugin.get_stream_states().values())
        delivered = sum(1 for x in plugin.release_log if x["released_value"] is not None)
        assert delivered == len(r["mock"].published), "every delivered value is published"
        # Every count share the plugin drew is inside some stream's charges.
        assert plugin.eps_count_spent <= charged + TOL, (wait, plugin.eps_count_spent, charged)
    print("  [ok] plugin: window invariant with wait + walk; every drawn count share is charged")


def main():
    tests = [
        test_cached_walk_charges_every_level,
        test_cached_walk_reuses_paid_leaf_and_skips_same_levels,
        test_walk_respects_max_units,
        test_svt_stops_when_threshold_met,
        test_deferral_repeats_unless_ablation,
        test_predictive_interval_sizing,
        test_centered_sum_keeps_sensitivity_R,
        test_predictive_widens_sparse_scopes_for_free,
        test_release_utility_penalizes_missing,
        test_latest_message_per_publisher,
        test_plugin_pools_latest_across_topics,
        test_skill_metric,
        test_smoothed_clipped_denominator,
        test_release_is_unbiased_far_from_zero,
        test_simulator_window_invariant,
        test_no_crumb_releases,
        test_cached_walk_raises_coverage_at_leaf,
        test_clip_penalized_nmae_punishes_missing,
        test_walk_kl_measures_semantic_shift,
        test_plugin_charges_wait_and_walk_counts,
    ]
    print(f"Running {len(tests)} pipeline tests...")
    for t in tests:
        t()
    print("ALL PIPELINE TESTS PASSED")


if __name__ == "__main__":
    main()
