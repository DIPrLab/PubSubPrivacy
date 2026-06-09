#!/usr/bin/env python3
"""Tests for the Introduction's extreme baselines (no dataset files needed).

Verifies:
  * Extreme 2 / local DP  (run_ldp_on_per_pub): per-publisher input perturbation
    with Laplace scale R*w/eps, attribution advantage 1.0, and that it is
    strictly noisier than the pooled output-DP release (the paper's whole point).
  * Extreme 1.1 (extreme1_1_per_type): one stream per publisher type, producing
    a finite per-type-vs-topic distortion for every sensor type.
  * Extreme 1 (extreme1_global) and Extreme 2 figure (extreme2_per_publisher)
    run end-to-end and write their artifacts.

Run:  python test_extremes.py      (exits non-zero on any failure)
"""
from __future__ import annotations

import os
import tempfile

import numpy as np

from experiments import engine as R
from experiments import intro as I   # extreme1_global / extreme1_1_per_type / extreme2_per_publisher moved here


def _synth_per_pub(n_pub=6, T=200, base=50.0, amp=10.0, seed=0):
    """Deterministic per-publisher trace: each publisher a phase-shifted sine."""
    rng = np.random.default_rng(seed)
    per_pub = {}
    for p in range(n_pub):
        phase = 2 * np.pi * p / n_pub
        series = [float(base + amp * np.sin(0.1 * t + phase) + rng.normal(0, 1.0))
                  for t in range(T)]
        per_pub[f"dev{p}"] = series
    return per_pub


def test_local_dp_noise_scale():
    per_pub = _synth_per_pub()
    R_clamp, w, eps = 100.0, 8, 1.0
    res = R.run_ldp_on_per_pub(per_pub, R_clamp, eps, w, seed=1)
    m = res["metrics"]
    expected_scale = R_clamp * w / eps
    assert abs(m["ldp_noise_scale"] - expected_scale) < 1e-9, \
        f"LDP scale {m['ldp_noise_scale']} != R*w/eps {expected_scale}"
    assert m["attribution_advantage"] == 1.0, "LDP must expose every publisher (adv=1)"
    assert m["release_rate"] == 1.0, "LDP releases every populated timestamp"
    assert np.isfinite(m["normalized_mae"]) and m["normalized_mae"] > 0
    print(f"  [ok] local DP: scale={m['ldp_noise_scale']:.1f} (=R*w/eps), "
          f"NMAE={m['normalized_mae']:.3f}, attr_adv=1.0")


def test_local_dp_noisier_than_pooled():
    """Extreme 2 (local/input DP) must be strictly noisier than pooled output-DP."""
    per_pub = _synth_per_pub(n_pub=8)
    R_clamp, w, eps = 100.0, 8, 1.0
    agg, cnt = R._adaptive_interval_rebuild(per_pub, P=1, k_ext=0)  # native Dt pool
    ldp = R.run_ldp_on_per_pub(per_pub, R_clamp, eps, w, seed=2)["metrics"]
    pooled = R.run_dp_on_stream(agg, cnt, epsilon=eps, window_size=w,
                                min_publishers=1, payload_bound=R_clamp,
                                strategy="uniform", seed=2)["metrics"]
    assert ldp["normalized_mae"] > pooled["normalized_mae"], (
        f"LDP NMAE {ldp['normalized_mae']:.3f} should exceed pooled "
        f"{pooled['normalized_mae']:.3f}")
    print(f"  [ok] LDP NMAE {ldp['normalized_mae']:.3f} > pooled output-DP "
          f"{pooled['normalized_mae']:.3f} (input vs output privacy)")


def test_extreme1_1_per_type():
    # Two sensor types, each with several publishers.
    per_pub_all = {
        "temperature": (_synth_per_pub(n_pub=5, base=20, amp=5, seed=10), 80.0),
        "power_kw": (_synth_per_pub(n_pub=6, base=30, amp=15, seed=11), 50.0),
    }
    sensor_streams = {}
    for s, (pp, B) in per_pub_all.items():
        agg, cnt = R._adaptive_interval_rebuild(pp, P=1, k_ext=0)
        sensor_streams[s] = (agg, cnt, B)
    with tempfile.TemporaryDirectory() as d:
        df = I.extreme1_1_per_type(sensor_streams, per_pub_all, "synthtest", d)
        assert os.path.exists(
            os.path.join(d, "synthtest_figure_extreme1_1_per_type.csv"))
        # Pure-type aggregation: every publisher of a type pooled into one
        # release; subscriptions evaluated per hierarchy level.
        assert "kl_vs_type_release" in df.columns
        assert "subscription_level" in df.columns
        assert df["kl_vs_type_release"].notna().all(), "per-level KL must be finite"
        assert (df["n_publishers"] > 1).all(), "per-type must pool >1 publisher"
        assert set(df["sensor_type"]) == {"temperature", "power_kw"}
    print(f"  [ok] Extreme 1.1 pure-type: {len(df)} (type,level) rows, "
          f"mean KL vs type release={df['kl_vs_type_release'].mean():.3f}")


def test_extreme1_and_extreme2_figures():
    per_pub_all = {"power_kw": (_synth_per_pub(n_pub=5, base=30, amp=12, seed=20), 50.0)}
    sensor_streams = {}
    for s, (pp, B) in per_pub_all.items():
        agg, cnt = R._adaptive_interval_rebuild(pp, P=1, k_ext=0)
        sensor_streams[s] = (agg, cnt, B)
    with tempfile.TemporaryDirectory() as d:
        I.extreme1_global(sensor_streams, d, "synthtest")
        assert os.path.exists(os.path.join(d, "synthtest_figure_extreme1_global.csv"))
        pp, B = per_pub_all["power_kw"]
        I.extreme2_per_publisher(pp, B, "power_kw", "synthtest", d)
        assert os.path.exists(
            os.path.join(d, "synthtest_figure_extreme2_per_publisher.csv"))
    print("  [ok] Extreme 1 (global) + Extreme 2 (per-publisher) figures written")


def main():
    tests = [
        test_local_dp_noise_scale,
        test_local_dp_noisier_than_pooled,
        test_extreme1_1_per_type,
        test_extreme1_and_extreme2_figures,
    ]
    print(f"Running {len(tests)} extreme-baseline tests...")
    for t in tests:
        t()
    print("ALL EXTREME-BASELINE TESTS PASSED")


if __name__ == "__main__":
    main()
