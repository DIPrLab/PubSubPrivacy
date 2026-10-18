"""Offline simulation of the full PAWS broker pipeline, one subscription at a time.

The older offline experiments pre-build an (aggregate, count) stream with
``engine._adaptive_interval_rebuild`` and then privatize it.  That rebuild
decides how long to hold an interval open on the EXACT publisher count (not a
private decision) and never walks the topic hierarchy, so it cannot measure the
components this paper is about.  ``simulate_subscription`` instead replays the
raw per-publisher trace through the same decisions the broker plugin makes,
with the same privacy accounting:

  1. open a logical interval of base width Delta_t (``base_dt`` raw slots);
  2. adaptive interval (paper Sec. 4.1, ``wait="predictive"``, default): the
     width is fixed up front as clip(P_aim / lambda, Delta_t, K_ext * Delta_t)
     from the discounted rate of past noisy counts (free by post-processing);
     at the close, a too-short interval whose count misses P_min gets at most
     one corrective Delta_t extension when the window can pay for the extra
     count.  Alternatives: ``"early_stop"`` (Sparse Vector rule, slot by slot,
     close as soon as P_min is met, up to (1 + K_ext) * Delta_t),
     ``"reactive"`` (whole-Delta_t extension, one count per check), ``"none"``;
  3. hierarchy walk: draw the DP count of the subscribed scope and, while it is
     below P_min, walk up to the parent scope using cached delta counts
     (``count_cache=True``) or fresh per-level counts (``count_cache=False``),
     one count share per probed level either way;
  4. release through ``dp_engine.StreamState`` (noisy sum / noisy count, the
     configured budget strategy, the sliding-window invariant), charging every
     count share spent above.  A deferred timestamp repeats the previous
     output (``repeat_on_defer=False`` delivers nothing -- the ablation).

Utility is measured against the SUBSCRIBED scope's true aggregate over the
interval (``nmae``), which includes the semantic error of answering from an
ancestor after a walk; ``nmae_release_scope`` measures against the scope that
was actually released (DP noise only).  ``walk_kl`` measures the semantic
shift a walk introduces: D_KL(P || Q) between the readings of the subscribed
scope's publishers (P) and those of the walked-up pool actually released (Q),
over every fresh release that walked, binned on the public clamp interval.

Baselines run on the same base intervals: ``ldp`` (each publisher perturbs its
own value with Lap(R*w/eps), P_min = 1), ``wevent`` (Kellaris et al. uniform
w-event on the subscribed scope's aggregate, exact count, no pooling) and
``classic`` (no privacy).

All randomness goes through numpy's global RNG, seeded once per call, like the
rest of the engine.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from data_streams import DATASETS
from dp_engine import (
    BudgetStrategy,
    MonotoneAboveThreshold,
    PredictiveInterval,
    PrivacyConfig,
    StreamState,
    compute_kl_divergence,
    is_p_gated,
    release_utility,
    walk_population_counts,
)

WAIT_POLICIES = ("none", "predictive", "reactive", "early_stop")
APPROACHES = ("classic", "ldp", "wevent", "paws")

# Native duration of one raw slot per dataset (the resampling window the loader
# uses), so induced waits can be reported in seconds as well as Delta_t units.
# wearable rows carry no timestamp; its waits are reported in Delta_t only.
SLOT_SECONDS = {
    "energy": 300.0,
    "traffic": 10.0,
    "wearable": None,
    "pune": 3600.0,
    "mobility": 1800.0,
    "manufacturing": 600.0,
}

# The epsilon grid used by the Section 6 utility experiments.
PAPER_EPS_VALUES = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0]
PAPER_W_VALUES = [4, 8, 12, 16, 20, 24]


# ═════════════════════════════════════════════════════════════════════════
#  Per-publisher trace as dense arrays
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class PubMatrix:
    """A sensor's per-publisher trace as dense arrays over raw slots.

    present[i, t] -- publisher i reported in slot t
    vals[i, t]    -- its reading in slot t (0 where absent)
    ccnt          -- prefix counts of reports along time (activity in [s, e)
                     costs one vector subtraction)
    last[i, t]    -- the latest slot < t in which publisher i reported (-1 if
                     none), so each publisher's LATEST message in any interval
                     [s, e) is vals[i, last[i, e]] when last[i, e] >= s.
    """
    pubs: list
    present: np.ndarray
    vals: np.ndarray
    ccnt: np.ndarray
    last: np.ndarray
    T: int

    @classmethod
    def from_per_pub(cls, per_pub: dict, pubs: list | None = None) -> "PubMatrix":
        pubs = list(pubs) if pubs is not None else list(per_pub.keys())
        T = len(per_pub[pubs[0]]) if pubs else 0
        vals = np.zeros((len(pubs), T), dtype=float)
        present = np.zeros((len(pubs), T), dtype=bool)
        for i, p in enumerate(pubs):
            series = per_pub[p]
            for t in range(T):
                v = series[t]
                if v is not None and np.isfinite(v):
                    vals[i, t] = v
                    present[i, t] = True
        ccnt = np.zeros((len(pubs), T + 1))
        ccnt[:, 1:] = np.cumsum(present, axis=1)
        last = np.full((len(pubs), T + 1), -1, dtype=np.int64)
        if T:
            seen = np.where(present, np.arange(T)[None, :], -1)
            last[:, 1:] = np.maximum.accumulate(seen, axis=1)
        return cls(pubs, present, vals, ccnt, last, T)

    def window(self, s: int, e: int) -> tuple[np.ndarray, np.ndarray]:
        """(active mask, per-publisher value) over slots [s, e).

        Each active publisher contributes ONE value per logical timestamp --
        its LATEST message in the interval, as the broker keeps -- matching the
        DP model in which a publisher is one row of D_tau."""
        li = self.last[:, e]
        active = li >= s
        rows = np.arange(len(self.pubs))
        latest = np.where(active, self.vals[rows, np.maximum(li, 0)], 0.0)
        return active, latest


# ═════════════════════════════════════════════════════════════════════════
#  Subscriptions on the topic hierarchy
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class Subscription:
    """One subscription bound at a point of the topic tree.

    chain[k] is the publisher mask of the k-th scope a walk visits, starting at
    the subscribed scope and ending at the domain root (all scopes are nested).
    ``chain_key`` identifies the distinct publisher sets on the chain, so two
    subscriptions that see exactly the same pools (e.g. a device node and its
    single metric leaf) can share one simulation.
    """
    level: int
    scope: str
    depth: int
    n_pubs: int
    chain: list
    chain_key: tuple

    @property
    def scope_kind(self) -> str:
        if self.level == 1:
            return "root"
        return "leaf" if self.level == self.depth else "intermediate"


def subscriptions_for(dataset: str, sensor: str, pubs: list) -> list[Subscription]:
    """Every (level, subtree) subscription of a sensor's topic tree.

    Level 1 is the domain root; the deepest level is the per-publisher leaf.
    Without a topic mapping the sensor is one flat subscription."""
    spec = DATASETS.get(dataset, {})
    topic_of = spec.get("publisher_topic")
    paths = {}
    if topic_of is not None:
        for p in pubs:
            try:
                paths[p] = topic_of(p, sensor).split("/")
            except Exception:
                continue
    index = {p: i for i, p in enumerate(pubs)}
    if not paths:
        mask = np.ones(len(pubs), dtype=bool)
        return [Subscription(1, sensor, 1, len(pubs), [mask],
                             (tuple(range(len(pubs))),))]
    depth = max(len(v) for v in paths.values())

    def _mask(prefix_parts):
        L = len(prefix_parts)
        m = np.zeros(len(pubs), dtype=bool)
        for p, segs in paths.items():
            if segs[:L] == prefix_parts:
                m[index[p]] = True
        return m

    subs = []
    seen = set()
    for L in range(1, depth + 1):
        for p, segs in paths.items():
            prefix = tuple(segs[:L])
            if (L, prefix) in seen:
                continue
            seen.add((L, prefix))
            chain = [_mask(list(prefix[:k])) for k in range(L, 0, -1)]
            key = tuple(tuple(np.flatnonzero(m)) for i, m in enumerate(chain)
                        if i == 0 or not np.array_equal(m, chain[i - 1]))
            subs.append(Subscription(L, "/".join(prefix), depth,
                                     int(chain[0].sum()), chain, key))
    return subs


def busiest_chain(dataset: str, sensor: str, per_pub: dict) -> list[Subscription]:
    """The subscriptions along the walk of the busiest publisher's leaf, one
    per DISTINCT pool (leaf -> ... -> root).  Used as the grid search's
    representative set of subscription depths."""
    pubs = list(per_pub.keys())
    busiest = max(pubs, key=lambda p: sum(v is not None for v in per_pub[p]))
    i = pubs.index(busiest)
    subs = [s for s in subscriptions_for(dataset, sensor, pubs) if s.chain[0][i]]
    out, keys = [], set()
    for s in sorted(subs, key=lambda s: -s.level):
        if s.chain_key not in keys:
            keys.add(s.chain_key)
            out.append(s)
    return out


# ═════════════════════════════════════════════════════════════════════════
#  The simulator
# ═════════════════════════════════════════════════════════════════════════

def _nan_stats(values) -> tuple[float, float, float]:
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(a.mean()), float(np.percentile(a, 95)), float(a.max())


def simulate_subscription(
    mat: PubMatrix,
    chain: list,
    *,
    approach: str = "paws",
    epsilon: float,
    w: int,
    payload_bound: float,
    P_min: int = 1,
    P_max: int | None = None,
    strategy: str = "p_gated_uniform",
    rho: float = 0.2,
    base_dt: int = 1,
    k_ext: int = 0,
    wait: str = "predictive",
    walk: bool = True,
    count_cache: bool = True,
    seed: int = 0,
    keep_series: bool = False,
    repeat_on_defer: bool = True,
    center: float | None = None,
    count_smoothing: int = 0,
) -> dict:
    """Run one subscription through PAWS (or a baseline) and score it.

    Returns {"metrics": {...}} and, with ``keep_series``, the per-timestamp
    series {"start", "end", "truth", "released", "level", "n"} (slot indices).
    """
    if approach not in APPROACHES:
        raise ValueError(f"unknown approach {approach!r}")
    if wait not in WAIT_POLICIES:
        raise ValueError(f"unknown wait policy {wait!r}")
    np.random.seed(seed)
    R = float(payload_bound)
    base_dt = max(1, int(base_dt))
    k_ext = max(0, int(k_ext))
    # P_max is a PAWS knob; the baselines never truncate their pools.
    P_max = (int(P_max) if approach == "paws" and P_max and P_max == P_max
             else None)
    t_wall = time.perf_counter()

    if approach == "paws":
        strat = BudgetStrategy(strategy)
    else:
        strat = BudgetStrategy.UNIFORM
    pop_aware = approach == "paws" and is_p_gated(strat)
    masks = list(chain) if (pop_aware and walk) else [chain[0]]
    m0 = masks[0]
    same = [False] + [bool(np.array_equal(masks[k], masks[k - 1]))
                      for k in range(1, len(masks))]

    stream = None
    if approach in ("paws", "wevent"):
        stream = StreamState(config=PrivacyConfig(
            epsilon=float(epsilon), window_size=int(w),
            min_publishers=int(P_min) if pop_aware else 1,
            payload_bound=R, strategy=strat,
            rho_split=float(rho) if pop_aware else 0.0,
            max_publishers=None, repeat_on_defer=bool(repeat_on_defer),
            # Estimator (PAWS only): centered sum and smoothed, clipped count.
            center=(float(center) if (pop_aware and center is not None) else None),
            count_smoothing=int(count_smoothing)))
    ec = stream.config.count_epsilon() if (stream is not None and pop_aware) else 0.0
    ldp_scale = R * w / epsilon if epsilon > 0 else float("inf")
    can_wait = pop_aware and k_ext > 0 and wait != "none"
    sizer = (PredictiveInterval(base_dt, k_ext, int(P_min), P_max, w)
             if can_wait and wait == "predictive" else None)

    starts, ends, truths, released_l, levels, ns = [], [], [], [], [], []
    rel_truths, waits, units_l, fresh_l = [], [], [], []
    walk_p, walk_q = [], []   # subscribed-scope vs walked-pool readings
    T = mat.T
    s = 0
    while s < T:
        e = min(s + base_dt, T)
        cap = min(s + (1 + k_ext) * base_dt, T)
        units = 0
        leaf_noisy = None
        afford = stream.affordable_count_units() if pop_aware else 0

        # ---- adaptive interval ------------------------------------------
        if sizer is not None:
            # Eq. 3: width fixed up front from past released counts (whole
            # raw slots offline), capped at T_max = K_ext * Delta_t.
            t_max = min(s + int(sizer.t_max), T)
            e = min(s + max(base_dt, int(round(sizer.width()))), t_max)
            # A mis-sized interval is caught by the count the release pays for
            # anyway; while the window can also pay for a recount, an
            # interval below P_min gets one corrective Delta_t extension.
            if ec > 0 and afford >= 2 and e + base_dt <= t_max:
                active = (mat.ccnt[:, e] - mat.ccnt[:, s]) > 0
                leaf_noisy = (np.count_nonzero(active & m0)
                              + float(np.random.laplace(0.0, 1.0 / ec)))
                units += 1
                if max(0, int(round(leaf_noisy))) < P_min:
                    e += base_dt
                    leaf_noisy = None   # the release counts the longer interval
        elif can_wait and e < cap:
            if wait == "early_stop":
                # The stopping rule's share plus the release count's share
                # must both fit in the window; otherwise release at Delta_t.
                if afford >= 2:
                    svt = MonotoneAboveThreshold(P_min, ec)
                    units += 1
                    active = (mat.ccnt[:, e] - mat.ccnt[:, s]) > 0
                    while not svt.test(int(np.count_nonzero(active & m0))) and e < cap:
                        active |= mat.present[:, e]
                        e += 1
            else:  # reactive: one fresh leaf count per whole-Delta_t check
                while afford - units >= 1:
                    active = (mat.ccnt[:, e] - mat.ccnt[:, s]) > 0
                    leaf_noisy = (np.count_nonzero(active & m0)
                                  + float(np.random.laplace(0.0, 1.0 / ec)))
                    units += 1
                    if max(0, int(round(leaf_noisy))) >= P_min or e >= cap:
                        break
                    e = min(e + base_dt, cap)
                    leaf_noisy = None

        active, means = mat.window(s, e)
        sub_active = active & m0
        truth = float(means[sub_active].mean()) if sub_active.any() else None

        walked = None
        if approach == "classic":
            out, lvl, n = truth, 0, int(sub_active.sum())
            rel_truth = truth
        elif approach == "ldp":
            idx = np.flatnonzero(sub_active)
            if idx.size:
                noisy = means[idx] + np.random.laplace(0.0, ldp_scale, size=idx.size)
                out = float(noisy.mean())
            else:
                out = None
            lvl, n, rel_truth = 0, int(idx.size), truth
        else:
            if pop_aware:
                remaining = afford - units
                if leaf_noisy is None and ec > 0 and remaining < 1:
                    res_level, n_gate, walk_units = 0, 0, 0
                else:
                    res = walk_population_counts(
                        lambda k: int(np.count_nonzero(active & masks[k])),
                        len(masks), int(P_min), ec, cache=count_cache,
                        leaf_noisy=leaf_noisy, max_units=remaining,
                        same_as_below=lambda k: same[k])
                    res_level, n_gate, walk_units = res.level, res.noisy, res.units
                    if sizer is not None:
                        # Eq. 2: learn the arrival rate from the subscribed
                        # scope's released count over this interval's width.
                        sizer.observe(res.leaf, e - s)
                units += walk_units
                lvl = res_level
            else:
                lvl, n_gate = 0, None
            idx = np.flatnonzero(active & masks[lvl])
            if P_max is not None and idx.size > P_max:
                # P_max truncation (deterministic order, as in the plugin); the
                # denominator is capped the same way (post-processing).
                idx = idx[:P_max]
                if n_gate is not None:
                    n_gate = min(n_gate, P_max)
            n = int(idx.size)
            if lvl > 0 and n:
                walked = (means[sub_active], means[idx])
            pooled_sum = float(means[idx].sum())
            agg = pooled_sum / n if n else 0.0
            rel_truth = agg if n else None
            # Public cap on the denominator: publishers registered under the
            # released scope (the topic tree is public), after P_max.
            n_cap = int(masks[lvl].sum())
            if P_max is not None:
                n_cap = min(n_cap, P_max)
            out = stream.release(
                agg, n,
                n_gate=n_gate if pop_aware else None,
                n_count_draws=units if pop_aware else 1,
                true_sum=pooled_sum,
                count_key=tuple(np.flatnonzero(masks[lvl])), n_cap=n_cap)

        starts.append(s)
        ends.append(e)
        truths.append(truth)
        released_l.append(out)
        levels.append(lvl)
        ns.append(n)
        rel_truths.append(rel_truth)
        fresh_l.append(out is not None and not (stream is not None
                                                and stream.last_was_deferred))
        if walked is not None and fresh_l[-1]:
            walk_p.extend(walked[0].tolist())
            walk_q.extend(walked[1].tolist())
        waits.append(max(0.0, (e - s - base_dt) / base_dt))
        units_l.append(units)
        s = e

    # ---- metrics --------------------------------------------------------
    ru = release_utility(truths, released_l, R, prior=center)
    ru_rel = release_utility(rel_truths, released_l, R)
    fresh = [i for i, f in enumerate(fresh_l) if f]
    n_ts = len(released_l)
    walk_levels = [levels[i] for i in fresh]
    pooled = [ns[i] for i in fresh if ns[i] > 0]
    mean_wait, p95_wait, max_wait = _nan_stats(waits)
    # Attribution advantage 1/n_tau over releases; LDP exposes every publisher.
    if approach == "ldp":
        attribution = 1.0
    else:
        attribution = float(np.mean([1.0 / n for n in pooled])) if pooled else float("nan")
    metrics = {
        "normalized_mae": ru["nmae"],
        "mae": ru["mae"],
        "coverage": ru["coverage"],
        "nmae_penalized": ru["nmae_penalized"],
        "nmae_clip_penalized": ru["nmae_clip_penalized"],
        "skill": ru["skill"],
        "skill_penalized": ru["skill_penalized"],
        "rel_mae_penalized": ru["rel_mae_penalized"],
        "corr": ru["corr"],
        "skill_prior": ru["skill_prior"],
        "nmae_release_scope": ru_rel["nmae"],
        "release_rate": len(fresh) / n_ts if n_ts else float("nan"),
        "n_timestamps": n_ts,
        "n_scored": ru["n_scored"],
        "attribution_advantage": attribution,
        "avg_pooled_n": float(np.mean(pooled)) if pooled else float("nan"),
        "walk_rate": (float(np.mean([lv > 0 for lv in walk_levels]))
                      if walk_levels else float("nan")),
        "avg_walk_levels": float(np.mean(walk_levels)) if walk_levels else float("nan"),
        "walk_kl": compute_kl_divergence(
            walk_p, walk_q,
            value_range=(center - R / 2.0, center + R / 2.0)
            if center is not None else None),
        "mean_wait_dt": mean_wait,
        "p95_wait_dt": p95_wait,
        "max_wait_dt": max_wait,
        "wait_rate": float(np.mean([x > 0 for x in waits])) if waits else float("nan"),
        "count_units_per_ts": float(np.mean(units_l)) if units_l else 0.0,
        "eps_window_max": (max(sum(stream.budgets_spent[max(0, i - w + 1):i + 1])
                               for i in range(len(stream.budgets_spent)))
                           if stream is not None and stream.budgets_spent else 0.0),
        "compute_ms_per_ts": 1000.0 * (time.perf_counter() - t_wall) / max(1, n_ts),
    }
    out = {"metrics": metrics}
    if keep_series:
        out["series"] = {"start": starts, "end": ends, "truth": truths,
                         "released": released_l, "level": levels, "n": ns}
    return out


# ═════════════════════════════════════════════════════════════════════════
#  Parallel task plumbing shared by the Section 6 experiment modules
# ═════════════════════════════════════════════════════════════════════════
#
# Workers receive the per-sensor PubMatrix objects and subscription chains once
# through the pool initializer; a task only names them by key.

_WORKER_CTX: dict | None = None


def init_sim_worker(mats: dict, chains: dict) -> None:
    global _WORKER_CTX
    _WORKER_CTX = {"mats": mats, "chains": chains}


def sim_task(task) -> dict:
    """task = (meta, mat_key, chain_key, sim_kwargs) -> {**meta, **metrics}."""
    meta, mat_key, chain_key, kwargs = task
    mat = _WORKER_CTX["mats"][mat_key]
    chain = _WORKER_CTX["chains"][chain_key]
    res = simulate_subscription(mat, chain, **kwargs)
    return {**meta, **res["metrics"]}


def resolve_paws_params(core, grid_config, dataset, clamp_mode, strategy,
                        epsilon, args) -> dict:
    """PAWS hyperparameters for (dataset, strategy, epsilon): the Sec. 6.1 grid
    optimum when available, else the CLI defaults."""
    prm = core._resolve_params(
        grid_config, dataset, clamp_mode, strategy, epsilon,
        {"P_min": getattr(args, "ablation_P", 3), "P_max": None,
         "k_ext": getattr(args, "k_ext", 3), "delta_t": 1})
    return {
        "P_min": int(prm["P_min"]),
        "P_max": (int(prm["P_max"]) if prm.get("P_max") is not None
                  and prm["P_max"] == prm["P_max"] else None),
        "k_ext": int(prm.get("k_ext") or 0),
        "base_dt": int(prm.get("delta_t") or 1),
        "rho": float(prm["rho_split"]),
    }
