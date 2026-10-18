"""
MQTT Privacy Plugin: broker-side middleware for clamped w-event DP with
P-allocation (paper Sections 3-5).

Pipeline:

    Publishers --> [raw topics] --> PrivacyPlugin --> [protected topics] --> Subscribers

The plugin:
  1. Clamps each incoming payload to the publisher's declared interval
     [a_p, b_p] (paper Sec. 6.1; static operator-declared bounds).
  2. Buffers messages per logical-timestamp interval.  With the default
     ``wait_policy="predictive"`` (paper Sec. 4.1) each interval's width is
     fixed up front as clip(P_aim / lambda, Delta_t, K_ext * Delta_t), where
     lambda is a discounted average of the leaf's past noisy counts per unit
     width -- free by post-processing; an interval whose closing count misses
     P_min gets at most one corrective Delta_t extension when the window can
     pay for the recount.  ``"early_stop"`` closes as soon as P_min is met (a
     Sparse Vector rule, one count share per interval), ``"reactive"`` is the
     original whole-Delta_t extension, ``"none"`` never waits.
  3. If the DP count is still below P_min, walks up the topic hierarchy
     (Algorithm 1) to the nearest clamp-compatible ancestor whose DP count
     reaches P_min, otherwise defers.  With ``count_cache`` each level adds a
     noisy count of only the publishers new at that level to the cached count
     below; each probed level is charged one count share (Algorithm 1).
  4. Releases the noisy sum over the noisy count with the configured
     budget-allocation strategy; every count share spent in steps 2-3 is
     charged inside the w-event window.
  5. Delivers only (hat{e}_tau, t_start_tau) on the protected topic; a
     deferred timestamp repeats the previous output (``repeat_on_defer``).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import paho.mqtt.client as mqtt

from dp_engine import (
    BudgetStrategy,
    MonotoneAboveThreshold,
    PredictiveInterval,
    PrivacyConfig,
    StreamState,
    is_p_gated,
    walk_population_counts,
)

logger = logging.getLogger(__name__)

WAIT_POLICIES = ("none", "predictive", "reactive", "early_stop")


@dataclass
class TopicBuffer:
    """Per-timestamp buffer: publisher_id -> clamped value.

    ``start_time`` is the wall-clock moment the current interval began
    accumulating messages, stamped on the first add after a clear.  Paper
    Definition 3.1 says adaptive extensions change the closing boundary
    but never the start, so we do NOT touch ``start_time`` across
    extensions — only ``clear()`` resets it.

    ``interval_start`` is when the current logical interval opened (the
    previous close, or the first message), against which the base width
    Delta_t and the cap (1 + K_ext) * Delta_t are measured.  ``svt``,
    ``leaf_noisy`` and ``count_units`` carry the waiting state of the open
    interval: the stopping rule, the latest reactive leaf count, and the count
    shares already spent on it (charged when the interval is released).
    """
    payloads: dict[str, float] = field(default_factory=dict)
    # Arrival time of each publisher's kept payload: a publisher contributes
    # its LATEST message in the interval (one value per publisher).
    arrivals: dict[str, float] = field(default_factory=dict)
    extensions: int = 0
    start_time: Optional[float] = None
    interval_start: Optional[float] = None
    svt: Optional[MonotoneAboveThreshold] = None
    leaf_noisy: Optional[float] = None
    count_units: int = 0
    target_width: Optional[float] = None   # predictive: width fixed at open
    corrected: bool = False                # predictive: extension used

    @property
    def num_publishers(self) -> int:
        return len(self.payloads)

    def mean(self) -> float:
        if not self.payloads:
            return 0.0
        return sum(self.payloads.values()) / len(self.payloads)

    def total(self) -> float:
        """Actual pooled sum sum_{p in P_tau} x_{p,tau} (one value per
        publisher), i.e. the real data the noisy sum S~_tau is built from."""
        return float(sum(self.payloads.values()))

    def add(self, publisher_id: str, value: float, now: Optional[float] = None):
        now = now if now is not None else time.time()
        if self.start_time is None:
            self.start_time = now
        if self.interval_start is None:
            self.interval_start = now
        # Latest message wins: a newer (or same-time, later-arriving) message
        # from the same publisher replaces the earlier one.
        if publisher_id not in self.arrivals or now >= self.arrivals[publisher_id]:
            self.payloads[publisher_id] = value
            self.arrivals[publisher_id] = now

    def clear(self, now: Optional[float] = None):
        self.payloads.clear()
        self.arrivals.clear()
        self.extensions = 0
        self.start_time = None
        self.interval_start = now
        self.svt = None
        self.leaf_noisy = None
        self.count_units = 0
        self.target_width = None
        self.corrected = False


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class PrivacyPlugin:
    """
    Broker-side privacy middleware.  All constructor arguments map directly
    onto the paper's parameter taxonomy (Section 3.2):

      DP parameters: epsilon, window_size, sensor_bounds (R),
      Scheduling:    timestamp_interval (Delta_t), min_publishers (P),
                     k_ext (K_ext), strategy (A), ba_threshold (theta).
    """

    def __init__(
        self,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        raw_prefix: str = "factory/raw",
        protected_prefix: str = "factory/protected",
        epsilon: float = 1.0,
        window_size: int = 10,
        min_publishers: int = 3,
        strategy: str = "uniform",
        timestamp_interval: float = 5.0,
        k_ext: int = 0,
        ba_threshold: float = 0.1,
        epsilon_count: float = 0.0,
        rho_split: float = 0.2,
        max_publishers: int | None = None,
        enable_hierarchy_walk: bool = True,
        sensor_bounds: dict[str, tuple[float, float]] | None = None,
        publisher_clamps: dict[str, tuple[float, float]] | None = None,
        client_id: str | None = None,
        wait_policy: str = "predictive",
        wait_resolution: int = 4,
        count_cache: bool = True,
        repeat_on_defer: bool = True,
    ):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.raw_prefix = raw_prefix
        self.protected_prefix = protected_prefix
        self.epsilon = epsilon
        self.window_size = window_size
        self.min_publishers = min_publishers  # P_min
        self.strategy = BudgetStrategy(strategy)
        self.timestamp_interval = timestamp_interval
        self.k_ext = int(k_ext)
        self.ba_threshold = ba_threshold
        # eps_count (DEPRECATED): legacy per-level count budget.
        # Superseded by ``rho_split`` -- the count is now the rho share of the
        # per-step budget eps_tau = epsilon/window_size (Definition: Aggregate
        # Stream Element), so every probe spends rho * eps_tau and the noisy
        # count uses Laplace scale 1/(rho * eps_tau).
        self.epsilon_count = float(epsilon_count)
        # Split parameter rho_tau in (0, 1): the count's share of eps_tau.  The
        # sum then gets (1-rho) * eps_tau.  rho <= 0 disables the DP count.
        self.rho_split = float(rho_split)
        self.max_publishers = int(max_publishers) if max_publishers else None
        # Ablation toggle (paper Sec. 6.4): when False the broker never walks
        # up the topic hierarchy; an under-P scope simply defers.  Lets the
        # ablation isolate the walk-up module's contribution to utility.
        self.enable_hierarchy_walk = bool(enable_hierarchy_walk)
        # Adaptive interval (Sec. 4.1).  predictive fixes each width up front
        # from past noisy counts (cap K_ext * Delta_t, one corrective
        # extension); early_stop closes as soon as the stopping rule says P_min
        # is met; reactive extends by whole Delta_t with a fresh count per
        # check; none never waits.  early_stop / reactive cap at
        # (1 + K_ext) * Delta_t.
        if wait_policy not in WAIT_POLICIES:
            raise ValueError(f"wait_policy must be one of {WAIT_POLICIES}")
        self.wait_policy = wait_policy
        self.wait_resolution = max(1, int(wait_resolution))
        # Cached delta counts on the hierarchy walk (one count share per probed level).
        self.count_cache = bool(count_cache)
        # A deferred timestamp repeats the previous output (paper); False
        # delivers nothing instead.
        self.repeat_on_defer = bool(repeat_on_defer)
        # Per-leaf predictive interval sizers (Eqs. 2-3), created lazily.
        self._sizers: dict[str, PredictiveInterval] = {}
        # Timer period: sub-Delta_t checks only when a policy can close an
        # interval between Delta_t boundaries.
        self._tick = (self.timestamp_interval / self.wait_resolution
                      if (self.wait_policy in ("early_stop", "predictive")
                          and self.k_ext > 0)
                      else self.timestamp_interval)
        # Running total of eps_count charged across the whole run, exposed for
        # composed-cost reporting (parallels StreamState.eps_count_spent).
        self.eps_count_spent = 0.0
        self.eps_count_releases = 0
        # Broker-side processing cost (latency experiment): CPU time spent
        # ingesting messages and closing/releasing intervals.
        self.ingest_seconds = 0.0
        self.ingest_messages = 0
        # Per-sensor-type clamp bounds [a, b].  The global payload range R is
        # derived as the max width across the bounds that apply to a topic.
        self.sensor_bounds = sensor_bounds or {}
        # Per-publisher clamp intervals (Definition 4.1).  Keyed by publisher_id.
        self.publisher_clamps: dict[str, tuple[float, float]] = publisher_clamps or {}

        # Per-logical-topic buffering + DP state.  Keys are logical
        # subscription scopes ("line01/machine01/temperature",
        # "line01/machine01", ..., "") expanded lazily by the hierarchy walk.
        self._buffers: dict[str, TopicBuffer] = defaultdict(TopicBuffer)
        # Publishers registered per topic (public deployment structure): the
        # upper clip of the released mean's denominator.
        self._registry: dict[str, set] = defaultdict(set)
        self._streams: dict[str, StreamState] = {}
        self._lock = threading.Lock()

        import uuid as _uuid
        self.client_id = client_id or f"privacy-plugin-{_uuid.uuid4().hex[:8]}"
        self._client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=self.client_id,
        )
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message

        self._running = False
        self._timer_thread: threading.Thread | None = None
        self.release_log: list[dict] = []

    # -------------------------------------------------------------- clamp

    def _clamp_bounds_for(self, topic: str, publisher_id: str) -> tuple[float, float]:
        """Resolve [a_p, b_p] for a given publisher/topic (Definition 4.1)."""
        if publisher_id in self.publisher_clamps:
            return self.publisher_clamps[publisher_id]
        for sensor_type, (lo, hi) in self.sensor_bounds.items():
            if sensor_type in topic:
                return (lo, hi)
        return (-1e9, 1e9)

    def _center_for(self, topic: str) -> Optional[float]:
        """Midpoint of the declared clamp for the topic's sensor type (public),
        or None when no declared bound applies."""
        for st, (lo, hi) in self.sensor_bounds.items():
            if st in topic:
                return (lo + hi) / 2.0
        return None

    def _registered_under(self, scope: str) -> int:
        """Publishers known under ``scope`` -- the public deployment, not the
        publishers active in this interval.  LOCKING: caller holds the lock."""
        prefix = scope + "/" if scope else ""
        pubs: set = set()
        for t, ids in self._registry.items():
            if scope == "" or t == scope or t.startswith(prefix):
                pubs |= ids
        return len(pubs)

    def _payload_range_for(self, topic: str) -> float:
        """Global clamp range R for a subscription scope.

        Takes the sup over all sensor_bounds that match any substring of the
        topic.  The scope-walk ensures only clamp-compatible publishers are
        pooled (Algorithm 1), so R remains the paper's data-independent bound.
        """
        widths = [hi - lo for st, (lo, hi) in self.sensor_bounds.items() if st in topic]
        if widths:
            return float(max(widths))
        # Fallback: sup across all known types.
        if self.sensor_bounds:
            return float(max(hi - lo for lo, hi in self.sensor_bounds.values()))
        return 100.0

    # ------------------------------------------------------ stream state

    def _get_or_create_stream(self, topic: str) -> StreamState:
        if topic not in self._streams:
            config = PrivacyConfig(
                epsilon=self.epsilon,
                window_size=self.window_size,
                min_publishers=self.min_publishers,
                payload_bound=self._payload_range_for(topic),
                strategy=self.strategy,
                ba_threshold=self.ba_threshold,
                # The plugin draws the DP counts itself during the scope walk
                # (Algorithm 1) and hands the engine the count via n_gate, so
                # the engine never re-draws.  rho_split must match the plugin's
                # so the engine reserves the same count share (rho * eps_tau)
                # across the window -- leaving pub_eps = (1-rho)*eps_tau for the
                # noisy sum S~_tau, per the Aggregate Stream Element definition.
                rho_split=self.rho_split,
                epsilon_count=0.0,
                repeat_on_defer=self.repeat_on_defer,
                # Centered sum at the midpoint of the leaf's declared clamp
                # (public), so count error pulls toward the range's middle.
                center=self._center_for(topic),
                # P_max truncation is applied to the pooled buffer before the
                # mean is taken, so the stream sees an already-capped n_tau.
                max_publishers=None,
            )
            self._streams[topic] = StreamState(config=config)
        return self._streams[topic]

    # ------------------------------------------- DP count (eps_count) ----

    def _count_epsilon(self) -> float:
        """Count share rho * eps_tau = rho * (epsilon / window_size).

        Each probed publisher count (Algorithm 1 step 3) spends this much, and
        the noisy count uses Laplace scale 1 / (rho * eps_tau).  rho <= 0
        disables the DP count (exact count, no charge).
        """
        if self.rho_split <= 0 or self.window_size <= 0:
            return 0.0
        return self.rho_split * (self.epsilon / self.window_size)

    def _dp_count(self, n: int) -> int:
        """Release a publisher count n~ = n + Lap(1/(rho*eps_tau)) (sensitivity 1).

        This is the count component of the aggregate stream element
        (Definition: Aggregate Stream Element).  Charges one count share
        rho * eps_tau and returns max(0, round(n + noise)).  When the count
        share is 0 (rho <= 0) the exact count is used (no charge), so the
        Kellaris baselines and a 'free count' configuration both behave sensibly.
        """
        eps_n = self._count_epsilon()
        if eps_n <= 0:
            return int(n)
        self.eps_count_spent += eps_n
        self.eps_count_releases += 1
        noisy = n + float(np.random.laplace(loc=0.0, scale=1.0 / eps_n))
        return max(0, int(round(noisy)))

    @staticmethod
    def _truncate_to_pmax(buf: "TopicBuffer", p_max: int | None) -> "TopicBuffer":
        """Keep at most p_max publishers (implementation knob, not in the paper): when n_tau > P_max the
        broker aggregates only the first P_max publishers and discards the
        rest, bounding the per-element sensitivity change.  Deterministic
        insertion order (dict preserves it) makes the truncation reproducible.
        """
        if p_max is None or buf.num_publishers <= p_max:
            return buf
        kept = TopicBuffer()
        for i, (pid, val) in enumerate(buf.payloads.items()):
            if i >= p_max:
                break
            kept.payloads[pid] = val
            if pid in buf.arrivals:
                kept.arrivals[pid] = buf.arrivals[pid]
        kept.start_time = buf.start_time
        return kept

    # -------------------------------------------------------- MQTT hooks

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        sub = f"{self.raw_prefix}/#"
        client.subscribe(sub)
        logger.info(f"Privacy plugin subscribed to {sub}")

    def _on_message(self, client, userdata, msg):
        t0 = time.perf_counter()
        try:
            self._ingest(msg)
        finally:
            self.ingest_seconds += time.perf_counter() - t0
            self.ingest_messages += 1

    def _ingest(self, msg):
        try:
            payload = json.loads(msg.payload.decode())
            publisher_id = payload.get("publisher_id", "unknown")
            value = float(payload.get("value", 0.0))
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Malformed message on {msg.topic}: {e}")
            return

        logical = msg.topic
        if logical.startswith(self.raw_prefix + "/"):
            logical = logical[len(self.raw_prefix) + 1:]

        lo, hi = self._clamp_bounds_for(logical, publisher_id)
        clamped = _clamp(value, lo, hi)

        now = time.time()
        with self._lock:
            self._buffers[logical].add(publisher_id, clamped, now=now)
            self._registry[logical].add(publisher_id)

    # --------------------------------- Algorithm 1: hierarchy walk ------

    @staticmethod
    def _ancestors(topic: str) -> list[str]:
        """Return [topic, parent(topic), ..., ''] (root)."""
        parts = topic.split("/")
        out = []
        for i in range(len(parts), 0, -1):
            out.append("/".join(parts[:i]))
        out.append("")  # root
        return out

    def _clamp_compatible_buffer(
        self,
        scope: str,
        reference_bounds: tuple[float, float],
        R: float,
    ) -> TopicBuffer:
        """Pool clamp-compatible publishers under `scope` (Definition 4.1).

        A publisher is pooled if the hull of its clamp with the reference has
        width <= R.  The pooled buffer inherits the earliest ``start_time``
        among the contributing source buffers so the walked-up release still
        carries the real wall-clock interval start (paper Def 3.1), not the
        moment the walk happened to run.

        LOCKING: the caller MUST hold ``self._lock`` (the whole per-leaf
        critical section in ``_flush_and_release`` runs under one lock hold so
        the buffer snapshot is consistent and cannot be mutated mid-walk).
        """
        a_star, b_star = reference_bounds
        merged = TopicBuffer()
        earliest_start: Optional[float] = None
        prefix = scope + "/" if scope else ""
        for t, buf in self._buffers.items():
            if scope != "" and not (t == scope or t.startswith(prefix)):
                continue
            contributed = False
            for pub_id, val in buf.payloads.items():
                a_p, b_p = self._clamp_bounds_for(t, pub_id)
                hull = max(b_star, b_p) - min(a_star, a_p)
                if hull <= R + 1e-9:
                    # A publisher seen on several topics under the scope
                    # contributes its latest message only.
                    ts = buf.arrivals.get(pub_id, float("-inf"))
                    if (pub_id not in merged.payloads
                            or ts >= merged.arrivals.get(pub_id, float("-inf"))):
                        merged.payloads[pub_id] = val
                        merged.arrivals[pub_id] = ts
                    contributed = True
            if contributed and buf.start_time is not None:
                if earliest_start is None or buf.start_time < earliest_start:
                    earliest_start = buf.start_time
        merged.start_time = earliest_start
        return merged

    def _reference_bounds(self, topic: str, R: float) -> tuple[float, float]:
        """The subscription's reference clamp for range compatibility: the leaf
        topic's declared bounds (an interval of width R centred on 0 when no
        declared bound matches)."""
        for st, (lo, hi) in self.sensor_bounds.items():
            if st in topic:
                return (lo, hi)
        return (-R / 2, R / 2)

    def _scope_topics(self, scope: str) -> frozenset:
        """Topics the broker knows under ``scope`` -- the public structure the
        walk uses to skip a level whose publisher set equals the one below."""
        prefix = scope + "/" if scope else ""
        return frozenset(t for t in self._buffers
                         if scope == "" or t == scope or t.startswith(prefix))

    def _walk(self, scopes: list[str], ref: tuple[float, float], R: float, *,
              leaf_noisy: float | None = None, max_units: int | None = None):
        """Algorithm 1 over ``scopes`` (subscribed scope first, root last).

        Pools are built lazily, only for the levels the walk probes.  Returns
        (WalkResult, pooled buffer at the chosen level).  Count shares are NOT
        charged here; the caller charges ``result.units``.

        LOCKING: the caller holds ``self._lock``."""
        pools: dict[int, TopicBuffer] = {}
        topics: dict[int, frozenset] = {}

        def pool(k: int) -> TopicBuffer:
            if k not in pools:
                pools[k] = self._clamp_compatible_buffer(scopes[k], ref, R)
            return pools[k]

        def topics_at(k: int) -> frozenset:
            if k not in topics:
                topics[k] = self._scope_topics(scopes[k])
            return topics[k]

        res = walk_population_counts(
            lambda k: pool(k).num_publishers, len(scopes),
            self.min_publishers, self._count_epsilon(),
            cache=self.count_cache, leaf_noisy=leaf_noisy, max_units=max_units,
            same_as_below=lambda k: topics_at(k) == topics_at(k - 1))
        return res, pool(res.level)

    def _scope_walk(self, topic: str, skip_leaf: bool = False,
                    max_probes: int | None = None
                    ) -> tuple[str, TopicBuffer, int]:
        """Walk up from ``topic``; returns (scope, pooled buffer, DP count) and
        charges the walk's count shares.  Kept for callers of the original
        signature; ``skip_leaf`` is ignored (the leaf is the walk's level 0).

        LOCKING: the caller holds ``self._lock``."""
        R = self._payload_range_for(topic)
        scopes = self._ancestors(topic)
        res, pooled = self._walk(scopes, self._reference_bounds(topic, R), R,
                                 max_units=max_probes)
        self._charge_counts(res.units)
        return scopes[res.level], pooled, res.noisy

    def _charge_counts(self, units: int) -> None:
        ec = self._count_epsilon()
        if units > 0 and ec > 0:
            self.eps_count_spent += units * ec
            self.eps_count_releases += units

    # ---------------------------------------- adaptive interval (Sec. 4.1)

    def _keep_waiting(self, buf: TopicBuffer, stream: StreamState) -> bool:
        """One check on an open interval that is past Delta_t but under the cap.
        True = P_min not met yet, keep the interval open.

        early_stop: the first check opens a Sparse Vector stopping rule (one
        count share, and only if the window can also pay the release count);
        every later check is free.  reactive: a fresh leaf count per check, one
        share each; the count from the closing check is reused by the walk.

        LOCKING: the caller holds ``self._lock``."""
        ec = self._count_epsilon()
        n = buf.num_publishers
        if ec <= 0:
            return n < self.min_publishers
        spare = stream.affordable_count_units() - buf.count_units
        if self.wait_policy == "early_stop":
            if buf.svt is None:
                if spare < 2:
                    return False
                buf.svt = MonotoneAboveThreshold(self.min_publishers, ec)
                buf.count_units += 1
                self._charge_counts(1)
            return not buf.svt.test(n)
        # reactive
        if spare < 1:
            buf.leaf_noisy = None
            return False
        buf.leaf_noisy = n + float(np.random.laplace(0.0, 1.0 / ec))
        buf.count_units += 1
        self._charge_counts(1)
        if max(0, int(round(buf.leaf_noisy))) < self.min_publishers:
            buf.extensions += 1
            return True
        return False

    def _sizer(self, leaf: str) -> PredictiveInterval:
        if leaf not in self._sizers:
            self._sizers[leaf] = PredictiveInterval(
                self.timestamp_interval, self.k_ext, self.min_publishers,
                self.max_publishers, self.window_size)
        return self._sizers[leaf]

    def _predictive_wait(self, leaf: str, buf: TopicBuffer, stream: StreamState,
                         width: float, tol: float) -> bool:
        """Predictive interval (Sec. 4.1).  True = keep the interval open.

        The width is fixed when the interval is first examined, from the
        leaf's discounted arrival rate (Eq. 2) -- no count is spent on it.  At
        that width the interval closes; if the window can pay for the release
        count AND a recount, the closing count is drawn here, and an interval
        whose count misses P_min gets one corrective Delta_t extension (the
        count is then redrawn by the walk over the longer interval).  Otherwise
        the count is drawn by the walk and the ordinary walk/defer path
        handles a miss.

        LOCKING: the caller holds ``self._lock``."""
        dt = self.timestamp_interval
        if buf.target_width is None:
            buf.target_width = self._sizer(leaf).width()
        if width + tol < buf.target_width:
            return True
        ec = self._count_epsilon()
        t_max = self._sizer(leaf).t_max
        if (buf.corrected or ec <= 0 or width + dt > t_max + tol
                or stream.affordable_count_units() - buf.count_units < 2):
            return False
        buf.leaf_noisy = buf.num_publishers + float(np.random.laplace(0.0, 1.0 / ec))
        buf.count_units += 1
        self._charge_counts(1)
        if max(0, int(round(buf.leaf_noisy))) < self.min_publishers:
            buf.corrected = True
            buf.target_width = width + dt
            buf.leaf_noisy = None
            return True
        return False

    # ------------------------------------------------- release pipeline

    def _flush_and_release(self):
        now = time.time()
        with self._lock:
            # Snapshot leaf-level topics currently holding buffered messages.
            leaf_topics = [t for t, buf in self._buffers.items() if buf.num_publishers > 0]
            # Preserve empty topics (still counted as logical timestamps).
            leaf_topics += [t for t in self._buffers if t not in leaf_topics]

        needs_walk = is_p_gated(self.strategy)   # P-gated wrappers + n-weighted
        waiting = needs_walk and self.k_ext > 0 and self.wait_policy != "none"
        dt = self.timestamp_interval
        tol = 0.5 * self._tick
        for leaf in leaf_topics:
            # The ENTIRE read -> decide -> pool -> clear critical section runs
            # under ONE lock hold, so a message arriving via _on_message cannot
            # slip into the gap between reading the buffer and clearing it (it
            # would otherwise be silently dropped).  stream.release() is called
            # AFTER the lock since only this timer thread touches stream state.
            t_proc = time.perf_counter()
            defer_no_budget = False
            scope, walk_levels, n_gate_decision = leaf, 0, None
            n_cap = None
            aggregate, n_tau, pooled_sum = 0.0, 0, 0.0
            with self._lock:
                buf = self._buffers[leaf]
                stream = self._get_or_create_stream(leaf)
                if buf.interval_start is None:
                    buf.interval_start = now
                width = now - buf.interval_start

                # Adaptive interval: hold an under-P_min interval open past
                # Delta_t, up to (1 + K_ext) * Delta_t, until P_min is met.
                if waiting and self.wait_policy == "predictive":
                    if self._predictive_wait(leaf, buf, stream, width, tol):
                        continue
                elif waiting:
                    if width + tol < dt:
                        continue                      # base width not reached
                    at_cap = width + tol >= (1 + self.k_ext) * dt
                    if not at_cap and self._keep_waiting(buf, stream):
                        continue
                    if at_cap:
                        buf.leaf_noisy = None         # stale: covers a shorter interval

                units = buf.count_units
                spare = (stream.affordable_count_units() - units) if needs_walk else 0
                if (needs_walk and self._count_epsilon() > 0 and spare < 1
                        and buf.leaf_noisy is None):
                    # The window cannot pay for the release count -> defer.
                    defer_no_budget = True
                    t_start = (buf.start_time if buf.start_time is not None
                               else buf.interval_start)
                else:
                    if needs_walk:
                        # Algorithm 1 with cached delta counts: the leaf's DP
                        # count (or the reactive check's) gates the release;
                        # below P_min the walk climbs to the first ancestor
                        # whose DP count of range-compatible publishers meets it.
                        R = self._payload_range_for(leaf)
                        scopes = (self._ancestors(leaf) if self.enable_hierarchy_walk
                                  else [leaf])
                        res, pooled = self._walk(
                            scopes, self._reference_bounds(leaf, R), R,
                            leaf_noisy=buf.leaf_noisy, max_units=spare)
                        self._charge_counts(res.units)
                        units += res.units
                        scope, walk_levels = scopes[res.level], res.level
                        n_gate_decision = res.noisy
                        if waiting and self.wait_policy == "predictive":
                            # Eq. 2: learn the leaf's arrival rate from the
                            # count this interval released.
                            self._sizer(leaf).observe(res.leaf, width)
                    else:
                        pooled = buf
                    # P_max sensitivity cap (implementation knob, not in the paper): fold at most P_max
                    # publishers into the aggregate; cap the noisy denominator
                    # the same way (post-processing of the DP count).
                    if self.max_publishers is not None:
                        pooled = self._truncate_to_pmax(pooled, self.max_publishers)
                        if n_gate_decision is not None:
                            n_gate_decision = min(n_gate_decision, self.max_publishers)
                    aggregate = pooled.mean()
                    n_tau = pooled.num_publishers
                    pooled_sum = pooled.total()   # actual sum_p x_{p,tau}
                    # Denominator clip: publishers registered under the
                    # released scope (after P_max).
                    n_cap = self._registered_under(scope)
                    if self.max_publishers is not None:
                        n_cap = min(n_cap, self.max_publishers)
                    # t_start = start of the buffering interval (Def. 3.1), not
                    # shifted by the adaptive wait.
                    t_start = (pooled.start_time if pooled.start_time is not None
                               else buf.interval_start)
                # Clear the LEAF buffer (ancestor buffers drain on their own
                # tick) -- still under the lock, no message dropped.  The next
                # interval opens now.
                buf.clear(now)

            if defer_no_budget:
                # Deferred timestamp: charge the shares the wait already spent
                # (repeats the previous output unless repeat_on_defer is off).
                released = stream.release(0.0, 0, n_gate=0, n_count_draws=units)
            else:
                # Hand the engine the SAME DP count the walk gated on, so it
                # does not re-gate on the exact pooled count.  Kellaris
                # baselines (not needs_walk) are not gated at all.
                released = stream.release(
                    aggregate, n_tau,
                    n_gate=(n_gate_decision if needs_walk else None),
                    n_count_draws=(units if needs_walk else 1),
                    true_sum=pooled_sum,   # actual pooled sum, not aggregate*n
                    # Same publisher set -> same count history, whatever the
                    # level name (single-child levels share it).
                    count_key=self._scope_topics(scope), n_cap=n_cap)
            deferred = stream.last_was_deferred

            if released is not None:
                protected_topic = f"{self.protected_prefix}/{leaf}"
                # Paper Def 3.4: the delivered pair is (noisy aggregate,
                # wall-clock start time of the construction interval).  A
                # deferred timestamp re-delivers the previous value.
                out = {
                    "t_start": t_start,
                    "value": round(released, 4),
                }
                self._client.publish(protected_topic, json.dumps(out))
                logger.debug(
                    f"[tau={stream.current_tau}, t_start={t_start:.3f}] {leaf} "
                    f"(scope={scope or 'root'}, n_tau={n_tau}): "
                    f"true={aggregate:.4f} -> noisy={released:.4f}"
                )

            self.release_log.append({
                "leaf_topic": leaf,
                "release_scope": scope or "root",
                "tau": stream.current_tau,
                "t_start": t_start,
                "t_close": now,
                "interval_width": width,
                "wait_dt": max(0.0, width / dt - 1.0) if dt > 0 else 0.0,
                "true_aggregate": aggregate,
                "released_value": released,
                "n_tau": n_tau,
                # The DP count the P-gate decided on (None for the Kellaris
                # strategies, which do not gate); n_tau is the exact pool.
                "n_gate": (n_gate_decision if needs_walk and not defer_no_budget
                           else (0 if defer_no_budget else None)),
                "walk_up": scope != leaf,
                "walk_levels": walk_levels,
                "count_units": units if needs_walk else 0,
                "deferred": deferred,
                "processing_ms": 1000.0 * (time.perf_counter() - t_proc),
            })

    def _timer_loop(self):
        while self._running:
            time.sleep(self._tick)
            if self._running:
                try:
                    self._flush_and_release()
                except Exception as exc:
                    logger.exception(f"release loop error: {exc}")

    # -------------------------------------------------- lifecycle

    def start(self):
        self._client.connect(self.broker_host, self.broker_port)
        self._client.loop_start()
        self._running = True
        self._timer_thread = threading.Thread(target=self._timer_loop, daemon=True)
        self._timer_thread.start()
        logger.info(
            f"Privacy plugin started (epsilon={self.epsilon}, w={self.window_size}, "
            f"P={self.min_publishers}, strategy={self.strategy.value}, "
            f"wait={self.wait_policy}, K_ext={self.k_ext}, "
            f"max interval={(1 + self.k_ext) * self.timestamp_interval}s, "
            f"count_cache={self.count_cache})"
        )

    def stop(self):
        self._running = False
        if self._timer_thread:
            self._timer_thread.join(timeout=self.timestamp_interval * 2)
        self._client.loop_stop()
        self._client.disconnect()
        logger.info("Privacy plugin stopped")

    def get_stream_states(self) -> dict[str, StreamState]:
        return dict(self._streams)
