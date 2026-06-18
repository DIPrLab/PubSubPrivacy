"""
MQTT Privacy Plugin: broker-side middleware for clamped w-event DP with
P-allocation (paper Sections 3-5).

Pipeline:

    Publishers --> [raw topics] --> PrivacyPlugin --> [protected topics] --> Subscribers

The plugin:
  1. Clamps each incoming payload to the publisher's declared interval
     [a_p, b_p] (Definition 3.2, Option A; static operator-declared bounds).
  2. Buffers messages per logical-timestamp interval Delta_t and adaptively
     extends the interval up to T_max = K_ext * Delta_t if the leaf-scope
     publisher count is below P (Section 5.6).
  3. If n_tau < P at the leaf scope even after extension, walks up the topic
     hierarchy (Algorithm 1) to the nearest clamp-compatible ancestor that
     pools >= P publishers; otherwise defers.
  4. Applies the Laplace mechanism with sensitivity Delta_f = R/n_tau for the
     mean aggregation, using the configured budget-allocation strategy.
  5. Delivers only (hat{e}_tau, t_start_tau) on the protected topic.
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

from dp_engine import BudgetStrategy, PrivacyConfig, StreamState, is_p_gated

logger = logging.getLogger(__name__)


@dataclass
class TopicBuffer:
    """Per-timestamp buffer: publisher_id -> clamped value.

    ``start_time`` is the wall-clock moment the current interval began
    accumulating messages, stamped on the first add after a clear.  Paper
    Definition 3.1 says adaptive extensions change the closing boundary
    but never the start, so we do NOT touch ``start_time`` across
    extensions — only ``clear()`` resets it.
    """
    payloads: dict[str, float] = field(default_factory=dict)
    extensions: int = 0
    start_time: Optional[float] = None

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
        if self.start_time is None:
            self.start_time = now if now is not None else time.time()
        self.payloads[publisher_id] = value

    def clear(self):
        self.payloads.clear()
        self.extensions = 0
        self.start_time = None


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class PrivacyPlugin:
    """
    Broker-side privacy middleware.  All constructor arguments map directly
    onto the paper's parameter taxonomy (Section 3.1):

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
        # eps_count (Table 3, DEPRECATED): legacy per-level count budget.
        # Superseded by ``rho_split`` -- the count is now the rho share of the
        # per-step budget eps_tau = epsilon/window_size (Definition: Aggregate
        # Stream Element), so every probe spends rho * eps_tau and the noisy
        # count uses Laplace scale 1/(rho * eps_tau).
        self.epsilon_count = float(epsilon_count)
        # Split parameter rho_tau in (0, 1): the count's share of eps_tau.  The
        # sum then gets (1-rho) * eps_tau.  rho <= 0 disables the DP count.
        self.rho_split = float(rho_split)
        self.max_publishers = int(max_publishers) if max_publishers else None
        # Ablation toggle (paper Sec. 7.8): when False the broker never walks
        # up the topic hierarchy; an under-P scope simply defers.  Lets the
        # ablation isolate the walk-up module's contribution to utility.
        self.enable_hierarchy_walk = bool(enable_hierarchy_walk)
        # Running total of eps_count charged across the whole run, exposed for
        # composed-cost reporting (parallels StreamState.eps_count_spent).
        self.eps_count_spent = 0.0
        self.eps_count_releases = 0
        # Per-sensor-type clamp bounds [a, b].  The global payload range R is
        # derived as the max width across the bounds that apply to a topic.
        self.sensor_bounds = sensor_bounds or {}
        # Per-publisher clamp intervals (Definition 3.2).  Keyed by publisher_id.
        self.publisher_clamps: dict[str, tuple[float, float]] = publisher_clamps or {}

        # Per-logical-topic buffering + DP state.  Keys are logical
        # subscription scopes ("line01/machine01/temperature",
        # "line01/machine01", ..., "") expanded lazily by the hierarchy walk.
        self._buffers: dict[str, TopicBuffer] = defaultdict(TopicBuffer)
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
        """Resolve [a_p, b_p] for a given publisher/topic (Definition 3.2)."""
        if publisher_id in self.publisher_clamps:
            return self.publisher_clamps[publisher_id]
        for sensor_type, (lo, hi) in self.sensor_bounds.items():
            if sensor_type in topic:
                return (lo, hi)
        return (-1e9, 1e9)

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
        """Keep at most p_max publishers (Sec. 6.5): when n_tau > P_max the
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
        kept.start_time = buf.start_time
        return kept

    # -------------------------------------------------------- MQTT hooks

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        sub = f"{self.raw_prefix}/#"
        client.subscribe(sub)
        logger.info(f"Privacy plugin subscribed to {sub}")

    def _on_message(self, client, userdata, msg):
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
        """Pool clamp-compatible publishers under `scope` (Definition 5.2).

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
                    merged.payloads[pub_id] = val
                    contributed = True
            if contributed and buf.start_time is not None:
                if earliest_start is None or buf.start_time < earliest_start:
                    earliest_start = buf.start_time
        merged.start_time = earliest_start
        return merged

    def _scope_walk(self, topic: str, skip_leaf: bool = False,
                    max_probes: int | None = None
                    ) -> tuple[str, TopicBuffer, int]:
        """Walk up the topic tree to the first clamp-compatible ancestor with
        a DP count |P_tau^R(s)| >= P.  Returns (scope, pooled_buffer, n_gate),
        where ``n_gate`` is the differentially private count at the returned
        scope that the P-gate decision was based on (>= P_min on success;
        the last probed count < P_min when the root is reached without meeting
        P, so the caller's gate will defer).

        ``skip_leaf`` starts the walk at parent(topic) when the caller has
        already drawn (and charged) a DP count at the leaf scope, so the leaf
        is not re-probed / re-charged.  ``max_probes`` caps how many ancestor
        levels may be probed (each spends one eps_count); the caller sets it so
        the walk's count spend cannot exceed the remaining w-event budget --
        when it runs out the walk stops at the deepest scope reached so far.
        """
        R = self._payload_range_for(topic)
        # Reference clamp: use the leaf-topic's declared bounds.
        a_star, b_star = 0.0, 0.0
        for st, (lo, hi) in self.sensor_bounds.items():
            if st in topic:
                a_star, b_star = lo, hi
                break
        ref = (a_star, b_star) if (a_star, b_star) != (0.0, 0.0) else (-R / 2, R / 2)

        scopes = self._ancestors(topic)
        if skip_leaf and len(scopes) > 1:
            scopes = scopes[1:]
        if max_probes is not None:
            scopes = scopes[:max(0, int(max_probes))]
        last_scope, last_buf, last_n_gate = topic, TopicBuffer(), 0
        for scope in scopes:
            buf = self._clamp_compatible_buffer(scope, ref, R)
            # Algorithm 1 step 3: compare a DIFFERENTIALLY PRIVATE count of the
            # range-compatible publishers at this scope against P, spending
            # eps_count (sensitivity 1) at each level we probe.
            n_gate = self._dp_count(buf.num_publishers)
            last_scope, last_buf, last_n_gate = scope, buf, n_gate
            if n_gate >= self.min_publishers:
                return scope, buf, n_gate
        return last_scope, last_buf, last_n_gate

    # ------------------------------------------------- release pipeline

    def _flush_and_release(self):
        with self._lock:
            # Snapshot leaf-level topics currently holding buffered messages.
            leaf_topics = [t for t, buf in self._buffers.items() if buf.num_publishers > 0]
            # Preserve empty topics (still counted as logical timestamps).
            leaf_topics += [t for t in self._buffers if t not in leaf_topics]

        needs_walk = is_p_gated(self.strategy) or self.strategy == BudgetStrategy.N_WEIGHTED
        for leaf in leaf_topics:
            # The ENTIRE read -> decide -> pool -> clear critical section runs
            # under ONE lock hold, so a message arriving via _on_message cannot
            # slip into the gap between reading the buffer and clearing it (it
            # would otherwise be silently dropped).  stream.release() is called
            # AFTER the lock since only this timer thread touches stream state.
            extend = False
            defer_no_budget = False
            n_count_draws = 0
            with self._lock:
                buf_leaf = self._buffers[leaf]
                leaf_n = buf_leaf.num_publishers
                stream = self._get_or_create_stream(leaf)

                # Budget-aware DP counting: each count draw (leaf gate + every
                # walk probe) spends rho * eps_tau from the SAME w-event budget
                # eps, so cap the number of draws this tick to what the window
                # can still afford (floor(remaining / (rho*eps_tau))).  This
                # guarantees the leaf + walk count spend never breaches
                # Sum_{window} <= eps.
                ec = self._count_epsilon()
                affordable = ((1 << 30) if (not needs_walk or ec <= 0)
                              else int(stream._budget_remaining() / ec))
                ec_before = self.eps_count_spent

                if needs_walk and ec > 0 and affordable < 1:
                    # Window cannot pay for even the leaf count -> defer.
                    self._buffers[leaf].clear()
                    defer_no_budget = True
                else:
                    # Paper Sec. 6.3 step 1 / Sec. 6.7: any publisher count
                    # released to drive a decision (the interval-extension check
                    # AND the release-eligibility gate) is DIFFERENTIALLY PRIVATE
                    # -- one eps_count (sensitivity 1) per probe.  The same noisy
                    # leaf count drives both the extend and the gate decision.
                    leaf_n_gate = self._dp_count(leaf_n) if needs_walk else leaf_n

                    # Adaptive interval extension (Sec. 6.7): hold the buffer open
                    # one more Delta_t when the DP leaf count is below P.
                    if (needs_walk and leaf_n_gate < self.min_publishers
                            and self.k_ext > 0 and buf_leaf.extensions < self.k_ext):
                        buf_leaf.extensions += 1
                        extend = True
                    else:
                        scope, pooled = (leaf, buf_leaf)
                        # The DP count the P-gate decision is based on: the leaf
                        # count, or the walk's final scope count if we walk up.
                        # This single noisy count is handed to release(n_gate=...)
                        # so the engine does NOT re-gate on the exact pooled count.
                        n_gate_decision = leaf_n_gate
                        if (needs_walk and leaf_n_gate < self.min_publishers
                                and self.enable_hierarchy_walk):
                            # Algorithm 1 walk (charges eps_count per ancestor),
                            # capped to the remaining count budget (leaf already
                            # spent one draw), under the same lock so the pool
                            # snapshot is consistent.
                            scope, pooled, n_gate_decision = self._scope_walk(
                                leaf, skip_leaf=True,
                                max_probes=(None if ec <= 0 else affordable - 1))
                        # P_max sensitivity cap (Sec. 6.5): fold at most P_max
                        # publishers into the aggregate.
                        pooled = self._truncate_to_pmax(pooled, self.max_publishers)
                        aggregate = pooled.mean()
                        n_tau = pooled.num_publishers
                        pooled_sum = pooled.total()   # actual sum_p x_{p,tau}
                        # t_start = start of the buffering interval (Def. 3.1),
                        # NOT shifted by K_ext extensions.
                        t_start = (pooled.start_time if pooled.start_time is not None
                                   else time.time())
                        # Clear the LEAF buffer (ancestor buffers drain on their
                        # own tick) -- still under the lock, no message dropped.
                        self._buffers[leaf].clear()
                # Count of eps_count draws made this tick (leaf + walk probes),
                # passed to release() so the full count cost is charged in-window.
                n_count_draws = (round((self.eps_count_spent - ec_before) / ec)
                                 if ec > 0 else 0)
            if defer_no_budget:
                # Record a deferred timestamp (repeat last) without spending eps.
                self._get_or_create_stream(leaf).release(
                    0.0, 0, n_gate=0, n_count_draws=0)
                continue
            if extend:
                continue  # wait for another Delta_t tick

            # Use the leaf's DP stream state (each subscription binding is a
            # distinct stream in the paper; here we key by subscribed topic).
            # Hand the engine the SAME differentially private count the plugin
            # gated on (Algorithm 1), so it does not re-gate on the exact pooled
            # count.  Kellaris baselines (not needs_walk) pass n_gate=None and
            # are not gated at all.
            stream = self._get_or_create_stream(leaf)
            released = stream.release(
                aggregate, n_tau,
                n_gate=(n_gate_decision if needs_walk else None),
                n_count_draws=(n_count_draws if needs_walk else 1),
                true_sum=pooled_sum)   # actual pooled sum, not aggregate*n
            deferred = stream.last_was_deferred

            if released is not None:
                protected_topic = f"{self.protected_prefix}/{leaf}"
                # Paper Def 3.4: the delivered pair is (noisy aggregate,
                # wall-clock start time of the construction interval).
                out = {
                    "t_start": t_start,
                    "value": round(released, 4),
                }
                self._client.publish(protected_topic, json.dumps(out))
                logger.debug(
                    f"[tau={stream.current_tau}, t_start={t_start:.3f}] {leaf} "
                    f"(scope={scope or 'root'}, n_tau={n_tau}): "
                    f"true={aggregate:.4f} -> noisy={released:.4f}"
                    f"{' [DEFERRED]' if deferred else ''}"
                )

            self.release_log.append({
                "leaf_topic": leaf,
                "release_scope": scope or "root",
                "tau": stream.current_tau,
                "t_start": t_start,
                "true_aggregate": aggregate,
                "released_value": released,
                "n_tau": n_tau,
                "walk_up": scope != leaf,
                "deferred": deferred,
            })

    def _timer_loop(self):
        while self._running:
            time.sleep(self.timestamp_interval)
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
            f"K_ext={self.k_ext}, T_max={self.k_ext * self.timestamp_interval}s)"
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
