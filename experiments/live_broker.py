#!/usr/bin/env python3
"""Experiment E: live MQTT broker end-to-end (paper Sec. 7, live throughput /
latency).  Spins up an embedded broker, runs real paho publishers/subscribers
through the PrivacyPlugin, and measures live behaviour.  Kept out of the offline
engine because it needs a running broker; invoked only by the CLI.

Imports the offline engine for the shared DP runner + dataset prep.
"""
from __future__ import annotations

import json
import os
import threading
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.engine import *          # noqa: F401,F403  (shared engine surface)
from experiments.engine import (          # explicit: underscore names import * skips
    run_dp_on_stream, prepare_dataset, _dataset_max_rows, logger,
)

def _live_raw_topic(prefix: str, leaf: str) -> str:
    return f"{prefix}/{leaf}"


class _EmbeddedBroker:
    """Embedded amqtt (pure-Python) MQTT broker, started on a background thread.

    Used by ``--experiment E`` when no broker is already listening on the
    requested (host, port).  Runs an asyncio loop in a daemon thread hosting
    the ``amqtt.broker.Broker`` instance; ``stop()`` cleanly shuts both down.
    """

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self._loop: asyncio.AbstractEventLoop | None = None
        self._broker = None
        self._thread: threading.Thread | None = None
        self._started = threading.Event()
        self._start_error: BaseException | None = None

    def start(self, timeout: float = 10.0) -> None:
        import asyncio as _asyncio
        try:
            from amqtt.broker import Broker  # noqa: F401 (import-check only)
        except ImportError as exc:
            raise RuntimeError(
                "amqtt is not installed; either install it "
                "(`pip install amqtt`) or start an external MQTT broker "
                "(e.g., mosquitto) before running --experiment E."
            ) from exc

        def _runner():
            try:
                loop = _asyncio.new_event_loop()
                _asyncio.set_event_loop(loop)
                self._loop = loop
                from amqtt.broker import Broker
                config = {
                    "listeners": {
                        "default": {
                            "type": "tcp",
                            "bind": f"{self.host}:{self.port}",
                        }
                    },
                    "auth": {"allow-anonymous": True},
                    # Disable the $SYS plugin's periodic reporting; it tries
                    # to compare sys_interval (None) to 0 and spams warnings.
                    "sys_interval": 0,
                }

                async def _bring_up():
                    # Broker.__init__ and start() both need a running loop.
                    self._broker = Broker(config=config)
                    await self._broker.start()

                loop.run_until_complete(_bring_up())
                self._started.set()
                loop.run_forever()
            except BaseException as exc:  # pragma: no cover
                self._start_error = exc
                self._started.set()

        self._thread = threading.Thread(
            target=_runner, name="pubsubpriv-embedded-broker", daemon=True,
        )
        self._thread.start()
        if not self._started.wait(timeout=timeout):
            raise RuntimeError(
                f"Embedded broker did not start within {timeout}s"
            )
        if self._start_error is not None:
            raise self._start_error
        # Give paho clients a beat to be able to connect reliably.
        time.sleep(0.3)
        logger.info(f"[exp E] embedded amqtt broker listening at {self.host}:{self.port}")

    def stop(self, timeout: float = 5.0) -> None:
        if self._loop is None:
            return
        loop = self._loop

        async def _shutdown():
            try:
                if self._broker is not None:
                    await self._broker.shutdown()
            except Exception:
                pass

        try:
            fut = asyncio.run_coroutine_threadsafe(_shutdown(), loop)
            fut.result(timeout=timeout)
        except Exception:
            pass
        loop.call_soon_threadsafe(loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._loop = None
        logger.info("[exp E] embedded broker stopped")


def _broker_is_listening(host: str, port: int, timeout: float = 0.75) -> bool:
    import socket
    s = socket.socket()
    s.settimeout(timeout)
    try:
        s.connect((host, port))
        return True
    except Exception:
        return False
    finally:
        try:
            s.close()
        except Exception:
            pass


def _ensure_broker(host: str, port: int, auto_start: bool) -> "_EmbeddedBroker | None":
    """If a broker is already listening, return None.  Otherwise, start an
    embedded amqtt broker on (host, port) and return a handle whose ``stop()``
    tears it down when the experiment finishes.  Raises if ``auto_start`` is
    false and no external broker is found.
    """
    if _broker_is_listening(host, port):
        logger.info(f"[exp E] using existing broker at {host}:{port}")
        return None
    if not auto_start:
        raise RuntimeError(
            f"No MQTT broker listening at {host}:{port} and --no-auto-broker "
            f"was specified.  Start mosquitto (or similar) first."
        )
    logger.info(
        f"[exp E] no broker at {host}:{port}; starting embedded amqtt broker"
    )
    eb = _EmbeddedBroker(host, port)
    eb.start()
    return eb


class _LiveSubscriber:
    """paho subscriber that captures delivered releases on the protected prefix."""

    def __init__(self, broker_host, broker_port, protected_prefix, client_id):
        import paho.mqtt.client as mqtt  # local import: paho may not be installed
        self._mqtt = mqtt
        self.prefix = protected_prefix
        self.received: list[dict] = []
        self._ready = threading.Event()
        self._client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=client_id,
        )

        def _on_connect(client, *_a, **_kw):
            client.subscribe(f"{self.prefix}/#")
            self._ready.set()

        def _on_message(_c, _u, msg):
            try:
                body = json.loads(msg.payload.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return
            self.received.append({
                "topic": msg.topic,
                "t_start": float(body.get("t_start", 0.0)),
                "value": float(body.get("value", 0.0)),
                "wall_clock_recv": time.time(),
            })

        self._client.on_connect = _on_connect
        self._client.on_message = _on_message
        self._broker_host = broker_host
        self._broker_port = broker_port

    def start(self, connect_timeout: float = 5.0):
        self._client.connect(self._broker_host, self._broker_port)
        self._client.loop_start()
        if not self._ready.wait(timeout=connect_timeout):
            raise RuntimeError("Subscriber did not connect within timeout")

    def stop(self):
        self._client.loop_stop()
        try:
            self._client.disconnect()
        except Exception:
            pass


class _LivePublisher:
    """paho publisher that emits each tau's per-publisher readings at a wall-clock cadence."""

    def __init__(self, broker_host, broker_port, raw_prefix, client_id):
        import paho.mqtt.client as mqtt
        self.prefix = raw_prefix
        self._client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=client_id,
        )
        self._ready = threading.Event()
        self._client.on_connect = lambda *_a, **_kw: self._ready.set()
        self._broker_host = broker_host
        self._broker_port = broker_port
        self.num_published = 0

    def start(self, connect_timeout: float = 5.0):
        self._client.connect(self._broker_host, self._broker_port)
        self._client.loop_start()
        if not self._ready.wait(timeout=connect_timeout):
            raise RuntimeError("Publisher did not connect within timeout")

    def publish_one(self, leaf: str, publisher_id: str, value: float):
        topic = _live_raw_topic(self.prefix, leaf)
        payload = json.dumps({"publisher_id": str(publisher_id),
                              "value": float(value)}).encode("utf-8")
        # QoS 1 so the plugin reliably sees every publish even under load.
        info = self._client.publish(topic, payload, qos=1)
        info.wait_for_publish(timeout=2.0)
        self.num_published += 1

    def stop(self):
        self._client.loop_stop()
        try:
            self._client.disconnect()
        except Exception:
            pass


def _drive_live_config(
    per_pub: dict[str, list[float | None]],
    sensor: str,
    dataset_spec: dict,
    *,
    strategy: str,
    epsilon: float,
    w: int,
    P: int,
    broker_host: str,
    broker_port: int,
    live_dt: float,
    n_steps: int,
    seed: int,
    drain_ticks: int = 3,
    scenario: str = "pooled",
    n_subscribers: int = 1,
) -> dict:
    """Run one (strategy, epsilon, w, P, scenario) config end-to-end
    through a live broker.

    Scenarios:

      * ``pooled``    — every publisher emits on the SAME leaf topic so the
                        plugin aggregates all n_tau active publishers into a
                        single release per tau.  Matches the offline
                        ``run_dp_on_stream`` reference exactly (single stream
                        over the mean of all active publishers).  Exercises
                        the many-publishers path under a shared-leaf
                        subscription.
      * ``hierarchy`` — each publisher emits on its own leaf under the
                        dataset's normative MQTT tree (so each leaf's
                        n_tau = 1).  P >= 2 therefore forces Algorithm 1's
                        clamp-compatible walk-up on every release, and the
                        plugin's ``walk_up`` flag should be True for every
                        released record.

    Every component (plugin, publishers, subscribers) gets unique paho
    client_ids and a unique topic-prefix UUID so parallel configs can share
    the broker safely.  ``n_subscribers`` concurrent ``_LiveSubscriber``
    clients attach to the protected prefix to verify the broker correctly
    fans each release out to every subscribing client — each subscriber's
    received-count is reported alongside the plugin's published-count, so
    the caller can verify subscriber-fan-out integrity.

    Returns the plugin's release log, every subscriber's received releases
    (aggregated), per-subscriber counts, tau-truth log, and timing metadata.
    """
    import uuid as _uuid

    run_id = _uuid.uuid4().hex[:8]
    raw_prefix = f"pubsubpriv/{run_id}/raw"
    protected_prefix = f"pubsubpriv/{run_id}/protected"

    lo, hi = dataset_spec["static_clamps"][sensor]
    if scenario == "pooled":
        pooled_leaf = f"{sensor}"

        def _leaf_of(_pub_id: str) -> str:
            return pooled_leaf
        plugin_P = int(P)
    elif scenario == "hierarchy":
        # Each publisher gets its own leaf under the dataset's normative
        # topic tree.  With P > 1 the plugin's release gate will fire
        # on every leaf (n_tau=1 < P) and Algorithm 1's clamp-compatible
        # walk-up must climb to an ancestor before any release can emit.
        topic_of = dataset_spec["publisher_topic"]

        def _leaf_of(pub_id: str) -> str:
            return topic_of(pub_id, sensor)
        plugin_P = max(2, int(P))  # force walk-up (every leaf has n=1)
    else:
        raise ValueError(f"unknown scenario: {scenario}")

    plugin = PrivacyPlugin(
        broker_host=broker_host,
        broker_port=broker_port,
        raw_prefix=raw_prefix,
        protected_prefix=protected_prefix,
        epsilon=epsilon,
        window_size=w,
        min_publishers=plugin_P,
        strategy=strategy,
        timestamp_interval=live_dt,
        k_ext=0,
        sensor_bounds={sensor: (lo, hi)},
        client_id=f"plugin-{run_id}",
    )
    n_subscribers = max(1, int(n_subscribers))
    subscribers = [
        _LiveSubscriber(broker_host, broker_port, protected_prefix,
                        client_id=f"sub-{run_id}-{i}")
        for i in range(n_subscribers)
    ]
    publisher = _LivePublisher(broker_host, broker_port, raw_prefix,
                               client_id=f"pub-{run_id}")

    np.random.seed(seed)
    publishers = list(per_pub.keys())
    T = min(n_steps, len(per_pub[publishers[0]]))
    tau_truth: list[dict] = []
    t0_wall = None

    try:
        for sub in subscribers:
            sub.start()
        plugin.start()
        publisher.start()

        # Give the broker a moment to finish subscription ack-and-route before
        # the first publish.  This avoids a race where tau=0 messages arrive
        # before any subscription is live.
        time.sleep(max(0.2, live_dt))

        t0_wall = time.time()
        for tau in range(T):
            tau_start_wall = time.time()
            active_values = []
            for pub_id in publishers:
                v = per_pub[pub_id][tau]
                if v is None:
                    continue
                publisher.publish_one(_leaf_of(pub_id), pub_id, float(v))
                active_values.append(max(lo, min(hi, float(v))))
            tau_truth.append({
                "tau": tau,
                "true_clamped_mean": float(np.mean(active_values)) if active_values else 0.0,
                "n_active": len(active_values),
                "wall_clock_publish": tau_start_wall,
            })
            # Pace to one publish burst per live_dt so the plugin timer groups
            # this tau's messages into one flush.
            elapsed = time.time() - tau_start_wall
            if elapsed < live_dt:
                time.sleep(live_dt - elapsed)

        # Drain the tail: let the plugin timer fire a few more times so the
        # final window is released.
        time.sleep(drain_ticks * live_dt + 0.2)
    finally:
        try:
            plugin.stop()
        finally:
            publisher.stop()
            for sub in subscribers:
                try:
                    sub.stop()
                except Exception:
                    pass

    # Aggregate across every subscriber so downstream metrics still see one
    # canonical delivery list; also record per-subscriber counts so the
    # caller can verify broker fan-out was correct (every subscriber should
    # receive every release).
    subscriber_received: list[dict] = []
    per_subscriber_counts: list[int] = []
    for sub in subscribers:
        recv = list(sub.received)
        subscriber_received.extend(recv)
        per_subscriber_counts.append(len(recv))

    return {
        "plugin_log": list(plugin.release_log),
        "subscriber_received": subscriber_received,
        "per_subscriber_counts": per_subscriber_counts,
        "num_subscribers": n_subscribers,
        "scenario": scenario,
        "plugin_P": plugin_P,
        "tau_truth": tau_truth,
        "run_id": run_id,
        "raw_prefix": raw_prefix,
        "protected_prefix": protected_prefix,
        "num_published": publisher.num_published,
        "t0_wall": t0_wall,
    }


def experiment_E_live_broker(
    dataset_name: str,
    output_dir: str,
    args,
    *,
    broker_host: str = "localhost",
    broker_port: int = 1883,
    live_dt: float = 0.1,
    n_steps: int = 120,
    seed: int = 123,
    auto_start_broker: bool = True,
) -> pd.DataFrame:
    """Exp E: small sanity grid through a live MQTT broker.

    Grid (9 configs): 3 strategies × 3 epsilons × 1 (dataset, sensor, w, P).
    Intended for the smallest datasets ('wearable' default); do NOT run on
    'energy' (~36k windows per sensor -> hours of broker traffic).

    If no broker is listening at (broker_host, broker_port) and
    ``auto_start_broker`` is True, spins up an embedded amqtt broker on the
    same (host, port) for the duration of the experiment.
    """
    if dataset_name == "energy":
        logger.warning(
            "[exp E] 'energy' has ~36k windows/sensor; that is ~1 hour per "
            "config at live_dt=0.1s.  Strongly recommend 'wearable' (119 "
            "windows) or 'manufacturing' (1000)."
        )

    embedded_broker = _ensure_broker(broker_host, broker_port, auto_start_broker)
    try:
        return _experiment_E_run(
            dataset_name, output_dir, args,
            broker_host=broker_host, broker_port=broker_port,
            live_dt=live_dt, n_steps=n_steps, seed=seed,
        )
    finally:
        if embedded_broker is not None:
            embedded_broker.stop()


def _experiment_E_run(
    dataset_name: str,
    output_dir: str,
    args,
    *,
    broker_host: str,
    broker_port: int,
    live_dt: float,
    n_steps: int,
    seed: int,
) -> pd.DataFrame:
    prepared = prepare_dataset(
        dataset_name,
        clamp_mode="static",
        eps_clip=args.eps_clip,
        seed=args.seed,
        max_rows=_dataset_max_rows(dataset_name, args),
    )
    if prepared is None or not prepared.per_pubs:
        logger.warning(f"[exp E] nothing prepared for {dataset_name}; skip")
        return pd.DataFrame()

    sensor = next(
        (s for s in prepared.spec["sensors"]
         if s in prepared.per_pubs
         and s in prepared.spec["static_clamps"]
         and len(prepared.per_pubs[s][0]) >= 2),
        None,
    )
    if sensor is None:
        logger.warning(f"[exp E] no suitable sensor in {dataset_name}; skip")
        return pd.DataFrame()

    per_pub = prepared.per_pubs[sensor][0]
    lo, hi = prepared.spec["static_clamps"][sensor]
    B = float(hi - lo)

    strategies = ["uniform", "p_gated_ba", "p_gated_bd", "n_weighted"]
    epsilons = [0.5, 1.0, 2.0]
    w = 8
    P = 2

    # Scenarios exercised end-to-end:
    #   pooled    — many publishers on one shared leaf (no walk-up; matches
    #               offline reference exactly).
    #   hierarchy — each publisher on its own leaf under the normative
    #               topic tree, P>=2 forces Algorithm 1 walk-up on every
    #               release.
    # The pooled scenario is always valid; the hierarchy scenario is skipped
    # only when the dataset lacks a ``publisher_topic`` factory.
    raw_scenarios = getattr(args, "live_scenarios", None)
    if isinstance(raw_scenarios, str):
        scenarios: list[str] = [
            s.strip() for s in raw_scenarios.split(",") if s.strip()
        ]
    else:
        scenarios = list(raw_scenarios) if raw_scenarios else []
    if not scenarios:
        scenarios = ["pooled", "hierarchy"]
    if "hierarchy" in scenarios and "publisher_topic" not in prepared.spec:
        logger.info(
            f"[exp E] {dataset_name}: dataset lacks 'publisher_topic' spec; "
            "skipping hierarchy scenario"
        )
        scenarios = [s for s in scenarios if s != "hierarchy"]

    n_subscribers: int = max(1, int(getattr(args, "live_n_subscribers", 3)))

    # Dataset-specific output path so running E across multiple datasets
    # (e.g. --experiment E --dataset all) does not clobber earlier results.
    exp_dir = os.path.join(output_dir, "experiments", "E_live_broker",
                           dataset_name)
    os.makedirs(exp_dir, exist_ok=True)

    rows = []
    all_messages: list[dict] = []
    log_messages = getattr(args, "log_messages", True)
    t_exp_start = time.time()
    for strategy in strategies:
        for epsilon in epsilons:
            for scenario in scenarios:
                t_cfg = time.time()
                logger.info(
                    f"[exp E] {dataset_name}/{sensor} strategy={strategy} "
                    f"eps={epsilon} w={w} P={P} scenario={scenario} "
                    f"subscribers={n_subscribers} "
                    f"(live broker {broker_host}:{broker_port})"
                )
                try:
                    out = _drive_live_config(
                        per_pub, sensor, prepared.spec,
                        strategy=strategy, epsilon=epsilon, w=w, P=P,
                        broker_host=broker_host, broker_port=broker_port,
                        live_dt=live_dt, n_steps=n_steps, seed=seed,
                        scenario=scenario,
                        n_subscribers=n_subscribers,
                    )
                except Exception as exc:
                    logger.exception(
                        f"[exp E] config failed (strategy={strategy}, "
                        f"eps={epsilon}, scenario={scenario}): {exc}"
                    )
                    rows.append({
                        "dataset": dataset_name, "sensor": sensor,
                        "strategy": strategy, "epsilon": epsilon,
                        "w": w, "P": P, "scenario": scenario,
                        "num_subscribers": n_subscribers,
                        "status": "error", "error": str(exc),
                    })
                    continue

                log = out["plugin_log"]
                released = [r for r in log if not r["deferred"]
                            and r["released_value"] is not None]
                plugin_published = [r for r in log
                                    if r["released_value"] is not None]

                # Canonical metrics from the plugin log.
                live_nmae, live_mae, live_kl, live_attr = _live_metrics(
                    released, out["tau_truth"], B,
                )

                # Offline reference with the same seed; used for comparison
                # metrics and to populate epsilon_tau / lambda_tau in the
                # per-release message log (plugin doesn't expose budget
                # metadata directly).
                agg = [e["true_clamped_mean"] for e in out["tau_truth"]]
                cnt = [e["n_active"] for e in out["tau_truth"]]
                offline = run_dp_on_stream(
                    agg, cnt, epsilon=epsilon, window_size=w,
                    min_publishers=P, payload_bound=B,
                    strategy=strategy, seed=seed,
                )
                off_m = offline["metrics"]

                walkup_count = sum(1 for r in log if r.get("walk_up"))
                walkup_rate = walkup_count / max(1, len(log))
                # P-gate violation: a released (non-deferred) record with
                # n_tau < plugin_P indicates the gate is broken.
                p_gate_violations = sum(
                    1 for r in released
                    if int(r["n_tau"]) < int(out.get("plugin_P", P))
                )

                # Per-release message log.  Canonical rows from the live
                # plugin path, augmented with broker-delivery and walk-up
                # audit columns.
                if log_messages:
                    subscriber_by_t = {
                        float(m.get("t_start", -1)): m
                        for m in out["subscriber_received"]
                    }
                    off_budgets = offline.get("budgets_spent") or []
                    for rec in log:
                        tau_idx = int(rec["tau"])  # 1-indexed
                        n_tau = int(rec["n_tau"])
                        true_v = rec["true_aggregate"]
                        noisy_v = rec["released_value"]
                        eps_tau = (float(off_budgets[tau_idx - 1])
                                   if 0 <= tau_idx - 1 < len(off_budgets)
                                   else 0.0)
                        deferred = bool(rec["deferred"]) or eps_tau <= 0
                        if (true_v is not None and noisy_v is not None
                                and not deferred):
                            noise = float(noisy_v) - float(true_v)
                        else:
                            noise = 0.0
                        lam = (float(B) / (n_tau * eps_tau)
                               if eps_tau > 0 and n_tau > 0
                               else float("inf"))
                        delta_f = (float(B) / n_tau
                                   if n_tau > 0 else float("inf"))
                        t_start = float(rec.get("t_start", 0.0))
                        all_messages.append({
                            "dataset": dataset_name,
                            "clamp_mode": "static",  # E runs under Option A
                            "sensor": sensor,
                            "strategy": strategy,
                            "P": int(P),
                            "epsilon": float(epsilon),
                            "w": int(w),
                            "payload_bound": float(B),
                            "seed": int(seed),
                            "experiment": f"E_live_broker/{scenario}",
                            "config_id": (
                                f"{dataset_name}|E|{scenario}|{sensor}"
                                f"|{strategy}|P={P}|eps={epsilon}|w={w}"
                                f"|run={out['run_id']}"
                            ),
                            "tau": tau_idx,
                            "t_start_logical": t_start,
                            "true_aggregate": (float(true_v)
                                               if true_v is not None else None),
                            "noisy_value": (float(noisy_v)
                                            if noisy_v is not None else None),
                            "n_tau": n_tau,
                            "epsilon_tau": eps_tau,
                            "lambda_tau": lam,
                            "noise_sample": noise,
                            "deferred": deferred,
                            "delta_f": delta_f,
                            # Live-broker-specific audit columns:
                            "scenario": scenario,
                            "plugin_P": int(out.get("plugin_P", P)),
                            "num_subscribers": n_subscribers,
                            "leaf_topic": rec.get("leaf_topic"),
                            "release_scope": rec.get("release_scope"),
                            "walk_up": bool(rec.get("walk_up", False)),
                            "broker_delivered": t_start in subscriber_by_t,
                            "run_id": out["run_id"],
                        })

                # Broker-path integrity: every protected-topic publish the
                # plugin made should have been delivered to every subscriber.
                per_sub = out.get("per_subscriber_counts", [])
                broker_deliveries = len(out["subscriber_received"])
                expected_deliveries = len(plugin_published) * n_subscribers
                broker_delivery_ok = (
                    broker_deliveries == expected_deliveries
                    and all(c == len(plugin_published) for c in per_sub)
                )

                rows.append({
                    "dataset": dataset_name,
                    "sensor": sensor,
                    "strategy": strategy,
                    "epsilon": epsilon,
                    "w": w,
                    "P": P,
                    "scenario": scenario,
                    "plugin_P": int(out.get("plugin_P", P)),
                    "num_subscribers": n_subscribers,
                    "per_subscriber_counts": ";".join(str(c) for c in per_sub),
                    "run_id": out["run_id"],
                    "num_taus": len(log),
                    "num_released": len(released),
                    "num_plugin_published": len(plugin_published),
                    "num_walkups": walkup_count,
                    "walkup_rate": walkup_rate,
                    "p_gate_violations": p_gate_violations,
                    "release_rate_live": len(released) / max(1, len(log)),
                    "release_rate_offline": off_m.get("release_rate",
                                                      float("nan")),
                    "nmae_live": live_nmae,
                    "nmae_offline": off_m.get("normalized_mae", float("nan")),
                    "nmae_abs_delta": abs(
                        live_nmae - off_m.get("normalized_mae", float("nan"))),
                    "mae_live": live_mae,
                    "kl_live": live_kl,
                    "kl_offline": off_m.get("kl_divergence", float("nan")),
                    "attribution_advantage_live": live_attr,
                    "attribution_advantage_offline":
                        off_m.get("attribution_advantage", float("nan")),
                    "broker_deliveries": broker_deliveries,
                    "expected_deliveries": expected_deliveries,
                    "broker_delivery_ok": broker_delivery_ok,
                    "num_input_published": out["num_published"],
                    "wall_clock_seconds": round(time.time() - t_cfg, 2),
                    "status": "ok",
                })
                logger.info(
                    f"[exp E]   scenario={scenario} "
                    f"released={len(released)}/{len(log)} "
                    f"walkups={walkup_count} "
                    f"p_violations={p_gate_violations} "
                    f"broker_delivered={broker_deliveries}/{expected_deliveries} "
                    f"nmae_live={live_nmae:.4f} "
                    f"nmae_offline={off_m.get('normalized_mae', 0):.4f} "
                    f"(elapsed {time.time() - t_cfg:.1f}s)"
                )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(exp_dir, "experiment_E_live_broker.csv"), index=False)
    logger.info(
        f"  Experiment E wrote {len(df)} rows -> {exp_dir} "
        f"(total wall-clock {time.time() - t_exp_start:.1f}s)"
    )

    if log_messages and all_messages:
        from message_logger import write_messages_csv
        n = write_messages_csv(
            all_messages,
            os.path.join(exp_dir, "experiment_E_messages.csv"),
        )
        logger.info(f"  Experiment E wrote {n} per-release messages "
                    f"(live plugin path; broker_delivered flag included)")

    # Plot generation is delegated to generate_plots.py by default; inline
    # rendering only runs when the user passes --generate-plots (monkey-patch
    # of plt.savefig in main() no-ops this call otherwise).
    if getattr(args, "generate_plots", False):
        _plot_experiment_E(df,
                           os.path.join(exp_dir, "experiment_E_live_broker.png"),
                           dataset_name, sensor, live_dt,
                           broker_host, broker_port)
    return df


def _live_metrics(released_records, tau_truth, payload_bound):
    """Compute NMAE / MAE / KL / attribution advantage from the plugin's release log."""
    if not released_records:
        return float("nan"), float("nan"), float("nan"), float("nan")
    # Match by tau index (plugin's current_tau counts from 1; tau_truth from 0).
    truth_by_tau = {e["tau"]: e for e in tau_truth}
    abs_err = []
    attr = []
    fresh_true, fresh_noisy = [], []
    for r in released_records:
        t = truth_by_tau.get(r["tau"] - 1)
        if t is None:
            continue
        err = abs(r["released_value"] - t["true_clamped_mean"])
        abs_err.append(err)
        if r["n_tau"] > 0:
            attr.append(1.0 / r["n_tau"])
        fresh_true.append(t["true_clamped_mean"])
        fresh_noisy.append(r["released_value"])
    if not abs_err:
        return float("nan"), float("nan"), float("nan"), float("nan")
    mae = float(np.mean(abs_err))
    nmae = mae / max(payload_bound, 1e-9)
    # KL using the same binning rules as dp_engine (import-on-demand).
    try:
        from dp_engine import compute_kl_divergence
        kl = float(compute_kl_divergence(np.array(fresh_true),
                                         np.array(fresh_noisy)))
    except Exception:
        kl = float("nan")
    attr_adv = float(np.mean(attr)) if attr else float("nan")
    return nmae, mae, kl, attr_adv


def _plot_experiment_E(df, path, dataset_name, sensor, live_dt, broker_host, broker_port):
    """Bar + scatter plot comparing live-broker metrics to offline reference."""
    if df.empty:
        return
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    # (a) NMAE live vs offline per config
    labels = [f"{r.strategy}\neps={r.epsilon}" for r in ok.itertuples()]
    x = np.arange(len(labels))
    bw = 0.38
    axes[0].bar(x - bw / 2, ok["nmae_live"], bw, label="live broker", color="C0")
    axes[0].bar(x + bw / 2, ok["nmae_offline"], bw, label="offline (same seed)", color="C1")
    axes[0].set(xticks=x, ylabel="NMAE",
                title="(a) Utility: live broker vs offline engine")
    axes[0].set_xticklabels(labels, fontsize=8, rotation=0)
    axes[0].grid(True, alpha=0.3, axis="y")
    axes[0].legend(fontsize=8)

    # (b) KL live vs offline
    axes[1].bar(x - bw / 2, ok["kl_live"], bw, label="live broker", color="C0")
    axes[1].bar(x + bw / 2, ok["kl_offline"], bw, label="offline (same seed)", color="C1")
    axes[1].set(xticks=x, ylabel="KL divergence",
                title="(b) Distributional utility")
    axes[1].set_xticklabels(labels, fontsize=8, rotation=0)
    axes[1].grid(True, alpha=0.3, axis="y")
    axes[1].legend(fontsize=8)

    # (c) Broker delivery integrity
    axes[2].bar(x - bw / 2, ok["num_released"], bw, label="plugin released", color="C2")
    axes[2].bar(x + bw / 2, ok["broker_deliveries"], bw, label="subscriber received", color="C3")
    axes[2].set(xticks=x, ylabel="count",
                title="(c) MQTT path integrity")
    axes[2].set_xticklabels(labels, fontsize=8, rotation=0)
    axes[2].grid(True, alpha=0.3, axis="y")
    axes[2].legend(fontsize=8)

    fig.suptitle(
        f"Experiment E: live MQTT broker [{broker_host}:{broker_port}] "
        f"dataset={dataset_name} sensor={sensor} dt={live_dt}s",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


