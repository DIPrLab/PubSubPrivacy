#!/usr/bin/env python3
"""Experiment J: broker latency (paper Sec. 6, RQ2).

What PAWS costs the broker per message and per release, against a broker that
just forwards messages and against a w-event DP plugin.

In-process (default, no network): each dataset's per-publisher trace is
replayed through the broker-side code on a stubbed MQTT client, with every
publisher on its own leaf of the dataset's topic tree, under

  passthrough   decode + forward each message (no privacy)
  wevent        the plugin with Kellaris uniform w-event DP (no gate/walk/wait)
  paws_nowait   PAWS gate + cached walk, no adaptive interval
  paws_nocache  PAWS with the adaptive interval and per-level walk counts
  paws          PAWS as deployed: adaptive interval + cached walk, under our
                strategy and under every other budget strategy

and the real CPU time of the broker code is measured: per-message ingest
(us), per-interval close/release (ms, mean and p95), the whole release pass
per Delta_t tick (ms) -- the delay the last leaf of a tick waits for -- and
broker CPU per Delta_t.

Live (``--live``): through a real MQTT broker (a local Mosquitto is started
when none is listening; ``--broker-impl amqtt`` uses the embedded pure-Python
broker instead), the publish -> subscriber delivery latency of
a plain forward on the raw topic, and for PAWS the close -> delivery latency
of each release plus the data age at delivery (delivery - interval start,
which includes the interval itself and any adaptive wait).

Run: python -m experiments.latency --dataset all --trials 3 [--live]
"""
from __future__ import annotations

import json
import os
import threading
import time

import numpy as np
import pandas as pd

from experiments import _common as C
from experiments import engine as core
from experiments import pipeline as pl

CONFIGS = ("passthrough", "wevent", "paws_nowait", "paws_nocache", "paws")


def _plugin_options(config: str, prm: dict, strategy: str) -> dict:
    if config == "wevent":
        return dict(strategy="uniform", k_ext=0, wait_policy="none")
    k_ext = max(2, prm["k_ext"])   # predictive sizing needs T_max > Delta_t
    return {
        "paws_nowait": dict(strategy=strategy, k_ext=0, wait_policy="none",
                            count_cache=True),
        "paws_nocache": dict(strategy=strategy, k_ext=k_ext,
                             wait_policy="predictive", count_cache=False),
        "paws": dict(strategy=strategy, k_ext=k_ext, wait_policy="predictive",
                     count_cache=True),
    }[config]


def _passthrough_run(per_pub, sensor, spec, n_steps) -> dict:
    """Decode + forward every message, as a broker with no privacy layer."""
    mock = core._MockMQTTClient()
    topic_of = spec["publisher_topic"]
    seconds, n = 0.0, 0
    pubs = list(per_pub.keys())
    T = min(n_steps, len(per_pub[pubs[0]]))
    for tau in range(T):
        for p in pubs:
            v = per_pub[p][tau]
            if v is None:
                continue
            msg = core._FakeMsg(f"raw/{topic_of(p, sensor)}",
                                json.dumps({"publisher_id": str(p),
                                            "value": float(v)}).encode())
            t0 = time.perf_counter()
            body = json.loads(msg.payload.decode())
            mock.publish("protected/" + msg.topic[len("raw/"):],
                         json.dumps({"value": body["value"]}))
            seconds += time.perf_counter() - t0
            n += 1
    per_msg_ms = 1000.0 * seconds / max(1, n)
    return {"messages": n, "ticks": T, "releases": n,
            "ingest_us_per_msg": 1000.0 * per_msg_ms,
            "close_ms_mean": 0.0, "close_ms_p95": 0.0,
            "flush_ms_per_tick_mean": 0.0, "flush_ms_per_tick_p95": 0.0,
            "broker_cpu_ms_per_dt": 1000.0 * seconds / max(1, T),
            "mean_wait_dt": 0.0, "count_units_per_release": 0.0}


def _plugin_run(per_pub, sensor, spec, n_steps, seed, epsilon, w, prm,
                options) -> dict:
    res = core._drive_plugin_scenario(
        per_pub, sensor, spec, "hierarchy", epsilon=epsilon, w=w,
        P=prm["P_min"], delta_t=1.0, n_steps=n_steps, seed=seed,
        rho_split=prm["rho"], min_publishers=prm["P_min"],
        max_publishers=prm["P_max"], **options)
    plugin = res["plugin"]
    log = plugin.release_log
    proc = np.array([r["processing_ms"] for r in log]) if log else np.zeros(0)
    per_tick = (pd.DataFrame(log).groupby("t_close")["processing_ms"].sum().to_numpy()
                if log else np.zeros(0))
    released = [r for r in log if r["released_value"] is not None]
    ticks = len(res["tau_log"])
    return {
        "messages": plugin.ingest_messages, "ticks": ticks,
        "releases": len(released),
        "ingest_us_per_msg": 1e6 * plugin.ingest_seconds / max(1, plugin.ingest_messages),
        "close_ms_mean": float(proc.mean()) if proc.size else float("nan"),
        "close_ms_p95": float(np.percentile(proc, 95)) if proc.size else float("nan"),
        "flush_ms_per_tick_mean": float(per_tick.mean()) if per_tick.size else float("nan"),
        "flush_ms_per_tick_p95": (float(np.percentile(per_tick, 95))
                                  if per_tick.size else float("nan")),
        "broker_cpu_ms_per_dt": (1000.0 * plugin.ingest_seconds + float(proc.sum()))
                                / max(1, ticks),
        "mean_wait_dt": float(np.mean([r["wait_dt"] for r in log])) if log else float("nan"),
        "count_units_per_release": (float(np.mean([r["count_units"] for r in released]))
                                    if released else float("nan")),
    }


def inprocess_latency(datasets, clamp_mode, args, *, trials=1, n_steps=300):
    eps, w = float(args.fixed_epsilon), int(args.fixed_w)
    strategy = getattr(args, "our_strategy", "n_weighted")
    rows = []
    for ds in datasets:
        prepared = core.prepare_dataset(
            ds, clamp_mode=clamp_mode, eps_clip=args.eps_clip, seed=args.seed,
            max_rows=core._dataset_max_rows(ds, args))
        if prepared is None or not prepared.per_pubs:
            continue
        spec = prepared.spec
        sensor = next((s for s in spec["sensors"] if s in prepared.per_pubs), None)
        if sensor is None or "publisher_topic" not in spec:
            continue
        per_pub = prepared.per_pubs[sensor][0]
        # The component variants run under our strategy; the full PAWS broker
        # also runs under every other budget strategy, each at its own grid
        # optimum, so the per-strategy broker cost is measured too.
        runs = [(c, strategy) for c in CONFIGS]
        runs += [("paws", st) for st in (getattr(args, "strategies", None)
                                         or core.ALL_STRATEGIES) if st != strategy]
        for config, st in runs:
            prm = pl.resolve_paws_params(core, getattr(args, "grid_config", None),
                                         ds, clamp_mode, st, eps, args)
            for trial in range(max(1, trials)):
                seed = 123 + 1000 * trial
                if config == "passthrough":
                    m = _passthrough_run(per_pub, sensor, spec, n_steps)
                else:
                    m = _plugin_run(per_pub, sensor, spec, n_steps, seed, eps, w,
                                    prm, _plugin_options(config, prm, st))
                rows.append({"dataset": ds, "sensor": sensor, "clamp_mode": clamp_mode,
                             "config": config,
                             "strategy": ("uniform" if config == "wevent" else
                                          "-" if config == "passthrough" else st),
                             "ours": (config not in ("passthrough", "wevent")
                                      and st == strategy),
                             "n_publishers": len(per_pub),
                             "epsilon": eps, "w": w, "P_min": prm["P_min"],
                             "rho_split": prm["rho"], "trial": trial, **m})
        core.logger.info(f"  [exp J {ds}] in-process latency done")
    return pd.DataFrame(rows)


# ── live broker ─────────────────────────────────────────────────────────────

def _live_passthrough(host, port, per_pub, sensor, spec, live_dt, n_steps) -> dict:
    """Publish -> delivery latency of a plain forward through the broker."""
    import uuid
    import paho.mqtt.client as mqtt
    run = uuid.uuid4().hex[:8]
    prefix = f"pubsubpriv/{run}/raw"
    lat_ms, ready = [], threading.Event()

    def on_message(_c, _u, msg):
        try:
            sent = json.loads(msg.payload.decode())["sent"]
        except Exception:
            return
        lat_ms.append(1000.0 * (time.time() - sent))

    sub = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                      client_id=f"lat-sub-{run}")
    sub.on_connect = lambda c, *_a, **_k: (c.subscribe(f"{prefix}/#", qos=1), ready.set())
    sub.on_message = on_message
    pub = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                      client_id=f"lat-pub-{run}")
    sub.connect(host, port); sub.loop_start()
    pub.connect(host, port); pub.loop_start()
    ready.wait(5.0)
    time.sleep(0.3)
    topic_of = spec["publisher_topic"]
    pubs = list(per_pub.keys())
    sent = 0
    try:
        for tau in range(min(n_steps, len(per_pub[pubs[0]]))):
            t_tau = time.time()
            for p in pubs:
                v = per_pub[p][tau]
                if v is None:
                    continue
                pub.publish(f"{prefix}/{topic_of(p, sensor)}",
                            json.dumps({"value": float(v), "sent": time.time()}),
                            qos=1).wait_for_publish(timeout=2.0)
                sent += 1
            rest = live_dt - (time.time() - t_tau)
            if rest > 0:
                time.sleep(rest)
        time.sleep(1.0)
    finally:
        for c in (pub, sub):
            c.loop_stop()
            c.disconnect()
    a = np.array(lat_ms)
    return {"messages": sent, "delivered": int(a.size),
            "delivery_ms_mean": float(a.mean()) if a.size else float("nan"),
            "delivery_ms_p50": float(np.median(a)) if a.size else float("nan"),
            "delivery_ms_p95": float(np.percentile(a, 95)) if a.size else float("nan")}


def _live_paws(host, port, per_pub, sensor, spec, live_dt, n_steps, seed, eps, w,
               prm, options) -> dict:
    from experiments.live_broker import _drive_live_config
    out = _drive_live_config(
        per_pub, sensor, spec, strategy=options.pop("strategy"), epsilon=eps,
        w=w, P=prm["P_min"], broker_host=host, broker_port=port,
        live_dt=live_dt, n_steps=n_steps, seed=seed, scenario="hierarchy",
        n_subscribers=1,
        plugin_kwargs=dict(min_publishers=prm["P_min"], rho_split=prm["rho"],
                           max_publishers=prm["P_max"], **options))
    # Match each delivery to its release by (leaf, t_start, value): a leaf that
    # walks up can pool a sibling's still-open buffer on consecutive ticks, so
    # (leaf, t_start) alone is not unique.
    prefix = out["protected_prefix"] + "/"
    closes = {(r["leaf_topic"], round(float(r["t_start"]), 6),
               round(float(r["released_value"]), 4)): r
              for r in out["plugin_log"] if r["released_value"] is not None}
    delivery, age = [], []
    for m in out["subscriber_received"]:
        rec = closes.get((m["topic"][len(prefix):], round(m["t_start"], 6),
                          round(m["value"], 4)))
        if rec is None:
            continue
        delivery.append(1000.0 * (m["wall_clock_recv"] - rec["t_close"]))
        age.append(1000.0 * (m["wall_clock_recv"] - rec["t_start"]))
    d, a = np.array(delivery), np.array(age)
    log = out["plugin_log"]
    return {"messages": out["num_published"], "delivered": int(d.size),
            "releases": len(closes),
            "delivery_ms_mean": float(d.mean()) if d.size else float("nan"),
            "delivery_ms_p50": float(np.median(d)) if d.size else float("nan"),
            "delivery_ms_p95": float(np.percentile(d, 95)) if d.size else float("nan"),
            "data_age_ms_mean": float(a.mean()) if a.size else float("nan"),
            "data_age_ms_p95": float(np.percentile(a, 95)) if a.size else float("nan"),
            "close_ms_mean": (float(np.mean([r["processing_ms"] for r in log]))
                              if log else float("nan")),
            "mean_wait_dt": (float(np.mean([r["wait_dt"] for r in log]))
                             if log else float("nan"))}


def live_latency(datasets, clamp_mode, args, *, live_dt=0.2, n_steps=60):
    from experiments.live_broker import _ensure_broker
    eps, w = float(args.fixed_epsilon), int(args.fixed_w)
    strategy = getattr(args, "our_strategy", "n_weighted")
    host, port = args.broker_host, args.broker_port
    broker = _ensure_broker(host, port, auto_start=True,
                            impl=getattr(args, "broker_impl", "mosquitto"))
    rows = []
    try:
        for ds in datasets:
            prepared = core.prepare_dataset(
                ds, clamp_mode=clamp_mode, eps_clip=args.eps_clip, seed=args.seed,
                max_rows=core._dataset_max_rows(ds, args))
            if prepared is None or not prepared.per_pubs:
                continue
            spec = prepared.spec
            sensor = next((s for s in spec["sensors"] if s in prepared.per_pubs), None)
            if sensor is None or "publisher_topic" not in spec:
                continue
            per_pub = prepared.per_pubs[sensor][0]
            prm = pl.resolve_paws_params(core, getattr(args, "grid_config", None),
                                         ds, clamp_mode, strategy, eps, args)
            base = {"dataset": ds, "sensor": sensor, "clamp_mode": clamp_mode,
                    "n_publishers": len(per_pub), "live_dt_s": live_dt,
                    "epsilon": eps, "w": w}
            rows.append({**base, "config": "passthrough",
                         **_live_passthrough(host, port, per_pub, sensor, spec,
                                             live_dt, n_steps)})
            for config in ("wevent", "paws"):
                opts = _plugin_options(config, prm, strategy)
                rows.append({**base, "config": config,
                             **_live_paws(host, port, per_pub, sensor, spec, live_dt,
                                          n_steps, 123, eps, w, prm, opts)})
            core.logger.info(f"  [exp J live {ds}] done")
    finally:
        if broker is not None:
            broker.stop()
    return pd.DataFrame(rows)


def main():
    p = C.make_parser("Experiment J: broker latency (in-process, optional live).")
    p.add_argument("--latency-steps", type=int, default=None,
                   help="Logical timestamps replayed per run (default 300; "
                        "--quick 80; live runs use a fifth of it).")
    p.add_argument("--live-dt", type=float, default=0.2,
                   help="Wall-clock seconds per Delta_t in the live run.")
    args = C.resolve(p.parse_args())
    steps = args.latency_steps or (80 if args.quick else 300)
    for clamp_mode in args._clamp_modes:
        exp_dir = os.path.join(C.cross_dir(args, clamp_mode), "experiments", "J_latency")
        os.makedirs(exp_dir, exist_ok=True)
        df = inprocess_latency(args._targets, clamp_mode, args,
                               trials=max(1, args.trials), n_steps=steps)
        df.to_csv(os.path.join(exp_dir, "experiment_J_latency_inprocess.csv"), index=False)
        if not df.empty:
            metrics = [c for c in df.columns if c.endswith(("_us_per_msg", "_ms_mean",
                       "_ms_p95", "_per_dt", "mean_wait_dt", "count_units_per_release"))]
            core._aggregate_over_trials(
                df, ["dataset", "sensor", "clamp_mode", "config", "strategy", "ours",
                     "n_publishers"],
                metrics).to_csv(os.path.join(
                    exp_dir, "experiment_J_latency_inprocess_aggregate.csv"), index=False)
        core.logger.info(f"  Experiment J (in-process) wrote {len(df)} rows -> {exp_dir}")
        if args.live:
            live = live_latency(args._targets, clamp_mode, args, live_dt=args.live_dt,
                                n_steps=max(10, steps // 5))
            live.to_csv(os.path.join(exp_dir, "experiment_J_latency_live.csv"), index=False)
            core.logger.info(f"  Experiment J (live) wrote {len(live)} rows -> {exp_dir}")


if __name__ == "__main__":
    main()
