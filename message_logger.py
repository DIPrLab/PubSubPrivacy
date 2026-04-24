"""
Per-release message logging for clamped w-event DP with P-allocation.

Every DP run releases (or defers) one logical-timestamp element per tau.  This
module captures each of those as a row in an append-friendly CSV so downstream
analysis can audit the full message trail:

    - the "original message" = the clamped-mean aggregate e_tau the broker
      computed before noise injection (Definition 3.3),
    - the "output message" = the noisy released value hat{e}_tau delivered to
      the subscriber (or the repeat-of-last for deferred taus),
    - every piece of metadata that determines either (config context, DP
      parameters, broker-internal n_tau / epsilon_tau / lambda_tau / deferred
      flag / t_start_logical).

The DP engine never transmits broker-internal quantities to subscribers
(Definition 3.4); they are logged here only for offline analysis of the
mechanism.  The logger is side-channel-only and does not feed back into any
DP calculation.
"""

from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import pandas as pd


# Canonical column order for every per-release log written by this module.
MESSAGE_LOG_COLUMNS = [
    "dataset",
    "clamp_mode",
    "sensor",
    "strategy",
    "P",
    "epsilon",
    "w",
    "payload_bound",
    "seed",
    "experiment",
    "config_id",
    "tau",                   # logical timestamp (broker-internal, for analysis)
    "t_start_logical",       # (tau-1) in Delta_t units (offline proxy for Def 3.1)
    "true_aggregate",        # clamped-mean e_tau BEFORE noise ("original message")
    "noisy_value",           # hat{e}_tau delivered to subscriber ("output message")
    "n_tau",                 # multiplicity |P_tau| (broker-internal)
    "epsilon_tau",           # per-element budget spent at tau
    "lambda_tau",            # Laplace scale Delta_f / epsilon_tau
    "noise_sample",          # noisy - true (0.0 for deferred)
    "deferred",              # True when release gate deferred (repeat-last)
    "delta_f",               # per-element sensitivity R / n_tau (default mean)
]


def _safe_lambda(payload_bound: float, n_tau: int, eps_tau: float) -> float:
    if eps_tau is None or eps_tau <= 0 or n_tau is None or n_tau <= 0:
        return float("inf")
    return float(payload_bound) / (float(n_tau) * float(eps_tau))


def _safe_delta_f(payload_bound: float, n_tau: int) -> float:
    if n_tau is None or n_tau <= 0:
        return float("inf")
    return float(payload_bound) / float(n_tau)


def build_message_rows(
    result: dict,
    *,
    dataset: str,
    clamp_mode: str,
    sensor: str,
    strategy: str,
    P: int,
    epsilon: float,
    w: int,
    payload_bound: float,
    seed: int,
    experiment: str = "sweep",
    config_id: str | None = None,
) -> list[dict]:
    """Flatten one ``run_dp_on_stream`` result into per-tau message rows.

    ``result`` must be the dict returned by ``run_experiment.run_dp_on_stream``
    (it contains aligned ``true_values``, ``noisy_values``, ``budgets_spent``,
    and ``pub_counts`` lists — one entry per logical timestamp).

    The returned rows share the ``MESSAGE_LOG_COLUMNS`` schema so multiple
    configurations can be concatenated into a single CSV downstream.
    """
    true_vals = result.get("true_values") or []
    noisy_vals = result.get("noisy_values") or []
    budgets = result.get("budgets_spent") or []
    pub_counts = result.get("pub_counts") or []

    n = min(len(true_vals), len(noisy_vals), len(budgets), len(pub_counts))
    rows = []
    if config_id is None:
        config_id = (
            f"{dataset}|{clamp_mode}|{sensor}|{strategy}|P={P}|eps={epsilon}|"
            f"w={w}|seed={seed}"
        )
    for tau in range(n):
        true_v = true_vals[tau]
        noisy_v = noisy_vals[tau]
        eps_tau = float(budgets[tau]) if budgets[tau] is not None else 0.0
        n_tau = int(pub_counts[tau]) if pub_counts[tau] is not None else 0
        deferred = eps_tau <= 0
        if true_v is not None and noisy_v is not None and not deferred:
            noise = float(noisy_v) - float(true_v)
        else:
            noise = 0.0
        rows.append({
            "dataset": dataset,
            "clamp_mode": clamp_mode,
            "sensor": sensor,
            "strategy": strategy,
            "P": int(P),
            "epsilon": float(epsilon),
            "w": int(w),
            "payload_bound": float(payload_bound),
            "seed": int(seed),
            "experiment": experiment,
            "config_id": config_id,
            "tau": int(tau + 1),             # 1-indexed to match StreamState.current_tau
            "t_start_logical": float(tau),   # (tau-1) * Delta_t with Delta_t=1 offline
            "true_aggregate": (float(true_v) if true_v is not None else None),
            "noisy_value": (float(noisy_v) if noisy_v is not None else None),
            "n_tau": n_tau,
            "epsilon_tau": eps_tau,
            "lambda_tau": _safe_lambda(payload_bound, n_tau, eps_tau),
            "noise_sample": noise,
            "deferred": bool(deferred),
            "delta_f": _safe_delta_f(payload_bound, n_tau),
        })
    return rows


def write_messages_csv(
    rows: Iterable[dict],
    path: str,
    *,
    append: bool = False,
) -> int:
    """Write ``rows`` to ``path`` with a stable column order.  Returns count."""
    rows = list(rows)
    if not rows:
        return 0
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df = pd.DataFrame(rows)
    # Stabilize column ordering; keep any extras at the end.
    ordered = [c for c in MESSAGE_LOG_COLUMNS if c in df.columns]
    extras = [c for c in df.columns if c not in ordered]
    df = df[ordered + extras]
    mode = "a" if append and os.path.exists(path) else "w"
    header = not (append and os.path.exists(path))
    df.to_csv(path, index=False, mode=mode, header=header)
    return len(df)


def flatten_runs_to_frame(
    runs: Iterable[tuple[dict, dict]],
) -> pd.DataFrame:
    """Given an iterable of (result, context) pairs, return one DataFrame.

    ``context`` supplies the non-result columns (dataset, strategy, ...) and
    must contain every argument required by ``build_message_rows``.
    """
    all_rows: list[dict] = []
    for result, ctx in runs:
        all_rows.extend(build_message_rows(result, **ctx))
    if not all_rows:
        return pd.DataFrame(columns=MESSAGE_LOG_COLUMNS)
    df = pd.DataFrame(all_rows)
    ordered = [c for c in MESSAGE_LOG_COLUMNS if c in df.columns]
    extras = [c for c in df.columns if c not in ordered]
    return df[ordered + extras]
