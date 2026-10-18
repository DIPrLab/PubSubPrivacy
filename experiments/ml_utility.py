#!/usr/bin/env python3
"""Experiment M: downstream machine-learning utility (paper Sec. 6, RQ1).

A subscriber that trains a model on the stream it receives.  For every dataset,
sensor and subscription level, each approach -- classic (no privacy, the
ceiling), local DP, w-event DP and PAWS under every budget strategy (each at
its own grid optimum; n-weighted is ours) -- produces the subscriber's stream,
and the subscriber fits two standard downstream models on it:

  forecast  Ridge regression predicting the next interval's value from the
            last ``--ml-lags`` values it holds; scored as NMAE against the TRUE
            next value.
  classify  logistic regression predicting whether the next value exceeds the
            task threshold (the median true value of the training period, the
            same for every approach); scored as balanced accuracy and F1
            against the true labels.

Protocol, on the base-Delta_t grid shared by all approaches: the subscriber's
view at step j is the most recent value it has received by the end of step j
(forward-filled when nothing has arrived yet; on a deferral the broker
repeats its previous output).  The first ``--ml-train-frac`` of the
steps train the model, using as targets only the steps at which a fresh value
arrived; the rest are the test set, scored against the truth.  Features at test
time are the subscriber's view, exactly as in deployment.

Run: python -m experiments.ml_utility --dataset all --trials 6
"""
from __future__ import annotations

import warnings

import numpy as np

from experiments import _common as C
from experiments import _section7 as S7
from experiments import pipeline as pl

BASELINES = ("classic", "ldp", "wevent")
ML_METRIC_COLS = S7.ML_METRIC_COLS


def subscriber_grid(mat: pl.PubMatrix, sub_mask, series, base_dt: int):
    """(truth, view, fresh) on the base-Delta_t grid.

    truth[j] -- true subscribed-scope mean over base step j (NaN if no data)
    view[j]  -- latest value delivered by the end of step j (NaN before any)
    fresh[j] -- a release arrived during step j
    """
    T = mat.T
    J = (T + base_dt - 1) // base_dt
    truth = np.full(J, np.nan)
    for j in range(J):
        active, means = mat.window(j * base_dt, min((j + 1) * base_dt, T))
        sel = active & sub_mask
        if sel.any():
            truth[j] = means[sel].mean()
    arrived = np.full(J, np.nan)
    for end, value in zip(series["end"], series["released"]):
        if value is not None and np.isfinite(value):
            arrived[min(J - 1, (end - 1) // base_dt)] = value
    fresh = np.isfinite(arrived)
    view = arrived.copy()
    last = np.nan
    for j in range(J):
        if np.isfinite(view[j]):
            last = view[j]
        view[j] = last
    return truth, view, fresh, arrived


def evaluate_downstream(truth, view, fresh, arrived, payload_bound, *,
                        lags: int = 4, train_frac: float = 0.7) -> dict:
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import balanced_accuracy_score, f1_score
    from sklearn.preprocessing import StandardScaler

    nan = float("nan")
    out = {"ml_forecast_nmae": nan, "ml_class_bal_acc": nan, "ml_class_f1": nan,
           "ml_n_train": 0, "ml_n_test": 0}
    J = len(truth)
    split = int(train_frac * J)
    if J < lags + 4 or split <= lags:
        return out
    X_tr, y_tr, X_te, y_te = [], [], [], []
    for j in range(lags - 1, J - 1):
        x = view[j - lags + 1:j + 1]
        if not np.all(np.isfinite(x)):
            continue
        if j + 1 < split:
            if fresh[j + 1]:
                X_tr.append(x)
                y_tr.append(arrived[j + 1])
        elif np.isfinite(truth[j + 1]):
            X_te.append(x)
            y_te.append(truth[j + 1])
    out["ml_n_train"], out["ml_n_test"] = len(y_tr), len(y_te)
    if len(y_tr) < 5 or len(y_te) < 5:
        return out
    X_tr, y_tr = np.array(X_tr), np.array(y_tr)
    X_te, y_te = np.array(X_te), np.array(y_te)
    scaler = StandardScaler().fit(X_tr)
    Z_tr, Z_te = scaler.transform(X_tr), scaler.transform(X_te)

    reg = Ridge(alpha=1.0).fit(Z_tr, y_tr)
    out["ml_forecast_nmae"] = float(np.mean(np.abs(reg.predict(Z_te) - y_te))
                                    / payload_bound)

    threshold = float(np.nanmedian(truth[:split]))
    c_tr, c_te = y_tr > threshold, y_te > threshold
    if c_tr.all() or not c_tr.any():
        pred = np.full(len(c_te), bool(c_tr[0]))
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = LogisticRegression(max_iter=1000).fit(Z_tr, c_tr).predict(Z_te)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out["ml_class_bal_acc"] = float(balanced_accuracy_score(c_te, pred))
        out["ml_class_f1"] = float(f1_score(c_te, pred, zero_division=0))
    return out


def ml_task(task) -> dict:
    """Simulate the subscription, then train/evaluate the downstream models."""
    meta, mat_key, chain_key, kwargs = task
    kwargs = dict(kwargs)
    lags = kwargs.pop("_ml_lags", 4)
    train_frac = kwargs.pop("_ml_train_frac", 0.7)
    mat = pl._WORKER_CTX["mats"][mat_key]
    chain = pl._WORKER_CTX["chains"][chain_key]
    res = pl.simulate_subscription(mat, chain, keep_series=True, **kwargs)
    truth, view, fresh, arrived = subscriber_grid(
        mat, chain[0], res["series"], int(kwargs.get("base_dt", 1)))
    ml = evaluate_downstream(truth, view, fresh, arrived, kwargs["payload_bound"],
                             lags=lags, train_frac=train_frac)
    return {**meta, **res["metrics"], **ml}


def ml_experiment(datasets, clamp_mode, output_dir, args, *, trials: int = 1):
    points = [{"epsilon": float(e), "w": int(args.fixed_w)} for e in args.eps_values]

    def kwargs_fn(ds, sensor, point, approach):
        kw, meta = S7.approach_kwargs(args, ds, clamp_mode, approach,
                                      point["epsilon"], point["w"])
        kw.update(_ml_lags=args.ml_lags, _ml_train_frac=args.ml_train_frac)
        return kw, meta

    approaches = BASELINES + S7.paws_approaches(args)
    df = S7.run_points(datasets, clamp_mode, args, points=points,
                       approaches=approaches, kwargs_fn=kwargs_fn,
                       label="exp M", trials=trials, task_fn=ml_task)
    S7.write_experiment(df, output_dir, "experiment_M_ml", trials)
    return df


def main():
    p = C.make_parser("Experiment M: downstream ML utility on the released stream.")
    p.add_argument("--ml-lags", type=int, default=4,
                   help="Lagged values the subscriber's models use as features.")
    p.add_argument("--ml-train-frac", type=float, default=0.7,
                   help="Leading fraction of the stream used for training.")
    args = C.resolve(p.parse_args())
    for clamp_mode in args._clamp_modes:
        ml_experiment(args._targets, clamp_mode, C.cross_dir(args, clamp_mode),
                      args, trials=max(1, args.trials))


if __name__ == "__main__":
    main()
