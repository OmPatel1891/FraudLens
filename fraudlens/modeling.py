"""Evaluation, calibration and threshold selection."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import COST_FALSE_NEGATIVE, COST_FALSE_POSITIVE


class ScaledModel:
    """Pairs a scaler with a linear model behind a plain predict_proba.

    Must stay at module level so joblib can round-trip it. A class defined in a
    notebook lives in __main__ and cannot be unpickled by the API process.
    """

    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    def predict_proba(self, X):
        return self.model.predict_proba(self.scaler.transform(X))

    def predict(self, X):
        return self.model.predict(self.scaler.transform(X))


class CalibratedModel:
    """Maps a tree model's scores onto calibrated probabilities.

    Training with scale_pos_weight to counter a 27:1 imbalance inflates raw
    predict_proba output, so those numbers cannot be shown to a risk team as
    probabilities. An isotonic fit on held-out data corrects the scale while
    preserving ranking, which leaves ROC-AUC and PR-AUC untouched.

    Implemented directly rather than via CalibratedClassifierCV because that
    estimator's `cv="prefit"` interface has changed across recent scikit-learn
    releases, and this artifact has to unpickle in whatever the API is running.
    """

    def __init__(self, base_model, calibrator: IsotonicRegression):
        self.base_model = base_model
        self.calibrator = calibrator

    @classmethod
    def fit(cls, base_model, X_calib, y_calib) -> CalibratedModel:
        raw = base_model.predict_proba(X_calib)[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(raw, np.asarray(y_calib))
        return cls(base_model, iso)

    def predict_proba(self, X):
        raw = self.base_model.predict_proba(X)[:, 1]
        calibrated = np.clip(self.calibrator.predict(raw), 0.0, 1.0)
        return np.column_stack([1.0 - calibrated, calibrated])

    def predict(self, X, threshold: float = 0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


def evaluate(y_true, proba, threshold: float = 0.5) -> dict:
    """Threshold-free ranking metrics plus operating-point metrics."""
    preds = (proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "threshold": float(threshold),
        "flag_rate": float(preds.mean()),
    }


@dataclass
class ThresholdChoice:
    """A candidate operating point with its business consequences."""

    threshold: float
    precision: float
    recall: float
    f1: float
    expected_cost: float
    flag_rate: float
    extra: dict = field(default_factory=dict)


def sweep_thresholds(
    y_true,
    proba,
    amounts=None,
    cost_fn: float = COST_FALSE_NEGATIVE,
    cost_fp: float = COST_FALSE_POSITIVE,
    n_steps: int = 200,
) -> pd.DataFrame:
    """Score every candidate threshold on both statistical and cost terms.

    `amounts` lets a missed fraud be charged at the real transaction value
    instead of a flat rate, which is what a risk team actually cares about.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    amounts = None if amounts is None else np.asarray(amounts, dtype=float)

    rows = []
    for t in np.linspace(0.001, 0.999, n_steps):
        preds = (proba >= t).astype(int)
        tp = int(((preds == 1) & (y_true == 1)).sum())
        fp = int(((preds == 1) & (y_true == 0)).sum())
        fn = int(((preds == 0) & (y_true == 1)).sum())

        if amounts is None:
            fn_cost = fn * cost_fn
        else:
            fn_cost = float(amounts[(preds == 0) & (y_true == 1)].sum())

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        rows.append(
            {
                "threshold": float(t),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "expected_cost": fn_cost + fp * cost_fp,
                "flag_rate": float(preds.mean()),
            }
        )

    return pd.DataFrame(rows)


def pick_threshold(sweep: pd.DataFrame, strategy: str = "min_cost", **kwargs) -> ThresholdChoice:
    """Choose an operating point from a completed sweep.

    strategy:
      min_cost        - lowest total expected cost (the business default)
      max_f1          - highest F1, kept for comparison against the baseline
      min_precision   - cheapest point meeting a precision floor, which is how
                        you cap the number of blocked legitimate customers
    """
    if sweep.empty:
        raise ValueError("threshold sweep is empty")

    if strategy == "min_cost":
        row = sweep.loc[sweep["expected_cost"].idxmin()]
    elif strategy == "max_f1":
        row = sweep.loc[sweep["f1"].idxmax()]
    elif strategy == "min_precision":
        floor = kwargs.get("min_precision", 0.9)
        eligible = sweep[sweep["precision"] >= floor]
        if eligible.empty:
            # No threshold reaches the floor; fall back to the best available.
            row = sweep.loc[sweep["precision"].idxmax()]
        else:
            row = eligible.loc[eligible["recall"].idxmax()]
    else:
        raise ValueError(f"unknown strategy: {strategy}")

    return ThresholdChoice(
        threshold=float(row["threshold"]),
        precision=float(row["precision"]),
        recall=float(row["recall"]),
        f1=float(row["f1"]),
        expected_cost=float(row["expected_cost"]),
        flag_rate=float(row["flag_rate"]),
        extra={
            "strategy": strategy,
            "tp": int(row["tp"]),
            "fp": int(row["fp"]),
            "fn": int(row["fn"]),
        },
    )


def precision_at_recall(y_true, proba, target_recall: float = 0.8) -> float:
    """Precision achievable at a required recall - a common risk-team SLA."""
    precision, recall, _ = precision_recall_curve(y_true, proba)
    viable = precision[recall >= target_recall]
    return float(viable.max()) if len(viable) else 0.0
