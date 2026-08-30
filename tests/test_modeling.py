"""Calibration and threshold-selection tests."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score

from fraudlens.modeling import (
    CalibratedModel,
    evaluate,
    pick_threshold,
    precision_at_recall,
    sweep_thresholds,
)


class _Overconfident:
    """Stand-in for a model trained with scale_pos_weight: right order, wrong scale."""

    def __init__(self, scores):
        self.scores = np.asarray(scores, dtype=float)

    def predict_proba(self, X):
        idx = np.asarray(X).ravel().astype(int)
        p = np.clip(self.scores[idx], 1e-6, 1 - 1e-6)
        return np.column_stack([1 - p, p])


def _imbalanced(n=4000, rate=0.035, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < rate).astype(int)
    # Ranking is informative but the scale is inflated, like a weighted GBM.
    raw = np.clip(rng.beta(2, 5, n) + y * 0.45, 0, 1)
    return y, raw


def test_calibration_lowers_brier_without_hurting_ranking():
    y, raw = _imbalanced()
    X = np.arange(len(y)).reshape(-1, 1)
    model = _Overconfident(raw)

    before_auc = roc_auc_score(y, raw)
    before_brier = brier_score_loss(y, raw)

    calibrated = CalibratedModel.fit(model, X, y)
    after = calibrated.predict_proba(X)[:, 1]

    assert brier_score_loss(y, after) < before_brier
    # Isotonic regression is monotonic, so discrimination must be preserved.
    assert roc_auc_score(y, after) >= before_auc - 1e-6


def test_calibrated_output_is_a_valid_distribution():
    y, raw = _imbalanced()
    X = np.arange(len(y)).reshape(-1, 1)
    proba = CalibratedModel.fit(_Overconfident(raw), X, y).predict_proba(X)

    assert np.all((proba >= 0) & (proba <= 1))
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_cost_threshold_differs_from_f1_threshold():
    """The business optimum should not coincide with the F1 optimum."""
    y, raw = _imbalanced()
    sweep = sweep_thresholds(y, raw, cost_fn=100.0, cost_fp=5.0)

    cost = pick_threshold(sweep, strategy="min_cost")
    f1 = pick_threshold(sweep, strategy="max_f1")

    assert cost.threshold != f1.threshold
    # Missing fraud is 20x costlier here, so the cost optimum must catch more.
    assert cost.recall > f1.recall


def test_amount_weighting_lowers_the_threshold_for_costly_frauds():
    """Charging a missed fraud at its real value should move the threshold.

    Constructed so the effect is unambiguous: a handful of very expensive
    frauds sit at low scores. A flat per-incident cost writes them off as not
    worth the false positives; amount weighting makes catching them essential.
    """
    n_legit, n_cheap, n_pricey = 900, 90, 10

    y = np.r_[np.zeros(n_legit), np.ones(n_cheap), np.ones(n_pricey)].astype(int)
    scores = np.r_[
        np.linspace(0.01, 0.60, n_legit),   # legitimate, spread across low scores
        np.linspace(0.80, 0.95, n_cheap),   # cheap fraud the model finds easily
        np.linspace(0.20, 0.30, n_pricey),  # expensive fraud hiding at low scores
    ]
    amounts = np.r_[np.full(n_legit, 50.0), np.full(n_cheap, 20.0), np.full(n_pricey, 25000.0)]

    flat = pick_threshold(sweep_thresholds(y, scores, cost_fn=100.0, cost_fp=5.0), "min_cost")
    weighted = pick_threshold(
        sweep_thresholds(y, scores, amounts=amounts, cost_fp=5.0), "min_cost"
    )

    assert weighted.threshold < flat.threshold
    assert weighted.recall > flat.recall


def test_precision_floor_strategy_respects_the_floor():
    y, raw = _imbalanced()
    sweep = sweep_thresholds(y, raw)
    choice = pick_threshold(sweep, strategy="min_precision", min_precision=0.5)
    assert choice.precision >= 0.5 or choice.precision == sweep["precision"].max()


def test_evaluate_reports_ranking_and_operating_metrics():
    y, raw = _imbalanced()
    metrics = evaluate(y, raw, threshold=0.3)
    for key in ("roc_auc", "pr_auc", "brier", "f1", "precision", "recall", "flag_rate"):
        assert key in metrics
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_precision_at_recall_target():
    y, raw = _imbalanced()
    assert 0.0 <= precision_at_recall(y, raw, 0.8) <= 1.0
