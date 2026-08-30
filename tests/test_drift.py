"""PSI drift detection tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraudlens import drift
from fraudlens.config import MONITOR_ALWAYS


def _frame(rng, n=4000, shift=0.0, scale=1.0):
    return pd.DataFrame(
        {
            "a": rng.normal(shift, scale, n),
            "b": rng.gamma(2.0, 1.0, n) + shift,
        }
    )


def test_identical_distributions_are_stable():
    rng = np.random.default_rng(0)
    reference = drift.build_reference(_frame(rng), ["a", "b"])
    report = drift.compute_drift(_frame(rng), reference)

    assert report["status"] == "ok"
    assert report["features_drifted"] == 0
    assert not report["alert"]


def test_large_shift_is_flagged_major():
    rng = np.random.default_rng(1)
    reference = drift.build_reference(_frame(rng), ["a", "b"])
    report = drift.compute_drift(_frame(rng, shift=3.0), reference)

    assert report["features_drifted"] == 2
    assert set(report["major_drift"]) == {"a", "b"}
    assert report["alert"]


def test_psi_grows_with_shift():
    rng = np.random.default_rng(2)
    reference = drift.build_reference(_frame(rng), ["a"])
    psi = [
        drift.compute_drift(_frame(rng, shift=s), reference)["features"]["a"]["psi"]
        for s in (0.0, 0.5, 2.0)
    ]
    assert psi[0] < psi[1] < psi[2]


def test_small_window_reports_insufficient_data():
    rng = np.random.default_rng(3)
    reference = drift.build_reference(_frame(rng), ["a"])
    report = drift.compute_drift(_frame(rng, n=10), reference, min_rows=50)

    assert report["status"] == "insufficient_data"
    assert report["rows_observed"] == 10


def test_reference_survives_a_round_trip(tmp_path):
    rng = np.random.default_rng(4)
    reference = drift.build_reference(_frame(rng), ["a", "b"])
    path = tmp_path / "ref.json"
    drift.save_reference(reference, path)
    assert drift.load_reference(path).keys() == reference.keys()


def test_constant_columns_are_skipped():
    frame = pd.DataFrame({"constant": np.ones(500), "varying": np.arange(500.0)})
    reference = drift.build_reference(frame, ["constant", "varying"])
    assert "constant" not in reference
    assert "varying" in reference


def test_severity_bands():
    assert drift.classify(0.02) == "stable"
    assert drift.classify(0.15) == "minor"
    assert drift.classify(0.40) == "major"


def test_amount_is_monitored_even_when_shap_ignores_it():
    """A currency or scaling break in TransactionAmt must not slip through.

    Monitoring only the top features by SHAP can select anonymous V columns and
    leave amount unwatched, so an upstream feed switching to cents would look
    perfectly stable. The always-on list exists to close that hole.
    """
    rng = np.random.default_rng(5)
    train = pd.DataFrame(
        {
            "TransactionAmt": rng.lognormal(4.0, 1.0, 4000),
            "V1": rng.normal(0, 1, 4000),
        }
    )
    ranked = ["V1"]  # SHAP ranking on its own would stop here
    monitored = list(dict.fromkeys([*MONITOR_ALWAYS, *ranked]))
    reference = drift.build_reference(train, monitored)
    assert "TransactionAmt" in reference

    live = train.copy()
    live["TransactionAmt"] *= 100  # dollars reported as cents
    report = drift.compute_drift(live, reference)

    assert report["features"]["TransactionAmt"]["severity"] == "major"
    assert report["alert"]
    assert "TransactionAmt" in report["alert_reason"]


def test_one_major_shift_alerts_even_when_most_features_are_stable():
    """Breadth-based alerting would stay silent on a single catastrophic break."""
    rng = np.random.default_rng(6)
    columns = {f"v{i}": rng.normal(0, 1, 3000) for i in range(20)}
    train = pd.DataFrame(columns)
    reference = drift.build_reference(train, list(columns))

    live = train.copy()
    live["v0"] = live["v0"] * 50 + 500

    report = drift.compute_drift(live, reference)
    assert report["share_drifted"] <= 0.2
    assert report["major_drift"] == ["v0"]
    assert report["alert"]
