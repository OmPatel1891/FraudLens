"""API contract tests against the real exported artifacts.

Skipped when models/ is empty, so a fresh clone does not fail before training.
"""

from __future__ import annotations

import io
import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fraudlens.config import MODEL_DIR

REQUIRED = ["model.joblib", "feature_engineer.joblib", "preprocessor.joblib", "model_meta.json"]

pytestmark = pytest.mark.skipif(
    not all((MODEL_DIR / f).exists() for f in REQUIRED),
    reason="no trained artifacts; run scripts/train.py first",
)


@pytest.fixture(scope="module")
def client():
    from api.main import app

    with TestClient(app) as c:
        yield c


def test_health_reports_a_loaded_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["error"] is None


def test_ready_is_200_when_the_model_is_loaded(client):
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_root_describes_the_service(client):
    body = client.get("/").json()
    assert body["service"] == "FraudLens"
    assert "/predict" in " ".join(body["endpoints"])


def test_shipped_artifacts_monitor_transaction_amount():
    """Guards against shipping artifacts trained before the drift fix.

    The committed model is what deploys, so a stale drift_reference here means
    a live service blind to a currency or scaling break in the amount field.
    """
    meta = json.loads((MODEL_DIR / "model_meta.json").read_text(encoding="utf-8"))
    monitored = meta.get("monitored_features", [])
    assert any("TransactionAmt" in f for f in monitored), (
        f"amount is unmonitored in the shipped artifacts: {monitored}. "
        "Retrain with scripts/train.py and commit the result."
    )


def test_serving_stack_matches_the_training_stack(client):
    """A mismatch means estimators are being rebuilt from different library code."""
    mismatches = client.get("/health").json()["version_mismatches"]
    assert mismatches == [], f"pickle-bearing library drift: {mismatches}"


def test_ready_fails_the_probe_when_artifacts_are_missing(monkeypatch):
    """A pod that cannot score must be pulled from rotation, not sent traffic.

    `/health` deliberately stays 200 so an operator can read the reason, which
    is why readiness needs its own endpoint that actually fails.
    """
    from api import main

    monkeypatch.setattr(main.artifacts, "ready", False)
    monkeypatch.setattr(main.artifacts, "error", "missing artifact")

    with TestClient(main.app) as unready:
        # TestClient's lifespan reloads real artifacts, so re-apply the failure.
        monkeypatch.setattr(main.artifacts, "ready", False)
        monkeypatch.setattr(main.artifacts, "error", "missing artifact")
        response = unready.get("/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"
    assert main.artifacts.error == "missing artifact"


def test_reload_is_disabled_without_an_admin_key(client, monkeypatch):
    """An unset key must close the endpoint, not leave it open."""
    monkeypatch.setattr("api.main.ADMIN_API_KEY", "")
    assert client.post("/reload").status_code == 404


def test_reload_rejects_a_wrong_key(client, monkeypatch):
    monkeypatch.setattr("api.main.ADMIN_API_KEY", "correct-horse")
    assert client.post("/reload").status_code == 401
    assert client.post("/reload", headers={"X-API-Key": "wrong"}).status_code == 401


def test_reload_accepts_the_configured_key(client, monkeypatch):
    monkeypatch.setattr("api.main.ADMIN_API_KEY", "correct-horse")
    response = client.post("/reload", headers={"X-API-Key": "correct-horse"})
    assert response.status_code == 200
    assert response.json()["status"] == "reloaded"


def test_metrics_reports_sealed_test_performance(client):
    body = client.get("/metrics").json()
    assert body["calibrated"] is True
    assert 0.0 <= body["roc_auc"] <= 1.0
    # Headline numbers must come from the slice never used for tuning.
    assert body["test"]["roc_auc"] == body["roc_auc"]


def test_predict_minimal_payload(client):
    """An amount-only payload must score; it is the realistic minimum request."""
    response = client.post("/predict", json={"TransactionAmt": 299.99})
    assert response.status_code == 200, response.text

    body = response.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["is_fraud"] == (body["fraud_probability"] >= body["threshold_used"])


def test_predict_full_payload_with_explanations(client):
    payload = {
        "TransactionAmt": 1499.00,
        "TransactionDT": 86400 * 3 + 3600 * 23,
        "ProductCD": "W",
        "card1": 9500,
        "card2": 360.0,
        "card4": "visa",
        "card6": "credit",
        "addr1": 325.0,
        "addr2": 87.0,
        "P_emaildomain": "gmail.com",
        "R_emaildomain": "anonymous.com",
        "DeviceType": "mobile",
    }
    body = client.post("/predict", json=payload).json()

    assert body["explanation_available"] is True
    assert 1 <= len(body["top_shap_drivers"]) <= 5
    driver = body["top_shap_drivers"][0]
    assert driver["direction"] in {"increases fraud risk", "decreases fraud risk"}


def test_risk_level_is_consistent_with_the_flag(client):
    """A flagged transaction must never be labelled LOW."""
    for amount in (5.0, 250.0, 900.0, 5000.0):
        body = client.post("/predict", json={"TransactionAmt": amount}).json()
        if body["is_fraud"]:
            assert body["risk_level"] in {"HIGH", "CRITICAL"}
        else:
            assert body["risk_level"] in {"LOW", "MEDIUM"}


def test_different_inputs_produce_different_scores(client):
    """Guards the skew bug: constant output means features are not reaching the model."""
    payloads = [
        {"TransactionAmt": 5.0, "ProductCD": "W", "card1": 1001},
        {"TransactionAmt": 4800.0, "ProductCD": "C", "card1": 17999,
         "P_emaildomain": "anonymous.com", "TransactionDT": 86400 * 2 + 3600 * 2},
        {"TransactionAmt": 120.0, "ProductCD": "H", "card1": 8000, "DeviceType": "mobile"},
    ]
    scores = {client.post("/predict", json=p).json()["fraud_probability"] for p in payloads}
    assert len(scores) > 1, "identical scores suggest features are being median-filled"


def test_predict_rejects_invalid_amount(client):
    assert client.post("/predict", json={"TransactionAmt": -10}).status_code == 422
    assert client.post("/predict", json={}).status_code == 422


def test_batch_scoring(client):
    frame = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "TransactionAmt": [10.0, 500.0, 3000.0],
            "ProductCD": ["W", "C", "W"],
            "card1": [1200, 9000, 15000],
        }
    )
    buffer = io.BytesIO(frame.to_csv(index=False).encode())
    response = client.post("/batch", files={"file": ("txns.csv", buffer, "text/csv")})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total_transactions"] == 3
    assert len(body["results"]) == 3
    assert body["results"][0]["transaction_id"] == 1


def test_batch_rejects_non_csv(client):
    buffer = io.BytesIO(b"not a csv")
    assert client.post("/batch", files={"file": ("x.txt", buffer, "text/plain")}).status_code == 400


def test_batch_requires_amount_column(client):
    buffer = io.BytesIO(b"TransactionID\n1\n")
    response = client.post("/batch", files={"file": ("x.csv", buffer, "text/csv")})
    assert response.status_code == 422


def test_drift_endpoint(client):
    """Below the row floor it must say so rather than report a bogus PSI."""
    body = client.get("/drift", params={"min_rows": 1000000}).json()
    assert body["status"] == "insufficient_data"

    frame = pd.DataFrame({"TransactionAmt": [50.0 + i for i in range(120)]})
    buffer = io.BytesIO(frame.to_csv(index=False).encode())
    client.post("/batch", files={"file": ("t.csv", buffer, "text/csv")})

    body = client.get("/drift", params={"min_rows": 50}).json()
    assert body["status"] == "ok"
    assert body["features_monitored"] > 0
    assert 0.0 <= body["share_drifted"] <= 1.0
