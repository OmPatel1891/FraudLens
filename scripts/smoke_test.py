"""Exercise the API in-process and print a readable summary.

Useful as a post-training sanity check: if every transaction comes back with
the same probability, features are not reaching the model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CASES = [
    {
        "label": "small daytime purchase",
        "payload": {
            "TransactionAmt": 12.50, "ProductCD": "W", "card1": 1200, "card4": "visa",
            "card6": "debit", "P_emaildomain": "gmail.com",
            "TransactionDT": 86400 * 3 + 3600 * 14,
        },
    },
    {
        "label": "large late-night anonymous purchase",
        "payload": {
            "TransactionAmt": 4800.00, "ProductCD": "C", "card1": 17999,
            "card4": "discover", "card6": "credit", "P_emaildomain": "anonymous.com",
            "R_emaildomain": "anonymous.com", "DeviceType": "mobile",
            "TransactionDT": 86400 * 6 + 3600 * 3,
        },
    },
    {"label": "amount only (minimal payload)", "payload": {"TransactionAmt": 299.99}},
    {
        "label": "unseen card and unseen network",
        "payload": {"TransactionAmt": 750.00, "card1": 999999, "card4": "brand-new-network"},
    },
]


def main() -> int:
    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as client:
        health = client.get("/health").json()
        print("HEALTH")
        print(json.dumps(health, indent=2))
        if not health.get("model_loaded"):
            print("\nModel not loaded; run python scripts/train.py first.")
            return 1

        print("\nPREDICTIONS")
        probabilities = []
        for case in CASES:
            response = client.post("/predict", json=case["payload"])
            if response.status_code != 200:
                print(f"  {case['label']}: HTTP {response.status_code} {response.text}")
                return 1

            body = response.json()
            probabilities.append(body["fraud_probability"])
            print(
                f"\n  {case['label']}"
                f"\n    probability {body['fraud_probability']:.4f}"
                f" | risk {body['risk_level']}"
                f" | flagged {body['is_fraud']}"
                f" | {body['latency_ms']:.1f} ms"
            )
            for driver in body["top_shap_drivers"][:3]:
                print(
                    f"      {driver['feature']:<28} "
                    f"shap {driver['shap_value']:+.4f}  {driver['direction']}"
                )

        spread = max(probabilities) - min(probabilities)
        print(f"\nProbability spread across cases: {spread:.4f}")
        if spread < 1e-6:
            print("FAIL: identical scores indicate features are being median-filled.")
            return 1

        # Separate the scoring cost from the explanation cost, so the latency
        # figure quoted for the authorisation path is the honest one.
        import time

        payload = CASES[0]["payload"]
        modes = (("with SHAP", "true"), ("scoring only", "false"))
        for label, explain in modes:
            timings = []
            for _ in range(20):
                t0 = time.perf_counter()
                client.post("/predict", json=payload, params={"explain": explain})
                timings.append((time.perf_counter() - t0) * 1000)
            timings.sort()
            p50 = timings[len(timings) // 2]
            p95 = timings[int(len(timings) * 0.95) - 1]
            print(f"\nLATENCY {label}: p50 {p50:.1f} ms | p95 {p95:.1f} ms")

        metrics = client.get("/metrics").json()
        print(
            f"\nMETRICS (sealed test slice)"
            f"\n  model      {metrics['model_name']} (calibrated={metrics['calibrated']})"
            f"\n  ROC-AUC    {metrics['roc_auc']}"
            f"\n  PR-AUC     {metrics['pr_auc']}"
            f"\n  Brier      {metrics['brier']}"
            f"\n  threshold  {metrics['optimal_threshold']} ({metrics['threshold_strategy']})"
            f"\n  flag rate  {metrics['flag_rate']}"
        )

        print("\nAll smoke checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
