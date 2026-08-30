"""Break single-transaction scoring down by stage."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fraudlens.config import MODEL_DIR

PAYLOAD = {
    "TransactionAmt": 299.99, "ProductCD": "W", "card1": 9500.0, "card4": "visa",
    "card6": "credit", "P_emaildomain": "gmail.com", "TransactionDT": 86400.0,
}


def bench(label, fn, n=50):
    fn()  # warm up
    timings = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        timings.append((time.perf_counter() - t0) * 1000)
    timings.sort()
    print(f"  {label:<34} p50 {timings[n // 2]:7.2f} ms   p95 {timings[int(n * 0.95) - 1]:7.2f} ms")
    return timings[n // 2]


def main() -> None:
    engineer = joblib.load(MODEL_DIR / "feature_engineer.joblib")
    preprocessor = joblib.load(MODEL_DIR / "preprocessor.joblib")
    model = joblib.load(MODEL_DIR / "model.joblib")

    raw = pd.DataFrame([PAYLOAD])
    engineered = engineer.transform(raw)
    X = preprocessor.transform(engineered)

    print("Single-transaction scoring breakdown")
    bench("FeatureEngineer.transform", lambda: engineer.transform(raw))
    bench("Preprocessor.transform", lambda: preprocessor.transform(engineered))
    bench("model.predict_proba", lambda: model.predict_proba(X))
    def full_pipeline():
        return model.predict_proba(preprocessor.transform(engineer.transform(raw)))

    bench("full pipeline", full_pipeline)


if __name__ == "__main__":
    main()
