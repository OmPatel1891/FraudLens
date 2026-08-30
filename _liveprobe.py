"""Probe the deployed Render service the same way the local container was probed."""

import io
import time

import httpx
import pandas as pd

BASE = "https://fraudlens-1wjy.onrender.com"
tx = {
    "TransactionID": 1,
    "TransactionDT": 2592000,
    "TransactionAmt": 2500.0,
    "ProductCD": "C",
    "card1": 9999,
    "P_emaildomain": "protonmail.com",
    "DeviceType": "mobile",
}

with httpx.Client(base_url=BASE, timeout=180, follow_redirects=True) as c:
    t = time.perf_counter()
    r = c.get("/ready")
    print(f"/ready   -> {r.status_code} {r.json()}  ({(time.perf_counter()-t)*1000:.0f} ms)")

    h = c.get("/health").json()
    print(f"/health  -> {h['status']} | {h['model']} | {h['features']} features "
          f"| explanations {h['explanations']}")

    m = c.get("/metrics").json()
    print(f"/metrics -> PR-AUC {m['pr_auc']:.4f} | ROC-AUC {m['roc_auc']:.4f} "
          f"| thr {m['optimal_threshold']:.3f} | calibrated {m['calibrated']}")

    r = c.post("/predict", json=tx).json()
    print(f"/predict -> {r['fraud_probability']:.4f} {r['risk_level']} "
          f"| server {r['latency_ms']:.0f} ms "
          f"| drivers {[d['feature'] for d in r['top_shap_drivers'][:3]]}")

    print(f"/reload  -> {c.post('/reload').status_code} (401 if admin key set)")
    print(f"bad amt  -> {c.post('/predict', json=dict(tx, TransactionAmt=-5)).status_code} "
          "(expect 422)")

    lat = []
    for _ in range(15):
        t = time.perf_counter()
        c.post("/predict?explain=false", json=tx)
        lat.append((time.perf_counter() - t) * 1000)
    lat.sort()
    print(f"latency  -> p50 {lat[7]:.0f} ms  max {lat[-1]:.0f} ms (0.1 CPU + internet)")

    df = pd.DataFrame([tx] * 100)
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    b = c.post("/batch", files={"file": ("t.csv", buf, "text/csv")}).json()
    print(f"/batch   -> {b['scored']} scored, {b['failed']} failed, {b['latency_ms']:.0f} ms")

    d = c.get("/drift").json()
    print(f"/drift   -> {{k: v for scalars}} "
          f"{ {k: v for k, v in d.items() if not isinstance(v, (dict, list))} }")

    print(f"/docs    -> {c.get('/docs').status_code}")
    print(f"/        -> {c.get('/').status_code} (404 = no root route)")
