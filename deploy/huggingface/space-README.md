---
title: FraudLens
emoji: 🔍
colorFrom: indigo
colorTo: red
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# FraudLens

Real-time transaction fraud scoring with SHAP explanations and PSI drift
monitoring. Binary classification on IEEE-CIS style tabular data with ~3.5%
fraud, served behind FastAPI.

The model deployed here is trained at image build time on **synthetic data**
that reproduces the IEEE-CIS schema and class imbalance. It demonstrates the
full pipeline end to end; the scores are not comparable to a model trained on
the real Kaggle dataset.

## Try it

- `GET /docs` — interactive OpenAPI console
- `GET /health` — liveness plus which artifacts loaded
- `GET /ready` — readiness, 503 until the model is loaded
- `GET /metrics` — held-out test performance and the tuned threshold
- `POST /predict` — score one transaction, with SHAP drivers
- `POST /batch` — score a CSV upload
- `GET /drift` — PSI of recent traffic against the training reference

```bash
curl -X POST https://<your-space>.hf.space/predict \
  -H 'Content-Type: application/json' \
  -d '{"TransactionID":1,"TransactionDT":2592000,"TransactionAmt":2500.0,
       "ProductCD":"C","card1":9999,"P_emaildomain":"protonmail.com"}'
```

Source and full documentation: see the project repository.
