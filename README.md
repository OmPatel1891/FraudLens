<h1 align="center">FraudLens</h1>

<p align="center">
  Real-time transaction fraud detection on the IEEE-CIS dataset: a calibrated
  gradient-boosted model served behind a FastAPI endpoint that returns a fraud
  probability, the SHAP drivers behind it, and live drift monitoring.
</p>

<p align="center">
  <a href="https://fraudlens-1wjy.onrender.com/docs"><b>Live API</b></a> ·
  <a href="https://fraudlens-1wjy.onrender.com/redoc">API reference</a> ·
  <a href="FraudLens.ipynb">Notebook walkthrough</a> ·
  <a href="DEPLOYMENT.md">Deployment runbook</a>
</p>

<p align="center">
  <a href="https://github.com/OmPatel1891/FraudLens/actions/workflows/ci.yml">
    <img alt="CI" src="https://github.com/OmPatel1891/FraudLens/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.12%2B-blue">
  <img alt="Tests" src="https://img.shields.io/badge/tests-53%20passing-brightgreen">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
</p>

> The live demo runs on a free instance that sleeps after 15 minutes idle, so the
> first request may take 30–60 seconds to wake. Try
> [`/docs`](https://fraudlens-1wjy.onrender.com/docs) for an interactive form.

---

## The problem

Online payment fraud has to be caught before the payment clears, without
blocking so many legitimate customers that the business suffers. As an ML
problem that means binary classification on tabular data with:

- **Severe class imbalance** (~3.5% fraud), which makes accuracy meaningless
- **400+ mostly anonymous features** (`V1`–`V339`, `C1`–`C14`, `D1`–`D15`)
- **Two joinable sources** — transactions and identity, on `TransactionID`
- **Temporal structure**, which makes a random train/test split leak the future

---

## Quick start

Requires **Python 3.12+** (`xgboost` 3.4 sets that floor). No Kaggle account
needed — a synthetic generator reproduces the dataset's shape, sparsity,
temporal axis and fraud rate.

```bash
pip install -r requirements.txt

python scripts/make_synthetic_data.py   # or: python scripts/download_data.py
python scripts/train.py
python scripts/smoke_test.py
```

Then serve it:

```bash
python -m uvicorn api.main:app --reload --port 8000   # http://localhost:8000/docs
python -m streamlit run dashboard.py                  # http://localhost:8501
```

Or bring up the whole stack (API, MLflow, dashboard):

```bash
docker compose up -d
```

`make help` lists every target.

---

## Scoring a transaction

Only the amount is required; anything else you can supply sharpens the estimate.

```bash
curl -X POST https://fraudlens-1wjy.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"TransactionAmt": 2500.00, "ProductCD": "C", "card1": 9999,
       "P_emaildomain": "protonmail.com", "TransactionDT": 2592000,
       "DeviceType": "mobile"}'
```

```json
{
  "fraud_probability": 0.3778,
  "is_fraud": true,
  "risk_level": "HIGH",
  "top_shap_drivers": [
    {"feature": "is_night", "shap_value": 0.3093, "feature_value": 1.0,
     "direction": "increases fraud risk"},
    {"feature": "C_sum", "shap_value": 0.2668, "feature_value": 24.0,
     "direction": "increases fraud risk"}
  ],
  "threshold_used": 0.1163,
  "calibrated": true,
  "latency_ms": 31.4
}
```

Risk bands are anchored to the tuned threshold rather than a fixed 0.5, so a
transaction can never be flagged `is_fraud=true` while being labelled `LOW`.

| Endpoint | Purpose |
|---|---|
| `GET /` | Service card and endpoint index |
| `POST /predict` | Score one transaction. `?explain=false` skips SHAP for lower latency |
| `POST /batch` | Score a CSV upload, vectorised |
| `GET /health` | Liveness, loaded artifacts, and any training/serving version skew. Always 200, so failures are readable |
| `GET /ready` | Readiness. 503 until the model loads, for load-balancer probes |
| `GET /metrics` | Held-out test performance |
| `GET /drift` | PSI of recent live traffic against the training reference |
| `POST /reload` | Reload artifacts without a restart. Requires `X-API-Key` |

---

## Architecture

```
fraudlens/              shared library - imported by BOTH training and serving
  config.py             paths, cost matrix, split boundaries
  data.py               table join + the id-/id_ column fix
  features.py           FeatureEngineer: fit learns lookup tables, transform applies them
  preprocessing.py      Preprocessor: frozen column contract, encoding, imputation
  modeling.py           calibration, evaluation, cost-based threshold selection
  drift.py              PSI reference + scoring
  versions.py           records the stack a model was pickled with

api/main.py             FastAPI service
dashboard.py            Streamlit UI
scripts/                data generation, download, training, smoke test, profiling
tests/                  53 tests, including batch-vs-row feature parity
deploy/render.yaml      Render blueprint
.github/workflows/      CI: lint, unit tests, full pipeline, Docker build-and-serve
FraudLens.ipynb         annotated walkthrough of the same pipeline
DEPLOYMENT.md           git to production runbook
```

The single most important design decision is that `api/main.py` contains **no
feature engineering**. It loads the fitted `FeatureEngineer` and `Preprocessor`
from disk and calls them. Reimplementing those transforms in the serving layer
is the classic way to ship a model that silently scores garbage.

---

## Design decisions

### Split before you engineer

Aggregate features like "how many transactions has this card made" are computed
with a `value_counts()` over whatever frame you hand them. Compute them before
splitting and training rows absorb statistics from the validation period.

Worse, a single incoming API request has no frame to count over, so those
features cannot be recomputed at serving time at all. `FeatureEngineer.fit`
learns the lookup tables from training rows only and ships them with the model,
which fixes both problems at once. A card never seen in training gets frequency
0 — itself a useful signal — rather than a missing value.

The `tests/test_features.py` parity suite asserts that a row scored alone is
byte-identical to that row scored inside a batch, in any order, in any split.

### A three-way temporal split

`train` (0–70%) → `val` (70–85%) → `test` (85–100%) by `TransactionDT`.

Two slices are not enough. Early stopping, model selection and threshold tuning
all consume `val`, so reporting on `val` would be reporting on data the model
was tuned against. Every headline number in this repo comes from `test`, which
is touched exactly once.

### Selection on PR-AUC, not ROC-AUC

At a 3.5% positive rate ROC-AUC is dominated by the majority class. On
validation, all three candidates look nearly identical by ROC-AUC while PR-AUC
separates them cleanly:

| Model | ROC-AUC | PR-AUC | Brier | Train time |
|---|---|---|---|---|
| Logistic Regression | 0.9376 | 0.3445 | 0.0967 | 0.2 s |
| XGBoost | 0.9431 | 0.4121 | 0.0327 | 6.2 s |
| **LightGBM** (selected) | **0.9468** | **0.4494** | **0.0225** | 5.5 s |

Picking on ROC-AUC would have made this a coin toss between three models whose
precision-recall behaviour differs by 30%.

### Calibration

Training with `scale_pos_weight ≈ 27` deliberately distorts the output scale, so
raw `predict_proba` values are not probabilities and must not be shown to a risk
team as such. An isotonic fit on `val` corrects the scale. Being monotonic, it
leaves ROC-AUC and PR-AUC untouched and only moves the Brier score.

### A cost-based threshold, not F1

0.5 is arbitrary, but maximising F1 is arbitrary too: it weights precision and
recall equally, which encodes no business reality. Here a missed fraud is
charged at the disputed transaction amount and a false positive at a flat rate
covering manual review plus the churn risk of blocking a real customer. Both
are set in `fraudlens/config.py`.

On the bundled synthetic data the two disagree, which is the whole point:

| Strategy | Threshold | Precision | Recall | Flag rate |
|---|---|---|---|---|
| Min expected cost | 0.116 | 0.349 | 0.669 | 5.7% |
| Max F1 | 0.197 | 0.443 | 0.526 | 3.5% |

`--threshold-strategy min_precision --min-precision 0.9` is also available for
teams that operate against a precision SLA instead.

### Drift monitoring in two layers

Evidently produces the rich offline HTML report for periodic review. The API
serves PSI from a small frozen reference at `GET /drift`, because Evidently is
too heavy for a request path and its report object does not serialise into an
API response. Both are exercised in section 10 of the notebook.

Two details decide whether this actually catches anything. The monitored set is
the model's top features by SHAP *unioned with* `MONITOR_ALWAYS` (amount, its
derivatives, and hour): ranking by SHAP alone tends to pick anonymous `V`
columns and leave amount unwatched, so an upstream feed switching to cents would
read as perfectly stable. And a single feature at major PSI raises the alert on
its own rather than requiring a share of the population to move, since that same
cents bug shifts three columns out of twenty-three and would sit under any
breadth-based threshold.

### Pinning the stack that touches the pickle

Estimators pickle by *class reference*, not by value, so loading a model under a
different library version rebuilds it from whatever the installed code now does.
scikit-learn emits `InconsistentVersionWarning` and carries on. For a fraud
model that is the worst available failure mode: predictions that are wrong but
plausible, returned as a `200`.

So the six libraries whose objects live inside the artifacts — `numpy`, `scipy`,
`scikit-learn`, `xgboost`, `lightgbm`, `joblib` — carry compatible-release pins
that are identical in `requirements.txt` and `requirements-api.txt`. Pins drift
and base images get rebuilt, so belt and braces: `scripts/train.py` records the
training versions in `model_meta.json`, and the API compares them on load,
logging any mismatch and exposing it at `GET /health`:

```json
{"status": "ok", "model": "LightGBM", "version_mismatches": []}
```

An empty list is the healthy case. `tests/test_api.py` asserts it stays empty,
which is what caught a real 1.8.0-vs-1.9.0 skew in the deployed container.

---

## Results on the bundled synthetic data

These are **not** IEEE-CIS numbers. The generator produces a learnable but much
simpler problem; run `scripts/download_data.py` for the real thing.

| Metric (sealed test slice) | Value |
|---|---|
| ROC-AUC | 0.9377 |
| PR-AUC | 0.5056 (logistic baseline 0.3445) |
| Brier | 0.0225 |
| Precision / Recall @ threshold | 0.391 / 0.680 |
| Precision @ 80% recall | 0.218 |
| Flag rate | 6.0% |

Single-transaction latency, measured by `scripts/profile_latency.py`:

| Stage | p50 |
|---|---|
| Feature engineering | 8.5 ms |
| Preprocessing | 8.0 ms |
| Model inference | 2.4 ms |
| **Full pipeline** | **31 ms** |

SHAP roughly doubles the request (measured 50–100 ms end to end with it, 30–50 ms
without, varying with machine load). Disable it with `?explain=false` on the
authorisation path and fetch explanations asynchronously.

Measured on Python 3.14 / Windows with the 361-feature synthetic model. The real
IEEE-CIS model has a similar feature count, so expect the same order of
magnitude. The free-tier demo adds network round-trip and runs on 0.1 vCPU, so
it serves the same model at roughly 200–300 ms p50.

---

## Tests

```bash
python -m pytest tests/ -v
```

53 tests. The ones worth knowing about:

- `test_batch_and_single_row_agree` — the training/serving parity guarantee
- `test_output_independent_of_batch_composition` / `_row_order`
- `test_sparse_request_produces_full_matrix` — a 5-field payload yields the full matrix
- `test_unknown_category_is_isolated` — unseen categories don't impersonate real ones
- `test_calibration_lowers_brier_without_hurting_ranking`
- `test_different_inputs_produce_different_scores` — catches silent skew regressions
- `test_amount_is_monitored_even_when_shap_ignores_it` — the drift blind spot above
- `test_shipped_artifacts_monitor_transaction_amount` — blocks shipping stale artifacts
- `test_serving_stack_matches_the_training_stack` — blocks silent pickle skew

CI runs lint, the unit suite, a full synthetic train-and-evaluate pipeline, and a
Docker build that boots the image and probes every endpoint.

---

## Known limitations

- **Aggregates are a snapshot, not a stream.** `card1_freq` reflects the
  training window. In production these belong in a feature store updated
  continuously; retraining is currently the only way to refresh them.
- **No true velocity features.** Time-windowed counts ("transactions on this
  card in the last hour") are the strongest fraud signal and are absent. They
  need per-entity state the current design has no place for.
- **SHAP explains the uncalibrated model**, since `TreeExplainer` needs the raw
  booster. Calibration is monotonic so the ranking and sign of each contribution
  carry over, but the magnitudes are on the log-odds scale.
- **The drift window is in-process and bounded** (`DRIFT_WINDOW`, default 5000).
  It resets on restart and is not shared across replicas; a real deployment
  would log predictions to a store and compute drift as a batch job.
- **Scoring endpoints are unauthenticated.** `/reload` is gated behind
  `FRAUDLENS_ADMIN_KEY`, but `/predict` and `/batch` are open and nothing is rate
  limited. Set `FRAUDLENS_CORS_ORIGINS` and put a gateway in front before
  exposing this beyond a VPC.
- **The cost matrix is illustrative.** Replace
  `COST_FALSE_NEGATIVE` / `COST_FALSE_POSITIVE` with your own numbers — the
  chosen threshold is only as good as those two constants.
- **Evidently does not import on Python 3.14** (its pydantic v1 shim raises
  `ConfigError`). Only the optional offline HTML report is affected; section 10
  of the notebook catches it and the API's own PSI monitoring has no Evidently
  dependency. Use Python 3.12 or 3.13 if you want the HTML report.

---

## Deploying

`make release` trains a model and builds an image with it baked in, so the image
tag identifies the model version and rollback is the previous tag. If the build
context has no trained model, the image trains one on synthetic data so it stays
self-contained rather than shipping something that answers 503 to everything.

The container runs Python 3.13 on `python:3.13-slim`, installs only
`requirements-api.txt`, and drops the ~450 MB CUDA collective-communication
wheel that `xgboost` pulls in but CPU inference never touches.

`DEPLOYMENT.md` is the full runbook: CI, hardening, container verification,
free-tier deployment on Render, and what to monitor afterwards.

---

## Dataset

[IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) —
590,540 e-commerce transactions from Vesta Corporation, hosted by the IEEE
Computational Intelligence Society.

One gotcha worth flagging: `test_identity.csv` names its columns `id-01`–`id-38`
with hyphens, while `train_identity.csv` uses underscores. Merging without
normalising them silently drops ~20 identity features from the test frame.
`fraudlens/data.py` handles it and raises if the schemas still disagree.

---

## License

[MIT](LICENSE).
