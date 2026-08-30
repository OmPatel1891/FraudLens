# Deploying FraudLens

From a working tree to a live, monitored API. Each phase is independently
useful; stop wherever your needs end.

---

## Phase 1 — Version control

```bash
git add -A
git commit -m "FraudLens: calibrated fraud scoring with SHAP and PSI drift monitoring"
git remote add origin <your-remote-url>
git push -u origin main
```

Two things worth checking before the first push. `git status --short` should
show no `data/`, `models/*`, `mlruns/` or `.env` — those are gitignored, and
`models/.gitkeep` is the single deliberate exception so the Dockerfile's
`COPY models/` works on a clean checkout. And the notebook should be committed
with its outputs cleared; a notebook carrying execution output turns every diff
into an unreviewable blob.

---

## Phase 2 — Continuous integration

`.github/workflows/ci.yml` runs three jobs on every push and pull request.

**Lint and unit tests** runs `ruff` and `pytest`. The API tests skip themselves
when `models/` is empty, so this job needs no trained model and finishes in
about a minute.

**End-to-end pipeline** generates synthetic data, trains, then re-runs the full
suite against the real exported artifacts and smoke-tests the served model.
This is the job that catches training/serving skew, which unit tests structurally
cannot: skew only appears once training has produced artifacts and something
tries to score with them.

**Docker** builds the image and curls `/ready`, `/health` and `/predict` against
the running container, so a green build means the container actually serves
rather than merely compiles.

---

## Phase 3 — How the model reaches production

The model is **committed to git and baked into the image**. The artifacts total
~3.6 MB, so `COPY models/ ./models/` puts the exact tested model inside the
image. The commit identifies the model version, rollback is `docker run` on the
previous tag, and there is no object-store dependency during a cold start —
which matters when you sit in the authorization path.

Committing binaries is a deliberate trade. At this size the history cost is
negligible, and it buys deterministic deploys: a platform builds the model you
tested rather than retraining on whatever hardware it happens to allocate.
Retrain with `python scripts/train.py`, then commit the changed artifacts.

Build a release from a model you trained:

```bash
make release              # trains, then builds fraudlens:<timestamp> and :latest
# or explicitly:
python scripts/train.py
docker build -t fraudlens:2026-08-29 .
```

If the build context has no trained model — a clean CI checkout, or a fresh
clone before the first train — the Dockerfile installs the training stack and
trains on synthetic data during the build, so the image stays self-contained
instead of answering 503 to every request. The guard is `TRAIN_IN_BUILD=true`
**and** no `models/model.joblib` present, so it never overwrites a real model,
and with artifacts committed the training dependencies are never installed at
all. Pass `--build-arg TRAIN_IN_BUILD=false` for a deliberately model-less image
to mount into at runtime.

A model trained on synthetic data is fine for a demo and dishonest as a
production claim. Ship a real one before anyone depends on the scores.

---

## Phase 4 — Hardening before exposure

Already in place:

| Concern | Where |
|---|---|
| Readiness distinct from liveness | `GET /ready` returns 503 until artifacts load |
| Model replacement is authenticated | `POST /reload` requires `X-API-Key` |
| Container runs unprivileged | `USER fraudlens` (uid 1000) |
| Batch size is bounded | `MAX_BATCH_ROWS` |
| Drift buffer cannot grow without limit | `DRIFT_WINDOW`, a bounded deque |

Set these before exposing the service:

```bash
FRAUDLENS_ADMIN_KEY=<random-32-bytes>     # unset disables /reload entirely
FRAUDLENS_CORS_ORIGINS=https://your-ui    # defaults to *
DRIFT_WINDOW=5000
MAX_BATCH_ROWS=50000
```

`FRAUDLENS_ADMIN_KEY` unset makes `/reload` return 404 rather than leaving it
open — an operator who never configured a key did not intend to expose model
replacement to anyone who can reach the port.

Still missing, and worth adding before real traffic: authentication on
`/predict` and `/batch` (`/reload` is the only gated endpoint), rate limiting,
and TLS. Do all three at a gateway or reverse proxy rather than in the app.

---

## Phase 5 — Verify the container locally

```bash
docker compose build
docker compose up -d
docker compose ps           # api should reach "healthy"
```

Verified on Docker 29.7.2. The image is a two-stage build: a builder installs
the full requirements and produces `models/`, then the runtime stage installs
only `requirements-api.txt` and copies the artifacts across. That split, plus
dropping the CUDA collective libraries xgboost pulls in for multi-GPU training,
takes the image from 3.73 GB to **1.44 GB**. A clean-checkout build takes about
five minutes, most of it dependency installation with roughly 70 seconds of
training; a release build with `models/` already populated skips training and
finishes in about 20 seconds. The container becomes ready in under 10 seconds
and holds ~169 MiB resident while serving.

Then run the same probes I ran against the local process:

```bash
curl -fsS localhost:8000/ready
curl -fsS localhost:8000/metrics
curl -fsS -X POST localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"TransactionID":1,"TransactionDT":2592000,"TransactionAmt":2500.0,
       "ProductCD":"C","card1":9999,"P_emaildomain":"protonmail.com"}'
curl -fsS localhost:8000/drift
```

Expect a fraud probability with populated `top_shap_drivers`, and `/drift`
reporting `insufficient_data` until the window fills.

The stack also brings up MLflow on :5000 and the Streamlit dashboard on :8501.

---

## Phase 6 — Deploy to a free platform

### Render free tier (recommended)

Render's free compute plan is 0.1 CPU and 512 MB RAM, needs no credit card, and
supports the Docker runtime. Measured steady-state memory is ~169 MiB, so the
cap is comfortable. Two limits to accept: the service spins down after 15
minutes idle and takes about a minute to wake, and 0.1 CPU makes scoring
noticeably slower than the ~50 ms measured locally.

Because the trained artifacts are committed, the build installs serving
dependencies and copies files — it does not train. That matters here: training
is the memory- and CPU-hungry step and the one most likely to be killed on a
free builder.

1. Push this repo to GitHub (Render deploys from a git remote).
2. At <https://dashboard.render.com>, choose **New → Web Service** and connect
   the repository.
3. Set **Language** to `Docker`, **Branch** to `main`, and **Instance Type** to
   **Free**.
4. Under **Advanced**, set **Health Check Path** to `/ready`, and add the
   environment variables below.
5. Click **Create Web Service**.

```
DRIFT_WINDOW=1000
MAX_BATCH_ROWS=5000
FRAUDLENS_ADMIN_KEY=<random-32-bytes>    # optional; omit to leave /reload off
```

Do not set `PORT` — Render injects it and the Dockerfile's `CMD` reads it.

`deploy/render.yaml` holds the same configuration as a Blueprint if you would
rather commit the infrastructure than click through the form.

The service goes live at `https://<name>.onrender.com`, with interactive docs at
`/docs`. Watch the first deploy from the **Logs** tab.

### Hugging Face Spaces (requires PRO)

Since July 2026, Gradio and Docker Spaces run on compute that requires a paid
plan — PRO for personal accounts, Team or Enterprise for organizations. Free
personal accounts get Static Spaces, plus up to two Gradio Spaces on ZeroGPU;
neither can host this container. If you do subscribe, deployment is a `git push`
to the Space remote once you add this YAML header to the top of `README.md`:

```yaml
---
title: FraudLens
sdk: docker
app_port: 8000
---
```

`app_port` matches the Dockerfile's default `PORT`, so nothing else needs
configuring.

### Anywhere else

The image is a plain uvicorn service reading `PORT`, listening on `0.0.0.0`,
running as a non-root user, with `/ready` for probes. That is all Cloud Run, Fly,
Railway, ECS or Kubernetes need. For Kubernetes, point `readinessProbe` at
`/ready` and `livenessProbe` at `/health` — using `/health` for readiness is the
specific mistake to avoid, since it answers 200 even when the model failed to
load.

---

## Phase 7 — Operating it

**Drift.** `GET /drift` computes PSI for recent traffic against the frozen
training reference. Alert on the `alert` flag. The monitored set is the model's
top SHAP features unioned with amount, its derivatives and hour — ranking by
SHAP alone leaves amount unwatched, and an upstream feed switching to cents is
exactly the failure that then reads as perfectly stable. A single feature at
major PSI raises the alert on its own rather than waiting for a share of the
population to move.

**The drift window is in-process.** It is a bounded deque that resets on restart
and is not shared across replicas, so with more than one replica each sees only
its own slice. Log predictions to durable storage and compute drift as a batch
job once you scale out.

**Also alert on flag rate.** A sudden jump in the share of transactions flagged
is far more often a broken upstream feed than a genuine fraud wave, and it shows
up sooner than a PSI shift.

**Retraining.** Fraud patterns move, so retrain on a schedule. Gate promotion on
the new model beating the incumbent on PR-AUC on a fresh time-sliced holdout,
never on a random split. Then `make release`, deploy the new tag, and keep the
previous tag available for rollback.

**Revisit the cost matrix.** The tuned threshold is only as good as
`COST_FALSE_NEGATIVE` and `COST_FALSE_POSITIVE` in `fraudlens/config.py`. The
defaults are illustrative. Replace them with your real chargeback cost and your
real cost of blocking a good customer, then retune.
