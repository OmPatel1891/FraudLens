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

The model is **baked into the image**. `COPY models/ ./models/` puts the trained
artifacts inside, so the image tag identifies the model version and a rollback is
`docker run` on the previous tag rather than a retrain. There is no object-store
dependency during a cold start, which matters when you sit in the authorization
path.

Build a release from a model you trained:

```bash
make release              # trains, then builds fraudlens:<timestamp> and :latest
# or explicitly:
python scripts/train.py
docker build -t fraudlens:2026-08-29 .
```

If the build context has no trained model — a clean CI checkout, or a platform
building straight from your git remote — the Dockerfile trains one on synthetic
data during the build. That keeps the image self-contained and demonstrable
instead of shipping something that answers 503 to every request. The guard is
`TRAIN_IN_BUILD=true` **and** no `models/model.joblib` present, so it never
overwrites a real model. Pass `--build-arg TRAIN_IN_BUILD=false` if you
deliberately want a model-less image to mount into at runtime.

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

### Hugging Face Spaces (recommended)

Free tier gives 2 vCPU and 16 GB RAM with native Docker support and no credit
card.

On memory: the serving process measures ~292 MB resident after 200 scored
requests, so it does fit the 512 MB free tiers elsewhere, though without much
headroom for a traffic spike. The reasons to prefer Spaces are that the build
itself trains a model — which is the memory- and CPU-hungry step, and the one
most likely to be killed on a constrained free builder — and that Spaces does
not spin down between requests.

The Space configuration lives in this repo's root `README.md` front-matter
(`sdk: docker`, `app_port: 8000`), so pushing the repo is the whole deployment.
`app_port` deliberately matches the Dockerfile's default `PORT`, which means no
Space variables are required to get it running.

1. Create a Space at <https://huggingface.co/new-space>: pick a name, choose
   **Docker** → **Blank**, and set it public.
2. Add the Space as a git remote and push. Authenticate with a **write** access
   token from <https://huggingface.co/settings/tokens> — use it as the password
   when git prompts, with your HF username as the username.

```bash
git remote add space https://huggingface.co/spaces/<user>/<space-name>
git push space main
```

3. Optional but recommended: in **Settings → Variables and secrets**, add
   `FRAUDLENS_ADMIN_KEY` as a secret to enable `/reload`. Left unset, that
   endpoint stays disabled, which is the safe default for a public Space.

The first build takes several minutes: it installs dependencies and trains the
demo model. Watch the build log. When it goes live the API is at
`https://<user>-<space-name>.hf.space`, with interactive docs at `/docs`.

If the push is rejected because the Space already has a commit, reconcile with
`git pull space main --allow-unrelated-histories` and keep this repo's
`README.md` so the front-matter survives.

The Dockerfile reads `PORT` at startup, so the same image also runs on Render
(10000) or Cloud Run (8080) with only that variable changed.

### Render (fallback)

`deploy/render.yaml` is a blueprint pointing at the same Dockerfile with
`healthCheckPath: /ready`. Runtime memory fits the free plan, but `DRIFT_WINDOW`
and `MAX_BATCH_ROWS` are still reduced there to keep headroom, and the risk to
watch is the build step training a model rather than the steady-state process.
Free services also spin down after inactivity, so the first request after idle
takes tens of seconds. Render blueprints cannot pass Docker build args, which is
the reason `TRAIN_IN_BUILD` defaults to true rather than being set per-platform.

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
