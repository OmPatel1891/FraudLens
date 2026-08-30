# ── builder ───────────────────────────────────────────────────────────────────
# Exists only to produce models/. Everything it installs — MLflow, plotting, the
# notebook stack — is unreachable from api/main.py, so none of it belongs in the
# image that actually serves traffic.
FROM python:3.13-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    GIT_PYTHON_REFRESH=quiet

WORKDIR /app

COPY requirements.txt .
COPY fraudlens/ ./fraudlens/
COPY scripts/ ./scripts/
COPY models/ ./models/

# Trained artifacts are committed, so this stage is normally a pass-through and
# the whole training stack is never installed. The fallback exists for a context
# with no model — CI, or a fresh clone before the first train — where an image
# that answers 503 to every request would be worse than useless. Installing the
# training dependencies inside the guard keeps them off the critical path of an
# ordinary build, which matters on memory- and CPU-constrained free builders.
# Pass --build-arg TRAIN_IN_BUILD=false to force a deliberately model-less image.
ARG TRAIN_IN_BUILD=true
RUN if [ "$TRAIN_IN_BUILD" = "true" ] && [ ! -f models/model.joblib ]; then \
        echo "No model in build context; installing training deps and training" && \
        pip install --no-cache-dir -r requirements.txt && \
        python scripts/make_synthetic_data.py --train-rows 30000 --test-rows 10000 && \
        python scripts/train.py --fast --shap-sample 800 ; \
    else \
        echo "Model present in build context; skipping training" ; \
    fi


# ── runtime ───────────────────────────────────────────────────────────────────
FROM python:3.13-slim AS runtime

# libgomp1 is the OpenMP runtime LightGBM and XGBoost link against; curl backs
# the container healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Copied first so dependency layers survive source-only changes.
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# xgboost declares an nvidia-nccl wheel on linux/x86_64, ~450 MB of CUDA
# collective communication libraries used only for multi-GPU distributed
# training. This image scores on CPU, so it is dead weight; xgboost trains,
# predicts and unpickles without it. The suffix tracks the CUDA major (cu12,
# cu13, ...), so match on the prefix rather than pinning one name.
#
# Kept as its own step: chaining it onto the install with `|| true` would also
# swallow a failed install and ship a broken image.
RUN pip freeze | grep -i '^nvidia-' | cut -d= -f1 | xargs -r pip uninstall -y

COPY fraudlens/ ./fraudlens/
COPY api/ ./api/

# The model is baked in rather than mounted, so the image tag identifies the
# model version and a rollback is just the previous tag.
COPY --from=builder /app/models/ ./models/

RUN useradd --create-home --uid 1000 fraudlens && chown -R fraudlens:fraudlens /app
USER fraudlens

# Read at startup so one image serves Hugging Face Spaces, Render and localhost
# without an edit.
ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/ready" || exit 1

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
