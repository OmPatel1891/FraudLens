FROM python:3.11-slim

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
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY fraudlens/ ./fraudlens/
COPY api/ ./api/
COPY scripts/ ./scripts/

# The model is baked in rather than mounted, so the image tag identifies the
# model version and a rollback is just the previous tag. models/ carries a
# .gitkeep so this COPY succeeds even on a clean checkout.
COPY models/ ./models/

# A clean checkout has no trained model, and an image that answers 503 to every
# request is worse than useless, so the fallback is to train one on synthetic
# data and stay self-contained. This is a no-op for a real release, where
# models/ already holds a trained model and the guard below skips it. Pass
# --build-arg TRAIN_IN_BUILD=false to force a deliberately model-less image.
ARG TRAIN_IN_BUILD=true
RUN if [ "$TRAIN_IN_BUILD" = "true" ] && [ ! -f models/model.joblib ]; then \
        echo "No model in build context; training on synthetic data" && \
        python scripts/make_synthetic_data.py --train-rows 30000 --test-rows 10000 && \
        python scripts/train.py --fast --shap-sample 800 && \
        rm -rf data plots reports mlruns ; \
    fi

RUN useradd --create-home --uid 1000 fraudlens && chown -R fraudlens:fraudlens /app
USER fraudlens

# Hugging Face Spaces routes to 7860; everything else here defaults to 8000.
ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
