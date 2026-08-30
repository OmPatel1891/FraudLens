"""FraudLens serving layer.

POST /predict  - score one transaction, with SHAP drivers
POST /batch    - score a CSV upload
GET  /health   - liveness and artifact status
GET  /metrics  - held-out model performance
GET  /drift    - PSI of recent live traffic against the training reference

Feature construction is delegated entirely to fraudlens.FeatureEngineer, the
same object fitted during training and loaded from disk here. Reimplementing
it in this file is what caused the model to be served median-filled constants
for most of its inputs.
"""

from __future__ import annotations

import io
import json
import logging
import os
import secrets
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from fraudlens import drift
from fraudlens.config import MODEL_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fraudlens.api")

MAX_BATCH_ROWS = int(os.getenv("MAX_BATCH_ROWS", "50000"))
DRIFT_WINDOW = int(os.getenv("DRIFT_WINDOW", "5000"))

# Swapping the model under a live service is an administrative action, so it is
# gated on a shared secret. Unset means the endpoint is disabled rather than
# open: an operator who never configured a key almost certainly did not intend
# to expose model replacement to anyone who can reach the port.
ADMIN_API_KEY = os.getenv("FRAUDLENS_ADMIN_KEY", "")

# Bounded so a long-running container cannot grow without limit.
_recent_features: deque = deque(maxlen=DRIFT_WINDOW)
_recent_lock = Lock()


class Artifacts:
    """Everything loaded from models/, kept together so /health can report gaps."""

    def __init__(self) -> None:
        self.ready = False
        self.error: Optional[str] = None
        self.model = None
        self.base_model = None
        self.engineer = None
        self.preprocessor = None
        self.meta: dict = {}
        self.drift_reference: dict = {}
        self.explainer = None
        self.threshold = 0.5

    def load(self, model_dir: Path = MODEL_DIR) -> None:
        try:
            self.model = joblib.load(model_dir / "model.joblib")
            self.engineer = joblib.load(model_dir / "feature_engineer.joblib")
            self.preprocessor = joblib.load(model_dir / "preprocessor.joblib")
            self.meta = json.loads((model_dir / "model_meta.json").read_text(encoding="utf-8"))
            self.threshold = float(self.meta.get("optimal_threshold", 0.5))

            base_path = model_dir / "base_model.joblib"
            self.base_model = joblib.load(base_path) if base_path.exists() else self.model

            ref_path = model_dir / "drift_reference.json"
            if ref_path.exists():
                self.drift_reference = drift.load_reference(ref_path)

            self._build_explainer()
            self.ready = True
            self.error = None
            logger.info(
                "loaded %s | test PR-AUC=%s | threshold=%.4f",
                self.meta.get("model_name"), self.meta.get("pr_auc"), self.threshold,
            )
        except FileNotFoundError as exc:
            # Start anyway so /health can explain the problem instead of the
            # container crash-looping on an import-time traceback.
            self.ready = False
            self.error = f"missing artifact: {exc}. Run `python scripts/train.py` first."
            logger.error(self.error)
        except Exception as exc:
            self.ready = False
            self.error = f"failed to load artifacts: {exc}"
            logger.exception(self.error)

    def _build_explainer(self) -> None:
        try:
            import shap

            self.explainer = shap.TreeExplainer(self.base_model)
        except Exception as exc:
            self.explainer = None
            logger.warning("SHAP explainer unavailable: %s", exc)


artifacts = Artifacts()


@asynccontextmanager
async def lifespan(_: FastAPI):
    artifacts.load()
    yield


app = FastAPI(
    title="FraudLens API",
    description="Real-time transaction fraud scoring with SHAP explanations and drift monitoring",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    # Tighten via FRAUDLENS_CORS_ORIGINS before exposing this beyond a VPC.
    allow_origins=os.getenv("FRAUDLENS_CORS_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def require_admin(x_api_key: str = Header(default="")) -> None:
    if not ADMIN_API_KEY:
        raise HTTPException(
            status_code=404,
            detail="Admin endpoints are disabled. Set FRAUDLENS_ADMIN_KEY to enable them.",
        )
    # compare_digest keeps the check constant-time so the key cannot be
    # recovered a byte at a time by timing repeated requests.
    if not secrets.compare_digest(x_api_key, ADMIN_API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


# ── schemas ───────────────────────────────────────────────────────────────────
class TransactionRequest(BaseModel):
    """One transaction. Only the amount is required; anything else may be sent."""

    TransactionAmt: float = Field(..., gt=0, description="Transaction amount in USD")
    TransactionDT: Optional[float] = Field(None, description="Seconds from the dataset epoch")
    ProductCD: Optional[str] = Field(None, description="Product code (W/H/C/S/R)")
    card1: Optional[float] = None
    card2: Optional[float] = None
    card3: Optional[float] = None
    card4: Optional[str] = None
    card5: Optional[float] = None
    card6: Optional[str] = None
    addr1: Optional[float] = None
    addr2: Optional[float] = None
    P_emaildomain: Optional[str] = None
    R_emaildomain: Optional[str] = None
    DeviceType: Optional[str] = None
    DeviceInfo: Optional[str] = None

    model_config = {"extra": "allow"}

    @field_validator("TransactionAmt")
    @classmethod
    def _finite_amount(cls, v: float) -> float:
        if not np.isfinite(v):
            raise ValueError("TransactionAmt must be finite")
        return v


class ShapDriver(BaseModel):
    feature: str
    shap_value: float
    feature_value: Optional[float]
    direction: str


class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    risk_level: str
    top_shap_drivers: list[ShapDriver]
    threshold_used: float
    model_name: str
    calibrated: bool
    latency_ms: float
    explanation_available: bool


class BatchResponse(BaseModel):
    total_transactions: int
    scored: int
    failed: int
    fraud_count: int
    fraud_rate: float
    threshold_used: float
    latency_ms: float
    results: list[dict[str, Any]]


# ── core scoring ──────────────────────────────────────────────────────────────
def _require_ready() -> None:
    if not artifacts.ready:
        raise HTTPException(status_code=503, detail=artifacts.error or "model not loaded")


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw request rows to model matrix, via the exact training transforms."""
    engineered = artifacts.engineer.transform(raw)
    return artifacts.preprocessor.transform(engineered)


def risk_level(prob: float) -> str:
    """Bands anchored to the tuned threshold, not to a fixed 0.5.

    Hardcoded cutoffs let a transaction be flagged `is_fraud=True` while being
    labelled LOW, because the tuned threshold is usually far below 0.5.
    """
    t = artifacts.threshold
    if prob >= min(1.0, t + (1.0 - t) * 0.5):
        return "CRITICAL"
    if prob >= t:
        return "HIGH"
    if prob >= t * 0.5:
        return "MEDIUM"
    return "LOW"


def shap_drivers(X: pd.DataFrame, n_top: int = 5) -> list[dict]:
    if artifacts.explainer is None:
        return []
    try:
        values = artifacts.explainer.shap_values(X)
        if isinstance(values, list):
            values = values[1]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, 1]
        row = values[0]

        order = np.argsort(np.abs(row))[::-1][:n_top]
        drivers = []
        for i in order:
            raw_value = X.iloc[0, i]
            drivers.append(
                {
                    "feature": str(X.columns[i]),
                    "shap_value": round(float(row[i]), 4),
                    "feature_value": (
                        round(float(raw_value), 4) if pd.notna(raw_value) else None
                    ),
                    "direction": (
                        "increases fraud risk" if row[i] > 0 else "decreases fraud risk"
                    ),
                }
            )
        return drivers
    except Exception:
        # Never fail a score because an explanation could not be produced, but
        # do surface it in the logs rather than swallowing it silently.
        logger.exception("SHAP explanation failed")
        return []


def record_for_drift(X: pd.DataFrame) -> None:
    monitored = artifacts.meta.get("monitored_features") or []
    cols = [c for c in monitored if c in X.columns]
    if not cols:
        return
    with _recent_lock:
        for record in X[cols].to_dict("records"):
            _recent_features.append(record)


# ── endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok" if artifacts.ready else "degraded",
        "model_loaded": artifacts.ready,
        "error": artifacts.error,
        "model": artifacts.meta.get("model_name"),
        "calibrated": artifacts.meta.get("calibrated", False),
        "threshold": artifacts.threshold,
        "features": artifacts.meta.get("feature_count"),
        "trained_on": artifacts.meta.get("train_size"),
        "explanations": artifacts.explainer is not None,
        "drift_buffer": len(_recent_features),
    }


@app.get("/ready")
def ready():
    """Readiness, kept separate from liveness.

    `/health` answers 200 even when artifacts are missing, so an operator can
    read the reason. A load balancer needs the opposite: a pod that cannot score
    must fail its probe and be pulled from rotation rather than be sent traffic
    it will only answer with 503.
    """
    if artifacts.ready:
        return {"status": "ready", "model": artifacts.meta.get("model_name")}
    return JSONResponse(
        status_code=503,
        content={"status": "not_ready", "error": artifacts.error},
    )


@app.get("/metrics")
def metrics():
    _require_ready()
    meta = artifacts.meta
    return {
        "model_name": meta.get("model_name"),
        "calibrated": meta.get("calibrated", False),
        "threshold_strategy": meta.get("threshold_strategy"),
        "optimal_threshold": meta.get("optimal_threshold"),
        # Reported on the sealed test slice, never on the tuning slice.
        "test": meta.get("test", {}),
        "validation": meta.get("validation", {}),
        "roc_auc": meta.get("roc_auc"),
        "pr_auc": meta.get("pr_auc"),
        "f1_at_threshold": meta.get("f1"),
        "brier": meta.get("brier"),
        "flag_rate": meta.get("flag_rate"),
        "train_size": meta.get("train_size"),
        "val_size": meta.get("val_size"),
        "test_size": meta.get("test_size"),
        "feature_count": meta.get("feature_count"),
        "top_10_features": meta.get("top_features", []),
        "all_model_results": meta.get("all_model_results", {}),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(
    transaction: TransactionRequest,
    explain: bool = Query(True, description="Compute SHAP drivers; disable for lowest latency"),
):
    _require_ready()
    started = time.perf_counter()

    try:
        X = build_features(pd.DataFrame([transaction.model_dump()]))
    except Exception as exc:
        logger.exception("preprocessing failed")
        raise HTTPException(status_code=422, detail=f"Preprocessing error: {exc}") from exc

    try:
        prob = float(artifacts.model.predict_proba(X)[0, 1])
    except Exception as exc:
        logger.exception("inference failed")
        raise HTTPException(status_code=500, detail=f"Model inference error: {exc}") from exc

    # SHAP dominates the request budget, so callers on the authorisation path
    # can turn it off and fetch an explanation asynchronously instead.
    drivers = shap_drivers(X) if explain else []
    record_for_drift(X)

    return PredictionResponse(
        fraud_probability=round(prob, 6),
        is_fraud=bool(prob >= artifacts.threshold),
        risk_level=risk_level(prob),
        top_shap_drivers=[ShapDriver(**d) for d in drivers],
        threshold_used=artifacts.threshold,
        model_name=artifacts.meta.get("model_name", "unknown"),
        calibrated=bool(artifacts.meta.get("calibrated", False)),
        latency_ms=round((time.perf_counter() - started) * 1000, 2),
        explanation_available=bool(drivers),
    )


@app.post("/batch", response_model=BatchResponse)
async def batch_predict(file: UploadFile = File(...)):
    _require_ready()
    started = time.perf_counter()

    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a CSV")

    try:
        raw = pd.read_csv(io.BytesIO(await file.read()))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"CSV parse error: {exc}") from exc

    if raw.empty:
        raise HTTPException(status_code=400, detail="CSV contains no rows")
    if len(raw) > MAX_BATCH_ROWS:
        raise HTTPException(
            status_code=413,
            detail=f"Batch of {len(raw):,} exceeds the {MAX_BATCH_ROWS:,} row limit",
        )
    if "TransactionAmt" not in raw.columns:
        raise HTTPException(status_code=422, detail="CSV must contain a TransactionAmt column")

    # Transform the whole frame at once; scoring row by row would re-run the
    # full pipeline per transaction.
    try:
        X = build_features(raw)
        probs = artifacts.model.predict_proba(X)[:, 1]
    except Exception as exc:
        logger.exception("batch scoring failed")
        raise HTTPException(status_code=422, detail=f"Batch scoring error: {exc}") from exc

    record_for_drift(X)

    ids = raw["TransactionID"] if "TransactionID" in raw.columns else pd.Series(range(len(raw)))
    flags = probs >= artifacts.threshold
    results = [
        {
            "transaction_id": (int(i) if pd.notna(i) and float(i).is_integer() else i),
            "fraud_probability": round(float(p), 6),
            "is_fraud": bool(f),
            "risk_level": risk_level(float(p)),
        }
        for i, p, f in zip(ids, probs, flags, strict=True)
    ]

    return BatchResponse(
        total_transactions=len(raw),
        scored=len(results),
        failed=0,
        fraud_count=int(flags.sum()),
        fraud_rate=round(float(flags.mean()), 4),
        threshold_used=artifacts.threshold,
        latency_ms=round((time.perf_counter() - started) * 1000, 2),
        results=results,
    )


@app.get("/drift")
def drift_report(min_rows: int = Query(50, ge=1, description="Rows needed before PSI is computed")):
    """PSI of recent live traffic against the frozen training reference."""
    _require_ready()
    if not artifacts.drift_reference:
        raise HTTPException(
            status_code=503,
            detail="No drift reference in models/. Retrain to generate drift_reference.json.",
        )

    with _recent_lock:
        window = pd.DataFrame(list(_recent_features))

    report = drift.compute_drift(window, artifacts.drift_reference, min_rows=min_rows)
    report["window_capacity"] = DRIFT_WINDOW
    return report


@app.post("/reload", dependencies=[Depends(require_admin)])
def reload_artifacts():
    """Pick up newly trained artifacts without restarting the process."""
    artifacts.load()
    if not artifacts.ready:
        raise HTTPException(status_code=503, detail=artifacts.error)
    return {"status": "reloaded", "model": artifacts.meta.get("model_name")}
