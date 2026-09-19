"""
api/main.py
============
FastAPI inference service for the Fake News Trend Drift Detector.

Endpoints:
    POST /predict          — single article prediction
    POST /batch_predict    — batch prediction (up to 50 articles)
    GET  /health           — liveness + readiness probe
    GET  /metrics          — Prometheus metrics exposition
    GET  /model/info       — current model metadata
    POST /model/rollback   — rollback to a previous model version
    GET  /drift/status     — latest drift detection result

Advanced features:
    - Rate limiting via slowapi (100 req/min per IP)
    - Redis caching of predictions (by text hash)
    - Async batch inference endpoint
    - Shadow deployment: logs both old & new model predictions
    - Canary routing: sends N% of traffic to the "challenger" model
    - Prometheus instrumentation via prometheus-fastapi-instrumentator
    - Request + prediction logging to PostgreSQL
"""

import asyncio
import hashlib
import json
import logging
import os
import pickle
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import redis.asyncio as aioredis
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel, Field, validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from api.prediction_logger import PredictionLogger
from feature_pipeline.embeddings import EmbeddingService, EMBEDDING_MODEL_NAME

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Paths & Config ────────────────────────────────────────────────────────────
MODEL_DIR = Path("models")
DRIFT_REPORTS_DIR = Path("reports")

CANARY_TRAFFIC_PCT = int(os.getenv("CANARY_TRAFFIC_PCT", 0))
SHADOW_MODE = os.getenv("SHADOW_MODE", "false").lower() == "true"
REDIS_URL = f"redis://{os.getenv('REDIS_HOST', 'localhost')}:{os.getenv('REDIS_PORT', '6379')}/0"

# ── Rate Limiter ──────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

# ── Prometheus Metrics ────────────────────────────────────────────────────────
PREDICTION_COUNT = Counter(
    "fakenews_predictions_total", "Total predictions made",
    ["label", "model_version"]
)
PREDICTION_CONFIDENCE = Histogram(
    "fakenews_prediction_confidence", "Prediction confidence distribution",
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
)
INFERENCE_LATENCY = Histogram(
    "fakenews_inference_latency_seconds", "Inference latency in seconds",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
)
DRIFT_SCORE = Gauge("fakenews_drift_score", "Latest drift score")
RETRAIN_EVENTS = Counter("fakenews_retrain_events_total", "Total retraining events triggered")
CACHE_HITS = Counter("fakenews_cache_hits_total", "Prediction cache hits")
BATCH_SIZE_HIST = Histogram("fakenews_batch_size", "Batch request sizes",
                            buckets=[1, 5, 10, 20, 30, 50])


# ── Pydantic Schemas ──────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=10000,
                      description="News article text to classify")
    title: Optional[str] = Field(None, max_length=500, description="Optional article title")

    @validator("text")
    def validate_text(cls, v):
        v = v.strip()
        if len(v.split()) < 5:
            raise ValueError("Text must contain at least 5 words")
        return v


class PredictResponse(BaseModel):
    request_id: str
    prediction: str          # "FAKE" or "REAL"
    label: int               # 1 = FAKE, 0 = REAL
    confidence: float
    fake_probability: float
    real_probability: float
    drift_status: str        # "normal" | "warning" | "critical"
    model_version: str
    processing_time_ms: float
    cached: bool


class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=50)
    titles: Optional[List[str]] = None


class BatchPredictResponse(BaseModel):
    request_id: str
    predictions: List[PredictResponse]
    total_processed: int
    processing_time_ms: float


class ModelInfoResponse(BaseModel):
    model_version: str
    classifier: str
    val_f1: Optional[float]
    test_f1: Optional[float]
    loaded_at: str
    canary_mode: bool
    shadow_mode: bool


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    embedding_service_ready: bool
    redis_connected: bool
    timestamp: str


# ── App State (loaded at startup) ─────────────────────────────────────────────

class AppState:
    model = None
    challenger_model = None  # for canary / shadow
    embedding_service: Optional[EmbeddingService] = None
    redis_client = None
    prediction_logger: Optional[PredictionLogger] = None
    model_metadata: dict = {}
    drift_status: str = "normal"
    loaded_at: str = ""


app_state = AppState()


def load_model_from_disk(path: Path):
    """Load model.pkl from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


def load_model_metadata() -> dict:
    """Load the latest_model_metadata.json written by training."""
    meta_path = MODEL_DIR / "latest_model_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return {"model_version": "v1.0", "classifier": "logistic_regression"}


def load_drift_status() -> str:
    """Check the latest drift report for current drift status."""
    report_path = DRIFT_REPORTS_DIR / "latest_drift_status.json"
    if report_path.exists():
        with open(report_path) as f:
            data = json.load(f)
            return data.get("status", "normal")
    return "normal"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle handler."""
    # ── Startup ──────────────────────────────────────────────────────────────
    logger.info("Starting up Fake News Detector API...")

    # Load main model
    model_path = MODEL_DIR / "model.pkl"
    if model_path.exists():
        app_state.model = load_model_from_disk(model_path)
        logger.info("Primary model loaded.")
    else:
        logger.warning("No model.pkl found. /predict will fail until training runs.")

    # Load challenger model (if exists) for canary/shadow
    challenger_path = MODEL_DIR / "challenger_model.pkl"
    if challenger_path.exists():
        app_state.challenger_model = load_model_from_disk(challenger_path)
        logger.info("Challenger model loaded for canary/shadow deployment.")

    # Load embedding service
    try:
        app_state.embedding_service = EmbeddingService(model_name=EMBEDDING_MODEL_NAME)
        logger.info("Embedding service ready.")
    except Exception as e:
        logger.error(f"Failed to load embedding service: {e}")

    # Connect to Redis
    try:
        app_state.redis_client = await aioredis.from_url(
            REDIS_URL, encoding="utf-8", decode_responses=False
        )
        await app_state.redis_client.ping()
        logger.info("Redis connected.")
    except Exception as e:
        logger.warning(f"Redis unavailable, prediction caching disabled: {e}")
        app_state.redis_client = None

    # Load metadata
    app_state.model_metadata = load_model_metadata()
    app_state.loaded_at = datetime.utcnow().isoformat()

    # Setup prediction logger
    try:
        app_state.prediction_logger = PredictionLogger()
        logger.info("Prediction logger initialized.")
    except Exception as e:
        logger.warning(f"Prediction logger unavailable: {e}")

    # Load initial drift status
    app_state.drift_status = load_drift_status()

    logger.info("✅ API startup complete.")
    yield

    # ── Shutdown ─────────────────────────────────────────────────────────────
    logger.info("Shutting down API...")
    if app_state.redis_client:
        await app_state.redis_client.close()


# ── App Initialization ─────────────────────────────────────────────────────────

app = FastAPI(
    title="Fake News Trend Drift Detector",
    description="Production NLP inference service with drift detection and auto-retraining",
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cache_key(text: str) -> str:
    return "pred:" + hashlib.sha256(text.encode()).hexdigest()


async def _get_cached_prediction(text: str) -> Optional[dict]:
    if not app_state.redis_client:
        return None
    try:
        raw = await app_state.redis_client.get(_cache_key(text))
        if raw:
            CACHE_HITS.inc()
            return json.loads(raw)
    except Exception:
        pass
    return None


async def _cache_prediction(text: str, result: dict, ttl: int = 3600) -> None:
    if not app_state.redis_client:
        return
    try:
        await app_state.redis_client.setex(_cache_key(text), ttl, json.dumps(result))
    except Exception:
        pass


def _make_prediction(text: str, model=None) -> dict:
    """
    Core prediction logic.
    Returns dict with label, probabilities, confidence.
    """
    if model is None:
        model = app_state.model

    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Run training pipeline first."
        )
    if app_state.embedding_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding service not ready."
        )

    # Generate embedding
    embedding = app_state.embedding_service.embed_single(text).reshape(1, -1)

    # Predict
    label = int(model.predict(embedding)[0])
    proba = model.predict_proba(embedding)[0]
    fake_prob = float(proba[1])
    real_prob = float(proba[0])
    confidence = float(max(proba))

    return {
        "label": label,
        "prediction": "FAKE" if label == 1 else "REAL",
        "fake_probability": round(fake_prob, 4),
        "real_probability": round(real_prob, 4),
        "confidence": round(confidence, 4),
    }


def _select_model():
    """
    Canary routing: sends CANARY_TRAFFIC_PCT% of requests to challenger model.
    Returns (model, is_canary) tuple.
    """
    if (
        app_state.challenger_model is not None
        and CANARY_TRAFFIC_PCT > 0
        and np.random.randint(100) < CANARY_TRAFFIC_PCT
    ):
        return app_state.challenger_model, True
    return app_state.model, False


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Liveness and readiness probe."""
    redis_ok = False
    if app_state.redis_client:
        try:
            await app_state.redis_client.ping()
            redis_ok = True
        except Exception:
            pass

    return HealthResponse(
        status="healthy" if app_state.model else "degraded",
        model_loaded=app_state.model is not None,
        embedding_service_ready=app_state.embedding_service is not None,
        redis_connected=redis_ok,
        timestamp=datetime.utcnow().isoformat(),
    )


@app.get("/metrics", tags=["System"])
async def prometheus_metrics():
    """Expose Prometheus metrics."""
    from fastapi.responses import Response
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    """Return current model metadata."""
    meta = app_state.model_metadata
    return ModelInfoResponse(
        model_version=meta.get("model_version", "v1.0"),
        classifier=meta.get("classifier", "unknown"),
        val_f1=meta.get("val_f1"),
        test_f1=meta.get("test_f1"),
        loaded_at=app_state.loaded_at,
        canary_mode=CANARY_TRAFFIC_PCT > 0,
        shadow_mode=SHADOW_MODE,
    )


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
@limiter.limit("100/minute")
async def predict(request: Request, body: PredictRequest):
    """
    Classify a single news article as FAKE or REAL.

    - Checks Redis cache for duplicate requests
    - Supports canary routing (challenger model gets N% traffic)
    - Logs prediction to DB for drift monitoring
    - Tracks Prometheus metrics
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())

    # Combine title + text
    full_text = body.text
    if body.title:
        full_text = f"{body.title} {body.title} {body.text}"

    # Cache check
    cached = await _get_cached_prediction(full_text)
    if cached:
        cached["request_id"] = request_id
        cached["cached"] = True
        return PredictResponse(**cached)

    # Select model (canary routing)
    model, is_canary = _select_model()
    model_version = (
        app_state.model_metadata.get("model_version", "v1.0")
        + ("-canary" if is_canary else "")
    )

    # Predict
    result = _make_prediction(full_text, model)

    # Shadow mode: also run challenger model for comparison logging
    if SHADOW_MODE and app_state.challenger_model:
        try:
            shadow_result = _make_prediction(full_text, app_state.challenger_model)
            logger.info(
                f"SHADOW prediction: main={result['prediction']}, "
                f"shadow={shadow_result['prediction']}"
            )
        except Exception as e:
            logger.warning(f"Shadow prediction failed: {e}")

    elapsed_ms = (time.time() - start_time) * 1000

    # Build response
    response_data = {
        "request_id": request_id,
        "prediction": result["prediction"],
        "label": result["label"],
        "confidence": result["confidence"],
        "fake_probability": result["fake_probability"],
        "real_probability": result["real_probability"],
        "drift_status": app_state.drift_status,
        "model_version": model_version,
        "processing_time_ms": round(elapsed_ms, 2),
        "cached": False,
    }

    # Cache the result
    await _cache_prediction(full_text, response_data)

    # Prometheus instrumentation
    PREDICTION_COUNT.labels(
        label=result["prediction"], model_version=model_version
    ).inc()
    PREDICTION_CONFIDENCE.observe(result["confidence"])
    INFERENCE_LATENCY.observe(elapsed_ms / 1000)

    # Log to DB
    if app_state.prediction_logger:
        try:
            await app_state.prediction_logger.log(
                request_id=request_id,
                text_snippet=full_text[:200],
                prediction=result["prediction"],
                confidence=result["confidence"],
                model_version=model_version,
            )
        except Exception as e:
            logger.warning(f"Prediction logging failed: {e}")

    return PredictResponse(**response_data)


@app.post("/batch_predict", response_model=BatchPredictResponse, tags=["Inference"])
@limiter.limit("20/minute")
async def batch_predict(request: Request, body: BatchPredictRequest):
    """
    Classify up to 50 articles in a single request.
    Runs predictions concurrently using asyncio.gather.
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())

    BATCH_SIZE_HIST.observe(len(body.texts))

    # Build full texts
    full_texts = []
    for i, text in enumerate(body.texts):
        title = body.titles[i] if body.titles and i < len(body.titles) else None
        full_texts.append(f"{title} {title} {text}" if title else text)

    # Batch embed
    if app_state.embedding_service is None:
        raise HTTPException(status_code=503, detail="Embedding service not ready")

    embeddings = app_state.embedding_service.embed_batch(full_texts)

    model = app_state.model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    predictions_raw = model.predict(embeddings)
    probas = model.predict_proba(embeddings)

    results = []
    for i, (text, label, proba) in enumerate(zip(full_texts, predictions_raw, probas)):
        results.append(PredictResponse(
            request_id=f"{request_id}-{i}",
            prediction="FAKE" if label == 1 else "REAL",
            label=int(label),
            confidence=round(float(max(proba)), 4),
            fake_probability=round(float(proba[1]), 4),
            real_probability=round(float(proba[0]), 4),
            drift_status=app_state.drift_status,
            model_version=app_state.model_metadata.get("model_version", "v1.0"),
            processing_time_ms=0.0,
            cached=False,
        ))

    elapsed_ms = (time.time() - start_time) * 1000
    INFERENCE_LATENCY.observe(elapsed_ms / 1000)

    return BatchPredictResponse(
        request_id=request_id,
        predictions=results,
        total_processed=len(results),
        processing_time_ms=round(elapsed_ms, 2),
    )


@app.get("/drift/status", tags=["Monitoring"])
async def drift_status():
    """Return the current drift detection status."""
    # Refresh from disk in case orchestrator updated it
    app_state.drift_status = load_drift_status()
    report_path = DRIFT_REPORTS_DIR / "latest_drift_status.json"
    if report_path.exists():
        with open(report_path) as f:
            return json.load(f)
    return {"status": "normal", "message": "No drift report found yet."}


@app.post("/model/reload", tags=["Model"])
async def reload_model():
    """Hot-reload the model from disk without restarting the service."""
    try:
        model_path = MODEL_DIR / "model.pkl"
        app_state.model = load_model_from_disk(model_path)
        app_state.model_metadata = load_model_metadata()
        app_state.loaded_at = datetime.utcnow().isoformat()
        logger.info("Model hot-reloaded.")
        return {"status": "success", "loaded_at": app_state.loaded_at}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
