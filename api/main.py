"""
api/main.py
------------
FastAPI prediction service for the churn model.

Run locally:
    uvicorn api.main:app --reload --port 8000

Swagger UI: http://localhost:8000/docs
"""
from __future__ import annotations

import os
import time
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.predictor import get_predictor
from api.schemas import CustomerFeatures, HealthResponse, PredictionResponse
from src.config import settings
from src.logging_config import configure_logging

configure_logging(
    level=os.getenv("LOG_LEVEL", "INFO"),
    json=os.getenv("LOG_FORMAT", "text") == "json",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Churn Prediction API",
    description="Predicts customer churn probability from multi-source features.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Middleware: request timing ────────────────────────────────────────────

@app.middleware("http")
async def add_timing(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = round((time.perf_counter() - start) * 1000, 2)
    response.headers["X-Response-Time-Ms"] = str(elapsed)
    return response


# ── Startup: pre-load model ───────────────────────────────────────────────

@app.on_event("startup")
async def startup_event() -> None:
    logger.info("API starting up — pre-loading churn model …")
    try:
        get_predictor()
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.warning("Model not yet available: %s. Will retry on first request.", e)


# ── Routes ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    """Liveness check — returns model load status."""
    try:
        predictor = get_predictor()
        loaded = True
        version = predictor.model_version
    except Exception:
        loaded = False
        version = "unavailable"

    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        mlflow_uri=settings.mlflow_tracking_uri,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
async def predict(customer: CustomerFeatures) -> PredictionResponse:
    """
    Predict churn probability for a single customer.

    Returns probability, binary prediction, risk tier (low/medium/high),
    and the active model version.
    """
    try:
        predictor = get_predictor()
        prob, pred, tier = predictor.predict(customer.model_dump())
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail="Prediction error — see server logs.")

    logger.info(
        "Prediction: prob=%.4f tier=%s version=%s",
        prob, tier, predictor.model_version,
    )
    return PredictionResponse(
        churn_probability=round(prob, 4),
        churn_prediction=pred,
        risk_tier=tier,
        model_version=predictor.model_version,
    )


@app.post("/predict/batch", tags=["prediction"])
async def predict_batch(customers: list[CustomerFeatures]) -> list[PredictionResponse]:
    """Predict churn for a batch of up to 500 customers."""
    if len(customers) > 500:
        raise HTTPException(status_code=400, detail="Batch size limit is 500.")
    try:
        predictor = get_predictor()
        results = []
        for c in customers:
            prob, pred, tier = predictor.predict(c.model_dump())
            results.append(PredictionResponse(
                churn_probability=round(prob, 4),
                churn_prediction=pred,
                risk_tier=tier,
                model_version=predictor.model_version,
            ))
        return results
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Batch prediction failed: %s", e)
        raise HTTPException(status_code=500, detail="Batch prediction error.")
