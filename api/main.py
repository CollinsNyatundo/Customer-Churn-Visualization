"""
api/main.py
------------
FastAPI prediction service — auth, predict, explain, batch, health.

Run locally:
    uvicorn api.main:app --reload --port 8000
    make run-api

Swagger UI: http://localhost:8000/docs
"""

from __future__ import annotations

import logging
import os
import time

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from api.auth import verify_api_key
from api.explain import explain_prediction
from api.predictor import get_predictor
from api.schemas import (
    CustomerFeatures,
    ExplainResponse,
    FeatureContribution,
    HealthResponse,
    PredictionResponse,
)
from src.config import settings
from src.logging_config import configure_logging

configure_logging(
    level=os.getenv("LOG_LEVEL", "INFO"),
    json=os.getenv("LOG_FORMAT", "text") == "json",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Churn Prediction API",
    description=(
        "Multi-source churn prediction with SHAP explainability.\n\n"
        "**Authentication**: pass your API key in the `X-API-Key` header.\n"
        "Set `API_KEYS=key1,key2` env var to enable enforcement "
        "(omit for development bypass)."
    ),
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Middleware ────────────────────────────────────────────────────────────────


@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    ms = round((time.perf_counter() - start) * 1000, 2)
    response.headers["X-Response-Time-Ms"] = str(ms)
    return response


# ── Startup ───────────────────────────────────────────────────────────────────


@app.on_event("startup")
async def startup_event() -> None:
    logger.info("API starting — pre-loading churn model ...")
    try:
        get_predictor()
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.warning("Model not yet available: %s. Will retry on first request.", e)


# ── Ops ───────────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    """Liveness check — returns model load status and auth mode."""
    import os

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
        auth_enabled=bool(os.getenv("API_KEYS", "")),
    )


# ── Prediction ────────────────────────────────────────────────────────────────


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["prediction"],
)
async def predict(
    customer: CustomerFeatures,
    _: str = Depends(verify_api_key),
) -> PredictionResponse:
    """Predict churn probability for a single customer."""
    try:
        predictor = get_predictor()
        prob, pred, tier = predictor.predict(customer.model_dump())
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail="Prediction error.")

    return PredictionResponse(
        churn_probability=round(prob, 4),
        churn_prediction=pred,
        risk_tier=tier,
        model_version=predictor.model_version,
    )


@app.post(
    "/predict/batch",
    response_model=list[PredictionResponse],
    tags=["prediction"],
)
async def predict_batch(
    customers: list[CustomerFeatures],
    _: str = Depends(verify_api_key),
) -> list[PredictionResponse]:
    """Predict churn for a batch of up to 500 customers."""
    if len(customers) > 500:
        raise HTTPException(status_code=400, detail="Batch size limit is 500.")
    try:
        predictor = get_predictor()
        return [
            PredictionResponse(
                churn_probability=round(p, 4),
                churn_prediction=pred,
                risk_tier=tier,
                model_version=predictor.model_version,
            )
            for c in customers
            for p, pred, tier in [predictor.predict(c.model_dump())]
        ]
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Batch prediction failed: %s", e)
        raise HTTPException(status_code=500, detail="Batch prediction error.")


# ── Explainability ────────────────────────────────────────────────────────────


@app.post(
    "/explain",
    response_model=ExplainResponse,
    tags=["explainability"],
)
async def explain(
    customer: CustomerFeatures,
    top_n: int = 10,
    _: str = Depends(verify_api_key),
) -> ExplainResponse:
    """
    Predict churn AND return the top N SHAP feature attributions.

    - `top_n`: number of features to return (default 10, max 30)
    - Each feature shows its raw value, SHAP contribution, and direction
    """
    top_n = min(top_n, 30)
    try:
        predictor = get_predictor()
        prob, pred, tier = predictor.predict(customer.model_dump())
        contributions = explain_prediction(
            predictor._pipeline,
            customer.model_dump(),
            top_n=top_n,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Explain failed: %s", e)
        raise HTTPException(status_code=500, detail="Explanation error.")

    return ExplainResponse(
        churn_probability=round(prob, 4),
        churn_prediction=pred,
        risk_tier=tier,
        model_version=predictor.model_version,
        top_features=[FeatureContribution(**c) for c in contributions],
    )
