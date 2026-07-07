"""
api/main.py
------------
FastAPI prediction service.

Run:
    uvicorn api.main:app --reload --port 8000
    make run-api
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from api.auth import _load_valid_keys, verify_api_key
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

limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])

configure_logging(
    level=os.getenv("LOG_LEVEL", "INFO"),
    json=os.getenv("LOG_FORMAT", "text") == "json",
)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan: startup + shutdown logic."""
    # ── Startup ───────────────────────────────────────────────────────────
    try:
        _load_valid_keys()
    except RuntimeError as e:
        log.error("Auth misconfiguration: %s", e)
        raise

    log.info("Pre-loading churn model ...")
    if settings.env == "production":
        try:
            get_predictor()
            log.info("Model loaded successfully.")
        except Exception as e:
            log.error("FATAL: model could not be loaded at startup: %s", e)
            raise
    else:
        try:
            get_predictor()
            log.info("Model loaded successfully.")
        except Exception as e:
            log.warning(
                "Model not available at startup (dev mode): %s. " "Inference will fail until model is registered.", e
            )
    yield
    # ── Shutdown ──────────────────────────────────────────────────────────
    log.info("API shutting down.")


app = FastAPI(
    lifespan=lifespan,
    title="Churn Prediction API",
    description=(
        "Multi-source churn prediction with SHAP explainability.\n\n"
        "**Auth**: pass your key in `X-API-Key` header. "
        "Set `API_KEYS` env var to enforce; omit for dev bypass (not for production)."
    ),
    version="1.2.0",
)

# ── CORS — configurable, not hardcoded open ──────────────────────────────────
_cors_origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
if "*" in _cors_origins and settings.env == "production":
    log.error("CORS is wide open (allow_origins=['*']) in ENV=production. Set CORS_ORIGINS.")

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Middleware ────────────────────────────────────────────────────────────────


@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Response-Time-Ms"] = str(round((time.perf_counter() - t0) * 1000, 2))
    return response


# ── Ops ───────────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    """Liveness check — returns model status, auth mode, env."""
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


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
@limiter.limit("100/minute")
async def predict(
    request: Request,
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
        log.exception("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail="Prediction error.")

    return PredictionResponse(
        churn_probability=round(prob, 4),
        churn_prediction=pred,
        risk_tier=tier,
        model_version=predictor.model_version,
    )


@app.post("/predict/batch", response_model=list[PredictionResponse], tags=["prediction"])
@limiter.limit("20/minute")
async def predict_batch(
    request: Request,
    customers: list[CustomerFeatures],
    _: str = Depends(verify_api_key),
) -> list[PredictionResponse]:
    """
    Predict churn for up to 500 customers in a single vectorised call.
    Single sklearn pipeline.predict_proba() call — not a Python loop.
    """
    if len(customers) > 500:
        raise HTTPException(status_code=400, detail="Batch size limit is 500.")
    if not customers:
        return []

    try:
        predictor = get_predictor()
        results = predictor.predict_batch([c.model_dump() for c in customers])
        return [
            PredictionResponse(
                churn_probability=round(prob, 4),
                churn_prediction=pred,
                risk_tier=tier,
                model_version=predictor.model_version,
            )
            for prob, pred, tier in results
        ]
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        log.exception("Batch prediction failed: %s", e)
        raise HTTPException(status_code=500, detail="Batch prediction error.")


# ── Explainability ────────────────────────────────────────────────────────────


@app.post("/explain", response_model=ExplainResponse, tags=["explainability"])
@limiter.limit("30/minute")
async def explain(
    request: Request,
    customer: CustomerFeatures,
    top_n: int = 10,
    _: str = Depends(verify_api_key),
) -> ExplainResponse:
    """
    Predict + return top-N SHAP feature attributions.
    Uses the public get_pipeline() accessor — no internal attribute access.
    """
    top_n = min(top_n, 30)
    try:
        predictor = get_predictor()
        prob, pred, tier = predictor.predict(customer.model_dump())
        # Use public accessor, not predictor._pipeline
        contributions = explain_prediction(
            predictor.get_pipeline(),
            customer.model_dump(),
            top_n=top_n,
            thresholds=predictor.thresholds,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        log.exception("Explain failed: %s", e)
        raise HTTPException(status_code=500, detail="Explanation error.")

    return ExplainResponse(
        churn_probability=round(prob, 4),
        churn_prediction=pred,
        risk_tier=tier,
        model_version=predictor.model_version,
        top_features=[FeatureContribution(**c) for c in contributions],
    )


@app.post("/explain/batch", response_model=list[ExplainResponse], tags=["explainability"])
@limiter.limit("10/minute")
async def explain_batch(
    request: Request,
    customers: list[CustomerFeatures],
    top_n: int = 5,
    _: str = Depends(verify_api_key),
) -> list[ExplainResponse]:
    """
    SHAP explanations for a batch of customers (up to 50).
    Required for compliance-grade explainability at scale.
    """
    if len(customers) > 50:
        raise HTTPException(status_code=400, detail="Batch explain limit is 50.")
    if not customers:
        return []
    try:
        predictor = get_predictor()
        results = []
        for customer in customers:
            prob, pred, tier = predictor.predict(customer.model_dump())
            contributions = explain_prediction(
                predictor.get_pipeline(),
                customer.model_dump(),
                top_n=min(top_n, 30),
                thresholds=predictor.thresholds,
            )
            results.append(
                ExplainResponse(
                    churn_probability=round(prob, 4),
                    churn_prediction=pred,
                    risk_tier=tier,
                    model_version=predictor.model_version,
                    top_features=[FeatureContribution(**c) for c in contributions],
                )
            )
        return results
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        log.exception("Batch explain failed: %s", e)
        raise HTTPException(status_code=500, detail="Batch explanation error.")
