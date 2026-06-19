"""
api/predictor.py
-----------------
Loads the latest Production (or Staging) model from the MLflow
registry and exposes a predict() method. The model is loaded once
at startup and cached — thread-safe for gunicorn multi-worker use.
"""
from __future__ import annotations

import logging
import threading
from functools import lru_cache

import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient

from src.config import settings
from src.features.engineering import build_feature_set

logger = logging.getLogger(__name__)
_lock = threading.Lock()


class ChurnPredictor:
    def __init__(self, stage: str = "Production"):
        self.stage = stage
        self._pipeline = None
        self._model_version: str = "unknown"

    def load(self) -> None:
        """Load model from MLflow registry. Falls back to Staging if no Production model."""
        client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
        name = settings.mlflow_registered_model_name

        for s in [self.stage, "Staging"]:
            try:
                versions = client.get_latest_versions(name, stages=[s])
                if versions:
                    mv = versions[0]
                    uri = f"models:/{name}/{s}"
                    self._pipeline = mlflow.sklearn.load_model(uri)
                    self._model_version = f"{s}-v{mv.version}"
                    logger.info("Loaded model %s from stage=%s", name, s)
                    return
            except Exception as e:
                logger.warning("Could not load model from stage=%s: %s", s, e)

        raise RuntimeError(
            f"No model found in registry '{name}'. "
            "Run the full pipeline first: python -m src.pipeline.flows"
        )

    @property
    def model_version(self) -> str:
        return self._model_version

    def predict(self, features: dict) -> tuple[float, bool, str]:
        """
        Parameters
        ----------
        features : flat dict matching CustomerFeatures schema

        Returns
        -------
        (churn_probability, churn_prediction, risk_tier)
        """
        if self._pipeline is None:
            with _lock:
                if self._pipeline is None:
                    self.load()

        df = pd.DataFrame([features])
        df = build_feature_set(df)

        prob = float(self._pipeline.predict_proba(df)[:, 1][0])
        pred = prob >= 0.5

        if prob < 0.30:
            tier = "low"
        elif prob < 0.60:
            tier = "medium"
        else:
            tier = "high"

        return prob, pred, tier


@lru_cache(maxsize=1)
def get_predictor() -> ChurnPredictor:
    """Singleton accessor — safe to call from multiple workers."""
    predictor = ChurnPredictor(stage="Production")
    predictor.load()
    return predictor
