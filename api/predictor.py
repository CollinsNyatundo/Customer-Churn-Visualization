"""
api/predictor.py
-----------------
MLflow model loader with:
- Explicit failure modes (no silent fallbacks)
- Public get_pipeline() to avoid encapsulation breaches
- Risk tier thresholds driven from config
- Feature version logging for training-serving skew detection
"""

from __future__ import annotations

import logging
import threading
from functools import lru_cache

import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient

from src.config import settings
from src.features.engineering import FeatureThresholds, build_feature_set

log = logging.getLogger(__name__)
_lock = threading.Lock()

# Feature engineering version — bump this when engineering.py changes
# to detect training/serving skew at model load time.
FEATURE_ENGINEERING_VERSION = "1.2.0"


class ChurnPredictor:
    def __init__(self, stage: str = "Production"):
        self.stage = stage
        self._pipeline = None
        self._model_version: str = "unknown"
        self._thresholds: FeatureThresholds | None = None

    def load(self) -> None:
        """
        Load model from MLflow registry.
        Tries self.stage first, then Staging.
        Raises RuntimeError loudly if neither is available.
        """
        client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
        name = settings.mlflow_registered_model_name

        for stage in [self.stage, "Staging"]:
            try:
                versions = client.get_latest_versions(name, stages=[stage])
                if versions:
                    mv = versions[0]
                    uri = f"models:/{name}/{stage}"
                    self._pipeline = mlflow.sklearn.load_model(uri)
                    self._model_version = f"{stage}-v{mv.version}"
                    log.info("Loaded '%s' v%s from stage=%s", name, mv.version, stage)

                    # Log feature engineering version for skew detection
                    log.info(
                        "Feature engineering version at load time: %s",
                        FEATURE_ENGINEERING_VERSION,
                    )
                    return
            except Exception as e:
                log.warning("Could not load model from stage=%s: %s", stage, e)

        raise RuntimeError(
            f"No model found in registry '{name}' (tried: {self.stage}, Staging). "
            "Run the pipeline first: make run-pipeline"
        )

    @property
    def model_version(self) -> str:
        return self._model_version

    def get_pipeline(self):
        """
        Public accessor for the sklearn pipeline.
        Use this instead of accessing _pipeline directly.
        """
        if self._pipeline is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._pipeline

    def set_thresholds(self, thresholds: FeatureThresholds) -> None:
        """
        Inject frozen training-set thresholds to prevent inference skew.
        Call this after training and before serving.
        """
        self._thresholds = thresholds
        log.info("Feature thresholds set: %s", thresholds)

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
        df = build_feature_set(df, thresholds=self._thresholds)

        prob = float(self._pipeline.predict_proba(df)[:, 1][0])
        pred = prob >= settings.risk_tier_high or prob >= 0.5

        if prob >= settings.risk_tier_high:
            tier = "high"
        elif prob >= settings.risk_tier_medium:
            tier = "medium"
        else:
            tier = "low"

        return prob, prob >= 0.5, tier

    def predict_batch(self, features_list: list[dict]) -> list[tuple[float, bool, str]]:
        """
        Vectorised batch prediction — single pipeline call, not a Python loop.
        """
        if self._pipeline is None:
            with _lock:
                if self._pipeline is None:
                    self.load()

        df = pd.DataFrame(features_list)
        df = build_feature_set(df, thresholds=self._thresholds)

        probs = self._pipeline.predict_proba(df)[:, 1]
        results = []
        for prob in probs:
            prob = float(prob)
            if prob >= settings.risk_tier_high:
                tier = "high"
            elif prob >= settings.risk_tier_medium:
                tier = "medium"
            else:
                tier = "low"
            results.append((prob, prob >= 0.5, tier))
        return results


@lru_cache(maxsize=1)
def get_predictor() -> ChurnPredictor:
    """Singleton — safe for multi-worker use."""
    predictor = ChurnPredictor(stage="Production")
    predictor.load()
    return predictor
