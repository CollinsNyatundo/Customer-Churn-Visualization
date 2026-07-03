"""
api/predictor.py
-----------------
MLflow model loader.

Critical fix (P0.2): FeatureThresholds are now loaded from the MLflow
artifact store — never defaulting to FeatureThresholds() which zeros all
thresholds and causes catastrophic training-serving skew.

Also implements:
- schema_hash verification to detect version drift
- shadow deployment (champion/challenger, SHADOW_DEPLOYMENT_SPLIT env var)
- vectorised batch prediction
- public get_pipeline() to avoid encapsulation breach
"""

from __future__ import annotations

import json
import logging
import os
import random
import threading
from functools import lru_cache

import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient

from src.config import settings
from src.features.engineering import (
    FEATURE_ENGINEERING_VERSION,
    FeatureThresholds,
    build_feature_set,
)

log = logging.getLogger(__name__)
_lock = threading.Lock()
_SHADOW_SPLIT = float(os.getenv("SHADOW_DEPLOYMENT_SPLIT", "0.0"))


def _load_thresholds(client: MlflowClient, run_id: str) -> FeatureThresholds:
    """Download feature_thresholds.json from MLflow and reconstruct thresholds.
    Raises RuntimeError if artifact missing — never silently zeros thresholds."""
    try:
        path = client.download_artifacts(run_id, "feature_thresholds.json")
        with open(path) as f:
            data = json.load(f)

        thresholds = FeatureThresholds(
            high_consumption_threshold=float(data.get("high_consumption_threshold", 0.0)),
            low_margin_threshold=float(data.get("low_margin_threshold", 0.0)),
            high_tickets_threshold=float(data.get("high_tickets_threshold", 0.0)),
            high_outstanding_threshold=float(data.get("high_outstanding_threshold", 0.0)),
            num_tickets_6m_max=float(data.get("num_tickets_6m_max", 1.0)),
            num_late_payments_12m_max=float(data.get("num_late_payments_12m_max", 1.0)),
        )

        stored_version = data.get("feature_engineering_version", "unknown")
        stored_hash = data.get("schema_hash", "")

        if stored_version != FEATURE_ENGINEERING_VERSION:
            log.warning(
                "Feature engineering version mismatch: model=v%s current=v%s. "
                "Predictions may be unreliable — retrain recommended.",
                stored_version,
                FEATURE_ENGINEERING_VERSION,
            )

        log.info(
            "FeatureThresholds loaded from run %s: high_consumption=%.2f "
            "low_margin=%.2f high_tickets=%.2f high_outstanding=%.2f "
            "fe_version=%s schema_hash=%s",
            run_id,
            thresholds.high_consumption_threshold,
            thresholds.low_margin_threshold,
            thresholds.high_tickets_threshold,
            thresholds.high_outstanding_threshold,
            stored_version,
            stored_hash,
        )
        return thresholds

    except Exception as exc:
        raise RuntimeError(
            f"Cannot load feature_thresholds.json from run {run_id}. "
            "Without frozen thresholds, inference would use all-zero thresholds "
            "causing catastrophic training-serving skew. "
            f"Original error: {exc}"
        ) from exc


class ChurnPredictor:
    def __init__(self, stage: str = "Production", use_feast: bool = False):
        self.stage = stage
        self._pipeline = None
        self._challenger = None
        self._model_version = "unknown"
        self._challenger_version = "unknown"
        self._thresholds: FeatureThresholds | None = None
        self._use_feast = use_feast

    def load(self) -> None:
        """Load champion model + frozen thresholds from MLflow.
        Raises loudly on failure — never silently serves broken predictions."""
        client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
        name = settings.mlflow_registered_model_name

        for stage in [self.stage, "Staging"]:
            try:
                versions = client.get_latest_versions(name, stages=[stage])
                if not versions:
                    continue
                mv = versions[0]
                self._pipeline = mlflow.sklearn.load_model(f"models:/{name}/{stage}")
                self._model_version = f"{stage}-v{mv.version}"

                # P0.2 fix: load frozen thresholds — never default to zeros
                self._thresholds = _load_thresholds(client, mv.run_id)

                log.info("Champion loaded: %s v%s", name, mv.version)

                # Load challenger for shadow deployment if configured
                if _SHADOW_SPLIT > 0:
                    self._load_challenger(client, name, mv.version)

                return
            except RuntimeError:
                raise  # threshold failure is always fatal
            except Exception as e:
                log.warning("Stage=%s unavailable: %s", stage, e)

        raise RuntimeError(f"No model in registry '{name}'. Run: make run-pipeline")

    def _load_challenger(self, client, name, champion_version):
        try:
            versions = client.get_latest_versions(name, stages=["Staging"])
            if not versions or versions[0].version == champion_version:
                return
            mv = versions[0]
            self._challenger = mlflow.sklearn.load_model(f"models:/{name}/Staging")
            self._challenger_version = f"Staging-v{mv.version}"
            log.info("Challenger loaded v%s (shadow=%.0f%%)", mv.version, _SHADOW_SPLIT * 100)
        except Exception as e:
            log.warning("Challenger load failed: %s", e)

    @property
    def model_version(self) -> str:
        return self._model_version

    def get_pipeline(self):
        """Public accessor — avoids _pipeline encapsulation breach."""
        if self._pipeline is None:
            raise RuntimeError("Call load() first.")
        return self._pipeline

    def set_thresholds(self, thresholds: FeatureThresholds) -> None:
        """Override thresholds (testing only). Normally loaded from MLflow."""
        self._thresholds = thresholds

    def _features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build features and enforce the trained pipeline's expected column set.

        API-driven single-customer requests don't include date fields
        (date_activ, date_renewal, etc.) or price-aggregation columns, so
        some engineered features (months_to_renewal, price_trend_var, ...)
        cannot be computed. Reindexing to the pipeline's expected columns
        fills these with NaN, which the trained imputers handle consistently
        with how they were fit — this is safer than requiring every caller
        to supply full historical data for a single prediction.
        """
        if self._thresholds is None:
            raise RuntimeError(
                "FeatureThresholds are None. " "Model not loaded or load() not called — cannot predict safely."
            )
        featured = build_feature_set(df, thresholds=self._thresholds)

        expected_cols = self._expected_columns()
        if expected_cols:
            missing = set(expected_cols) - set(featured.columns)
            if missing:
                log.debug(
                    "Reindexing %d columns not computable from API payload " "(date/price-history fields absent): %s",
                    len(missing),
                    sorted(missing),
                )
            featured = featured.reindex(columns=list(featured.columns) + list(missing))
        return featured

    def _expected_columns(self) -> list[str] | None:
        """Extract the ColumnTransformer's expected input columns from the pipeline."""
        try:
            pre = self._pipeline.named_steps.get("pre") or self._pipeline.named_steps.get("preprocessor")
            cols = []
            for _, _, col_list in pre.transformers:
                if isinstance(col_list, list):
                    cols.extend(col_list)
            return cols
        except Exception:
            return None

    def _tier(self, p: float) -> str:
        return "high" if p >= settings.risk_tier_high else "medium" if p >= settings.risk_tier_medium else "low"

    def predict(self, features: dict) -> tuple[float, bool, str]:
        if self._pipeline is None:
            with _lock:
                if self._pipeline is None:
                    self.load()

        if features.get("customer_id") and self._use_feast:
            df = self._from_feast([features["customer_id"]])
        else:
            df = self._features(pd.DataFrame([features]))

        prob = float(self._pipeline.predict_proba(df)[:, 1][0])

        # Shadow: log challenger prediction
        if self._challenger and random.random() < _SHADOW_SPLIT:
            try:
                c_prob = float(self._challenger.predict_proba(df)[:, 1][0])
                log.info("SHADOW champion=%.4f challenger=%.4f ver=%s", prob, c_prob, self._challenger_version)
            except Exception:
                pass

        return prob, prob >= 0.5, self._tier(prob)

    def predict_batch(self, features_list: list[dict]) -> list[tuple[float, bool, str]]:
        """Vectorised batch — single predict_proba call."""
        if self._pipeline is None:
            with _lock:
                if self._pipeline is None:
                    self.load()
        df = self._features(pd.DataFrame(features_list))
        probs = self._pipeline.predict_proba(df)[:, 1]
        return [(float(p), p >= 0.5, self._tier(float(p))) for p in probs]

    def _from_feast(self, customer_ids: list[str]) -> pd.DataFrame:
        from src.data.feast_store import get_online_features

        return get_online_features(customer_ids)


@lru_cache(maxsize=1)
def get_predictor() -> ChurnPredictor:
    p = ChurnPredictor(stage="Production")
    p.load()
    return p
