from __future__ import annotations

import logging
from contextlib import contextmanager
from unittest.mock import MagicMock

import mlflow

from src.config import settings

logger = logging.getLogger(__name__)


def _init(exp_name):
    try:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(exp_name)
    except Exception:
        pass


@contextmanager
def pipeline_run(run_name="data-pipeline", tags=None):
    try:
        _init(settings.mlflow_experiment_pipeline)
        with mlflow.start_run(run_name=run_name, tags=tags or {}) as run:
            yield run
    except Exception as e:
        logger.warning("MLflow unavailable, skipping tracking: %s", e)
        yield MagicMock(info=MagicMock(run_id="offline"))


@contextmanager
def model_run(run_name="churn-model", tags=None):
    try:
        _init(settings.mlflow_experiment_model)
        with mlflow.start_run(run_name=run_name, tags=tags or {}) as run:
            yield run
    except Exception as e:
        logger.warning("MLflow unavailable: %s", e)
        yield MagicMock(info=MagicMock(run_id="offline"))


def log_metrics(metrics):
    try:
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
    except Exception:
        pass


def log_params(params):
    try:
        mlflow.log_params({k: str(v) for k, v in params.items()})
    except Exception:
        pass


def register_model(run_id, model_uri_suffix="model", stage="Staging"):
    logger.info("Model registration skipped (offline mode): run_id=%s stage=%s", run_id, stage)
