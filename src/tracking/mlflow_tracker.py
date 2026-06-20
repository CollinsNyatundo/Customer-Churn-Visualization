"""
src/tracking/mlflow_tracker.py
--------------------------------
MLflow context managers with explicit, visible failure modes.

Design rules
------------
- Failures are LOUD: log at ERROR level with full traceback context.
- Never return MagicMock or fake run IDs — callers must handle failure.
- Use MLFLOW_OFFLINE=true to explicitly opt in to offline (no-op) mode
  rather than silently degrading when the server is unreachable.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import mlflow
import mlflow.sklearn

from src.config import settings

log = logging.getLogger(__name__)

_OFFLINE = os.getenv("MLFLOW_OFFLINE", "false").lower() in ("1", "true", "yes")


def _is_offline() -> bool:
    return _OFFLINE


def _init_experiment(experiment_name: str) -> None:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)


@dataclass
class TrackingResult:
    """Returned by context managers — carries the run ID even in offline mode."""

    run_id: str
    offline: bool = False


@contextmanager
def pipeline_run(run_name: str = "data-pipeline", tags: dict | None = None):
    """
    Context manager for a data pipeline MLflow run.

    Offline mode (MLFLOW_OFFLINE=true): logs a warning, yields a no-op result.
    Server unreachable: raises with a clear message — not silently swallowed.
    """
    if _is_offline():
        log.warning("MLflow offline mode — pipeline metrics will NOT be tracked.")
        yield TrackingResult(run_id="offline", offline=True)
        return

    try:
        _init_experiment(settings.mlflow_experiment_pipeline)
        with mlflow.start_run(run_name=run_name, tags=tags or {}) as run:
            log.info("MLflow pipeline run started: %s", run.info.run_id)
            yield TrackingResult(run_id=run.info.run_id)
            log.info("MLflow pipeline run complete: %s", run.info.run_id)
    except Exception as exc:
        log.error(
            "MLflow pipeline tracking FAILED (uri=%s). " "Set MLFLOW_OFFLINE=true to suppress. Error: %s",
            settings.mlflow_tracking_uri,
            exc,
            exc_info=True,
        )
        raise


@contextmanager
def model_run(run_name: str = "churn-model", tags: dict | None = None):
    """
    Context manager for a model training MLflow run.
    Raises on server failure — never returns fake run IDs.
    """
    if _is_offline():
        log.warning("MLflow offline mode — model metrics will NOT be tracked.")
        yield TrackingResult(run_id="offline", offline=True)
        return

    try:
        _init_experiment(settings.mlflow_experiment_model)
        with mlflow.start_run(run_name=run_name, tags=tags or {}) as run:
            log.info("MLflow model run started: %s", run.info.run_id)
            yield TrackingResult(run_id=run.info.run_id)
            log.info("MLflow model run complete: %s", run.info.run_id)
    except Exception as exc:
        log.error(
            "MLflow model tracking FAILED (uri=%s). " "Set MLFLOW_OFFLINE=true to suppress. Error: %s",
            settings.mlflow_tracking_uri,
            exc,
            exc_info=True,
        )
        raise


def log_metrics(metrics: dict[str, Any]) -> None:
    if _is_offline():
        return
    numeric = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
    if numeric:
        mlflow.log_metrics(numeric)


def log_params(params: dict[str, Any]) -> None:
    if _is_offline():
        return
    mlflow.log_params({k: str(v) for k, v in params.items()})


def register_model(run_id: str, model_uri_suffix: str = "model", stage: str = "Staging") -> None:
    """
    Register a logged sklearn model and promote it to the given stage.
    Raises on failure — never silently skips.
    """
    if _is_offline():
        log.warning("MLflow offline — model registration skipped for run_id=%s", run_id)
        return

    if run_id == "offline":
        log.warning("Offline run_id — skipping model registration.")
        return

    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    name = settings.mlflow_registered_model_name
    model_uri = f"runs:/{run_id}/{model_uri_suffix}"

    try:
        client.create_registered_model(name)
    except Exception:
        pass  # Already exists — expected on subsequent runs

    mv = client.create_model_version(name=name, source=model_uri, run_id=run_id)
    client.transition_model_version_stage(name=name, version=mv.version, stage=stage)
    log.info("Model '%s' v%s promoted to %s", name, mv.version, stage)
