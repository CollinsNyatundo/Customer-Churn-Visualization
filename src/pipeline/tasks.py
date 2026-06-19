"""
src/pipeline/tasks.py
----------------------
Prefect @task definitions — thin wrappers around src/ logic.
"""

from __future__ import annotations

import logging

import pandas as pd
from prefect import task
from prefect.logging import get_run_logger

from src.data.pipeline import MultiSourcePipeline
from src.features.engineering import build_feature_set
from src.models.churn_model import train as train_model
from src.tracking.mlflow_tracker import log_metrics, pipeline_run

logger = logging.getLogger(__name__)


@task(name="extract-and-merge-sources", retries=2, retry_delay_seconds=30)
def extract_sources() -> tuple[pd.DataFrame, dict]:
    """Run the multi-source pipeline and return merged df + metadata."""
    log = get_run_logger()
    log.info("Starting multi-source extraction ...")
    pipeline = MultiSourcePipeline()
    df, meta = pipeline.run()
    log.info("Extraction complete: %d rows, %d cols", len(df), len(df.columns))
    return df, meta


@task(name="engineer-features")
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering transformations."""
    log = get_run_logger()
    before = set(df.columns)
    df = build_feature_set(df)
    new_cols = set(df.columns) - before
    log.info("Feature engineering added %d columns", len(new_cols))
    return df


@task(name="log-pipeline-metrics")
def log_pipeline_metrics(meta: dict) -> None:
    """Log data pipeline metadata to MLflow."""
    log = get_run_logger()
    with pipeline_run(run_name="prefect-pipeline-run"):
        log_metrics(meta)
    log.info("Pipeline metrics logged to MLflow")


@task(name="train-churn-model", retries=1, retry_delay_seconds=60)
def train_churn_model(df: pd.DataFrame) -> str:
    """Train the gradient-boosted model and register it in MLflow."""
    log = get_run_logger()
    log.info("Starting model training ...")
    run_id = train_model(df)
    log.info("Model training complete. MLflow run_id: %s", run_id)
    return run_id


@task(name="generate-sweetviz-report")
def generate_report(
    df: pd.DataFrame,
    output_path: str = "reports/sweetviz_report.html",
) -> str:
    """Generate the Sweetviz EDA report and save to disk."""
    from pathlib import Path

    import sweetviz as sv

    log = get_run_logger()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = sv.analyze(df, target_feat="churn")
    report.show_html(str(path), open_browser=False)
    log.info("Sweetviz report saved to %s", path)
    return str(path)


@task(name="save-processed-data")
def save_processed(df: pd.DataFrame, filename: str = "merged_features.parquet") -> str:
    """Persist the processed dataset as Parquet for downstream use."""
    from src.config import settings

    settings.ensure_dirs()
    path = settings.data_processed_dir / filename
    df.to_parquet(path, index=False)
    return str(path)
