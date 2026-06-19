"""
src/pipeline/flows.py
----------------------
Prefect @flow definitions. Run locally:
    python -m src.pipeline.flows
"""

from __future__ import annotations

import logging

from prefect import flow
from prefect.logging import get_run_logger

from src.pipeline.tasks import (
    engineer_features,
    extract_sources,
    generate_report,
    log_pipeline_metrics,
    save_processed,
    train_churn_model,
)

logger = logging.getLogger(__name__)


@flow(
    name="churn-data-pipeline",
    description="Extract → merge → feature engineer → report",
    version="1.0.0",
)
def data_pipeline_flow() -> str:
    """Run data ingestion and feature engineering only."""
    log = get_run_logger()
    log.info("=== Data pipeline flow started ===")
    df, meta = extract_sources()
    df = engineer_features(df)
    log_pipeline_metrics(meta)
    parquet_path = save_processed(df)
    report_path = generate_report(df)
    log.info("Pipeline complete. Parquet: %s | Report: %s", parquet_path, report_path)
    return parquet_path


@flow(
    name="churn-full-pipeline",
    description="Full pipeline: data + model training + MLflow registration",
    version="1.0.0",
)
def full_pipeline_flow(train: bool = True) -> dict:
    """Run the complete pipeline including model training."""
    log = get_run_logger()
    log.info("=== Full pipeline flow started (train=%s) ===", train)
    df, meta = extract_sources()
    df = engineer_features(df)
    log_pipeline_metrics(meta)
    parquet_path = save_processed(df)
    report_path = generate_report(df)
    run_id = train_churn_model(df) if train else None
    result = {
        "parquet_path": parquet_path,
        "report_path": report_path,
        "mlflow_run_id": run_id,
    }
    log.info("Full pipeline complete: %s", result)
    return result


if __name__ == "__main__":
    full_pipeline_flow()
