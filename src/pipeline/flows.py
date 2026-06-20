"""
src/pipeline/flows.py
----------------------
Prefect @flow definitions.

Shared pipeline steps live in _run_data_steps() — both flows call it
rather than duplicating the same 5 task calls.

Run locally:
    python -m src.pipeline.flows
"""

from __future__ import annotations

import logging

import pandas as pd
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

log = logging.getLogger(__name__)


def _run_data_steps() -> tuple[pd.DataFrame, str, str]:
    """
    Shared sequence: extract → feature-engineer → log → save → report.
    Returns (featured_df, parquet_path, report_path).
    Called by both flows to avoid duplication.
    """
    run_log = get_run_logger()
    df, meta = extract_sources()
    df = engineer_features(df)
    log_pipeline_metrics(meta)
    parquet_path = save_processed(df)
    report_path = generate_report(df)
    run_log.info("Data steps complete: %d rows, parquet=%s", len(df), parquet_path)
    return df, parquet_path, report_path


@flow(
    name="churn-data-pipeline",
    description="Extract → merge → feature engineer → save → report (no training)",
    version="1.1.0",
)
def data_pipeline_flow() -> str:
    """Data-only pipeline: ingest and feature engineering, no model training."""
    get_run_logger().info("=== Data pipeline flow started ===")
    _, parquet_path, _ = _run_data_steps()
    return parquet_path


@flow(
    name="churn-full-pipeline",
    description="Full pipeline: data + model training + MLflow registration",
    version="1.1.0",
)
def full_pipeline_flow(train: bool = True) -> dict:
    """Full pipeline: data steps + optional model training."""
    run_log = get_run_logger()
    run_log.info("=== Full pipeline flow started (train=%s) ===", train)
    df, parquet_path, report_path = _run_data_steps()
    run_id = train_churn_model(df) if train else None
    result = {
        "parquet_path": parquet_path,
        "report_path": report_path,
        "mlflow_run_id": run_id,
    }
    run_log.info("Full pipeline complete: %s", result)
    return result


if __name__ == "__main__":
    full_pipeline_flow()
