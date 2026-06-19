"""
src/monitoring/monitor_flow.py
-------------------------------
Prefect flow for drift detection + alert dispatch.
Schedule: daily at 04:00 UTC (see prefect.yaml).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from prefect import flow, task
from prefect.logging import get_run_logger

from src.config import settings
from src.monitoring.alerts import AlertManager
from src.monitoring.drift import DriftDetector

logger = logging.getLogger(__name__)

REFERENCE_PATH = settings.data_processed_dir / "reference.parquet"
CURRENT_PATH = settings.data_processed_dir / "merged_features.parquet"
_alerts = AlertManager()


@task(name="load-reference-data")
def load_reference(path: Path = REFERENCE_PATH) -> pd.DataFrame:
    log = get_run_logger()
    if not path.exists():
        raise FileNotFoundError(f"Reference dataset not found at {path}. Run the pipeline first.")
    df = pd.read_parquet(path)
    log.info("Reference data loaded: %d rows", len(df))
    return df


@task(name="load-current-data")
def load_current(path: Path = CURRENT_PATH) -> pd.DataFrame:
    log = get_run_logger()
    if not path.exists():
        raise FileNotFoundError(f"Current dataset not found at {path}.")
    df = pd.read_parquet(path)
    log.info("Current data loaded: %d rows", len(df))
    return df


@task(name="run-drift-detection")
def run_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> dict:
    log = get_run_logger()
    detector = DriftDetector(reference_df)
    report = detector.run(current_df)
    report_path = detector.save_report(report, settings.reports_dir / "drift_report.html")
    detector.log_to_mlflow(report)
    summary = detector.get_drift_summary(report)
    log.info("Drift summary: %s", summary)

    if summary.get("dataset_drift_detected"):
        n = summary.get("n_drifted_features", 0)
        share = summary.get("share_drifted", 0.0)
        log.warning("DRIFT DETECTED: %d features drifted (%.1f%%)", n, share * 100)
        _alerts.send_drift_alert(
            n_drifted=n,
            share=share,
            report_path=str(report_path),
        )

    return summary


@flow(name="drift-monitoring", description="Detect data drift vs reference dataset + alert")
def drift_monitoring_flow() -> dict:
    reference_df = load_reference()
    current_df = load_current()
    return run_drift(reference_df, current_df)


if __name__ == "__main__":
    drift_monitoring_flow()
