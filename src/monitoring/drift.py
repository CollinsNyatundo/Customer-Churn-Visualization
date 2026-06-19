"""
src/monitoring/drift.py
------------------------
Data drift detection using Evidently AI.
Compares a reference dataset (training data) against a current
production dataset and flags columns that have drifted.

Usage
-----
    from src.monitoring.drift import DriftDetector
    detector = DriftDetector(reference_df)
    report = detector.run(current_df)
    detector.save_report(report, "reports/drift_report.html")
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
from evidently.report import Report

logger = logging.getLogger(__name__)

# Columns to include in drift analysis
NUMERIC_COLS = [
    "cons_12m", "cons_gas_12m", "cons_last_month", "net_margin",
    "num_years_antig", "pow_max", "nb_prod_act", "imp_cons",
    "nps_score", "satisfaction_score", "num_contacts_6m",
    "num_tickets_6m", "avg_resolution_hours", "escalations_6m",
    "num_late_payments_12m", "total_outstanding",
    "cross_source_risk_score", "engagement_score",
]

CATEGORICAL_COLS = [
    "channel_sales", "activity_new", "origin_up",
    "contract_type", "payment_method", "top_ticket_category",
]


class DriftDetector:
    """
    Wraps Evidently drift detection with MLflow artifact logging.

    Parameters
    ----------
    reference_df : the baseline dataset (e.g. training split)
    """

    def __init__(self, reference_df: pd.DataFrame):
        self.reference_df = reference_df
        self._column_mapping = ColumnMapping(
            target="churn",
            numerical_features=[c for c in NUMERIC_COLS if c in reference_df.columns],
            categorical_features=[c for c in CATEGORICAL_COLS if c in reference_df.columns],
        )

    def run(self, current_df: pd.DataFrame) -> Report:
        """
        Run drift + quality + target drift analysis.

        Returns
        -------
        Evidently Report object (call .as_dict() or .save_html())
        """
        logger.info(
            "Running drift detection: reference=%d rows, current=%d rows",
            len(self.reference_df), len(current_df),
        )
        report = Report(metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
            TargetDriftPreset(),
        ])
        report.run(
            reference_data=self.reference_df,
            current_data=current_df,
            column_mapping=self._column_mapping,
        )
        return report

    def save_report(
        self,
        report: Report,
        output_path: str | Path = "reports/drift_report.html",
    ) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        report.save_html(str(path))
        logger.info("Drift report saved to %s", path)
        return path

    def get_drift_summary(self, report: Report) -> dict:
        """
        Extract a flat dict of drift metrics suitable for MLflow logging.

        Returns keys like: n_drifted_features, share_drifted, dataset_drift
        """
        result = report.as_dict()
        metrics = result.get("metrics", [])
        summary: dict = {}

        for m in metrics:
            if m.get("metric") == "DatasetDriftMetric":
                r = m.get("result", {})
                summary["n_drifted_features"] = r.get("number_of_drifted_columns", 0)
                summary["share_drifted"] = round(r.get("share_of_drifted_columns", 0), 4)
                summary["dataset_drift_detected"] = int(r.get("dataset_drift", False))

        return summary

    def log_to_mlflow(self, report: Report, run_id: str | None = None) -> None:
        """Log drift summary metrics and the HTML report as an MLflow artifact."""
        import mlflow
        from src.config import settings

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        summary = self.get_drift_summary(report)

        ctx = mlflow.start_run(run_id=run_id) if run_id else mlflow.start_run(
            run_name="drift-monitoring",
            experiment_id=mlflow.set_experiment(
                settings.mlflow_experiment_pipeline
            ).experiment_id,
        )
        with ctx:
            mlflow.log_metrics({k: float(v) for k, v in summary.items()
                                 if isinstance(v, (int, float))})
            path = self.save_report(report)
            mlflow.log_artifact(str(path), artifact_path="drift_reports")

        logger.info("Drift metrics logged to MLflow: %s", summary)
