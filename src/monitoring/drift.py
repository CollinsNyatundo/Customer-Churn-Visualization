"""
src/monitoring/drift.py
------------------------
Data drift detection using Evidently AI.
Compares a reference dataset (training data) against a current
production dataset and flags columns that have drifted.

Rewritten for Evidently 0.7.x (code review fix)
--------------------------------------------------
Evidently made breaking API changes between 0.4.x and 0.7.x:
  evidently.ColumnMapping              -> removed entirely (no legacy shim)
  evidently.metric_preset.*            -> evidently.presets.*
  evidently.report.Report              -> evidently.Report
  report.run(..., column_mapping=...)  -> Dataset.from_pandas(df, data_definition=...)
  report.as_dict()                     -> snapshot.dict()
  metric name "DatasetDriftMetric"     -> "DriftedColumnsCount(drift_share=X)"

The old API was never importable against the pinned evidently==0.7.21 in
requirements.txt — this module could not be imported at all, meaning
drift detection has been non-functional since it was written. This
rewrite uses the current native API and was verified against a real
Report.run() call.

Usage
-----
    from src.monitoring.drift import DriftDetector
    detector = DriftDetector(reference_df)
    snapshot = detector.run(current_df)
    detector.save_report(snapshot, "reports/drift_report.html")
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

logger = logging.getLogger(__name__)

# Columns to include in drift analysis
NUMERIC_COLS = [
    "cons_12m",
    "cons_gas_12m",
    "cons_last_month",
    "net_margin",
    "num_years_antig",
    "pow_max",
    "nb_prod_act",
    "imp_cons",
    "nps_score",
    "satisfaction_score",
    "num_contacts_6m",
    "num_tickets_6m",
    "avg_resolution_hours",
    "escalations_6m",
    "num_late_payments_12m",
    "total_outstanding",
    "cross_source_risk_score",
    "engagement_score",
]

CATEGORICAL_COLS = [
    "channel_sales",
    "activity_new",
    "origin_up",
    "contract_type",
    "payment_method",
    "top_ticket_category",
]


class DriftDetector:
    """
    Wraps Evidently drift detection with MLflow artifact logging.

    Parameters
    ----------
    reference_df : the baseline dataset (e.g. training split, or the
                    snapshot saved by src.pipeline.tasks.save_reference_snapshot)
    """

    def __init__(self, reference_df: pd.DataFrame):
        self.reference_df = reference_df
        self._numeric_cols = [c for c in NUMERIC_COLS if c in reference_df.columns]
        self._categorical_cols = [c for c in CATEGORICAL_COLS if c in reference_df.columns]
        self._data_definition = DataDefinition(
            numerical_columns=self._numeric_cols,
            categorical_columns=self._categorical_cols,
        )

    def run(self, current_df: pd.DataFrame):
        """
        Run data drift + summary analysis.

        Returns
        -------
        Evidently Snapshot object (call .dict(), .save_html(), or .json())
        """
        logger.info(
            "Running drift detection: reference=%d rows, current=%d rows",
            len(self.reference_df),
            len(current_df),
        )
        ref_dataset = Dataset.from_pandas(self.reference_df, data_definition=self._data_definition)
        cur_dataset = Dataset.from_pandas(current_df, data_definition=self._data_definition)

        report = Report(metrics=[DataDriftPreset(), DataSummaryPreset()])
        snapshot = report.run(reference_data=ref_dataset, current_data=cur_dataset)
        return snapshot

    def save_report(
        self,
        snapshot,
        output_path: str | Path = "reports/drift_report.html",
    ) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(path))
        logger.info("Drift report saved to %s", path)
        return path

    def get_drift_summary(self, snapshot) -> dict:
        """
        Extract a flat dict of drift metrics suitable for MLflow logging.

        Returns keys: n_drifted_features, share_drifted, dataset_drift_detected
        """
        result = snapshot.dict()
        metrics = result.get("metrics", [])
        summary: dict = {}

        for m in metrics:
            name = m.get("metric_name", "")
            if name.startswith("DriftedColumnsCount"):
                value = m.get("value", {})
                count = value.get("count", 0) if isinstance(value, dict) else 0
                share = value.get("share", 0) if isinstance(value, dict) else 0
                summary["n_drifted_features"] = int(count)
                summary["share_drifted"] = round(float(share), 4)
                # Dataset-level drift flagged when >50% of columns drifted
                # (matches DriftedColumnsCount's default drift_share threshold)
                summary["dataset_drift_detected"] = int(share > 0.5)

        return summary

    def log_to_mlflow(self, snapshot, run_id: str | None = None) -> None:
        """Log drift summary metrics and the HTML report as an MLflow artifact."""
        import mlflow

        from src.config import settings

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        summary = self.get_drift_summary(snapshot)

        ctx = (
            mlflow.start_run(run_id=run_id)
            if run_id
            else mlflow.start_run(
                run_name="drift-monitoring",
                experiment_id=mlflow.set_experiment(settings.mlflow_experiment_pipeline).experiment_id,
            )
        )
        with ctx:
            mlflow.log_metrics({k: float(v) for k, v in summary.items() if isinstance(v, (int, float))})
            path = self.save_report(snapshot)
            mlflow.log_artifact(str(path), artifact_path="drift_reports")

        logger.info("Drift metrics logged to MLflow: %s", summary)
