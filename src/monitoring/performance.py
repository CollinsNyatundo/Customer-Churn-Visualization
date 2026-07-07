"""
src/monitoring/performance.py
-------------------------------
Rolling model performance monitoring.

Tracks precision, recall, F1, Brier score, and AUC over a sliding
window of recent predictions with observed outcomes. Alerts when
performance degrades below configurable thresholds.

Usage (in a Prefect flow or cron):
    from src.monitoring.performance import PerformanceMonitor
    monitor = PerformanceMonitor()
    monitor.record_prediction(customer_id, prob, actual_churn)
    report = monitor.compute_metrics(window_days=30)
    monitor.alert_if_degraded(report)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

_DEFAULT_LOG = Path("reports/prediction_log.jsonl")
_ALERT_THRESHOLDS = {
    "auc": 0.90,  # alert if AUC drops below this
    "precision": 0.50,
    "recall": 0.80,  # high recall is critical for churn
    "brier": 0.10,  # alert if calibration degrades
}


class PerformanceMonitor:
    """
    Append-only log of (prediction, outcome) pairs.
    Computes rolling metrics and fires alerts on degradation.
    """

    def __init__(self, log_path: Path = _DEFAULT_LOG):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def record_prediction(
        self,
        customer_id: str,
        churn_probability: float,
        actual_churn: bool | None = None,
        model_version: str = "unknown",
    ) -> None:
        """Append one prediction record. actual_churn filled in later."""
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "customer_id": customer_id,
            "prob": round(churn_probability, 6),
            "actual": actual_churn,
            "version": model_version,
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def record_outcome(self, customer_id: str, actual_churn: bool) -> None:
        """Update the most recent prediction for a customer with its outcome."""
        if not self.log_path.exists():
            return
        lines = self.log_path.read_text().splitlines()
        updated = []
        found = False
        for line in reversed(lines):
            if not found:
                r = json.loads(line)
                if r["customer_id"] == customer_id and r["actual"] is None:
                    r["actual"] = actual_churn
                    line = json.dumps(r)
                    found = True
            updated.append(line)
        self.log_path.write_text("\n".join(reversed(updated)) + "\n")

    def load_window(self, window_days: int = 30) -> pd.DataFrame:
        """Load predictions from the past N days that have observed outcomes."""
        if not self.log_path.exists():
            return pd.DataFrame()
        records = [json.loads(line) for line in self.log_path.read_text().splitlines() if line]
        df = pd.DataFrame(records)
        if df.empty:
            return df
        df["ts"] = pd.to_datetime(df["ts"])
        cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
        df = df[df["ts"] >= cutoff]
        # Only rows with observed outcomes
        return df[df["actual"].notna()].copy()

    def compute_metrics(self, window_days: int = 30) -> dict[str, Any]:
        """Compute precision, recall, AUC, Brier for the rolling window."""
        df = self.load_window(window_days)
        if len(df) < 30:
            log.warning(
                "Only %d labeled predictions in window — metrics may be unreliable.",
                len(df),
            )
            return {"n": len(df), "insufficient_data": True}

        from sklearn.metrics import (
            average_precision_score,
            brier_score_loss,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        y_true = df["actual"].astype(int).values
        y_prob = df["prob"].values
        y_pred = (y_prob >= 0.5).astype(int)

        metrics: dict[str, Any] = {
            "n": int(len(df)),
            "window_days": window_days,
            "auc": round(float(roc_auc_score(y_true, y_prob)), 4),
            "avg_prec": round(float(average_precision_score(y_true, y_prob)), 4),
            "brier": round(float(brier_score_loss(y_true, y_prob)), 4),
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            "churn_rate": round(float(y_true.mean()), 4),
            "computed_at": datetime.now(timezone.utc).isoformat(),
        }
        log.info("Performance metrics (window=%dd): %s", window_days, metrics)
        return metrics

    def alert_if_degraded(
        self,
        metrics: dict[str, Any],
        thresholds: dict[str, float] | None = None,
    ) -> list[str]:
        """
        Check metrics against thresholds. Returns list of alert messages.
        Empty list = all good.
        """
        if metrics.get("insufficient_data"):
            return []

        t = thresholds or _ALERT_THRESHOLDS
        alerts = []

        if metrics.get("auc", 1.0) < t.get("auc", 0.90):
            alerts.append(f"⚠️ AUC degraded: {metrics['auc']:.4f} < threshold {t['auc']:.2f}")
        if metrics.get("precision", 1.0) < t.get("precision", 0.50):
            alerts.append(f"⚠️ Precision degraded: {metrics['precision']:.4f} < {t['precision']:.2f}")
        if metrics.get("recall", 1.0) < t.get("recall", 0.80):
            alerts.append(f"⚠️ Recall degraded: {metrics['recall']:.4f} < {t['recall']:.2f}")
        if metrics.get("brier", 0.0) > t.get("brier", 0.10):
            alerts.append(f"⚠️ Brier score degraded: {metrics['brier']:.4f} > {t['brier']:.2f}")

        if alerts:
            log.error("Model performance degradation detected:\n%s", "\n".join(alerts))
            # Fire alert via existing AlertManager
            try:
                from src.monitoring.alerts import AlertManager

                AlertManager().send(
                    title="⚠️ Churn Model Performance Degraded",
                    body="\n".join(alerts),
                )
            except Exception as e:
                log.warning("Alert send failed: %s", e)

        return alerts
