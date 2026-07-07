"""
tests/test_performance.py
----------------------------
Real functional tests for PerformanceMonitor (src/monitoring/performance.py).
This module had zero test coverage — the compute_metrics()/alert_if_degraded()
logic (window filtering, threshold comparisons, degradation detection) was
never actually exercised end-to-end against real prediction logs.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from src.monitoring.performance import PerformanceMonitor


@pytest.fixture()
def monitor(tmp_path):
    return PerformanceMonitor(log_path=tmp_path / "prediction_log.jsonl")


class TestRecordPrediction:
    def test_record_prediction_creates_log_file(self, monitor):
        monitor.record_prediction("cust_1", 0.75, actual_churn=True)
        assert monitor.log_path.exists()

    def test_record_prediction_appends_multiple_rows(self, monitor):
        for i in range(5):
            monitor.record_prediction(f"cust_{i}", 0.5, actual_churn=bool(i % 2))
        lines = monitor.log_path.read_text().splitlines()
        assert len(lines) == 5


class TestLoadWindow:
    def test_excludes_predictions_without_outcomes(self, monitor):
        monitor.record_prediction("cust_1", 0.5, actual_churn=None)
        monitor.record_prediction("cust_2", 0.5, actual_churn=True)
        df = monitor.load_window(window_days=30)
        assert len(df) == 1
        assert df.iloc[0]["customer_id"] == "cust_2"

    def test_returns_empty_dataframe_when_no_log_exists(self, tmp_path):
        monitor = PerformanceMonitor(log_path=tmp_path / "nonexistent.jsonl")
        df = monitor.load_window()
        assert df.empty


class TestComputeMetrics:
    def _populate_realistic_predictions(self, monitor, n=100, seed=42):
        """
        Populate with predictions that have real discriminative signal
        (high prob -> more likely to actually churn) so AUC/precision/
        recall come out non-degenerate rather than exactly 0.5/undefined.
        """
        rng = np.random.default_rng(seed)
        for i in range(n):
            true_churn = rng.random() < 0.3
            # Predicted probability correlates with the true label
            prob = rng.uniform(0.6, 0.95) if true_churn else rng.uniform(0.05, 0.4)
            monitor.record_prediction(f"cust_{i}", prob, actual_churn=bool(true_churn))

    def test_insufficient_data_flagged(self, monitor):
        """Fewer than 30 labeled predictions should be flagged as insufficient."""
        for i in range(10):
            monitor.record_prediction(f"cust_{i}", 0.5, actual_churn=True)
        metrics = monitor.compute_metrics()
        assert metrics.get("insufficient_data") is True

    def test_sufficient_data_computes_real_metrics(self, monitor):
        self._populate_realistic_predictions(monitor, n=100)
        metrics = monitor.compute_metrics(window_days=30)
        assert "insufficient_data" not in metrics
        assert 0 <= metrics["auc"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert metrics["brier"] >= 0
        assert metrics["n"] == 100

    def test_good_predictions_yield_high_auc(self, monitor):
        """Strongly discriminative predictions should score a high AUC."""
        self._populate_realistic_predictions(monitor, n=200)
        metrics = monitor.compute_metrics()
        assert metrics["auc"] > 0.8


class TestAlertIfDegraded:
    def test_no_alerts_on_good_metrics(self, monitor):
        good_metrics = {"auc": 0.95, "precision": 0.8, "recall": 0.9, "brier": 0.05}
        alerts = monitor.alert_if_degraded(good_metrics)
        assert alerts == []

    def test_alert_fires_on_low_auc(self, monitor):
        bad_metrics = {"auc": 0.60, "precision": 0.8, "recall": 0.9, "brier": 0.05}
        alerts = monitor.alert_if_degraded(bad_metrics)
        assert any("AUC" in a for a in alerts)

    def test_alert_fires_on_low_recall(self, monitor):
        bad_metrics = {"auc": 0.95, "precision": 0.8, "recall": 0.5, "brier": 0.05}
        alerts = monitor.alert_if_degraded(bad_metrics)
        assert any("Recall" in a for a in alerts)

    def test_alert_fires_on_high_brier(self, monitor):
        bad_metrics = {"auc": 0.95, "precision": 0.8, "recall": 0.9, "brier": 0.30}
        alerts = monitor.alert_if_degraded(bad_metrics)
        assert any("Brier" in a for a in alerts)

    def test_no_alert_on_insufficient_data(self, monitor):
        """Should not fire spurious alerts when there isn't enough data to trust."""
        alerts = monitor.alert_if_degraded({"n": 5, "insufficient_data": True})
        assert alerts == []

    def test_alert_uses_generic_alertmanager_send(self, monitor, monkeypatch):
        """
        This exact call pattern (AlertManager().send(title=..., body=...))
        previously raised AttributeError because AlertManager had no
        generic send() method — only typed methods like send_drift_alert().
        This test locks in that the integration point stays fixed.
        """
        from unittest.mock import MagicMock

        mock_manager = MagicMock()
        monkeypatch.setattr("src.monitoring.alerts.AlertManager", lambda: mock_manager)

        bad_metrics = {"auc": 0.50, "precision": 0.3, "recall": 0.4, "brier": 0.40}
        monitor.alert_if_degraded(bad_metrics)

        mock_manager.send.assert_called_once()
