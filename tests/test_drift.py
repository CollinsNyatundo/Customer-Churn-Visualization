"""
tests/test_drift.py
---------------------
Real functional tests for DriftDetector — no mocking of Evidently itself.

This module previously had zero test coverage under either the old
(evidently==0.4.30) or new (evidently==0.7.x) API, meaning a breaking
dependency version bump could silently make drift detection non-functional
without any test ever catching it. These tests exercise the real
Report.run() call end-to-end against real (small, synthetic) DataFrames.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift import DriftDetector


@pytest.fixture()
def reference_df():
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "cons_12m": rng.exponential(10000, 200),
            "net_margin": rng.normal(200, 80, 200),
            "nps_score": rng.uniform(-100, 100, 200),
            "channel_sales": rng.choice(["CH01", "CH02", "CH03"], 200),
            "churn": rng.integers(0, 2, 200),
        }
    )


@pytest.fixture()
def similar_current_df():
    """Same distribution as reference — should show minimal/no drift."""
    rng = np.random.default_rng(99)
    return pd.DataFrame(
        {
            "cons_12m": rng.exponential(10000, 200),
            "net_margin": rng.normal(200, 80, 200),
            "nps_score": rng.uniform(-100, 100, 200),
            "channel_sales": rng.choice(["CH01", "CH02", "CH03"], 200),
            "churn": rng.integers(0, 2, 200),
        }
    )


@pytest.fixture()
def drifted_current_df():
    """Deliberately shifted distribution on cons_12m — should be flagged."""
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "cons_12m": rng.exponential(40000, 200),  # 4x scale shift
            "net_margin": rng.normal(200, 80, 200),
            "nps_score": rng.uniform(-100, 100, 200),
            "channel_sales": rng.choice(["CH01", "CH02", "CH03"], 200),
            "churn": rng.integers(0, 2, 200),
        }
    )


class TestDriftDetectorReal:
    def test_run_returns_snapshot(self, reference_df, similar_current_df):
        detector = DriftDetector(reference_df)
        snapshot = detector.run(similar_current_df)
        assert snapshot is not None
        assert hasattr(snapshot, "dict")

    def test_get_drift_summary_has_expected_keys(self, reference_df, similar_current_df):
        detector = DriftDetector(reference_df)
        snapshot = detector.run(similar_current_df)
        summary = detector.get_drift_summary(snapshot)
        assert "n_drifted_features" in summary
        assert "share_drifted" in summary
        assert "dataset_drift_detected" in summary

    def test_detects_genuine_drift(self, reference_df, drifted_current_df):
        """A deliberately shifted column should be flagged as drifted."""
        detector = DriftDetector(reference_df)
        snapshot = detector.run(drifted_current_df)
        summary = detector.get_drift_summary(snapshot)
        assert summary["n_drifted_features"] >= 1, "Deliberately drifted column was not detected"

    def test_similar_data_shows_low_drift(self, reference_df, similar_current_df):
        """Same-distribution data should not show significant drift."""
        detector = DriftDetector(reference_df)
        snapshot = detector.run(similar_current_df)
        summary = detector.get_drift_summary(snapshot)
        assert summary["share_drifted"] < 0.5

    def test_save_report_writes_html(self, reference_df, similar_current_df, tmp_path):
        detector = DriftDetector(reference_df)
        snapshot = detector.run(similar_current_df)
        output = tmp_path / "drift_report.html"
        path = detector.save_report(snapshot, output)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_handles_missing_optional_columns(self, similar_current_df):
        """Reference with only a subset of known columns should not crash."""
        minimal_ref = pd.DataFrame({"cons_12m": [1000, 2000, 3000], "churn": [0, 1, 0]})
        minimal_cur = pd.DataFrame({"cons_12m": [1100, 2100, 3100], "churn": [0, 1, 0]})
        detector = DriftDetector(minimal_ref)
        snapshot = detector.run(minimal_cur)
        summary = detector.get_drift_summary(snapshot)
        assert "n_drifted_features" in summary
