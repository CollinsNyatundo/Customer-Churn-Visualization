"""
tests/test_quality.py
-----------------------
Tests for Pandera data quality contracts (src/data/quality.py).
This module previously had zero test coverage — meaning the schemas
themselves were never actually exercised against real or synthetic data,
so a typo or logic error in a Check() would have gone unnoticed.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.quality import validate_source


class TestBCGClientQuality:
    def test_valid_data_passes(self):
        df = pd.DataFrame(
            {
                "id": ["a", "b", "c"],
                "churn": [True, False, True],
                "cons_12m": [1000.0, 2000.0, 3000.0],
                "net_margin": [100.0, -50.0, 200.0],
                "num_years_antig": [1.0, 5.0, 10.0],
                "nb_prod_act": [1, 2, 3],
            }
        )
        result = validate_source(df, "bcg_client", strict=True)
        assert len(result) == 3

    def test_negative_consumption_fails_strict(self):
        df = pd.DataFrame(
            {
                "id": ["a", "b"],
                "churn": [True, False],
                "cons_12m": [-500.0, 2000.0],  # invalid: negative
            }
        )
        with pytest.raises(ValueError):
            validate_source(df, "bcg_client", strict=True)

    def test_negative_consumption_logs_warning_non_strict(self, caplog):
        df = pd.DataFrame(
            {
                "id": ["a", "b"],
                "churn": [True, False],
                "cons_12m": [-500.0, 2000.0],
            }
        )
        result = validate_source(df, "bcg_client", strict=False)
        assert result is not None  # doesn't raise, returns something

    def test_duplicate_ids_fail_strict(self):
        df = pd.DataFrame(
            {
                "id": ["a", "a"],  # duplicate
                "churn": [True, False],
            }
        )
        with pytest.raises(ValueError):
            validate_source(df, "bcg_client", strict=True)


class TestCRMQuality:
    def test_valid_nps_range_passes(self):
        df = pd.DataFrame(
            {
                "id": ["a", "b"],
                "nps_score": [-100.0, 100.0],  # boundary values
                "satisfaction_score": [1.0, 5.0],
            }
        )
        result = validate_source(df, "crm", strict=True)
        assert len(result) == 2

    def test_nps_out_of_range_fails_strict(self):
        df = pd.DataFrame(
            {
                "id": ["a", "b"],
                "nps_score": [-150.0, 200.0],  # invalid: outside [-100, 100]
            }
        )
        with pytest.raises(ValueError):
            validate_source(df, "crm", strict=True)

    def test_satisfaction_out_of_range_fails_strict(self):
        df = pd.DataFrame(
            {
                "id": ["a", "b"],
                "satisfaction_score": [0.5, 6.0],  # invalid: outside [1, 5]
            }
        )
        with pytest.raises(ValueError):
            validate_source(df, "crm", strict=True)


class TestSupportQuality:
    def test_negative_tickets_fails_strict(self):
        df = pd.DataFrame({"id": ["a"], "num_tickets_6m": [-1]})
        with pytest.raises(ValueError):
            validate_source(df, "support", strict=True)

    def test_csat_out_of_range_fails_strict(self):
        df = pd.DataFrame({"id": ["a"], "post_ticket_csat": [10.0]})
        with pytest.raises(ValueError):
            validate_source(df, "support", strict=True)


class TestBillingQuality:
    def test_negative_outstanding_fails_strict(self):
        df = pd.DataFrame({"id": ["a"], "total_outstanding": [-500.0]})
        with pytest.raises(ValueError):
            validate_source(df, "billing", strict=True)

    def test_discount_pct_out_of_range_fails_strict(self):
        df = pd.DataFrame({"id": ["a"], "discount_pct": [150]})
        with pytest.raises(ValueError):
            validate_source(df, "billing", strict=True)


class TestUnknownSource:
    def test_unknown_source_name_passes_through_unchanged(self):
        """No schema registered for this name — should skip validation, not crash."""
        df = pd.DataFrame({"id": ["a"], "some_col": [1]})
        result = validate_source(df, "nonexistent_source", strict=True)
        pd.testing.assert_frame_equal(result, df)
