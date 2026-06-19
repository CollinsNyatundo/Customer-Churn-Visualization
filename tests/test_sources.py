"""tests/test_sources.py — Unit tests for all DataSource classes."""

import pandas as pd

from src.data.sources.billing_source import BillingSource
from src.data.sources.crm_source import CRMSource
from src.data.sources.support_source import SupportSource


class TestCRMSource:
    def test_extract_returns_correct_shape(self, client_ids):
        df, meta = CRMSource(client_ids=client_ids).load()
        assert len(df) == len(client_ids)
        assert "nps_score" in df.columns

    def test_nps_within_range(self, client_ids):
        df, _ = CRMSource(client_ids=client_ids).load()
        assert df["nps_score"].between(-100, 100).all()

    def test_satisfaction_within_range(self, client_ids):
        df, _ = CRMSource(client_ids=client_ids).load()
        assert df["satisfaction_score"].between(1, 5).all()

    def test_reproducibility_with_seed(self, client_ids):
        df1, _ = CRMSource(client_ids=client_ids, seed=42).load()
        df2, _ = CRMSource(client_ids=client_ids, seed=42).load()
        pd.testing.assert_frame_equal(df1, df2)

    def test_empty_ids_returns_empty_df(self):
        df, _ = CRMSource(client_ids=[]).load()
        assert df.empty

    def test_metadata_populated(self, client_ids):
        _, meta = CRMSource(client_ids=client_ids).load()
        assert meta.row_count == len(client_ids)
        assert meta.source_name == "crm"


class TestSupportSource:
    def test_extract_returns_correct_shape(self, client_ids):
        df, _ = SupportSource(client_ids=client_ids).load()
        assert len(df) == len(client_ids)

    def test_escalations_never_exceed_tickets(self, client_ids):
        df, _ = SupportSource(client_ids=client_ids).load()
        assert (df["escalations_6m"] <= df["num_tickets_6m"]).all()

    def test_resolution_hours_non_negative(self, client_ids):
        df, _ = SupportSource(client_ids=client_ids).load()
        assert (df["avg_resolution_hours"] >= 0).all()


class TestBillingSource:
    def test_extract_returns_correct_shape(self, client_ids):
        df, _ = BillingSource(client_ids=client_ids).load()
        assert len(df) == len(client_ids)

    def test_no_negative_outstanding(self, client_ids):
        df, _ = BillingSource(client_ids=client_ids).load()
        assert (df["total_outstanding"] >= 0).all()

    def test_valid_payment_methods(self, client_ids):
        from src.data.sources.billing_source import PAYMENT_METHODS

        df, _ = BillingSource(client_ids=client_ids).load()
        assert set(df["payment_method"]).issubset(set(PAYMENT_METHODS))

    def test_discount_pct_range(self, client_ids):
        df, _ = BillingSource(client_ids=client_ids).load()
        assert df["discount_pct"].between(0, 100).all()
