"""
tests/test_feature_store.py
----------------------------
Tests for the Feast feature store integration.

Tests verify:
  - Parquet sink writes correct shapes and required columns
  - Feast FeatureStore can retrieve online features
  - Historical retrieval returns point-in-time correct data
  - feast_sink handles missing columns gracefully
"""

from __future__ import annotations

from datetime import timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def sample_featured_df():
    """500-row fully-featured DataFrame."""
    from src.data.preprocessing import aggregate_price_data, clean_client_data
    from src.data.sources.bcg_source import BCGClientSource, BCGPriceSource
    from src.data.sources.billing_source import BillingSource
    from src.data.sources.crm_source import CRMSource
    from src.data.sources.support_source import SupportSource
    from src.features.engineering import FeatureThresholds, build_feature_set

    client_df, _ = BCGClientSource().load()
    price_df, _ = BCGPriceSource().load()
    cleaned = clean_client_data(client_df.head(500))
    price_agg = aggregate_price_data(price_df[price_df["id"].isin(cleaned["id"])])
    ids = cleaned["id"].tolist()

    crm_df, _ = CRMSource(client_ids=ids).load()
    sup_df, _ = SupportSource(client_ids=ids).load()
    bill_df, _ = BillingSource(client_ids=ids).load()

    merged = (
        cleaned.merge(price_agg, on="id", how="left")
        .merge(crm_df, on="id", how="left")
        .merge(sup_df, on="id", how="left")
        .merge(bill_df, on="id", how="left")
    )

    return build_feature_set(merged, thresholds=FeatureThresholds.from_dataframe(merged))


# ── Feast sink tests ──────────────────────────────────────────────────────────


class TestFeastSink:
    def test_writes_all_six_groups(self, sample_featured_df, tmp_path):
        import src.data.feast_sink as sink_module
        from src.data.feast_sink import (
            BCG_COLS,
            BILLING_COLS,
            CRM_COLS,
            ENGINEERED_COLS,
            FEAST_DATA_DIR,
            PRICE_COLS,
            SUPPORT_COLS,
            write_feast_parquet,
        )

        original = sink_module.FEAST_DATA_DIR

        try:
            sink_module.FEAST_DATA_DIR = tmp_path
            written = write_feast_parquet(sample_featured_df)
            assert len(written) == 6
            expected = {
                "bcg_features",
                "price_features",
                "crm_features",
                "support_features",
                "billing_features",
                "engineered_features",
            }
            assert set(written.keys()) == expected
        finally:
            sink_module.FEAST_DATA_DIR = original

    def test_parquet_has_required_feast_columns(self, sample_featured_df, tmp_path):
        import src.data.feast_sink as sink_module
        from src.data.feast_sink import write_feast_parquet

        original = sink_module.FEAST_DATA_DIR
        try:
            sink_module.FEAST_DATA_DIR = tmp_path
            written = write_feast_parquet(sample_featured_df)
            for name, path in written.items():
                df = pd.read_parquet(path)
                assert "customer_id" in df.columns, f"{name}: missing customer_id"
                assert "event_timestamp" in df.columns, f"{name}: missing event_timestamp"
                assert "created" in df.columns, f"{name}: missing created"
        finally:
            sink_module.FEAST_DATA_DIR = original

    def test_row_count_matches_input(self, sample_featured_df, tmp_path):
        import src.data.feast_sink as sink_module
        from src.data.feast_sink import write_feast_parquet

        original = sink_module.FEAST_DATA_DIR
        try:
            sink_module.FEAST_DATA_DIR = tmp_path
            written = write_feast_parquet(sample_featured_df)
            for name, path in written.items():
                df = pd.read_parquet(path)
                assert len(df) == len(sample_featured_df), f"{name}: row count mismatch"
        finally:
            sink_module.FEAST_DATA_DIR = original

    def test_handles_missing_optional_columns(self, sample_featured_df, tmp_path):
        """Sink should write what's available, not crash on missing columns."""
        import src.data.feast_sink as sink_module
        from src.data.feast_sink import write_feast_parquet

        original = sink_module.FEAST_DATA_DIR
        try:
            sink_module.FEAST_DATA_DIR = tmp_path
            df_missing = sample_featured_df.drop(columns=["nps_score"], errors="ignore")
            written = write_feast_parquet(df_missing)
            assert len(written) == 6  # still writes all groups
        finally:
            sink_module.FEAST_DATA_DIR = original

    def test_event_timestamp_is_utc(self, sample_featured_df, tmp_path):
        import src.data.feast_sink as sink_module
        from src.data.feast_sink import write_feast_parquet

        original = sink_module.FEAST_DATA_DIR
        try:
            sink_module.FEAST_DATA_DIR = tmp_path
            written = write_feast_parquet(sample_featured_df)
            df = pd.read_parquet(list(written.values())[0])
            ts = df["event_timestamp"].iloc[0]
            assert ts.tzinfo is not None, "event_timestamp must be timezone-aware"
        finally:
            sink_module.FEAST_DATA_DIR = original


# ── Feast store tests (requires materialized store) ───────────────────────────


class TestFeastOnlineStore:
    """
    These tests require the Feast store to be materialised.
    Run `python feature_store/materialize.py` before running these tests.
    They are skipped automatically if the online store doesn't exist.
    """

    @pytest.fixture(autouse=True)
    def require_store(self):
        online_store = Path("feature_store/online_store.db")
        if not online_store.exists():
            pytest.skip("Feast online store not materialised — run: python feature_store/materialize.py")

    def test_online_retrieval_returns_dataframe(self):
        from src.data.feast_store import get_online_features

        df = get_online_features(["CL00001", "CL00002"])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_online_retrieval_has_expected_features(self):
        from src.data.feast_store import get_online_features

        df = get_online_features(["CL00001"])
        # Core features from each source group should be present
        for col in ["nps_score", "num_tickets_6m", "num_late_payments_12m", "cross_source_risk_score", "cons_12m"]:
            assert col in df.columns, f"Missing expected feature: {col}"

    def test_online_retrieval_values_in_range(self):
        from src.data.feast_store import get_online_features

        df = get_online_features(["CL00001"])
        assert df["nps_score"].iloc[0] >= -100
        assert df["nps_score"].iloc[0] <= 100
        assert df["cross_source_risk_score"].iloc[0] >= 0

    def test_historical_retrieval_returns_training_data(self):
        from src.data.feast_store import get_training_data

        entity_df = pd.DataFrame(
            {
                "customer_id": ["CL00001", "CL00002", "CL00003"],
                "event_timestamp": pd.Timestamp.now(tz=timezone.utc),
            }
        )
        df = get_training_data(entity_df)
        assert len(df) == 3
        assert "nps_score" in df.columns
        assert "cross_source_risk_score" in df.columns

    def test_historical_and_online_consistent(self):
        """Same customer should have consistent feature values from both paths."""
        from src.data.feast_store import get_online_features, get_training_data

        cid = "CL00005"
        online = get_online_features([cid])
        hist = get_training_data(
            pd.DataFrame(
                {
                    "customer_id": [cid],
                    "event_timestamp": pd.Timestamp.now(tz=timezone.utc),
                }
            )
        )
        # nps_score should match (within float precision)
        online_nps = float(online["nps_score"].iloc[0])
        hist_nps = float(hist["nps_score"].iloc[0])
        assert (
            abs(online_nps - hist_nps) < 0.01
        ), f"Online ({online_nps}) and historical ({hist_nps}) nps_score mismatch for {cid}"
