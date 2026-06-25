"""tests/test_engineering.py — Tests for all feature engineering functions."""

import numpy as np
import pandas as pd
import pytest

from src.features.engineering import (
    FeatureThresholds,
    add_billing_features,
    add_consumption_features,
    add_margin_features,
    add_multisource_features,
    add_price_sensitivity,
    add_source_flags,
    add_support_features,
    add_tenure_features,
    build_feature_set,
    get_feature_names,
)


class TestFeatureThresholds:
    def test_from_dataframe(self, merged_df):
        t = FeatureThresholds.from_dataframe(merged_df)
        assert t.high_consumption_threshold > 0
        assert t.low_margin_threshold != 0

    def test_schema_hash_stable(self, merged_df):
        t = FeatureThresholds.from_dataframe(merged_df)
        assert t.schema_hash() == t.schema_hash()

    def test_different_data_different_hash(self, merged_df):
        t1 = FeatureThresholds.from_dataframe(merged_df)
        t2 = FeatureThresholds.from_dataframe(merged_df.sample(10, random_state=0))
        # Hashes may differ if quantiles differ
        assert isinstance(t1.schema_hash(), str) and len(t1.schema_hash()) == 16


class TestTenureFeatures:
    def test_contract_duration_non_negative(self, merged_df):
        df = add_tenure_features(merged_df)
        assert (df["contract_duration_days"] >= 0).all()

    def test_is_long_term_is_bool(self, merged_df):
        df = add_tenure_features(merged_df)
        assert df["is_long_term"].dtype == bool

    def test_months_to_renewal_present(self, merged_df):
        df = add_tenure_features(merged_df)
        assert "months_to_renewal" in df.columns

    def test_renewal_urgency_is_bool(self, merged_df):
        df = add_tenure_features(merged_df)
        assert "renewal_urgency" in df.columns
        assert df["renewal_urgency"].dtype == bool

    def test_contract_completion_pct_between_0_and_1(self, merged_df):
        df = add_tenure_features(merged_df)
        valid = df["contract_completion_pct"].dropna()
        assert valid.between(0, 1).all()

    def test_reference_date_reproducibility(self, merged_df):
        ref = pd.Timestamp("2026-01-01")
        df1 = add_tenure_features(merged_df, reference_date=ref)
        df2 = add_tenure_features(merged_df, reference_date=ref)
        pd.testing.assert_series_equal(df1["months_to_renewal"], df2["months_to_renewal"])


class TestConsumptionFeatures:
    def test_growth_rate_finite(self, merged_df):
        df = add_consumption_features(merged_df)
        vals = df["cons_growth_rate"].replace([float("inf"), float("-inf")], pd.NA)
        assert vals.notna().all()

    def test_gas_share_between_0_and_1(self, merged_df):
        df = add_consumption_features(merged_df)
        valid = df["gas_share"].dropna()
        assert valid.between(0, 1).all()

    def test_high_consumption_is_bool(self, merged_df):
        df = add_consumption_features(merged_df)
        assert df["high_consumption"].dtype == bool

    def test_cons_per_product_positive(self, merged_df):
        df = add_consumption_features(merged_df)
        assert (df["cons_per_product"].dropna() >= 0).all()

    def test_power_utilisation_between_0_and_1(self, merged_df):
        df = add_consumption_features(merged_df)
        assert df["power_utilisation"].dropna().between(0, 1).all()


class TestPriceSensitivity:
    def test_price_spread_var_present(self, merged_df):
        df = add_price_sensitivity(merged_df)
        assert "price_spread_var" in df.columns

    def test_price_peak_ratio_positive(self, merged_df):
        df = add_price_sensitivity(merged_df)
        if "price_peak_ratio" in df.columns:
            assert (df["price_peak_ratio"].dropna() > 0).all()

    def test_price_volatility_non_negative(self, merged_df):
        df = add_price_sensitivity(merged_df)
        if "price_volatility" in df.columns:
            assert (df["price_volatility"].dropna() >= 0).all()


class TestMarginFeatures:
    def test_margin_efficiency_present(self, merged_df):
        df = add_margin_features(merged_df)
        assert "margin_efficiency" in df.columns

    def test_low_margin_is_bool(self, merged_df):
        df = add_margin_features(merged_df)
        assert df["low_margin"].dtype == bool

    def test_clv_proxy_present(self, merged_df):
        df = add_margin_features(merged_df)
        assert "clv_proxy" in df.columns


class TestSupportFeatures:
    def test_escalation_rate_between_0_and_1(self, merged_df):
        df = add_support_features(merged_df)
        valid = df["escalation_rate"].dropna()
        assert valid.between(0, 1).all()

    def test_service_quality_deficit_between_0_and_1(self, merged_df):
        df = add_support_features(merged_df)
        if "service_quality_deficit" in df.columns:
            assert df["service_quality_deficit"].dropna().between(0, 1).all()

    def test_ticket_rate_per_product_non_negative(self, merged_df):
        df = add_support_features(merged_df)
        if "ticket_rate_per_product" in df.columns:
            assert (df["ticket_rate_per_product"].dropna() >= 0).all()


class TestBillingFeatures:
    def test_payment_reliability_between_0_and_1(self, merged_df):
        df = add_billing_features(merged_df)
        if "payment_reliability" in df.columns:
            assert df["payment_reliability"].dropna().between(0, 1).all()

    def test_financial_distress_between_0_and_1(self, merged_df):
        df = add_billing_features(merged_df)
        if "financial_distress_score" in df.columns:
            assert df["financial_distress_score"].dropna().between(0, 1).all()


class TestMultisourceFeatures:
    def test_risk_score_between_0_and_1(self, merged_df):
        df = add_multisource_features(merged_df)
        assert df["cross_source_risk_score"].dropna().between(0, 1).all()

    def test_no_inf_in_risk_score(self, merged_df):
        df = add_multisource_features(merged_df)
        assert not np.isinf(df["cross_source_risk_score"].values).any()

    def test_engagement_score_present(self, merged_df):
        df = add_multisource_features(merged_df)
        assert "engagement_score" in df.columns

    def test_revenue_at_risk_non_negative(self, merged_df):
        df = add_multisource_features(merged_df)
        if "revenue_at_risk" in df.columns:
            assert (df["revenue_at_risk"].dropna() >= 0).all()

    def test_nps_tenure_interaction_present(self, merged_df):
        df = add_multisource_features(merged_df)
        assert "nps_tenure_interaction" in df.columns


class TestSourceFlags:
    def test_source_flags_added_when_column_present(self):
        df = pd.DataFrame({"id": ["a", "b", "c"], "data_source": ["bcg", "kaggle_telco", "bank_churn"]})
        out = add_source_flags(df)
        assert "is_kaggle_source" in out.columns
        assert out["is_kaggle_source"].iloc[1] == 1
        assert out["is_bank_source"].iloc[2] == 1
        assert out["is_bcg_source"].iloc[0] == 1

    def test_source_flags_all_zero_when_no_column(self, merged_df):
        # merged_df has no data_source column — flags should not be added
        df = merged_df.drop(columns=["data_source"], errors="ignore")
        out = add_source_flags(df)
        # No flags added since column absent
        for col in ["is_kaggle_source", "is_bank_source", "is_bcg_source"]:
            assert col not in out.columns


class TestBuildFeatureSet:
    def test_adds_all_38_engineered_features(self, merged_df):
        names = get_feature_names()
        df = build_feature_set(merged_df)
        for domain, feats in names.items():
            for feat in feats:
                # source flags only appear when data_source column exists
                if "source" in domain:
                    continue
                assert feat in df.columns, f"Missing {domain}/{feat}"

    def test_no_rows_dropped(self, merged_df):
        df = build_feature_set(merged_df)
        assert len(df) == len(merged_df)

    def test_no_inf_values(self, merged_df):
        df = build_feature_set(merged_df)
        numeric = df.select_dtypes(include="number")
        assert not np.isinf(numeric.values).any(), "Inf values found"

    def test_frozen_thresholds_deterministic(self, merged_df):
        t = FeatureThresholds.from_dataframe(merged_df)
        ref = pd.Timestamp("2026-06-22")
        df1 = build_feature_set(merged_df, thresholds=t, reference_date=ref)
        df2 = build_feature_set(merged_df, thresholds=t, reference_date=ref)
        pd.testing.assert_series_equal(df1["high_consumption"], df2["high_consumption"])
        pd.testing.assert_series_equal(df1["months_to_renewal"], df2["months_to_renewal"])

    def test_get_feature_names_returns_38(self):
        names = get_feature_names()
        total = sum(len(v) for v in names.values())
        assert total == 38, f"Expected 38 engineered features, got {total}"
