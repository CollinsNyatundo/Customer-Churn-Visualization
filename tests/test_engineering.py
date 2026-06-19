"""tests/test_engineering.py — Tests for all feature engineering functions."""
import pandas as pd
import pytest
from src.features.engineering import (
    add_tenure_features, add_consumption_features,
    add_price_sensitivity, add_margin_features,
    add_multisource_features, build_feature_set,
)


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


class TestConsumptionFeatures:
    def test_growth_rate_finite(self, merged_df):
        df = add_consumption_features(merged_df)
        assert df["cons_growth_rate"].replace([float("inf"), float("-inf")], pd.NA).notna().all()

    def test_gas_share_between_0_and_1(self, merged_df):
        df = add_consumption_features(merged_df)
        valid = df["gas_share"].dropna()
        assert valid.between(0, 1).all()

    def test_high_consumption_is_bool(self, merged_df):
        df = add_consumption_features(merged_df)
        assert df["high_consumption"].dtype == bool


class TestPriceSensitivity:
    def test_price_spread_present(self, merged_df):
        df = add_price_sensitivity(merged_df)
        assert "price_spread_var" in df.columns

    def test_discount_flag_is_bool(self, merged_df):
        df = add_price_sensitivity(merged_df)
        assert df["discount_flag"].dtype == bool


class TestMarginFeatures:
    def test_margin_efficiency_present(self, merged_df):
        df = add_margin_features(merged_df)
        assert "margin_efficiency" in df.columns

    def test_low_margin_is_bool(self, merged_df):
        df = add_margin_features(merged_df)
        assert df["low_margin"].dtype == bool


class TestMultisourceFeatures:
    def test_risk_score_present(self, merged_df):
        df = add_multisource_features(merged_df)
        assert "cross_source_risk_score" in df.columns

    def test_risk_score_between_0_and_1(self, merged_df):
        df = add_multisource_features(merged_df)
        scores = df["cross_source_risk_score"].dropna()
        assert scores.between(0, 1).all()

    def test_engagement_score_present(self, merged_df):
        df = add_multisource_features(merged_df)
        assert "engagement_score" in df.columns


class TestBuildFeatureSet:
    def test_adds_all_expected_columns(self, merged_df):
        df = build_feature_set(merged_df)
        expected = [
            "contract_duration_days", "is_long_term", "months_to_renewal",
            "cons_growth_rate", "high_consumption", "price_spread_var",
            "margin_efficiency", "cross_source_risk_score",
        ]
        for col in expected:
            assert col in df.columns, f"Missing: {col}"

    def test_no_rows_dropped(self, merged_df):
        df = build_feature_set(merged_df)
        assert len(df) == len(merged_df)
