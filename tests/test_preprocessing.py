"""tests/test_preprocessing.py — Tests for cleaning, aggregation, and merging."""
import numpy as np
import pandas as pd
import pytest
from src.data.preprocessing import (
    clean_client_data, aggregate_price_data, merge_datasets, null_report
)


class TestCleanClientData:
    def test_churn_cast_to_bool(self, sample_client_df):
        df = clean_client_data(sample_client_df)
        assert df["churn"].dtype == bool

    def test_consumption_clipped_to_zero(self, sample_client_df):
        dirty = sample_client_df.copy()
        dirty["cons_12m"] = dirty["cons_12m"] - 99999   # force negatives
        df = clean_client_data(dirty)
        assert (df["cons_12m"] >= 0).all()

    def test_categorical_nulls_filled(self, sample_client_df):
        df = clean_client_data(sample_client_df)
        assert df["activity_new"].isna().sum() == 0

    def test_no_duplicate_rows(self, sample_client_df):
        duped = pd.concat([sample_client_df, sample_client_df.iloc[:5]])
        df = clean_client_data(duped)
        assert len(df) == len(sample_client_df)

    def test_has_gas_mapped_to_bool(self, sample_client_df):
        df = clean_client_data(sample_client_df)
        assert df["has_gas"].dtype == bool


class TestAggregatePriceData:
    def test_one_row_per_client(self, sample_price_df, client_ids):
        agg = aggregate_price_data(sample_price_df)
        assert len(agg) == len(client_ids)

    def test_mean_and_std_columns_created(self, sample_price_df):
        agg = aggregate_price_data(sample_price_df)
        assert "price_off_peak_var_mean" in agg.columns
        assert "price_off_peak_var_std" in agg.columns

    def test_no_nulls_in_mean_columns(self, sample_price_df):
        agg = aggregate_price_data(sample_price_df)
        mean_cols = [c for c in agg.columns if c.endswith("_mean")]
        assert agg[mean_cols].isnull().sum().sum() == 0


class TestMergeDatasets:
    def test_row_count_preserved(self, sample_client_df, sample_price_df):
        cleaned = clean_client_data(sample_client_df)
        merged = merge_datasets(cleaned, sample_price_df)
        assert len(merged) == len(cleaned)

    def test_price_columns_present(self, sample_client_df, sample_price_df):
        cleaned = clean_client_data(sample_client_df)
        merged = merge_datasets(cleaned, sample_price_df)
        assert "price_off_peak_var_mean" in merged.columns


class TestNullReport:
    def test_returns_only_null_columns(self, sample_client_df):
        df = sample_client_df.copy()
        df["synthetic_null_col"] = np.nan
        report = null_report(df)
        assert "synthetic_null_col" in report.index
        assert (report["missing_count"] > 0).all()

    def test_empty_when_no_nulls(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        assert null_report(df).empty
