"""
tests/benchmarks/test_performance.py
--------------------------------------
pytest-benchmark performance tests for the core prediction path.

Run with:
    make benchmark
    pytest tests/benchmarks/ --benchmark-only -v
"""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import aggregate_price_data, clean_client_data
from src.data.sources.billing_source import BillingSource
from src.data.sources.crm_source import CRMSource
from src.data.sources.support_source import SupportSource
from src.features.engineering import build_feature_set

# ── Shared fixtures ───────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def small_df():
    """50-row dataset — simulates single-customer and small-batch scenarios."""
    return _make_df(50)


@pytest.fixture(scope="module")
def large_df():
    """5000-row dataset — simulates batch scoring."""
    return _make_df(5000)


def _make_df(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    ids = [f"c{i:05d}" for i in range(n)]
    client = pd.DataFrame(
        {
            "id": ids,
            "churn": rng.choice([0, 1], n, p=[0.85, 0.15]),
            "cons_12m": rng.exponential(10000, n).clip(0),
            "cons_gas_12m": rng.exponential(2000, n).clip(0),
            "cons_last_month": rng.exponential(900, n).clip(0),
            "imp_cons": rng.exponential(500, n).clip(0),
            "net_margin": rng.normal(200, 80, n),
            "margin_gross_pow_ele": rng.normal(150, 60, n),
            "num_years_antig": rng.integers(1, 15, n).astype(float),
            "pow_max": rng.exponential(40, n).clip(1),
            "nb_prod_act": rng.integers(1, 5, n),
            "forecast_discount_energy": rng.uniform(0, 0.3, n),
            "activity_new": rng.choice(["a", "b", "c"], n),
            "channel_sales": rng.choice(["online", "phone", "agent"], n),
            "origin_up": rng.choice(["camp_a", "camp_b"], n),
            "has_gas": rng.choice(["t", "f"], n),
            "date_activ": pd.date_range("2015-01-01", periods=n, freq="6h"),
            "date_end": pd.date_range("2026-01-01", periods=n, freq="6h"),
            "date_modif_prod": pd.date_range("2023-01-01", periods=n, freq="8h"),
            "date_renewal": pd.date_range("2026-06-01", periods=n, freq="4h"),
        }
    )
    cleaned = clean_client_data(client)
    crm_df, _ = CRMSource(client_ids=ids).load()
    sup_df, _ = SupportSource(client_ids=ids).load()
    bill_df, _ = BillingSource(client_ids=ids).load()
    return (
        cleaned.merge(crm_df, on="id", how="left")
        .merge(sup_df, on="id", how="left")
        .merge(bill_df, on="id", how="left")
    )


# ── Benchmarks ────────────────────────────────────────────────────────────────


def test_feature_engineering_50_rows(benchmark, small_df):
    """Feature engineering on 50 rows — simulates real-time API path."""
    result = benchmark(build_feature_set, small_df)
    assert len(result) == 50
    assert "cross_source_risk_score" in result.columns


def test_feature_engineering_5000_rows(benchmark, large_df):
    """Feature engineering on 5000 rows — simulates batch scoring."""
    result = benchmark(build_feature_set, large_df)
    assert len(result) == 5000


def test_clean_client_data(benchmark, small_df):
    """Cleaning step performance."""
    result = benchmark(clean_client_data, small_df)
    assert result["churn"].dtype == bool


def test_crm_source_extraction(benchmark):
    """CRM synthetic data generation for 1000 customers."""
    ids = [f"c{i:04d}" for i in range(1000)]
    src = CRMSource(client_ids=ids)
    df, _ = benchmark(src.load)
    assert len(df) == 1000


def test_billing_source_extraction(benchmark):
    """Billing synthetic data generation for 1000 customers."""
    ids = [f"c{i:04d}" for i in range(1000)]
    src = BillingSource(client_ids=ids)
    df, _ = benchmark(src.load)
    assert len(df) == 1000
