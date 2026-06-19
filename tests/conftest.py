"""tests/conftest.py — Shared fixtures for all test modules."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def sample_client_df() -> pd.DataFrame:
    """Minimal client DataFrame that mirrors client_data.csv structure."""
    n = 100
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "id": [f"client_{i:04d}" for i in range(n)],
        "churn": rng.choice([0, 1], size=n, p=[0.85, 0.15]),
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
        "forecast_cons_12m": rng.exponential(10000, n).clip(0),
        "forecast_cons_year": rng.exponential(10000, n).clip(0),
        "forecast_meter_rent_12m": rng.exponential(30, n).clip(0),
        "forecast_price_energy_off_peak": rng.uniform(0.05, 0.15, n),
        "forecast_price_energy_peak": rng.uniform(0.10, 0.25, n),
        "forecast_price_pow_off_peak": rng.uniform(0.01, 0.05, n),
        "activity_new": rng.choice(["a", "b", "c", None], n),
        "channel_sales": rng.choice(["online", "phone", "agent"], n),
        "origin_up": rng.choice(["campaign_a", "campaign_b"], n),
        "has_gas": rng.choice(["t", "f"], n),
        "margin_net_pow_ele": rng.normal(100, 40, n),
        "date_activ": pd.date_range("2015-01-01", periods=n, freq="3D"),
        "date_end": pd.date_range("2026-01-01", periods=n, freq="3D"),
        "date_modif_prod": pd.date_range("2023-01-01", periods=n, freq="5D"),
        "date_renewal": pd.date_range("2026-06-01", periods=n, freq="2D"),
    })


@pytest.fixture(scope="session")
def sample_price_df(sample_client_df: pd.DataFrame) -> pd.DataFrame:
    """Monthly price rows for each client."""
    ids = sample_client_df["id"].tolist()
    rows = []
    rng = np.random.default_rng(1)
    for cid in ids:
        for month in range(12):
            rows.append({
                "id": cid,
                "price_date": pd.Timestamp("2023-01-01") + pd.DateOffset(months=month),
                "price_off_peak_var": rng.uniform(0.05, 0.10),
                "price_peak_var": rng.uniform(0.10, 0.20),
                "price_mid_peak_var": rng.uniform(0.07, 0.14),
                "price_off_peak_fix": rng.uniform(0.01, 0.03),
                "price_peak_fix": rng.uniform(0.02, 0.05),
                "price_mid_peak_fix": rng.uniform(0.015, 0.04),
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def client_ids(sample_client_df: pd.DataFrame) -> list[str]:
    return sample_client_df["id"].tolist()


@pytest.fixture(scope="session")
def merged_df(sample_client_df, sample_price_df, client_ids):
    """Full merged + featured DataFrame for model/pipeline tests."""
    from src.data.preprocessing import clean_client_data, merge_datasets
    from src.data.sources.crm_source import CRMSource
    from src.data.sources.support_source import SupportSource
    from src.data.sources.billing_source import BillingSource
    from src.features.engineering import build_feature_set

    cleaned = clean_client_data(sample_client_df)
    merged = merge_datasets(cleaned, sample_price_df)

    crm_df, _ = CRMSource(client_ids=client_ids).load()
    support_df, _ = SupportSource(client_ids=client_ids).load()
    billing_df, _ = BillingSource(client_ids=client_ids).load()

    merged = (merged
              .merge(crm_df, on="id", how="left")
              .merge(support_df, on="id", how="left")
              .merge(billing_df, on="id", how="left"))
    return build_feature_set(merged)
