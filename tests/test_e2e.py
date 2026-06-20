"""
tests/test_e2e.py
------------------
True end-to-end tests: real feature construction + real model training
+ real FastAPI serving. The predictor is NOT mocked.

These tests are slower (~10-30s) but prove the actual inference path works —
not just the FastAPI wiring with a mocked predictor.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.preprocessing import aggregate_price_data, clean_client_data
from src.data.sources.billing_source import BillingSource
from src.data.sources.crm_source import CRMSource
from src.data.sources.support_source import SupportSource
from src.features.engineering import FeatureThresholds, build_feature_set

# ── Shared fixtures ───────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def real_df():
    """500-row fully-featured DataFrame built through the real pipeline."""
    rng = np.random.default_rng(42)
    n = 500
    ids = [f"e2e_{i:04d}" for i in range(n)]

    client_df = pd.DataFrame(
        {
            "id": ids,
            "churn": rng.choice([0, 1], n, p=[0.88, 0.12]),
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
            "forecast_cons_12m": rng.exponential(10000, n).clip(0),
            "forecast_cons_year": rng.exponential(10000, n).clip(0),
            "margin_net_pow_ele": rng.normal(100, 40, n),
        }
    )
    cleaned = clean_client_data(client_df)
    crm_df, _ = CRMSource(client_ids=ids).load()
    sup_df, _ = SupportSource(client_ids=ids).load()
    bill_df, _ = BillingSource(client_ids=ids).load()
    merged = (
        cleaned.merge(crm_df, on="id", how="left")
        .merge(sup_df, on="id", how="left")
        .merge(bill_df, on="id", how="left")
    )
    return build_feature_set(merged)


@pytest.fixture(scope="module")
def trained_pipeline(real_df):
    """Train a real (small) sklearn pipeline on the E2E dataset."""
    NUM = [
        "cons_12m",
        "cons_gas_12m",
        "net_margin",
        "num_years_antig",
        "nps_score",
        "num_tickets_6m",
        "num_late_payments_12m",
        "cross_source_risk_score",
    ]
    CAT = ["channel_sales", "contract_type", "payment_method"]
    avail_num = [c for c in NUM if c in real_df.columns]
    avail_cat = [c for c in CAT if c in real_df.columns]

    X = real_df[avail_num + avail_cat]
    y = real_df["churn"].astype(int)

    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat_pipe = Pipeline(
        [
            ("imp", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    pre = ColumnTransformer([("num", num_pipe, avail_num), ("cat", cat_pipe, avail_cat)], remainder="drop")
    clf = GradientBoostingClassifier(n_estimators=50, random_state=42)
    pipeline = Pipeline([("pre", pre), ("clf", clf)])
    pipeline.fit(X, y)
    return pipeline, avail_num, avail_cat


# ── Feature engineering tests ─────────────────────────────────────────────────


class TestFeatureEngineeringE2E:
    def test_no_inf_values(self, real_df):
        numeric = real_df.select_dtypes(include="number")
        assert not np.isinf(numeric.values).any(), "Inf values found in feature set"

    def test_cross_source_risk_between_0_and_1(self, real_df):
        scores = real_df["cross_source_risk_score"].dropna()
        assert scores.between(0, 1).all(), "Risk score outside [0, 1]"

    def test_frozen_thresholds_match_batch(self, real_df):
        thresholds = FeatureThresholds.from_dataframe(real_df)
        df_frozen = build_feature_set(real_df, thresholds=thresholds)
        df_batch = build_feature_set(real_df)
        # high_consumption should be identical when thresholds match batch
        pd.testing.assert_series_equal(
            df_frozen["high_consumption"],
            df_batch["high_consumption"],
            check_names=True,
        )

    def test_months_to_renewal_deterministic(self, real_df):
        ref = pd.Timestamp("2026-06-20")
        df1 = build_feature_set(real_df, reference_date=ref)
        df2 = build_feature_set(real_df, reference_date=ref)
        pd.testing.assert_series_equal(df1["months_to_renewal"], df2["months_to_renewal"])


# ── Real model inference path ─────────────────────────────────────────────────


class TestRealInferencePath:
    def test_pipeline_produces_probabilities(self, trained_pipeline, real_df):
        pipeline, avail_num, avail_cat = trained_pipeline
        X = real_df[avail_num + avail_cat].head(10)
        probs = pipeline.predict_proba(X)[:, 1]
        assert len(probs) == 10
        assert np.all((probs >= 0) & (probs <= 1))

    def test_pipeline_handles_all_unknown_categoricals(self, trained_pipeline):
        pipeline, avail_num, avail_cat = trained_pipeline
        row = {c: 0 for c in avail_num}
        row.update({c: "UNSEEN_VALUE" for c in avail_cat})
        X = pd.DataFrame([row])
        probs = pipeline.predict_proba(X)[:, 1]
        assert len(probs) == 1
        assert 0 <= probs[0] <= 1

    def test_pipeline_handles_null_inputs(self, trained_pipeline):
        pipeline, avail_num, avail_cat = trained_pipeline
        row = {c: None for c in avail_num}
        row.update({c: None for c in avail_cat})
        X = pd.DataFrame([row])
        probs = pipeline.predict_proba(X)[:, 1]
        assert len(probs) == 1
        assert 0 <= probs[0] <= 1


# ── FastAPI with real model (no mocked predictor) ─────────────────────────────


class TestAPIWithRealModel:
    @pytest.fixture(scope="class")
    def client_real(self, trained_pipeline, real_df):
        """TestClient wired to a real (not mocked) predictor."""
        from unittest.mock import MagicMock, patch

        from api.predictor import ChurnPredictor

        pipeline, avail_num, avail_cat = trained_pipeline
        thresholds = FeatureThresholds.from_dataframe(real_df)

        real_predictor = ChurnPredictor.__new__(ChurnPredictor)
        real_predictor._pipeline = pipeline
        real_predictor._model_version = "e2e-test-v1"
        real_predictor._thresholds = thresholds
        real_predictor.stage = "test"

        with patch("api.main.get_predictor", return_value=real_predictor):
            from api.main import app

            yield TestClient(app)

    def test_predict_returns_valid_probability(self, client_real):
        resp = client_real.post("/predict", json={"nps_score": -80})
        assert resp.status_code == 200
        prob = resp.json()["churn_probability"]
        assert 0 <= prob <= 1

    def test_high_risk_profile_scores_higher(self, client_real):
        low_risk = client_real.post(
            "/predict",
            json={
                "nps_score": 90,
                "satisfaction_score": 4.8,
                "num_late_payments_12m": 0,
                "contract_type": "two-year",
            },
        ).json()["churn_probability"]

        high_risk = client_real.post(
            "/predict",
            json={
                "nps_score": -90,
                "satisfaction_score": 1.1,
                "num_late_payments_12m": 7,
                "contract_type": "month-to-month",
            },
        ).json()["churn_probability"]

        assert high_risk > low_risk, (
            f"High-risk profile ({high_risk:.3f}) should score higher " f"than low-risk ({low_risk:.3f})"
        )

    def test_batch_same_as_individual(self, client_real):
        payload = {"nps_score": -50, "num_tickets_6m": 8}
        single = client_real.post("/predict", json=payload).json()["churn_probability"]
        batch = client_real.post("/predict/batch", json=[payload]).json()[0]["churn_probability"]
        assert abs(single - batch) < 1e-4, "Batch and single prediction should match"

    def test_risk_tier_consistent_with_probability(self, client_real):
        resp = client_real.post("/predict", json={"nps_score": -90}).json()
        prob = resp["churn_probability"]
        tier = resp["risk_tier"]
        if prob >= 0.60:
            assert tier == "high"
        elif prob >= 0.30:
            assert tier == "medium"
        else:
            assert tier == "low"
