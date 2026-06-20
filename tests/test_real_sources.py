"""
tests/test_real_sources.py
---------------------------
Tests for real-world CSV sources and API source contracts.
API sources are tested via mocks (no live server required in CI).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ── KaggleTelcoSource ─────────────────────────────────────────────────────────


class TestKaggleTelcoSource:
    def _make_csv(self, tmp_path: Path) -> Path:
        """Create a minimal fake Kaggle CSV."""
        csv = tmp_path / "real_world_churn.csv"
        pd.DataFrame(
            {
                "customerID": [f"K{i:03d}" for i in range(50)],
                "Churn": np.random.default_rng(0).choice(["Yes", "No"], 50),
                "tenure": np.random.default_rng(1).integers(1, 72, 50),
                "MonthlyCharges": np.random.default_rng(2).uniform(20, 120, 50),
                "TotalCharges": np.random.default_rng(3).uniform(100, 5000, 50),
                "Contract": np.random.default_rng(4).choice(["Month-to-month", "One year", "Two year"], 50),
                "PaymentMethod": np.random.default_rng(5).choice(
                    ["Bank transfer (automatic)", "Credit card", "Mailed check"], 50
                ),
                "InternetService": np.random.default_rng(6).choice(["DSL", "Fiber optic"], 50),
            }
        ).to_csv(csv, index=False)
        return csv

    def test_extract_returns_correct_shape(self, tmp_path):
        from src.data.sources.kaggle_telco_source import KaggleTelcoSource

        csv = self._make_csv(tmp_path)
        src = KaggleTelcoSource(path=csv)
        df, meta = src.load()
        assert len(df) == 50
        assert "id" in df.columns
        assert "churn" in df.columns

    def test_churn_is_bool(self, tmp_path):
        from src.data.sources.kaggle_telco_source import KaggleTelcoSource

        csv = self._make_csv(tmp_path)
        df, _ = KaggleTelcoSource(path=csv).load()
        assert df["churn"].dtype == bool

    def test_tenure_converted_to_years(self, tmp_path):
        from src.data.sources.kaggle_telco_source import KaggleTelcoSource

        csv = self._make_csv(tmp_path)
        df, _ = KaggleTelcoSource(path=csv).load()
        assert df["num_years_antig"].max() <= 6.1  # 72 months / 12

    def test_contract_type_lowercased(self, tmp_path):
        from src.data.sources.kaggle_telco_source import KaggleTelcoSource

        csv = self._make_csv(tmp_path)
        df, _ = KaggleTelcoSource(path=csv).load()
        assert df["contract_type"].str.islower().all()

    def test_missing_file_raises(self):
        from src.data.sources.kaggle_telco_source import KaggleTelcoSource

        src = KaggleTelcoSource(path=Path("/nonexistent/path.csv"))
        with pytest.raises(FileNotFoundError, match="Kaggle"):
            src.load()

    def test_source_tag_present(self, tmp_path):
        from src.data.sources.kaggle_telco_source import KaggleTelcoSource

        csv = self._make_csv(tmp_path)
        df, _ = KaggleTelcoSource(path=csv).load()
        assert (df["data_source"] == "kaggle_telco").all()


# ── BankChurnSource ───────────────────────────────────────────────────────────


class TestBankChurnSource:
    def _make_csv(self, tmp_path: Path) -> Path:
        csv = tmp_path / "bank_customer_churn.csv"
        pd.DataFrame(
            {
                "customer_id": range(30),
                "churn": np.random.default_rng(0).integers(0, 2, 30),
                "credit_score": np.random.default_rng(1).integers(300, 850, 30),
                "age": np.random.default_rng(2).integers(18, 70, 30),
                "balance": np.random.default_rng(3).uniform(0, 100000, 30),
                "num_of_products": np.random.default_rng(4).integers(1, 5, 30),
                "has_cr_card": np.random.default_rng(5).integers(0, 2, 30),
                "is_active_member": np.random.default_rng(6).integers(0, 2, 30),
                "estimated_salary": np.random.default_rng(7).uniform(30000, 150000, 30),
                "country": np.random.default_rng(8).choice(["France", "Germany", "Spain"], 30),
                "gender": np.random.default_rng(9).choice(["Male", "Female"], 30),
            }
        ).to_csv(csv, index=False)
        return csv

    def test_extract_returns_correct_shape(self, tmp_path):
        from src.data.sources.bank_source import BankChurnSource

        csv = self._make_csv(tmp_path)
        df, _ = BankChurnSource(path=csv).load()
        assert len(df) == 30
        assert "id" in df.columns

    def test_ids_prefixed(self, tmp_path):
        from src.data.sources.bank_source import BankChurnSource

        csv = self._make_csv(tmp_path)
        df, _ = BankChurnSource(path=csv).load()
        assert df["id"].str.startswith("bank_").all()

    def test_churn_is_bool(self, tmp_path):
        from src.data.sources.bank_source import BankChurnSource

        csv = self._make_csv(tmp_path)
        df, _ = BankChurnSource(path=csv).load()
        assert df["churn"].dtype == bool

    def test_missing_file_raises(self):
        from src.data.sources.bank_source import BankChurnSource

        with pytest.raises(FileNotFoundError):
            BankChurnSource(path=Path("/nonexistent.csv")).load()


# ── API source contracts (mocked) ─────────────────────────────────────────────


class TestCRMApiSourceContract:
    def test_raises_without_url(self):
        import os

        from src.data.sources.crm_api_source import CRMApiDataSource

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ESPOCRM_URL", None)
            with pytest.raises(RuntimeError, match="ESPOCRM_URL"):
                CRMApiDataSource(url="")

    def test_accepts_url_kwarg(self):
        from src.data.sources.crm_api_source import CRMApiDataSource

        src = CRMApiDataSource(url="http://localhost:8080", api_key="test-key")
        assert src.url == "http://localhost:8080"


class TestSupportApiSourceContract:
    def test_raises_without_url(self):
        import os

        from src.data.sources.support_api_source import SupportApiDataSource

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ZAMMAD_URL", None)
            with pytest.raises(RuntimeError, match="ZAMMAD_URL"):
                SupportApiDataSource(url="")

    def test_accepts_url_kwarg(self):
        from src.data.sources.support_api_source import SupportApiDataSource

        src = SupportApiDataSource(url="http://localhost:3000", token="test-token")
        assert src.url == "http://localhost:3000"


class TestBillingApiSourceContract:
    def test_raises_without_url(self):
        import os

        from src.data.sources.billing_api_source import BillingApiDataSource

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NOVABILLING_URL", None)
            os.environ.pop("LAGO_URL", None)
            with pytest.raises(RuntimeError):
                BillingApiDataSource(provider="novabilling")

    def test_rejects_unknown_provider(self):
        from src.data.sources.billing_api_source import BillingApiDataSource

        with pytest.raises(ValueError, match="Unknown billing provider"):
            BillingApiDataSource(provider="unknown_billing_system")


# ── Pipeline source toggle tests ──────────────────────────────────────────────


class TestPipelineSourceFlags:
    def test_use_crm_api_flag_false_uses_synthetic(self, client_ids):
        from src.data.pipeline import MultiSourcePipeline
        from src.data.sources.crm_source import CRMSource

        p = MultiSourcePipeline.__new__(MultiSourcePipeline)
        p._use_crm_api = False
        p._crm_src = None
        src = p._get_crm_source(client_ids[:5])
        assert isinstance(src, CRMSource)

    def test_use_support_api_flag_false_uses_synthetic(self, client_ids):
        from src.data.pipeline import MultiSourcePipeline
        from src.data.sources.support_source import SupportSource

        p = MultiSourcePipeline.__new__(MultiSourcePipeline)
        p._use_support_api = False
        p._support_src = None
        src = p._get_support_source(client_ids[:5])
        assert isinstance(src, SupportSource)

    def test_use_billing_api_flag_false_uses_synthetic(self, client_ids):
        from src.data.pipeline import MultiSourcePipeline
        from src.data.sources.billing_source import BillingSource

        p = MultiSourcePipeline.__new__(MultiSourcePipeline)
        p._use_billing_api = False
        p._billing_src = None
        src = p._get_billing_source(client_ids[:5])
        assert isinstance(src, BillingSource)
