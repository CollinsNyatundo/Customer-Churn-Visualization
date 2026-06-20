"""
src/data/sources/bank_source.py
---------------------------------
Bank Customer Churn dataset from Maven Analytics.
10,000 customers at a European bank — demographics, account info, churn label.

Dataset: https://mavenanalytics.io/data-playground/bank-customer-churn

Download instructions
---------------------
1. Visit https://mavenanalytics.io/data-playground/bank-customer-churn
2. Click Download → save CSV to data/raw/bank/bank_customer_churn.csv

This acts as a complementary/cross-industry churn signal alongside telco data.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import settings
from src.data.sources.base import BaseDataSource

logger = logging.getLogger(__name__)

DEFAULT_PATH = settings.data_raw_dir / "bank" / "bank_customer_churn.csv"

# Maven Analytics column → internal schema
COLUMN_MAP = {
    "customer_id": "id",
    "churn": "churn",
    "credit_score": "net_margin",
    "age": "num_years_antig",
    "balance": "total_outstanding",
    "num_of_products": "nb_prod_act",
    "has_cr_card": "has_gas",
    "is_active_member": "autopay_enrolled",
    "estimated_salary": "imp_cons",
    "country": "activity_new",
    "gender": "channel_sales",
}

REQUIRED_COLS = {"id", "churn"}


class BankChurnSource(BaseDataSource):
    """
    Loads the Maven Analytics Bank Customer Churn dataset and normalises
    it to the internal schema. Provides cross-industry churn signals
    (banking vs telco) for richer model training.
    """

    name = "bank_churn"

    def __init__(self, path: Path | None = None):
        self.path = path or DEFAULT_PATH

    def extract(self) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(
                f"Bank churn dataset not found at {self.path}.\n"
                "Download from https://mavenanalytics.io/data-playground/bank-customer-churn\n"
                "and place at data/raw/bank/bank_customer_churn.csv"
            )

        df = pd.read_csv(self.path)
        df = df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns})

        # Normalise churn
        if "churn" in df.columns:
            df["churn"] = df["churn"].map({1: True, 0: False, "Yes": True, "No": False})

        # Convert age → years of antiquity (approximate)
        if "num_years_antig" in df.columns:
            df["num_years_antig"] = ((df["num_years_antig"] - 18) / 5).clip(lower=0).round(1)

        # Prefix IDs to avoid collision with telco/BCG IDs
        if "id" in df.columns:
            df["id"] = "bank_" + df["id"].astype(str)

        # Add source tag
        df["data_source"] = "bank_churn"

        logger.info("BankChurnSource loaded %d rows from %s", len(df), self.path)
        return df

    def validate(self, df: pd.DataFrame) -> None:
        missing = REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"BankChurnSource missing columns: {missing}")
        if df["id"].duplicated().any():
            raise ValueError("BankChurnSource: duplicate customer IDs found")
