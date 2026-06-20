"""
src/data/sources/kaggle_telco_source.py
-----------------------------------------
Real-World Customer Churn Dataset from Kaggle.
60,000+ anonymized records from a large telco company.

Dataset: https://www.kaggle.com/datasets/lasaljaywardena/real-world-churn

Download instructions
---------------------
1. Install kaggle CLI:  pip install kaggle
2. Place your kaggle.json token at ~/.kaggle/kaggle.json
3. Run:
       kaggle datasets download lasaljaywardena/real-world-churn
       unzip real-world-churn.zip -d data/raw/kaggle_telco/
4. Set KAGGLE_TELCO_PATH in .env (or use the default below).

Alternatively, download manually from the Kaggle link above and place the
CSV at data/raw/kaggle_telco/real_world_churn.csv.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import settings
from src.data.sources.base import BaseDataSource

logger = logging.getLogger(__name__)

DEFAULT_PATH = settings.data_raw_dir / "kaggle_telco" / "real_world_churn.csv"

# Column mapping: Kaggle names → our internal schema
# Semantic mapping note
# -----------------------
# MonthlyCharges → imp_cons  : Approximate. Monthly billing charge is not the
#   same as current-period consumption (imp_cons), but it is the closest
#   Kaggle field. Models trained on mixed BCG+Kaggle data should treat imp_cons
#   as "billing proxy" rather than true consumption.
#
# TotalCharges → cons_12m    : Approximate. Total charges ≠ 12-month kWh
#   consumption, but it correlates with usage volume. Treat as a proxy.
#
# These mappings are intentionally coarse. When precision matters,
# train separate models per source and ensemble rather than mixing rows.

COLUMN_MAP = {
    "customerID": "id",
    "Churn": "churn",
    "tenure": "num_years_antig",
    "MonthlyCharges": "imp_cons",  # billing proxy — see note above
    "TotalCharges": "cons_12m",  # usage volume proxy — see note above
    "Contract": "contract_type",
    "PaymentMethod": "payment_method",
    "InternetService": "activity_new",
    "OnlineSecurity": "channel_sales",
}

REQUIRED_COLS = {"id", "churn"}


class KaggleTelcoSource(BaseDataSource):
    """
    Loads the Kaggle Real-World Churn dataset and normalises it
    to match the internal schema used by MultiSourcePipeline.
    """

    name = "kaggle_telco"

    def __init__(self, path: Path | None = None):
        self.path = path or DEFAULT_PATH

    def extract(self) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(
                f"Kaggle telco dataset not found at {self.path}.\n"
                "Download it from https://www.kaggle.com/datasets/lasaljaywardena/real-world-churn\n"
                "and place it at data/raw/kaggle_telco/real_world_churn.csv"
            )

        df = pd.read_csv(self.path)

        # Rename known columns to internal schema
        df = df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns})

        # Normalise churn to bool
        if "churn" in df.columns:
            df["churn"] = df["churn"].map({"Yes": True, "No": False, 1: True, 0: False})

        # Normalise contract type to match CRM source format
        if "contract_type" in df.columns:
            df["contract_type"] = df["contract_type"].str.lower().str.replace(" ", "-").fillna("month-to-month")

        # Normalise tenure (Kaggle uses months, we use years)
        if "num_years_antig" in df.columns:
            df["num_years_antig"] = (df["num_years_antig"] / 12).round(2)

        # TotalCharges may be string with spaces
        if "cons_12m" in df.columns:
            df["cons_12m"] = pd.to_numeric(df["cons_12m"], errors="coerce").fillna(0)

        # Add source tag
        df["data_source"] = "kaggle_telco"

        logger.info("KaggleTelcoSource loaded %d rows from %s", len(df), self.path)
        return df

    def validate(self, df: pd.DataFrame) -> None:
        missing = REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"KaggleTelcoSource missing columns: {missing}")
        if df["id"].duplicated().any():
            raise ValueError("KaggleTelcoSource: duplicate customer IDs found")
        if df["churn"].isnull().any():
            raise ValueError("KaggleTelcoSource: null values in churn column")
