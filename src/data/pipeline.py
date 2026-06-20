"""
src/data/pipeline.py
---------------------
MultiSourcePipeline — extracts, validates, and merges all data sources.

Source modes (controlled via env vars or constructor flags):

  BCG CSVs (always on):
    - BCGClientSource  ← client_data.csv
    - BCGPriceSource   ← price_data.csv

  CRM (pick one via USE_CRM_API=true/false):
    - CRMApiDataSource  ← EspoCRM REST API  (USE_CRM_API=true)
    - CRMSource         ← synthetic generator (default)

  Support (pick one via USE_SUPPORT_API=true/false):
    - SupportApiDataSource ← Zammad REST API  (USE_SUPPORT_API=true)
    - SupportSource        ← synthetic generator (default)

  Billing (pick one via USE_BILLING_API=true/false):
    - BillingApiDataSource ← NovaBilling/Lago API (USE_BILLING_API=true)
    - BillingSource        ← synthetic generator (default)

  Supplemental CSV datasets (opt-in via USE_KAGGLE=true / USE_BANK=true):
    - KaggleTelcoSource ← real-world telco churn CSV
    - BankChurnSource   ← Maven Analytics bank churn CSV

Usage
-----
# Default (all synthetic):
pipeline = MultiSourcePipeline()

# With real APIs:
pipeline = MultiSourcePipeline(use_crm_api=True, use_support_api=True, use_billing_api=True)

# With supplemental CSV datasets:
pipeline = MultiSourcePipeline(use_kaggle=True, use_bank=True)

# Via environment variables:
USE_CRM_API=true USE_KAGGLE=true python -m src.pipeline.flows
"""

from __future__ import annotations

import logging
import os

import pandas as pd

from src.data.preprocessing import aggregate_price_data, clean_client_data, null_report
from src.data.sources.bank_source import BankChurnSource
from src.data.sources.bcg_source import BCGClientSource, BCGPriceSource
from src.data.sources.billing_source import BillingSource
from src.data.sources.crm_source import CRMSource
from src.data.sources.kaggle_telco_source import KaggleTelcoSource
from src.data.sources.support_source import SupportSource

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).lower() in ("1", "true", "yes")


class MultiSourcePipeline:
    """
    Orchestrates extraction from all enabled sources, validates each,
    merges them on customer ID, and returns the unified DataFrame
    plus a flat metadata dict suitable for MLflow logging.
    """

    def __init__(
        self,
        # BCG sources (always required)
        client_source: BCGClientSource | None = None,
        price_source: BCGPriceSource | None = None,
        # CRM: synthetic or API-backed
        crm_source=None,
        use_crm_api: bool | None = None,
        # Support: synthetic or API-backed
        support_source=None,
        use_support_api: bool | None = None,
        # Billing: synthetic or API-backed
        billing_source=None,
        use_billing_api: bool | None = None,
        # Supplemental real-world CSV datasets
        use_kaggle: bool | None = None,
        kaggle_source: KaggleTelcoSource | None = None,
        use_bank: bool | None = None,
        bank_source: BankChurnSource | None = None,
    ):
        self._client_src = client_source or BCGClientSource()
        self._price_src = price_source or BCGPriceSource()

        # CRM mode
        self._use_crm_api = use_crm_api if use_crm_api is not None else _env_flag("USE_CRM_API")
        self._crm_src = crm_source

        # Support mode
        self._use_support_api = use_support_api if use_support_api is not None else _env_flag("USE_SUPPORT_API")
        self._support_src = support_source

        # Billing mode
        self._use_billing_api = use_billing_api if use_billing_api is not None else _env_flag("USE_BILLING_API")
        self._billing_src = billing_source

        # Supplemental datasets
        self._use_kaggle = use_kaggle if use_kaggle is not None else _env_flag("USE_KAGGLE")
        self._kaggle_src = kaggle_source
        self._use_bank = use_bank if use_bank is not None else _env_flag("USE_BANK")
        self._bank_src = bank_source

    # ── Source factories ──────────────────────────────────────────────────────

    def _get_crm_source(self, client_ids: list[str]):
        if self._crm_src:
            return self._crm_src
        if self._use_crm_api:
            from src.data.sources.crm_api_source import CRMApiDataSource

            logger.info("Using CRMApiDataSource (EspoCRM)")
            return CRMApiDataSource()
        logger.info("Using synthetic CRMSource")
        return CRMSource(client_ids=client_ids)

    def _get_support_source(self, client_ids: list[str]):
        if self._support_src:
            return self._support_src
        if self._use_support_api:
            from src.data.sources.support_api_source import SupportApiDataSource

            logger.info("Using SupportApiDataSource (Zammad)")
            return SupportApiDataSource()
        logger.info("Using synthetic SupportSource")
        return SupportSource(client_ids=client_ids)

    def _get_billing_source(self, client_ids: list[str]):
        if self._billing_src:
            return self._billing_src
        if self._use_billing_api:
            from src.data.sources.billing_api_source import BillingApiDataSource

            logger.info("Using BillingApiDataSource (NovaBilling/Lago)")
            return BillingApiDataSource()
        logger.info("Using synthetic BillingSource")
        return BillingSource(client_ids=client_ids)

    # ── Main run ──────────────────────────────────────────────────────────────

    def run(self) -> tuple[pd.DataFrame, dict]:
        """
        Execute full extraction → validate → merge pipeline.

        Returns
        -------
        df       : merged DataFrame ready for feature engineering
        all_meta : flat dict of source metadata (for mlflow.log_metrics)
        """
        all_meta: dict = {}

        # 1. BCG client base
        client_df, meta = self._client_src.load()
        all_meta.update(meta.as_dict())
        client_df = clean_client_data(client_df)
        client_ids = client_df["id"].tolist()

        # 2. BCG price (aggregate → one row per client)
        price_df, meta = self._price_src.load()
        all_meta.update(meta.as_dict())
        price_agg = aggregate_price_data(price_df)

        # 3. CRM
        crm_src = self._get_crm_source(client_ids)
        if hasattr(crm_src, "client_ids") and not crm_src.client_ids:
            crm_src.client_ids = client_ids
        crm_df, meta = crm_src.load()
        all_meta.update(meta.as_dict())

        # 4. Support
        support_src = self._get_support_source(client_ids)
        if hasattr(support_src, "client_ids") and not support_src.client_ids:
            support_src.client_ids = client_ids
        support_df, meta = support_src.load()
        all_meta.update(meta.as_dict())

        # 5. Billing
        billing_src = self._get_billing_source(client_ids)
        if hasattr(billing_src, "client_ids") and not billing_src.client_ids:
            billing_src.client_ids = client_ids
        billing_df, meta = billing_src.load()
        all_meta.update(meta.as_dict())

        # 6. Merge BCG + enrichment sources
        df = (
            client_df.merge(price_agg, on="id", how="left")
            .merge(crm_df, on="id", how="left")
            .merge(support_df, on="id", how="left")
            .merge(billing_df, on="id", how="left")
        )

        # 7. Supplemental CSV datasets (appended as additional rows)
        if self._use_kaggle:
            try:
                kaggle_src = self._kaggle_src or KaggleTelcoSource()
                kaggle_df, meta = kaggle_src.load()
                all_meta.update(meta.as_dict())
                df = pd.concat([df, kaggle_df], ignore_index=True)
                logger.info("Appended %d Kaggle telco rows", len(kaggle_df))
            except FileNotFoundError as e:
                logger.warning("Kaggle dataset skipped: %s", e)

        if self._use_bank:
            try:
                bank_src = self._bank_src or BankChurnSource()
                bank_df, meta = bank_src.load()
                all_meta.update(meta.as_dict())
                df = pd.concat([df, bank_df], ignore_index=True)
                logger.info("Appended %d bank churn rows", len(bank_df))
            except FileNotFoundError as e:
                logger.warning("Bank dataset skipped: %s", e)

        # 8. Final metadata
        all_meta.update(
            {
                "merged_rows": len(df),
                "merged_cols": len(df.columns),
                "merged_null_rate": round(df.isnull().sum().sum() / max(df.size, 1), 4),
                "use_crm_api": int(self._use_crm_api),
                "use_support_api": int(self._use_support_api),
                "use_billing_api": int(self._use_billing_api),
                "use_kaggle": int(self._use_kaggle),
                "use_bank": int(self._use_bank),
            }
        )

        nulls = null_report(df)
        logger.info(
            "Pipeline complete: %d rows, %d cols, %d columns with nulls",
            len(df),
            len(df.columns),
            len(nulls),
        )
        return df, all_meta
