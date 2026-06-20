"""
src/data/sources/billing_api_source.py
-----------------------------------------
Billing data source backed by NovaBilling or Lago REST API.
Replaces the synthetic BillingSource when a live billing system is available.

NovaBilling setup
-----------------
1. docker compose up -d novabilling   (see docker-compose.yml)
2. API available at http://localhost:4000
3. Interactive docs at http://localhost:4000/api/reference
4. Set BILLING_PROVIDER=novabilling, NOVABILLING_URL, NOVABILLING_API_KEY in .env

Lago setup (alternative)
------------------------
1. docker compose up -d lago
2. Visit http://localhost:3000, create an account
3. Developers → API Keys → copy key
4. Set BILLING_PROVIDER=lago, LAGO_URL, LAGO_API_KEY in .env

API docs:
  NovaBilling: http://localhost:4000/api/reference
  Lago:        https://doc.getlago.com/api-reference/intro

Falls back to synthetic BillingSource if no billing URL is configured.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request

import pandas as pd

from src.data.sources.base import BaseDataSource

logger = logging.getLogger(__name__)

_PROVIDER = os.getenv("BILLING_PROVIDER", "novabilling")
_NOVABILLING_URL = os.getenv("NOVABILLING_URL", "")
_NOVABILLING_KEY = os.getenv("NOVABILLING_API_KEY", "")
_LAGO_URL = os.getenv("LAGO_URL", "")
_LAGO_KEY = os.getenv("LAGO_API_KEY", "")


class BillingApiDataSource(BaseDataSource):
    """
    Fetches per-customer billing metrics from NovaBilling or Lago:
    - num_late_payments_12m, avg_days_late, payment_method,
      total_outstanding, discount_applied, discount_pct, autopay_enrolled
    """

    name = "billing_api"
    REQUIRED_COLS = {"id", "num_late_payments_12m", "total_outstanding"}

    def __init__(self, provider: str | None = None):
        self.provider = (provider or _PROVIDER).lower()
        if self.provider == "novabilling":
            self.base_url = _NOVABILLING_URL.rstrip("/")
            self.api_key = _NOVABILLING_KEY
        elif self.provider == "lago":
            self.base_url = _LAGO_URL.rstrip("/")
            self.api_key = _LAGO_KEY
        else:
            raise ValueError(f"Unknown billing provider: {self.provider}")

        if not self.base_url:
            raise RuntimeError(
                f"{self.provider.upper()}_URL not set. Either:\n"
                f"  1. Set {self.provider.upper()}_URL + {self.provider.upper()}_API_KEY in .env\n"
                "  2. Use BillingSource (synthetic): pipeline = MultiSourcePipeline(use_api=False)"
            )

    def _get(self, path: str, params: dict | None = None) -> dict | list:
        url = f"{self.base_url}/{path.lstrip('/')}"
        if params:
            from urllib.parse import urlencode

            url += "?" + urlencode(params)
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())

    def _fetch_novabilling(self) -> pd.DataFrame:
        customers = self._get("/customers", {"page": 1, "per_page": 1000}).get("data", [])
        rows = []
        for c in customers:
            invoices = self._get(f"/customers/{c['id']}/invoices").get("data", [])
            late = [i for i in invoices if i.get("status") == "overdue"]
            outstanding = sum(float(i.get("amount_due", 0)) for i in invoices if i.get("status") in ("open", "overdue"))
            rows.append(
                {
                    "id": str(c["id"]),
                    "num_late_payments_12m": len(late),
                    "avg_days_late": round(sum(i.get("days_overdue", 0) for i in late) / max(len(late), 1), 1),
                    "payment_method": c.get("payment_method", "bank_transfer"),
                    "total_outstanding": round(outstanding, 2),
                    "discount_applied": bool(c.get("discount_percent", 0)),
                    "discount_pct": int(c.get("discount_percent", 0)),
                    "autopay_enrolled": bool(c.get("auto_charge", False)),
                    "data_source": "billing_api_novabilling",
                }
            )
        return pd.DataFrame(rows)

    def _fetch_lago(self) -> pd.DataFrame:
        customers = self._get("/api/v1/customers", {"per_page": 1000}).get("customers", [])
        rows = []
        for c in customers:
            cid = c.get("external_id", c.get("lago_id"))
            invoices = self._get(f"/api/v1/invoices?external_customer_id={cid}").get("invoices", [])
            late = [i for i in invoices if i.get("payment_status") == "failed"]
            outstanding = sum(
                float(i.get("total_amount_cents", 0)) / 100 for i in invoices if i.get("status") != "finalized"
            )
            rows.append(
                {
                    "id": str(cid),
                    "num_late_payments_12m": len(late),
                    "avg_days_late": 0.0,
                    "payment_method": c.get("billing_configuration", {}).get("payment_provider", "bank_transfer"),
                    "total_outstanding": round(outstanding, 2),
                    "discount_applied": False,
                    "discount_pct": 0,
                    "autopay_enrolled": False,
                    "data_source": "billing_api_lago",
                }
            )
        return pd.DataFrame(rows)

    def extract(self) -> pd.DataFrame:
        logger.info("Fetching billing data from %s at %s", self.provider, self.base_url)
        if self.provider == "novabilling":
            df = self._fetch_novabilling()
        else:
            df = self._fetch_lago()
        logger.info("BillingApiDataSource fetched %d records", len(df))
        return df

    def validate(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        missing = self.REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"BillingApiDataSource missing columns: {missing}")
        if (df["total_outstanding"] < 0).any():
            raise ValueError("BillingApiDataSource: negative outstanding balance")
