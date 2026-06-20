"""
src/data/sources/crm_api_source.py
------------------------------------
CRM data source backed by EspoCRM REST API.
Replaces the synthetic CRMSource when a live EspoCRM instance is available.

EspoCRM setup
-------------
1. docker compose up -d espocrm   (see docker-compose.yml)
2. Visit http://localhost:8080, complete setup wizard
3. Admin → API Users → create a user, copy the API key
4. Set ESPOCRM_URL and ESPOCRM_API_KEY in .env
5. Seed customers: Admin → Import → upload a CSV of customer records

API docs: https://docs.espocrm.com/development/api/

Falls back to synthetic CRMSource if ESPOCRM_URL is not set.
"""

from __future__ import annotations

import logging
import os
from typing import Any
from urllib.parse import urljoin

import pandas as pd

from src.data.sources.base import BaseDataSource

logger = logging.getLogger(__name__)

_URL = os.getenv("ESPOCRM_URL", "")
_KEY = os.getenv("ESPOCRM_API_KEY", "")
_PAGE_SIZE = 200


class CRMApiDataSource(BaseDataSource):
    """
    Pulls customer NPS, satisfaction, contact history, and contract type
    from a live EspoCRM instance via its REST API.

    If ESPOCRM_URL is not configured, raises RuntimeError with setup instructions.
    """

    name = "crm_api"
    REQUIRED_COLS = {"id", "nps_score", "satisfaction_score", "contract_type"}

    def __init__(self, url: str | None = None, api_key: str | None = None):
        self.url = (url or _URL).rstrip("/")
        self.api_key = api_key or _KEY
        if not self.url:
            raise RuntimeError(
                "ESPOCRM_URL not set. Either:\n"
                "  1. Set ESPOCRM_URL + ESPOCRM_API_KEY in .env and run docker compose up espocrm\n"
                "  2. Use CRMSource (synthetic) instead: pipeline = MultiSourcePipeline(use_api=False)"
            )

    def _get(self, endpoint: str, params: dict | None = None) -> dict[str, Any]:
        import json
        import urllib.request

        url = urljoin(self.url + "/api/v1/", endpoint)
        if params:
            from urllib.parse import urlencode

            url += "?" + urlencode(params)

        req = urllib.request.Request(url, headers={"X-Api-Key": self.api_key})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())

    def _fetch_all(self, entity: str, select: list[str]) -> list[dict]:
        """Paginate through all records of a given EspoCRM entity."""
        records, offset = [], 0
        while True:
            resp = self._get(entity, {"select": ",".join(select), "maxSize": _PAGE_SIZE, "offset": offset})
            batch = resp.get("list", [])
            records.extend(batch)
            if len(batch) < _PAGE_SIZE:
                break
            offset += _PAGE_SIZE
        return records

    def extract(self) -> pd.DataFrame:
        logger.info("Fetching CRM data from EspoCRM at %s", self.url)

        contacts = self._fetch_all(
            "Contact",
            ["id", "accountId", "npsScore", "satisfactionScore", "lastContactDate", "contractType", "accountManagerId"],
        )
        if not contacts:
            logger.warning("EspoCRM returned 0 contacts — returning empty DataFrame")
            return pd.DataFrame()

        df = pd.DataFrame(contacts)
        df = df.rename(
            columns={
                "accountId": "id",
                "npsScore": "nps_score",
                "satisfactionScore": "satisfaction_score",
                "lastContactDate": "last_contact_days_ago",
                "contractType": "contract_type",
            }
        )

        # Convert last contact date to days ago
        if "last_contact_days_ago" in df.columns:
            df["last_contact_days_ago"] = (
                (pd.Timestamp.today() - pd.to_datetime(df["last_contact_days_ago"], errors="coerce"))
                .dt.days.clip(lower=0)
                .fillna(90)
            )

        df["data_source"] = "crm_api"
        logger.info("CRMApiDataSource fetched %d records", len(df))
        return df

    def validate(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        missing = self.REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"CRMApiDataSource missing columns: {missing}")
