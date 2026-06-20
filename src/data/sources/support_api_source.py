"""
src/data/sources/support_api_source.py
-----------------------------------------
Support ticket data source backed by Zammad REST API.
Replaces the synthetic SupportSource when a live Zammad instance is available.

Zammad setup
------------
1. docker compose up -d zammad   (see docker-compose.yml)
2. Visit http://localhost:3000, complete setup wizard
3. Admin → API Token Access → create a token
4. Set ZAMMAD_URL and ZAMMAD_API_TOKEN in .env
5. Seed tickets: upload via Zammad's import feature or API

API docs: https://docs.zammad.org/en/latest/api/intro.html

Falls back to synthetic SupportSource if ZAMMAD_URL is not set.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from collections import defaultdict

import pandas as pd

from src.data.sources.base import BaseDataSource

logger = logging.getLogger(__name__)

_URL = os.getenv("ZAMMAD_URL", "")
_TOKEN = os.getenv("ZAMMAD_API_TOKEN", "")
_PAGE_SIZE = 100


class SupportApiDataSource(BaseDataSource):
    """
    Aggregates Zammad tickets per customer:
    - num_tickets_6m, avg_resolution_hours, escalations_6m, open_tickets,
      top_ticket_category, post_ticket_csat
    """

    name = "support_api"
    REQUIRED_COLS = {"id", "num_tickets_6m", "avg_resolution_hours"}

    def __init__(self, url: str | None = None, token: str | None = None):
        self.url = (url or _URL).rstrip("/")
        self.token = token or _TOKEN
        if not self.url:
            raise RuntimeError(
                "ZAMMAD_URL not set. Either:\n"
                "  1. Set ZAMMAD_URL + ZAMMAD_API_TOKEN in .env and run docker compose up zammad\n"
                "  2. Use SupportSource (synthetic) instead: pipeline = MultiSourcePipeline(use_api=False)"
            )

    def _get(self, endpoint: str, params: dict | None = None) -> list | dict:
        url = f"{self.url}/api/v1/{endpoint}"
        if params:
            from urllib.parse import urlencode

            url += "?" + urlencode(params)
        req = urllib.request.Request(url, headers={"Authorization": f"Token token={self.token}"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())

    def _fetch_tickets(self) -> list[dict]:
        cutoff = (pd.Timestamp.today() - pd.DateOffset(months=6)).isoformat()
        tickets, page = [], 1
        while True:
            batch = self._get("tickets", {"per_page": _PAGE_SIZE, "page": page, "created_after": cutoff})
            if not batch:
                break
            tickets.extend(batch if isinstance(batch, list) else [batch])
            if len(batch) < _PAGE_SIZE:
                break
            page += 1
        return tickets

    def extract(self) -> pd.DataFrame:
        logger.info("Fetching support tickets from Zammad at %s", self.url)
        tickets = self._fetch_tickets()

        if not tickets:
            logger.warning("Zammad returned 0 tickets — returning empty DataFrame")
            return pd.DataFrame()

        # Aggregate per customer (customer_id field in Zammad tickets)
        agg: dict[str, dict] = defaultdict(
            lambda: {
                "num_tickets_6m": 0,
                "resolution_hours": [],
                "escalations_6m": 0,
                "open_tickets": 0,
                "categories": [],
                "csat_scores": [],
            }
        )

        for t in tickets:
            cid = str(t.get("customer_id", ""))
            if not cid:
                continue
            a = agg[cid]
            a["num_tickets_6m"] += 1

            # Resolution time
            created = pd.to_datetime(t.get("created_at"), errors="coerce")
            closed = pd.to_datetime(t.get("close_at"), errors="coerce")
            if pd.notna(created) and pd.notna(closed):
                hrs = max((closed - created).total_seconds() / 3600, 0)
                a["resolution_hours"].append(hrs)

            # State
            if t.get("state_id") in (1, 2):  # new / open
                a["open_tickets"] += 1
            if t.get("escalation_at"):
                a["escalations_6m"] += 1

            # Category (Zammad "group" name)
            a["categories"].append(t.get("group", "other"))
            if t.get("satisfaction"):
                a["csat_scores"].append(float(t["satisfaction"]))

        rows = []
        for cid, a in agg.items():
            rows.append(
                {
                    "id": cid,
                    "num_tickets_6m": a["num_tickets_6m"],
                    "avg_resolution_hours": round(sum(a["resolution_hours"]) / max(len(a["resolution_hours"]), 1), 1),
                    "escalations_6m": a["escalations_6m"],
                    "open_tickets": a["open_tickets"],
                    "top_ticket_category": (
                        max(set(a["categories"]), key=a["categories"].count) if a["categories"] else "other"
                    ),
                    "post_ticket_csat": round(sum(a["csat_scores"]) / max(len(a["csat_scores"]), 1), 1),
                    "data_source": "support_api",
                }
            )

        df = pd.DataFrame(rows)
        logger.info("SupportApiDataSource aggregated %d customers from %d tickets", len(df), len(tickets))
        return df

    def validate(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        missing = self.REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"SupportApiDataSource missing columns: {missing}")
