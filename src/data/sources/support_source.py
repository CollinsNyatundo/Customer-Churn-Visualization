import numpy as np
import pandas as pd

from src.config import settings
from src.data.sources.base import BaseDataSource

TICKET_CATEGORIES = ["billing", "technical", "contract", "outage", "pricing", "other"]


class SupportSource(BaseDataSource):
    name = "support"
    REQUIRED_COLS = {"id", "num_tickets_6m", "avg_resolution_hours", "escalations_6m"}

    def __init__(self, client_ids=None, seed=None):
        self.client_ids = client_ids
        self.seed = seed if seed is not None else settings.synthetic_seed + 1

    def extract(self):
        if not self.client_ids:
            return pd.DataFrame()
        rng = np.random.default_rng(self.seed)
        n = len(self.client_ids)
        num_tickets = rng.integers(0, 20, n)
        return pd.DataFrame(
            {
                "id": self.client_ids,
                "num_tickets_6m": num_tickets,
                "avg_resolution_hours": (rng.exponential(24, n) + num_tickets * 0.5).round(1),
                "escalations_6m": np.minimum(rng.integers(0, 5, n), num_tickets),
                "open_tickets": np.minimum(rng.integers(0, 4, n), num_tickets),
                "top_ticket_category": rng.choice(TICKET_CATEGORIES, n, p=[0.30, 0.25, 0.20, 0.10, 0.10, 0.05]),
                "post_ticket_csat": rng.uniform(1, 5, n).round(1),
            }
        )

    def validate(self, df):
        if df.empty:
            return
        missing = self.REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"SupportSource missing: {missing}")
        if (df["escalations_6m"] > df["num_tickets_6m"]).any():
            raise ValueError("escalations > tickets")
