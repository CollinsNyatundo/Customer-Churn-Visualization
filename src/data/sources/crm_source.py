import numpy as np
import pandas as pd

from src.config import settings
from src.data.sources.base import BaseDataSource


class CRMSource(BaseDataSource):
    name = "crm"
    CONTRACT_TYPES = ["month-to-month", "one-year", "two-year"]
    REQUIRED_COLS = {"id", "nps_score", "contract_type", "satisfaction_score"}

    def __init__(self, client_ids=None, seed=None):
        self.client_ids = client_ids
        self.seed = seed if seed is not None else settings.synthetic_seed

    def extract(self):
        if not self.client_ids:
            return pd.DataFrame()
        rng = np.random.default_rng(self.seed)
        n = len(self.client_ids)
        return pd.DataFrame(
            {
                "id": self.client_ids,
                "nps_score": rng.integers(-100, 101, n),
                "satisfaction_score": rng.uniform(1, 5, n).round(1),
                "num_contacts_6m": rng.integers(0, 15, n),
                "contract_type": rng.choice(self.CONTRACT_TYPES, n, p=[0.50, 0.30, 0.20]),
                "last_contact_days_ago": rng.integers(1, 365, n),
                "account_manager_changed": rng.choice([True, False], n, p=[0.20, 0.80]),
            }
        )

    def validate(self, df):
        if df.empty:
            return
        missing = self.REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"CRMSource missing: {missing}")
        if not df["nps_score"].between(-100, 100).all():
            raise ValueError("nps_score out of range")
        if not df["satisfaction_score"].between(1, 5).all():
            raise ValueError("satisfaction_score out of range")
