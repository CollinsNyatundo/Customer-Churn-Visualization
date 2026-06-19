import numpy as np
import pandas as pd
from src.config import settings
from src.data.sources.base import BaseDataSource

PAYMENT_METHODS = ["bank_transfer","credit_card","direct_debit","check"]

class BillingSource(BaseDataSource):
    name = "billing"
    REQUIRED_COLS = {"id","num_late_payments_12m","payment_method","total_outstanding"}

    def __init__(self, client_ids=None, seed=None):
        self.client_ids = client_ids
        self.seed = seed if seed is not None else settings.synthetic_seed+2

    def extract(self):
        if not self.client_ids: return pd.DataFrame()
        rng = np.random.default_rng(self.seed)
        n = len(self.client_ids)
        late = rng.integers(0, 8, n)
        discount_applied = rng.choice([True,False], n, p=[0.35,0.65])
        return pd.DataFrame({
            "id": self.client_ids,
            "num_late_payments_12m": late,
            "avg_days_late": np.where(late>0, rng.integers(1,60,n), 0).astype(float),
            "payment_method": rng.choice(PAYMENT_METHODS, n, p=[0.40,0.25,0.25,0.10]),
            "total_outstanding": rng.exponential(200, n).round(2),
            "discount_applied": discount_applied,
            "discount_pct": np.where(discount_applied, rng.choice([5,10,15,20],n), 0),
            "autopay_enrolled": rng.choice([True,False], n, p=[0.60,0.40]),
        })

    def validate(self, df):
        if df.empty: return
        missing = self.REQUIRED_COLS - set(df.columns)
        if missing: raise ValueError(f"BillingSource missing: {missing}")
        if (df["total_outstanding"] < 0).any():
            raise ValueError("negative outstanding")
