from pathlib import Path
import pandas as pd
from src.config import settings
from src.data.sources.base import BaseDataSource

class BCGClientSource(BaseDataSource):
    name = "bcg_client"
    DATE_COLS = ["date_activ","date_end","date_modif_prod","date_renewal"]
    REQUIRED_COLS = {"id","churn","cons_12m","num_years_antig"}

    def __init__(self, path=None):
        self.path = path or settings.data_raw_dir/"client_data.csv"

    def extract(self):
        return pd.read_csv(self.path, parse_dates=self.DATE_COLS)

    def validate(self, df):
        missing = self.REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"BCGClientSource missing: {missing}")

class BCGPriceSource(BaseDataSource):
    name = "bcg_price"
    REQUIRED_COLS = {"id","price_date","price_off_peak_var"}

    def __init__(self, path=None):
        self.path = path or settings.data_raw_dir/"price_data.csv"

    def extract(self):
        return pd.read_csv(self.path, parse_dates=["price_date"])

    def validate(self, df):
        missing = self.REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"BCGPriceSource missing: {missing}")
