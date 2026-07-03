import pandas as pd

from src.config import settings
from src.data.sources.base import BaseDataSource


class BCGClientSource(BaseDataSource):
    name = "bcg_client"
    DATE_COLS = ["date_activ", "date_end", "date_modif_prod", "date_renewal"]
    REQUIRED_COLS = {"id", "churn", "cons_12m", "num_years_antig"}

    def __init__(self, path=None):
        self.path = path or settings.data_raw_dir / "client_data.csv"

    def extract(self):
        return pd.read_csv(self.path, parse_dates=self.DATE_COLS)

    def validate(self, df):
        import logging

        log = logging.getLogger(__name__)
        missing = self.REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"BCGClientSource missing: {missing}")
        # Duplicate ID check
        dupes = df["id"].duplicated().sum()
        if dupes:
            raise ValueError(f"BCGClientSource: {dupes} duplicate customer IDs")
        # Date consistency
        if "date_activ" in df.columns and "date_end" in df.columns:
            bad = (df["date_end"] < df["date_activ"]).sum()
            if bad:
                log.warning("BCGClientSource: %d rows where date_end < date_activ", bad)
        # Negative consumption
        for col in ["cons_12m", "cons_gas_12m", "cons_last_month", "imp_cons"]:
            if col in df.columns and (df[col] < 0).any():
                log.warning("BCGClientSource: negative values in '%s'", col)


class BCGPriceSource(BaseDataSource):
    name = "bcg_price"
    REQUIRED_COLS = {"id", "price_date", "price_off_peak_var"}

    def __init__(self, path=None):
        self.path = path or settings.data_raw_dir / "price_data.csv"

    def extract(self):
        return pd.read_csv(self.path, parse_dates=["price_date"])

    def validate(self, df):
        missing = self.REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"BCGPriceSource missing: {missing}")
