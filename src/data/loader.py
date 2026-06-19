"""
src/data/loader.py  —  Load raw CSV files. Path resolution is centralised here.
"""
from pathlib import Path
import pandas as pd
from src.config import settings

DATE_COLS_CLIENT = ["date_activ", "date_end", "date_modif_prod", "date_renewal"]


def load_client_data(path: Path | None = None) -> pd.DataFrame:
    filepath = path or settings.data_raw_dir / "client_data.csv"
    return pd.read_csv(filepath, parse_dates=DATE_COLS_CLIENT)


def load_price_data(path: Path | None = None) -> pd.DataFrame:
    filepath = path or settings.data_raw_dir / "price_data.csv"
    return pd.read_csv(filepath, parse_dates=["price_date"])


def load_all(
    client_path: Path | None = None,
    price_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_client_data(client_path), load_price_data(price_path)
