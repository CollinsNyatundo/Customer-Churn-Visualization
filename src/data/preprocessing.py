"""src/data/preprocessing.py — Cleaning, aggregation, and merge helpers."""

import pandas as pd


def clean_client_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().copy()
    df["churn"] = df["churn"].astype(bool)
    for col in ["cons_12m", "cons_gas_12m", "cons_last_month", "forecast_cons_12m", "forecast_cons_year", "imp_cons"]:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)
    for col in ["activity_new", "channel_sales", "origin_up"]:
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype("category")
    if "has_gas" in df.columns:
        df["has_gas"] = df["has_gas"].map({"t": True, "f": False, True: True, False: False})
    return df


def aggregate_price_data(price_df: pd.DataFrame) -> pd.DataFrame:
    price_cols = [
        "price_off_peak_var",
        "price_peak_var",
        "price_mid_peak_var",
        "price_off_peak_fix",
        "price_peak_fix",
        "price_mid_peak_fix",
    ]
    existing = [c for c in price_cols if c in price_df.columns]
    agg = price_df.groupby("id")[existing].agg(["mean", "std"])
    agg.columns = ["_".join(c) for c in agg.columns]
    return agg.reset_index()


def merge_datasets(client_df, price_df):
    return client_df.merge(aggregate_price_data(price_df), on="id", how="left")


def null_report(df):
    total = df.isnull().sum()
    pct = (total / len(df) * 100).round(2)
    return (
        pd.DataFrame({"missing_count": total, "missing_pct": pct})
        .loc[lambda x: x["missing_count"] > 0]
        .sort_values("missing_pct", ascending=False)
    )
