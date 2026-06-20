"""
src/data/preprocessing.py
--------------------------
Cleaning, aggregation, and merge helpers.
All functions are pure (input → output, no side effects).
"""

from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)

_CHURN_VALID_VALUES = {0, 1, True, False, "Yes", "No", "0", "1"}
_CHURN_MAP = {
    0: False,
    1: True,
    "0": False,
    "1": True,
    "Yes": True,
    "No": False,
    True: True,
    False: False,
}
_HAS_GAS_MAP = {"t": True, "f": False, True: True, False: False}


def validate_churn_column(df: pd.DataFrame) -> None:
    """Raise ValueError if churn contains unexpected values."""
    if "churn" not in df.columns:
        raise ValueError("DataFrame is missing required 'churn' column.")
    unexpected = set(df["churn"].dropna().unique()) - _CHURN_VALID_VALUES
    if unexpected:
        raise ValueError(
            f"Unexpected values in 'churn' column: {unexpected}. " f"Expected one of: {_CHURN_VALID_VALUES}"
        )


def clean_client_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean BCG client data:
    - Validate and normalise churn label
    - Drop exact duplicates
    - Clip negative consumption values to 0
    - Fill missing categoricals with 'unknown'
    - Map has_gas → bool, with unmapped values replaced by False + warning
    """
    df = df.drop_duplicates().copy()

    # ── Churn: explicit validation then normalise ──────────────────────────
    validate_churn_column(df)
    df["churn"] = df["churn"].map(_CHURN_MAP).astype(bool)

    # ── Consumption: must be non-negative ────────────────────────────────
    for col in [
        "cons_12m",
        "cons_gas_12m",
        "cons_last_month",
        "forecast_cons_12m",
        "forecast_cons_year",
        "imp_cons",
    ]:
        if col in df.columns:
            neg = (df[col] < 0).sum()
            if neg:
                log.warning("Clipping %d negative values in '%s'", neg, col)
            df[col] = df[col].clip(lower=0)

    # ── Categoricals: fill NaN ────────────────────────────────────────────
    for col in ["activity_new", "channel_sales", "origin_up"]:
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype("category")

    # ── has_gas: map t/f → bool, warn on unmapped values ─────────────────
    if "has_gas" in df.columns:
        unmapped = ~df["has_gas"].isin(_HAS_GAS_MAP.keys()) & df["has_gas"].notna()
        if unmapped.any():
            log.warning(
                "%d unexpected 'has_gas' values (will be set False): %s",
                unmapped.sum(),
                df.loc[unmapped, "has_gas"].unique().tolist(),
            )
        df["has_gas"] = df["has_gas"].map(_HAS_GAS_MAP).fillna(False).astype(bool)

    return df


def aggregate_price_data(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate monthly price rows to one row per client.
    Computes mean, std, and last-period value per price column to preserve
    some temporal signal (last-period = most recent pricing state).
    """
    price_cols = [
        "price_off_peak_var",
        "price_peak_var",
        "price_mid_peak_var",
        "price_off_peak_fix",
        "price_peak_fix",
        "price_mid_peak_fix",
    ]
    existing = [c for c in price_cols if c in price_df.columns]
    if not existing:
        log.warning("No price columns found in price_df; returning id-only frame.")
        return price_df[["id"]].drop_duplicates()

    # Sort so "last" is genuinely the most recent price date
    sorted_df = price_df.sort_values("price_date") if "price_date" in price_df.columns else price_df

    agg_mean_std = sorted_df.groupby("id")[existing].agg(["mean", "std"])
    agg_mean_std.columns = ["_".join(c) for c in agg_mean_std.columns]

    agg_last = sorted_df.groupby("id")[existing].last()
    agg_last.columns = [f"{c}_last" for c in agg_last.columns]

    return pd.concat([agg_mean_std, agg_last], axis=1).reset_index()


def merge_datasets(client_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    return client_df.merge(aggregate_price_data(price_df), on="id", how="left")


def null_report(df: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame:
    """
    Report missing values per column.

    Parameters
    ----------
    threshold : float | None
        If set, raises ValueError if any column exceeds this missing fraction.
    """
    total = df.isnull().sum()
    pct = (total / len(df) * 100).round(2)
    report = (
        pd.DataFrame({"missing_count": total, "missing_pct": pct})
        .loc[lambda x: x["missing_count"] > 0]
        .sort_values("missing_pct", ascending=False)
    )
    if threshold is not None:
        bad = report[report["missing_pct"] / 100 > threshold]
        if not bad.empty:
            raise ValueError(f"Columns exceed null threshold ({threshold:.0%}): " f"{bad.index.tolist()}")
    return report
