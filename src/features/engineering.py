"""src/features/engineering.py — Domain-driven feature construction."""

import numpy as np
import pandas as pd


def add_tenure_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ref = pd.Timestamp.today().normalize()
    if "date_activ" in df.columns and "date_end" in df.columns:
        df["contract_duration_days"] = (df["date_end"] - df["date_activ"]).dt.days.clip(lower=0)
        df["is_long_term"] = df["contract_duration_days"] > 365
    if "date_renewal" in df.columns:
        df["months_to_renewal"] = ((df["date_renewal"] - ref).dt.days / 30.44).round(1)
    return df


def add_consumption_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "cons_last_month" in df.columns and "cons_12m" in df.columns:
        df["cons_growth_rate"] = ((df["cons_last_month"] * 12 - df["cons_12m"]) / (df["cons_12m"] + 1)).round(4)
    if "cons_gas_12m" in df.columns and "cons_12m" in df.columns:
        total = df["cons_12m"] + df["cons_gas_12m"]
        df["gas_share"] = (df["cons_gas_12m"] / total.replace(0, np.nan)).round(4)
    if "cons_12m" in df.columns:
        df["high_consumption"] = df["cons_12m"] > df["cons_12m"].quantile(0.75)
    return df


def add_price_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "price_off_peak_var_mean" in df.columns and "price_peak_var_mean" in df.columns:
        df["price_spread_var"] = (df["price_peak_var_mean"] - df["price_off_peak_var_mean"]).round(4)
    if "price_off_peak_fix_mean" in df.columns and "price_peak_fix_mean" in df.columns:
        df["price_spread_fix"] = (df["price_peak_fix_mean"] - df["price_off_peak_fix_mean"]).round(4)
    if "forecast_discount_energy" in df.columns:
        df["discount_flag"] = df["forecast_discount_energy"] > 0
    return df


def add_margin_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "net_margin" in df.columns and "margin_gross_pow_ele" in df.columns:
        df["margin_efficiency"] = (df["net_margin"] / (df["margin_gross_pow_ele"] + 1)).round(4)
    if "net_margin" in df.columns:
        df["low_margin"] = df["net_margin"] < df["net_margin"].quantile(0.25)
    return df


def add_multisource_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features derived from cross-source signals (CRM + Support + Billing)."""
    df = df.copy()
    # Risk score: combines NPS, tickets, and late payments into a single signal
    components = []
    if "nps_score" in df.columns:
        components.append((-df["nps_score"]).clip(lower=0) / 100)
    if "num_tickets_6m" in df.columns:
        components.append((df["num_tickets_6m"] / df["num_tickets_6m"].max()).fillna(0))
    if "num_late_payments_12m" in df.columns:
        components.append((df["num_late_payments_12m"] / df["num_late_payments_12m"].max()).fillna(0))
    if components:
        import numpy as np

        df["cross_source_risk_score"] = np.mean(components, axis=0).round(4)
    # Engagement score: satisfaction + low contact recency
    if "satisfaction_score" in df.columns and "last_contact_days_ago" in df.columns:
        df["engagement_score"] = (
            df["satisfaction_score"] / 5 - (df["last_contact_days_ago"] / 365).clip(0, 1) * 0.5
        ).round(4)
    return df


def build_feature_set(df: pd.DataFrame) -> pd.DataFrame:
    """Chain all feature engineering steps."""
    df = add_tenure_features(df)
    df = add_consumption_features(df)
    df = add_price_sensitivity(df)
    df = add_margin_features(df)
    df = add_multisource_features(df)
    return df
