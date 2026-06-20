"""
src/features/engineering.py
----------------------------
Domain-driven feature construction.

Design rules
------------
1. All functions are pure (return new DataFrame, never modify in place).
2. Quantile thresholds (low_margin, high_consumption) must be frozen from
   the training set and passed explicitly at inference to prevent batch-
   dependent skew.  Use FeatureThresholds for that contract.
3. Temporal features (months_to_renewal) accept an explicit reference_date
   so they are reproducible across runs and time-stable at inference.
4. Normalisation denominators are guarded against zero/NaN.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class FeatureThresholds:
    """
    Frozen thresholds computed on the training set.
    Pass to build_feature_set() at inference to prevent batch-distribution skew.

    Usage
    -----
    # Training
    thresholds = FeatureThresholds.from_dataframe(train_df)
    train_features = build_feature_set(train_df, thresholds=thresholds)

    # Inference (single row or batch)
    features = build_feature_set(inference_df, thresholds=thresholds)
    """

    high_consumption_threshold: float = 0.0
    low_margin_threshold: float = 0.0

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "FeatureThresholds":
        return cls(
            high_consumption_threshold=float(df["cons_12m"].quantile(0.75)) if "cons_12m" in df.columns else 0.0,
            low_margin_threshold=float(df["net_margin"].quantile(0.25)) if "net_margin" in df.columns else 0.0,
        )


def _safe_normalize(series: pd.Series, label: str) -> pd.Series:
    """Divide series by its max, guarding against zero/NaN denominators."""
    max_val = series.max()
    if pd.isna(max_val) or max_val == 0:
        log.warning("Cannot normalise '%s': max is %s. Returning zeros.", label, max_val)
        return pd.Series(0.0, index=series.index)
    return series / max_val


def add_tenure_features(
    df: pd.DataFrame,
    reference_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Derive contract duration and time-to-renewal.

    Parameters
    ----------
    reference_date : explicit 'today' for reproducibility.
                     Defaults to pd.Timestamp.today() when None (dev/notebook use).
    """
    df = df.copy()
    ref = reference_date or pd.Timestamp.today().normalize()

    if reference_date is None:
        log.debug("months_to_renewal using runtime date %s — pass reference_date for reproducibility.", ref)

    if "date_activ" in df.columns and "date_end" in df.columns:
        df["contract_duration_days"] = (df["date_end"] - df["date_activ"]).dt.days.clip(lower=0)
        df["is_long_term"] = df["contract_duration_days"] > 365

    if "date_renewal" in df.columns:
        df["months_to_renewal"] = ((df["date_renewal"] - ref).dt.days / 30.44).round(1)

    return df


def add_consumption_features(
    df: pd.DataFrame,
    thresholds: Optional[FeatureThresholds] = None,
) -> pd.DataFrame:
    df = df.copy()

    if "cons_last_month" in df.columns and "cons_12m" in df.columns:
        df["cons_growth_rate"] = ((df["cons_last_month"] * 12 - df["cons_12m"]) / (df["cons_12m"] + 1)).round(4)

    if "cons_gas_12m" in df.columns and "cons_12m" in df.columns:
        total = df["cons_12m"] + df["cons_gas_12m"]
        df["gas_share"] = (df["cons_gas_12m"] / total.replace(0, np.nan)).round(4)

    if "cons_12m" in df.columns:
        threshold = thresholds.high_consumption_threshold if thresholds else float(df["cons_12m"].quantile(0.75))
        if thresholds is None:
            log.debug(
                "high_consumption threshold computed from batch (%.2f). "
                "Pass FeatureThresholds at inference to freeze.",
                threshold,
            )
        df["high_consumption"] = df["cons_12m"] > threshold

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


def add_margin_features(
    df: pd.DataFrame,
    thresholds: Optional[FeatureThresholds] = None,
) -> pd.DataFrame:
    df = df.copy()

    if "net_margin" in df.columns and "margin_gross_pow_ele" in df.columns:
        df["margin_efficiency"] = (df["net_margin"] / (df["margin_gross_pow_ele"] + 1)).round(4)

    if "net_margin" in df.columns:
        threshold = thresholds.low_margin_threshold if thresholds else float(df["net_margin"].quantile(0.25))
        if thresholds is None:
            log.debug(
                "low_margin threshold computed from batch (%.2f). " "Pass FeatureThresholds at inference to freeze.",
                threshold,
            )
        df["low_margin"] = df["net_margin"] < threshold

    return df


def add_multisource_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-source risk and engagement scores.

    Each component is normalised to [0, 1] using _safe_normalize(),
    which guards against zero-max and all-NaN columns.  The resulting
    cross_source_risk_score is a heuristic index, not a calibrated
    probability — treat it as a relative ranking signal.
    """
    df = df.copy()
    components: list[pd.Series] = []

    if "nps_score" in df.columns:
        # NPS in [-100, 100]: invert and scale so high negative NPS → high risk
        components.append((-df["nps_score"]).clip(lower=0).div(100).fillna(0.0))

    if "num_tickets_6m" in df.columns:
        components.append(_safe_normalize(df["num_tickets_6m"], "num_tickets_6m").fillna(0.0))

    if "num_late_payments_12m" in df.columns:
        components.append(_safe_normalize(df["num_late_payments_12m"], "num_late_payments_12m").fillna(0.0))

    if components:
        stacked = np.stack([c.values for c in components], axis=1)
        df["cross_source_risk_score"] = np.mean(stacked, axis=1).round(4)
    else:
        df["cross_source_risk_score"] = 0.0

    if "satisfaction_score" in df.columns and "last_contact_days_ago" in df.columns:
        df["engagement_score"] = (
            df["satisfaction_score"] / 5 - (df["last_contact_days_ago"] / 365).clip(0, 1) * 0.5
        ).round(4)

    return df


def build_feature_set(
    df: pd.DataFrame,
    thresholds: Optional[FeatureThresholds] = None,
    reference_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Apply all feature engineering steps in sequence.

    Parameters
    ----------
    thresholds    : frozen training-set thresholds for batch-stable features.
                    If None, thresholds are computed from the current batch
                    (acceptable for training, risky for inference).
    reference_date: explicit reference date for temporal features.
                    If None, pd.Timestamp.today() is used (logs a warning).
    """
    df = add_tenure_features(df, reference_date=reference_date)
    df = add_consumption_features(df, thresholds=thresholds)
    df = add_price_sensitivity(df)
    df = add_margin_features(df, thresholds=thresholds)
    df = add_multisource_features(df)
    return df
