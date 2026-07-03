"""
src/features/engineering.py
----------------------------
Domain-driven feature construction for churn prediction.

Design rules
------------
1. Pure functions — return new DataFrame, never modify in place.
2. FeatureThresholds must be frozen from training set and passed at
   inference to prevent batch-distribution skew.
3. Temporal features accept explicit reference_date for reproducibility.
4. All normalisation denominators are guarded against zero/NaN.
5. Feature version is tracked in FEATURE_ENGINEERING_VERSION.

Feature count: 35+ engineered features across 7 domains.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

FEATURE_ENGINEERING_VERSION = "2.0.0"


# ── Feature threshold contract ────────────────────────────────────────────────


@dataclass
class FeatureThresholds:
    """
    Frozen thresholds computed on the training set.
    Pass to build_feature_set() at inference to prevent batch-distribution skew.
    """

    high_consumption_threshold: float = 0.0
    low_margin_threshold: float = 0.0
    high_tickets_threshold: float = 0.0
    high_outstanding_threshold: float = 0.0

    # Normalisation maxima for cross_source_risk_score components.
    # _safe_normalize(series) divides by series.max() — on a single-row
    # inference request that degenerates to always-0-or-1. Freezing
    # training-set maxima makes the risk score meaningful per-customer.
    num_tickets_6m_max: float = 1.0
    num_late_payments_12m_max: float = 1.0

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "FeatureThresholds":
        return cls(
            high_consumption_threshold=float(df["cons_12m"].quantile(0.75)) if "cons_12m" in df.columns else 0.0,
            low_margin_threshold=float(df["net_margin"].quantile(0.25)) if "net_margin" in df.columns else 0.0,
            high_tickets_threshold=(
                float(df["num_tickets_6m"].quantile(0.75)) if "num_tickets_6m" in df.columns else 0.0
            ),
            high_outstanding_threshold=(
                float(df["total_outstanding"].quantile(0.75)) if "total_outstanding" in df.columns else 0.0
            ),
            num_tickets_6m_max=(
                float(df["num_tickets_6m"].max())
                if "num_tickets_6m" in df.columns and df["num_tickets_6m"].max() > 0
                else 1.0
            ),
            num_late_payments_12m_max=(
                float(df["num_late_payments_12m"].max())
                if "num_late_payments_12m" in df.columns and df["num_late_payments_12m"].max() > 0
                else 1.0
            ),
        )

    def schema_hash(self) -> str:
        """SHA-256 of threshold values — use to detect contract drift."""
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ── Safe normalisation ────────────────────────────────────────────────────────


def _safe_normalize(series: pd.Series, label: str) -> pd.Series:
    max_val = series.max()
    if pd.isna(max_val) or max_val == 0:
        log.warning("Cannot normalise '%s': max=%s. Returning zeros.", label, max_val)
        return pd.Series(0.0, index=series.index)
    return series / max_val


def _safe_div(a: pd.Series, b: pd.Series, label: str = "") -> pd.Series:
    b_safe = b.replace(0, np.nan)
    result = a / b_safe
    if result.isnull().any() and label:
        log.debug("Safe divide '%s': %d zeros replaced with NaN.", label, (b == 0).sum())
    return result


# ── 1. Tenure & contract features ────────────────────────────────────────────


def add_tenure_features(
    df: pd.DataFrame,
    reference_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    df = df.copy()
    ref = reference_date or pd.Timestamp.today().normalize()
    if reference_date is None:
        log.debug("months_to_renewal using runtime date %s — pass reference_date for reproducibility.", ref)

    if "date_activ" in df.columns and "date_end" in df.columns:
        df["contract_duration_days"] = (df["date_end"] - df["date_activ"]).dt.days.clip(lower=0)
        df["is_long_term"] = df["contract_duration_days"] > 365

        # How far through the contract is the customer? (0=start, 1=end)
        elapsed = (ref - df["date_activ"]).dt.days.clip(lower=0)
        df["contract_completion_pct"] = (
            _safe_div(elapsed, df["contract_duration_days"].replace(0, np.nan), "contract_completion")
            .round(4)
            .clip(0, 1)
        )
    elif "num_years_antig" in df.columns:
        # Proxy fallback for the single-customer API path: date_activ/date_end
        # aren't in CustomerFeatures schema, but contract_duration_days and
        # is_long_term are the model's strongest tenure predictors. Without
        # this proxy they'd be NaN on every API prediction, silently
        # crippling inference (the model would fall back to weaker signals).
        log.debug(
            "date_activ/date_end absent — deriving contract_duration_days "
            "from num_years_antig proxy (API single-customer path)."
        )
        df["contract_duration_days"] = (df["num_years_antig"] * 365).clip(lower=0)
        df["is_long_term"] = df["num_years_antig"] > 1.0
        df["contract_completion_pct"] = np.nan  # genuinely unknowable without dates

    if "date_renewal" in df.columns:
        days_to_renewal = (df["date_renewal"] - ref).dt.days
        df["months_to_renewal"] = (days_to_renewal / 30.44).round(1)
        df["renewal_urgency"] = days_to_renewal <= 90  # within 3 months

    if "date_modif_prod" in df.columns:
        df["days_since_modification"] = (ref - df["date_modif_prod"]).dt.days.clip(lower=0)

    return df


# ── 2. Consumption features ───────────────────────────────────────────────────


def add_consumption_features(
    df: pd.DataFrame,
    thresholds: Optional[FeatureThresholds] = None,
) -> pd.DataFrame:
    df = df.copy()

    if "cons_last_month" in df.columns and "cons_12m" in df.columns:
        # Month-over-month growth rate
        df["cons_growth_rate"] = _safe_div(
            df["cons_last_month"] * 12 - df["cons_12m"], df["cons_12m"] + 1, "cons_growth"
        ).round(4)

    if "cons_12m" in df.columns:
        threshold = thresholds.high_consumption_threshold if thresholds else float(df["cons_12m"].quantile(0.75))
        if not thresholds:
            log.debug("high_consumption threshold from batch (%.2f) — freeze for inference.", threshold)
        df["high_consumption"] = df["cons_12m"] > threshold

        # Consumption per active product — usage intensity
        if "nb_prod_act" in df.columns:
            df["cons_per_product"] = _safe_div(df["cons_12m"], df["nb_prod_act"], "cons_per_product").round(2)

        # Revenue efficiency: net margin per kWh consumed
        if "net_margin" in df.columns:
            df["revenue_per_kwh"] = _safe_div(df["net_margin"], df["cons_12m"], "revenue_per_kwh").round(6)

        # Power utilisation: how much of subscribed capacity is used?
        if "pow_max" in df.columns:
            annual_capacity = df["pow_max"] * 8760  # kWh if running at max
            df["power_utilisation"] = (
                _safe_div(df["cons_12m"], annual_capacity, "power_utilisation").clip(0, 1).round(4)
            )

    if "cons_gas_12m" in df.columns and "cons_12m" in df.columns:
        total = df["cons_12m"] + df["cons_gas_12m"]
        df["gas_share"] = _safe_div(df["cons_gas_12m"], total.replace(0, np.nan), "gas_share").round(4)

    # Forecast accuracy: how well does the forecast match actual consumption?
    if "forecast_cons_12m" in df.columns and "cons_12m" in df.columns:
        df["forecast_vs_actual"] = _safe_div(
            (df["forecast_cons_12m"] - df["cons_12m"]).abs(), df["cons_12m"] + 1, "forecast_accuracy"
        ).round(4)

    return df


# ── 3. Price sensitivity features ────────────────────────────────────────────


def add_price_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "price_off_peak_var_mean" in df.columns and "price_peak_var_mean" in df.columns:
        df["price_spread_var"] = (df["price_peak_var_mean"] - df["price_off_peak_var_mean"]).round(4)
        # Peak-to-off-peak ratio: >1 means peak is more expensive (normal); extremes signal anomaly
        df["price_peak_ratio"] = _safe_div(
            df["price_peak_var_mean"], df["price_off_peak_var_mean"], "price_peak_ratio"
        ).round(4)

    if "price_off_peak_fix_mean" in df.columns and "price_peak_fix_mean" in df.columns:
        df["price_spread_fix"] = (df["price_peak_fix_mean"] - df["price_off_peak_fix_mean"]).round(4)

    # Price trend: direction of price change (last period vs mean)
    if "price_off_peak_var_last" in df.columns and "price_off_peak_var_mean" in df.columns:
        df["price_trend_var"] = (df["price_off_peak_var_last"] - df["price_off_peak_var_mean"]).round(6)

    if "price_off_peak_fix_last" in df.columns and "price_off_peak_fix_mean" in df.columns:
        df["price_trend_fix"] = (df["price_off_peak_fix_last"] - df["price_off_peak_fix_mean"]).round(6)

    # Price volatility: coefficient of variation for variable price
    if "price_off_peak_var_std" in df.columns and "price_off_peak_var_mean" in df.columns:
        df["price_volatility"] = _safe_div(
            df["price_off_peak_var_std"], df["price_off_peak_var_mean"], "price_volatility"
        ).round(4)

    if "forecast_discount_energy" in df.columns:
        df["discount_flag"] = df["forecast_discount_energy"] > 0

    return df


# ── 4. Margin & financial features ────────────────────────────────────────────


def add_margin_features(
    df: pd.DataFrame,
    thresholds: Optional[FeatureThresholds] = None,
) -> pd.DataFrame:
    df = df.copy()

    if "net_margin" in df.columns and "margin_gross_pow_ele" in df.columns:
        df["margin_efficiency"] = _safe_div(
            df["net_margin"], df["margin_gross_pow_ele"] + 1, "margin_efficiency"
        ).round(4)

    if "net_margin" in df.columns:
        threshold = thresholds.low_margin_threshold if thresholds else float(df["net_margin"].quantile(0.25))
        if not thresholds:
            log.debug("low_margin threshold from batch (%.2f) — freeze for inference.", threshold)
        df["low_margin"] = df["net_margin"] < threshold

        # CLV proxy: how much total value has this customer delivered?
        if "num_years_antig" in df.columns:
            df["clv_proxy"] = (df["net_margin"] * df["num_years_antig"]).round(2)

        # Margin per product
        if "nb_prod_act" in df.columns:
            df["margin_per_product"] = _safe_div(df["net_margin"], df["nb_prod_act"], "margin_per_product").round(2)

    return df


# ── 5. Cross-source behavioural features ─────────────────────────────────────


def add_support_features(
    df: pd.DataFrame,
    thresholds: Optional[FeatureThresholds] = None,
) -> pd.DataFrame:
    """Features derived from the Support source."""
    df = df.copy()

    if "num_tickets_6m" in df.columns:
        threshold = thresholds.high_tickets_threshold if thresholds else float(df["num_tickets_6m"].quantile(0.75))
        df["high_ticket_volume"] = df["num_tickets_6m"] > threshold

        # Escalation rate: fraction of tickets that escalated
        if "escalations_6m" in df.columns:
            df["escalation_rate"] = (
                _safe_div(df["escalations_6m"], df["num_tickets_6m"], "escalation_rate").fillna(0).round(4)
            )

        # Ticket rate per product: support burden per service
        if "nb_prod_act" in df.columns:
            df["ticket_rate_per_product"] = _safe_div(
                df["num_tickets_6m"], df["nb_prod_act"], "ticket_rate_per_product"
            ).round(4)

    # Service quality composite: higher = worse service experience
    components = []
    if "avg_resolution_hours" in df.columns:
        components.append(_safe_normalize(df["avg_resolution_hours"].clip(lower=0), "resolution_hours"))
    if "open_tickets" in df.columns:
        components.append(_safe_normalize(df["open_tickets"].clip(lower=0), "open_tickets"))
    if "post_ticket_csat" in df.columns:
        # Invert CSAT: low satisfaction → high service quality deficit
        components.append((5 - df["post_ticket_csat"].clip(1, 5)) / 4)

    if components:
        df["service_quality_deficit"] = np.mean(np.stack([c.values for c in components], axis=1), axis=1).round(4)

    return df


def add_billing_features(
    df: pd.DataFrame,
    thresholds: Optional[FeatureThresholds] = None,
) -> pd.DataFrame:
    """Features derived from the Billing source."""
    df = df.copy()

    if "num_late_payments_12m" in df.columns:
        df["payment_reliability"] = (1 - (df["num_late_payments_12m"] / 12).clip(0, 1)).round(4)

        threshold = (
            thresholds.high_outstanding_threshold
            if thresholds
            else float(df.get("total_outstanding", pd.Series([0])).quantile(0.75))
        )
        if "total_outstanding" in df.columns:
            df["high_outstanding"] = df["total_outstanding"] > threshold

    # Financial distress composite: late payments + outstanding balance
    components = []
    if "num_late_payments_12m" in df.columns:
        components.append(_safe_normalize(df["num_late_payments_12m"].clip(lower=0), "late_payments"))
    if "total_outstanding" in df.columns:
        components.append(_safe_normalize(df["total_outstanding"].clip(lower=0), "outstanding"))

    if components:
        df["financial_distress_score"] = np.mean(np.stack([c.values for c in components], axis=1), axis=1).round(4)

    return df


# ── 6. Multi-source interaction features ─────────────────────────────────────


def add_multisource_features(
    df: pd.DataFrame,
    thresholds: Optional["FeatureThresholds"] = None,
) -> pd.DataFrame:
    """
    Cross-source risk and engagement scores.

    Normalisation maxima for num_tickets_6m and num_late_payments_12m are
    frozen from the training set (via thresholds) when provided. Without
    this, a single-row inference request would normalise against its own
    value — always producing 0 or 1, which is meaningless. When thresholds
    is None (e.g. during training before freezing), falls back to batch max.

    Scores are heuristic relative rankings, not calibrated probabilities.
    """
    df = df.copy()

    # Cross-source risk: combines NPS deficit, ticket pressure, payment failure
    risk_components: list[pd.Series] = []
    if "nps_score" in df.columns:
        risk_components.append((-df["nps_score"]).clip(lower=0).div(100).fillna(0.0))
    if "num_tickets_6m" in df.columns:
        if thresholds is not None:
            comp = (df["num_tickets_6m"] / thresholds.num_tickets_6m_max).clip(0, 1).fillna(0.0)
        else:
            comp = _safe_normalize(df["num_tickets_6m"], "num_tickets_6m").fillna(0.0)
        risk_components.append(comp)
    if "num_late_payments_12m" in df.columns:
        if thresholds is not None:
            comp = (df["num_late_payments_12m"] / thresholds.num_late_payments_12m_max).clip(0, 1).fillna(0.0)
        else:
            comp = _safe_normalize(df["num_late_payments_12m"], "num_late_payments_12m").fillna(0.0)
        risk_components.append(comp)

    if risk_components:
        df["cross_source_risk_score"] = np.mean(np.stack([c.values for c in risk_components], axis=1), axis=1).round(4)
    else:
        df["cross_source_risk_score"] = 0.0

    # Engagement score: equal-weight average of normalised satisfaction, recency,
    # and contact frequency. Weights are intentionally uniform (1/3 each) rather
    # than fitted — use only as a relative ranking signal, not a calibrated score.
    engagement_components: list[pd.Series] = []
    if "satisfaction_score" in df.columns:
        sat_norm = (df["satisfaction_score"].clip(1, 5) - 1) / 4  # [0, 1]
        engagement_components.append(sat_norm)
    if "last_contact_days_ago" in df.columns:
        recency_score = 1 - (df["last_contact_days_ago"] / 365).clip(0, 1)
        engagement_components.append(recency_score)
    if "num_contacts_6m" in df.columns:
        contact_score = (_safe_normalize(df["num_contacts_6m"], "contacts_6m") * 0.5).clip(0, 1)
        engagement_components.append(contact_score)

    if engagement_components:
        df["engagement_score"] = np.mean(np.stack([c.values for c in engagement_components], axis=1), axis=1).round(4)

    # Revenue at risk: how much margin is exposed given behavioural signals?
    if "net_margin" in df.columns and "cross_source_risk_score" in df.columns:
        df["revenue_at_risk"] = (df["net_margin"].clip(lower=0) * df["cross_source_risk_score"]).round(2)

    # NPS × tenure interaction: long-tenured dissatisfied customers are high risk
    if "nps_score" in df.columns and "num_years_antig" in df.columns:
        df["nps_tenure_interaction"] = ((-df["nps_score"].clip(upper=0)) * df["num_years_antig"]).round(2)

    return df


# ── 7. Source harmonisation flag ──────────────────────────────────────────────


def add_source_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a source indicator so models can learn source-specific patterns.
    Prevents contamination from mixed-dataset rows silently biasing predictions.
    """
    df = df.copy()
    if "data_source" in df.columns:
        df["is_kaggle_source"] = (df["data_source"] == "kaggle_telco").astype(int)
        df["is_bank_source"] = (df["data_source"] == "bank_churn").astype(int)
        df["is_bcg_source"] = (~df["data_source"].isin(["kaggle_telco", "bank_churn"])).astype(int)
    return df


# ── Build full feature set ────────────────────────────────────────────────────


def build_feature_set(
    df: pd.DataFrame,
    thresholds: Optional[FeatureThresholds] = None,
    reference_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Apply all feature engineering steps.

    Parameters
    ----------
    thresholds    : frozen training-set thresholds (prevents batch skew at inference)
    reference_date: explicit reference date for temporal features
    """
    df = add_tenure_features(df, reference_date=reference_date)
    df = add_consumption_features(df, thresholds=thresholds)
    df = add_price_sensitivity(df)
    df = add_margin_features(df, thresholds=thresholds)
    df = add_support_features(df, thresholds=thresholds)
    df = add_billing_features(df, thresholds=thresholds)
    df = add_multisource_features(df, thresholds=thresholds)
    df = add_source_flags(df)
    return df


def get_feature_names() -> dict[str, list[str]]:
    """
    Return the canonical feature name registry.
    Used to enforce strict feature contracts at training and inference.
    """
    return {
        "tenure": [
            "contract_duration_days",
            "is_long_term",
            "months_to_renewal",
            "renewal_urgency",
            "contract_completion_pct",
            "days_since_modification",
        ],
        "consumption": [
            "cons_growth_rate",
            "high_consumption",
            "gas_share",
            "cons_per_product",
            "revenue_per_kwh",
            "power_utilisation",
            "forecast_vs_actual",
        ],
        "price": [
            "price_spread_var",
            "price_spread_fix",
            "price_peak_ratio",
            "price_trend_var",
            "price_trend_fix",
            "price_volatility",
            "discount_flag",
        ],
        "margin": [
            "margin_efficiency",
            "low_margin",
            "clv_proxy",
            "margin_per_product",
        ],
        "support": [
            "high_ticket_volume",
            "escalation_rate",
            "ticket_rate_per_product",
            "service_quality_deficit",
        ],
        "billing": [
            "payment_reliability",
            "high_outstanding",
            "financial_distress_score",
        ],
        "multisource": [
            "cross_source_risk_score",
            "engagement_score",
            "revenue_at_risk",
            "nps_tenure_interaction",
        ],
        "source_flags": [
            "is_kaggle_source",
            "is_bank_source",
            "is_bcg_source",
        ],
    }
