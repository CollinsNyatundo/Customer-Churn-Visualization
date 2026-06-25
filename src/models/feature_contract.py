"""
src/models/feature_contract.py
--------------------------------
Strict feature contract enforcement — closes audit gap B15.

Every model training run and inference call validates that the
required features are present. Silently dropping unavailable features
causes silent prediction degradation; this makes it loud.
"""
from __future__ import annotations

import hashlib
import json
import logging

import pandas as pd

from src.features.engineering import get_feature_names

log = logging.getLogger(__name__)

# Raw (pre-engineering) features — must always be present for BCG models
RAW_NUMERIC_REQUIRED = [
    "cons_12m", "cons_gas_12m", "cons_last_month", "imp_cons",
    "net_margin", "margin_gross_pow_ele", "num_years_antig", "pow_max",
    "nb_prod_act", "forecast_discount_energy",
]
RAW_CATEGORICAL_REQUIRED = ["channel_sales", "activity_new", "origin_up"]

# Full training feature list (raw + engineered)
NUMERIC_FEATURES = RAW_NUMERIC_REQUIRED + [
    "nps_score", "satisfaction_score", "num_contacts_6m", "last_contact_days_ago",
    "num_tickets_6m", "avg_resolution_hours", "escalations_6m", "open_tickets",
    "post_ticket_csat",
    "num_late_payments_12m", "avg_days_late", "total_outstanding", "discount_pct",
] + [
    f for domain in ["tenure", "consumption", "price", "margin", "support", "billing", "multisource"]
    for f in get_feature_names()[domain]
    if f not in ("is_long_term", "renewal_urgency", "high_consumption", "discount_flag",
                 "low_margin", "high_ticket_volume", "high_outstanding")
]

BOOLEAN_AS_INT_FEATURES = [
    "is_long_term", "renewal_urgency", "high_consumption", "discount_flag",
    "low_margin", "high_ticket_volume", "high_outstanding",
    "is_kaggle_source", "is_bank_source", "is_bcg_source",
]

CATEGORICAL_FEATURES = RAW_CATEGORICAL_REQUIRED + [
    "contract_type", "payment_method", "top_ticket_category",
]

ALL_FEATURES = NUMERIC_FEATURES + BOOLEAN_AS_INT_FEATURES + CATEGORICAL_FEATURES
TARGET = "churn"


def validate_features(df: pd.DataFrame, strict: bool = False) -> dict[str, list[str]]:
    """
    Validate that expected features are present.

    Parameters
    ----------
    strict : if True, raise ValueError on any missing feature.
             if False, log WARNING and return missing list.

    Returns
    -------
    dict with "missing_numeric", "missing_categorical", "missing_bool"
    """
    missing_num  = [c for c in NUMERIC_FEATURES        if c not in df.columns]
    missing_cat  = [c for c in CATEGORICAL_FEATURES     if c not in df.columns]
    missing_bool = [c for c in BOOLEAN_AS_INT_FEATURES  if c not in df.columns]

    if missing_num or missing_cat or missing_bool:
        msg = (
            f"Feature contract violation — "
            f"missing numeric: {missing_num}, "
            f"categorical: {missing_cat}, "
            f"boolean: {missing_bool}"
        )
        if strict:
            raise ValueError(msg)
        log.warning(msg)

    return {
        "missing_numeric":    missing_num,
        "missing_categorical": missing_cat,
        "missing_bool":       missing_bool,
    }


def get_available(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (available_numeric + bool_as_int, available_categorical)."""
    num_and_bool = [
        c for c in NUMERIC_FEATURES + BOOLEAN_AS_INT_FEATURES
        if c in df.columns
    ]
    cat = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    return num_and_bool, cat


def feature_schema_hash(df: pd.DataFrame) -> str:
    """
    SHA-256 of the available feature column set.
    Log at training time; compare at inference to detect schema drift.
    """
    available = sorted([c for c in ALL_FEATURES if c in df.columns])
    return hashlib.sha256(json.dumps(available).encode()).hexdigest()[:16]
