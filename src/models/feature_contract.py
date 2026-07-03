"""
src/models/feature_contract.py
--------------------------------
P0.3 Fix: Feature contract auto-derived from engineering.get_feature_names().
No more hard-coded lists that must be manually synced.
"""

from __future__ import annotations

import hashlib
import json
import logging

import pandas as pd

log = logging.getLogger(__name__)
TARGET = "churn"

_BOOLEAN_FEATURES = frozenset(
    {
        "is_long_term",
        "renewal_urgency",
        "high_consumption",
        "discount_flag",
        "low_margin",
        "high_ticket_volume",
        "high_outstanding",
        "is_kaggle_source",
        "is_bank_source",
        "is_bcg_source",
    }
)

_RAW_NUMERIC = [
    "cons_12m",
    "cons_gas_12m",
    "cons_last_month",
    "imp_cons",
    "net_margin",
    "margin_gross_pow_ele",
    "num_years_antig",
    "pow_max",
    "nb_prod_act",
    "forecast_discount_energy",
]
_MULTISOURCE_NUMERIC = [
    "nps_score",
    "satisfaction_score",
    "num_contacts_6m",
    "last_contact_days_ago",
    "num_tickets_6m",
    "avg_resolution_hours",
    "escalations_6m",
    "open_tickets",
    "post_ticket_csat",
    "num_late_payments_12m",
    "avg_days_late",
    "total_outstanding",
    "discount_pct",
]
_RAW_CATEGORICAL = ["channel_sales", "activity_new", "origin_up"]
_SOURCE_CATEGORICAL = ["contract_type", "payment_method", "top_ticket_category"]


def _build() -> tuple[list, list, list]:
    from src.features.engineering import get_feature_names

    numeric, boolean = [], []
    for feats in get_feature_names().values():
        for f in feats:
            (boolean if f in _BOOLEAN_FEATURES else numeric).append(f)
    return (
        _RAW_NUMERIC + _MULTISOURCE_NUMERIC + numeric,
        boolean,
        _RAW_CATEGORICAL + _SOURCE_CATEGORICAL,
    )


NUMERIC_FEATURES, BOOLEAN_AS_INT_FEATURES, CATEGORICAL_FEATURES = _build()
ALL_FEATURES = NUMERIC_FEATURES + BOOLEAN_AS_INT_FEATURES + CATEGORICAL_FEATURES


def validate_features(df: pd.DataFrame, strict: bool = False) -> dict[str, list[str]]:
    missing = {
        "missing_numeric": [c for c in NUMERIC_FEATURES if c not in df.columns],
        "missing_categorical": [c for c in CATEGORICAL_FEATURES if c not in df.columns],
        "missing_bool": [c for c in BOOLEAN_AS_INT_FEATURES if c not in df.columns],
    }
    if any(missing.values()):
        msg = f"Feature contract violation: {missing}"
        if strict:
            raise ValueError(msg)
        log.warning(msg)
    return missing


def get_available(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    num = [c for c in NUMERIC_FEATURES + BOOLEAN_AS_INT_FEATURES if c in df.columns]
    cat = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    return num, cat


def feature_schema_hash(df: pd.DataFrame) -> str:
    available = sorted(c for c in ALL_FEATURES if c in df.columns)
    return hashlib.sha256(json.dumps(available).encode()).hexdigest()[:16]
