"""
src/data/feast_sink.py
-----------------------
Writes pipeline output to the Feast offline store (Parquet files).

Called after MultiSourcePipeline.run() + build_feature_set() to persist
features in a Feast-compatible format:
  - customer_id column (entity key)
  - event_timestamp column (required by Feast for point-in-time joins)
  - created column (when the row was written)

One file per feature group so each FeatureView has its own source.
"""

from __future__ import annotations

import logging
from datetime import timezone
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

FEAST_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "feast"

BCG_COLS = [
    "cons_12m",
    "cons_gas_12m",
    "cons_last_month",
    "imp_cons",
    "net_margin",
    "margin_gross_pow_ele",
    "margin_net_pow_ele",
    "num_years_antig",
    "pow_max",
    "nb_prod_act",
    "forecast_discount_energy",
    "has_gas",
    "channel_sales",
    "activity_new",
    "origin_up",
]
PRICE_COLS = [
    "price_off_peak_var_mean",
    "price_off_peak_var_std",
    "price_off_peak_var_last",
    "price_peak_var_mean",
    "price_peak_var_std",
    "price_peak_var_last",
    "price_mid_peak_var_mean",
    "price_mid_peak_var_std",
    "price_mid_peak_var_last",
    "price_off_peak_fix_mean",
    "price_off_peak_fix_std",
    "price_off_peak_fix_last",
    "price_peak_fix_mean",
    "price_peak_fix_std",
    "price_peak_fix_last",
    "price_mid_peak_fix_mean",
    "price_mid_peak_fix_std",
    "price_mid_peak_fix_last",
]
CRM_COLS = [
    "nps_score",
    "satisfaction_score",
    "num_contacts_6m",
    "last_contact_days_ago",
    "contract_type",
    "account_manager_changed",
]
SUPPORT_COLS = [
    "num_tickets_6m",
    "avg_resolution_hours",
    "escalations_6m",
    "open_tickets",
    "top_ticket_category",
    "post_ticket_csat",
]
BILLING_COLS = [
    "num_late_payments_12m",
    "avg_days_late",
    "payment_method",
    "total_outstanding",
    "discount_applied",
    "discount_pct",
    "autopay_enrolled",
]
ENGINEERED_COLS = [
    "contract_duration_days",
    "is_long_term",
    "months_to_renewal",
    "cons_growth_rate",
    "gas_share",
    "high_consumption",
    "price_spread_var",
    "price_spread_fix",
    "margin_efficiency",
    "low_margin",
    "cross_source_risk_score",
    "engagement_score",
]


def _prep(df: pd.DataFrame, cols: list[str], timestamp: pd.Timestamp) -> pd.DataFrame:
    """Add Feast-required columns and select only the relevant feature columns."""
    available = [c for c in cols if c in df.columns]
    missing = set(cols) - set(available)
    if missing:
        log.warning("Feast sink: missing columns will be skipped: %s", missing)

    id_col = "customer_id" if "customer_id" in df.columns else "id"
    out = df[[id_col] + available].copy()
    out = out.rename(columns={id_col: "customer_id"})
    out["event_timestamp"] = timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp
    out["created"] = pd.Timestamp.now(tz=timezone.utc)
    return out


def write_feast_parquet(
    df: pd.DataFrame,
    timestamp: pd.Timestamp | None = None,
) -> dict[str, Path]:
    """
    Write one Parquet file per feature group to data/feast/.

    Parameters
    ----------
    df        : fully merged + feature-engineered DataFrame
    timestamp : event timestamp for all rows (defaults to now)

    Returns
    -------
    dict mapping group name → written path
    """
    FEAST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ts = timestamp or pd.Timestamp.now()

    groups = {
        "bcg_features": BCG_COLS,
        "price_features": PRICE_COLS,
        "crm_features": CRM_COLS,
        "support_features": SUPPORT_COLS,
        "billing_features": BILLING_COLS,
        "engineered_features": ENGINEERED_COLS,
    }

    written: dict[str, Path] = {}
    for name, cols in groups.items():
        path = FEAST_DATA_DIR / f"{name}.parquet"
        _prep(df, cols, ts).to_parquet(path, index=False)
        log.info("Feast sink: wrote %s → %s (%d rows)", name, path, len(df))
        written[name] = path

    return written
