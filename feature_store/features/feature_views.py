"""
feature_store/features/feature_views.py
-----------------------------------------
One FeatureView per data source group.

FeatureViews define:
  - Which entity they belong to (customer)
  - Which FileSource they read from
  - How long features are valid (ttl)
  - The schema (Field name + dtype)

TTL is set per source:
  - BCG / engineered: 30 days (pipeline runs nightly)
  - CRM: 7 days (updated frequently)
  - Support: 1 day (near real-time tickets)
  - Billing: 7 days (monthly billing cycle)
  - Price: 30 days (monthly price updates)
"""

from datetime import timedelta

from feast import FeatureView, Field
from feast.types import Bool, Float64, Int32, String

from feature_store.features.data_sources import (
    bcg_source,
    billing_source,
    crm_source,
    engineered_source,
    price_source,
    support_source,
)
from feature_store.features.entities import customer

# ── BCG core features ─────────────────────────────────────────────────────────

bcg_feature_view = FeatureView(
    name="bcg_features",
    entities=[customer],
    ttl=timedelta(days=30),
    schema=[
        Field(name="cons_12m", dtype=Float64),
        Field(name="cons_gas_12m", dtype=Float64),
        Field(name="cons_last_month", dtype=Float64),
        Field(name="imp_cons", dtype=Float64),
        Field(name="net_margin", dtype=Float64),
        Field(name="margin_gross_pow_ele", dtype=Float64),
        Field(name="margin_net_pow_ele", dtype=Float64),
        Field(name="num_years_antig", dtype=Float64),
        Field(name="pow_max", dtype=Float64),
        Field(name="nb_prod_act", dtype=Int32),
        Field(name="forecast_discount_energy", dtype=Float64),
        Field(name="has_gas", dtype=Bool),
        Field(name="channel_sales", dtype=String),
        Field(name="activity_new", dtype=String),
        Field(name="origin_up", dtype=String),
    ],
    source=bcg_source,
    description="BCG telco client core features",
    tags={"source": "bcg", "team": "data-engineering"},
)


# ── Price features ────────────────────────────────────────────────────────────

price_feature_view = FeatureView(
    name="price_features",
    entities=[customer],
    ttl=timedelta(days=30),
    schema=[
        Field(name="price_off_peak_var_mean", dtype=Float64),
        Field(name="price_off_peak_var_std", dtype=Float64),
        Field(name="price_off_peak_var_last", dtype=Float64),
        Field(name="price_peak_var_mean", dtype=Float64),
        Field(name="price_peak_var_std", dtype=Float64),
        Field(name="price_peak_var_last", dtype=Float64),
        Field(name="price_mid_peak_var_mean", dtype=Float64),
        Field(name="price_mid_peak_var_std", dtype=Float64),
        Field(name="price_mid_peak_var_last", dtype=Float64),
        Field(name="price_off_peak_fix_mean", dtype=Float64),
        Field(name="price_off_peak_fix_std", dtype=Float64),
        Field(name="price_off_peak_fix_last", dtype=Float64),
        Field(name="price_peak_fix_mean", dtype=Float64),
        Field(name="price_peak_fix_std", dtype=Float64),
        Field(name="price_peak_fix_last", dtype=Float64),
        Field(name="price_mid_peak_fix_mean", dtype=Float64),
        Field(name="price_mid_peak_fix_std", dtype=Float64),
        Field(name="price_mid_peak_fix_last", dtype=Float64),
    ],
    source=price_source,
    description="Aggregated monthly price features (mean, std, last period)",
    tags={"source": "bcg_price", "team": "data-engineering"},
)


# ── CRM features ──────────────────────────────────────────────────────────────

crm_feature_view = FeatureView(
    name="crm_features",
    entities=[customer],
    ttl=timedelta(days=7),
    schema=[
        Field(name="nps_score", dtype=Float64),
        Field(name="satisfaction_score", dtype=Float64),
        Field(name="num_contacts_6m", dtype=Int32),
        Field(name="last_contact_days_ago", dtype=Int32),
        Field(name="contract_type", dtype=String),
        Field(name="account_manager_changed", dtype=Bool),
    ],
    source=crm_source,
    description="CRM signals: NPS, satisfaction, contact history, contract type",
    tags={"source": "crm", "team": "crm-engineering"},
)


# ── Support features ──────────────────────────────────────────────────────────

support_feature_view = FeatureView(
    name="support_features",
    entities=[customer],
    ttl=timedelta(days=1),
    schema=[
        Field(name="num_tickets_6m", dtype=Int32),
        Field(name="avg_resolution_hours", dtype=Float64),
        Field(name="escalations_6m", dtype=Int32),
        Field(name="open_tickets", dtype=Int32),
        Field(name="top_ticket_category", dtype=String),
        Field(name="post_ticket_csat", dtype=Float64),
    ],
    source=support_source,
    description="Support ticket signals: volume, resolution, escalations",
    tags={"source": "support", "team": "support-engineering"},
)


# ── Billing features ──────────────────────────────────────────────────────────

billing_feature_view = FeatureView(
    name="billing_features",
    entities=[customer],
    ttl=timedelta(days=7),
    schema=[
        Field(name="num_late_payments_12m", dtype=Int32),
        Field(name="avg_days_late", dtype=Float64),
        Field(name="payment_method", dtype=String),
        Field(name="total_outstanding", dtype=Float64),
        Field(name="discount_applied", dtype=Bool),
        Field(name="discount_pct", dtype=Int32),
        Field(name="autopay_enrolled", dtype=Bool),
    ],
    source=billing_source,
    description="Billing signals: late payments, outstanding balance, payment method",
    tags={"source": "billing", "team": "billing-engineering"},
)


# ── Engineered features ───────────────────────────────────────────────────────

engineered_feature_view = FeatureView(
    name="engineered_features",
    entities=[customer],
    ttl=timedelta(days=30),
    schema=[
        Field(name="contract_duration_days", dtype=Float64),
        Field(name="is_long_term", dtype=Bool),
        Field(name="months_to_renewal", dtype=Float64),
        Field(name="cons_growth_rate", dtype=Float64),
        Field(name="gas_share", dtype=Float64),
        Field(name="high_consumption", dtype=Bool),
        Field(name="price_spread_var", dtype=Float64),
        Field(name="price_spread_fix", dtype=Float64),
        Field(name="margin_efficiency", dtype=Float64),
        Field(name="low_margin", dtype=Bool),
        Field(name="cross_source_risk_score", dtype=Float64),
        Field(name="engagement_score", dtype=Float64),
    ],
    source=engineered_source,
    description="Cross-source engineered features (risk score, tenure, growth)",
    tags={"source": "engineered", "team": "ml-engineering"},
)
