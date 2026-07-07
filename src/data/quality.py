"""
src/data/quality.py
--------------------
Pandera data quality contracts for each source.

Run automatically in MultiSourcePipeline after extraction.
Raises SchemaError on breach — makes data problems loud and early.

Fix applied after code review
------------------------------
Every non-key column below is marked required=False. Pandera's default
is required=True for every declared column, meaning a DataFrame missing
even one optional field (e.g. a partial payload, or a source that hasn't
populated every column yet) would fail validation entirely on a
"column_in_dataframe" error — regardless of whether the columns that
WERE present were perfectly valid. This module had zero test coverage
before this review, so that default was never actually exercised against
real data. Only the entity key ("id") and, for bcg_client, the target
("churn") are genuinely required; every quality/range check should apply
only when the column is present, matching the "validate what's there"
philosophy used elsewhere (e.g. validate_features()).
"""

from __future__ import annotations

import logging

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

log = logging.getLogger(__name__)


# ── BCG client schema ─────────────────────────────────────────────────────────
BCG_CLIENT_SCHEMA = DataFrameSchema(
    columns={
        "id": Column(str, nullable=False, unique=True),
        "churn": Column(bool, nullable=False),
        "cons_12m": Column(float, checks=Check.ge(0), nullable=True, required=False),
        "cons_gas_12m": Column(float, checks=Check.ge(0), nullable=True, required=False),
        "net_margin": Column(float, nullable=True, required=False),
        "num_years_antig": Column(float, checks=Check.ge(0), nullable=True, required=False),
        "pow_max": Column(float, checks=Check.ge(0), nullable=True, required=False),
        "nb_prod_act": Column(int, checks=Check.ge(1), nullable=True, required=False),
    },
    coerce=True,
)

# ── CRM schema ────────────────────────────────────────────────────────────────
CRM_SCHEMA = DataFrameSchema(
    columns={
        "id": Column(str, nullable=False),
        "nps_score": Column(float, checks=[Check.ge(-100), Check.le(100)], nullable=True, required=False),
        "satisfaction_score": Column(float, checks=[Check.ge(1), Check.le(5)], nullable=True, required=False),
        "num_contacts_6m": Column(int, checks=Check.ge(0), nullable=True, required=False),
        "last_contact_days_ago": Column(int, checks=Check.ge(0), nullable=True, required=False),
    },
    coerce=True,
)

# ── Support schema ────────────────────────────────────────────────────────────
SUPPORT_SCHEMA = DataFrameSchema(
    columns={
        "id": Column(str, nullable=False),
        "num_tickets_6m": Column(int, checks=Check.ge(0), nullable=True, required=False),
        "avg_resolution_hours": Column(float, checks=Check.ge(0), nullable=True, required=False),
        "escalations_6m": Column(int, checks=Check.ge(0), nullable=True, required=False),
        "open_tickets": Column(int, checks=Check.ge(0), nullable=True, required=False),
        "post_ticket_csat": Column(float, checks=[Check.ge(1), Check.le(5)], nullable=True, required=False),
    },
    coerce=True,
)

# ── Billing schema ────────────────────────────────────────────────────────────
BILLING_SCHEMA = DataFrameSchema(
    columns={
        "id": Column(str, nullable=False),
        "num_late_payments_12m": Column(int, checks=Check.ge(0), nullable=True, required=False),
        "avg_days_late": Column(float, checks=Check.ge(0), nullable=True, required=False),
        "total_outstanding": Column(float, checks=Check.ge(0), nullable=True, required=False),
        "discount_pct": Column(int, checks=[Check.ge(0), Check.le(100)], nullable=True, required=False),
    },
    coerce=True,
)

_SCHEMAS = {
    "bcg_client": BCG_CLIENT_SCHEMA,
    "crm": CRM_SCHEMA,
    "support": SUPPORT_SCHEMA,
    "billing": BILLING_SCHEMA,
}


def validate_source(df: pd.DataFrame, source_name: str, strict: bool = False) -> pd.DataFrame:
    """
    Validate a source DataFrame against its Pandera schema.

    Parameters
    ----------
    df          : DataFrame to validate
    source_name : one of 'bcg_client', 'crm', 'support', 'billing'
    strict      : if True, raise on any violation; if False, log WARNING

    Returns
    -------
    Validated (and coerced) DataFrame
    """
    schema = _SCHEMAS.get(source_name)
    if schema is None:
        log.debug("No quality schema for source '%s' — skipping", source_name)
        return df

    try:
        validated = schema.validate(df, lazy=True)
        log.debug("Data quality: '%s' passed validation (%d rows)", source_name, len(df))
        return validated
    except pa.errors.SchemaErrors as exc:
        failures = exc.failure_cases
        msg = f"Data quality breach in '{source_name}': {len(failures)} failures\n{failures}"
        if strict:
            raise ValueError(msg) from exc
        log.warning(msg)
        return df
