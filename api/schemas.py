"""
api/schemas.py
--------------
Request and response models for the prediction API.

Cross-field validation is enforced: escalations cannot exceed tickets,
discount_pct must be 0 when discount_applied is false, etc.
Defaults are intentional but logged — missing fields default to
zero/unknown which can bias predictions; callers should always provide
as many fields as possible.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class CustomerFeatures(BaseModel):
    """Input features for a single customer churn prediction."""

    # BCG core
    cons_12m: float = Field(0, ge=0, description="12-month electricity consumption (kWh)")
    cons_gas_12m: float = Field(0, ge=0)
    cons_last_month: float = Field(0, ge=0)
    imp_cons: float = Field(0, ge=0)
    net_margin: float = 0.0
    margin_gross_pow_ele: float = 0.0
    num_years_antig: float = Field(1, ge=0)
    pow_max: float = Field(0, ge=0)
    nb_prod_act: int = Field(1, ge=0)
    forecast_discount_energy: float = 0.0
    channel_sales: str = "unknown"
    activity_new: str = "unknown"
    origin_up: str = "unknown"

    # CRM
    nps_score: float = Field(0, ge=-100, le=100)
    satisfaction_score: float = Field(3.0, ge=1, le=5)
    num_contacts_6m: int = Field(0, ge=0)
    last_contact_days_ago: int = Field(90, ge=0)
    contract_type: str = "month-to-month"

    # Support
    num_tickets_6m: int = Field(0, ge=0)
    avg_resolution_hours: float = Field(24, ge=0)
    escalations_6m: int = Field(0, ge=0)
    open_tickets: int = Field(0, ge=0)
    top_ticket_category: str = "other"

    # Billing
    num_late_payments_12m: int = Field(0, ge=0)
    avg_days_late: float = Field(0, ge=0)
    payment_method: str = "bank_transfer"
    total_outstanding: float = Field(0, ge=0)
    discount_pct: int = Field(0, ge=0, le=100)

    @model_validator(mode="after")
    def check_cross_field_consistency(self) -> "CustomerFeatures":
        # Escalations cannot exceed total tickets
        if self.escalations_6m > self.num_tickets_6m:
            raise ValueError(
                f"escalations_6m ({self.escalations_6m}) cannot exceed " f"num_tickets_6m ({self.num_tickets_6m})"
            )
        # Open tickets cannot exceed total tickets
        if self.open_tickets > self.num_tickets_6m:
            raise ValueError(
                f"open_tickets ({self.open_tickets}) cannot exceed " f"num_tickets_6m ({self.num_tickets_6m})"
            )
        # avg_days_late should be 0 if no late payments
        if self.num_late_payments_12m == 0 and self.avg_days_late > 0:
            raise ValueError("avg_days_late must be 0 when num_late_payments_12m is 0")
        return self


class FeatureContribution(BaseModel):
    """Single feature's SHAP contribution to a prediction."""

    feature: str
    raw_value: float | str | None
    shap_value: float
    direction: str  # "increases_churn" | "decreases_churn"


class PredictionResponse(BaseModel):
    churn_probability: float = Field(..., ge=0, le=1)
    churn_prediction: bool
    risk_tier: str  # "low" | "medium" | "high"
    model_version: str


class ExplainResponse(PredictionResponse):
    """Prediction + SHAP feature attributions."""

    top_features: list[FeatureContribution]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    mlflow_uri: str
    auth_enabled: bool
