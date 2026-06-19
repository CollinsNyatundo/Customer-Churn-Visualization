"""api/schemas.py — Request and response models for the prediction API."""

from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    cons_12m: float = Field(0, ge=0)
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
    nps_score: float = Field(0, ge=-100, le=100)
    satisfaction_score: float = Field(3.0, ge=1, le=5)
    num_contacts_6m: int = Field(0, ge=0)
    last_contact_days_ago: int = Field(90, ge=0)
    contract_type: str = "month-to-month"
    num_tickets_6m: int = Field(0, ge=0)
    avg_resolution_hours: float = Field(24, ge=0)
    escalations_6m: int = Field(0, ge=0)
    open_tickets: int = Field(0, ge=0)
    top_ticket_category: str = "other"
    num_late_payments_12m: int = Field(0, ge=0)
    avg_days_late: float = Field(0, ge=0)
    payment_method: str = "bank_transfer"
    total_outstanding: float = Field(0, ge=0)
    discount_pct: int = Field(0, ge=0, le=100)


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: bool
    risk_tier: str
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    mlflow_uri: str
