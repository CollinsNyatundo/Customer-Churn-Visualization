# API Reference

Base URL: `http://localhost:8000`  
Interactive docs: `http://localhost:8000/docs`

## Authentication

Pass your API key in every request header:
```
X-API-Key: your-api-key-here
```

Configure valid keys via environment variable (comma-separated):
```bash
export API_KEYS="key1,key2,key3"
```

Omit `API_KEYS` entirely for development (all requests pass through).

---

## Endpoints

### `GET /health`
Liveness check. No auth required.

**Response**
```json
{
  "status": "ok",
  "model_loaded": true,
  "mlflow_uri": "http://localhost:5000",
  "auth_enabled": true
}
```

---

### `POST /predict`
Predict churn probability for a single customer.

**Request body** — all fields optional, defaults applied for missing values.

| Field | Type | Description |
|---|---|---|
| `cons_12m` | float | 12-month electricity consumption (kWh) |
| `nps_score` | float | NPS score [-100, 100] |
| `satisfaction_score` | float | Satisfaction [1.0, 5.0] |
| `contract_type` | string | `month-to-month`, `one-year`, `two-year` |
| `num_late_payments_12m` | int | Late payments in past 12 months |
| `num_tickets_6m` | int | Support tickets in past 6 months |
| *(+22 more fields)* | | See `/docs` for full schema |

**Response**
```json
{
  "churn_probability": 0.7231,
  "churn_prediction": true,
  "risk_tier": "high",
  "model_version": "Production-v3"
}
```

**Risk tiers**: `low` < 0.30 ≤ `medium` < 0.60 ≤ `high`

---

### `POST /predict/batch`
Score up to 500 customers in one call.

**Request**: JSON array of `CustomerFeatures` objects.  
**Response**: JSON array of `PredictionResponse` objects (same order).  
**Limit**: 400 Bad Request if > 500 items.

---

### `POST /explain`
Predict + return top-N SHAP feature attributions.

**Query params**: `top_n` (default 10, max 30)

**Response**
```json
{
  "churn_probability": 0.7231,
  "churn_prediction": true,
  "risk_tier": "high",
  "model_version": "Production-v3",
  "top_features": [
    {
      "feature": "nps_score",
      "raw_value": -85,
      "shap_value": 0.142,
      "direction": "increases_churn"
    },
    {
      "feature": "satisfaction_score",
      "raw_value": 1.2,
      "shap_value": 0.098,
      "direction": "increases_churn"
    }
  ]
}
```

---

## Error codes

| Code | Meaning |
|---|---|
| 401 | Missing `X-API-Key` header |
| 403 | Invalid API key |
| 400 | Validation error or batch limit exceeded |
| 422 | Field value out of allowed range |
| 503 | Model not yet loaded — run the pipeline first |
| 500 | Internal server error |
