# Feast Feature Store

Versioned feature definitions, offline training data retrieval, and
low-latency online serving for the churn prediction system.

## Architecture

```
Pipeline run (MultiSourcePipeline + build_feature_set)
        │
        ▼
src/data/feast_sink.py        ← writes data/feast/*.parquet
        │
        ▼
feast apply                   ← registers feature definitions in registry.db
        │
        ▼
feast materialize             ← loads parquet → online_store.db (SQLite dev / Redis prod)
        │
   ┌────┴────────────────────────────┐
   ▼                                  ▼
Training                          Serving
store.get_historical_features()   store.get_online_features()
→ point-in-time correct DF        → low-latency dict per customer
```

## Feature groups

| FeatureView | TTL | Features | Source |
|---|---|---|---|
| `bcg_features` | 30d | consumption, margin, tenure, products | BCG CSV |
| `price_features` | 30d | off/mid/peak price (mean, std, last) | BCG price CSV |
| `crm_features` | 7d | NPS, satisfaction, contact history | CRM source |
| `support_features` | 1d | ticket count, resolution, escalations | Support source |
| `billing_features` | 7d | late payments, outstanding, method | Billing source |
| `engineered_features` | 30d | risk score, growth rate, price spread | Computed |

## Feature services

| Service | Views included | Use case |
|---|---|---|
| `churn_prediction_v1` | All 6 views | Full prediction (all sources available) |
| `churn_prediction_bcg_only` | BCG + price + engineered | Fallback when CRM/Support/Billing unavailable |

## Setup

```bash
# 1. Run the data pipeline to generate Parquet files
make run-pipeline
# or directly:
python scripts/run_pipeline_e2e.py

# 2. Register feature definitions
python feature_store/apply.py
# or: cd feature_store && feast apply

# 3. Materialise to online store
python feature_store/materialize.py
# or: make feast-materialize

# 4. Verify
python -c "
from src.data.feast_store import get_online_features
df = get_online_features(['CL00001'])
print(df[['nps_score', 'cross_source_risk_score', 'cons_12m']])
"
```

## Production online store (Redis)

The default online store is SQLite (no extra services needed for dev).
For production, switch to Redis in `feature_store.yaml`:

```yaml
online_store:
  type: redis
  connection_string: "localhost:6379"
```

Then add Redis to docker-compose:
```bash
docker compose --profile redis up -d
```

## Using features in model training

```python
from src.data.feast_store import get_training_data
import pandas as pd
from datetime import timezone

# Entity DataFrame: one row per customer with event timestamp
entity_df = pd.DataFrame({
    "customer_id":    customer_ids,
    "event_timestamp": pd.Timestamp.now(tz=timezone.utc),
    "churn":          churn_labels,   # your target
})

# Returns point-in-time correct features joined to entity_df
training_df = get_training_data(entity_df, feature_service="churn_prediction_v1")
```

## Using features in inference

```python
from src.data.feast_store import get_online_features

# Low-latency retrieval from SQLite/Redis online store
features_df = get_online_features(["CL00001", "CL00042"])
# Pass to predictor directly (skips build_feature_set() recomputation)
```

## Updating features

After adding a new feature:
1. Add the column to `src/data/feast_sink.py` in the relevant group constant
2. Add the `Field` to `feature_store/features/feature_views.py`
3. Run `python feature_store/apply.py` to register the change
4. Run `python feature_store/materialize.py` to populate the online store
5. Bump `FEATURE_ENGINEERING_VERSION` in `api/predictor.py`
