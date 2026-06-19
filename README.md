# BCG Customer Churn — Production ML System

End-to-end production-grade churn prediction platform built on top of the BCG Telco dataset, extended with a multi-source data pipeline, MLflow experiment tracking, Prefect orchestration, FastAPI prediction API, Evidently drift monitoring, and a Locust load test suite.

## Architecture

```
Data Sources (BCG CSV + CRM + Support + Billing)
    ↓  MultiSourcePipeline  (src/data/pipeline.py)
Feature Engineering          (src/features/engineering.py)
    ↓
┌─────────────────┬──────────────────────┐
│  Prefect Flows  │   MLflow Tracking    │
│  (orchestration)│   (experiments +     │
│                 │    model registry)   │
└────────┬────────┴──────────────────────┘
         ↓
┌────────────────────────────────────────┐
│  FastAPI  /predict  /predict/batch     │
│  Dash Dashboard  (5 source tabs)       │
│  Evidently Drift Monitor               │
└────────────────────────────────────────┘
```

## Quick start

```bash
# 1. Clone and install
git clone https://github.com/CollinsNyatundo/BCG-customer-churn-visualization.git
cd BCG-customer-churn-visualization
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env

# 3. Start services (MLflow + Prefect + Dashboard)
docker compose up -d

# 4. Run the full pipeline (ingest → features → train → register)
python -m src.pipeline.flows

# 5. Start the prediction API
uvicorn api.main:app --reload --port 8000

# 6. Run tests
pytest tests/ -v --cov=src --cov=api

# 7. Load test / generate demo data
locust -f locustfile.py --host http://localhost:8000          # web UI
python locustfile.py --generate --rows 2000                  # demo CSV
```

## Project structure

```
src/
├── config.py                  # Pydantic settings
├── logging_config.py          # structlog setup
├── data/
│   ├── sources/               # BaseDataSource + BCG/CRM/Support/Billing
│   ├── pipeline.py            # MultiSourcePipeline
│   ├── loader.py
│   └── preprocessing.py
├── features/engineering.py    # 13 engineered features
├── models/churn_model.py      # GBM + MLflow tracking
├── tracking/mlflow_tracker.py # context managers
├── pipeline/
│   ├── flows.py               # Prefect @flow definitions
│   └── tasks.py               # Prefect @task definitions
├── monitoring/
│   ├── drift.py               # Evidently drift detection
│   └── monitor_flow.py        # Prefect drift flow
└── visualizations/
    ├── eda.py                 # Plotly figure factory
    └── dashboard.py           # Dash layout + callbacks
api/
├── main.py                    # FastAPI app
├── predictor.py               # MLflow model loader
└── schemas.py                 # Pydantic request/response
tests/                         # 54 tests, 62% coverage
locustfile.py                  # Load test + demo data generator
```

## Data sources

| Source | Type | Key features |
|---|---|---|
| BCG client_data.csv | CSV | consumption, margins, tenure, churn label |
| BCG price_data.csv | CSV | off-peak/peak/mid-peak pricing (aggregated) |
| CRM | Synthetic | NPS score, satisfaction, contact history, contract type |
| Support | Synthetic | ticket volume, resolution time, escalations |
| Billing | Synthetic | late payments, outstanding balance, payment method |

## Services

| Service | Port | URL |
|---|---|---|
| Dash Dashboard | 8050 | http://localhost:8050 |
| FastAPI Prediction API | 8000 | http://localhost:8000/docs |
| MLflow Tracking | 5000 | http://localhost:5000 |
| Prefect UI | 4200 | http://localhost:4200 |
| Locust UI | 8089 | http://localhost:8089 |

## License

MIT
