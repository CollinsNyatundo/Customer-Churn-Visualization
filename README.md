# Customer Churn Visualization — Production ML System

> End-to-end churn prediction platform: multi-source data pipeline → MLflow experiment tracking → Prefect orchestration → FastAPI prediction API with SHAP explainability → Evidently AI drift monitoring → Dash dashboard → Kubernetes deployment.

[![CI](https://github.com/CollinsNyatundo/Customer-Churn-Visualization/actions/workflows/ci.yml/badge.svg)](https://github.com/CollinsNyatundo/Customer-Churn-Visualization/actions/workflows/ci.yml)
[![CD](https://github.com/CollinsNyatundo/Customer-Churn-Visualization/actions/workflows/cd.yml/badge.svg)](https://github.com/CollinsNyatundo/Customer-Churn-Visualization/actions/workflows/cd.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Architecture

```
Data Sources (BCG CSV · CRM · Support · Billing)
        │
        ▼
MultiSourcePipeline  →  Feature Engineering (63 cols, 13 engineered)
        │
        ├──▶  Prefect Orchestration  ──▶  MLflow Tracking + Registry
        │
        ├──▶  FastAPI  /predict · /predict/batch · /explain · /health
        │          └── API key auth · SHAP explainability · OTel tracing
        │
        ├──▶  Dash Dashboard  (5-tab multi-source view)
        │
        └──▶  Evidently Drift Monitor  ──▶  Slack + Email Alerts
```

---

## Quick start

```bash
git clone https://github.com/CollinsNyatundo/Customer-Churn-Visualization.git
cd Customer-Churn-Visualization

make install-dev          # deps + pre-commit hooks
cp .env.example .env      # configure environment

make docker-up            # start MLflow (5000) + Prefect (4200)
make run-pipeline         # ingest → features → train → register
make run-api              # FastAPI at http://localhost:8000/docs
make run-dashboard        # Dash at http://localhost:8050
```

---

## Services

| Service | Port | URL |
|---|---|---|
| FastAPI Prediction API | 8000 | http://localhost:8000/docs |
| Dash Dashboard | 8050 | http://localhost:8050 |
| MLflow Tracking | 5000 | http://localhost:5000 |
| Prefect UI | 4200 | http://localhost:4200 |
| Locust Load Test UI | 8089 | http://localhost:8089 |

---

## Data sources

| Source | Type | Key signals |
|---|---|---|
| `client_data.csv` | BCG CSV | consumption, margins, tenure, churn label |
| `price_data.csv` | BCG CSV | off/mid/peak pricing (aggregated monthly → per client) |
| CRM | Synthetic generator | NPS, satisfaction, contact history, contract type |
| Support | Synthetic generator | ticket volume, resolution time, escalations |
| Billing | Synthetic generator | late payments, outstanding balance, payment method |

> Synthetic sources mirror real CRM/helpdesk/billing APIs. Swap `extract()` in each source class to connect live systems.

---

## API usage

```bash
# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"nps_score": -80, "num_late_payments_12m": 5, "contract_type": "month-to-month"}'

# Prediction + SHAP explanation
curl -X POST "http://localhost:8000/explain?top_n=5" \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"nps_score": -80, "satisfaction_score": 1.2}'

# Health check
curl http://localhost:8000/health
```

Full API reference → [`docs/api.md`](docs/api.md)

---

## Project structure

```
.
├── api/                        # FastAPI prediction service
│   ├── main.py                 #   routes + middleware
│   ├── auth.py                 #   API key authentication
│   ├── predictor.py            #   MLflow model loader
│   ├── explain.py              #   SHAP explainability
│   └── schemas.py              #   Pydantic models
├── src/
│   ├── config.py               # pydantic-settings
│   ├── logging_config.py       # structlog JSON/console
│   ├── data/
│   │   ├── sources/            # BaseDataSource + BCG/CRM/Support/Billing
│   │   ├── pipeline.py         # MultiSourcePipeline
│   │   └── preprocessing.py   # clean · aggregate · merge · null_report
│   ├── features/
│   │   └── engineering.py      # 13 engineered features
│   ├── models/
│   │   └── churn_model.py      # GBM + MLflow tracking + registry
│   ├── tracking/
│   │   └── mlflow_tracker.py   # context managers for MLflow runs
│   ├── pipeline/
│   │   ├── flows.py            # Prefect @flow — data + full + drift
│   │   └── tasks.py            # Prefect @task wrappers
│   ├── monitoring/
│   │   ├── drift.py            # Evidently AI drift detection
│   │   ├── monitor_flow.py     # Prefect drift flow + alert dispatch
│   │   └── alerts.py           # Slack + email alerters
│   ├── telemetry/
│   │   └── tracing.py          # OpenTelemetry span helpers
│   └── visualizations/
│       ├── eda.py              # Plotly figure factory (10 charts)
│       └── dashboard.py        # Dash 5-tab layout + callbacks
├── tests/
│   ├── conftest.py             # shared fixtures (500-row synthetic dataset)
│   ├── test_sources.py         # 13 data source tests
│   ├── test_preprocessing.py   # 12 cleaning/merge tests
│   ├── test_engineering.py     # 15 feature engineering tests
│   ├── test_pipeline.py        # 4 integration tests (mocked)
│   ├── test_api.py             # 10 endpoint tests
│   ├── test_auth.py            # 7 authentication tests
│   ├── test_explain.py         # 6 explainability tests
│   └── benchmarks/
│       └── test_performance.py # 5 pytest-benchmark tests
├── k8s/                        # Kubernetes manifests
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── secrets.yaml            # template only
│   ├── mlflow-deployment.yaml
│   ├── api-deployment.yaml     # + HPA (2–8 replicas)
│   ├── dashboard-deployment.yaml
│   ├── ingress.yaml
│   └── README.md
├── docs/
│   ├── api.md                  # full endpoint reference
│   ├── architecture.md         # system design + module map
│   └── development.md          # setup, workflow, adding sources
├── .github/workflows/
│   ├── ci.yml                  # lint + test matrix (py3.11/3.12) + Docker build
│   └── cd.yml                  # build + push to GHCR + Slack notify
├── locustfile.py               # load test + demo data generator
├── app.py                      # Dash entry point
├── Makefile                    # dev commands
├── Dockerfile                  # multi-stage build
├── docker-compose.yml          # MLflow + Prefect + Dashboard
├── prefect.yaml                # 3 scheduled deployments
├── dvc.yaml                    # data versioning pipeline
├── pyproject.toml              # black + ruff + pytest config
├── .pre-commit-config.yaml     # pre-commit hooks
└── requirements.txt            # pinned dependencies
```

---

## Make commands

```bash
make help             # show all commands
make install          # pip install -r requirements.txt
make install-dev      # install + pre-commit hooks
make lint             # ruff + black --check
make format           # black + ruff --fix
make test             # 70+ tests
make test-cov         # tests + HTML coverage at reports/coverage/
make benchmark        # pytest-benchmark performance suite
make run-api          # uvicorn api.main:app --reload
make run-dashboard    # python app.py
make run-pipeline     # python -m src.pipeline.flows
make run-drift        # python -m src.monitoring.monitor_flow
make docker-up        # docker compose up -d
make generate-demo    # 2000-row demo prediction CSV
make clean            # remove caches + artefacts
```

---

## Load testing

```bash
# Web UI
locust -f locustfile.py --host http://localhost:8000

# Headless: 50 users for 60 seconds
locust -f locustfile.py --host http://localhost:8000 \
  --headless -u 50 -r 5 -t 60s --csv=reports/locust_results

# Demo CSV (no server needed)
python locustfile.py --generate --rows 5000 --high-risk-pct 0.25
```

---

## Kubernetes deployment

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secrets.yaml        # fill in values first
kubectl apply -f k8s/mlflow-deployment.yaml
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/dashboard-deployment.yaml
kubectl apply -f k8s/ingress.yaml
```

API HPA auto-scales from **2 → 8 replicas** at 70% CPU.  
Full guide → [`k8s/README.md`](k8s/README.md)

---

## Documentation

| Doc | Contents |
|---|---|
| [`docs/api.md`](docs/api.md) | All endpoints, request/response schemas, error codes |
| [`docs/architecture.md`](docs/architecture.md) | System diagram, data flow, module map |
| [`docs/development.md`](docs/development.md) | Setup, workflow, adding new data sources, env vars |

---

## Tech stack

| Layer | Technology |
|---|---|
| Data pipeline | pandas, pyarrow, DVC |
| Feature engineering | numpy, pandas |
| ML model | scikit-learn (GradientBoostingClassifier) |
| Experiment tracking | MLflow |
| Orchestration | Prefect 3 |
| API | FastAPI + Pydantic + Uvicorn |
| Explainability | SHAP |
| Drift monitoring | Evidently AI |
| Alerting | Slack webhooks + SMTP |
| Tracing | OpenTelemetry |
| Dashboard | Plotly Dash |
| EDA | Sweetviz |
| Logging | structlog |
| Load testing | Locust |
| Testing | pytest + pytest-cov + pytest-benchmark |
| Linting | ruff + black + mypy |
| CI/CD | GitHub Actions |
| Containerisation | Docker + docker-compose |
| Orchestration (k8s) | Kubernetes + HPA + Ingress + cert-manager |

---

## License

MIT — see [LICENSE](LICENSE)
