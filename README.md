# Customer Churn Visualization — Production ML System

> End-to-end churn prediction platform: multi-source data pipeline → MLflow experiment tracking → Prefect orchestration → FastAPI prediction API with SHAP explainability → Evidently AI drift monitoring → Dash dashboard → Kubernetes deployment.

[![CI](https://github.com/CollinsNyatundo/Customer-Churn-Visualization/actions/workflows/ci.yml/badge.svg)](https://github.com/CollinsNyatundo/Customer-Churn-Visualization/actions/workflows/ci.yml)
[![CD](https://github.com/CollinsNyatundo/Customer-Churn-Visualization/actions/workflows/cd.yml/badge.svg)](https://github.com/CollinsNyatundo/Customer-Churn-Visualization/actions/workflows/cd.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org)
[![Tests](https://img.shields.io/badge/tests-97%20passing-brightgreen)](#testing)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## What this is

A **modular, audited ML scaffold** for customer churn prediction, built to production standards:

- **Multi-source ingestion** — BCG CSV files as ground truth, with synthetic CRM / Support / Billing generators by default and real API-backed connectors (EspoCRM, Zammad, NovaBilling/Lago) as opt-in replacements
- **Frozen feature contracts** — `FeatureThresholds` are computed on the training set and injected at inference, preventing batch-distribution skew
- **Explicit failure modes** — MLflow failures raise loudly; production startup fails hard if the model is missing; auth dev-bypass logs a warning
- **Vectorised serving** — batch prediction is a single `predict_proba()` call, not a Python loop
- **Real E2E test coverage** — tests that train an actual model and assert directional sanity on live predictions, not just API wiring with mocked predictors

### What it is not (yet)

- A fully wired production ingest stack — CRM / Support / Billing default to **synthetic generators**; real APIs require `USE_CRM_API=true` + a running service
- A feature store — `FeatureThresholds` are serialised to MLflow but not versioned in a dedicated store
- A multi-tenant or authn-hardened system — CORS and API key auth are configurable but require explicit production setup

---

## Architecture

```
Data Sources
  BCG client_data.csv + price_data.csv   ← ground truth, always on
  CRM  (synthetic default / EspoCRM API) ← USE_CRM_API=true
  Support (synthetic / Zammad API)        ← USE_SUPPORT_API=true
  Billing (synthetic / NovaBilling API)   ← USE_BILLING_API=true
  Kaggle telco CSV (60k rows, opt-in)     ← USE_KAGGLE=true
  Maven bank CSV   (10k rows, opt-in)     ← USE_BANK=true
        │
        ▼
MultiSourcePipeline  →  FeatureThresholds.from_dataframe(train_df)
        │                        │
        ▼                        ▼ (frozen at training, injected at inference)
Feature Engineering (63 cols, 13 engineered)
        │
        ├──▶  Prefect Flows  ──▶  MLflow Tracking + Registry
        │         data_pipeline_flow()         churn-data-pipeline exp
        │         full_pipeline_flow()         churn-model-comparison exp
        │         drift_monitoring_flow()      feature_thresholds.json artifact
        │
        ├──▶  FastAPI  (port 8000)
        │       POST /predict          → single customer, risk tier
        │       POST /predict/batch    → vectorised, up to 500 customers
        │       POST /explain          → SHAP top-N feature attributions
        │       GET  /health           → model status, auth mode, env
        │       Auth: X-API-Key header (dev bypass logs WARNING)
        │
        ├──▶  Dash Dashboard  (port 8050)
        │       5-tab multi-source view: BCG · CRM · Support · Billing · Cross-source
        │       Lazy-loaded at first request (not at import time)
        │
        └──▶  Evidently Drift Monitor  ──▶  Slack + Email Alerts
```

---

## Baseline model performance

Verified on 3,000 BCG customers (11% churn rate), 80/10/10 stratified split:

| Model | CV AUC | Test AUC | Test F1 | Brier | Lift@10% |
|---|---|---|---|---|---|
| Logistic Regression | 0.9564 | 0.9553 | 0.7333 | 0.0647 | 5.15× |
| **Random Forest** ◀ | 0.9555 | **0.9660** | **0.7416** | 0.0571 | **6.06×** |
| Gradient Boosting | 0.9583 | 0.9645 | 0.6269 | 0.0472 | 5.76× |

**Random Forest** — best AUC (0.966) and F1 (0.742). Recall = 1.0 on both LR and RF means every churner is caught; precision (0.59) reflects manageable false-positive rate for retention outreach.

**Business lift** — scoring the top 10% of customers by predicted churn probability captures **6× more actual churners** than random selection.

Regenerate: `make run-e2e`

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
make run-e2e              # full end-to-end verification with printed report
```

---

## Services

| Service | Port | URL |
|---|---|---|
| FastAPI Prediction API | 8000 | http://localhost:8000/docs |
| Dash Dashboard | 8050 | http://localhost:8050 |
| MLflow Tracking | 5000 | http://localhost:5000 |
| Prefect UI | 4200 | http://localhost:4200 |
| EspoCRM (opt-in) | 8080 | http://localhost:8080 |
| Zammad (opt-in) | 3000 | http://localhost:3000 |
| NovaBilling (opt-in) | 4000 | http://localhost:4000 |
| Locust Load Test UI | 8089 | http://localhost:8089 |

---

## Data sources

| Source | Type | Default | Activate |
|---|---|---|---|
| BCG `client_data.csv` | CSV | ✅ always on | — |
| BCG `price_data.csv` | CSV | ✅ always on | — |
| CRM | Synthetic generator | ✅ default | `USE_CRM_API=true` → EspoCRM |
| Support | Synthetic generator | ✅ default | `USE_SUPPORT_API=true` → Zammad |
| Billing | Synthetic generator | ✅ default | `USE_BILLING_API=true` → NovaBilling/Lago |
| Kaggle telco (60k rows) | CSV download | ❌ opt-in | `USE_KAGGLE=true` |
| Maven bank (10k rows) | CSV download | ❌ opt-in | `USE_BANK=true` |

> Synthetic sources use `FeatureThresholds`-compatible schemas. To replace a synthetic source with a real API: set the env flag + run `docker compose --profile real-sources up -d` and configure the service credentials in `.env`.

---

## API usage

```bash
# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"nps_score": -80, "num_late_payments_12m": 5, "contract_type": "month-to-month"}'

# Prediction + SHAP explanation (top 5 features)
curl -X POST "http://localhost:8000/explain?top_n=5" \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"nps_score": -80, "satisfaction_score": 1.2}'

# Batch (vectorised — single predict_proba call)
curl -X POST http://localhost:8000/predict/batch \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '[{"nps_score": -80}, {"nps_score": 70}]'

# Health check
curl http://localhost:8000/health
```

Full API reference → [`docs/api.md`](docs/api.md)

---

## Key design decisions

### Feature contract
`FeatureThresholds` are computed from the training set via `FeatureThresholds.from_dataframe(train_df)` and saved as `feature_thresholds.json` in the MLflow run artifacts. They are injected at inference via `predictor.set_thresholds(thresholds)` to ensure `high_consumption` and `low_margin` flags are stable across batch sizes and time.

### Lazy dashboard loading
`app.py` uses a `create_server()` factory and `_LazyServer` WSGI proxy. The pipeline and feature engineering run on first HTTP request, not at module import or gunicorn worker fork.

### Failure philosophy
- **MLflow unreachable** → raises `RuntimeError` with `ERROR` log (set `MLFLOW_OFFLINE=true` to explicitly bypass)
- **Model missing at startup in `ENV=production`** → raises, server does not start
- **`API_KEYS` unset in `ENV=production`** → raises at startup
- **`API_KEYS` unset in dev** → logs `WARNING`, requests pass through

### Kaggle/bank data appending
`KaggleTelcoSource` maps `MonthlyCharges → imp_cons` and `TotalCharges → cons_12m` as billing/usage proxies — semantically approximate. When precision matters, train separate models per source and ensemble rather than mixing rows.

---

## Project structure

```
.
├── api/
│   ├── main.py          # FastAPI routes + lifespan (not on_event)
│   ├── auth.py          # API key auth + production guard
│   ├── predictor.py     # MLflow model loader + get_pipeline() accessor
│   ├── explain.py       # SHAP feature attributions
│   └── schemas.py       # Pydantic models + cross-field validators
├── src/
│   ├── config.py        # pydantic-settings (risk tiers, CORS, env mode)
│   ├── logging_config.py
│   ├── data/
│   │   ├── sources/     # BaseDataSource + BCG/CRM/Support/Billing/Kaggle/Bank + API sources
│   │   ├── pipeline.py  # MultiSourcePipeline (source toggle flags)
│   │   └── preprocessing.py  # clean · aggregate (mean+std+last) · merge · null_report
│   ├── features/
│   │   └── engineering.py  # FeatureThresholds + 13 engineered features
│   ├── models/
│   │   └── churn_model.py  # GBM + frozen thresholds + MLflow artifacts
│   ├── tracking/
│   │   └── mlflow_tracker.py  # TrackingResult + explicit failures + MLFLOW_OFFLINE
│   ├── pipeline/
│   │   ├── flows.py     # Prefect flows (_run_data_steps shared, no duplication)
│   │   └── tasks.py     # Prefect tasks
│   ├── monitoring/
│   │   ├── drift.py     # Evidently AI
│   │   ├── monitor_flow.py  # drift flow + AlertManager
│   │   └── alerts.py    # Slack + email
│   ├── telemetry/
│   │   └── tracing.py   # OpenTelemetry (activate via OTEL_EXPORTER_OTLP_ENDPOINT)
│   └── visualizations/
│       ├── eda.py       # Plotly figure factory (BCG + CRM + Support + Billing + cross-source)
│       └── dashboard.py # Dash 5-tab layout + callbacks
├── tests/
│   ├── conftest.py
│   ├── test_sources.py      # 13 tests
│   ├── test_preprocessing.py # 12 tests
│   ├── test_engineering.py  # 15 tests
│   ├── test_pipeline.py     # 4 tests
│   ├── test_api.py          # 10 tests
│   ├── test_auth.py         # 7 tests
│   ├── test_explain.py      # 6 tests
│   ├── test_real_sources.py # 19 tests
│   ├── test_e2e.py          # 11 real E2E tests (no mocked predictor)
│   └── benchmarks/
│       └── test_performance.py  # 5 pytest-benchmark tests
├── scripts/
│   └── run_pipeline_e2e.py  # full end-to-end: ingest→train→evaluate→infer
├── k8s/                     # Kubernetes manifests + HPA
├── docs/
│   ├── api.md           # endpoint reference
│   ├── architecture.md  # system diagram + source modes
│   └── development.md   # setup, workflow, adding sources
├── reports/
│   ├── baseline_metrics.json
│   ├── roc_comparison.png
│   ├── pr_comparison.png
│   ├── calibration_comparison.png
│   └── confusion_*.png
├── app.py               # Dash entry point (lazy-loaded via _LazyServer)
├── locustfile.py        # load test + demo data generator
├── Makefile             # 19 dev commands
├── Dockerfile           # multi-stage, non-root user
├── docker-compose.yml   # core + --profile real-sources (EspoCRM/Zammad/NovaBilling)
├── prefect.yaml         # 3 scheduled deployments
├── dvc.yaml             # 3-stage data versioning pipeline
├── pyproject.toml       # black + ruff + pytest + coverage config
└── .pre-commit-config.yaml
```

---

## Testing

```bash
make test           # 97 unit + integration tests
make test-cov       # tests + HTML coverage report
make benchmark      # 5 pytest-benchmark performance tests (main only in CI)
```

**97 tests across 10 modules:**

| Module | Tests | What's covered |
|---|---|---|
| `test_sources.py` | 13 | shape, range constraints, reproducibility, metadata |
| `test_preprocessing.py` | 12 | churn validation, has_gas NaN, clip, merge, null threshold |
| `test_engineering.py` | 15 | all feature functions, FeatureThresholds, zero-max guard |
| `test_pipeline.py` | 4 | integration (mocked BCG sources) |
| `test_api.py` | 10 | endpoint wiring, schema validation, error codes |
| `test_auth.py` | 7 | dev bypass, key enforcement, production guard |
| `test_explain.py` | 6 | SHAP schema, top_n param, probability range |
| `test_real_sources.py` | 19 | Kaggle/bank CSV, API source contracts, pipeline flags |
| `test_e2e.py` | 11 | **real trained model** — directional sanity, batch parity, risk tier consistency |
| `benchmarks/` | 5 | feature engineering throughput, source extraction latency |

CI excludes benchmarks from the standard matrix (they run separately on `main` only and upload results as artifacts).

---

## Make commands

```bash
make help             # show all commands
make install          # pip install -r requirements.txt
make install-dev      # install + pre-commit hooks
make lint             # ruff + black --check
make format           # black + ruff --fix
make test             # 97 tests
make test-cov         # tests + HTML coverage at reports/coverage/
make benchmark        # pytest-benchmark performance suite
make run-api          # uvicorn api.main:app --reload
make run-dashboard    # python app.py
make run-pipeline     # python -m src.pipeline.flows
make run-drift        # python -m src.monitoring.monitor_flow
make run-e2e          # python scripts/run_pipeline_e2e.py
make docker-up        # docker compose up -d (core services)
make generate-demo    # 2000-row demo prediction CSV
make clean            # remove caches + artefacts
```

---

## Load testing

```bash
# Web UI at http://localhost:8089
locust -f locustfile.py --host http://localhost:8000

# Headless — 50 users, ramp 5/s, 60s
locust -f locustfile.py --host http://localhost:8000 \
  --headless -u 50 -r 5 -t 60s --csv=reports/locust_results

# Demo CSV (no server needed)
python locustfile.py --generate --rows 5000 --high-risk-pct 0.25
```

---

## Kubernetes

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secrets.yaml   # fill in values first
kubectl apply -f k8s/mlflow-deployment.yaml
kubectl apply -f k8s/api-deployment.yaml      # includes HPA 2→8 replicas
kubectl apply -f k8s/dashboard-deployment.yaml
kubectl apply -f k8s/ingress.yaml
```

Full guide → [`k8s/README.md`](k8s/README.md)

---

## Documentation

| Doc | Contents |
|---|---|
| [`docs/api.md`](docs/api.md) | Endpoints, schemas, error codes, auth setup |
| [`docs/architecture.md`](docs/architecture.md) | System diagram, data flow, source mode table |
| [`docs/development.md`](docs/development.md) | Setup, workflow, adding sources, env vars, dataset downloads |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Contribution guide, commit conventions |
| [`CHANGELOG.md`](CHANGELOG.md) | Version history |

---

## Tech stack

| Layer | Technology |
|---|---|
| Data pipeline | pandas, pyarrow, DVC |
| Feature engineering | numpy, pandas · FeatureThresholds contract |
| ML model | scikit-learn (GradientBoostingClassifier, RandomForest, LogisticRegression) |
| Experiment tracking | MLflow (explicit failures, MLFLOW_OFFLINE opt-in) |
| Orchestration | Prefect 3 |
| API | FastAPI + Pydantic v2 + Uvicorn |
| Explainability | SHAP (TreeExplainer) |
| Drift monitoring | Evidently AI |
| Alerting | Slack webhooks + SMTP |
| Tracing | OpenTelemetry (activate via env var) |
| Dashboard | Plotly Dash (lazy-loaded) |
| EDA | Sweetviz |
| Logging | structlog (JSON / console) |
| Load testing | Locust |
| Testing | pytest · pytest-cov · pytest-benchmark |
| Linting | ruff · black · mypy |
| CI/CD | GitHub Actions (lint → test matrix → benchmark → Docker build → GHCR push) |
| Containerisation | Docker multi-stage + docker-compose |
| Real data services | EspoCRM · Zammad · NovaBilling/Lago (--profile real-sources) |
| Kubernetes | Deployments · HPA · Ingress · cert-manager |

---

## License

MIT — see [LICENSE](LICENSE)
