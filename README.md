# Customer Churn ML System

[![CI](https://github.com/CollinsNyatundo/customer-churn-ml-system/actions/workflows/ci.yml/badge.svg)](https://github.com/CollinsNyatundo/customer-churn-ml-system/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An end-to-end reference project for customer-churn modeling: multi-source ingestion, reproducible feature engineering, model comparison, experiment tracking, a prediction API, monitoring utilities, and a dashboard.

The repository is production-shaped, not a turnkey production service. Synthetic CRM, support, and billing generators are enabled by default; live connectors, durable infrastructure, credentials, and operational ownership are deployment work.

## Highlights

- Scikit-learn pipelines for logistic regression, random forest, gradient boosting, XGBoost, LightGBM, CatBoost, and MLP
- Train/validation/test separation: model and threshold selection use validation data; the chosen model is evaluated once on the holdout test split
- Optional Optuna optimization and out-of-fold stacking
- Frozen feature thresholds and schema hashing to reduce training/serving skew
- MLflow tracking, Prefect flows, Feast definitions, FastAPI prediction endpoints, SHAP explanations, and drift/performance utilities
- CI on Python 3.11 and 3.12, plus Docker and benchmark jobs

## Scope and limitations

| Area | Current state |
|---|---|
| Data | BCG inputs plus synthetic generators by default; external APIs are opt-in |
| Feature store | Feast definitions with SQLite for local development |
| Tracking | MLflow, with an offline mode for tests/local workflows |
| Serving | FastAPI API-key boundary and Dash demonstration dashboard |
| Deployment | Example Docker Compose and Kubernetes manifests |
| Persistence | Requires deployment-specific databases, registries, and secret management |

The checked-in `reports/baseline_metrics.json` is a historical development artifact from before the holdout-selection correction. Do not present it as an independently validated production result. Regenerate metrics with the current pipeline and record the data version, commit, environment, and random seed before quoting them.

## Architecture

```text
Source adapters -> validation/merge -> feature engineering -> Feast sink
                                              |
                                              v
                              model sweep -> validation selection
                                              |
                                              v
                                  one holdout evaluation -> MLflow
                                              |
                           FastAPI + dashboard + monitoring utilities
```

## Local setup

```bash
git clone https://github.com/CollinsNyatundo/customer-churn-ml-system.git
cd customer-churn-ml-system
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Run the isolated tests:

```bash
pytest tests/ --ignore=tests/benchmarks
```

Run the local pipeline and services:

```bash
make run-e2e
docker compose up --build
```

Review `.env.example` before enabling external data sources. Never commit source datasets, service credentials, or generated model artifacts.

## Repository map

```text
src/data/          ingestion, validation, and feature-store sink
src/features/      deterministic feature engineering and thresholds
src/models/        algorithms, optimization, stacking, selection
src/monitoring/    drift, performance, and alert utilities
src/pipeline/      Prefect tasks and flows
api/               prediction and explanation API
feature_store/     Feast entities, views, and services
tests/             unit, integration, and benchmark tests
```

## License

[MIT](LICENSE)
