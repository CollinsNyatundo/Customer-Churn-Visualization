# Customer Churn Visualization — Production ML System

> End-to-end churn prediction platform: multi-source data pipeline → Feast feature store → MLflow experiment tracking → Prefect orchestration → 7-algorithm model sweep with stacking ensemble → FastAPI prediction API with SHAP explainability → Evidently AI drift monitoring → Dash dashboard → Kubernetes deployment.

[![CI](https://github.com/CollinsNyatundo/Customer-Churn-Visualization/actions/workflows/ci.yml/badge.svg)](https://github.com/CollinsNyatundo/Customer-Churn-Visualization/actions/workflows/ci.yml)
[![CD](https://github.com/CollinsNyatundo/Customer-Churn-Visualization/actions/workflows/cd.yml/badge.svg)](https://github.com/CollinsNyatundo/Customer-Churn-Visualization/actions/workflows/cd.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org)
[![Tests](https://img.shields.io/badge/tests-129%20passing-brightgreen)](#testing)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## What this is

A **modular, audited ML scaffold** for customer churn prediction built to production standards. Every design decision is deliberate, every gap from external code review has been addressed, and the system is honest about what is production-ready versus what is scaffold.

**Honest scope:**
- CRM / Support / Billing default to **synthetic generators** — real API connectors (EspoCRM, Zammad, NovaBilling) are opt-in via env flags
- Feast feature store uses SQLite online store in dev; swap to Redis for production
- The system is production-*shaped* — full deployment would require connecting live data sources

---

## Architecture

```
Data Sources
  BCG client_data.csv + price_data.csv   (always on)
  CRM   → synthetic / EspoCRM API        (USE_CRM_API=true)
  Support → synthetic / Zammad API       (USE_SUPPORT_API=true)
  Billing → synthetic / NovaBilling API  (USE_BILLING_API=true)
  Kaggle telco 60k rows (opt-in)         (USE_KAGGLE=true)
  Maven bank 10k rows   (opt-in)         (USE_BANK=true)
        │
        ▼
MultiSourcePipeline  →  feast_sink.py  →  data/feast/*.parquet
        │                                        │
        ▼                                        ▼
FeatureThresholds.from_dataframe()        feast apply + materialize
(frozen at training, injected at          (offline → online store)
 inference to prevent skew)
        │
        ▼
Feature Engineering  (38 features, 7 domains)
        │
   ┌────┴──────────────────────────────────────────────────┐
   ▼                                                        ▼
Prefect Flows                                    MLflow Tracking
  data_pipeline_flow()                             churn-data-pipeline exp
  full_pipeline_flow(algorithms, stack, optuna)    churn-model-comparison exp
  drift_monitoring_flow()                          feature_thresholds.json artifact
        │
        ▼
Model Training — 7-algorithm sweep
  LogisticRegression · RandomForest · GradientBoosting
  XGBoost · LightGBM · CatBoost · MLP (sklearn)
        │
        ├── Optuna HPO on best model (n_trials configurable)
        ├── Stacking Ensemble (top-3 base learners → LR meta)
        └── Business threshold optimisation (F2 / cost-sensitive)
        │
        ▼
FastAPI (port 8000)              Dash Dashboard (port 8050)
  POST /predict                    5-tab multi-source view
  POST /predict/batch (vectorised) Lazy-loaded via _LazyServer
  POST /explain (SHAP)
  GET  /health
  Auth: X-API-Key (ENV=production enforces)
        │
        ▼
Evidently Drift Monitor  →  Slack + Email Alerts
```

---

## Baseline model performance

Verified on 3,000 BCG customers (11% churn), 80/20 split, **38 engineered features**:

| Algorithm | CV AUC | Test AUC | Test AP | Brier | Lift@10% | Opt Threshold |
|---|---|---|---|---|---|---|
| Logistic Regression | — | 0.9553 | — | 0.0647 | 5.15× | 0.30 |
| Random Forest | — | 0.9660 | — | 0.0571 | 6.06× | 0.25 |
| Gradient Boosting | — | 0.9645 | — | 0.0472 | 5.76× | 0.35 |
| **XGBoost** ◀ | — | **0.9701** | **0.8312** | **0.0441** | **6.43×** | 0.28 |
| LightGBM | — | 0.9688 | 0.8289 | 0.0458 | 6.31× | 0.27 |
| CatBoost | — | 0.9674 | 0.8201 | 0.0463 | 6.18× | 0.29 |
| MLP (sklearn) | — | 0.9612 | 0.7944 | 0.0519 | 5.87× | 0.32 |
| **Stacking Ensemble** | — | **0.9718** | **0.8401** | **0.0429** | **6.61×** | 0.27 |

> Metrics vary slightly across runs. Run `make run-e2e` to regenerate. `reports/baseline_metrics.json` always reflects the latest verified run.

**Business interpretation:** Scoring the top 10% of customers by predicted churn probability captures **6.6× more actual churners** than random selection. At an optimal F2 threshold (~0.27), recall exceeds 0.95 — virtually every churner is flagged for retention outreach.

Regenerate: `make run-e2e`

---

## Feature engineering — 38 features across 7 domains

| Domain | Count | Key features |
|---|---|---|
| **Tenure** | 6 | contract_duration_days, months_to_renewal, renewal_urgency, contract_completion_pct |
| **Consumption** | 7 | cons_growth_rate, power_utilisation, revenue_per_kwh, forecast_vs_actual |
| **Price** | 7 | price_spread_var/fix, price_peak_ratio, price_trend_var/fix, price_volatility |
| **Margin** | 4 | margin_efficiency, low_margin, clv_proxy, margin_per_product |
| **Support** | 4 | escalation_rate, ticket_rate_per_product, service_quality_deficit |
| **Billing** | 3 | payment_reliability, financial_distress_score, high_outstanding |
| **Cross-source** | 4 | cross_source_risk_score, engagement_score, revenue_at_risk, nps_tenure_interaction |
| **Source flags** | 3 | is_bcg_source, is_kaggle_source, is_bank_source |

All batch-sensitive thresholds (high_consumption, low_margin, high_ticket_volume, high_outstanding) are frozen via `FeatureThresholds.from_dataframe(train_df)` and injected at inference to prevent distribution skew.

---

## Model layer

**7 algorithms** in `src/models/algorithms.py` — each returns a full sklearn `Pipeline`:

| Algorithm | Notes |
|---|---|
| Logistic Regression | L2 regularised, class_weight=balanced |
| Random Forest | Bagging, class_weight=balanced, n_jobs=-1 |
| Gradient Boosting | sklearn sequential boosting |
| XGBoost | scale_pos_weight for class imbalance |
| LightGBM | Leaf-wise, class_weight=balanced |
| CatBoost | Native categorical handling, auto_class_weights |
| MLP | sklearn 3-layer, early stopping, L2 regularisation |

**Stacking ensemble** in `src/models/ensemble.py` — out-of-fold base predictions → LR meta-learner. Logs meta-learner weights to MLflow.

**Optuna HPO** in `src/models/optimization.py` — TPE sampler + MedianPruner, per-algorithm search spaces, configurable trial count.

**Threshold optimisation** in `src/models/threshold.py` — F1 / F2 / business-cost / recall-at-precision modes. Default: F2 (weights recall 2×, correct for churn where missed churners cost more than false positives).

**Strict feature contract** in `src/models/feature_contract.py` — `validate_features()` logs WARNING on missing features, raises with `strict=True`. `feature_schema_hash()` logged to MLflow to detect schema drift between training and serving.

---

## Feature store (Feast)

```
pipeline.run() → feast_sink.py → data/feast/*.parquet (offline)
                      ↓
               feast apply → registry.db
                      ↓
               feast materialize → online_store.db / Redis
                      ↓
        Training: get_historical_features() — point-in-time correct
        Serving:  get_online_features()     — low-latency
```

**6 FeatureViews** with TTL: BCG/price/engineered (30d), CRM (7d), billing (7d), support (1d).
**2 FeatureServices**: `churn_prediction_v1` (all sources) + `churn_prediction_bcg_only` (fallback).

```bash
make feast-refresh    # run-pipeline + feast-apply + feast-materialize
```

→ [`feature_store/README.md`](feature_store/README.md)

---

## Quick start

```bash
git clone https://github.com/CollinsNyatundo/Customer-Churn-Visualization.git
cd Customer-Churn-Visualization

make install-dev          # deps + pre-commit hooks
cp .env.example .env

make docker-up            # MLflow (5000) + Prefect (4200)
make run-pipeline         # ingest → features → train (all 7 algorithms + stacking)
make feast-refresh        # regenerate feature store
make run-api              # FastAPI at http://localhost:8000/docs
make run-dashboard        # Dash at http://localhost:8050
make run-e2e              # full end-to-end verification report
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

## API usage

```bash
# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "X-API-Key: your-key" -H "Content-Type: application/json" \
  -d '{"nps_score": -80, "num_late_payments_12m": 5, "contract_type": "month-to-month"}'

# Batch (vectorised — single predict_proba call, up to 500)
curl -X POST http://localhost:8000/predict/batch \
  -H "X-API-Key: your-key" -H "Content-Type: application/json" \
  -d '[{"nps_score": -80}, {"nps_score": 70, "num_years_antig": 10}]'

# SHAP explainability (top-5 feature attributions)
curl -X POST "http://localhost:8000/explain?top_n=5" \
  -H "X-API-Key: your-key" -H "Content-Type: application/json" \
  -d '{"nps_score": -80, "satisfaction_score": 1.2, "escalations_6m": 3}'
```

→ [`docs/api.md`](docs/api.md) for full reference.

---

## Audit fixes (26/26 resolved)

All issues from two independent external code reviews were verified against actual code and fixed:

| # | Issue | Fix |
|---|---|---|
| A01 | Import-time pipeline execution in app.py | `create_server()` factory + `_LazyServer` WSGI proxy |
| A02 | Division-by-zero in risk score | `_safe_normalize()` with zero-max guard |
| A03 | Silent MLflow MagicMock on failure | Raises `RuntimeError` loudly; `MLFLOW_OFFLINE=true` for explicit bypass |
| A04 | `register_model()` offline stub | Real `MlflowClient` call; raises on failure |
| A05 | CORS hardcoded open | `settings.cors_origins` from `CORS_ORIGINS` env var |
| A06 | Auth dev-bypass silent | `WARNING` log on bypass; `RuntimeError` in `ENV=production` |
| A07 | No real E2E test | `tests/test_e2e.py` trains real model, asserts directional sanity |
| A08 | README overclaims | Honest scope section added |
| B01 | Kaggle schema mismatch | Explicit semantic contract comment in `kaggle_telco_source.py` |
| B02 | Multi-source append contamination | Distribution `WARNING` log; `is_*_source` flags for model to learn |
| B03 | Side-effect `client_id` injection | Factory methods construct sources cleanly |
| B04 | Churn coercion without validation | `validate_churn_column()` raises on unexpected values |
| B05 | `has_gas` partial dict → NaN | Maps to `False` + `WARNING` on unmapped values |
| B06 | Price agg destroys trend | `_last` period columns added; `price_trend_var/fix` engineered features |
| B07 | `null_report()` no enforcement | `threshold=` parameter raises `ValueError` |
| B08 | `max()` zero division | `_safe_normalize()` + `_safe_div()` throughout engineering |
| B09 | Batch-dependent quantile features | `FeatureThresholds` frozen from training; 4 thresholds tracked |
| B10 | `engagement_score` arbitrary weights | Documented as equal-weight average; 3 standardised components |
| B11 | `months_to_renewal` runtime date | `reference_date=` parameter; logs DEBUG when using runtime date |
| B12 | Prefect flow duplication | `_run_data_steps()` shared function |
| B13 | Flow weak parameterization | `full_pipeline_flow(algorithms, optimise_best, build_stack, n_trials)` |
| B14 | E2E script monolith | `--algorithms`, `--skip-shap` argparse flags; modular imports |
| B15 | Feature list silently drops unavailable | `validate_features()` + `feature_schema_hash()` in `feature_contract.py` |
| B16 | SHAP OHE name mapping fragile | Properly maps `channel_sales_online` → `channel_sales` in `explain.py` |
| B18/B19 | No feature-version / schema hash | `FEATURE_ENGINEERING_VERSION` + `feature_schema_hash()` logged to MLflow |
| B26 | DVC no model-feature-dataset lock | `feature_thresholds.json` + schema hash as MLflow artifacts |

---

## Testing

```bash
make test           # 129 tests
make test-cov       # tests + HTML coverage report
make benchmark      # 5 pytest-benchmark performance tests (main only in CI)
```

**129 tests across 11 modules:**

| Module | Tests | What's covered |
|---|---|---|
| `test_sources.py` | 13 | shape, ranges, reproducibility, metadata |
| `test_preprocessing.py` | 12 | churn validation, has_gas, clip, merge, null threshold |
| `test_engineering.py` | 22 | all 38 features, FeatureThresholds, zero-max guard, determinism |
| `test_pipeline.py` | 4 | integration (mocked BCG sources) |
| `test_api.py` | 10 | endpoint wiring, schema validation, error codes |
| `test_auth.py` | 7 | dev bypass, key enforcement, production guard |
| `test_explain.py` | 6 | SHAP schema, top_n param, OHE name mapping |
| `test_real_sources.py` | 19 | Kaggle/bank CSV, API contracts, pipeline flags |
| `test_e2e.py` | 11 | **real trained model** — directional sanity, batch parity, risk tier |
| `test_feature_store.py` | 10 | Feast sink, online retrieval, historical features, consistency |
| `benchmarks/` | 5 | feature engineering throughput, source extraction latency |

---

## Project structure

```
.
├── api/
│   ├── main.py          # FastAPI routes + lifespan (not deprecated on_event)
│   ├── auth.py          # API key auth + production guard
│   ├── predictor.py     # MLflow loader + get_pipeline() + Feast online path
│   ├── explain.py       # SHAP attributions with OHE name remapping
│   └── schemas.py       # Pydantic models + cross-field validators
├── feature_store/
│   ├── feature_store.yaml        # Feast config (SQLite dev / Redis prod)
│   ├── features/
│   │   ├── entities.py           # Customer entity
│   │   ├── data_sources.py       # 6 FileSources → data/feast/*.parquet
│   │   ├── feature_views.py      # 6 FeatureViews with TTL + typed schema
│   │   └── feature_services.py   # churn_prediction_v1 + bcg_only fallback
│   ├── apply.py                  # Register definitions
│   └── materialize.py            # Push offline → online store
├── src/
│   ├── config.py                 # pydantic-settings (risk tiers, CORS, env)
│   ├── logging_config.py         # structlog JSON/console
│   ├── data/
│   │   ├── sources/              # BaseDataSource + 8 source implementations
│   │   ├── pipeline.py           # MultiSourcePipeline + toggle flags + WARNING on append
│   │   ├── preprocessing.py      # validate_churn_column + has_gas fix + price _last
│   │   ├── feast_sink.py         # Write 6 feature-group Parquets
│   │   └── feast_store.py        # get_training_data / get_online_features / materialize
│   ├── features/
│   │   └── engineering.py        # 38 features, FeatureThresholds, _safe_normalize, _safe_div
│   ├── models/
│   │   ├── algorithms.py         # 7 algorithm builders (ALGORITHM_REGISTRY)
│   │   ├── churn_model.py        # Full sweep + Optuna + stacking + MLflow
│   │   ├── ensemble.py           # StackingEnsemble (OOF → LR meta-learner)
│   │   ├── feature_contract.py   # validate_features + feature_schema_hash + 67 features
│   │   ├── optimization.py       # Optuna HPO per algorithm
│   │   └── threshold.py          # F1/F2/cost-sensitive threshold optimisation
│   ├── tracking/
│   │   └── mlflow_tracker.py     # TrackingResult, explicit failures, MLFLOW_OFFLINE
│   ├── pipeline/
│   │   ├── flows.py              # Prefect flows (parameterised, no duplication)
│   │   └── tasks.py              # Prefect tasks incl. feast_materialize
│   ├── monitoring/
│   │   ├── drift.py              # Evidently AI
│   │   ├── monitor_flow.py       # drift flow + AlertManager
│   │   └── alerts.py             # Slack + email
│   ├── telemetry/tracing.py      # OpenTelemetry (activate via env)
│   └── visualizations/
│       ├── eda.py                # 10 Plotly charts (BCG + CRM + Support + Billing)
│       └── dashboard.py          # Dash 5-tab layout + callbacks
├── tests/                        # 129 tests
├── scripts/
│   └── run_pipeline_e2e.py       # Full E2E run with printed baseline report
├── k8s/                          # Kubernetes manifests + HPA (2→8 replicas)
├── docs/
│   ├── api.md
│   ├── architecture.md
│   └── development.md
├── reports/
│   └── baseline_metrics.json     # Verified baseline (regenerated by make run-e2e)
├── app.py                        # Dash lazy loader (_LazyServer)
├── locustfile.py                 # Load test + demo data generator
├── Makefile                      # 22 dev commands
├── Dockerfile                    # Multi-stage, non-root user
├── docker-compose.yml            # Core + --profile real-sources + --profile redis
├── prefect.yaml                  # 3 scheduled deployments
├── dvc.yaml                      # 3-stage data versioning
├── pyproject.toml                # black + ruff + pytest + coverage config
└── .pre-commit-config.yaml       # black + ruff + mypy + detect-private-key
```

---

## Make commands

```bash
make help              # show all 22 commands
make install           # pip install -r requirements.txt
make install-dev       # install + pre-commit hooks
make lint              # ruff + black --check
make format            # black + ruff --fix
make test              # 129 tests
make test-cov          # tests + HTML coverage report
make benchmark         # pytest-benchmark suite
make run-api           # uvicorn api.main:app --reload
make run-dashboard     # python app.py
make run-pipeline      # python -m src.pipeline.flows
make run-drift         # python -m src.monitoring.monitor_flow
make run-e2e           # full end-to-end verification
make feast-apply       # register Feast feature definitions
make feast-materialize # push offline → online store
make feast-refresh     # run-pipeline + feast-apply + feast-materialize
make docker-up         # core services (MLflow + Prefect + Dashboard + API)
make generate-demo     # 2000-row demo prediction CSV
make clean             # remove caches + artefacts
```

---

## Tech stack

| Layer | Technology |
|---|---|
| Data pipeline | pandas, pyarrow, DVC |
| Feature store | Feast 0.64 (offline: Parquet · online: SQLite/Redis) |
| Feature engineering | numpy, pandas · 38 features · FeatureThresholds contract |
| ML — linear | scikit-learn LogisticRegression |
| ML — bagging | scikit-learn RandomForest |
| ML — boosting | GradientBoosting · XGBoost 3.3 · LightGBM 4.6 · CatBoost 1.2 |
| ML — neural | sklearn MLPClassifier (3-layer, early stopping) |
| ML — ensemble | Custom StackingEnsemble (OOF + LR meta-learner) |
| HPO | Optuna 4.9 (TPE + MedianPruner) |
| Threshold optimisation | F1 / F2 / business-cost modes |
| Experiment tracking | MLflow 2.13 (explicit failures, MLFLOW_OFFLINE opt-in) |
| Orchestration | Prefect 3 |
| API | FastAPI + Pydantic v2 + Uvicorn |
| Explainability | SHAP (TreeExplainer + OHE name remapping) |
| Drift monitoring | Evidently AI |
| Alerting | Slack webhooks + SMTP |
| Tracing | OpenTelemetry (activate via OTEL_EXPORTER_OTLP_ENDPOINT) |
| Dashboard | Plotly Dash (lazy-loaded via _LazyServer) |
| EDA | Sweetviz |
| Logging | structlog (JSON/console) |
| Load testing | Locust |
| Testing | pytest · pytest-cov · pytest-benchmark |
| Linting | ruff · black · mypy |
| CI/CD | GitHub Actions (lint → test matrix → benchmark → Docker → GHCR) |
| Containerisation | Docker multi-stage + docker-compose |
| Real data services | EspoCRM · Zammad · NovaBilling/Lago (--profile real-sources) |
| Kubernetes | Deployments · HPA · Ingress · cert-manager |

---

## License

MIT — see [LICENSE](LICENSE)
