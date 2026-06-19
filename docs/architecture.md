# System Architecture

## Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Data Sources Layer                       │
│  BCG CSV ─── CRM (synthetic) ─── Support ─── Billing        │
│         └──────── MultiSourcePipeline ──────────┘           │
└────────────────────────┬────────────────────────────────────┘
                         │ extract → validate → merge
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  Feature Engineering                         │
│  Tenure · Consumption · Price sensitivity · Margin          │
│  Cross-source risk score · Engagement score                  │
└────────────────────────┬────────────────────────────────────┘
                         │ 63 total features
          ┌──────────────┴──────────────┐
          ▼                             ▼
┌──────────────────┐         ┌─────────────────────┐
│  Prefect Flows   │         │   MLflow Tracking    │
│  ─────────────── │         │  ─────────────────── │
│  data_pipeline   │────────▶│  Experiments         │
│  full_pipeline   │         │  Model registry      │
│  drift_monitor   │         │  Artifact store      │
└──────────────────┘         └─────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│                    Serving Layer                             │
│                                                             │
│  FastAPI (port 8000)          Dash Dashboard (port 8050)    │
│  ─────────────────            ────────────────────────────  │
│  POST /predict                5-tab multi-source view       │
│  POST /predict/batch          BCG · CRM · Support           │
│  POST /explain  (SHAP)        Billing · Cross-source        │
│  GET  /health                                               │
│                                                             │
│  Auth: X-API-Key header       Sweetviz EDA Report           │
│  Tracing: OpenTelemetry       (reports/sweetviz_report.html)│
└─────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│                  Monitoring & Alerts                         │
│  Evidently AI drift detection  →  Slack + Email alerts      │
│  MLflow metric logging         →  Prefect run history       │
└─────────────────────────────────────────────────────────────┘
```

## Data flow

1. **Extract** — `MultiSourcePipeline` loads BCG CSVs and generates synthetic CRM/Support/Billing data (swappable with real connectors in production)
2. **Validate** — each `BaseDataSource` subclass runs schema checks before merge
3. **Merge** — left joins on `id`, producing one row per customer with 40+ raw columns
4. **Engineer** — `build_feature_set()` adds 13 derived features (risk score, growth rate, price spread, etc.)
5. **Track** — Prefect tasks log source metadata, row counts, and null rates to MLflow
6. **Train** — GradientBoostingClassifier with 3-fold CV; AUC, F1, confusion matrix logged as artifacts
7. **Register** — model promoted to MLflow `Staging` automatically; manual promotion to `Production`
8. **Serve** — FastAPI loads `Production` model at startup; falls back to `Staging`
9. **Monitor** — daily Evidently drift check against reference dataset; alerts fired if drift detected

## Module map

| Module | Responsibility |
|---|---|
| `src/config.py` | All settings via pydantic-settings + .env |
| `src/logging_config.py` | structlog JSON/console setup |
| `src/data/sources/` | BaseDataSource ABC + 4 source implementations |
| `src/data/pipeline.py` | MultiSourcePipeline orchestrator |
| `src/data/preprocessing.py` | Cleaning, aggregation, merge helpers |
| `src/features/engineering.py` | 13 engineered features across 5 domains |
| `src/models/churn_model.py` | sklearn Pipeline + MLflow tracking |
| `src/tracking/mlflow_tracker.py` | Context managers for MLflow runs |
| `src/pipeline/flows.py` | Prefect @flow definitions |
| `src/pipeline/tasks.py` | Prefect @task wrappers |
| `src/monitoring/drift.py` | Evidently drift detection |
| `src/monitoring/alerts.py` | Slack + email alert dispatch |
| `src/telemetry/tracing.py` | OpenTelemetry span helpers |
| `src/visualizations/eda.py` | Plotly figure factory (10 chart types) |
| `src/visualizations/dashboard.py` | Dash layout + callbacks |
| `api/main.py` | FastAPI routes |
| `api/auth.py` | API key authentication |
| `api/predictor.py` | MLflow model loader (singleton) |
| `api/explain.py` | SHAP feature attributions |
| `api/schemas.py` | Pydantic request/response models |
