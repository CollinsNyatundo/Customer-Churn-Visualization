# Changelog

All notable changes are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Added
- Real-world data sources: `KaggleTelcoSource`, `BankChurnSource`
- API-backed sources: `CRMApiDataSource` (EspoCRM), `SupportApiDataSource` (Zammad), `BillingApiDataSource` (NovaBilling/Lago)
- Docker services for EspoCRM, Zammad, NovaBilling
- Source toggle flags in `MultiSourcePipeline`

---

## [1.1.0] — 2026-06-19

### Added
- `api/auth.py` — API key authentication (X-API-Key header)
- `api/explain.py` — SHAP-based feature attribution
- `POST /explain` endpoint with top-N feature contributions
- `src/monitoring/alerts.py` — Slack + email alert dispatch
- `src/telemetry/tracing.py` — OpenTelemetry span helpers
- `k8s/` — full Kubernetes manifests (API HPA 2–8 replicas)
- `.github/workflows/cd.yml` — build + push to GHCR + Slack notify
- `Makefile` — 18 dev commands
- `.pre-commit-config.yaml` — black + ruff + mypy + secret detection
- `dvc.yaml` — 3-stage data versioning pipeline
- `docs/` — API reference, architecture, development guide
- `tests/test_auth.py`, `tests/test_explain.py`, `tests/benchmarks/`

### Fixed
- CI: pinned black/ruff versions, slimmed test install
- Missing `src/pipeline/flows.py` and `tasks.py` from initial push
- Black formatting across all 48 source files

---

## [1.0.0] — 2026-06-19

### Added
- `MultiSourcePipeline` merging BCG CSV + CRM + Support + Billing
- `BaseDataSource` ABC with extract → validate → metadata contract
- 13 engineered features: tenure, consumption, price sensitivity, cross-source risk
- GradientBoostingClassifier with MLflow experiment tracking + model registry
- Prefect flows: `data_pipeline_flow`, `full_pipeline_flow`, `drift_monitoring_flow`
- FastAPI: `POST /predict`, `POST /predict/batch`, `GET /health`
- Evidently AI drift detection
- Dash dashboard with 5-tab multi-source view
- Sweetviz EDA report generation
- Locust load test suite + demo data generator
- GitHub Actions CI: lint + test matrix (py3.11/3.12) + Docker build
- Multi-stage Dockerfile + docker-compose (MLflow + Prefect + Dashboard)
- 54 pytest tests
