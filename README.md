# Customer Churn Visualization — Production ML System

> End-to-end churn prediction platform: multi-source data pipeline → Feast feature store → MLflow experiment tracking → Prefect orchestration → 7-algorithm model sweep with stacking ensemble → FastAPI prediction API with SHAP explainability → Evidently AI drift monitoring → Dash dashboard → Kubernetes deployment.

[![CI](https://github.com/CollinsNyatundo/Customer-Churn-Visualization/actions/workflows/ci.yml/badge.svg)](https://github.com/CollinsNyatundo/Customer-Churn-Visualization/actions/workflows/ci.yml)
[![CD](https://github.com/CollinsNyatundo/Customer-Churn-Visualization/actions/workflows/cd.yml/badge.svg)](https://github.com/CollinsNyatundo/Customer-Churn-Visualization/actions/workflows/cd.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org)
[![Tests](https://img.shields.io/badge/tests-129%20passing-brightgreen)](#testing)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## What this is

A **modular ML scaffold** for customer churn prediction, refined through three
rounds of code review — an initial audit (26 issues), a deeper forensic pass (3
critical bugs the first round missed), and a line-by-line source review (this
round, which found code that had never actually been exercised — broken API
usage, missing methods, dead code paths). Each round found real things the
previous one missed; the honest expectation is that further review would too.

**Honest scope:**
- CRM / Support / Billing default to **synthetic generators** — real API connectors (EspoCRM, Zammad, NovaBilling) are opt-in via env flags
- Feast feature store uses SQLite online store in dev; swap to Redis for production
- The system is production-*shaped* — full deployment would require connecting live data sources
- Test coverage is genuinely 67% (`src`+`api`), not the higher figure an earlier, mis-scoped coverage config reported — see [Testing](#testing) for exactly what is and isn't covered

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

## Critical bug fixes (forensic audit round 2)

A second, deeper audit pass found 3 P0 bugs that the first fix round missed, plus a
training-serving skew bug discovered only through genuine end-to-end verification
(not just unit tests). All are fixed and verified with real trained models.

| Bug | Root cause | Fix | Verified |
|---|---|---|---|
| **MLP broken (CV AUC 0.531)** | Architecture `(76,38,64)` violated funnel pattern; LR=0.001 too low to converge in 200 iters | Fixed to `(128,64,32)` funnel, LR=0.01, max_iter=500 | AUC 0.531 → 0.9309 |
| **FeatureThresholds defaulted to zero at inference** | `api/predictor.py` never loaded `feature_thresholds.json` from the MLflow artifact — every threshold-based feature (`high_consumption`, `low_margin`, etc.) was silently wrong for every prediction | `ChurnPredictor.load()` downloads and reconstructs `FeatureThresholds` from the training run's artifact; raises `RuntimeError` if missing | Confirmed non-zero thresholds loaded (e.g. `high_consumption_threshold=16633.49`) |
| **Feature contract hard-coded, drift-prone** | `feature_contract.py` had 50+ manually-typed feature names that had to be kept in sync with `engineering.py` by hand | Auto-derived from `engineering.get_feature_names()` — impossible to drift | Verified 67 features match exactly |
| **Cross-source risk score degenerate on single-row inference** | `_safe_normalize()` divides by the *current batch's* max — for a single API request, that's always the value itself, producing meaningless 0-or-1 scores | Normalisation maxima (`num_tickets_6m_max`, `num_late_payments_12m_max`) frozen in `FeatureThresholds` from training, injected at inference | Confirmed real maxima (19.0, 7.0) load correctly, not defaulting to 1.0 |
| **Threshold artifact only saved 2 of 6 fields** | `mlflow.log_dict()` call hard-coded only `high_consumption_threshold` and `low_margin_threshold` | Uses `dataclasses.asdict(thresholds)` so any future field is automatically included | Full 6-field round-trip verified |
| **Tenure features silently NaN on API path** | `contract_duration_days`/`is_long_term` (the model's strongest predictors) require `date_activ`/`date_end`, which aren't in the `CustomerFeatures` API schema — every single-customer prediction was missing its top features | Proxy fallback derives `contract_duration_days` from `num_years_antig` (which *is* in the schema) when dates are absent | Directional sanity restored on isolated tenure signal |
| **MLflow 3.x registry incompatibility** | `log_model(name=...)` creates a "Logged Model" entity that doesn't auto-register for `models:/name/stage` loading | Switched to `artifact_path=` + `registered_model_name=` for direct registry compatibility | Model loads correctly via registry path |

### A meaningful lesson about synthetic test data

While chasing what looked like an inverted-prediction bug, we discovered the fixture's
CRM/Support/Billing synthetic sources generate features **independently of the churn
label** — there is no real causal link between NPS score, ticket count, or contract
type and whether a customer churns (only `tenure < 2 years` drives the label). This
means a trained model can and will pick up spurious, run-dependent correlations in
these fields, and a directional sanity test that assumes "bad NPS → higher churn"
without also holding other confounding synthetic fields constant will be unreliable.
`tests/test_e2e.py`'s fixture and assertions were rewritten to tie `churn` causally to
tenure (matching the production generator) and to isolate the tested variable from
confounds — this is what makes an E2E "sanity check" actually meaningful, and it's a
good reminder that synthetic data design has to match its own test's assumptions.

---

## Code review round 3 — bugs found through direct source inspection

A third pass, this time a line-by-line code review rather than an issue-list audit,
found a different class of problem: code that was never actually exercised, so nobody
had verified it worked. Several of these were completely non-functional despite
looking correct on the page.

| Bug | Root cause | Fix | Verified |
|---|---|---|---|
| **`/explain` and `/explain/batch` totally broken** | `explain.py` hard-coded the pipeline step name to `"preprocessor"`, but every real pipeline in `algorithms.py` uses `"pre"` — every call raised `KeyError`. `test_explain.py` never caught this because it mocks the entire pipeline with `MagicMock()`, which silently absorbs any attribute access without raising | Rewrote to resolve `"pre"` (with fallback), extract expected columns directly from the fitted `ColumnTransformer`, accept `thresholds=` (was building features with all-zero defaults — the same P0.2 skew bug, unfixed in this file), and handle `StackingEnsemble` champions | Real trained model + real `/explain` call succeeds end-to-end; 2 new tests added in `test_e2e.py` |
| **Drift detection subsystem non-functional** | `src/monitoring/drift.py` used Evidently's pre-0.5 API (`ColumnMapping`, `evidently.report.Report`); nothing in the codebase ever wrote `data/processed/reference.parquet`, so `drift_monitoring_flow()` always raised `FileNotFoundError` on first run regardless of API version | Rewrote for the native Evidently 0.7.x API (`Report`, `Dataset`, `DataDefinition`, `presets.DataDriftPreset`); added `save_reference_snapshot()` task, wired to run at the end of every successful training run | Full `drift_monitoring_flow()` run verified end-to-end for the first time; 6 new tests in `test_drift.py` |
| **`AlertManager` had no generic `.send()` method** | `src/monitoring/performance.py`'s degradation-alert path calls `AlertManager().send(title=..., body=...)`, but `AlertManager` only exposed typed methods (`send_drift_alert`, `send_pipeline_failure`, `send_model_registered`) — the first real performance-degradation event would have crashed with `AttributeError` | Added a generic `send()` method | Locked in with a dedicated regression test in `test_performance.py` |
| **Pandera schemas rejected any dataframe missing an optional column** | Every `Column()` in `quality.py` defaulted to `required=True`; `validate_source()` had zero test coverage before this review, so this was never caught | Marked all non-key columns `required=False` | 12 tests in `test_quality.py`, including the exact failure this caused |
| **Data quality contracts were defined but never called** | `quality.py`'s own docstring said it runs "automatically in `MultiSourcePipeline`," but nothing actually invoked `validate_source()` anywhere in the codebase | Wired into `MultiSourcePipeline.run()` after each source load, non-strict (warns, doesn't crash) | Integration test confirms a WARNING fires on injected bad data without failing the pipeline |
| **Test suite wasn't portable — CI has likely been failing silently** | `test_feature_store.py`'s own fixture called `BCGClientSource().load()`, which reads `data/raw/client_data.csv` — a gitignored path never generated by test infrastructure or the CI workflow | Fixture rewritten to use `conftest.py`'s synthetic, in-memory fixtures like the rest of the suite | Confirmed passing on a genuinely fresh clone with no local state |
| **Coverage config hid the buggiest modules** | `pyproject.toml` excluded `src/monitoring/*` and `src/pipeline/*` from coverage — exactly where the bugs above were found. A previously reported "92% coverage" never measured these paths | Exclusion removed; honest current figure is 67% (see Testing section) | — |
| Dead/unreachable code: `add_timing_header()` middleware had ~20 lines after its `return`; `pipeline.py` had `hasattr()` mutation branches that could never execute; `src/data/loader.py` was entirely unused; `full_pipeline_flow()` had two docstrings back-to-back | Leftover artifacts from earlier edits | Removed | AST-based scan across the repo confirms no remaining unreachable-code-after-`return` patterns |
| `StackingEnsemble` champion models would break `_expected_columns()` in both `predictor.py` and `explain.py` (no `.named_steps`) | Never triggered in practice because a single algorithm has consistently won the AUC comparison in testing, but this is not guaranteed | Both call sites now check for `_fitted_bases` and fall back to a representative base pipeline | Verified against a `StackingEnsemble` instance directly |
| OpenTelemetry packages never listed in `requirements.txt` | `tracing.py` degrades gracefully (returns `False`, logs a warning) rather than crashing, but the advertised feature could never actually activate | Added `opentelemetry-api`, `opentelemetry-sdk`, `opentelemetry-exporter-otlp-proto-grpc` | Exporter import confirmed working |
| `evidently` pinned to `0.4.30` | Confirmed the old code *did* work against this exact pin — not a live regression — but it's an old, unmaintained version with zero test coverage either way | Re-pinned to `0.7.21` alongside the native-API rewrite | — |

### A word on process

Nearly every bug in this pass shares a pattern: the affected code had no real test
exercising it, or the test that existed mocked away the exact thing that was broken.
`MagicMock()` doesn't raise `KeyError` on a wrong dictionary key, doesn't care if a
required method doesn't exist, and doesn't complain about a `required=True` Pandera
column receiving no data. Every fix in this round shipped with a test that calls the
real thing — a real `Report.run()`, a real fitted `ColumnTransformer`, a real
`AlertManager()` — specifically because that's the only way this class of bug gets
caught before it reaches an actual failure in production.

---

## Rate limiting & circuit breakers

- **API rate limits** (`slowapi`): 100/min on `/predict`, 20/min on `/predict/batch`,
  30/min on `/explain`, 10/min on `/explain/batch`
- **Circuit breaker** (`src/data/sources/_resilience.py`): all 3 API sources
  (EspoCRM, Zammad, NovaBilling) get exponential-backoff retry + circuit breaker;
  opens after 3 failures, resets after 60s

## Data quality contracts

Pandera schemas (`src/data/quality.py`) validate BCG/CRM/Support/Billing sources
for range violations, nulls, and type mismatches. Wired into
`MultiSourcePipeline.run()` in non-strict mode — violations log a `WARNING` and the
pipeline continues; pass `strict=True` at the call site for a deployment profile
where bad data should halt the pipeline outright.

## Model performance monitoring

`src/monitoring/performance.py` tracks rolling AUC/precision/recall/Brier over a
configurable window and fires Slack/email alerts on degradation. Wired into a Prefect
`performance_check_flow()` with optional auto-retraining trigger.

## Shadow deployment (champion/challenger)

Set `SHADOW_DEPLOYMENT_SPLIT=0.1` to route 10% of predictions to the Staging
("challenger") model alongside the Production ("champion") model, logging both for
comparison before promoting a challenger.

## Full DVC data lineage

`dvc.yaml` now tracks 5 stages: raw validation → merge → feature engineering →
Feast sink → train, each with explicit deps/outs for full reproducibility.

## Customer segmentation

`src.models.churn_model.train_segmented()` trains separate models per customer
segment (e.g. high-value vs low-value) when a business case calls for it.

---

## Testing

```bash
make test           # 167 tests
make test-cov       # tests + HTML coverage report
make benchmark      # 5 pytest-benchmark performance tests (main only in CI)
```

**167 tests across 14 modules, 67% real statement coverage** (`src`+`api`, honestly measured — see note below):

| Module | Tests | What's covered |
|---|---|---|
| `test_sources.py` | 13 | shape, ranges, reproducibility, metadata |
| `test_preprocessing.py` | 12 | churn validation, has_gas, clip, merge, null threshold |
| `test_engineering.py` | 22 | all 38 features, FeatureThresholds, zero-max guard, determinism |
| `test_pipeline.py` | 5 | integration (mocked BCG sources) + quality-validation firing |
| `test_api.py` | 10 | endpoint wiring, schema validation, error codes |
| `test_auth.py` | 7 | dev bypass, key enforcement, production guard |
| `test_explain.py` | 6 | response schema/wiring only (predictor mocked — see `test_e2e.py` for functional coverage) |
| `test_real_sources.py` | 19 | Kaggle/bank CSV, API contracts, pipeline flags |
| `test_e2e.py` | 13 | **real trained model** — directional sanity, batch parity, risk tier, `/explain` and `/explain/batch` against a real fitted pipeline |
| `test_feature_store.py` | 10 | Feast sink, online retrieval, historical features, consistency |
| `test_quality.py` | 12 | Pandera schemas for all 4 sources — range/type/uniqueness checks |
| `test_drift.py` | 6 | real `Report.run()` calls — genuine drift detected, HTML export, missing-column handling |
| `test_alerts.py` | 9 | Slack/email dispatch, `AlertManager.send()` generic method |
| `test_performance.py` | 13 | rolling metric computation, degradation alerting, insufficient-data handling |
| `benchmarks/` | 5 | feature engineering throughput, source extraction latency |

**A note on coverage honesty.** An earlier pass reported "92% coverage," but `pyproject.toml` silently excluded `src/monitoring/*` and `src/pipeline/*` from measurement — precisely the modules where the most severe bugs surfaced during a later code-review pass (see below). That exclusion has been removed. The current 67% is measured across all of `src`/`api` except `src/visualizations/*` and `src/telemetry/*`, which are disclosed, not-yet-covered gaps (Dash callbacks and OTel tracing need different test strategies than the rest of this suite). `api/predictor.py` and `src/tracking/mlflow_tracker.py` have real but partial coverage — the core prediction and threshold-loading paths are tested via `test_e2e.py` against a real model, but not every branch (e.g. shadow-deployment routing, some MLflow failure paths) has a dedicated unit test yet.

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
│   │   ├── sources/               # BaseDataSource + 8 source implementations + _resilience.py (circuit breaker)
│   │   ├── pipeline.py            # MultiSourcePipeline + toggle flags + quality validation wired in
│   │   ├── preprocessing.py       # validate_churn_column + has_gas fix + price _last
│   │   ├── quality.py             # Pandera schemas for BCG/CRM/Support/Billing (optional columns)
│   │   ├── feast_sink.py          # Write 6 feature-group Parquets
│   │   └── feast_store.py         # get_training_data / get_online_features / materialize
│   ├── features/
│   │   └── engineering.py        # 38 features, FeatureThresholds, _safe_normalize, _safe_div
│   ├── models/
│   │   ├── algorithms.py         # 7 algorithm builders (ALGORITHM_REGISTRY)
│   │   ├── churn_model.py        # Full sweep + Optuna + stacking + MLflow + reference snapshot
│   │   ├── ensemble.py           # StackingEnsemble (OOF → LR meta-learner, per-fold fallback)
│   │   ├── feature_contract.py   # auto-derived from engineering.get_feature_names() — 67 features
│   │   ├── optimization.py       # Optuna HPO per algorithm
│   │   └── threshold.py          # F1/F2/cost-sensitive threshold optimisation
│   ├── tracking/
│   │   └── mlflow_tracker.py     # TrackingResult, explicit failures, MLFLOW_OFFLINE
│   ├── pipeline/
│   │   ├── flows.py              # Prefect flows + performance_check_flow + reference snapshot
│   │   └── tasks.py              # Prefect tasks incl. feast_materialize, save_reference_snapshot
│   ├── monitoring/
│   │   ├── drift.py              # Evidently AI (native 0.7.x API)
│   │   ├── monitor_flow.py       # drift flow + AlertManager
│   │   ├── alerts.py             # Slack + email + generic .send()
│   │   └── performance.py        # rolling AUC/precision/recall/Brier + degradation alerts
│   ├── telemetry/tracing.py      # OpenTelemetry (activate via env; deps in requirements.txt)
│   └── visualizations/
│       ├── eda.py                # 10 Plotly charts (BCG + CRM + Support + Billing)
│       └── dashboard.py          # Dash 5-tab layout + callbacks
├── tests/                        # 167 tests, 14 modules
├── scripts/
│   └── run_pipeline_e2e.py       # Full E2E run with printed baseline report
├── k8s/                          # Kubernetes manifests + HPA (2→8 replicas)
├── docs/
│   ├── api.md                    # includes /explain/batch + rate limits
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
├── dvc.yaml                      # 5-stage data lineage (raw_validation → merge → features → feast → train)
├── pyproject.toml                # black + ruff + pytest + coverage config (honestly scoped)
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
