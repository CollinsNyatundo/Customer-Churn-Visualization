# Development Guide

## Setup

```bash
git clone https://github.com/CollinsNyatundo/Customer-Churn-Visualization.git
cd Customer-Churn-Visualization
make install-dev          # installs deps + pre-commit hooks
cp .env.example .env      # configure your environment
```

## Daily workflow

```bash
make lint                 # check code style
make format               # auto-fix style issues
make test                 # run 70+ tests
make test-cov             # tests + HTML coverage report
make benchmark            # performance benchmarks
```

## Running services locally

```bash
make docker-up            # MLflow + Prefect (background)
make run-pipeline         # full data + train pipeline
make run-api              # FastAPI at http://localhost:8000
make run-dashboard        # Dash at http://localhost:8050
```

## Adding a new data source

1. Create `src/data/sources/my_source.py` extending `BaseDataSource`
2. Implement `extract()` and `validate()`
3. Add to `MultiSourcePipeline.run()` in `src/data/pipeline.py`
4. Add relevant features to `src/features/engineering.py`
5. Add test cases in `tests/test_sources.py`

## Environment variables

See `.env.example` for the full list. Key variables:

| Variable | Default | Description |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow server |
| `API_KEYS` | *(unset)* | Comma-separated API keys (dev bypass if unset) |
| `SLACK_WEBHOOK_URL` | *(unset)* | Slack alerts webhook |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | *(unset)* | OpenTelemetry collector |
| `SYNTHETIC_SEED` | `42` | Reproducibility seed for synthetic sources |

## Pre-commit hooks

Installed via `make install-dev`. Runs on every commit:
- `trailing-whitespace`, `end-of-file-fixer`, `check-yaml`
- `black` — auto-format
- `ruff` — lint + fix
- `mypy` — type checking
- `detect-private-key` — prevent secret leaks

## Load testing

```bash
# Web UI (http://localhost:8089)
locust -f locustfile.py --host http://localhost:8000

# Headless: 50 users, ramp 5/s, 60s
locust -f locustfile.py --host http://localhost:8000 \
  --headless -u 50 -r 5 -t 60s --csv=reports/locust_results

# Generate demo prediction CSV (no server needed)
make generate-demo
```
