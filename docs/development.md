# Development Guide

## Setup

```bash
git clone https://github.com/CollinsNyatundo/customer-churn-ml-system.git
cd customer-churn-ml-system
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

---

## Downloading real-world datasets

### Kaggle Real-World Churn (60k rows)

```bash
pip install kaggle
# Place your kaggle.json at ~/.kaggle/kaggle.json
kaggle datasets download lasaljaywardena/real-world-churn
mkdir -p data/raw/kaggle_telco
unzip real-world-churn.zip -d data/raw/kaggle_telco/
# Then set USE_KAGGLE=true in .env
```

### Maven Analytics Bank Churn (10k rows)

1. Visit https://mavenanalytics.io/data-playground/bank-customer-churn
2. Download the CSV
3. Save to `data/raw/bank/bank_customer_churn.csv`
4. Set `USE_BANK=true` in `.env`

## Starting real API services

```bash
# Start EspoCRM (CRM), Zammad (Support), NovaBilling (Billing)
docker compose --profile real-sources up -d

# Verify they are up
docker compose ps

# Activate in pipeline
USE_CRM_API=true USE_SUPPORT_API=true USE_BILLING_API=true make run-pipeline
```

After starting, configure each service:
- **EspoCRM** → http://localhost:8080 → Admin → API Users → create key → set `ESPOCRM_API_KEY`
- **Zammad** → http://localhost:3000 → Admin → Token Access → create token → set `ZAMMAD_API_TOKEN`
- **NovaBilling** → http://localhost:4000/api/reference → create key → set `NOVABILLING_API_KEY`
