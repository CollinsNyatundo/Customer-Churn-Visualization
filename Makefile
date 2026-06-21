# ──────────────────────────────────────────────────────────────────────────────
# Customer Churn — Developer Makefile
# Usage: make <target>
# ──────────────────────────────────────────────────────────────────────────────

.PHONY: help install install-dev lint format test test-cov benchmark \
        run-api run-dashboard run-pipeline run-drift docker-up docker-down \
        dvc-pull dvc-push generate-demo clean

PYTHON  := python
PYTEST  := pytest
SRC     := src api tests
PORT_API := 8000
PORT_DASH := 8050

# ── Help ──────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  Customer Churn ML System — available commands"
	@echo "  ─────────────────────────────────────────────"
	@echo "  make install        Install all dependencies"
	@echo "  make install-dev    Install + pre-commit hooks"
	@echo "  make lint           Run ruff linter"
	@echo "  make format         Auto-format with black + ruff --fix"
	@echo "  make test           Run full test suite"
	@echo "  make test-cov       Run tests + open HTML coverage report"
	@echo "  make benchmark      Run pytest-benchmark performance tests"
	@echo "  make run-api        Start FastAPI prediction server (port $(PORT_API))"
	@echo "  make run-dashboard  Start Dash dashboard (port $(PORT_DASH))"
	@echo "  make run-pipeline   Run Prefect full pipeline locally"
	@echo "  make run-drift      Run drift monitoring flow"
	@echo "  make docker-up      Start all services via docker compose"
	@echo "  make docker-down    Stop all docker services"
	@echo "  make dvc-pull       Pull latest data from DVC remote"
	@echo "  make dvc-push       Push data to DVC remote"
	@echo "  make generate-demo  Generate 2000 demo predictions CSV"
	@echo "  make clean          Remove caches, pyc files, coverage artefacts"
	@echo ""

# ── Setup ─────────────────────────────────────────────────────────────────────
install:
	pip install --upgrade pip
	pip install -r requirements.txt

install-dev: install
	pip install pre-commit
	pre-commit install
	@echo "Pre-commit hooks installed."

# ── Quality ───────────────────────────────────────────────────────────────────
lint:
	ruff check $(SRC)
	black --check $(SRC)

format:
	black $(SRC)
	ruff check $(SRC) --fix

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	$(PYTEST) tests/ -v --tb=short -x

test-cov:
	$(PYTEST) tests/ \
		--cov=src --cov=api \
		--cov-report=term-missing \
		--cov-report=html:reports/coverage
	@echo "Coverage report: reports/coverage/index.html"

benchmark:
	$(PYTEST) tests/benchmarks/ -v --benchmark-only \
		--benchmark-json=reports/benchmark.json \
		--benchmark-sort=mean

# ── Services ──────────────────────────────────────────────────────────────────
run-api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port $(PORT_API)

run-dashboard:
	$(PYTHON) app.py

run-pipeline:
	$(PYTHON) -m src.pipeline.flows

run-drift:
	$(PYTHON) -m src.monitoring.monitor_flow

# ── Docker ────────────────────────────────────────────────────────────────────
docker-up:
	docker compose up -d
	@echo "Services:"
	@echo "  MLflow    → http://localhost:5000"
	@echo "  Prefect   → http://localhost:4200"
	@echo "  Dashboard → http://localhost:8050"

docker-down:
	docker compose down

# ── Data ──────────────────────────────────────────────────────────────────────
dvc-pull:
	dvc pull

dvc-push:
	dvc push

generate-demo:
	$(PYTHON) locustfile.py --generate --rows 2000 --out reports/demo_predictions.csv
	@echo "Demo data: reports/demo_predictions.csv"

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	rm -rf reports/coverage reports/benchmark.json
	@echo "Cleaned."

run-e2e:
	$(PYTHON) scripts/run_pipeline_e2e.py

feast-apply:
	python feature_store/apply.py

feast-materialize:
	python feature_store/materialize.py

feast-refresh: run-pipeline feast-apply feast-materialize
	@echo "Feature store refreshed end-to-end."
