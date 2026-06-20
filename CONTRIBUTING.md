# Contributing Guide

Thank you for your interest in contributing to the Customer Churn ML System.

## Setup

```bash
git clone https://github.com/CollinsNyatundo/Customer-Churn-Visualization.git
cd Customer-Churn-Visualization
make install-dev   # installs deps + pre-commit hooks
cp .env.example .env
```

## Workflow

1. **Create a branch** from `main`:
   ```bash
   git checkout -b feat/your-feature-name
   ```

2. **Make your changes** following the conventions below.

3. **Run checks** before committing:
   ```bash
   make format   # auto-fix style
   make lint     # verify
   make test     # all 67+ tests must pass
   ```

4. **Open a pull request** against `main`. CI runs automatically.

## Adding a new data source

All data sources live in `src/data/sources/` and extend `BaseDataSource`.

```python
# src/data/sources/my_source.py
from src.data.sources.base import BaseDataSource

class MySource(BaseDataSource):
    name = "my_source"

    def extract(self) -> pd.DataFrame:
        # pull from file, DB, or API
        ...

    def validate(self, df: pd.DataFrame) -> None:
        # raise ValueError on bad data
        ...
```

Then register it in `src/data/pipeline.py` inside `MultiSourcePipeline.run()`.

Add tests in `tests/test_sources.py` following the existing patterns.

## Code style

- **Formatter**: black (line length 120)
- **Linter**: ruff
- **Type hints**: encouraged; mypy runs in pre-commit
- Pre-commit hooks enforce everything automatically

## Commit messages

Follow conventional commits:
```
feat: add KaggleTelcoSource
fix: handle null NPS scores in CRMApiSource
docs: update architecture diagram
test: add benchmark for BillingApiSource
```

## Environment variables

Document any new env vars in both `.env.example` and `docs/development.md`.

## Questions?

Open a GitHub issue or discussion.
