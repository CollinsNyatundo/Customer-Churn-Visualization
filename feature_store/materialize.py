"""
feature_store/materialize.py
------------------------------
Push latest offline features to the online store.

Run after every pipeline execution to keep online features fresh:

    python feature_store/materialize.py
    # or
    make feast-materialize

For production, this is triggered automatically by the Prefect
feast_materialize() task at the end of every pipeline run.
"""

import sys
from datetime import timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feast import FeatureStore

STORE_DIR = Path(__file__).resolve().parent

if __name__ == "__main__":
    store = FeatureStore(repo_path=str(STORE_DIR))
    end = pd.Timestamp.now(tz=timezone.utc)
    start = end - timedelta(days=30)

    print(f"Materialising features from {start.date()} to {end.date()} ...")
    store.materialize(start_date=start, end_date=end)
    print("✓ Online store updated.")
