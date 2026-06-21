"""
src/data/feast_store.py
------------------------
Thin wrapper around the Feast FeatureStore.

Provides:
  get_training_data()   — offline historical features for model training
  get_online_features() — low-latency online features for serving
  materialize()         — push latest offline features to online store

All methods fail loudly if the store is misconfigured.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

_STORE_DIR = Path(__file__).resolve().parents[2] / "feature_store"
_CHURN_SERVICE = "churn_prediction_v1"


@lru_cache(maxsize=1)
def _get_store():
    """Singleton Feast FeatureStore — cached after first call."""
    try:
        from feast import FeatureStore

        store = FeatureStore(repo_path=str(_STORE_DIR))
        log.info("Feast FeatureStore initialised from %s", _STORE_DIR)
        return store
    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialise Feast FeatureStore at {_STORE_DIR}. "
            "Run 'feast apply' in the feature_store/ directory first. "
            f"Original error: {exc}"
        ) from exc


def get_training_data(
    entity_df: pd.DataFrame,
    feature_service: str = _CHURN_SERVICE,
) -> pd.DataFrame:
    """
    Retrieve point-in-time correct historical features for model training.

    Parameters
    ----------
    entity_df       : DataFrame with columns [customer_id, event_timestamp, label]
    feature_service : name of the Feast FeatureService to retrieve

    Returns
    -------
    DataFrame with all features joined, ready for sklearn training
    """
    store = _get_store()
    fs = store.get_feature_service(feature_service)

    log.info(
        "Retrieving historical features for %d customers via '%s' ...",
        len(entity_df),
        feature_service,
    )
    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=fs,
    ).to_df()

    log.info("Training data retrieved: %s", training_df.shape)
    return training_df


def get_online_features(
    customer_ids: list[str],
    feature_service: str = _CHURN_SERVICE,
) -> pd.DataFrame:
    """
    Retrieve current features from the online store for real-time inference.

    Parameters
    ----------
    customer_ids    : list of customer IDs to retrieve features for
    feature_service : Feast FeatureService name

    Returns
    -------
    DataFrame with one row per customer_id, all online features filled
    """
    store = _get_store()
    fs = store.get_feature_service(feature_service)

    entity_rows = [{"customer_id": cid} for cid in customer_ids]
    response = store.get_online_features(
        features=fs,
        entity_rows=entity_rows,
    )
    df = pd.DataFrame(response.to_dict())
    log.info("Online features retrieved for %d customers", len(df))
    return df


def materialize(
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
) -> None:
    """
    Push features from the offline store to the online store.

    Call this after running the data pipeline to make fresh features
    available for low-latency inference.

    Parameters
    ----------
    start_date : materialise from this date (defaults to 30 days ago)
    end_date   : materialise up to this date (defaults to now)
    """
    from datetime import timedelta, timezone

    store = _get_store()
    end = end_date or pd.Timestamp.now(tz=timezone.utc)
    start = start_date or (end - timedelta(days=30))

    log.info("Materialising features from %s to %s ...", start, end)
    store.materialize(start_date=start, end_date=end)
    log.info("Materialisation complete.")


def apply() -> None:
    """Register all feature definitions with the Feast registry."""
    store = _get_store()
    store.apply([])  # Reads definitions from feature_store/features/
    log.info("Feast feature definitions applied to registry.")
