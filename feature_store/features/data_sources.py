"""
feature_store/features/data_sources.py
----------------------------------------
Feast FileSource definitions — one per feature group.

Each source points to a Parquet file written by the pipeline.
In production, swap FileSource for BigQuerySource, RedshiftSource, etc.

Files are written by src/data/feast_sink.py after pipeline.run().
"""

from pathlib import Path

from feast import FileSource

_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "feast"

bcg_source = FileSource(
    name="bcg_features_source",
    path=str(_DATA_DIR / "bcg_features.parquet"),
    timestamp_field="event_timestamp",
    description="BCG client core features (consumption, margin, tenure)",
)

price_source = FileSource(
    name="price_features_source",
    path=str(_DATA_DIR / "price_features.parquet"),
    timestamp_field="event_timestamp",
    description="Aggregated price features (mean, std, last period per customer)",
)

crm_source = FileSource(
    name="crm_features_source",
    path=str(_DATA_DIR / "crm_features.parquet"),
    timestamp_field="event_timestamp",
    description="CRM features: NPS, satisfaction, contacts, contract type",
)

support_source = FileSource(
    name="support_features_source",
    path=str(_DATA_DIR / "support_features.parquet"),
    timestamp_field="event_timestamp",
    description="Support ticket features: volume, resolution, escalations",
)

billing_source = FileSource(
    name="billing_features_source",
    path=str(_DATA_DIR / "billing_features.parquet"),
    timestamp_field="event_timestamp",
    description="Billing features: late payments, outstanding balance",
)

engineered_source = FileSource(
    name="engineered_features_source",
    path=str(_DATA_DIR / "engineered_features.parquet"),
    timestamp_field="event_timestamp",
    description="Engineered features: risk score, tenure, consumption growth",
)
