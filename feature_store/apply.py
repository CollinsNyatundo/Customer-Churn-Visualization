"""
feature_store/apply.py
-----------------------
Register all feature definitions with the Feast registry.
Run once after cloning, and again after changing feature_views.py.

    python feature_store/apply.py
    # or
    cd feature_store && feast apply
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feast import FeatureStore

from feature_store.features.entities import customer
from feature_store.features.feature_services import (
    churn_prediction_bcg_only,
    churn_prediction_v1,
)
from feature_store.features.feature_views import (
    bcg_feature_view,
    billing_feature_view,
    crm_feature_view,
    engineered_feature_view,
    price_feature_view,
    support_feature_view,
)

STORE_DIR = Path(__file__).resolve().parent

if __name__ == "__main__":
    store = FeatureStore(repo_path=str(STORE_DIR))
    store.apply(
        [
            customer,
            bcg_feature_view,
            price_feature_view,
            crm_feature_view,
            support_feature_view,
            billing_feature_view,
            engineered_feature_view,
            churn_prediction_v1,
            churn_prediction_bcg_only,
        ]
    )
    print("✓ Feast feature definitions applied to registry.")
    print(f"  Registry: {STORE_DIR / 'registry.db'}")
    print(f"  Online store: {STORE_DIR / 'online_store.db'}")
