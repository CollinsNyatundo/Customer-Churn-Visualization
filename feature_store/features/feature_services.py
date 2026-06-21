"""
feature_store/features/feature_services.py
-------------------------------------------
FeatureService definitions — named bundles of features for specific use cases.

A FeatureService pins exactly which features go into a model, creating a
versioned contract between feature engineering and model training/serving.
"""

from feast import FeatureService

from feature_store.features.feature_views import (
    bcg_feature_view,
    billing_feature_view,
    crm_feature_view,
    engineered_feature_view,
    price_feature_view,
    support_feature_view,
)

# ── Full churn prediction service ─────────────────────────────────────────────
# Used for training and batch/online inference.
# Adding or removing features here constitutes a breaking change — bump version.

churn_prediction_v1 = FeatureService(
    name="churn_prediction_v1",
    features=[
        bcg_feature_view,
        price_feature_view,
        crm_feature_view,
        support_feature_view,
        billing_feature_view,
        engineered_feature_view,
    ],
    description=(
        "Version 1 — all feature groups for churn prediction. "
        "Trained with GradientBoostingClassifier, test AUC 0.9645."
    ),
    tags={"version": "1", "model": "GradientBoosting"},
)

# ── Lightweight service (BCG + engineered only) ────────────────────────────────
# For when CRM/Support/Billing APIs are unavailable — reduced feature set.

churn_prediction_bcg_only = FeatureService(
    name="churn_prediction_bcg_only",
    features=[
        bcg_feature_view,
        price_feature_view,
        engineered_feature_view,
    ],
    description="BCG-only fallback service (no CRM/Support/Billing features).",
    tags={"version": "1", "mode": "fallback"},
)
