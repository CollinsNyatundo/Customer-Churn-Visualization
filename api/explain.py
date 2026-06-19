"""
api/explain.py
---------------
SHAP-based feature explainability for individual predictions.
Returns the top N features driving a churn prediction, with direction
and magnitude — suitable for CRM tooltips, dashboards, and audit logs.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.features.engineering import build_feature_set

logger = logging.getLogger(__name__)

# Features the model was trained on (must match churn_model.py)
NUMERIC_FEATURES = [
    "cons_12m",
    "cons_gas_12m",
    "cons_last_month",
    "imp_cons",
    "net_margin",
    "margin_gross_pow_ele",
    "num_years_antig",
    "pow_max",
    "nb_prod_act",
    "forecast_discount_energy",
    "nps_score",
    "satisfaction_score",
    "num_contacts_6m",
    "last_contact_days_ago",
    "num_tickets_6m",
    "avg_resolution_hours",
    "escalations_6m",
    "open_tickets",
    "num_late_payments_12m",
    "avg_days_late",
    "total_outstanding",
    "discount_pct",
    "contract_duration_days",
    "months_to_renewal",
    "cons_growth_rate",
    "price_spread_var",
    "margin_efficiency",
]
CATEGORICAL_FEATURES = [
    "channel_sales",
    "activity_new",
    "origin_up",
    "contract_type",
    "payment_method",
    "top_ticket_category",
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def explain_prediction(pipeline, features: dict, top_n: int = 10) -> list[dict]:
    """
    Compute SHAP values for a single prediction and return the top N
    most influential features.

    Parameters
    ----------
    pipeline : fitted sklearn Pipeline (preprocessor + clf)
    features : raw feature dict (same schema as CustomerFeatures)
    top_n    : number of top features to return

    Returns
    -------
    List of dicts sorted by |shap_value| descending:
        [{"feature": str, "value": any, "shap_value": float,
          "direction": "increases_churn" | "decreases_churn"}, ...]
    """
    try:
        import shap
    except ImportError:
        logger.warning("shap not installed — returning empty explanation")
        return []

    df_raw = pd.DataFrame([features])
    df_feat = build_feature_set(df_raw)

    available_num = [c for c in NUMERIC_FEATURES if c in df_feat.columns]
    available_cat = [c for c in CATEGORICAL_FEATURES if c in df_feat.columns]
    X = df_feat[available_num + available_cat]

    preprocessor = pipeline.named_steps["preprocessor"]
    clf = pipeline.named_steps["clf"]
    X_transformed = preprocessor.transform(X)

    # Get transformed feature names
    try:
        cat_names = (
            preprocessor.named_transformers_["cat"].named_steps["ohe"].get_feature_names_out(available_cat).tolist()
        )
    except Exception:
        cat_names = []
    feature_names = available_num + cat_names

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_transformed)

    # For binary classifiers shap_values may be a list [neg_class, pos_class]
    if isinstance(shap_values, list):
        sv = shap_values[1][0]
    else:
        sv = shap_values[0]

    n = min(len(feature_names), len(sv))
    pairs = sorted(
        zip(feature_names[:n], sv[:n]),
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:top_n]

    results = []
    for fname, sval in pairs:
        # Map transformed name back to original for readability
        original = fname.split("_")[0] if "_" in fname else fname
        raw_val = features.get(original, features.get(fname, None))
        results.append(
            {
                "feature": fname,
                "raw_value": raw_val,
                "shap_value": round(float(sval), 6),
                "direction": "increases_churn" if sval > 0 else "decreases_churn",
            }
        )

    return results
