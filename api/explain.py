"""
api/explain.py
---------------
SHAP-based feature explainability for individual predictions.
Returns the top N features driving a churn prediction, with direction
and magnitude — suitable for CRM tooltips, dashboards, and audit logs.

Fixes applied after code review
--------------------------------
1. Pipeline step name was hard-coded to "preprocessor", but every real
   pipeline built in src/models/algorithms.py uses the step name "pre".
   This meant every /explain call raised KeyError in production — fixed
   to resolve "pre" first, "preprocessor" as fallback.
2. NUMERIC_FEATURES/CATEGORICAL_FEATURES were a separate, incomplete,
   independently-drifted copy of the feature contract (~27 features vs
   the real 67, and missing BOOLEAN_AS_INT_FEATURES entirely). Now reads
   the exact expected columns directly from the fitted ColumnTransformer's
   .transformers_ attribute — the single source of truth for what a
   *specific* trained model actually expects, immune to drift between
   feature_contract.py and any given historical model version.
3. build_feature_set() was called with no thresholds, silently repeating
   the P0.2 training-serving-skew bug for every SHAP explanation. Now
   accepts a thresholds parameter that the caller (api/main.py) supplies
   from predictor.thresholds.
4. StackingEnsemble champions don't have .named_steps like a plain
   sklearn Pipeline — explanations now fall back to one of its fitted
   base learners, since all base learners share the same feature
   contract by construction.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.features.engineering import FeatureThresholds, build_feature_set

logger = logging.getLogger(__name__)


def _resolve_preprocessor_and_clf(pipeline):
    """
    Extract (preprocessor, classifier) from either a plain sklearn Pipeline
    (step name "pre") or a StackingEnsemble (falls back to one of its
    fitted base pipelines, which all share the same feature contract).
    """
    target = pipeline
    if hasattr(target, "_fitted_bases") and target._fitted_bases:
        target = next(iter(target._fitted_bases.values()))

    named_steps = getattr(target, "named_steps", None)
    if named_steps is None:
        raise ValueError(
            f"Cannot extract preprocessor/classifier from pipeline of type "
            f"{type(pipeline).__name__} — expected sklearn Pipeline or StackingEnsemble."
        )

    preprocessor = named_steps.get("pre") or named_steps.get("preprocessor")
    clf = named_steps.get("clf")
    if preprocessor is None or clf is None:
        raise ValueError(
            f"Pipeline steps {list(named_steps.keys())} do not contain expected "
            "'pre'/'preprocessor' and 'clf' names."
        )
    return preprocessor, clf


def _expected_columns(preprocessor) -> tuple[list[str], list[str]]:
    """
    Read the exact (numeric, categorical) column lists the fitted
    ColumnTransformer expects, straight from its .transformers_ attribute.
    This guarantees exact parity with whatever that specific model was
    actually trained on — safer than re-deriving lists from
    feature_contract.py, which could drift from an older model version.
    """
    num_cols, cat_cols = [], []
    for name, _, col_list in preprocessor.transformers_:
        if not isinstance(col_list, list):
            continue
        if name == "num":
            num_cols = col_list
        elif name == "cat":
            cat_cols = col_list
    return num_cols, cat_cols


def explain_prediction(
    pipeline,
    features: dict,
    top_n: int = 10,
    thresholds: FeatureThresholds | None = None,
) -> list[dict]:
    """
    Compute SHAP values for a single prediction and return the top N
    most influential features.

    Parameters
    ----------
    pipeline   : fitted sklearn Pipeline or StackingEnsemble
    features   : raw feature dict (same schema as CustomerFeatures)
    top_n      : number of top features to return
    thresholds : frozen FeatureThresholds from training (via predictor.thresholds).
                 Without this, threshold-based features (high_consumption,
                 low_margin, etc.) would be computed with all-zero defaults,
                 producing misleading SHAP attributions — the same class of
                 bug fixed in api/predictor.py for regular predictions.

    Returns
    -------
    List of dicts sorted by |shap_value| descending:
        [{"feature": str, "raw_value": any, "shap_value": float,
          "direction": "increases_churn" | "decreases_churn"}, ...]
    """
    try:
        import shap
    except ImportError:
        logger.warning("shap not installed — returning empty explanation")
        return []

    if thresholds is None:
        logger.warning(
            "explain_prediction() called without thresholds — threshold-based "
            "features (high_consumption, low_margin, ...) will use zero "
            "defaults, potentially producing misleading SHAP attributions."
        )

    df_raw = pd.DataFrame([features])
    df_feat = build_feature_set(df_raw, thresholds=thresholds)

    preprocessor, clf = _resolve_preprocessor_and_clf(pipeline)
    available_num, available_cat = _expected_columns(preprocessor)

    # Reindex to the fitted model's exact expected columns so the
    # ColumnTransformer doesn't raise on engineered features that couldn't
    # be computed from a raw API payload (e.g. date-derived features) —
    # same reasoning as predictor._features(). Imputers fill these NaNs
    # with the training-time median/constant, consistent with how the
    # model was fit.
    all_expected = available_num + available_cat
    missing = [c for c in all_expected if c not in df_feat.columns]
    if missing:
        logger.debug("Reindexing %d columns not computable from API payload: %s", len(missing), sorted(missing))
    df_feat = df_feat.reindex(columns=list(df_feat.columns) + missing)

    X = df_feat[all_expected]
    X_transformed = preprocessor.transform(X)

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
        # Recover original feature name from OHE-expanded names
        # e.g. "channel_sales_online" → "channel_sales", value="online"
        raw_val = None
        display_name = fname
        for cat_feat in available_cat:
            if fname.startswith(cat_feat + "_"):
                display_name = cat_feat
                category_value = fname[len(cat_feat) + 1 :]
                raw_val = features.get(cat_feat, category_value)
                break
        if raw_val is None:
            raw_val = features.get(fname, None)

        results.append(
            {
                "feature": display_name,
                "raw_value": raw_val,
                "shap_value": round(float(sval), 6),
                "direction": "increases_churn" if sval > 0 else "decreases_churn",
            }
        )

    return results
