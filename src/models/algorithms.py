"""
src/models/algorithms.py
--------------------------
Sklearn-compatible pipeline builders for every algorithm in the sweep.
Each returns a full Pipeline (preprocessor + classifier) ready to fit.

Algorithms
----------
- Logistic Regression  (baseline, linear)
- Random Forest        (bagging ensemble)
- Gradient Boosting    (sklearn, sequential boosting)
- XGBoost              (optimised gradient boosting)
- LightGBM             (leaf-wise gradient boosting)
- CatBoost             (native categorical handling)
- MLP                  (sklearn neural net, 3-layer)
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import settings


def _preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    num_pipe = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        [
            ("imp", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("num", num_pipe, numeric_cols), ("cat", cat_pipe, categorical_cols)],
        remainder="drop",
    )


def build_logistic_regression(
    num_cols: list[str],
    cat_cols: list[str],
    params: dict | None = None,
) -> Pipeline:
    p = params or {}
    clf = LogisticRegression(
        C=p.get("C", 0.1),
        max_iter=p.get("max_iter", 1000),
        class_weight="balanced",
        solver="lbfgs",
        random_state=settings.model_random_state,
    )
    return Pipeline([("pre", _preprocessor(num_cols, cat_cols)), ("clf", clf)])


def build_random_forest(
    num_cols: list[str],
    cat_cols: list[str],
    params: dict | None = None,
) -> Pipeline:
    p = params or {}
    clf = RandomForestClassifier(
        n_estimators=p.get("n_estimators", 300),
        max_depth=p.get("max_depth", 8),
        min_samples_leaf=p.get("min_samples_leaf", 20),
        class_weight="balanced",
        random_state=settings.model_random_state,
        n_jobs=-1,
    )
    return Pipeline([("pre", _preprocessor(num_cols, cat_cols)), ("clf", clf)])


def build_gradient_boosting(
    num_cols: list[str],
    cat_cols: list[str],
    params: dict | None = None,
) -> Pipeline:
    p = params or {}
    clf = GradientBoostingClassifier(
        n_estimators=p.get("n_estimators", 300),
        learning_rate=p.get("learning_rate", 0.05),
        max_depth=p.get("max_depth", 4),
        subsample=p.get("subsample", 0.8),
        min_samples_leaf=p.get("min_samples_leaf", 20),
        random_state=settings.model_random_state,
    )
    return Pipeline([("pre", _preprocessor(num_cols, cat_cols)), ("clf", clf)])


def build_xgboost(
    num_cols: list[str],
    cat_cols: list[str],
    params: dict | None = None,
) -> Pipeline:
    from xgboost import XGBClassifier

    p = params or {}
    clf = XGBClassifier(
        n_estimators=p.get("n_estimators", 300),
        learning_rate=p.get("learning_rate", 0.05),
        max_depth=p.get("max_depth", 4),
        subsample=p.get("subsample", 0.8),
        colsample_bytree=p.get("colsample_bytree", 0.8),
        scale_pos_weight=p.get("scale_pos_weight", 8),  # handles 11% churn
        eval_metric="logloss",
        random_state=settings.model_random_state,
        n_jobs=-1,
        verbosity=0,
    )
    return Pipeline([("pre", _preprocessor(num_cols, cat_cols)), ("clf", clf)])


def build_lightgbm(
    num_cols: list[str],
    cat_cols: list[str],
    params: dict | None = None,
) -> Pipeline:
    from lightgbm import LGBMClassifier

    p = params or {}
    clf = LGBMClassifier(
        n_estimators=p.get("n_estimators", 300),
        learning_rate=p.get("learning_rate", 0.05),
        max_depth=p.get("max_depth", 6),
        num_leaves=p.get("num_leaves", 31),
        subsample=p.get("subsample", 0.8),
        colsample_bytree=p.get("colsample_bytree", 0.8),
        class_weight="balanced",
        random_state=settings.model_random_state,
        n_jobs=-1,
        verbose=-1,
    )
    return Pipeline([("pre", _preprocessor(num_cols, cat_cols)), ("clf", clf)])


def build_catboost(
    num_cols: list[str],
    cat_cols: list[str],
    params: dict | None = None,
) -> Pipeline:
    from catboost import CatBoostClassifier

    p = params or {}
    # CatBoost handles categoricals natively — pass indices after preprocessing
    clf = CatBoostClassifier(
        iterations=p.get("iterations", 300),
        learning_rate=p.get("learning_rate", 0.05),
        depth=p.get("depth", 6),
        auto_class_weights="Balanced",
        random_seed=settings.model_random_state,
        verbose=0,
    )
    return Pipeline([("pre", _preprocessor(num_cols, cat_cols)), ("clf", clf)])


def build_mlp(
    num_cols: list[str],
    cat_cols: list[str],
    params: dict | None = None,
) -> Pipeline:
    """
    3-layer MLP via sklearn MLPClassifier.
    Fixed funnel (128→64→32). LR=0.01 ensures convergence in 500 iters.
    """
    p = params or {}
    n_features = len(num_cols) + len(cat_cols)
    clf = MLPClassifier(
        hidden_layer_sizes=p.get("hidden_layer_sizes", (128, 64, 32)),  # proper funnel
        activation=p.get("activation", "relu"),
        learning_rate_init=p.get("learning_rate_init", 0.01),  # was 0.001, too low
        alpha=p.get("alpha", 0.001),
        batch_size=p.get("batch_size", 128),
        max_iter=p.get("max_iter", 500),
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=settings.model_random_state,
    )
    return Pipeline([("pre", _preprocessor(num_cols, cat_cols)), ("clf", clf)])


ALGORITHM_REGISTRY = {
    "LogisticRegression": build_logistic_regression,
    "RandomForest": build_random_forest,
    "GradientBoosting": build_gradient_boosting,
    "XGBoost": build_xgboost,
    "LightGBM": build_lightgbm,
    "CatBoost": build_catboost,
    "MLP": build_mlp,
}
