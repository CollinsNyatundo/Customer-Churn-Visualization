"""
src/models/optimization.py
----------------------------
Optuna-based hyperparameter optimisation for each algorithm.

Usage
-----
    from src.models.optimization import optimise
    best_params = optimise("XGBoost", X_train, y_train, n_trials=50)
"""

from __future__ import annotations

import logging
import warnings

import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score

optuna.logging.set_verbosity(optuna.logging.WARNING)
log = logging.getLogger(__name__)


def _objective(trial, algorithm: str, X, y, cv) -> float:
    """Optuna objective — returns mean CV ROC-AUC."""
    from src.models.algorithms import ALGORITHM_REGISTRY

    num_cols = [c for c in X.columns if X[c].dtype in (float, int, "bool")]
    cat_cols = [c for c in X.columns if X[c].dtype == object]

    if algorithm == "LogisticRegression":
        params = {"C": trial.suggest_float("C", 1e-3, 10, log=True)}

    elif algorithm == "RandomForest":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50),
        }

    elif algorithm == "GradientBoosting":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        }

    elif algorithm == "XGBoost":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 4, 15),
        }

    elif algorithm == "LightGBM":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }

    elif algorithm == "CatBoost":
        params = {
            "iterations": trial.suggest_int("iterations", 100, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "depth": trial.suggest_int("depth", 4, 8),
        }

    elif algorithm == "MLP":
        n1 = trial.suggest_int("layer1", 64, 256)
        n2 = trial.suggest_int("layer2", 32, 128)
        n3 = trial.suggest_int("layer3", 16, 64)
        params = {
            "hidden_layer_sizes": (n1, n2, n3),
            "alpha": trial.suggest_float("alpha", 1e-4, 0.1, log=True),
            "learning_rate_init": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        }
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    builder = ALGORITHM_REGISTRY[algorithm]
    pipeline = builder(num_cols, cat_cols, params)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)

    return float(scores.mean())


def optimise(
    algorithm: str,
    X,
    y,
    n_trials: int = 30,
    cv_folds: int = 3,
    seed: int = 42,
) -> dict:
    """
    Run Optuna hyperparameter search for a given algorithm.

    Parameters
    ----------
    algorithm : name from ALGORITHM_REGISTRY
    X, y      : training data
    n_trials  : number of Optuna trials
    cv_folds  : folds for cross-validation inside each trial

    Returns
    -------
    best_params dict suitable for passing to ALGORITHM_REGISTRY[algorithm]()
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )
    study.optimize(
        lambda trial: _objective(trial, algorithm, X, y, cv),
        n_trials=n_trials,
        show_progress_bar=False,
    )
    log.info(
        "Optuna %s: best AUC=%.4f in %d trials. Params: %s",
        algorithm,
        study.best_value,
        n_trials,
        study.best_params,
    )
    return study.best_params
