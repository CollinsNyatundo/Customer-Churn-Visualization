"""
src/models/churn_model.py
--------------------------
Full model training pipeline:
  - 7-algorithm sweep (LR, RF, GBM, XGBoost, LightGBM, CatBoost, MLP)
  - Optuna hyperparameter optimisation for the best algorithm
  - Stacking ensemble across top-N models
  - Business-aware threshold optimisation
  - MLflow tracking with feature schema hash
  - FeatureThresholds frozen from training split
"""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from src.config import settings
from src.features.engineering import FEATURE_ENGINEERING_VERSION, FeatureThresholds, build_feature_set
from src.models.algorithms import ALGORITHM_REGISTRY
from src.models.ensemble import StackingEnsemble
from src.models.feature_contract import (
    TARGET,
    feature_schema_hash,
    get_available,
    validate_features,
)
from src.models.optimization import optimise
from src.models.threshold import optimise_threshold, threshold_sweep
from src.tracking.mlflow_tracker import log_metrics, log_params, model_run, register_model

log = logging.getLogger(__name__)


def _lift_at(y_true: np.ndarray, y_prob: np.ndarray, pct: float) -> float:
    n = max(int(len(y_prob) * pct), 1)
    top_idx = np.argsort(y_prob)[::-1][:n]
    baseline = y_true.mean()
    return float(y_true[top_idx].mean() / baseline) if baseline else 0.0


def _log_threshold_sweep(y_test, y_prob) -> None:
    rows = threshold_sweep(y_test, y_prob)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    thresholds = [r["threshold"] for r in rows]
    axes[0].plot(thresholds, [r["precision"] for r in rows], label="Precision")
    axes[0].plot(thresholds, [r["recall"] for r in rows], label="Recall")
    axes[0].plot(thresholds, [r["f1"] for r in rows], label="F1")
    axes[0].axvline(0.5, color="gray", linestyle="--", alpha=0.5, label="Default (0.5)")
    axes[0].set(title="Precision / Recall / F1 by threshold", xlabel="Threshold")
    axes[0].legend()
    axes[1].plot(thresholds, [r["flagged"] for r in rows], color="#E24B4A")
    axes[1].set(title="Customers flagged for outreach", xlabel="Threshold", ylabel="Count")
    fig.tight_layout()
    mlflow.log_figure(fig, "threshold_sweep.png")
    plt.close(fig)


def train_single(
    name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    num_cols: list[str],
    cat_cols: list[str],
    params: dict | None = None,
    optimise_hp: bool = False,
    n_trials: int = 30,
) -> dict:
    """Train one algorithm, log to MLflow, return result dict."""

    if optimise_hp:
        log.info("Running Optuna for %s (%d trials)...", name, n_trials)
        params = optimise(name, X_train, y_train, n_trials=n_trials)
        log.info("%s best params: %s", name, params)

    builder = ALGORITHM_REGISTRY[name]
    pipeline = builder(num_cols, cat_cols, params or {})

    cv = StratifiedKFold(
        n_splits=settings.model_cv_folds,
        shuffle=True,
        random_state=settings.model_random_state,
    )
    cv_res = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=["roc_auc", "f1", "average_precision", "precision", "recall"],
    )
    pipeline.fit(X_train, y_train)

    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    # Business-optimal threshold
    opt_threshold, _ = optimise_threshold(y_test.values, y_prob, metric="f2")
    y_pred_opt = (y_prob >= opt_threshold).astype(int)
    rpt_opt = classification_report(y_test, y_pred_opt, output_dict=True)

    return {
        "name": name,
        "pipeline": pipeline,
        "y_test": y_test.values,
        "y_prob": y_prob,
        "y_pred": y_pred,
        "cv_auc": float(cv_res["test_roc_auc"].mean()),
        "cv_auc_std": float(cv_res["test_roc_auc"].std()),
        "cv_f1": float(cv_res["test_f1"].mean()),
        "cv_ap": float(cv_res["test_average_precision"].mean()),
        "test_auc": float(roc_auc_score(y_test, y_prob)),
        "test_ap": float(average_precision_score(y_test, y_prob)),
        "test_f1": float(cv_res["test_f1"].mean()),
        "brier": float(brier_score_loss(y_test, y_prob)),
        "lift_10": _lift_at(y_test.values, y_prob, 0.10),
        "lift_20": _lift_at(y_test.values, y_prob, 0.20),
        "opt_threshold": opt_threshold,
        "opt_f1": float(rpt_opt["weighted avg"]["f1-score"]),
        "opt_recall": float(rpt_opt["1"]["recall"]),
        "report": classification_report(y_test, y_pred, output_dict=True),
        "params": params or {},
    }


def train(
    df: pd.DataFrame,
    algorithms: list[str] | None = None,
    optimise_best: bool = True,
    build_stack: bool = True,
    n_optuna_trials: int = 30,
) -> str:
    """
    Full training run: sweep → optimise best → stack ensemble → register.

    Parameters
    ----------
    df              : fully merged + feature-engineered DataFrame
    algorithms      : list of algorithm names (default: all 7)
    optimise_best   : run Optuna on the top-performing algorithm
    build_stack     : build stacking ensemble over top-3 algorithms
    n_optuna_trials : Optuna trials for the best algorithm

    Returns
    -------
    MLflow run_id of the best single model (or ensemble if better)
    """
    algorithms = algorithms or list(ALGORITHM_REGISTRY.keys())

    # ── Split & freeze thresholds ────────────────────────────────────────────
    df_train, df_test = train_test_split(
        df,
        test_size=settings.model_test_size,
        stratify=df[TARGET].astype(int),
        random_state=settings.model_random_state,
    )
    df_train, df_val = train_test_split(
        df_train,
        test_size=0.125,
        stratify=df_train[TARGET].astype(int),
        random_state=settings.model_random_state,
    )

    thresholds = FeatureThresholds.from_dataframe(df_train)
    ref_date = pd.Timestamp("2026-06-22")  # frozen reference date

    X_train = build_feature_set(df_train, thresholds=thresholds, reference_date=ref_date)
    X_val = build_feature_set(df_val, thresholds=thresholds, reference_date=ref_date)
    X_test = build_feature_set(df_test, thresholds=thresholds, reference_date=ref_date)

    validate_features(X_train, strict=False)
    schema_hash = feature_schema_hash(X_train)
    num_cols, cat_cols = get_available(X_train)

    y_train = df_train[TARGET].astype(int)
    y_val = df_val[TARGET].astype(int)
    y_test = df_test[TARGET].astype(int)

    log.info(
        "Training split: train=%d val=%d test=%d | churn=%.2f%% | schema_hash=%s",
        len(X_train),
        len(X_val),
        len(X_test),
        float(y_train.mean()) * 100,
        schema_hash,
    )

    # ── Algorithm sweep ───────────────────────────────────────────────────────
    results: dict[str, dict] = {}
    for name in algorithms:
        log.info("Training %s...", name)
        try:
            res = train_single(name, X_train, X_test, y_train, y_test, num_cols, cat_cols)
            results[name] = res
            log.info(
                "  %s → AUC=%.4f  F1=%.4f  Brier=%.4f  Lift@10%%=%.2fx",
                name,
                res["test_auc"],
                res["test_f1"],
                res["brier"],
                res["lift_10"],
            )
        except Exception as e:
            log.error("Algorithm %s failed: %s", name, e, exc_info=True)

    if not results:
        raise RuntimeError("All algorithms failed during training sweep.")

    # ── Optimise best single model ────────────────────────────────────────────
    best_name = max(results, key=lambda n: results[n]["test_auc"])
    if optimise_best:
        log.info("Optimising %s with Optuna (%d trials)...", best_name, n_optuna_trials)
        try:
            res_opt = train_single(
                best_name,
                X_train,
                X_test,
                y_train,
                y_test,
                num_cols,
                cat_cols,
                optimise_hp=True,
                n_trials=n_optuna_trials,
            )
            if res_opt["test_auc"] > results[best_name]["test_auc"]:
                results[f"{best_name}_Optuna"] = res_opt
                log.info(
                    "Optuna improved %s: %.4f → %.4f",
                    best_name,
                    results[best_name]["test_auc"],
                    res_opt["test_auc"],
                )
        except Exception as e:
            log.warning("Optuna optimisation failed: %s", e)

    # ── Stacking ensemble ─────────────────────────────────────────────────────
    if build_stack and len(results) >= 3:
        top3 = sorted(results, key=lambda n: results[n]["test_auc"], reverse=True)[:3]
        log.info("Building stacking ensemble over: %s", top3)
        try:
            base_pipes = {n: results[n]["pipeline"] for n in top3}
            ensemble = StackingEnsemble(base_pipes, cv_folds=3)
            ensemble.fit(X_train, y_train)

            ens_prob = ensemble.predict_proba(X_test)[:, 1]
            ens_auc = roc_auc_score(y_test, ens_prob)
            ens_ap = average_precision_score(y_test, ens_prob)
            ens_brier = brier_score_loss(y_test, ens_prob)
            ens_threshold, _ = optimise_threshold(y_test.values, ens_prob, metric="f2")
            ens_pred = (ens_prob >= ens_threshold).astype(int)
            ens_rpt = classification_report(y_test, ens_pred, output_dict=True)

            results["StackingEnsemble"] = {
                "name": "StackingEnsemble",
                "pipeline": ensemble,
                "y_test": y_test.values,
                "y_prob": ens_prob,
                "y_pred": ens_pred,
                "cv_auc": ens_auc,
                "cv_auc_std": 0.0,
                "cv_f1": ens_rpt["weighted avg"]["f1-score"],
                "cv_ap": ens_ap,
                "test_auc": ens_auc,
                "test_ap": ens_ap,
                "test_f1": ens_rpt["weighted avg"]["f1-score"],
                "brier": ens_brier,
                "lift_10": _lift_at(y_test.values, ens_prob, 0.10),
                "lift_20": _lift_at(y_test.values, ens_prob, 0.20),
                "opt_threshold": ens_threshold,
                "opt_recall": ens_rpt["1"]["recall"],
                "opt_f1": ens_rpt["weighted avg"]["f1-score"],
                "report": ens_rpt,
                "meta_weights": ensemble.meta_weights,
                "params": {"base_models": top3},
            }
            log.info("StackingEnsemble AUC=%.4f (vs best single %.4f)", ens_auc, results[best_name]["test_auc"])
        except Exception as e:
            log.warning("Stacking ensemble failed: %s", e, exc_info=True)

    # ── Log everything to MLflow ──────────────────────────────────────────────
    overall_best = max(results, key=lambda n: results[n]["test_auc"])
    best = results[overall_best]

    with model_run(run_name=f"ChurnModel-{overall_best}") as tracking:
        log_params(
            {
                "best_algorithm": overall_best,
                "algorithms_swept": list(results.keys()),
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "num_features": len(num_cols),
                "cat_features": len(cat_cols),
                "churn_rate": round(float(y_train.mean()), 4),
                "feature_engineering_version": "2.0.0",
                "schema_hash": schema_hash,
                "thresholds_high_consumption": thresholds.high_consumption_threshold,
                "thresholds_low_margin": thresholds.low_margin_threshold,
                "opt_threshold": best["opt_threshold"],
            }
        )

        # All model metrics
        for name, res in results.items():
            prefix = name.replace(" ", "_")
            log_metrics(
                {
                    f"{prefix}_test_auc": res["test_auc"],
                    f"{prefix}_test_ap": res.get("test_ap", 0),
                    f"{prefix}_test_f1": res["test_f1"],
                    f"{prefix}_brier": res["brier"],
                    f"{prefix}_lift_10": res["lift_10"],
                    f"{prefix}_lift_20": res["lift_20"],
                    f"{prefix}_opt_f1": res.get("opt_f1", 0),
                    f"{prefix}_opt_recall": res.get("opt_recall", 0),
                }
            )

        # Threshold sweep for best model
        _log_threshold_sweep(best["y_test"], best["y_prob"])

        # Feature importance (tree-based)
        clf = getattr(best["pipeline"], "named_steps", {}).get("clf") or getattr(best["pipeline"], "_fitted_bases", {})
        if hasattr(clf, "feature_importances_"):
            try:
                pre = best["pipeline"].named_steps["pre"]
                cat_names = pre.named_transformers_["cat"].named_steps["ohe"].get_feature_names_out(cat_cols).tolist()
                feat_names = num_cols + cat_names
                imps = clf.feature_importances_[: len(feat_names)]
                top15 = np.argsort(imps)[-15:]
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.barh([feat_names[i] for i in top15], imps[top15], color="#1D9E75")
                ax.set_title(f"Top 15 Feature Importances — {overall_best}")
                mlflow.log_figure(fig, "feature_importance.png")
                plt.close(fig)
            except Exception:
                pass

        # Save complete thresholds artifact — use asdict() so any new fields
        # added to FeatureThresholds are automatically included, preventing
        # this from silently going stale again.
        from dataclasses import asdict as _asdict

        thresholds_payload = _asdict(thresholds)
        thresholds_payload["feature_engineering_version"] = FEATURE_ENGINEERING_VERSION
        thresholds_payload["schema_hash"] = schema_hash
        mlflow.log_dict(thresholds_payload, "feature_thresholds.json")

        # Log best model
        if not tracking.offline:
            _TRUSTED_TYPES = [
                "numpy.dtype",
                "numpy.ndarray",
                "numpy.core.multiarray._reconstruct",
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBClassifier",
                "lightgbm.sklearn.LGBMClassifier",
                "lightgbm.basic.Booster",
                "catboost.core.CatBoostClassifier",
                "sklearn.ensemble._forest.RandomForestClassifier",
                "sklearn.neural_network._multilayer_perceptron.MLPClassifier",
            ]
            try:
                # Use artifact_path (not name=) + registered_model_name for
                # compatibility with the models:/name/stage registry loading
                # pattern used by api/predictor.py. This registers the model
                # version directly, so register_model() below is a no-op if
                # the version is already registered.
                mlflow.sklearn.log_model(
                    best["pipeline"],
                    artifact_path="model",
                    registered_model_name=settings.mlflow_registered_model_name,
                    pip_requirements=["scikit-learn", "xgboost", "lightgbm", "catboost", "numpy", "pandas"],
                    skops_trusted_types=_TRUSTED_TYPES,
                )
            except Exception as e:
                log.error("Model artifact logging failed: %s", e)

        run_id = tracking.run_id

    register_model(run_id=run_id, stage="Staging")
    log.info(
        "Training complete. Best: %s | AUC=%.4f | Lift@10%%=%.2fx | run_id=%s",
        overall_best,
        best["test_auc"],
        best["lift_10"],
        run_id,
    )
    return run_id


def train_segmented(
    df: pd.DataFrame,
    segment_col: str = "high_consumption",
    min_segment_size: int = 200,
    **train_kwargs,
) -> dict[str, str]:
    """
    Train separate models per customer segment (e.g. high-value vs low-value).

    Parameters
    ----------
    df               : full training DataFrame (must include segment_col after
                        feature engineering, or a raw column to segment on)
    segment_col       : boolean/categorical column to split on
    min_segment_size  : skip segments with fewer rows than this
    **train_kwargs    : passed through to train()

    Returns
    -------
    dict mapping segment value -> MLflow run_id
    """
    if segment_col not in df.columns:
        raise ValueError(
            f"segment_col '{segment_col}' not in DataFrame. "
            "Run build_feature_set() first if using an engineered column."
        )

    run_ids: dict[str, str] = {}
    for segment_value, segment_df in df.groupby(segment_col):
        if len(segment_df) < min_segment_size:
            log.warning(
                "Segment %s=%s has only %d rows (< %d) — skipping.",
                segment_col,
                segment_value,
                len(segment_df),
                min_segment_size,
            )
            continue

        log.info(
            "Training segment %s=%s (%d rows, churn=%.2f%%)...",
            segment_col,
            segment_value,
            len(segment_df),
            float(segment_df[TARGET].mean()) * 100,
        )
        run_id = train(segment_df, **train_kwargs)
        run_ids[str(segment_value)] = run_id

    log.info("Segmented training complete: %d segments trained.", len(run_ids))
    return run_ids
