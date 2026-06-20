"""
scripts/run_pipeline_e2e.py
----------------------------
End-to-end pipeline runner:
  1. Multi-source data ingestion
  2. Feature engineering
  3. Multi-model comparison (Logistic Regression, Random Forest, Gradient Boosting)
  4. Class imbalance handling
  5. Full MLflow tracking (params, metrics, artifacts)
  6. SHAP feature importance
  7. Calibration analysis
  8. Business lift metrics
  9. Inference demo on holdout set
  10. Printed baseline report

Run:
    python scripts/run_pipeline_e2e.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


import mlflow
import mlflow.sklearn
import shap
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

# ── Add project root to path ─────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///" + str(ROOT / "mlflow.db")

from src.data.pipeline import MultiSourcePipeline
from src.data.sources.bcg_source import BCGClientSource, BCGPriceSource
from src.features.engineering import build_feature_set
from src.logging_config import configure_logging

configure_logging(level="INFO", json=False)
log = logging.getLogger("e2e_pipeline")

# ── Feature sets ─────────────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    "cons_12m", "cons_gas_12m", "cons_last_month", "imp_cons",
    "net_margin", "margin_gross_pow_ele", "num_years_antig", "pow_max",
    "nb_prod_act", "forecast_discount_energy",
    "nps_score", "satisfaction_score", "num_contacts_6m", "last_contact_days_ago",
    "num_tickets_6m", "avg_resolution_hours", "escalations_6m", "open_tickets",
    "num_late_payments_12m", "avg_days_late", "total_outstanding", "discount_pct",
    "contract_duration_days", "months_to_renewal", "cons_growth_rate",
    "price_spread_var", "price_spread_fix", "margin_efficiency",
    "cross_source_risk_score", "engagement_score",
]
CATEGORICAL_FEATURES = [
    "channel_sales", "activity_new", "origin_up",
    "contract_type", "payment_method", "top_ticket_category",
]
TARGET = "churn"

MODELS = {
    "LogisticRegression": LogisticRegression(
        max_iter=1000, class_weight="balanced", C=0.1, random_state=42
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=200, max_depth=6, class_weight="balanced",
        random_state=42, n_jobs=-1,
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        subsample=0.8, random_state=42,
    ),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_sklearn_pipeline(clf) -> Pipeline:
    available_num = [c for c in NUMERIC_FEATURES]
    available_cat = [c for c in CATEGORICAL_FEATURES]
    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    pre = ColumnTransformer([
        ("num", num_pipe, available_num),
        ("cat", cat_pipe, available_cat),
    ], remainder="drop")
    return Pipeline([("pre", pre), ("clf", clf)])


def compute_lift(y_true: np.ndarray, y_prob: np.ndarray, pct: float) -> float:
    """Lift at top pct% of predicted probabilities."""
    n = max(int(len(y_prob) * pct), 1)
    top_idx = np.argsort(y_prob)[::-1][:n]
    baseline = y_true.mean()
    if baseline == 0:
        return 0.0
    return y_true[top_idx].mean() / baseline


def plot_roc(models_results: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#1D9E75", "#0C7BB3", "#E24B4A"]
    for (name, res), col in zip(models_results.items(), colors):
        fpr, tpr, _ = roc_curve(res["y_test"], res["y_prob"])
        ax.plot(fpr, tpr, color=col, lw=2,
                label=f"{name} (AUC={res['test_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="FPR", ylabel="TPR", title="ROC Curves — Model Comparison")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("ROC curve saved → %s", path)


def plot_pr(models_results: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#1D9E75", "#0C7BB3", "#E24B4A"]
    for (name, res), col in zip(models_results.items(), colors):
        prec, rec, _ = precision_recall_curve(res["y_test"], res["y_prob"])
        ap = res["avg_precision"]
        ax.plot(rec, prec, color=col, lw=2, label=f"{name} (AP={ap:.3f})")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curves")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("PR curve saved → %s", path)


def plot_calibration(models_results: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#1D9E75", "#0C7BB3", "#E24B4A"]
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    for (name, res), col in zip(models_results.items(), colors):
        frac_pos, mean_pred = calibration_curve(res["y_test"], res["y_prob"], n_bins=10)
        ax.plot(mean_pred, frac_pos, "o-", color=col, lw=2,
                label=f"{name} (Brier={res['brier']:.3f})")
    ax.set(xlabel="Mean predicted probability", ylabel="Fraction of positives",
           title="Calibration Curves")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Calibration curve saved → %s", path)


def plot_confusion(y_test, y_pred, name: str, path: Path) -> None:
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=["Retained", "Churned"],
           yticklabels=["Retained", "Churned"],
           xlabel="Predicted", ylabel="Actual",
           title=f"Confusion Matrix — {name}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14, fontweight="bold")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_shap(pipeline: Pipeline, X_sample: pd.DataFrame,
              available_num: list, available_cat: list, path: Path) -> None:
    pre  = pipeline.named_steps["pre"]
    clf  = pipeline.named_steps["clf"]
    X_t  = pre.transform(X_sample)

    try:
        cat_names = (
            pre.named_transformers_["cat"]
            .named_steps["ohe"]
            .get_feature_names_out(available_cat)
            .tolist()
        )
    except Exception:
        cat_names = []

    feature_names = available_num + cat_names
    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(X_t)
    if isinstance(sv, list):
        sv = sv[1]

    n = min(len(feature_names), sv.shape[1])
    mean_abs = np.abs(sv[:, :n]).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[-15:]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([feature_names[i] for i in top_idx], mean_abs[top_idx], color="#1D9E75")
    ax.set(title="Top 15 Features — Mean |SHAP value|",
           xlabel="Mean absolute SHAP value")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("SHAP importance saved → %s", path)
    return feature_names, mean_abs


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)

    print("\n" + "═" * 65)
    print("  CUSTOMER CHURN — END-TO-END PIPELINE")
    print("═" * 65)

    # ── 1. Data ingestion ────────────────────────────────────────────────────
    print("\n[1/6] Running MultiSourcePipeline …")
    t0 = time.time()

    pipeline = MultiSourcePipeline(
        client_source=BCGClientSource(path=ROOT / "data/raw/client_data.csv"),
        price_source=BCGPriceSource(path=ROOT / "data/raw/price_data.csv"),
    )
    df_raw, pipe_meta = pipeline.run()
    print(f"      Shape after merge : {df_raw.shape}")
    print(f"      Sources active    : BCG client + BCG price + CRM (syn) + Support (syn) + Billing (syn)")
    print(f"      Elapsed           : {time.time()-t0:.1f}s")

    # ── 2. Feature engineering ───────────────────────────────────────────────
    print("\n[2/6] Feature engineering …")
    df = build_feature_set(df_raw)
    new_cols = sorted(set(df.columns) - set(df_raw.columns))
    print(f"      Final shape       : {df.shape}")
    print(f"      New features ({len(new_cols)})  : {new_cols}")
    print(f"      Churn rate        : {df[TARGET].mean():.2%}")
    print(f"      Null rate         : {df.isnull().mean().mean():.2%}")

    # ── 3. Train / test split ────────────────────────────────────────────────
    print("\n[3/6] Splitting data …")
    avail_num = [c for c in NUMERIC_FEATURES if c in df.columns]
    avail_cat = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    X = df[avail_num + avail_cat]
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=0.50, stratify=y_test, random_state=42
    )
    print(f"      Train : {len(X_train):>5} rows  ({y_train.mean():.2%} churn)")
    print(f"      Val   : {len(X_val):>5} rows  ({y_val.mean():.2%} churn)")
    print(f"      Test  : {len(X_test):>5} rows  ({y_test.mean():.2%} churn)")
    print(f"      Features: {len(avail_num)} numeric + {len(avail_cat)} categorical")

    # ── 4. Multi-model training + MLflow ─────────────────────────────────────
    print("\n[4/6] Training models with 5-fold CV …")
    mlflow.set_tracking_uri("sqlite:///" + str(ROOT / "mlflow.db"))
    mlflow.set_experiment("churn-model-comparison")

    cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for model_name, clf in MODELS.items():
        print(f"\n      ── {model_name} ──")
        sk_pipe = build_sklearn_pipeline(clf)

        with mlflow.start_run(run_name=model_name) as run:
            # Log params
            mlflow.log_params({
                "model": model_name,
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "n_numeric": len(avail_num),
                "n_categorical": len(avail_cat),
                "churn_rate_train": round(float(y_train.mean()), 4),
                **{f"clf_{k}": str(v)
                   for k, v in list(clf.get_params().items())[:8]},
            })

            # Cross-validation
            cv_res = cross_validate(
                sk_pipe, X_train, y_train, cv=cv,
                scoring=["roc_auc", "f1", "average_precision",
                         "precision", "recall"],
                return_train_score=True,
            )
            cv_auc   = cv_res["test_roc_auc"].mean()
            cv_f1    = cv_res["test_f1"].mean()
            cv_ap    = cv_res["test_average_precision"].mean()
            cv_prec  = cv_res["test_precision"].mean()
            cv_rec   = cv_res["test_recall"].mean()
            cv_auc_std = cv_res["test_roc_auc"].std()

            print(f"         CV AUC  : {cv_auc:.4f} ± {cv_auc_std:.4f}")
            print(f"         CV F1   : {cv_f1:.4f}")
            print(f"         CV AP   : {cv_ap:.4f}")

            mlflow.log_metrics({
                "cv_auc_mean": round(cv_auc, 4),
                "cv_auc_std":  round(cv_auc_std, 4),
                "cv_f1_mean":  round(cv_f1, 4),
                "cv_ap_mean":  round(cv_ap, 4),
                "cv_precision_mean": round(cv_prec, 4),
                "cv_recall_mean":    round(cv_rec, 4),
            })

            # Final fit on full training set
            sk_pipe.fit(X_train, y_train)

            # Hold-out test metrics
            y_prob = sk_pipe.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
            test_auc = roc_auc_score(y_test, y_prob)
            test_f1  = f1_score(y_test, y_pred)
            test_ap  = average_precision_score(y_test, y_prob)
            brier    = brier_score_loss(y_test, y_prob)
            lift10   = compute_lift(y_test.values, y_prob, 0.10)
            lift20   = compute_lift(y_test.values, y_prob, 0.20)
            report   = classification_report(y_test, y_pred, output_dict=True)

            print(f"         Test AUC: {test_auc:.4f}")
            print(f"         Test F1 : {test_f1:.4f}")
            print(f"         Brier   : {brier:.4f}")
            print(f"         Lift@10%: {lift10:.2f}x")
            print(f"         Lift@20%: {lift20:.2f}x")

            mlflow.log_metrics({
                "test_auc":       round(test_auc, 4),
                "test_f1":        round(test_f1, 4),
                "test_avg_prec":  round(test_ap, 4),
                "test_brier":     round(brier, 4),
                "test_precision": round(report["1"]["precision"], 4),
                "test_recall":    round(report["1"]["recall"], 4),
                "test_accuracy":  round(report["accuracy"], 4),
                "lift_at_10pct":  round(lift10, 3),
                "lift_at_20pct":  round(lift20, 3),
                **{f"pipe_meta_{k}": float(v)
                   for k, v in pipe_meta.items()
                   if isinstance(v, (int, float))},
            })

            # Confusion matrix artifact
            cm_path = reports / f"confusion_{model_name}.png"
            plot_confusion(y_test, y_pred, model_name, cm_path)
            mlflow.log_artifact(str(cm_path), artifact_path="confusion_matrices")

            # Log model
            mlflow.sklearn.log_model(sk_pipe, name="model",
                                  skops_trusted_types=["numpy.dtype",
                                      "numpy.ndarray", "numpy.int64",
                                      "numpy.float64", "builtins.int",
                                      "builtins.float", "builtins.str",
                                      "builtins.list", "builtins.dict",
                                      "builtins.tuple", "builtins.bool"])

            results[model_name] = {
                "pipeline":   sk_pipe,
                "y_test":     y_test.values,
                "y_prob":     y_prob,
                "y_pred":     y_pred,
                "test_auc":   test_auc,
                "test_f1":    test_f1,
                "avg_precision": test_ap,
                "brier":      brier,
                "lift10":     lift10,
                "lift20":     lift20,
                "cv_auc":     cv_auc,
                "run_id":     run.info.run_id,
                "report":     report,
            }

    # ── 5. Comparison plots ───────────────────────────────────────────────────
    print("\n[5/6] Generating comparison artifacts …")

    roc_path  = reports / "roc_comparison.png"
    pr_path   = reports / "pr_comparison.png"
    cal_path  = reports / "calibration_comparison.png"
    shap_path = reports / "shap_importance.png"

    plot_roc(results, roc_path)
    plot_pr(results, pr_path)
    plot_calibration(results, cal_path)

    # SHAP on best model (GradientBoosting)
    best_name = max(results, key=lambda n: results[n]["test_auc"])
    best_pipe = results[best_name]["pipeline"]

    # Only tree-based models support TreeExplainer
    if best_name in ("GradientBoosting", "RandomForest"):
        sample = X_test.sample(min(300, len(X_test)), random_state=42)
        try:
            feat_names, shap_vals = plot_shap(
                best_pipe, sample, avail_num, avail_cat, shap_path
            )
        except Exception as e:
            log.warning("SHAP plot skipped: %s", e)

    # Log comparison plots to MLflow under the best run
    with mlflow.start_run(run_id=results[best_name]["run_id"]):
        for p in [roc_path, pr_path, cal_path, shap_path]:
            if p.exists():
                mlflow.log_artifact(str(p), artifact_path="comparison_plots")

    # ── 6. Inference demo ─────────────────────────────────────────────────────
    print("\n[6/6] Running inference demo on 10 holdout customers …")
    sample_customers = X_val.head(10).copy()
    best_pipeline    = results[best_name]["pipeline"]
    probs = best_pipeline.predict_proba(sample_customers)[:, 1]
    tiers = ["high" if p >= 0.60 else "medium" if p >= 0.30 else "low" for p in probs]

    print(f"\n      Model used: {best_name} (Test AUC={results[best_name]['test_auc']:.4f})")
    print(f"\n      {'Customer':<12} {'Churn Prob':>11} {'Risk Tier':<10} {'Actual':<8}")
    print("      " + "─" * 45)
    for i, (prob, tier) in enumerate(zip(probs, tiers)):
        actual = "CHURN" if y_val.values[i] else "retain"
        cid    = df.index[i] if i < len(df) else f"CL{i:05d}"
        print(f"      CL{i:05d}     {prob:>10.4f}   {tier:<10} {actual:<8}")

    # ── Baseline report ───────────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("  BASELINE METRICS REPORT")
    print("═" * 65)

    print(f"\n  Dataset")
    print(f"  {'Total customers':<30} {len(df):>8,}")
    print(f"  {'Train / Val / Test':<30} {len(X_train):>4,} / {len(X_val):>4,} / {len(X_test):>4,}")
    print(f"  {'Churn rate (overall)':<30} {y.mean():>8.2%}")
    print(f"  {'Numeric features':<30} {len(avail_num):>8}")
    print(f"  {'Categorical features':<30} {len(avail_cat):>8}")
    print(f"  {'Engineered features':<30} {len(new_cols):>8}")
    print(f"  {'Total features':<30} {len(avail_num)+len(avail_cat):>8}")

    print(f"\n  {'Model':<22} {'CV AUC':>8} {'Test AUC':>9} {'Test F1':>8} {'Brier':>7} {'Lift@10%':>9} {'Lift@20%':>9}")
    print("  " + "─" * 74)
    for name, res in results.items():
        marker = " ◀ BEST" if name == best_name else ""
        print(f"  {name:<22} {res['cv_auc']:>8.4f} {res['test_auc']:>9.4f} "
              f"{res['test_f1']:>8.4f} {res['brier']:>7.4f} "
              f"{res['lift10']:>8.2f}x {res['lift20']:>8.2f}x{marker}")

    best = results[best_name]
    rpt  = best["report"]
    print(f"\n  Best model ({best_name}) — detailed classification report")
    print(f"  {'':>20} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>9}")
    print("  " + "─" * 60)
    for label, lname in [(0, "Retained"), (1, "Churned (positive)")]:
        r = rpt[str(label)]
        print(f"  {lname:<20} {r['precision']:>10.4f} {r['recall']:>8.4f} "
              f"{r['f1-score']:>8.4f} {int(r['support']):>9,}")
    print(f"  {'Accuracy':<20} {'':>10} {'':>8} {rpt['accuracy']:>8.4f} "
          f"{int(rpt['macro avg']['support']):>9,}")

    print(f"\n  Business lift ({best_name})")
    print(f"  {'Lift at top 10% scored':<35} {best['lift10']:>6.2f}x baseline churn rate")
    print(f"  {'Lift at top 20% scored':<35} {best['lift20']:>6.2f}x baseline churn rate")

    print(f"\n  MLflow tracking")
    print(f"  {'Tracking URI':<30} {mlflow.get_tracking_uri()}")
    print(f"  {'Experiment':<30} churn-model-comparison")
    print(f"  {'Runs logged':<30} {len(results)}")
    print(f"  {'Best run ID':<30} {best['run_id']}")
    print(f"  Artifacts: {reports}")

    # Save baseline JSON
    baseline = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "dataset": {
            "total_rows": int(len(df)),
            "train_rows": int(len(X_train)),
            "val_rows":   int(len(X_val)),
            "test_rows":  int(len(X_test)),
            "churn_rate": round(float(y.mean()), 4),
            "n_features": len(avail_num) + len(avail_cat),
            "n_engineered": len(new_cols),
        },
        "models": {
            name: {
                "cv_auc":    round(res["cv_auc"], 4),
                "test_auc":  round(res["test_auc"], 4),
                "test_f1":   round(res["test_f1"], 4),
                "brier":     round(res["brier"], 4),
                "lift_10":   round(res["lift10"], 3),
                "lift_20":   round(res["lift20"], 3),
                "avg_prec":  round(res["avg_precision"], 4),
                "precision": round(res["report"]["1"]["precision"], 4),
                "recall":    round(res["report"]["1"]["recall"], 4),
                "run_id":    res["run_id"],
            }
            for name, res in results.items()
        },
        "best_model": best_name,
    }
    baseline_path = reports / "baseline_metrics.json"
    baseline_path.write_text(json.dumps(baseline, indent=2))
    print(f"\n  Baseline saved → {baseline_path}")
    print("\n" + "═" * 65)
    print("  PIPELINE COMPLETE ✓")
    print("═" * 65 + "\n")


if __name__ == "__main__":
    main()
