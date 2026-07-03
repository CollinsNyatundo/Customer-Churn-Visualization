"""
src/models/threshold.py
------------------------
Business-aware decision threshold optimisation.

The default 0.5 threshold maximises accuracy but is often wrong for churn:
  - False negatives (missed churners) = lost revenue
  - False positives (unnecessary outreach) = small cost

This module finds the threshold that optimises a configurable business metric.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


def optimise_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
    fn_cost: float = 10.0,
    fp_cost: float = 1.0,
) -> tuple[float, float]:
    """
    Find the decision threshold that optimises a business metric.

    Parameters
    ----------
    y_true   : true binary labels
    y_prob   : predicted probabilities for the positive class
    metric   : "f1" | "f2" | "business_cost" | "recall_at_precision"
    fn_cost  : relative cost of a false negative (missed churner)
    fp_cost  : relative cost of a false positive (unnecessary outreach)

    Returns
    -------
    (optimal_threshold, metric_value_at_threshold)
    """
    thresholds = np.linspace(0.05, 0.95, 181)
    best_threshold, best_value = 0.5, -np.inf

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)

        if metric == "f1":
            value = 2 * precision * recall / max(precision + recall, 1e-9)
        elif metric == "f2":
            # F2 weights recall 2× — good for churn where missing churners is costly
            beta = 2
            value = (1 + beta**2) * precision * recall / max(beta**2 * precision + recall, 1e-9)
        elif metric == "business_cost":
            # Minimise total cost — negate so we maximise
            value = -(fn * fn_cost + fp * fp_cost)
        elif metric == "recall_at_precision":
            # Maximise recall while keeping precision above 0.5
            value = recall if precision >= 0.5 else -1.0
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if value > best_value:
            best_value, best_threshold = value, t

    log.info(
        "Threshold optimised for '%s': threshold=%.3f, value=%.4f",
        metric,
        best_threshold,
        best_value,
    )
    return best_threshold, best_value


def threshold_sweep(y_true: np.ndarray, y_prob: np.ndarray) -> list[dict]:
    """
    Return a summary table of key metrics at every threshold.
    Useful for choosing a threshold based on business requirements.
    """
    rows = []
    for t in np.linspace(0.05, 0.95, 19):
        y_pred = (y_prob >= t).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        rows.append(
            {
                "threshold": round(t, 2),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "flagged": int(y_pred.sum()),
                "caught_churners": tp,
                "missed_churners": fn,
            }
        )
    return rows
