"""
src/models/ensemble.py
-----------------------
Stacking ensemble: base learners → meta-learner.

Architecture
------------
Level 0 (base learners): XGBoost, LightGBM, CatBoost, RandomForest, MLP
Level 1 (meta-learner):  Logistic Regression trained on out-of-fold predictions

This is harder to overfit than simple averaging because:
  - Base learners use out-of-fold predictions (no data leakage)
  - Meta-learner learns optimal weighting from held-out predictions
  - Diverse algorithms cover different parts of the hypothesis space
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src.config import settings

log = logging.getLogger(__name__)


class StackingEnsemble:
    """
    Two-level stacking ensemble.

    Usage
    -----
        ensemble = StackingEnsemble(base_pipelines)
        ensemble.fit(X_train, y_train)
        probs = ensemble.predict_proba(X_test)[:, 1]
    """

    def __init__(
        self,
        base_pipelines: dict[str, object],
        cv_folds: int = 5,
        meta_C: float = 1.0,
    ):
        """
        Parameters
        ----------
        base_pipelines : {name: sklearn Pipeline} — the level-0 learners
        cv_folds       : folds for out-of-fold generation
        meta_C         : regularisation for the meta-learner
        """
        self.base_pipelines = base_pipelines
        self.cv_folds = cv_folds
        self.meta_C = meta_C
        self._fitted_bases: dict[str, object] = {}
        self._meta: LogisticRegression | None = None
        self._base_names = list(base_pipelines.keys())

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "StackingEnsemble":
        cv = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=settings.model_random_state,
        )
        n = len(X)
        oof_preds = np.zeros((n, len(self._base_names)))

        log.info("Stacking: generating out-of-fold predictions (%d base learners, %d folds)...",
                 len(self._base_names), self.cv_folds)

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr        = y.iloc[train_idx]

            for col_idx, (name, pipeline) in enumerate(self.base_pipelines.items()):
                import copy
                fold_pipe = copy.deepcopy(pipeline)
                fold_pipe.fit(X_tr, y_tr)
                oof_preds[val_idx, col_idx] = fold_pipe.predict_proba(X_val)[:, 1]

            log.debug("Stacking fold %d/%d complete", fold_idx + 1, self.cv_folds)

        # Train meta-learner on out-of-fold predictions
        self._meta = LogisticRegression(
            C=self.meta_C,
            max_iter=1000,
            random_state=settings.model_random_state,
        )
        self._meta.fit(oof_preds, y)
        log.info("Meta-learner weights: %s",
                 dict(zip(self._base_names, self._meta.coef_[0].round(4))))

        # Refit all base learners on full training data
        for name, pipeline in self.base_pipelines.items():
            pipeline.fit(X, y)
            self._fitted_bases[name] = pipeline

        log.info("Stacking ensemble fitted on %d samples.", n)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted_bases or self._meta is None:
            raise RuntimeError("StackingEnsemble not fitted. Call fit() first.")

        base_preds = np.column_stack([
            self._fitted_bases[name].predict_proba(X)[:, 1]
            for name in self._base_names
        ])
        probs_churn = self._meta.predict_proba(base_preds)[:, 1]
        return np.column_stack([1 - probs_churn, probs_churn])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    @property
    def meta_weights(self) -> dict[str, float]:
        if self._meta is None:
            return {}
        return dict(zip(self._base_names, self._meta.coef_[0].round(4)))
