"""Leakage-safe model training, hyperparameter tuning, and cross-validation engine."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from src.config import (
    CV_FOLDS,
    RANDOM_SEED,
    TARGET_COL,
    TEST_SIZE,
)
from src.preprocessing.ml_pipeline import build_ml_pipeline


@dataclass
class TrainingResult:
    """Artifacts and metrics produced by training a single candidate model."""
    model_name: str
    best_pipeline: Any
    best_params: Dict[str, Any]
    cv_mean_roc_auc: float
    cv_std_roc_auc: float
    feature_importances: Optional[Dict[str, float]]
    search_object: Any


class ModelTrainer:
    """Orchestrates leakage-safe cross-validation and hyperparameter optimization."""

    def __init__(
        self,
        test_size: float = TEST_SIZE,
        cv_folds: int = CV_FOLDS,
        random_seed: int = RANDOM_SEED,
    ):
        self.test_size = test_size
        self.cv_folds = cv_folds
        self.random_seed = random_seed

    def split_data(
        self, df: pd.DataFrame, target_col: str = TARGET_COL
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Perform stratified train/test split.
        
        The test set is isolated here and must remain completely untouched
        throughout cross-validation and hyperparameter optimization.
        """
        X = df.drop(columns=[target_col]).copy()
        y = df[target_col].copy()

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            stratify=y,
            random_state=self.random_seed,
        )
        return X_train, X_test, y_train, y_test

    def get_candidate_models(self, y_train: pd.Series) -> Dict[str, Dict[str, Any]]:
        """Define candidate models, preprocessing configurations, and hyperparameter search grids.
        
        Args:
            y_train: Training labels used to compute exact class imbalance ratio.
        """
        neg_count = int((y_train == 0).sum())
        pos_count = int((y_train == 1).sum())
        scale_pos_weight = float(neg_count / max(pos_count, 1))

        # 1. Logistic Regression (Linear Baseline)
        lr_base = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=self.random_seed,
        )
        lr_params = {
            "classifier__C": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0],
            "classifier__solver": ["lbfgs"],
        }

        # 2. Random Forest
        rf_base = RandomForestClassifier(
            class_weight="balanced",
            random_state=self.random_seed,
            n_jobs=-1,
        )
        rf_params = {
            "classifier__n_estimators": [100, 150, 200, 250],
            "classifier__max_depth": [6, 8, 10, 12, 15, None],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4],
            "classifier__max_features": ["sqrt", "log2"],
        }

        # 3. XGBoost
        xgb_base = XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=self.random_seed,
            n_jobs=-1,
        )
        xgb_params = {
            "classifier__n_estimators": [100, 150, 200],
            "classifier__max_depth": [3, 4, 5, 6],
            "classifier__learning_rate": [0.02, 0.05, 0.1, 0.15],
            "classifier__subsample": [0.7, 0.8, 0.9, 1.0],
            "classifier__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            "classifier__min_child_weight": [1, 3, 5],
            "classifier__gamma": [0.0, 0.1, 0.2],
        }

        return {
            "Logistic Regression": {
                "model": lr_base,
                "params": lr_params,
                "scale_numeric": True,
                "n_iter": 9,
            },
            "Random Forest": {
                "model": rf_base,
                "params": rf_params,
                "scale_numeric": False,
                "n_iter": 15,
            },
            "XGBoost": {
                "model": xgb_base,
                "params": xgb_params,
                "scale_numeric": False,
                "n_iter": 15,
            },
        }

    def train_model(
        self,
        name: str,
        config: Dict[str, Any],
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> TrainingResult:
        """Run hyperparameter search with stratified cross-validation for a single model."""
        base_pipeline = build_ml_pipeline(
            model=config["model"], scale_numeric=config["scale_numeric"]
        )

        cv = StratifiedKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_seed
        )

        search = RandomizedSearchCV(
            estimator=base_pipeline,
            param_distributions=config["params"],
            n_iter=config["n_iter"],
            cv=cv,
            scoring="roc_auc",
            random_state=self.random_seed,
            n_jobs=-1,
            refit=True,
            verbose=0,
        )

        search.fit(X_train, y_train)

        best_pipeline = search.best_estimator_
        best_params = search.best_params_
        best_idx = search.best_index_
        mean_auc = float(search.cv_results_["mean_test_score"][best_idx])
        std_auc = float(search.cv_results_["std_test_score"][best_idx])

        # Extract feature importances / coefficients if accessible
        feature_importances = self._extract_feature_importances(best_pipeline)

        return TrainingResult(
            model_name=name,
            best_pipeline=best_pipeline,
            best_params=best_params,
            cv_mean_roc_auc=round(mean_auc, 4),
            cv_std_roc_auc=round(std_auc, 4),
            feature_importances=feature_importances,
            search_object=search,
        )

    def train_all_models(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> Dict[str, TrainingResult]:
        """Train and optimize all candidate models."""
        candidates = self.get_candidate_models(y_train)
        results: Dict[str, TrainingResult] = {}

        for name, config in candidates.items():
            result = self.train_model(name, config, X_train, y_train)
            results[name] = result

        return results

    def _extract_feature_importances(
        self, pipeline: Any
    ) -> Optional[Dict[str, float]]:
        """Extract feature names and their importance or coefficient magnitude."""
        try:
            preprocessor = pipeline.named_steps["preprocessor"]
            classifier = pipeline.named_steps["classifier"]

            feature_names = preprocessor.get_feature_names_out()
            # Clean feature names (remove num__ and cat__)
            clean_names = [f.replace("num__", "").replace("cat__", "") for f in feature_names]

            if hasattr(classifier, "feature_importances_"):
                importances = classifier.feature_importances_
            elif hasattr(classifier, "coef_"):
                importances = np.abs(classifier.coef_[0])
            else:
                return None

            imp_dict = {
                name: round(float(val), 5) for name, val in zip(clean_names, importances)
            }
            # Sort descending
            sorted_imp = dict(sorted(imp_dict.items(), key=lambda item: item[1], reverse=True))
            return sorted_imp
        except Exception:
            return None

