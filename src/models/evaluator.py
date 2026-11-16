"""Comprehensive model evaluation, curves, probability calibration, and threshold optimization."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import DEFAULT_COST_FN, DEFAULT_COST_FP


@dataclass
class ThresholdOptimizationResult:
    """Detailed summary of cost-optimal classification threshold search."""
    optimal_threshold: float
    min_total_cost: float
    cost_at_default_threshold: float
    cost_savings: float
    cost_savings_pct: float
    default_metrics: Dict[str, float]
    optimal_metrics: Dict[str, float]
    threshold_search_table: pd.DataFrame


@dataclass
class EvaluationMetrics:
    """Core evaluation metrics for credit risk classification."""
    model_name: str
    threshold: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    pr_auc: float
    brier_score: float
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int
    total_cost: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ModelEvaluator:
    """Evaluates credit risk classifiers with financial cost awareness and calibration inspection."""

    def __init__(
        self,
        cost_fn: float = DEFAULT_COST_FN,
        cost_fp: float = DEFAULT_COST_FP,
    ):
        self.cost_fn = cost_fn
        self.cost_fp = cost_fp

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: float = 0.50,
        model_name: str = "Model",
    ) -> EvaluationMetrics:
        """Compute complete suite of classification and financial metrics at a specified threshold.
        
        Args:
            y_true: True binary target array.
            y_prob: Predicted probabilities for the positive class (Default / 1).
            threshold: Decision threshold for positive prediction.
            model_name: Label for reporting.
            
        Returns:
            EvaluationMetrics dataclass instance.
        """
        y_true = np.asarray(y_true).astype(int)
        y_prob = np.asarray(y_prob)
        y_pred = (y_prob >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        roc_auc = float(roc_auc_score(y_true, y_prob))
        pr_auc = float(average_precision_score(y_true, y_prob))
        brier = float(brier_score_loss(y_true, y_prob))

        total_cost = float(fn * self.cost_fn + fp * self.cost_fp)

        return EvaluationMetrics(
            model_name=model_name,
            threshold=round(float(threshold), 4),
            precision=round(prec, 4),
            recall=round(rec, 4),
            f1_score=round(f1, 4),
            roc_auc=round(roc_auc, 4),
            pr_auc=round(pr_auc, 4),
            brier_score=round(brier, 4),
            true_negatives=int(tn),
            false_positives=int(fp),
            false_negatives=int(fn),
            true_positives=int(tp),
            total_cost=round(total_cost, 2),
        )

    def optimize_threshold_for_cost(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        model_name: str = "Model",
        n_thresholds: int = 100,
    ) -> ThresholdOptimizationResult:
        """Find the decision threshold that minimizes expected financial cost.
        
        In credit risk, Cost = Cost_FN * False_Negatives + Cost_FP * False_Positives.
        Since approving a defaulting borrower is far costlier than turning away a good applicant,
        the optimal operating threshold is typically lower than 0.50.
        
        Args:
            y_true: Ground truth binary targets.
            y_prob: Predicted probabilities for Default.
            model_name: Model identifier.
            n_thresholds: Number of threshold evaluation points.
            
        Returns:
            ThresholdOptimizationResult with optimal threshold, savings, and comparison table.
        """
        thresholds = np.linspace(0.01, 0.99, n_thresholds)
        results = []

        for th in thresholds:
            y_pred = (y_prob >= th).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            cost = fn * self.cost_fn + fp * self.cost_fp
            f1 = f1_score(y_true, y_pred, zero_division=0)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            results.append(
                {
                    "threshold": th,
                    "cost": cost,
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                    "tp": tp,
                }
            )

        res_df = pd.DataFrame(results)
        best_idx = res_df["cost"].idxmin()
        best_row = res_df.loc[best_idx]
        optimal_th = float(best_row["threshold"])
        min_cost = float(best_row["cost"])

        # Cost at default threshold 0.50
        def_metrics = self.compute_metrics(y_true, y_prob, threshold=0.50, model_name=model_name)
        opt_metrics = self.compute_metrics(y_true, y_prob, threshold=optimal_th, model_name=model_name)

        cost_at_default = def_metrics.total_cost
        cost_savings = cost_at_default - min_cost
        cost_savings_pct = (cost_savings / max(cost_at_default, 1e-6)) * 100.0

        return ThresholdOptimizationResult(
            optimal_threshold=round(optimal_th, 4),
            min_total_cost=round(min_cost, 2),
            cost_at_default_threshold=round(cost_at_default, 2),
            cost_savings=round(cost_savings, 2),
            cost_savings_pct=round(cost_savings_pct, 2),
            default_metrics=def_metrics.to_dict(),
            optimal_metrics=opt_metrics.to_dict(),
            threshold_search_table=res_df,
        )

    def get_roc_curve_data(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
        """Compute ROC curve false positive rate, true positive rate, and thresholds."""
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        return {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
            "roc_auc": float(roc_auc_score(y_true, y_prob)),
        }

    def get_pr_curve_data(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
        """Compute Precision-Recall curve data and PR-AUC."""
        prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
        return {
            "precision": prec.tolist(),
            "recall": rec.tolist(),
            "thresholds": thresholds.tolist(),
            "pr_auc": float(average_precision_score(y_true, y_prob)),
        }

    def get_calibration_data(
        self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
    ) -> Dict[str, Any]:
        """Compute reliability diagram / calibration curve data."""
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
        return {
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist(),
            "brier_score": float(brier_score_loss(y_true, y_prob)),
        }

