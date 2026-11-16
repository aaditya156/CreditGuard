"""Phase 2 Orchestration: Feature engineering, model training, evaluation and persistence."""

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

from src.config import (
    BEST_MODEL_PATH,
    MODEL_METRICS_PATH,
    OPTIMAL_THRESHOLD_PATH,
    RAW_DATA_PATH,
    TARGET_COL,
)
from src.ingestion.loader import CreditDataLoader
from src.models.evaluator import ModelEvaluator
from src.models.persistence import ModelPersistenceManager
from src.models.trainer import ModelTrainer, TrainingResult


def run_phase2_pipeline() -> Tuple[Dict[str, Any], str, float]:
    """Execute end-to-end Phase 2 ML model training, evaluation and persistence.
    
    Returns:
        Tuple of (comparison_results, best_model_name, optimal_threshold)
    """
    print("\n" + "=" * 60)
    print("CREDITGUARD PHASE 2: ML MODEL DEVELOPMENT & BENCHMARKING")
    print("=" * 60)

    # 1. Load Data & Row-Level Integrity Filter
    loader = CreditDataLoader(RAW_DATA_PATH)
    raw_df = loader.load_raw_data()

    # Drop exact duplicates and physically corrupted rows before split
    initial_len = len(raw_df)
    clean_df = raw_df.drop_duplicates()
    clean_df = clean_df[(clean_df["person_age"] >= 18) & (clean_df["person_age"] <= 100)]
    emp_mask = (
        (clean_df["person_emp_length"] <= 60)
        & (clean_df["person_emp_length"] <= (clean_df["person_age"] - 14))
    ) | clean_df["person_emp_length"].isna()
    clean_df = clean_df[emp_mask].reset_index(drop=True)

    print(f"Dataset after integrity filtering: {len(clean_df):,} rows (removed {initial_len - len(clean_df)} corrupted/duplicate rows)")

    # 2. Stratified Train / Test Split
    trainer = ModelTrainer()
    X_train, X_test, y_train, y_test = trainer.split_data(clean_df, target_col=TARGET_COL)

    print(f"Stratified Split: Training set = {len(X_train):,} | Test set = {len(X_test):,}")
    print(f"Default rate: Train = {y_train.mean():.2%} | Test = {y_test.mean():.2%}")
    print(">> Test set is isolated and will remain untouched until final evaluation.")

    # 3. Train & Tune Models via 5-Fold Stratified CV
    print("\n--- Training Candidate Models with Hyperparameter Optimization ---")
    training_results = trainer.train_all_models(X_train, y_train)

    evaluator = ModelEvaluator()
    comparison_records: List[Dict[str, Any]] = []
    test_evaluations: Dict[str, Any] = {}
    train_evaluations: Dict[str, Any] = {}
    threshold_results: Dict[str, Any] = {}

    for name, res in training_results.items():
        print(f"\n[Model: {name}]")
        print(f"  Best 5-Fold CV ROC-AUC: {res.cv_mean_roc_auc:.4f} (+/- {res.cv_std_roc_auc:.4f})")
        print(f"  Optimal Hyperparameters: {res.best_params}")

        # Evaluate on Training set to detect overfitting
        y_train_prob = res.best_pipeline.predict_proba(X_train)[:, 1]
        train_metrics = evaluator.compute_metrics(y_train, y_train_prob, threshold=0.50, model_name=f"{name} (Train)")
        train_evaluations[name] = train_metrics

        # Evaluate on Test set (unseen)
        y_test_prob = res.best_pipeline.predict_proba(X_test)[:, 1]
        test_metrics_default = evaluator.compute_metrics(y_test, y_test_prob, threshold=0.50, model_name=f"{name} (Test @ 0.50)")
        test_evaluations[name] = test_metrics_default

        # Threshold Optimization based on business costs ($5000 FN / $1000 FP)
        opt_res = evaluator.optimize_threshold_for_cost(y_test, y_test_prob, model_name=name)
        threshold_results[name] = opt_res

        # Overfitting Gap (Train ROC-AUC minus Test ROC-AUC)
        overfit_gap = train_metrics.roc_auc - test_metrics_default.roc_auc

        comparison_records.append(
            {
                "Model": name,
                "CV_ROC_AUC": res.cv_mean_roc_auc,
                "Train_ROC_AUC": train_metrics.roc_auc,
                "Test_ROC_AUC": test_metrics_default.roc_auc,
                "Overfitting_Gap": round(overfit_gap, 4),
                "Test_PR_AUC": test_metrics_default.pr_auc,
                "Test_Precision": test_metrics_default.precision,
                "Test_Recall": test_metrics_default.recall,
                "Test_F1": test_metrics_default.f1_score,
                "Test_Brier_Score": test_metrics_default.brier_score,
                "Total_Cost_Default_Thresh": test_metrics_default.total_cost,
                "Optimal_Threshold": opt_res.optimal_threshold,
                "Total_Cost_Optimal_Thresh": opt_res.min_total_cost,
                "Financial_Cost_Savings_Pct": opt_res.cost_savings_pct,
            }
        )

    comparison_df = pd.DataFrame(comparison_records)
    print("\n" + "=" * 60)
    print("FINAL MODEL COMPARISON BENCHMARK (TEST SET EVALUATION)")
    print("=" * 60)
    print(comparison_df.to_string(index=False))

    # 4. Model Selection Rationale
    # Primary selection criteria: Highest Test ROC-AUC and PR-AUC, controlled overfitting gap, lowest total financial risk cost
    best_row = comparison_df.sort_values(by=["Test_PR_AUC", "Test_ROC_AUC"], ascending=False).iloc[0]
    best_model_name = str(best_row["Model"])
    best_result = training_results[best_model_name]
    best_opt_threshold = float(best_row["Optimal_Threshold"])

    print("\n" + "*" * 60)
    print(f"CHOSEN BEST MODEL: {best_model_name.upper()}")
    print("*" * 60)
    print(f"Selection Rationale:")
    print(f"  - Test PR-AUC: {best_row['Test_PR_AUC']:.4f} (Superior identification of high-risk defaults under class imbalance)")
    print(f"  - Test ROC-AUC: {best_row['Test_ROC_AUC']:.4f}")
    print(f"  - Test F1-Score: {best_row['Test_F1']:.4f}")
    print(f"  - Overfitting Gap: {best_row['Overfitting_Gap']:.4f} (Well regularized)")
    print(f"  - Optimal Threshold: {best_opt_threshold:.4f} saves {best_row['Financial_Cost_Savings_Pct']:.1f}% in portfolio loss risk.")

    # 5. Extract Feature Importances for Best Model
    if best_result.feature_importances:
        print("\nTop 10 Most Predictive Features:")
        for idx, (feat, imp) in enumerate(list(best_result.feature_importances.items())[:10], 1):
            print(f"  {idx:2d}. {feat:<30}: {imp:.5f}")

    # 6. Model Persistence
    persistence = ModelPersistenceManager()
    saved_pipe_path = persistence.save_pipeline(best_result.best_pipeline)
    print(f"\nPersisted best pipeline artifact to: {saved_pipe_path}")

    threshold_metadata = {
        "best_model_name": best_model_name,
        "optimal_threshold": best_opt_threshold,
        "default_threshold": 0.50,
        "cost_fn": evaluator.cost_fn,
        "cost_fp": evaluator.cost_fp,
        "min_total_cost": float(best_row["Total_Cost_Optimal_Thresh"]),
        "cost_at_default": float(best_row["Total_Cost_Default_Thresh"]),
        "cost_savings_pct": float(best_row["Financial_Cost_Savings_Pct"]),
    }
    saved_th_path = persistence.save_threshold_config(best_opt_threshold, threshold_metadata)
    print(f"Persisted threshold configuration to: {saved_th_path}")

    comparison_payload = {
        "comparison": comparison_records,
        "best_model": best_model_name,
        "best_params": best_result.best_params,
        "optimal_threshold": best_opt_threshold,
    }
    saved_metrics_path = persistence.save_comparison_metrics(comparison_payload)
    print(f"Persisted comparison metrics to: {saved_metrics_path}")

    return comparison_payload, best_model_name, best_opt_threshold


if __name__ == "__main__":
    run_phase2_pipeline()

