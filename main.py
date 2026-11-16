"""CreditGuard - ML-powered Credit Risk Platform CLI Pipeline Entry Point."""

import argparse
import sys
from pathlib import Path

from src.analysis.financial import FinancialBehaviorAnalyzer
from src.analysis.stats import CreditRiskStatsAnalyzer
from src.config import (
    BEST_MODEL_PATH,
    MODEL_METRICS_PATH,
    OPTIMAL_THRESHOLD_PATH,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
)
from src.ingestion.loader import CreditDataLoader
from src.models.pipeline_runner import run_phase2_pipeline
from src.preprocessing.cleaner import CreditDataCleaner
from src.visualization.plots import CreditRiskVisualizer


def run_validation():
    print("\n>>> STEP 1: LOADING & VALIDATING RAW DATASET")
    loader = CreditDataLoader(RAW_DATA_PATH)
    raw_df = loader.load_raw_data()
    is_valid, report = loader.validate(raw_df)
    print(report.summary())
    return raw_df, is_valid


def run_cleaning(raw_df=None):
    print("\n>>> STEP 2: CLEANING, IMPUTATION & FEATURE ENGINEERING")
    if raw_df is None:
        loader = CreditDataLoader(RAW_DATA_PATH)
        raw_df = loader.load_raw_data()
    cleaner = CreditDataCleaner()
    clean_df, summary = cleaner.clean_and_transform(raw_df, save_processed=True)
    print(summary.summary())
    return clean_df


def run_analysis(clean_df=None):
    print("\n>>> STEP 3: STATISTICAL & FINANCIAL BEHAVIOR ANALYSIS")
    if clean_df is None:
        if not PROCESSED_DATA_PATH.exists():
            print("Cleaned data not found. Running cleaning first...")
            clean_df = run_cleaning()
        else:
            import pandas as pd
            clean_df = pd.read_csv(PROCESSED_DATA_PATH)

    stats_engine = CreditRiskStatsAnalyzer(clean_df)
    print("\n--- Target Distribution (loan_status) ---")
    for k, v in stats_engine.get_target_distribution().items():
        print(f"  {k}: {v}")

    print("\n--- Feature Information Value (IV) Ranking ---")
    iv_df = stats_engine.rank_all_features_by_iv()
    print(iv_df.to_string(index=False))

    fin_engine = FinancialBehaviorAnalyzer(clean_df)
    print("\n" + fin_engine.portfolio_exposure().summary())

    print("\n--- Debt-to-Income Risk Stress Analysis ---")
    debt_df = fin_engine.debt_burden_analysis()
    print(debt_df[["burden_bracket", "borrower_count", "default_rate", "total_loan_volume"]].to_string(index=False))

    return clean_df, iv_df


def run_visualizations(clean_df=None, iv_df=None):
    print("\n>>> STEP 4: GENERATING INTERACTIVE STATISTICAL VISUALIZATIONS")
    if clean_df is None:
        import pandas as pd
        clean_df = pd.read_csv(PROCESSED_DATA_PATH)
    if iv_df is None:
        stats_engine = CreditRiskStatsAnalyzer(clean_df)
        iv_df = stats_engine.rank_all_features_by_iv()

    viz = CreditRiskVisualizer(clean_df)
    viz.plot_target_distribution()
    viz.plot_loan_grade_default_analysis()
    viz.plot_risk_heatmap("loan_grade", "loan_intent")
    viz.plot_risk_heatmap("loan_grade", "person_home_ownership", save_name="risk_heatmap_homeownership")
    viz.plot_correlation_matrix()
    viz.plot_financial_behavior_scatter()
    viz.plot_iv_ranking(iv_df)
    viz.plot_distributions_by_target("loan_percent_income")
    viz.plot_distributions_by_target("loan_amnt")
    viz.plot_distributions_by_target("person_income")

    print(f"All interactive charts successfully generated in: {viz.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="CreditGuard Credit Risk Platform CLI")
    parser.add_argument("--validate", action="store_true", help="Run data validation on raw dataset")
    parser.add_argument("--clean", action="store_true", help="Clean dataset and impute missing values")
    parser.add_argument("--analyze", action="store_true", help="Run statistical and financial analysis")
    parser.add_argument("--visualize", action="store_true", help="Generate interactive Plotly visualizations")
    parser.add_argument("--phase1", action="store_true", help="Run complete Phase 1 foundation pipeline")
    parser.add_argument("--phase2", action="store_true", help="Run Phase 2 ML model training, tuning and evaluation")
    parser.add_argument("--all", action="store_true", help="Run both Phase 1 and Phase 2 pipelines end-to-end")

    args = parser.parse_args()

    # Default to Phase 1 + Phase 2 if no flag provided
    if not (args.validate or args.clean or args.analyze or args.visualize or args.phase1 or args.phase2 or args.all):
        args.phase2 = True

    if args.phase1 or args.all:
        raw_df, _ = run_validation()
        clean_df = run_cleaning(raw_df)
        clean_df, iv_df = run_analysis(clean_df)
        run_visualizations(clean_df, iv_df)

    if args.phase2 or args.all:
        run_phase2_pipeline()

    if args.validate and not args.phase1 and not args.all:
        run_validation()
    if args.clean and not args.phase1 and not args.all:
        run_cleaning()
    if args.analyze and not args.phase1 and not args.all:
        run_analysis()
    if args.visualize and not args.phase1 and not args.all:
        run_visualizations()


if __name__ == "__main__":
    main()
