"""Unit tests for statistical and financial risk analyzers."""

import pytest
import pandas as pd
from src.ingestion.loader import CreditDataLoader
from src.preprocessing.cleaner import CreditDataCleaner
from src.analysis.stats import CreditRiskStatsAnalyzer, WOEResult
from src.analysis.financial import FinancialBehaviorAnalyzer, PortfolioExposureSummary


@pytest.fixture(scope="module")
def clean_data():
    loader = CreditDataLoader()
    df = loader.load_raw_data()
    cleaner = CreditDataCleaner()
    cleaned_df, _ = cleaner.clean_and_transform(df, save_processed=False)
    return cleaned_df


def test_target_distribution(clean_data):
    stats_engine = CreditRiskStatsAnalyzer(clean_data)
    dist = stats_engine.get_target_distribution()
    assert dist["total_records"] == len(clean_data)
    assert 0.20 < dist["default_rate"] < 0.25
    assert dist["default_odds"] > 0


def test_default_rate_by_feature(clean_data):
    stats_engine = CreditRiskStatsAnalyzer(clean_data)
    grade_df = stats_engine.default_rate_by_feature("loan_grade")
    assert "default_rate" in grade_df.columns
    assert len(grade_df) == 7  # Grades A through G
    assert grade_df["portfolio_share"].sum() == pytest.approx(1.0, 0.01)


def test_woe_iv_calculation(clean_data):
    stats_engine = CreditRiskStatsAnalyzer(clean_data)
    woe_res = stats_engine.compute_woe_iv("loan_grade")
    assert isinstance(woe_res, WOEResult)
    assert woe_res.iv_value > 0.5  # High predictive power
    assert "woe" in woe_res.table.columns


def test_portfolio_exposure_calculation(clean_data):
    fin_engine = FinancialBehaviorAnalyzer(clean_data)
    exposure = fin_engine.portfolio_exposure()
    assert isinstance(exposure, PortfolioExposureSummary)
    assert exposure.total_originated_volume_usd > 0
    assert exposure.defaulted_volume_usd > 0
    assert exposure.non_defaulted_volume_usd > 0
    assert exposure.portfolio_default_rate_by_volume > 0.20


def test_debt_burden_analysis(clean_data):
    fin_engine = FinancialBehaviorAnalyzer(clean_data)
    debt_df = fin_engine.debt_burden_analysis()
    assert not debt_df.empty
    assert "default_rate" in debt_df.columns
    # Check that default rate strictly increases between <10% and >40%
    rate_low = debt_df.loc[debt_df["burden_bracket"] == "<10%", "default_rate"].values[0]
    rate_high = debt_df.loc[debt_df["burden_bracket"] == "40-50%", "default_rate"].values[0]
    assert rate_high > rate_low

