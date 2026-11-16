"""Financial behavior, debt exposure, and portfolio risk concentration analysis."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.config import TARGET_COL


@dataclass
class PortfolioExposureSummary:
    """Summary of financial exposure across borrower risk segments."""
    total_originated_volume_usd: float
    defaulted_volume_usd: float
    non_defaulted_volume_usd: float
    portfolio_default_rate_by_volume: float
    avg_loan_size: float
    avg_default_loan_size: float
    avg_non_default_loan_size: float

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "CREDITGUARD PORTFOLIO EXPOSURE SUMMARY",
            "=" * 60,
            f"Total Originated Portfolio: ${self.total_originated_volume_usd:,.2f}",
            f"Defaulted Principal at Risk: ${self.defaulted_volume_usd:,.2f} ({self.portfolio_default_rate_by_volume:.2%})",
            f"Performing Loan Volume: ${self.non_defaulted_volume_usd:,.2f}",
            f"Average Overall Loan Amount: ${self.avg_loan_size:,.2f}",
            f"Average Defaulted Loan Amount: ${self.avg_default_loan_size:,.2f}",
            f"Average Performing Loan Amount: ${self.avg_non_default_loan_size:,.2f}",
            "=" * 60,
        ]
        return "\n".join(lines)


class FinancialBehaviorAnalyzer:
    """Analyzes financial behavior, debt-to-income stress, and credit loss exposures."""

    def __init__(self, df: pd.DataFrame, target_col: str = TARGET_COL):
        self.df = df.copy()
        self.target_col = target_col

    def portfolio_exposure(self) -> PortfolioExposureSummary:
        """Compute total dollar volume at risk vs performing volume."""
        total_vol = float(self.df["loan_amnt"].sum())
        default_mask = self.df[self.target_col] == 1

        default_vol = float(self.df.loc[default_mask, "loan_amnt"].sum())
        non_default_vol = float(self.df.loc[~default_mask, "loan_amnt"].sum())

        vol_default_rate = default_vol / max(total_vol, 1.0)

        avg_total = float(self.df["loan_amnt"].mean())
        avg_default = float(self.df.loc[default_mask, "loan_amnt"].mean())
        avg_non_default = float(self.df.loc[~default_mask, "loan_amnt"].mean())

        return PortfolioExposureSummary(
            total_originated_volume_usd=total_vol,
            defaulted_volume_usd=default_vol,
            non_defaulted_volume_usd=non_default_vol,
            portfolio_default_rate_by_volume=vol_default_rate,
            avg_loan_size=avg_total,
            avg_default_loan_size=avg_default,
            avg_non_default_loan_size=avg_non_default,
        )

    def debt_burden_analysis(self, bins: Optional[List[float]] = None) -> pd.DataFrame:
        """Segment borrowers by loan-to-income ratio (debt burden) and compute default rate.
        
        Args:
            bins: Threshold list for loan_percent_income.
            
        Returns:
            DataFrame with loan volume, default rate, and dollar exposure per debt bracket.
        """
        cut_bins = bins or [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 1.0]
        labels = ["<10%", "10-20%", "20-30%", "30-40%", "40-50%", "50%+"]

        temp = self.df.copy()
        temp["burden_bracket"] = pd.cut(
            temp["loan_percent_income"],
            bins=cut_bins,
            labels=labels,
            include_lowest=True,
        )

        agg = (
            temp.groupby("burden_bracket", observed=False)
            .agg(
                borrower_count=("loan_amnt", "count"),
                total_loan_volume=("loan_amnt", "sum"),
                default_count=(self.target_col, "sum"),
                default_rate=(self.target_col, "mean"),
                avg_int_rate=("loan_int_rate", "mean"),
                avg_income=("person_income", "mean"),
            )
            .reset_index()
        )
        agg["portfolio_share"] = agg["borrower_count"] / len(temp)
        return agg

    def cross_tabulate_risk(
        self, row_feature: str, col_feature: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Produce two 2D contingency tables: Default Rate (%) and Total Loan Amount ($).
        
        Args:
            row_feature: e.g. 'loan_grade'
            col_feature: e.g. 'loan_intent' or 'person_home_ownership'
            
        Returns:
            Tuple of (default_rate_pivot, total_volume_pivot)
        """
        rate_pivot = pd.pivot_table(
            self.df,
            values=self.target_col,
            index=row_feature,
            columns=col_feature,
            aggfunc="mean",
            observed=False,
        )

        volume_pivot = pd.pivot_table(
            self.df,
            values="loan_amnt",
            index=row_feature,
            columns=col_feature,
            aggfunc="sum",
            observed=False,
        )

        return rate_pivot, volume_pivot

    def identify_extreme_risk_segments(
        self,
        min_default_rate: float = 0.50,
        min_subpopulation_size: int = 50,
    ) -> pd.DataFrame:
        """Find subsegments where default rate exceeds high risk threshold (e.g. 50%+).
        
        Args:
            min_default_rate: Minimum default threshold to flag.
            min_subpopulation_size: Minimum sample size to ensure statistical significance.
            
        Returns:
            DataFrame of highest-risk segments.
        """
        temp = self.df.copy()
        grouped = (
            temp.groupby(
                ["loan_grade", "cb_person_default_on_file", "person_home_ownership"],
                observed=False,
            )
            .agg(
                count=(self.target_col, "count"),
                default_count=(self.target_col, "sum"),
                default_rate=(self.target_col, "mean"),
                avg_loan_amnt=("loan_amnt", "mean"),
            )
            .reset_index()
        )

        filtered = grouped[
            (grouped["count"] >= min_subpopulation_size)
            & (grouped["default_rate"] >= min_default_rate)
        ].sort_values(by="default_rate", ascending=False).reset_index(drop=True)

        return filtered

