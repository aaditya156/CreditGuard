"""Statistical analysis engine for CreditGuard credit risk platform."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

from src.config import CATEGORICAL_COLS, NUMERICAL_COLS, TARGET_COL


@dataclass
class WOEResult:
    """Weight of Evidence (WoE) and Information Value (IV) summary for a feature."""
    feature: str
    iv_value: float
    predictive_power: str
    table: pd.DataFrame


class CreditRiskStatsAnalyzer:
    """Comprehensive statistical analyzer for credit risk modeling data."""

    def __init__(self, df: pd.DataFrame, target_col: str = TARGET_COL):
        self.df = df.copy()
        self.target_col = target_col
        if self.target_col not in self.df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in dataframe.")

    def get_summary_statistics(self, cols: Optional[List[str]] = None) -> pd.DataFrame:
        """Compute advanced descriptive statistics for numerical columns.
        
        Args:
            cols: List of column names (default: all numerical columns present in df)
            
        Returns:
            DataFrame with mean, std, median, IQR, skewness, kurtosis, min, max, quantiles.
        """
        target_cols = cols or [c for c in NUMERICAL_COLS if c in self.df.columns]
        stats_dict = {}

        for col in target_cols:
            series = self.df[col].dropna()
            q25 = float(series.quantile(0.25))
            q75 = float(series.quantile(0.75))
            iqr = q75 - q25

            stats_dict[col] = {
                "count": int(len(series)),
                "mean": float(series.mean()),
                "std": float(series.std()),
                "median": float(series.median()),
                "iqr": float(iqr),
                "skewness": float(series.skew()),
                "kurtosis": float(series.kurtosis()),
                "min": float(series.min()),
                "q25": q25,
                "q75": q75,
                "max": float(series.max()),
            }

        return pd.DataFrame.from_dict(stats_dict, orient="index")

    def get_target_distribution(self) -> Dict[str, Union[int, float]]:
        """Calculate overall class count, proportions, and default odds.
        
        Returns:
            Dictionary with counts, rates, and default odds.
        """
        counts = self.df[self.target_col].value_counts().to_dict()
        non_default = counts.get(0, 0)
        default = counts.get(1, 0)
        total = non_default + default

        default_rate = default / max(total, 1)
        odds = default / max(non_default, 1)

        return {
            "total_records": total,
            "non_default_count (0)": non_default,
            "default_count (1)": default,
            "default_rate": float(default_rate),
            "non_default_rate": float(non_default / max(total, 1)),
            "default_odds": float(odds),
        }

    def default_rate_by_feature(self, feature_col: str) -> pd.DataFrame:
        """Compute loan default distribution broken down by any discrete or categorical feature.
        
        Args:
            feature_col: Categorical or discrete column name.
            
        Returns:
            DataFrame containing counts, default counts, default rate, and percentage of total portfolio.
        """
        if feature_col not in self.df.columns:
            raise KeyError(f"Feature '{feature_col}' not found.")

        grouped = (
            self.df.groupby(feature_col, observed=False)[self.target_col]
            .agg(
                total_volume="count",
                default_count="sum",
                default_rate="mean",
            )
            .reset_index()
        )

        total_portfolio = len(self.df)
        grouped["portfolio_share"] = grouped["total_volume"] / total_portfolio
        grouped["non_default_count"] = grouped["total_volume"] - grouped["default_count"]
        grouped["default_share"] = grouped["default_count"] / max(grouped["default_count"].sum(), 1)

        # Sort by default rate descending
        grouped = grouped.sort_values(by="default_rate", ascending=False).reset_index(drop=True)
        return grouped

    def compute_correlations(self, method: str = "pearson") -> pd.DataFrame:
        """Compute correlation matrix for all numerical features against loan_status.
        
        Args:
            method: 'pearson' or 'spearman'
            
        Returns:
            DataFrame of correlations sorted by magnitude of correlation with target.
        """
        num_cols = [c for c in self.df.select_dtypes(include=[np.number]).columns]
        corr_matrix = self.df[num_cols].corr(method=method)
        target_corr = corr_matrix[[self.target_col]].sort_values(
            by=self.target_col, ascending=False
        )
        return target_corr

    def compute_woe_iv(
        self, feature_col: str, num_bins: int = 5, is_categorical: Optional[bool] = None
    ) -> WOEResult:
        """Calculate Weight of Evidence (WoE) and Information Value (IV) for credit risk feature.
        
        WoE = ln( (% of non-defaults in bin) / (% of defaults in bin) )
        IV = sum( ((% non-default) - (% default)) * WoE )
        
        Rule of thumb for IV:
          < 0.02: Unpredictive
          0.02 to 0.1: Weak predictor
          0.1 to 0.3: Medium predictor
          0.3 to 0.5: Strong predictor
          > 0.5: Suspiciously high / potential leakage
          
        Args:
            feature_col: Feature column name.
            num_bins: Number of bins for continuous features.
            is_categorical: Boolean flag, auto-detected if None.
            
        Returns:
            WOEResult dataclass containing IV value and per-bin WoE table.
        """
        temp_df = self.df[[feature_col, self.target_col]].dropna().copy()
        total_good = (temp_df[self.target_col] == 0).sum()
        total_bad = (temp_df[self.target_col] == 1).sum()

        if total_good == 0 or total_bad == 0:
            raise ValueError("Target must have both 0 and 1 classes.")

        if is_categorical is None:
            is_categorical = (
                temp_df[feature_col].dtype == "object"
                or temp_df[feature_col].dtype.name == "category"
                or temp_df[feature_col].dtype.name == "str"
                or temp_df[feature_col].nunique() <= 10
            )

        if not is_categorical:
            temp_df["bin"] = pd.qcut(temp_df[feature_col], q=num_bins, duplicates="drop").astype(str)
        else:
            temp_df["bin"] = temp_df[feature_col].astype(str)

        agg = (
            temp_df.groupby("bin", observed=False)[self.target_col]
            .agg(
                total="count",
                bad="sum",
            )
            .reset_index()
        )
        agg["good"] = agg["total"] - agg["bad"]

        # Prevent division by zero
        agg["good"] = agg["good"].replace(0, 0.5)
        agg["bad"] = agg["bad"].replace(0, 0.5)

        agg["dist_good"] = agg["good"] / total_good
        agg["dist_bad"] = agg["bad"] / total_bad

        agg["woe"] = np.log(agg["dist_good"] / agg["dist_bad"])
        agg["iv_bin"] = (agg["dist_good"] - agg["dist_bad"]) * agg["woe"]

        total_iv = float(agg["iv_bin"].sum())

        if total_iv < 0.02:
            power = "Unpredictive (<0.02)"
        elif total_iv < 0.1:
            power = "Weak (0.02 - 0.1)"
        elif total_iv < 0.3:
            power = "Medium (0.1 - 0.3)"
        elif total_iv < 0.5:
            power = "Strong (0.3 - 0.5)"
        else:
            power = "Suspiciously High (>0.5)"

        return WOEResult(
            feature=feature_col,
            iv_value=total_iv,
            predictive_power=power,
            table=agg,
        )

    def rank_all_features_by_iv(
        self, candidate_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Compute Information Value for all feature candidates and rank them.
        
        Args:
            candidate_cols: List of features to score.
            
        Returns:
            DataFrame ranked by IV descending.
        """
        cols = candidate_cols or [
            c for c in self.df.columns if c != self.target_col and not c.endswith("_isna")
        ]
        results = []

        for col in cols:
            try:
                res = self.compute_woe_iv(col)
                results.append(
                    {
                        "feature": col,
                        "information_value": round(res.iv_value, 4),
                        "predictive_power": res.predictive_power,
                    }
                )
            except Exception:
                continue

        return pd.DataFrame(results).sort_values(by="information_value", ascending=False).reset_index(drop=True)

