"""Data cleaning, missing-value imputation, and outlier treatment for CreditGuard."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.config import (
    MAX_REASONABLE_AGE,
    MAX_REASONABLE_EMP_LENGTH,
    MIN_WORKING_AGE,
    PROCESSED_DATA_PATH,
    TARGET_COL,
)


@dataclass
class CleaningSummary:
    """Summary of operations performed during data cleaning."""
    initial_rows: int
    final_rows: int
    duplicates_removed: int
    age_outliers_removed: int
    emp_length_anomalies_removed: int
    emp_length_imputed_count: int
    int_rate_imputed_count: int
    features_engineered: List[str]

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "CREDITGUARD DATA CLEANING SUMMARY",
            "=" * 60,
            f"Initial Row Count: {self.initial_rows:,}",
            f"Duplicates Removed: {self.duplicates_removed:,}",
            f"Age Outliers Filtered (> {MAX_REASONABLE_AGE}): {self.age_outliers_removed:,}",
            f"Employment Length Anomalies Filtered: {self.emp_length_anomalies_removed:,}",
            f"Missing Values Imputed:",
            f"  - person_emp_length: {self.emp_length_imputed_count:,} (Group median by age brackets)",
            f"  - loan_int_rate: {self.int_rate_imputed_count:,} (Group median by loan grade)",
            f"Final Cleaned Dataset Shape: {self.final_rows:,} rows",
            f"Net Records Retained: {self.final_rows / max(self.initial_rows, 1):.2%}",
            f"Engineered Features: {', '.join(self.features_engineered)}",
            "=" * 60,
        ]
        return "\n".join(lines)


class CreditDataCleaner:
    """Cleans raw credit risk data, imputes missing values and engineers risk features."""

    def __init__(
        self,
        max_age: int = MAX_REASONABLE_AGE,
        max_emp_length: int = MAX_REASONABLE_EMP_LENGTH,
        min_working_age: int = MIN_WORKING_AGE,
        add_indicator_flags: bool = True,
    ):
        self.max_age = max_age
        self.max_emp_length = max_emp_length
        self.min_working_age = min_working_age
        self.add_indicator_flags = add_indicator_flags

    def fit_imputation_values(self, df: pd.DataFrame) -> Dict[str, Union[Dict, float]]:
        """Compute robust reference statistics for group-based imputation.
        
        Args:
            df: Cleaned dataframe without extreme outliers.
            
        Returns:
            Dictionary containing group imputation lookups and global fallbacks.
        """
        # Global fallback medians
        global_emp_median = float(df["person_emp_length"].median())
        global_rate_median = float(df["loan_int_rate"].median())

        # Group median interest rates by loan grade
        grade_rate_medians = (
            df.groupby("loan_grade", observed=False)["loan_int_rate"]
            .median()
            .to_dict()
        )

        # Group median employment lengths by age category
        # Create temporary age bins
        temp_age_bins = pd.cut(
            df["person_age"],
            bins=[0, 25, 35, 50, 150],
            labels=["18-25", "26-35", "36-50", "50+"],
        )
        age_emp_medians = (
            df.groupby(temp_age_bins, observed=False)["person_emp_length"]
            .median()
            .to_dict()
        )

        return {
            "global_emp_median": global_emp_median,
            "global_rate_median": global_rate_median,
            "grade_rate_medians": grade_rate_medians,
            "age_emp_medians": age_emp_medians,
        }

    def clean_and_transform(
        self,
        df: pd.DataFrame,
        save_processed: bool = True,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Tuple[pd.DataFrame, CleaningSummary]:
        """Execute the end-to-end cleaning pipeline without modifying input dataframe.
        
        Args:
            df: Input raw dataframe.
            save_processed: Whether to write clean dataset to disk.
            output_path: Destination path for cleaned CSV.
            
        Returns:
            Tuple of (cleaned_df, CleaningSummary)
        """
        initial_rows = len(df)
        working_df = df.copy()

        # 1. Remove duplicate records
        dups_mask = working_df.duplicated()
        duplicates_removed = int(dups_mask.sum())
        working_df = working_df.drop_duplicates().reset_index(drop=True)

        # 2. Filter unphysical / outlier records
        # Age outliers: e.g. age > 100 or age < 18
        age_outliers_mask = (working_df["person_age"] > self.max_age) | (working_df["person_age"] < 18)
        age_outliers_removed = int(age_outliers_mask.sum())
        working_df = working_df[~age_outliers_mask].copy()

        # Employment length anomalies:
        # e.g. emp_length > 60 OR emp_length > (person_age - 14)
        emp_anomaly_mask = (
            (working_df["person_emp_length"] > self.max_emp_length)
            | (working_df["person_emp_length"] > (working_df["person_age"] - self.min_working_age))
        ) & working_df["person_emp_length"].notna()
        emp_anomalies_removed = int(emp_anomaly_mask.sum())
        working_df = working_df[~emp_anomaly_mask].copy()

        # 3. Missing Value Imputation
        emp_missing_count = int(working_df["person_emp_length"].isna().sum())
        rate_missing_count = int(working_df["loan_int_rate"].isna().sum())

        if self.add_indicator_flags:
            working_df["person_emp_length_isna"] = working_df["person_emp_length"].isna().astype(int)
            working_df["loan_int_rate_isna"] = working_df["loan_int_rate"].isna().astype(int)

        imputation_stats = self.fit_imputation_values(working_df)

        # Impute loan_int_rate by loan_grade median
        grade_medians = imputation_stats["grade_rate_medians"]
        global_rate = imputation_stats["global_rate_median"]

        def impute_rate(row):
            if pd.isna(row["loan_int_rate"]):
                return grade_medians.get(row["loan_grade"], global_rate)
            return row["loan_int_rate"]

        working_df["loan_int_rate"] = working_df.apply(impute_rate, axis=1)

        # Impute person_emp_length by age bucket median
        age_bins = pd.cut(
            working_df["person_age"],
            bins=[0, 25, 35, 50, 150],
            labels=["18-25", "26-35", "36-50", "50+"],
        )
        age_medians = imputation_stats["age_emp_medians"]
        global_emp = imputation_stats["global_emp_median"]

        emp_imputed_series = working_df["person_emp_length"].copy()
        for label, median_val in age_medians.items():
            mask = working_df["person_emp_length"].isna() & (age_bins == label)
            emp_imputed_series[mask] = median_val
        emp_imputed_series = emp_imputed_series.fillna(global_emp)
        working_df["person_emp_length"] = emp_imputed_series

        # 4. Feature Enrichment / Domain Engineering
        engineered: List[str] = []

        # Age Groups
        working_df["age_group"] = pd.cut(
            working_df["person_age"],
            bins=[17, 25, 35, 50, 100],
            labels=["18-25", "26-35", "36-50", "50+"],
        ).astype(str)
        engineered.append("age_group")

        # Income Quartiles
        working_df["income_group"] = pd.qcut(
            working_df["person_income"],
            q=4,
            labels=["Low", "Lower-Middle", "Upper-Middle", "High"],
        ).astype(str)
        engineered.append("income_group")

        # Disposable Income Proxy (Annual income - Loan Amount)
        working_df["disposable_income"] = working_df["person_income"] - working_df["loan_amnt"]
        engineered.append("disposable_income")

        # High Risk Composite Flag: Prior default AND (Loan Grade >= D OR loan_percent_income >= 0.40)
        subprime_grades = {"D", "E", "F", "G"}
        working_df["high_risk_profile"] = (
            (working_df["cb_person_default_on_file"] == "Y")
            & (
                working_df["loan_grade"].isin(subprime_grades)
                | (working_df["loan_percent_income"] >= 0.40)
            )
        ).astype(int)
        engineered.append("high_risk_profile")

        if self.add_indicator_flags:
            engineered.extend(["person_emp_length_isna", "loan_int_rate_isna"])

        final_rows = len(working_df)

        # 5. Persist Cleaned Dataset
        dest_path = Path(output_path or PROCESSED_DATA_PATH)
        if save_processed:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            working_df.to_csv(dest_path, index=False)

        summary = CleaningSummary(
            initial_rows=initial_rows,
            final_rows=final_rows,
            duplicates_removed=duplicates_removed,
            age_outliers_removed=age_outliers_removed,
            emp_length_anomalies_removed=emp_anomalies_removed,
            emp_length_imputed_count=emp_missing_count,
            int_rate_imputed_count=rate_missing_count,
            features_engineered=engineered,
        )

        return working_df, summary

