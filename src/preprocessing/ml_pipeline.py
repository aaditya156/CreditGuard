"""Leakage-safe feature engineering and scikit-learn preprocessing pipelines."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_COLS,
    NUMERICAL_COLS,
    TARGET_COL,
)


class CreditRiskFeatureEngineer(BaseEstimator, TransformerMixin):
    """Leakage-safe domain feature engineering transformer.
    
    All reference statistics (such as median interest rates by loan grade and
    median employment length by age group) are learned strictly during fit()
    on training data, preventing data leakage into validation/test sets.
    """

    def __init__(self, add_indicators: bool = True, assume_loan_term_months: int = 36):
        self.add_indicators = add_indicators
        self.assume_loan_term_months = assume_loan_term_months
        self.grade_rate_medians_: Dict[str, float] = {}
        self.global_rate_median_: float = 11.0
        self.age_emp_medians_: Dict[str, float] = {}
        self.global_emp_median_: float = 4.0

    def fit(self, X: pd.DataFrame, y=None):
        """Fit feature engineer strictly on training data."""
        df = X.copy()

        # Learn global and grade-level median interest rates
        if "loan_int_rate" in df.columns:
            self.global_rate_median_ = float(df["loan_int_rate"].median(skipna=True))
            if "loan_grade" in df.columns:
                self.grade_rate_medians_ = (
                    df.groupby("loan_grade", observed=False)["loan_int_rate"]
                    .median()
                    .to_dict()
                )

        # Learn global and age-bracket median employment length
        if "person_emp_length" in df.columns and "person_age" in df.columns:
            self.global_emp_median_ = float(df["person_emp_length"].median(skipna=True))
            age_bins = pd.cut(
                df["person_age"],
                bins=[0, 25, 35, 50, 150],
                labels=["18-25", "26-35", "36-50", "50+"],
            )
            self.age_emp_medians_ = (
                df.groupby(age_bins, observed=False)["person_emp_length"]
                .median()
                .to_dict()
            )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering transformations using fitted training statistics."""
        df = X.copy()

        # 1. Missing Value Indicators (if requested)
        if self.add_indicators:
            if "loan_int_rate" in df.columns:
                df["loan_int_rate_isna"] = df["loan_int_rate"].isna().astype(float)
            if "person_emp_length" in df.columns:
                df["person_emp_length_isna"] = df["person_emp_length"].isna().astype(float)

        # 2. Impute loan_int_rate using fitted grade-level medians
        if "loan_int_rate" in df.columns and "loan_grade" in df.columns:
            imputed_rate = df["loan_int_rate"].copy()
            for grade, median_val in self.grade_rate_medians_.items():
                mask = imputed_rate.isna() & (df["loan_grade"] == grade)
                imputed_rate[mask] = median_val
            imputed_rate = imputed_rate.fillna(self.global_rate_median_)
            df["loan_int_rate"] = imputed_rate

        # 3. Impute person_emp_length using fitted age-bracket medians
        if "person_emp_length" in df.columns and "person_age" in df.columns:
            age_bins = pd.cut(
                df["person_age"],
                bins=[0, 25, 35, 50, 150],
                labels=["18-25", "26-35", "36-50", "50+"],
            )
            imputed_emp = df["person_emp_length"].copy()
            for bracket, median_val in self.age_emp_medians_.items():
                mask = imputed_emp.isna() & (age_bins == bracket)
                imputed_emp[mask] = median_val
            imputed_emp = imputed_emp.fillna(self.global_emp_median_)
            df["person_emp_length"] = imputed_emp

        # 4. Corrected Financial Features
        # (a) Income-to-loan ratio (debt coverage proxy)
        if "person_income" in df.columns and "loan_amnt" in df.columns:
            df["income_to_loan_ratio"] = df["person_income"] / (df["loan_amnt"] + 1.0)

        # (b) Estimated Annual Loan Debt Service (Amortized Monthly Payment * 12)
        # Using standard annuity amortization formula: P * [r(1+r)^n] / [(1+r)^n - 1]
        if "loan_amnt" in df.columns and "loan_int_rate" in df.columns:
            n_months = self.assume_loan_term_months
            monthly_rate = (df["loan_int_rate"] / 100.0) / 12.0
            
            # Vectorized annuity payment
            pow_factor = np.power(1.0 + monthly_rate, n_months)
            denom = np.where(pow_factor == 1.0, 1.0, pow_factor - 1.0)
            monthly_pmt = np.where(
                monthly_rate <= 0,
                df["loan_amnt"] / n_months,
                df["loan_amnt"] * (monthly_rate * pow_factor) / denom,
            )
            df["estimated_annual_debt_service"] = monthly_pmt * 12.0

            # (c) Net Annual Income After Loan Debt Service
            # Replaces flawed "disposable_income = income - loan_amnt"
            if "person_income" in df.columns:
                df["net_annual_income_after_debt"] = (
                    df["person_income"] - df["estimated_annual_debt_service"]
                )

        # (d) Credit History Depth Relative to Age
        if "cb_person_cred_hist_length" in df.columns and "person_age" in df.columns:
            # Adult age proxy (minimum 18)
            adult_age = np.maximum(df["person_age"] - 18.0, 1.0)
            df["cred_hist_to_age_ratio"] = df["cb_person_cred_hist_length"] / adult_age

        # (e) Empirical 30% Debt Burden Risk Cliff Flag (Audited finding)
        if "loan_percent_income" in df.columns:
            df["high_debt_burden_flag"] = (df["loan_percent_income"] >= 0.30).astype(float)

        # (f) Interest Rate Spread against Grade Benchmark
        if "loan_int_rate" in df.columns and "loan_grade" in df.columns:
            grade_benchmarks = df["loan_grade"].map(self.grade_rate_medians_).fillna(self.global_rate_median_)
            df["interest_rate_grade_spread"] = df["loan_int_rate"] - grade_benchmarks

        return df


def get_feature_groups() -> Tuple[List[str], List[str]]:
    """Get the updated list of numerical and categorical feature names after feature engineering."""
    engineered_numerical = [
        "income_to_loan_ratio",
        "estimated_annual_debt_service",
        "net_annual_income_after_debt",
        "cred_hist_to_age_ratio",
        "high_debt_burden_flag",
        "interest_rate_grade_spread",
        "loan_int_rate_isna",
        "person_emp_length_isna",
    ]
    all_numeric = NUMERICAL_COLS + engineered_numerical
    all_categorical = CATEGORICAL_COLS
    return all_numeric, all_categorical


def build_column_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    """Construct an sklearn ColumnTransformer for numerical and categorical features.
    
    Args:
        scale_numeric: Whether to apply StandardScaler to numerical features.
        
    Returns:
        Configured ColumnTransformer.
    """
    all_numeric, all_categorical = get_feature_groups()

    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))

    num_transformer = Pipeline(steps=num_steps)

    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, all_numeric),
            ("cat", cat_transformer, all_categorical),
        ],
        remainder="drop",
    )
    return preprocessor


def build_ml_pipeline(model: BaseEstimator, scale_numeric: bool = True) -> Pipeline:
    """Build a complete, leakage-safe pipeline combining feature engineering,
    column transformation, and the downstream estimator.
    
    Args:
        model: Estimator to train.
        scale_numeric: Whether to scale numeric features (True for LogisticRegression).
        
    Returns:
        Full sklearn Pipeline.
    """
    return Pipeline(
        steps=[
            ("feature_engineer", CreditRiskFeatureEngineer()),
            ("preprocessor", build_column_preprocessor(scale_numeric=scale_numeric)),
            ("classifier", model),
        ]
    )

