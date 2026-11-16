"""Unit tests for CreditDataCleaner, missing-value imputation and outlier treatment."""

import pytest
import pandas as pd
from src.ingestion.loader import CreditDataLoader
from src.preprocessing.cleaner import CreditDataCleaner, CleaningSummary
from src.config import MAX_REASONABLE_AGE, MAX_REASONABLE_EMP_LENGTH


@pytest.fixture
def raw_df():
    loader = CreditDataLoader()
    return loader.load_raw_data()


@pytest.fixture
def cleaner():
    return CreditDataCleaner()


def test_cleaner_removes_duplicates(raw_df, cleaner):
    clean_df, summary = cleaner.clean_and_transform(raw_df, save_processed=False)
    assert summary.duplicates_removed > 0
    assert clean_df.duplicated().sum() == 0


def test_cleaner_removes_age_and_emp_outliers(raw_df, cleaner):
    clean_df, summary = cleaner.clean_and_transform(raw_df, save_processed=False)
    assert (clean_df["person_age"] > MAX_REASONABLE_AGE).sum() == 0
    assert (clean_df["person_emp_length"] > MAX_REASONABLE_EMP_LENGTH).sum() == 0
    # Physical impossibility check: emp_length <= age - 14
    impossible = clean_df["person_emp_length"] > (clean_df["person_age"] - 14)
    assert impossible.sum() == 0


def test_cleaner_imputes_all_missing_values(raw_df, cleaner):
    clean_df, summary = cleaner.clean_and_transform(raw_df, save_processed=False)
    assert clean_df["loan_int_rate"].isna().sum() == 0
    assert clean_df["person_emp_length"].isna().sum() == 0
    assert clean_df.isna().sum().sum() == 0


def test_cleaner_engineers_expected_features(raw_df, cleaner):
    clean_df, summary = cleaner.clean_and_transform(raw_df, save_processed=False)
    expected_features = [
        "age_group",
        "income_group",
        "disposable_income",
        "high_risk_profile",
        "person_emp_length_isna",
        "loan_int_rate_isna",
    ]
    for feat in expected_features:
        assert feat in clean_df.columns, f"Engineered feature {feat} missing."

