"""Unit tests for leakage-safe ML preprocessing and feature engineering."""

import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import RAW_DATA_PATH, TARGET_COL
from src.ingestion.loader import CreditDataLoader
from src.preprocessing.ml_pipeline import (
    CreditRiskFeatureEngineer,
    build_column_preprocessor,
    build_ml_pipeline,
    get_feature_groups,
)


@pytest.fixture(scope="module")
def sample_data():
    loader = CreditDataLoader(RAW_DATA_PATH)
    df = loader.load_raw_data().drop_duplicates().head(500)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y


def test_feature_engineer_learns_only_on_fit(sample_data):
    X, _ = sample_data
    split_point = 250
    X_train, X_val = X.iloc[:split_point], X.iloc[split_point:]

    fe = CreditRiskFeatureEngineer()
    fe.fit(X_train)

    # Verify that reference statistics were learned from training data
    assert len(fe.grade_rate_medians_) > 0
    assert fe.global_rate_median_ > 0

    # Transform validation split
    X_val_trans = fe.transform(X_val)
    assert "income_to_loan_ratio" in X_val_trans.columns
    assert "estimated_annual_debt_service" in X_val_trans.columns
    assert "net_annual_income_after_debt" in X_val_trans.columns
    assert "cred_hist_to_age_ratio" in X_val_trans.columns
    assert "high_debt_burden_flag" in X_val_trans.columns
    assert "interest_rate_grade_spread" in X_val_trans.columns


def test_column_preprocessor_shape(sample_data):
    X, _ = sample_data
    fe = CreditRiskFeatureEngineer()
    X_eng = fe.fit_transform(X)

    preprocessor = build_column_preprocessor(scale_numeric=True)
    X_processed = preprocessor.fit_transform(X_eng)

    num_cols, cat_cols = get_feature_groups()
    assert X_processed.shape[0] == len(X)
    assert not np.isnan(X_processed).any()


def test_full_pipeline_fit_predict(sample_data):
    X, y = sample_data
    pipe = build_ml_pipeline(LogisticRegression(max_iter=200), scale_numeric=True)
    assert isinstance(pipe, Pipeline)

    pipe.fit(X, y)
    preds = pipe.predict(X)
    probs = pipe.predict_proba(X)

    assert len(preds) == len(X)
    assert probs.shape == (len(X), 2)
    assert np.all((probs >= 0.0) & (probs <= 1.0))

