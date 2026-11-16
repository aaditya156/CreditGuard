"""Unit tests for CreditDataLoader and schema verification."""

import pytest
import pandas as pd
from pathlib import Path
from src.config import RAW_DATA_PATH, TARGET_COL, ALL_EXPECTED_COLS
from src.ingestion.loader import CreditDataLoader, ValidationReport


@pytest.fixture
def loader():
    return CreditDataLoader()


def test_raw_data_file_exists():
    assert RAW_DATA_PATH.exists(), f"Raw dataset not found at {RAW_DATA_PATH}"


def test_loader_loads_dataframe(loader):
    df = loader.load_raw_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert len(df) > 30000


def test_loader_preserves_expected_schema(loader):
    df = loader.load_raw_data()
    for col in ALL_EXPECTED_COLS:
        assert col in df.columns, f"Missing required column: {col}"
    assert TARGET_COL in df.columns


def test_loader_validate_report(loader):
    df = loader.load_raw_data()
    is_valid, report = loader.validate(df)
    assert isinstance(report, ValidationReport)
    assert report.total_rows == len(df)
    assert report.duplicate_rows_count > 0
    assert "person_emp_length" in report.null_counts
    assert "loan_int_rate" in report.null_counts
    assert 0 in report.target_distribution
    assert 1 in report.target_distribution

