"""Dataset loading and schema integrity validation for CreditGuard."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

from src.config import (
    ALL_EXPECTED_COLS,
    CATEGORICAL_COLS,
    NUMERICAL_BOUNDS,
    NUMERICAL_COLS,
    RAW_DATA_PATH,
    TARGET_COL,
    VALID_CATEGORIES,
)


@dataclass
class ValidationReport:
    """Structured report produced during data validation."""
    total_rows: int
    total_columns: int
    missing_columns: List[str]
    unexpected_columns: List[str]
    duplicate_rows_count: int
    null_counts: Dict[str, int]
    null_percentages: Dict[str, float]
    invalid_category_counts: Dict[str, int]
    out_of_bounds_counts: Dict[str, int]
    target_distribution: Dict[Union[int, str], int]
    is_valid: bool = True
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Return human-readable summary string of validation."""
        lines = [
            "=" * 60,
            "CREDITGUARD DATA VALIDATION REPORT",
            "=" * 60,
            f"Shape: {self.total_rows:,} rows x {self.total_columns} columns",
            f"Duplicate Rows: {self.duplicate_rows_count:,} ({self.duplicate_rows_count / max(self.total_rows, 1):.2%})",
            f"Schema Integrity: {'PASSED' if not self.missing_columns else 'FAILED'}",
        ]
        if self.missing_columns:
            lines.append(f"  Missing required columns: {self.missing_columns}")
        if self.unexpected_columns:
            lines.append(f"  Unexpected columns: {self.unexpected_columns}")

        lines.append("\nMissing Values:")
        has_nulls = False
        for col, count in self.null_counts.items():
            if count > 0:
                has_nulls = True
                pct = self.null_percentages[col]
                lines.append(f"  - {col}: {count:,} ({pct:.2%})")
        if not has_nulls:
            lines.append("  None detected.")

        lines.append("\nDomain Violations:")
        has_domain_issues = False
        for col, count in self.out_of_bounds_counts.items():
            if count > 0:
                has_domain_issues = True
                lines.append(f"  - {col} out-of-bounds: {count:,} records")
        for col, count in self.invalid_category_counts.items():
            if count > 0:
                has_domain_issues = True
                lines.append(f"  - {col} unrecognized categories: {count:,} records")
        if not has_domain_issues:
            lines.append("  None detected.")

        lines.append("\nTarget Distribution (loan_status):")
        for k, v in self.target_distribution.items():
            pct = v / max(self.total_rows, 1)
            lines.append(f"  - Class {k}: {v:,} ({pct:.2%})")

        if self.warnings:
            lines.append("\nWarnings:")
            for w in self.warnings:
                lines.append(f"  ! {w}")

        lines.append("=" * 60)
        return "\n".join(lines)


class CreditDataLoader:
    """Loads and validates credit risk raw dataset while maintaining data immutability."""

    def __init__(self, filepath: Optional[Union[str, Path]] = None):
        self.filepath = Path(filepath or RAW_DATA_PATH)

    def load_raw_data(self) -> pd.DataFrame:
        """Load raw CSV data without mutations.
        
        Returns:
            pd.DataFrame: A copy of raw dataset.
        """
        if not self.filepath.exists():
            raise FileNotFoundError(f"Raw data file not found at: {self.filepath}")

        # Read CSV directly without inplace modifications
        df = pd.read_csv(self.filepath)
        return df.copy()

    def validate(self, df: pd.DataFrame) -> Tuple[bool, ValidationReport]:
        """Perform comprehensive validation against credit risk domain rules.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, ValidationReport)
        """
        warnings: List[str] = []
        is_valid = True

        total_rows, total_cols = df.shape
        present_cols = set(df.columns)
        expected_cols = set(ALL_EXPECTED_COLS)

        missing_cols = list(expected_cols - present_cols)
        unexpected_cols = list(present_cols - expected_cols)

        if missing_cols:
            is_valid = False
            warnings.append(f"Missing mandatory columns: {missing_cols}")

        # Duplicates
        dup_count = int(df.duplicated().sum())
        if dup_count > 0:
            warnings.append(f"Detected {dup_count} duplicate rows in raw data.")

        # Null analysis
        null_counts = {col: int(df[col].isna().sum()) for col in df.columns}
        null_percentages = {col: float(df[col].isna().mean()) for col in df.columns}

        # Categorical domain checks
        invalid_cat_counts: Dict[str, int] = {}
        for cat_col, valid_vals in VALID_CATEGORIES.items():
            if cat_col in df.columns:
                series = df[cat_col].dropna().astype(str).str.strip().str.upper()
                invalid_mask = ~series.isin(valid_vals)
                inv_count = int(invalid_mask.sum())
                if inv_count > 0:
                    invalid_cat_counts[cat_col] = inv_count
                    warnings.append(f"{cat_col} contains {inv_count} invalid category entries.")

        # Numerical domain checks
        out_of_bounds_counts: Dict[str, int] = {}
        for num_col, bounds in NUMERICAL_BOUNDS.items():
            if num_col in df.columns:
                series = df[num_col].dropna()
                oob_mask = (series < bounds["min"]) | (series > bounds["max"])
                oob_count = int(oob_mask.sum())
                if oob_count > 0:
                    out_of_bounds_counts[num_col] = oob_count
                    warnings.append(
                        f"{num_col} contains {oob_count} values outside expected [{bounds['min']}, {bounds['max']}]."
                    )

        # Target distribution
        target_dist: Dict[Union[int, str], int] = {}
        if TARGET_COL in df.columns:
            target_series = df[TARGET_COL].value_counts(dropna=False)
            target_dist = {k: int(v) for k, v in target_series.items()}
            if set(target_dist.keys()) - {0, 1}:
                warnings.append(f"Target variable {TARGET_COL} contains unexpected values: {list(target_dist.keys())}")
        else:
            is_valid = False

        report = ValidationReport(
            total_rows=total_rows,
            total_columns=total_cols,
            missing_columns=missing_cols,
            unexpected_columns=unexpected_cols,
            duplicate_rows_count=dup_count,
            null_counts=null_counts,
            null_percentages=null_percentages,
            invalid_category_counts=invalid_cat_counts,
            out_of_bounds_counts=out_of_bounds_counts,
            target_distribution=target_dist,
            is_valid=is_valid,
            warnings=warnings,
        )

        return is_valid, report

