"""Central configuration for CreditGuard Credit Risk Platform."""

from pathlib import Path
from typing import Dict, List, Set

# Base Directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "credit_risk_dataset.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "cleaned_credit_risk_dataset.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Target Variable
TARGET_COL = "loan_status"
TARGET_LABELS = {0: "Non-Default", 1: "Default"}

# Column Groups
NUMERICAL_COLS: List[str] = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]

CATEGORICAL_COLS: List[str] = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
]

ALL_FEATURE_COLS = NUMERICAL_COLS + CATEGORICAL_COLS
ALL_EXPECTED_COLS = ALL_FEATURE_COLS + [TARGET_COL]

# Domain Validation Rules
VALID_CATEGORIES: Dict[str, Set[str]] = {
    "person_home_ownership": {"RENT", "MORTGAGE", "OWN", "OTHER"},
    "loan_intent": {
        "PERSONAL",
        "EDUCATION",
        "MEDICAL",
        "VENTURE",
        "HOMEIMPROVEMENT",
        "DEBTCONSOLIDATION",
    },
    "loan_grade": {"A", "B", "C", "D", "E", "F", "G"},
    "cb_person_default_on_file": {"Y", "N"},
}

NUMERICAL_BOUNDS: Dict[str, Dict[str, float]] = {
    "person_age": {"min": 18, "max": 100},
    "person_income": {"min": 0, "max": 10_000_000},
    "person_emp_length": {"min": 0, "max": 60},
    "loan_amnt": {"min": 500, "max": 100_000},
    "loan_int_rate": {"min": 3.0, "max": 35.0},
    "loan_percent_income": {"min": 0.0, "max": 1.0},
    "cb_person_cred_hist_length": {"min": 0, "max": 60},
}

# Outlier & Cleaning Settings
MAX_REASONABLE_AGE = 100
MAX_REASONABLE_EMP_LENGTH = 60
MIN_WORKING_AGE = 14  # employment length cannot exceed (age - MIN_WORKING_AGE)

# Machine Learning & Modeling Hyperparameters
RANDOM_SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# Business Cost Matrix (USD Estimates for Expected Cost Optimization)
# False Negative: Approving a borrower who defaults (loss of unrecovered principal)
DEFAULT_COST_FN = 5000.0
# False Positive: Declining a creditworthy borrower (loss of interest income / relationship margin)
DEFAULT_COST_FP = 1000.0

# Model Artifact Paths
BEST_MODEL_PATH = MODELS_DIR / "best_creditguard_pipeline.joblib"
MODEL_METRICS_PATH = MODELS_DIR / "model_comparison_metrics.json"
OPTIMAL_THRESHOLD_PATH = MODELS_DIR / "optimal_threshold.json"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
