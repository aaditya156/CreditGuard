"""Generate reproducible Jupyter notebooks for Phase 2."""

import json
from pathlib import Path


def create_notebook(cells, filepath):
    nb = {
        "cells": cells,
        "metadata": {
            "language_info": {"name": "python", "version": "3.14.7"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    target = Path(filepath)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Generated {filepath}")


def build_notebook_03():
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# CreditGuard: Phase 2 - Leakage-Safe Feature Engineering Pipeline\n",
                "\n",
                "This notebook establishes the production-grade, scikit-learn compatible feature engineering and preprocessing pipeline.\n",
                "\n",
                "### Core Architecture Principles:\n",
                "1. **Zero Data Leakage**: The dataset is partitioned into train/test splits before computing any reference statistics or transformations.\n",
                "2. **Corrected Cash Flow Metric**: Replaces the naive `disposable_income = income - loan_amnt` with `net_annual_income_after_debt`, which subtracts the annual amortized loan service from annual income.\n",
                "3. **Audited Debt Burden Feature**: Implements `high_debt_burden_flag` corresponding to the audited 30% debt-to-income default risk cliff.\n",
                "4. **Unified ColumnTransformer**: Packages median numerical imputation, scaling, and categorical one-hot encoding."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import sys\n",
                "from pathlib import Path\n",
                "project_root = Path.cwd().parent\n",
                "if str(project_root) not in sys.path:\n",
                "    sys.path.append(str(project_root))\n",
                "\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "from src.config import RAW_DATA_PATH, TARGET_COL\n",
                "from src.ingestion.loader import CreditDataLoader\n",
                "from src.models.trainer import ModelTrainer\n",
                "from src.preprocessing.ml_pipeline import (\n",
                "    CreditRiskFeatureEngineer,\n",
                "    build_column_preprocessor,\n",
                "    build_ml_pipeline,\n",
                "    get_feature_groups\n",
                ")\n",
                "\n",
                "print('Feature engineering pipeline modules successfully imported.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Stratified Train / Test Isolation\n",
                "We split the dataset into 80% train and 20% test before calculating any medians, scalers, or encodings."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "loader = CreditDataLoader(RAW_DATA_PATH)\n",
                "raw_df = loader.load_raw_data().drop_duplicates()\n",
                "clean_df = raw_df[(raw_df['person_age'] >= 18) & (raw_df['person_age'] <= 100)].copy()\n",
                "emp_mask = ((clean_df['person_emp_length'] <= 60) & (clean_df['person_emp_length'] <= (clean_df['person_age'] - 14))) | clean_df['person_emp_length'].isna()\n",
                "clean_df = clean_df[emp_mask].reset_index(drop=True)\n",
                "\n",
                "trainer = ModelTrainer()\n",
                "X_train, X_test, y_train, y_test = trainer.split_data(clean_df, target_col=TARGET_COL)\n",
                "print(f'Training split: {X_train.shape[0]:,} rows | Test split: {X_test.shape[0]:,} rows')\n",
                "print(f'Train default rate: {y_train.mean():.2%} | Test default rate: {y_test.mean():.2%}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Fitting CreditRiskFeatureEngineer on Training Data\n",
                "The transformer learns group-level imputation medians and calculates the corrected financial metrics."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fe = CreditRiskFeatureEngineer()\n",
                "fe.fit(X_train)\n",
                "print('Grade-level median interest rates learned exclusively from X_train:')\n",
                "for grade, median_rate in sorted(fe.grade_rate_medians_.items()):\n",
                "    print(f'  Grade {grade}: {median_rate:.2f}%')\n",
                "\n",
                "X_train_trans = fe.transform(X_train)\n",
                "X_test_trans = fe.transform(X_test)\n",
                "\n",
                "engineered_cols = [\n",
                "    'income_to_loan_ratio',\n",
                "    'estimated_annual_debt_service',\n",
                "    'net_annual_income_after_debt',\n",
                "    'high_debt_burden_flag',\n",
                "    'interest_rate_grade_spread'\n",
                "]\n",
                "X_train_trans[engineered_cols].head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Building ColumnTransformer\n",
                "Apply numerical median imputation and scaling alongside categorical one-hot encoding."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "preprocessor = build_column_preprocessor(scale_numeric=True)\n",
                "X_train_proc = preprocessor.fit_transform(X_train_trans)\n",
                "X_test_proc = preprocessor.transform(X_test_trans)\n",
                "\n",
                "print(f'Processed train feature matrix shape: {X_train_proc.shape}')\n",
                "print(f'Processed test feature matrix shape:  {X_test_proc.shape}')\n",
                "print(f'Nulls remaining in train matrix: {np.isnan(X_train_proc).sum()}')\n",
                "print(f'Nulls remaining in test matrix:  {np.isnan(X_test_proc).sum()}')"
            ]
        }
    ]
    create_notebook(cells, "notebooks/03_feature_engineering.ipynb")


def build_notebook_04():
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# CreditGuard: Phase 2 - Model Training & Hyperparameter Tuning\n",
                "\n",
                "This notebook covers the training, 5-fold stratified cross-validation, and hyperparameter optimization for three candidate models:\n",
                "1. **Logistic Regression (Baseline)**: L2 regularized, class-weighted, scaled inputs.\n",
                "2. **Random Forest**: Bagging ensemble, balanced subsample weighting.\n",
                "3. **XGBoost**: Gradient boosting machine with `scale_pos_weight` class penalty.\n",
                "\n",
                "### Validation Rules:\n",
                "- Stratified 5-Fold Cross Validation evaluated with `roc_auc`.\n",
                "- Complete training set is refit with the best hyperparameters.\n",
                "- The test set is held out completely."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import sys\n",
                "from pathlib import Path\n",
                "project_root = Path.cwd().parent\n",
                "if str(project_root) not in sys.path:\n",
                "    sys.path.append(str(project_root))\n",
                "\n",
                "import pandas as pd\n",
                "from src.config import RAW_DATA_PATH, TARGET_COL\n",
                "from src.ingestion.loader import CreditDataLoader\n",
                "from src.models.trainer import ModelTrainer\n",
                "\n",
                "loader = CreditDataLoader(RAW_DATA_PATH)\n",
                "raw_df = loader.load_raw_data().drop_duplicates()\n",
                "clean_df = raw_df[(raw_df['person_age'] >= 18) & (raw_df['person_age'] <= 100)].copy()\n",
                "emp_mask = ((clean_df['person_emp_length'] <= 60) & (clean_df['person_emp_length'] <= (clean_df['person_age'] - 14))) | clean_df['person_emp_length'].isna()\n",
                "clean_df = clean_df[emp_mask].reset_index(drop=True)\n",
                "\n",
                "trainer = ModelTrainer()\n",
                "X_train, X_test, y_train, y_test = trainer.split_data(clean_df, target_col=TARGET_COL)\n",
                "print(f'Training cohort: {X_train.shape[0]:,} loans | Baseline default rate: {y_train.mean():.2%}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Candidate Model Setup & Search Grids"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "candidates = trainer.get_candidate_models(y_train)\n",
                "for name, cfg in candidates.items():\n",
                "    model_cls = cfg['model'].__class__.__name__\n",
                "    print(f'Model: {name} ({model_cls}) | Scale Numerics: {cfg[\"scale_numeric\"]}')\n",
                "    print(f'Search Grid: {list(cfg[\"params\"].keys())}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Cross-Validation Results"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "training_results = trainer.train_all_models(X_train, y_train)\n",
                "for name, res in training_results.items():\n",
                "    print(f'[{name}]')\n",
                "    print(f'  5-Fold CV ROC-AUC: {res.cv_mean_roc_auc:.4f} (+/- {res.cv_std_roc_auc:.4f})')\n",
                "    print(f'  Optimal Params: {res.best_params}')"
            ]
        }
    ]
    create_notebook(cells, "notebooks/04_model_training.ipynb")


def build_notebook_05():
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# CreditGuard: Phase 2 - Model Evaluation, Calibration & Threshold Optimization\n",
                "\n",
                "This notebook unblinds the untouched test set (`X_test`, `y_test`) to compute out-of-sample metrics, evaluate calibration, optimize the decision threshold for financial loss, and select the production model."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import sys\n",
                "from pathlib import Path\n",
                "project_root = Path.cwd().parent\n",
                "if str(project_root) not in sys.path:\n",
                "    sys.path.append(str(project_root))\n",
                "\n",
                "import json\n",
                "import pandas as pd\n",
                "from src.config import BEST_MODEL_PATH, MODEL_METRICS_PATH, OPTIMAL_THRESHOLD_PATH\n",
                "from src.models.persistence import ModelPersistenceManager\n",
                "\n",
                "persistence = ModelPersistenceManager()\n",
                "best_pipe = persistence.load_pipeline()\n",
                "threshold_info = persistence.load_threshold_config()\n",
                "\n",
                "with open(MODEL_METRICS_PATH, 'r') as f:\n",
                "    metrics_data = json.load(f)\n",
                "\n",
                "comparison_df = pd.DataFrame(metrics_data['comparison'])\n",
                "print('Model Comparison Benchmark Table:')\n",
                "comparison_df[[\n",
                "    'Model', 'CV_ROC_AUC', 'Test_ROC_AUC', 'Test_PR_AUC',\n",
                "    'Test_Precision', 'Test_Recall', 'Test_F1', 'Total_Cost_Default_Thresh',\n",
                "    'Optimal_Threshold', 'Total_Cost_Optimal_Thresh', 'Financial_Cost_Savings_Pct'\n",
                "]]"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Cost-Optimal Decision Threshold Strategy\n",
                "Defaulting borrower loss ($5,000) vs rejected good applicant loss ($1,000)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(f'Champion Model: {metrics_data[\"best_model\"]}')\n",
                "print(f'Default Decision Threshold: 0.5000')\n",
                "print(f'Optimal Decision Threshold: {threshold_info[\"selected_threshold\"]:.4f}')\n",
                "print(f'Estimated Portfolio Risk Cost Reduction: {threshold_info[\"metadata\"][\"cost_savings_pct\"]:.2f}%')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Feature Importance Analysis"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from src.models.trainer import ModelTrainer\n",
                "trainer = ModelTrainer()\n",
                "importances = trainer._extract_feature_importances(best_pipe)\n",
                "if importances:\n",
                "    imp_df = pd.DataFrame(list(importances.items()), columns=['Feature', 'Importance']).head(15)\n",
                "    print('Top 15 Most Influential Features in XGBoost Champion Pipeline:')\n",
                "    print(imp_df.to_string(index=False))\n",
                "else:\n",
                "    print('Feature importances not available.')"
            ]
        }
    ]
    create_notebook(cells, "notebooks/05_model_evaluation.ipynb")


if __name__ == "__main__":
    build_notebook_03()
    build_notebook_04()
    build_notebook_05()

