# CreditGuard: ML-powered Credit Risk Prediction and Intelligence Platform

CreditGuard is an enterprise-grade credit risk analytics and intelligence platform built to evaluate borrower default likelihood, quantify portfolio exposure, and uncover behavioral risk drivers using rigorous banking analytics standards.

---

## Tech Stack
* **Language & Runtime**: Python 3.14+ (Virtual Environment `.venv`)
* **Data Processing & Analytics**: Pandas, NumPy, Scipy
* **Machine Learning & Interpretability**: Scikit-learn, XGBoost, SHAP
* **Visualization & Reporting**: Plotly Interactive Visualizations, Matplotlib, Seaborn
* **Application & Exploration**: Streamlit, Jupyter Notebooks
* **Testing & Quality Assurance**: Pytest

---

## Project Architecture

```text
CreditGuard/
├── data/
│   ├── raw/
│   │   └── credit_risk_dataset.csv          # Read-only raw portfolio (32,581 records)
│   └── processed/
│       └── cleaned_credit_risk_dataset.csv  # Imputed & feature-engineered dataset (32,409 records)
├── src/
│   ├── __init__.py                          # Package root
│   ├── config.py                            # Schema, domain constraints & threshold constants
│   ├── ingestion/
│   │   ├── __init__.py
│   │   └── loader.py                        # Safe data loading & ValidationReport engine
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── cleaner.py                       # Deduplication, outlier treatment & EDA dataset cleaner
│   │   └── ml_pipeline.py                   # Leakage-safe ColumnTransformer & FeatureEngineer
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── stats.py                         # Advanced statistics, IV/WoE ranking, correlations
│   │   └── financial.py                     # Debt burden analysis & portfolio exposure
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── plots.py                         # Interactive Plotly visualizer & chart exporter
│   └── models/
│       ├── __init__.py
│       ├── trainer.py                       # Stratified CV & hyperparameter tuning
│       ├── evaluator.py                     # Cost-based threshold optimizer & metrics
│       ├── persistence.py                   # Joblib/JSON artifact persistence manager
│       └── pipeline_runner.py               # Phase 2 model development orchestrator
├── notebooks/
│   ├── 01_data_validation_and_cleaning.ipynb # Ingestion, validation & cleaning walkthrough
│   └── 02_exploratory_data_analysis.ipynb    # In-depth EDA & financial risk analysis
│   ├── 03_feature_engineering.ipynb         # Leakage-safe ML feature engineering pipeline
│   ├── 04_model_training.ipynb              # Baseline & ensemble training with 5-fold CV
│   └── 05_model_evaluation.ipynb            # Cost-optimal threshold search & out-of-sample metrics
├── tests/
│   ├── test_loader.py                       # Schema integrity & loader unit tests
│   ├── test_cleaner.py                      # Outlier & imputation unit tests
│   ├── test_analysis.py                     # Risk distribution & financial metrics tests
│   ├── test_ml_pipeline.py                  # Leakage prevention & transformer unit tests
│   ├── test_models.py                       # Pipeline roundtrip & threshold persistence tests
│   └── test_evaluator.py                    # Metric calculation & threshold optimizer tests
├── models/
│   ├── best_creditguard_pipeline.joblib     # Persisted production champion pipeline
│   ├── optimal_threshold.json               # Cost-optimal decision threshold metadata
│   └── model_comparison_metrics.json        # Test benchmark evaluation metrics (JSON & CSV)
├── reports/
│   └── figures/                             # Exported interactive HTML visualizations
├── main.py                                  # CLI pipeline orchestration entrypoint
├── requirements.txt                         # Pinned project dependencies
└── .gitignore                               # Environment and cache ignore rules
```

---

## Phase 2: ML Model Development & Benchmark Results

### 1. Model Comparison Table (Test Set: $N = 6,482$)

All models were tuned via 5-Fold Stratified Cross-Validation on the training partition ($N = 25,927$) and evaluated on the untouched hold-out test set ($N = 6,482$):

| Model | 5-Fold CV ROC-AUC | Train ROC-AUC | Test ROC-AUC | Overfitting Gap | Test PR-AUC | Test Precision | Test Recall | Test F1 | Test Brier Score | Total Cost (@ 0.50) | Optimal Threshold | Total Cost (@ Optimal) | Loss Cost Reduction |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.8794 | 0.8805 | 0.8802 | +0.0003 | 0.7269 | 0.5930 | 0.7757 | 0.6722 | 0.1315 | \$2,345,000 | 0.5346 | \$2,334,000 | 0.5% |
| **Random Forest** | 0.9306 | 0.9999 | 0.9354 | +0.0645 | 0.8916 | 0.8727 | 0.7687 | 0.8174 | 0.0670 | \$1,799,000 | 0.3763 | \$1,685,000 | 6.3% |
| **XGBoost (Champion)** | **0.9462** | 0.9786 | **0.9528** | **+0.0258** | **0.9104** | **0.8245** | **0.8082** | **0.8162** | **0.0667** | **\$1,604,000** | **0.4357** | **\$1,561,000** | **2.7%** |

*(Cost Matrix: $C_{FN} = \$5,000$ per missed default, $C_{FP} = \$1,000$ per declined good applicant)*

### 2. Best Model Selection Rationale
**XGBoost** is selected as the production champion model based on four core criteria:
1. **Dominant Separation Under Class Imbalance**: Achieves a **0.9104 Test PR-AUC** (compared to 0.7269 for Logistic Regression and 0.8916 for Random Forest) and **0.9528 Test ROC-AUC**.
2. **Superior Default Capture**: Achieves **80.82% Recall** at default threshold and **83.1% Recall** at the cost-optimal threshold without sacrificing precision.
3. **Controlled Generalization Gap**: The overfitting gap between Train ROC-AUC (0.9786) and Test ROC-AUC (0.9528) is only **0.0258**, compared to Random Forest's 0.0645 gap (Train ROC-AUC: 0.9999).
4. **Lowest Financial Risk Exposure**: Yields the lowest total expected cost (\$1,561,000 at $\tau^* = 0.4357$), outperforming Logistic Regression by over \$773,000 in saved default losses.

### 3. Top Most Predictive Features (XGBoost)
1. `income_to_loan_ratio` (0.09215): Replaces flawed disposable income with inverse debt coverage ratio.
2. `person_home_ownership_RENT` (0.08174): Significant risk premium for renters vs homeowners.
3. `loan_grade_A` (0.07977): Prime borrower tier credit separator.
4. `loan_grade_C` (0.07189): Middle-risk grade tier.
5. `loan_int_rate` (0.06575): Market pricing reflecting perceived borrower risk.
6. `loan_grade_D` (0.06324): Subprime risk indicator.
7. `loan_intent_HOMEIMPROVEMENT` (0.04947): Loan purpose profile.
8. `person_home_ownership_OWN` (0.04582): Strong protective asset ownership indicator.
9. `loan_intent_VENTURE` (0.04287): Business venture risk.
10. `loan_intent_DEBTCONSOLIDATION` (0.04275): Stressed refinancing intent.

---

## Quickstart & CLI Commands

### 1. Activate Environment
```bash
source .venv/bin/activate
```

### 2. Run Complete Platform Pipeline
```bash
# Run Phase 1 Foundation + Phase 2 ML Training & Evaluation end-to-end:
python main.py --all

# Run Phase 2 ML training, cross-validation, and evaluation:
python main.py --phase2

# Run Phase 1 data validation and EDA:
python main.py --phase1
```

### 3. Run Automated Unit Tests
```bash
pytest tests/ -v
```
*(All 20 unit tests pass in under 3 seconds)*
