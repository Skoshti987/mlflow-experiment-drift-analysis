# MLflow Installation, Experiment Tracking and Data Drift Analysis

[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-2.22.5-orange.svg)](https://mlflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.9+-green.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

An end-to-end, reproducible **MLOps In-Semester Evaluation (ISE)** practical project demonstrating:
- Local Python virtual environment configuration and MLflow setup on Windows.
- Machine learning experiment tracking (parameters, metrics, tags, models, and artifacts) using MLflow.
- Multiclass classification model training using Random Forest on the Scikit-Learn Iris dataset.
- Continuous statistical data drift detection using the two-sample Kolmogorov–Smirnov (KS) test from `scipy.stats`.
- Automated generation of drift audit reports (CSV, JSON) and diagnostic visualizations (PNG).
- Complete automated unit and integration test suite using `pytest`.

---

## 1. Project Overview & Architecture

Modern machine learning models degrade in production when real-world data distributions deviate from training data (known as **Covariate Shift** or **Data Drift**). This project implements an automated MLOps pipeline that tracks model training runs and establishes statistical guardrails to alert engineers when features drift significantly.

```
+-------------------------------------------------------------------------------+
|                             MLOps Pipeline Architecture                        |
+-------------------------------------------------------------------------------+
                                        |
               +------------------------+------------------------+
               |                                                 |
               v                                                 v
   [ 1. Data Ingestion & Split ]                    [ 4. Data Drift Simulation ]
   - Scikit-Learn Iris Dataset                      - Baseline Reference Data (50%)
   - 150 instances, 4 features                      - Shifted Current Data (+1.8cm Petal)
   - Stratified 80/20 Train/Test Split                           |
               |                                                 v
               v                                    [ 5. Two-Sample KS Test ]
   [ 2. Random Forest Classifier ]                  - Null Hypothesis: Distributions Equal
   - n_estimators = 100                             - Significance threshold: alpha = 0.05
   - Evaluation: Acc, Prec, Rec, F1                 - Computes KS Statistic & p-values
               |                                                 |
               v                                                 v
   [ 3. MLflow Tracking Run 1 ]                     [ 6. MLflow Tracking Run 2 ]
   - Log Hyperparameters & Tags                     - Log KS Statistics & p-values
   - Log Evaluation Metrics                         - Log CSV/JSON Reports
   - Log Trained Sklearn Model                      - Log Distribution Comparison Plots
               |                                                 |
               +------------------------+------------------------+
                                        |
                                        v
                       +---------------------------------+
                       |    MLflow Local Tracking Store  |
                       |    - Experiment: Iris_..._Drift |
                       |    - Web Dashboard: :5000       |
                       +---------------------------------+
```

---

## 2. Directory Structure

```text
mlflow-experiment-drift-analysis/
│
├── train_and_drift.py          # Master pipeline orchestrator (Training + Drift + MLflow)
├── requirements.txt            # Pinned, tested dependency specifications
├── README.md                   # Comprehensive project documentation
├── .gitignore                  # Git exclusions for venv, caches, and tracking files
├── .env.example                # Local environment variable template
│
├── src/                        # Modular source code package
│   ├── __init__.py             # Package descriptor
│   ├── data_loader.py          # Dataset loading, verification, and drift dataset synthesis
│   ├── train_model.py          # Random Forest training and metric computation
│   └── drift_detection.py      # SciPy KS-test drift calculation, report & plot generation
│
├── tests/                      # Automated test suite
│   ├── test_data.py            # Dataset integrity, shapes, and stratification tests
│   └── test_drift_detection.py # KS-test behavior, threshold, and report verification
│
├── reports/                    # Generated drift audit reports
│   ├── drift_results.csv       # Tabular summary of KS test statistics and decisions
│   ├── drift_summary.json      # Machine-readable JSON summary of pipeline results
│   └── classification_report.txt # Text classification report from model training
│
├── artifacts/                  # Generated diagnostic visualizations
│   ├── drift_distribution.png  # 2x2 comparison histograms (Reference vs Current)
│   └── feature_drift.png       # KS-statistic and p-value comparative bar charts
│
├── screenshots/                # Practical evaluation evidence
│   └── README.md               # Step-by-step checklist for capturing evaluation screenshots
│
└── docs/                       # Academic submission documentation
    └── ISE_REPORT.md           # Formal college evaluation report
```

---

## 3. Technology Stack

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| **Language** | Python | `3.11.9` | Primary runtime for data science & MLOps |
| **Tracking Server** | MLflow | `2.22.5` | Experiment tracking, artifact & model logging |
| **Machine Learning**| Scikit-Learn | `1.9.1` | Iris dataset loader & Random Forest classifier |
| **Statistical Engine** | SciPy | `1.17.1` | Two-sample Kolmogorov–Smirnov test (`ks_2samp`) |
| **Data Manipulation** | Pandas & NumPy | `2.3.3` / `1.26.4` | DataFrame structures and numerical array math |
| **Visualization** | Matplotlib | `3.8+` | Publication-grade comparative drift distribution plots |
| **Testing** | Pytest | `8.0+` | Unit tests for data loading and drift validation |
| **Version Control** | Git & GitHub | `2.47+` | Source code management and reproducibility |

---

## 4. Setup and Installation (Windows PowerShell / CMD)

### Step 1: Clone or Navigate to the Workspace
```powershell
cd d:\sem7\mldevops\ISE2Theory
```

### Step 2: Create a Dedicated Python 3.11 Virtual Environment
Inspect your available Python versions using:
```powershell
py -0
```
Create a clean virtual environment using Python 3.11:
```powershell
py -3.11 -m venv .venv
```

### Step 3: Activate the Virtual Environment
- **In Windows PowerShell**:
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
  *(If PowerShell displays an `Execution_Policies` restriction, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and re-activate).*
- **In Windows Command Prompt (cmd.exe)**:
  ```cmd
  .\.venv\Scripts\activate.bat
  ```

### Step 4: Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 5: Verify Installed Versions
```powershell
python -c "import mlflow, sklearn, scipy, pandas, numpy; print('MLflow:', mlflow.__version__, '| Sklearn:', sklearn.__version__, '| SciPy:', scipy.__version__)"
```

---

## 5. Execution Instructions

### A. Run the Complete MLOps Pipeline
Execute the master script to load the data, train the Random Forest model, log to MLflow, evaluate data drift, and export artifacts:
```powershell
python train_and_drift.py
```

### B. Run the Automated Test Suite
Ensure the implementation satisfies all statistical assertions and data integrity constraints:
```powershell
pytest -v
```

---

## 6. Accessing the MLflow UI Dashboard

Launch the local MLflow dashboard:
```powershell
mlflow ui --host 127.0.0.1 --port 5000
```

1. Open your web browser and navigate to: **[http://127.0.0.1:5000](http://127.0.0.1:5000)**
2. In the left panel, select the experiment: **`Iris_MLflow_Drift_Analysis`**
3. You will see two structured runs:
   - **`Model_Training_And_Evaluation`**:
     - **Parameters**: `n_estimators`, `random_seed`, `test_size`, `model_type`, `python_version`.
     - **Metrics**: `accuracy`, `precision`, `recall`, `weighted_f1_score`.
     - **Artifacts**: Serialized `random_forest_model` (MLmodel, conda.yaml, model.pkl) and `classification_report.txt`.
   - **`Data_Drift_Analysis`**:
     - **Parameters**: `statistical_test`, `significance_threshold_alpha`, `simulated_shift_feature`.
     - **Metrics**: `ks_stat_*`, `p_val_*`, `drifted_features_count`.
     - **Artifacts**: `drift_reports/drift_results.csv`, `drift_reports/drift_summary.json`, `drift_visualizations/drift_distribution.png`, `drift_visualizations/feature_drift.png`.

---

## 7. Statistical Data Drift Methodology

### The Two-Sample Kolmogorov–Smirnov (KS) Test
The KS test is a non-parametric statistical hypothesis test that compares the continuous empirical cumulative distribution functions (ECDFs) of two independent samples:
$$D = \sup_x |F_{\text{reference}}(x) - F_{\text{current}}(x)|$$

### Hypothesis Definition:
- **Null Hypothesis ($H_0$)**: The reference baseline and current incoming feature samples originate from the same continuous probability distribution.
- **Alternative Hypothesis ($H_1$)**: The feature distribution has shifted ($F_{\text{reference}}(x) \neq F_{\text{current}}(x)$).
- **Decision Rule**:
  - If **$p\text{-value} < 0.05$** ($\alpha$): Reject $H_0$. Statistically significant evidence of **Data Drift**.
  - If **$p\text{-value} \ge 0.05$**: Fail to reject $H_0$. **No Drift Detected**.

> **Important Statistical Note**:
> Rejection of the null hypothesis confirms **Covariate Shift** (Data Drift in $P(X)$). It demonstrates that the feature distribution has shifted significantly. However, it does **not** inherently prove that model accuracy has degraded (Concept Drift $P(Y|X)$). Continued model monitoring and ground-truth validation are recommended upon detecting drift.

---

## 8. Limitations & Future Extensions

1. **Simulated Shift**: Petal length was synthetically perturbed by $+1.8\text{ cm}$ to demonstrate detection capability. In production, real-time streaming batch windows would ingest live inference inputs.
2. **Multivariate Drift**: The KS test operates per univariate feature. Future iterations can incorporate multivariate statistical metrics such as Maximum Mean Discrepancy (MMD) or Wasserstein Distance.
3. **Automated Retraining**: Upon detecting drift exceeding a threshold (e.g. $\ge 2$ drifted features), an automated CI/CD retraining trigger via GitHub Actions or Airflow can be established.

---

## 9. GitHub Submission & Reproducibility Guide

To publish your repository to GitHub:
```powershell
# Initialize git if not already initialized
git init

# Check status (confirm .venv/ and mlruns/ are ignored)
git status

# Add files and commit
git add .
git commit -m "feat: complete MLOps ISE experiment tracking and data drift analysis project"

# Add your remote and push
git branch -M main
git remote add origin https://github.com/Skoshti987/mlflow-experiment-drift-analysis.git
git push -u origin main
```
