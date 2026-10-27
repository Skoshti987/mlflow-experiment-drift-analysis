# MLflow Installation, Experiment Tracking and Data Drift Analysis

## Details

### Student Entry
- **Student Name:** `[Your Full Name]`
- **Roll Number(s):** `[Your Roll Number / PRN]`
- **Department / Class:** `[Department of Computer Engineering / AI & DS / IT, Class: BE / B.Tech]`
- **Academic Year:** `A.Y. 2026–27`
- **GitHub Repository URL:** `https://github.com/Skoshti987/mlflow-experiment-drift-analysis`

---

## 1. MLflow Installation and Experiment Tracking

### 1.1 Environment Setup
- **Operating System:** Windows 11 (64-bit, Build: `Windows-10-10.0.26300-SP0`)
- **Python Version:** `3.11.9` (Isolated project virtual environment: `.venv`)
- **MLflow Version:** `2.22.5`
- **Installation and verification status:**
  Successfully installed in a dedicated virtual environment with all core scientific dependencies (`scikit-learn 1.9.1`, `scipy 1.17.1`, `pandas 2.3.3`, `numpy 1.26.4`, `matplotlib 3.11.2`, and `pytest 8.4.2`). Verified via direct CLI import, 100% passing automated test suite (11/11 tests green), and active HTTP status (200 OK) from the MLflow UI server.

### 1.2 Experiment Tracking
- **Experiment Name:** `Iris_MLflow_Drift_Analysis`
- **Experiment Objective:**
  Establish an end-to-end MLOps experiment tracking and continuous monitoring pipeline. The pipeline logs classification model hyperparameters, performance evaluation metrics, and serialized model binaries, while decoupling statistical data drift monitoring into a separate tracked run with analytical reports and visual distribution artifacts.
- **Parameters, metrics or artifacts recorded:**
  - **Run 1: `Model_Training_And_Evaluation`**
    - *Parameters:* `model_type="RandomForestClassifier"`, `n_estimators=100`, `random_seed=42`, `test_size=0.20`, `train_samples=120`, `test_samples=30`, `python_version="3.11.9"`, `mlflow_version="2.22.5"`, `sklearn_version="1.9.1"`.
    - *Metrics:* Accuracy = `0.9000` (90.00%), Precision = `0.9024` (Weighted), Recall = `0.9000` (Weighted), Weighted F1-Score = `0.8997`.
    - *Artifacts:* Serialized Scikit-Learn Model (`random_forest_model/` containing `MLmodel`, `model.pkl`, `conda.yaml`, `requirements.txt`), Classification Report (`reports/classification_report.txt`).
  - **Run 2: `Data_Drift_Analysis`**
    - *Parameters:* `statistical_test="Kolmogorov-Smirnov (2-sample)"`, `significance_threshold_alpha=0.05`, `simulated_shift_feature="petal length (cm)"`, `simulated_shift_magnitude=1.8`, `reference_sample_count=75`, `current_sample_count=75`, `total_features_tested=4`.
    - *Metrics:* `ks_stat_petal_length_cm=0.4533`, `p_val_petal_length_cm=0.0000`, `ks_stat_sepal_length_cm=0.0000`, `p_val_sepal_length_cm=1.0000`, `ks_stat_sepal_width_cm=0.0000`, `p_val_sepal_width_cm=1.0000`, `ks_stat_petal_width_cm=0.0000`, `p_val_petal_width_cm=1.0000`, `drifted_features_count=1`.
    - *Artifacts:* `reports/drift_results.csv`, `reports/drift_summary.json`, `artifacts/drift_distribution.png`, `artifacts/feature_drift.png`.
- **Brief description of successful execution:**
  The orchestrator script `train_and_drift.py` executed successfully in 18 seconds without errors. It loaded and validated the Iris dataset, trained a Random Forest model, logged all parameters and metrics to MLflow, performed two-sample Kolmogorov-Smirnov hypothesis tests, exported CSV/JSON reports, generated multi-panel distribution plots, and saved all runs to the local tracking store (`./mlruns`). The local dashboard was launched and verified on `http://127.0.0.1:5000`.

---

## 2. Dataset Identification

### 2.1 Problem Statement
In production machine learning systems, deployed models are vulnerable to silent degradation when real-world operational distributions diverge from historical training distributions. In domains such as automated agricultural classification, botanical monitoring, and biological quality inspection, environmental variations and sensor discrepancies cause "Covariate Shift" (Data Drift). This project addresses the challenge of building an automated MLOps monitoring mechanism that flags distribution shifts before they cause inference failures.

### 2.2 Dataset Details
- **Dataset Name:** Iris Flower Dataset
- **Dataset Source:** Scikit-Learn (`sklearn.datasets.load_iris`) / UCI Machine Learning Repository
- **Number of Records:** 150 instances (120 training, 30 testing via stratified 80/20 split)
- **Important Features:** 4 continuous numerical morphological measurements:
  1. `sepal length (cm)`
  2. `sepal width (cm)`
  3. `petal length (cm)`
  4. `petal width (cm)`
- **Target Variable (if applicable):** Flower species class (Categorical multiclass: `0: setosa`, `1: versicolor`, `2: virginica` — balanced at 50 records per class)
- **Use Case Category:** ML (Supervised Classification) & MLOps Continuous Drift Monitoring

### 2.3 Dataset Selection Rationale
The Iris dataset provides four well-defined, continuous numerical variables with zero missing values, making it an ideal statistical benchmark for the continuous Two-Sample Kolmogorov-Smirnov test. Because flower species exhibit distinct morphological petal dimensions, injecting a controlled shift into petal length provides a clear, verifiable demonstration of distribution divergence without introducing data-cleaning confounders.

---

## 3. Drift Scenario and Detection Method

### 3.1 Drift Type
- **Identified Drift Type:** **Data Drift (Covariate Shift)**
- *Taxonomy Context:*
  - **Data Drift:** The input feature distribution shifts: $P(X_{\text{current}}) \neq P(X_{\text{reference}})$, while the relationship to output labels $P(Y|X)$ remains constant. (Primary focus of this project).
  - **Concept Drift:** The conditional probability mapping shifts: $P(Y|X_{\text{current}}) \neq P(Y|X_{\text{reference}})$.
  - **Prediction Drift:** The distribution of model output predictions shifts: $P(\hat{Y}_{\text{current}}) \neq P(\hat{Y}_{\text{reference}})$.

### 3.2 Reference and Current Data
- **Reference Dataset:** 75 samples randomly and reproducibly drawn (50% sample, random seed 42) from the validated baseline Iris features, representing historical training data.
- **Current Dataset:** A duplicate copy of the reference baseline with a controlled, synthetic offset of **$+1.8\text{ cm}$** injected into `petal length (cm)`.
- *Simulation Clarification:* This shift is strictly simulated to model environmental sensor calibration error or seasonal growth shifts, and is not naturally observed in wild Iris measurements. The remaining three features were left untouched to test false-positive avoidance.

### 3.3 Detection Method
- **Method / Statistical Test / Tool Used:**
  Two-Sample Kolmogorov–Smirnov (KS) test implemented via `scipy.stats.ks_2samp`. The KS test calculates the maximum absolute vertical difference ($D$) between the empirical cumulative distribution functions (ECDFs) of the reference and current samples:
  $$D = \sup_{x} |F_{\text{reference}}(x) - F_{\text{current}}(x)|$$
- **Features or outputs analyzed:**
  All four numerical input features: `sepal length (cm)`, `sepal width (cm)`, `petal length (cm)`, and `petal width (cm)`.
- **Detection threshold or criteria, if applicable:**
  Significance level **$\alpha = 0.05$**.
  - **Null Hypothesis ($H_0$):** Reference and current samples are drawn from identical continuous probability distributions.
  - **Decision Rule:** Reject $H_0$ if $p\text{-value} < 0.05$ (Statistically significant evidence of Data Drift). Fail to reject $H_0$ if $p\text{-value} \ge 0.05$ (No Drift).

---

## 4. Results, Interpretation and Conclusion

### 4.1 Drift Analysis Results
Actual values obtained from execution:

| Feature Name | KS Statistic ($D$) | p-value | Threshold ($\alpha$) | Null Hypothesis ($H_0$) | Drift Decision |
|:---|:---:|:---:|:---:|:---:|:---:|
| **sepal length (cm)** | `0.00000` | `1.0000e+00` | `0.05` | Fail to Reject | **No Drift** |
| **sepal width (cm)** | `0.00000` | `1.0000e+00` | `0.05` | Fail to Reject | **No Drift** |
| **petal length (cm)** | `0.45333` | `0.0000e+00` | `0.05` | **Reject $H_0$** | **Drift Detected** |
| **petal width (cm)** | `0.00000` | `1.0000e+00` | `0.05` | Fail to Reject | **No Drift** |

- **Total Features Evaluated:** 4
- **Drifted Features Detected:** 1 (`petal length (cm)`)
- **Overall Pipeline Status:** `DRIFT_DETECTED`
- Detailed CSV reports (`reports/drift_results.csv`), JSON summaries (`reports/drift_summary.json`), and publication-grade plots (`artifacts/drift_distribution.png`, `artifacts/feature_drift.png`) were exported and logged as MLflow artifacts.

### 4.2 Interpretation
The Two-Sample KS test demonstrated precise statistical discernment:
1. **Successful Drift Identification:** On the perturbed feature `petal length (cm)`, the KS distance widened to $D = 0.4533$ with an infinitesimal p-value ($p = 0.0000 < 0.05$), decisively rejecting $H_0$.
2. **False-Alarm Immunity:** For all unperturbed features (`sepal length`, `sepal width`, `petal width`), the statistic was $D = 0.0000$ and $p = 1.0000$, proving zero false positives.
3. **Engineering Significance:** In an operational inspection pipeline, this indicates that the physical distribution of petal measurements has shifted significantly. While statistical data drift does not automatically imply that classification accuracy has collapsed (concept drift), it indicates that model predictions are operating on out-of-distribution inputs.

### 4.3 Conclusion and Recommended Action
- **Conclusion:** **Statistically significant data drift was detected** in feature `petal length (cm)`. No drift was observed in the remaining features.
- **Appropriate Next Actions:**
  1. *Immediate Alert:* Dispatch an automated notification to the MLOps engineering team regarding covariate shift on `petal length`.
  2. *Upstream Sensor Audit:* Inspect measurement devices or upstream data ingestion code to rule out physical sensor calibration error.
  3. *Inference Ground-Truth Auditing:* Sample recent inference queries and obtain verified ground-truth labels to evaluate if live accuracy has dropped below the baseline $0.9000$.
  4. *Automated Retraining:* If accuracy degradation is substantiated, trigger an automated retraining pipeline on recently collected data, register the updated model in MLflow Model Registry, and execute shadow validation before deployment.

---

## 5. GitHub Repository and Reproducibility

- **GitHub Repository URL:** `https://github.com/Skoshti987/mlflow-experiment-drift-analysis`
- **Source code / Notebook:** Modular Python implementation containing `train_and_drift.py`, `src/data_loader.py`, `src/train_model.py`, and `src/drift_detection.py`.
- **README and execution instructions:** Complete copy-paste ready instructions for Windows PowerShell / CMD provided in [`README.md`](file:///d:/sem7/mldevops/ISE2Theory/README.md).
- **Requirements / Dependencies:** Pinned and tested dependencies documented in [`requirements.txt`](file:///d:/sem7/mldevops/ISE2Theory/requirements.txt).
- **Screenshots and drift reports:**
  - Audit reports: `reports/drift_results.csv`, `reports/drift_summary.json`, `reports/classification_report.txt`.
  - Visual distribution plots: `artifacts/drift_distribution.png`, `artifacts/feature_drift.png`.
  - Step-by-step screenshot submission checklist: [`screenshots/README.md`](file:///d:/sem7/mldevops/ISE2Theory/screenshots/README.md).
- **MLflow experiment and drift-analysis evidence:**
  Local experiment `Iris_MLflow_Drift_Analysis` saved under `./mlruns` tracking runs `Model_Training_And_Evaluation` (Run ID: `cefc6b197971411fb14cf0581b37b94d`) and `Data_Drift_Analysis` (Run ID: `f033bc6ed2734a29bd416379654df506`) accessible at `http://127.0.0.1:5000`.
