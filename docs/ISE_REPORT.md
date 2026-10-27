# Academic In-Semester Evaluation (ISE) Report

---

### Project Title:
## MLflow Installation, Experiment Tracking and Data Drift Analysis

**Course:** MLOps / Machine Learning DevOps  
**Academic Year:** A.Y. 2026–27  
**Student Name:** `[Your Full Name]`  
**Roll Number / PRN:** `[Your Roll Number]`  
**Department / Class:** `[Department of Computer Engineering / AI & DS / IT, Class: BE / B.Tech]`  
**Institution:** `[Your College / Institute Name]`  
**GitHub Repository:** `[https://github.com/your-username/mlflow-experiment-drift-analysis]`  
**Date of Evaluation:** October 27, 2026  

---

## 1. MLflow Installation and Experiment Tracking

### 1.1 Environment Setup
The development and evaluation environment was established on **Windows 11 (build 10.0.26300)** using VS Code. Prior to environment initialization, the host system was inspected:
- A host default Python 3.15 runtime was identified.
- To prevent binary wheel incompatibilities with MLflow and SciPy compiled C-extensions on Windows, a dedicated, isolated project virtual environment (`.venv`) was instantiated using **Python 3.11.9 (64-bit)** via `py -3.11 -m venv .venv`.
- Core MLOps and scientific computing dependencies were pinned and installed into the virtual environment using `pip install -r requirements.txt`.

#### Installed and Verified Package Versions (From Actual Execution):
| Component / Library | Verified Runtime Version | Functional Role in Pipeline |
|:---|:---|:---|
| **Operating System** | `Windows-10-10.0.26300-SP0` | Host execution platform |
| **Python** | `3.11.9` | Primary runtime engine |
| **MLflow** | `2.22.5` | Experiment tracking server and artifact repository |
| **Scikit-Learn** | `1.9.1` | Iris dataset loader & Random Forest classifier |
| **SciPy** | `1.17.1` | Two-sample Kolmogorov-Smirnov test (`scipy.stats.ks_2samp`) |
| **Pandas** | `2.3.3` | Tabular data structures and report generation |
| **NumPy** | `1.26.4` | Numerical array computations |
| **Matplotlib** | `3.11.2` | Diagnostic drift distribution plotting |
| **Pytest** | `8.4.2` | Automated unit & integration testing framework |

### 1.2 Experiment Tracking
MLflow was integrated to guarantee reproducibility, auditability, and governance over machine learning experiments.
- An MLflow experiment titled **`Iris_MLflow_Drift_Analysis`** was initialized using the local file-system tracking store (`./mlruns`).
- Two independent, decoupled runs were tracked within this experiment:
  1. **`Model_Training_And_Evaluation` (Run ID: `6993254ceaae4f1cb16be36adac29d02`)**:
     - **Logged Hyperparameters:** `model_type="RandomForestClassifier"`, `n_estimators=100`, `random_seed=42`, `test_size=0.20`, `train_samples=120`, `test_samples=30`, `python_version="3.11.9"`, `mlflow_version="2.22.5"`.
     - **Logged Evaluation Metrics:** `accuracy=0.9000`, `precision=0.9024`, `recall=0.9000`, `weighted_f1_score=0.8997`.
     - **Logged Model Artifacts:** Full Scikit-learn model logged via `mlflow.sklearn.log_model()` under `random_forest_model` containing `MLmodel`, `model.pkl`, `conda.yaml`, and `requirements.txt`.
     - **Logged Diagnostics:** Classification report exported as `reports/classification_report.txt`.
  2. **`Data_Drift_Analysis` (Run ID: `95e11b0ba9ff41f5b469b2b5a8bf8df9`)**:
     - **Logged Parameters:** `statistical_test="Kolmogorov-Smirnov (2-sample)"`, `significance_threshold_alpha=0.05`, `simulated_shift_feature="petal length (cm)"`, `simulated_shift_magnitude=1.8`, `reference_sample_count=75`, `current_sample_count=75`.
     - **Logged Metrics:** Per-feature KS statistics (`ks_stat_*`) and p-values (`p_val_*`), plus aggregate `drifted_features_count=1`.
     - **Logged Artifacts:** `drift_reports/drift_results.csv`, `drift_reports/drift_summary.json`, `drift_visualizations/drift_distribution.png`, and `drift_visualizations/feature_drift.png`.

The MLflow web UI dashboard was launched locally via:
```powershell
mlflow ui --host 127.0.0.1 --port 5000
```
accessible at `http://127.0.0.1:5000`.

---

## 2. Dataset Identification

### 2.1 Problem Statement
In production machine learning systems, models are trained on static historical data, but real-world data distributions frequently change over time due to seasonal fluctuations, sensory degradation, or environmental variations. This phenomenon leads to **silent model decay** where a previously accurate model makes erroneous predictions without throwing system errors. The objective of this project is to construct an automated MLOps pipeline that tracks model lineage and implements statistical monitoring to detect data distribution changes before downstream inference failures occur.

### 2.2 Dataset Details
The **Iris Flower Dataset** curated in `scikit-learn.datasets` was programmatically ingested and audited:
- **Total Instances:** 150 records
- **Number of Features:** 4 numerical continuous features:
  1. `sepal length (cm)` [Range: 4.3 – 7.9 cm]
  2. `sepal width (cm)`  [Range: 2.0 – 4.4 cm]
  3. `petal length (cm)` [Range: 1.0 – 6.9 cm]
  4. `petal width (cm)`  [Range: 0.1 – 2.5 cm]
- **Target Variable:** Flower species multiclass label:
  - Class `0`: *Iris setosa* (50 samples)
  - Class `1`: *Iris versicolor* (50 samples)
  - Class `2`: *Iris virginica* (50 samples)
- **Data Integrity Verification:** Missing values = 0 (zero null or NaN values detected).
- **Data Splitting:** Stratified sampling split of 80% training (120 samples, 40 per class) and 20% testing (30 samples, 10 per class) with random seed `42`.

### 2.3 Dataset Selection Rationale
1. **Continuous Multivariate Nature:** The Iris dataset features are strictly continuous numerical variables, making them statistically ideal for non-parametric continuous hypothesis testing such as the Kolmogorov-Smirnov test.
2. **Clear Feature Demarcation:** Distinct morphological variations between classes (specifically petal dimensions) allow controlled drift simulations to mimic real-world sensor calibrations or biological shifts.
3. **Standard Benchmark:** Iris provides an unencumbered, universally accepted benchmark for testing MLOps tracking pipelines and statistical algorithms without data cleaning ambiguities.

---

## 3. Drift Scenario and Detection Method

### 3.1 Drift Type: Theoretical Taxonomy
To ensure conceptual clarity in MLOps governance, we distinguish between three fundamental forms of distribution drift:

```
               [ Input Space X ] ---------------------> [ Output Space Y ]
                       |                                        |
                       v                                        v
               Data Drift (Covariate)                 Prediction Drift
               P(X) shifts; P(Y|X) constant          P(Y_pred) shifts
                                        \              /
                                         v            v
                                        Concept Drift
                                        P(Y|X) shifts
```

1. **Data Drift (Covariate Shift):**
   - Occurs when the probability distribution of input features changes: $P(X_{\text{current}}) \neq P(X_{\text{reference}})$, while the conditional probability distribution of the target given the features remains invariant: $P(Y|X_{\text{current}}) = P(Y|X_{\text{reference}})$.
   - *Example:* A thermometer begins recording in a warmer environment, but the physical freezing point remains identical.
2. **Concept Drift:**
   - Occurs when the mapping between input features and target labels shifts: $P(Y|X_{\text{current}}) \neq P(Y|X_{\text{reference}})$, regardless of whether $P(X)$ changes.
   - *Example:* Macroeconomic inflation changes consumer purchasing power, meaning a past income level now corresponds to a different credit risk category.
3. **Prediction Drift (Prior Probability Shift):**
   - Occurs when the distribution of model output predictions $P(\hat{Y})$ shifts over time, reflecting changes in either input data or decision boundaries.

**Focus of this Project:** This project specifically focuses on **Data Drift (Covariate Shift)** in input features.

### 3.2 Reference and Current Data
- **Reference Dataset ($D_{\text{ref}}$):** A reproducibly sampled subset consisting of 50% of the Iris feature data ($N = 75$ records) sampled with random state `42`, representing the baseline historical data distribution.
- **Current Dataset ($D_{\text{curr}}$):** Created by duplicating the baseline reference dataset and injecting a controlled synthetic offset of **$+1.8\text{ cm}$** into the `petal length (cm)` feature:
  $$\text{petal length}_{\text{curr}} = \text{petal length}_{\text{ref}} + 1.8\text{ cm}$$
- **Controlled Simulation Note:** *This distribution shift is strictly synthetic and intentionally injected for educational verification of the hypothesis test. It is not naturally observed in wild Iris measurements.* The remaining three features (`sepal length`, `sepal width`, `petal width`) remain unperturbed to test false-positive avoidance.

### 3.3 Detection Method: Two-Sample Kolmogorov–Smirnov (KS) Test
The two-sample Kolmogorov–Smirnov test (`scipy.stats.ks_2samp`) is a non-parametric hypothesis test that determines whether two independent continuous univariate samples originate from identical distributions.

#### Mathematical Foundation:
Let $F_{\text{ref}}(x)$ and $F_{\text{curr}}(x)$ represent the empirical cumulative distribution functions (ECDFs) of the reference and current samples:
$$F_n(x) = \frac{1}{n}\sum_{i=1}^{n} \mathbf{1}_{(-\infty, x]}(X_i)$$
The Kolmogorov–Smirnov test statistic $D$ quantifies the supremum (maximum vertical distance) between the two ECDFs:
$$D = \sup_{x} |F_{\text{ref}}(x) - F_{\text{curr}}(x)|$$

#### Hypothesis Testing Framework:
- **Null Hypothesis ($H_0$):** Both samples are drawn from identical continuous probability distributions ($F_{\text{ref}}(x) = F_{\text{curr}}(x)$).
- **Alternative Hypothesis ($H_1$):** The two samples are drawn from different continuous probability distributions ($F_{\text{ref}}(x) \neq F_{\text{curr}}(x)$).
- **Significance Level ($\alpha$):** Set to $\alpha = 0.05$.
- **Decision Rule:**
  - If $p\text{-value} < \alpha$ (0.05): **Reject $H_0$**. Statistically significant evidence of Data Drift.
  - If $p\text{-value} \ge \alpha$ (0.05): **Fail to reject $H_0$**. No statistically significant evidence of Data Drift.

---

## 4. Results, Interpretation and Conclusion

### 4.1 Drift Analysis Results (From Actual Pipeline Execution)
The pipeline was executed in the active environment. The exact statistical metrics calculated by `scipy.stats.ks_2samp` are tabulated below:

| Feature Name | KS Statistic ($D$) | p-value | Threshold ($\alpha$) | Null Hypothesis ($H_0$) | Drift Decision |
|:---|:---:|:---:|:---:|:---:|:---:|
| **sepal length (cm)** | `0.00000` | `1.000000` | `0.05` | Fail to Reject $H_0$ | **No Drift** |
| **sepal width (cm)** | `0.00000` | `1.000000` | `0.05` | Fail to Reject $H_0$ | **No Drift** |
| **petal length (cm)** | `0.45333` | `0.000000` ($< 10^{-12}$) | `0.05` | **Reject $H_0$** | **Drift Detected** |
| **petal width (cm)** | `0.00000` | `1.000000` | `0.05` | Fail to Reject $H_0$ | **No Drift** |

#### Pipeline Model Evaluation Metrics (Baseline Run `6993254ceaae4f1cb16be36adac29d02`):
- **Accuracy:** `0.9000` (90.00%)
- **Precision:** `0.9024` (Weighted)
- **Recall:** `0.9000` (Weighted)
- **F1-Score:** `0.8997` (Weighted)

### 4.2 Interpretation
1. **Detection of Perturbed Feature:**
   For `petal length (cm)`, the KS statistic reached $D = 0.45333$ with a p-value of $0.000000$ ($p < 0.05$). The null hypothesis was decisively rejected, confirming that the two-sample KS test successfully identified the simulated covariate shift.
2. **Robustness Against False Positives:**
   For the three unperturbed features (`sepal length`, `sepal width`, `petal width`), the KS statistic remained $0.00000$ with $p = 1.000000$. The test did not flag false positives on identical distributions.
3. **Statistical vs. Performance Nuance:**
   Rejection of $H_0$ confirms **statistical data drift (covariate shift)**. It proves that the empirical distribution of `petal length` in current data differs significantly from reference data. **However, a statistically significant KS test does NOT prove by itself that model accuracy has decayed (concept drift).** If the shifted values remain within a region where the classifier's decision boundary is invariant, predictions may still be accurate. Conversely, if petal length was a dominant decision feature near the boundary, accuracy degradation is highly probable.

### 4.3 Conclusion and Recommended Action
- **Summary:** The MLOps pipeline demonstrated robust experiment tracking with MLflow and successful statistical drift detection using SciPy.
- **Recommended Operational Protocol:**
  1. **Automated Alerting:** Trigger an alert to the MLOps engineering on-call channel when $\ge 1$ feature exhibits $p < 0.05$.
  2. **Data Pipeline Inspection:** Verify upstream sensory instrumentation or data ingestion scripts to determine whether the shift is due to a faulty sensor or true environmental changes.
  3. **Performance Audit:** Collect ground-truth labels for recent inference requests to evaluate whether empirical accuracy has dipped below the baseline $0.9000$.
  4. **Retraining Strategy:** If concept drift is substantiated, initiate an automated retraining pipeline on recent data, log the new model to the MLflow Model Registry, and execute shadow deployment before promotion to production.

---

## 5. GitHub Repository and Reproducibility

### 5.1 Repository Readiness
The repository is completely configured with standard root files:
- `.gitignore`: Excludes `.venv/`, `__pycache__/`, `mlruns/`, `.pytest_cache/`, and local temporary files.
- `requirements.txt`: Pinned and verified dependencies.
- `train_and_drift.py`: One-command execution script.
- `tests/`: 11 automated test cases passing with 100% success rate.
- `reports/` and `artifacts/`: Generated drift CSV, JSON summary, and high-resolution PNG plots.

### 5.2 Verification Commands (Copy-Paste Ready)
```powershell
# 1. Activate Environment
.\.venv\Scripts\Activate.ps1

# 2. Run Test Suite
pytest -v

# 3. Execute Complete Pipeline
python train_and_drift.py

# 4. Launch MLflow Dashboard
mlflow ui --host 127.0.0.1 --port 5000
```
