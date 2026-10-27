# Practical Evaluation Screenshots Guide

This directory is designated for storing genuine execution screenshots to submit alongside your **MLOps ISE Evaluation Report**.

> **Important College Evaluation Note**:
> Evaluators check that screenshots are authentic and generated from your own machine. Do not use generic internet images. Follow the numbered checklist below to capture each required screenshot after running the pipeline.

---

## Required Screenshots Checklist

| # | Screenshot Filename | Description / Screen to Capture | How to Capture |
|---|---------------------|---------------------------------|----------------|
| **1** | `01_env_and_versions.png` | Terminal showing Python, Pip, MLflow, and Scikit-Learn versions. | Run `.\.venv\Scripts\python.exe -c "import mlflow, sklearn, scipy; print('MLflow:', mlflow.__version__, 'Sklearn:', sklearn.__version__, 'SciPy:', scipy.__version__)"` |
| **2** | `02_terminal_execution.png` | Terminal output after running `python train_and_drift.py` showing accuracy, precision, recall, F1, and the KS test table. | Capture the terminal window right after the script prints `[5/5] Pipeline Execution Completed Successfully!`. |
| **3** | `03_test_suite_passing.png` | Pytest passing all tests with 100% green status. | Run `.\.venv\Scripts\pytest.exe -v` and take a screenshot of the output. |
| **4** | `04_mlflow_experiments_list.png` | MLflow Web UI showing the experiment `Iris_MLflow_Drift_Analysis` with both runs (`Model_Training_And_Evaluation` and `Data_Drift_Analysis`). | Open `http://127.0.0.1:5000` in browser. Click on `Iris_MLflow_Drift_Analysis` in the left sidebar. |
| **5** | `05_model_run_details.png` | MLflow run details for `Model_Training_And_Evaluation` showing logged Parameters (`n_estimators`, `random_seed`, etc.) and Metrics (`accuracy`, `precision`, `recall`, `weighted_f1_score`). | In MLflow UI, click on the `Model_Training_And_Evaluation` run link. Scroll down to show parameters and metrics tables. |
| **6** | `06_logged_model_artifact.png` | MLflow Artifact viewer showing `random_forest_model` with `MLmodel`, `conda.yaml`, `model.pkl`. | On the same model run page, scroll to the **Artifacts** section and click on `random_forest_model`. |
| **7** | `07_drift_run_artifacts.png` | MLflow run details for `Data_Drift_Analysis` showing logged artifacts (`drift_reports/` and `drift_visualizations/`). | In MLflow UI, go to the `Data_Drift_Analysis` run. Scroll to the **Artifacts** section and expand both folders. |
| **8** | `08_drift_visualizations.png` | Generated drift distribution plot (`artifacts/drift_distribution.png`). | Open `artifacts/drift_distribution.png` in VS Code or image viewer. |

---

## Fast Capture Steps on Windows

1. **Start the MLflow UI**:
   ```powershell
   .\.venv\Scripts\mlflow.exe ui --host 127.0.0.1 --port 5000
   ```
2. **Open your browser** to:
   ```
   http://127.0.0.1:5000
   ```
3. Use Windows shortcut **`Win + Shift + S`** (Snipping Tool) to capture the screen or active window.
4. Paste and save the image into this `screenshots/` directory using the filenames listed above (e.g. `screenshots/04_mlflow_experiments_list.png`).
