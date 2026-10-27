"""
MLOps Pipeline Orchestrator: Experiment Tracking and Data Drift Analysis.

Executes:
1. Iris Dataset Loading and Preprocessing.
2. MLflow Experiment Setup ('Iris_MLflow_Drift_Analysis').
3. Run 1: Model Training, Evaluation, and MLflow Logging.
4. Run 2: Two-Sample Kolmogorov-Smirnov Data Drift Detection and MLflow Logging.
5. Generation of Reports (CSV, JSON) and Diagnostic Visualizations (PNG).
"""

import sys
import platform
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")

import pandas as pd
import numpy as np
import scipy
import sklearn
import mlflow
import mlflow.sklearn

from src.data_loader import load_iris_data, split_data, prepare_drift_datasets
from src.train_model import train_and_evaluate_model
from src.drift_detection import (
    compute_kolmogorov_smirnov_drift,
    save_drift_reports,
    generate_drift_visualizations
)

EXPERIMENT_NAME = "Iris_MLflow_Drift_Analysis"
MODEL_RUN_NAME = "Model_Training_And_Evaluation"
DRIFT_RUN_NAME = "Data_Drift_Analysis"


def sanitize_metric_name(name: str) -> str:
    """Format string for clean MLflow metric keys without special characters."""
    return (
        name.replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "_")
        .lower()
    )


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    print("=" * 75)
    print("  MLOps Pipeline: Experiment Tracking & Statistical Data Drift Analysis")
    print("=" * 75)

    # -------------------------------------------------------------------------
    # Step 1: Environment & Dependency Verification
    # -------------------------------------------------------------------------
    env_info = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "mlflow_version": mlflow.__version__,
        "sklearn_version": sklearn.__version__,
        "scipy_version": scipy.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
    }

    print("\n[INFO] Active Environment Details:")
    for k, v in env_info.items():
        print(f"  • {k.replace('_', ' ').capitalize()}: {v}")

    # -------------------------------------------------------------------------
    # Step 2: Load and Validate Iris Dataset
    # -------------------------------------------------------------------------
    print("\n[1/5] Loading and Validating Iris Dataset...")
    X, y, data_meta = load_iris_data()
    print(f"  • Shape: {X.shape[0]} records, {X.shape[1]} features")
    print(f"  • Features: {', '.join(data_meta['feature_names'])}")
    print(f"  • Classes: {', '.join(data_meta['target_names'])} (0 missing values)")

    # Stratified Train/Test Split
    test_size = 0.20
    random_seed = 42
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size, random_state=random_seed)
    print(f"  • Train Set Size: {len(X_train)} samples | Test Set Size: {len(X_test)} samples")

    # -------------------------------------------------------------------------
    # Step 3: Setup MLflow Experiment
    # -------------------------------------------------------------------------
    print(f"\n[2/5] Setting up MLflow Experiment: '{EXPERIMENT_NAME}'...")
    mlflow.set_experiment(EXPERIMENT_NAME)

    # -------------------------------------------------------------------------
    # Step 4: Run 1 — Model Training and MLflow Tracking
    # -------------------------------------------------------------------------
    print(f"\n[3/5] Starting MLflow Run: '{MODEL_RUN_NAME}'...")
    n_estimators = 100
    with mlflow.start_run(run_name=MODEL_RUN_NAME) as model_run:
        model_run_id = model_run.info.run_id
        print(f"  • Active Run ID: {model_run_id}")

        # Train and Evaluate Random Forest Classifier
        model, metrics, eval_details = train_and_evaluate_model(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            n_estimators=n_estimators,
            random_state=random_seed
        )

        print("\n  [Model Evaluation Metrics]")
        print(f"    - Accuracy:    {metrics['accuracy']:.4f}")
        print(f"    - Precision:   {metrics['precision']:.4f}")
        print(f"    - Recall:      {metrics['recall']:.4f}")
        print(f"    - F1-Score:    {metrics['f1_weighted']:.4f}")

        # Log Tags
        mlflow.set_tags({
            "project": "MLOps_ISE_Evaluation",
            "task": "classification",
            "model_type": "RandomForestClassifier",
            "dataset": "Iris",
            "author": "Student Candidate",
            "run_purpose": "Baseline model training & performance evaluation"
        })

        # Log Parameters
        mlflow.log_params({
            "model_type": "RandomForestClassifier",
            "n_estimators": n_estimators,
            "random_seed": random_seed,
            "test_size": test_size,
            "dataset_name": data_meta["dataset_name"],
            "dataset_records": data_meta["num_records"],
            "dataset_features": data_meta["num_features"],
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "python_version": env_info["python_version"],
            "mlflow_version": env_info["mlflow_version"],
            "sklearn_version": env_info["sklearn_version"]
        })

        # Log Metrics
        mlflow.log_metrics({
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "weighted_f1_score": metrics["f1_weighted"]
        })

        # Log Scikit-Learn Model Artifact
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="random_forest_model",
            input_example=X_test.iloc[:3]
        )

        # Save classification report as a text artifact
        report_path = Path("reports") / "classification_report.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=== Model Classification Report ===\n")
            f.write(eval_details["classification_report"])
        mlflow.log_artifact(str(report_path))
        print(f"  • Model & metrics successfully logged to MLflow Run: {model_run_id}")

    # -------------------------------------------------------------------------
    # Step 5: Run 2 — Data Drift Detection and MLflow Tracking
    # -------------------------------------------------------------------------
    print(f"\n[4/5] Starting MLflow Run: '{DRIFT_RUN_NAME}'...")
    shifted_feature = "petal length (cm)"
    shift_magnitude = 1.8
    alpha_threshold = 0.05

    reference_data, current_data, drift_scenario = prepare_drift_datasets(
        X,
        shift_feature=shifted_feature,
        shift_magnitude=shift_magnitude,
        sample_fraction=0.5,
        random_state=random_seed
    )

    drift_scenario["alpha_threshold"] = alpha_threshold
    drift_scenario["environment"] = env_info

    # Execute Two-Sample Kolmogorov-Smirnov Test
    drift_df = compute_kolmogorov_smirnov_drift(
        reference_df=reference_data,
        current_df=current_data,
        alpha=alpha_threshold
    )

    print("\n  [Kolmogorov-Smirnov Test Results (Alpha = 0.05)]")
    print(f"  {'Feature':<22} | {'KS Stat':<10} | {'p-value':<12} | {'Drift Status'}")
    print("  " + "-" * 65)
    for _, row in drift_df.iterrows():
        print(f"  {row['feature']:<22} | {row['ks_statistic']:<10.4f} | {row['p_value']:<12.4e} | {row['drift_status']}")

    # Generate Reports and Visualizations
    report_paths = save_drift_reports(drift_df, drift_scenario, output_dir="reports")
    viz_paths = generate_drift_visualizations(reference_data, current_data, drift_df, output_dir="artifacts")

    with mlflow.start_run(run_name=DRIFT_RUN_NAME) as drift_run:
        drift_run_id = drift_run.info.run_id
        print(f"\n  • Active Drift Run ID: {drift_run_id}")

        # Set Tags
        mlflow.set_tags({
            "project": "MLOps_ISE_Evaluation",
            "task": "data_drift_analysis",
            "statistical_test": "Two-Sample Kolmogorov-Smirnov",
            "simulated_shift_feature": shifted_feature,
            "run_purpose": "Continuous statistical data drift monitoring"
        })

        # Log Parameters
        mlflow.log_params({
            "statistical_test": "Kolmogorov-Smirnov (2-sample)",
            "significance_threshold_alpha": alpha_threshold,
            "simulated_shift_feature": shifted_feature,
            "simulated_shift_magnitude": shift_magnitude,
            "reference_sample_count": len(reference_data),
            "current_sample_count": len(current_data),
            "total_features_tested": len(drift_df)
        })

        # Log Metrics
        drift_metrics_to_log = {}
        for _, row in drift_df.iterrows():
            sanitized_name = sanitize_metric_name(row["feature"])
            drift_metrics_to_log[f"ks_stat_{sanitized_name}"] = float(row["ks_statistic"])
            drift_metrics_to_log[f"p_val_{sanitized_name}"] = float(row["p_value"])

        drift_metrics_to_log["drifted_features_count"] = int(drift_df["drift_detected"].sum())
        mlflow.log_metrics(drift_metrics_to_log)

        # Log Artifacts
        mlflow.log_artifact(report_paths["csv_path"], artifact_path="drift_reports")
        mlflow.log_artifact(report_paths["json_path"], artifact_path="drift_reports")
        mlflow.log_artifact(viz_paths["distribution_plot_path"], artifact_path="drift_visualizations")
        mlflow.log_artifact(viz_paths["metrics_plot_path"], artifact_path="drift_visualizations")
        print(f"  • Reports & diagnostic plots logged to MLflow Run: {drift_run_id}")

    # -------------------------------------------------------------------------
    # Final Pipeline Summary
    # -------------------------------------------------------------------------
    print("\n[5/5] Pipeline Execution Completed Successfully!")
    print("=" * 75)
    print("  Artifacts & Output Files Generated:")
    print(f"  1. Model Report:       reports/classification_report.txt")
    print(f"  2. Drift CSV Report:   {report_paths['csv_path']}")
    print(f"  3. Drift JSON Summary: {report_paths['json_path']}")
    print(f"  4. Drift Distributions:{viz_paths['distribution_plot_path']}")
    print(f"  5. Feature Drift Plot: {viz_paths['metrics_plot_path']}")
    print(f"  6. MLflow Tracking:    Experiment '{EXPERIMENT_NAME}' (Local mlruns/)")
    print("=" * 75)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[FATAL ERROR] Pipeline aborted: {str(exc)}", file=sys.stderr)
        sys.exit(1)
