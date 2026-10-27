"""
Data Loader and Preprocessing Module for Iris Dataset.

This module provides reproducible loading, validation, stratified splitting,
and baseline/current dataset generation for data drift evaluation.
"""

from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def load_iris_data() -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    Load the standard Iris dataset from scikit-learn and validate dataset integrity.

    Returns:
        X (pd.DataFrame): 4 numerical features.
        y (pd.Series): Target class labels (0, 1, 2).
        metadata (Dict[str, Any]): Dataset shape, feature names, target names,
                                   and null check results.

    Raises:
        RuntimeError: If data loading fails or dataset validation fails.
    """
    try:
        raw_iris = load_iris(as_frame=True)
        X: pd.DataFrame = raw_iris.data.copy()
        y: pd.Series = raw_iris.target.copy()
        target_names = list(raw_iris.target_names)
        feature_names = list(raw_iris.feature_names)

        # Dataset Integrity Verification
        num_records, num_features = X.shape
        missing_values_count = int(X.isnull().sum().sum())
        unique_classes = sorted(y.unique().tolist())

        if num_records != 150 or num_features != 4:
            raise ValueError(f"Unexpected Iris dataset shape: ({num_records}, {num_features}). Expected (150, 4).")

        if missing_values_count > 0:
            raise ValueError(f"Found {missing_values_count} missing values in standard Iris dataset.")

        if len(unique_classes) != 3:
            raise ValueError(f"Expected 3 target classes, found {len(unique_classes)}.")

        metadata = {
            "dataset_name": "Iris Flower Dataset",
            "num_records": num_records,
            "num_features": num_features,
            "feature_names": feature_names,
            "target_names": target_names,
            "unique_classes": unique_classes,
            "missing_values": missing_values_count,
        }

        return X, y, metadata

    except Exception as e:
        raise RuntimeError(f"Error loading and validating Iris dataset: {str(e)}") from e


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataset into training and testing subsets using stratified sampling.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        test_size: Proportion of dataset for test set (default: 0.2).
        random_state: Random seed for reproducibility (default: 42).

    Returns:
        X_train, X_test, y_train, y_test
    """
    if X.empty or y.empty:
        raise ValueError("Cannot split empty feature DataFrame or target Series.")

    if len(X) != len(y):
        raise ValueError(f"Feature count ({len(X)}) does not match label count ({len(y)}).")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    return X_train, X_test, y_train, y_test


def prepare_drift_datasets(
    X: pd.DataFrame,
    shift_feature: str = "petal length (cm)",
    shift_magnitude: float = 1.8,
    sample_fraction: float = 0.5,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Create reference and current datasets for drift detection.

    Note on Simulation:
    The reference dataset represents the baseline feature distribution.
    The current dataset is created by copying the reference baseline and
    injecting a controlled synthetic offset into one numerical feature
    ('petal length (cm)') to simulate environmental sensor shift or seasonal drift.
    This shift is strictly simulated for verification and educational demonstration.

    Args:
        X: Original feature DataFrame.
        shift_feature: Feature name to apply controlled shift to.
        shift_magnitude: Value added to shift_feature (default: +1.8 cm).
        sample_fraction: Fraction of dataset sampled for reference baseline.
        random_state: Seed for reproducible sampling.

    Returns:
        reference_data (pd.DataFrame): Baseline reference sample.
        current_data (pd.DataFrame): Current sample with controlled shift.
        drift_scenario_info (Dict[str, Any]): Metadata describing the scenario.
    """
    if X.empty:
        raise ValueError("Input DataFrame is empty.")

    if shift_feature not in X.columns:
        raise ValueError(f"Shift feature '{shift_feature}' not found in columns: {list(X.columns)}")

    # Reproducibly select reference sample
    reference_data = X.sample(frac=sample_fraction, random_state=random_state).copy()

    # Create current dataset with controlled distribution shift
    current_data = reference_data.copy()
    current_data[shift_feature] = current_data[shift_feature] + shift_magnitude

    drift_scenario_info = {
        "drift_type": "Covariate Shift (Data Drift / Distribution Shift)",
        "shift_feature": shift_feature,
        "shift_magnitude": shift_magnitude,
        "reference_sample_size": len(reference_data),
        "current_sample_size": len(current_data),
        "simulation_note": (
            "Simulated controlled shift added to "
            f"'{shift_feature}' (+{shift_magnitude:.1f} cm) to test Kolmogorov-Smirnov detection. "
            "Not naturally observed in wild data."
        ),
    }

    return reference_data, current_data, drift_scenario_info
