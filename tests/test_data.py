"""
Unit tests for data loading and preprocessing.
"""

import pytest
import pandas as pd
from src.data_loader import load_iris_data, split_data, prepare_drift_datasets

EXPECTED_FEATURES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]


def test_iris_data_loading():
    """Verify that the Iris dataset loads with expected shape and metadata."""
    X, y, metadata = load_iris_data()

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.shape == (150, 4)
    assert len(y) == 150
    assert metadata["num_records"] == 150
    assert metadata["num_features"] == 4
    assert metadata["missing_values"] == 0
    assert len(metadata["unique_classes"]) == 3


def test_expected_features_exist():
    """Verify that all expected feature columns are present."""
    X, _, _ = load_iris_data()
    for col in EXPECTED_FEATURES:
        assert col in X.columns, f"Expected column '{col}' missing from Iris dataset."


def test_no_missing_values():
    """Confirm zero null or NaN values in the dataset."""
    X, y, _ = load_iris_data()
    assert X.isnull().sum().sum() == 0
    assert y.isnull().sum() == 0


def test_stratified_split_proportions():
    """Verify train_test_split preserves stratification and proportions."""
    X, y, _ = load_iris_data()
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

    assert len(X_train) == 120
    assert len(X_test) == 30
    assert len(y_train) == 120
    assert len(y_test) == 30

    # Stratification check: Each of the 3 classes should have equal distribution
    assert set(y_train.value_counts()) == {40}
    assert set(y_test.value_counts()) == {10}


def test_empty_input_handling_in_split():
    """Ensure split_data raises ValueError when given empty data."""
    empty_df = pd.DataFrame()
    empty_s = pd.Series(dtype=int)
    with pytest.raises(ValueError):
        split_data(empty_df, empty_s)


def test_drift_dataset_preparation():
    """Verify reference and current dataset generation for drift."""
    X, _, _ = load_iris_data()
    ref_df, curr_df, info = prepare_drift_datasets(
        X,
        shift_feature="petal length (cm)",
        shift_magnitude=1.8,
        sample_fraction=0.5,
        random_state=42
    )

    assert len(ref_df) == 75
    assert len(curr_df) == 75
    # The non-shifted features should remain identical
    assert (ref_df["sepal length (cm)"] == curr_df["sepal length (cm)"]).all()
    # The shifted feature should differ by shift_magnitude
    diff = curr_df["petal length (cm)"] - ref_df["petal length (cm)"]
    assert pytest.approx(diff.iloc[0], 0.001) == 1.8
