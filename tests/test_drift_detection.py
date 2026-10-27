"""
Unit tests for statistical data drift detection and reporting.
"""

from pathlib import Path
import json
import pytest
import pandas as pd
from src.data_loader import load_iris_data, prepare_drift_datasets
from src.drift_detection import (
    compute_kolmogorov_smirnov_drift,
    save_drift_reports
)


@pytest.fixture
def iris_features():
    X, _, _ = load_iris_data()
    return X


def test_drift_detection_returns_valid_statistics(iris_features):
    """Verify KS statistic is in [0, 1] and p-value is in [0, 1]."""
    ref_df = iris_features.sample(n=70, random_state=42)
    curr_df = iris_features.sample(n=70, random_state=99)

    drift_df = compute_kolmogorov_smirnov_drift(ref_df, curr_df, alpha=0.05)

    assert isinstance(drift_df, pd.DataFrame)
    assert len(drift_df) == 4

    for _, row in drift_df.iterrows():
        assert 0.0 <= row["ks_statistic"] <= 1.0, f"Invalid KS stat: {row['ks_statistic']}"
        assert 0.0 <= row["p_value"] <= 1.0, f"Invalid p-value: {row['p_value']}"
        assert bool(row["drift_detected"]) in (True, False)


def test_identical_distributions_do_not_trigger_drift(iris_features):
    """Verify comparing identical distributions yields p-value ~ 1.0 and NO drift."""
    ref_df = iris_features.copy()
    curr_df = iris_features.copy()

    drift_df = compute_kolmogorov_smirnov_drift(ref_df, curr_df, alpha=0.05)

    for _, row in drift_df.iterrows():
        assert row["drift_detected"] == False, (
            f"False positive drift flagged on identical data for {row['feature']}"
        )
        assert row["p_value"] > 0.05
        assert pytest.approx(row["ks_statistic"], 0.001) == 0.0


def test_controlled_shift_is_detected(iris_features):
    """Verify that an intentional shift on 'petal length (cm)' is successfully detected."""
    ref_df, curr_df, _ = prepare_drift_datasets(
        iris_features,
        shift_feature="petal length (cm)",
        shift_magnitude=1.8,
        sample_fraction=0.5,
        random_state=42
    )

    drift_df = compute_kolmogorov_smirnov_drift(ref_df, curr_df, alpha=0.05)

    petal_row = drift_df[drift_df["feature"] == "petal length (cm)"].iloc[0]
    assert petal_row["drift_detected"] == True, "Expected drift on shifted feature 'petal length (cm)'"
    assert petal_row["p_value"] < 0.05, f"Expected p-value < 0.05, got {petal_row['p_value']}"
    assert petal_row["ks_statistic"] > 0.4, f"Expected prominent KS stat, got {petal_row['ks_statistic']}"

    # Verify unshifted features are NOT flagged
    sepal_len_row = drift_df[drift_df["feature"] == "sepal length (cm)"].iloc[0]
    assert sepal_len_row["drift_detected"] == False


def test_invalid_and_empty_inputs():
    """Verify appropriate exceptions for empty datasets or bad alpha."""
    empty_df = pd.DataFrame()
    valid_df = pd.DataFrame({"col1": [1.0, 2.0, 3.0]})

    with pytest.raises(ValueError, match="Cannot perform drift analysis on empty DataFrames"):
        compute_kolmogorov_smirnov_drift(empty_df, valid_df)

    with pytest.raises(ValueError, match="Significance threshold alpha must be between 0 and 1"):
        compute_kolmogorov_smirnov_drift(valid_df, valid_df, alpha=1.5)

    with pytest.raises(ValueError, match="Significance threshold alpha must be between 0 and 1"):
        compute_kolmogorov_smirnov_drift(valid_df, valid_df, alpha=-0.05)


def test_output_reports_contain_expected_columns(iris_features, tmp_path):
    """Verify saved CSV report has required schema and JSON summary is valid."""
    ref_df, curr_df, meta = prepare_drift_datasets(iris_features, random_state=42)
    drift_df = compute_kolmogorov_smirnov_drift(ref_df, curr_df, alpha=0.05)

    paths = save_drift_reports(drift_df, meta, output_dir=str(tmp_path))

    csv_file = Path(paths["csv_path"])
    json_file = Path(paths["json_path"])

    assert csv_file.exists()
    assert json_file.exists()

    loaded_csv = pd.read_csv(csv_file)
    expected_cols = ["feature", "ks_statistic", "p_value", "alpha_threshold", "drift_detected"]
    for col in expected_cols:
        assert col in loaded_csv.columns, f"Expected column '{col}' missing from drift CSV"

    with open(json_file, "r", encoding="utf-8") as f:
        loaded_json = json.load(f)

    assert "total_features_evaluated" in loaded_json
    assert "drifted_features_count" in loaded_json
    assert "statistical_interpretation" in loaded_json
    assert loaded_json["total_features_evaluated"] == 4
