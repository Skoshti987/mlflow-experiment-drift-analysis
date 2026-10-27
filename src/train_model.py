"""
Model Training and Evaluation Module.

Trains a Random Forest Classifier on Iris features, evaluates performance
using standard classification metrics (accuracy, precision, recall, weighted F1),
and produces detailed diagnostic summaries.
"""

from typing import Dict, Any, Tuple
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


def train_and_evaluate_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    n_estimators: int = 100,
    random_state: int = 42,
    criterion: str = "gini",
    max_depth: int | None = None
) -> Tuple[RandomForestClassifier, Dict[str, float], Dict[str, Any]]:
    """
    Train a Random Forest Classifier and evaluate its performance.

    Args:
        X_train: Training features.
        X_test: Testing features.
        y_train: Training target labels.
        y_test: Testing target labels.
        n_estimators: Number of trees in forest.
        random_state: Seed for reproducible tree generation.
        criterion: Splitting metric ("gini" or "entropy").
        max_depth: Maximum tree depth.

    Returns:
        model: Fitted RandomForestClassifier instance.
        metrics: Dictionary containing accuracy, precision, recall, f1_weighted.
        evaluation_details: Dictionary containing classification report string and parameters.
    """
    if X_train.empty or X_test.empty or y_train.empty or y_test.empty:
        raise ValueError("Training or testing inputs cannot be empty.")

    try:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            criterion=criterion,
            max_depth=max_depth
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = float(accuracy_score(y_test, y_pred))
        precision = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
        recall = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
        f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

        report_str = classification_report(y_test, y_pred, digits=4, zero_division=0)

        metrics = {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_weighted": round(f1_weighted, 4)
        }

        hyperparameters = {
            "n_estimators": n_estimators,
            "random_state": random_state,
            "criterion": criterion,
            "max_depth": "None" if max_depth is None else str(max_depth)
        }

        evaluation_details = {
            "classification_report": report_str,
            "hyperparameters": hyperparameters,
            "test_sample_count": len(y_test)
        }

        return model, metrics, evaluation_details

    except Exception as e:
        raise RuntimeError(f"Error during model training and evaluation: {str(e)}") from e
