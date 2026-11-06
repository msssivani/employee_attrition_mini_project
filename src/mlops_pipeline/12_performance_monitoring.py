
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from api.model_loading import ModelLoader
from api.preprocessing import clean_prediction_data


# ============================================================
# 1. File paths
# ============================================================

X_TEST_PATH = Path("data/processed/X_test_raw.csv")
Y_TEST_PATH = Path("data/processed/y_test.npy")


# ============================================================
# 2. Load test data
# ============================================================

def load_test_data():

    X_test = pd.read_csv(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH).astype(int)

    if len(X_test) != len(y_test):
        raise ValueError(
            f"X_test and y_test size mismatch: "
            f"{len(X_test)} != {len(y_test)}"
        )

    return X_test, y_test


# ============================================================
# 3. Evaluate production model
# ============================================================

def evaluate_model(model_loader, X_test, y_test):

    # Clean raw input data
    X_clean, employee_numbers = clean_prediction_data(X_test)

    # Apply saved preprocessing pipeline
    X_transformed = model_loader.preprocessor.transform(X_clean)

    # Generate predictions
    predictions = model_loader.model.predict(X_transformed)

    # Generate attrition probabilities
    probabilities = model_loader.model.predict_proba(
        X_transformed
    )[:, 1]

    # Calculate performance metrics
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),

        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "f1": f1_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "roc_auc": roc_auc_score(
            y_test,
            probabilities
        )
    }

    return metrics


# ============================================================
# 4. Display monitoring results
# ============================================================

def print_results(metrics, model_loader, X_test, y_test):

    print("\n" + "=" * 65)
    print("EMPLOYEE ATTRITION - MODEL PERFORMANCE MONITORING")
    print("=" * 65)

    print("\nEvaluation dataset : X_test_raw.csv")
    print(f"Test samples       : {len(X_test)}")

    print(
        f"Actual attrition   : {int(y_test.sum())}"
    )

    print(
        f"Actual no attrition: {int((y_test == 0).sum())}"
    )

    print("\nModel Information")
    print("-" * 65)

    print(f"Model name    : {model_loader.model_name}")
    print(f"Model version : {model_loader.model_version}")
    print(f"Run ID        : {model_loader.run_id}")

    print("\nPerformance Metrics")
    print("-" * 65)

    for metric, value in metrics.items():
        print(f"{metric.capitalize():<12}: {value:.4f}")

    print("\nQuality Gate")
    print("-" * 65)

    print(f"Recall threshold : 0.65")
    print(
        f"Recall status    : "
        f"{'PASS' if metrics['recall'] >= 0.65 else 'FAIL'}"
    )

    print("=" * 65)


# ============================================================
# 5. Main function
# ============================================================

def main():

    print("[INFO] Starting Model Performance Monitoring...")

    # Load test dataset
    X_test, y_test = load_test_data()

    # Load production model and preprocessor
    model_loader = ModelLoader()
    model_loader.load()

    if not model_loader.is_ready():
        raise RuntimeError(
            "Production model or preprocessor is not available"
        )

    # Evaluate model
    metrics = evaluate_model(
        model_loader,
        X_test,
        y_test
    )

    # Display results
    print_results(
        metrics,
        model_loader,
        X_test,
        y_test
    )

    print("\n[SUCCESS] Model performance monitoring completed!")


if __name__ == "__main__":
    main()
