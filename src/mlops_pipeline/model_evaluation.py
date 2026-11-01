
import json
import os
import sys

import joblib
import numpy as np
import yaml

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def run_evaluation():
    print("[INFO] Starting Employee Attrition Model Evaluation...")

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_path = config["paths"]["model"]
    processed_dir = config["data"]["processed_dir"]

    # Check whether the trained model exists
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found at {model_path}.")
        print("[INFO] Run File 5 (model training) first.")
        return False

    # Load test data
    try:
        X_test = np.load(
            os.path.join(processed_dir, "X_test_final.npy")
        )
        y_test = np.load(
            os.path.join(processed_dir, "y_test.npy")
        )

    except FileNotFoundError as exc:
        print(f"[ERROR] Test data not found: {exc}")
        print("[INFO] Run preprocessing first.")
        return False

    # Load trained model
    model = joblib.load(model_path)

    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    # Calculate evaluation metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_test, y_pred, zero_division=0
        ),
        "f1_score": f1_score(
            y_test, y_pred, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    # Display evaluation results
    print("\n--- Employee Attrition Model Evaluation ---")

    for name, value in metrics.items():
        print(f"{name.replace('_', ' ').title():10}: {value:.4f}")

    print(f"Confusion Matrix:\n{cm}")
    print("------------------------------------------\n")

    # Generate evaluation report
    report = {
        "stage": "Employee Attrition - Model Evaluation",
        "dataset": "HR Employee Attrition",
        "evaluated_on": "held-out test split",
        "test_rows": int(len(y_test)),
        "metrics": {
            name: round(float(value), 4)
            for name, value in metrics.items()
        },
        "confusion_matrix": cm.tolist(),
    }

    # Save evaluation report
    report_path = config["paths"]["evaluation_report"]

    report_dir = os.path.dirname(report_path)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print(f"[SUCCESS] Evaluation report saved to {report_path}")

    return True


if __name__ == "__main__":
    passed = run_evaluation()
    sys.exit(0 if passed else 1)
