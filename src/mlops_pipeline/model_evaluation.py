
import sys
import json
from pathlib import Path

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


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_evaluation():

    print("[INFO] Starting Employee Attrition Model Evaluation...")

    # Load configuration
    config_path = PROJECT_ROOT / "config.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Resolve model path
    model_path = PROJECT_ROOT / config["paths"]["model"]

    # Resolve processed data directory
    processed_dir = PROJECT_ROOT / config["data"]["processed_dir"]

    # Check model existence
    if not model_path.exists():
        print(f"[ERROR] Model not found at: {model_path}")
        print("[INFO] Run the training and registry stage first.")
        return False

    # Locate test data
    X_test_candidates = [
        "X_test_scaled.npy",
        "X_test.npy",
        "X_test_final.npy",
    ]

    X_test_path = next(
        (
            processed_dir / filename
            for filename in X_test_candidates
            if (processed_dir / filename).exists()
        ),
        None,
    )

    y_test_path = processed_dir / "y_test.npy"

    if X_test_path is None or not y_test_path.exists():
        print("[ERROR] Test data not found.")
        print(f"[INFO] Checked directory: {processed_dir}")
        print(f"[INFO] X_test candidates: {X_test_candidates}")
        print("[INFO] Expected target file: y_test.npy")
        return False

    try:
        X_test = np.load(X_test_path)
        y_test = np.load(y_test_path)

    except Exception as e:
        print(f"[ERROR] Unable to load test data: {e}")
        return False

    # Load trained Employee Attrition model
    try:
        model = joblib.load(model_path)

    except Exception as e:
        print(f"[ERROR] Unable to load trained model: {e}")
        return False

    print(f"[INFO] Loaded model: {model_path}")
    print(f"[INFO] Test features: {X_test_path}")
    print(f"[INFO] Test samples: {len(y_test)}")

    # Check feature dimensions
    if X_test.shape[0] != len(y_test):
        print("[ERROR] X_test and y_test have different sample counts.")
        return False

    try:
        # Generate predictions
        y_pred = model.predict(X_test)

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
        }

        # ROC-AUC using predicted probabilities
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
        else:
            print("[WARNING] Model does not support predict_proba.")
            metrics["roc_auc"] = None

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    except Exception as e:
        print(f"[ERROR] Model evaluation failed: {e}")
        return False

    # Display evaluation results
    print("\n--- Employee Attrition Model Evaluation ---")

    for name, value in metrics.items():
        if value is not None:
            print(f"{name.replace('_', ' ').title():12}: {value:.4f}")
        else:
            print(f"{name.replace('_', ' ').title():12}: N/A")

    print("\nConfusion Matrix:")
    print(cm)
    print("-------------------------------------------")

    # Prepare evaluation report
    report = {
        "stage": "Employee Attrition - Model Evaluation",
        "evaluated_on": "held-out test split",
        "model": str(model_path.relative_to(PROJECT_ROOT)),
        "test_features": str(X_test_path.relative_to(PROJECT_ROOT)),
        "test_rows": int(len(y_test)),
        "metrics": {
            name: (
                round(float(value), 4)
                if value is not None
                else None
            )
            for name, value in metrics.items()
        },
        "confusion_matrix": cm.tolist(),
        "class_labels": {
            "0": "No Attrition",
            "1": "Attrition",
        },
    }

    # Save evaluation report
    report_path = PROJECT_ROOT / config["paths"]["evaluation_report"]

    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print(f"\n[SUCCESS] Evaluation report saved to: {report_path}")

    return True


if __name__ == "__main__":
    passed = run_evaluation()
    sys.exit(0 if passed else 1)
