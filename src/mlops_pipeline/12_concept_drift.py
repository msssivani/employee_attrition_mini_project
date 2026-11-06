
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from api.model_loading import ModelLoader
from api.preprocessing import clean_prediction_data


# ---------------------------------------------------------
# FILE PATHS
# ---------------------------------------------------------

BASELINE_X_PATH = Path("data/processed/X_test_raw.csv")
BASELINE_Y_PATH = Path("data/processed/y_test.npy")

EVOLVED_PATH = Path(
    "data/monitoring/drift/evolved_labeled_data.csv"
)

REPORT_PATH = Path(
    "outputs/drift/concept_drift_report.csv"
)


# ---------------------------------------------------------
# EVALUATE MODEL
# ---------------------------------------------------------

def evaluate(model_loader, X, y):

    # Clean raw employee data
    X_clean, employee_numbers = clean_prediction_data(X)

    # Apply the saved preprocessing pipeline
    X_transformed = model_loader.preprocessor.transform(X_clean)

    # Generate predictions
    predictions = model_loader.model.predict(X_transformed)

    # Generate probabilities
    probabilities = model_loader.model.predict_proba(
        X_transformed
    )[:, 1]

    # Calculate evaluation metrics
    metrics = {
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(
            y, predictions, zero_division=0
        ),
        "recall": recall_score(
            y, predictions, zero_division=0
        ),
        "f1_score": f1_score(
            y, predictions, zero_division=0
        ),
    }

    # ROC-AUC requires both classes to be present
    if len(np.unique(y)) == 2:
        metrics["roc_auc"] = roc_auc_score(
            y, probabilities
        )
    else:
        metrics["roc_auc"] = np.nan

    return metrics


# ---------------------------------------------------------
# MAIN FUNCTION
# ---------------------------------------------------------

def main():

    print("=" * 70)
    print("EMPLOYEE ATTRITION - CONCEPT DRIFT ANALYSIS")
    print("=" * 70)

    # Load baseline test data
    baseline_X = pd.read_csv(BASELINE_X_PATH)
    baseline_y = np.load(BASELINE_Y_PATH).astype(int)

    # Load evolved labeled data
    evolved = pd.read_csv(EVOLVED_PATH)

    if "Attrition" not in evolved.columns:
        raise ValueError(
            "Attrition target column not found in evolved dataset."
        )

    evolved_X = evolved.drop(columns=["Attrition"])
    evolved_y = evolved["Attrition"].to_numpy(dtype=int)

    # Validate feature and label counts
    if len(baseline_X) != len(baseline_y):
        raise ValueError(
            "Baseline feature and label counts do not match."
        )

    if len(evolved_X) != len(evolved_y):
        raise ValueError(
            "Evolved feature and label counts do not match."
        )

    if list(baseline_X.columns) != list(evolved_X.columns):
        raise ValueError(
            "Feature columns in baseline and evolved datasets do not match."
        )

    if not set(np.unique(baseline_y)).issubset({0, 1}):
        raise ValueError("Invalid baseline Attrition labels.")

    if not set(np.unique(evolved_y)).issubset({0, 1}):
        raise ValueError("Invalid evolved Attrition labels.")

    print("\nDATASET INFORMATION")
    print("-" * 70)

    print(f"Baseline samples : {len(baseline_X)}")
    print(f"Evolved samples  : {len(evolved_X)}")

    print("\nBaseline Attrition distribution:")
    print(pd.Series(baseline_y).value_counts().sort_index())

    print("\nEvolved Attrition distribution:")
    print(pd.Series(evolved_y).value_counts().sort_index())

    # Load production model and preprocessing object
    model_loader = ModelLoader()
    model_loader.load()

    print("\nMODEL INFORMATION")
    print("-" * 70)

    print(f"Model name    : {model_loader.model_name}")
    print(f"Model version : {model_loader.model_version}")

    # Evaluate baseline dataset
    baseline_metrics = evaluate(
        model_loader,
        baseline_X,
        baseline_y,
    )

    # Evaluate evolved dataset
    evolved_metrics = evaluate(
        model_loader,
        evolved_X,
        evolved_y,
    )

    # Create comparison report
    report_rows = []

    for metric in baseline_metrics:

        baseline_value = baseline_metrics[metric]
        evolved_value = evolved_metrics[metric]

        report_rows.append({
            "metric": metric,
            "baseline": baseline_value,
            "evolved": evolved_value,
            "change": evolved_value - baseline_value,
        })

    report = pd.DataFrame(report_rows)

    # Save report
    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report.to_csv(REPORT_PATH, index=False)

    print("\nCONCEPT DRIFT PERFORMANCE COMPARISON")
    print("-" * 70)

    print(report.to_string(index=False))

    print("\nREPORT SAVED")
    print("-" * 70)
    print(REPORT_PATH)

    print("\n" + "=" * 70)
    print("CONCEPT DRIFT ANALYSIS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
