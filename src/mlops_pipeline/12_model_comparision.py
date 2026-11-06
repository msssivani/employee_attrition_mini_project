
import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from api.model_loading import ModelLoader
from api.preprocessing import clean_prediction_data


# ---------------------------------------------------------
# FILE PATHS
# ---------------------------------------------------------

TEST_FEATURES_PATH = Path(
    "data/processed/X_test_raw.csv"
)

TEST_LABELS_PATH = Path(
    "data/processed/y_test.npy"
)

V2_MODEL_PATH = Path(
    "models/retrained_model_v2.pkl"
)

V2_PREPROCESSOR_PATH = Path(
    "models/retrained_preprocessor_v2.pkl"
)

REPORT_PATH = Path(
    "outputs/retraining/model_comparison_report.csv"
)

JSON_REPORT_PATH = Path(
    "outputs/retraining/model_comparison_report.json"
)


# Existing project quality gate: recall >= 0.65
RECALL_THRESHOLD = 0.65


# ---------------------------------------------------------
# EVALUATE MODEL
# ---------------------------------------------------------

def evaluate(model, preprocessor, X_test, y_test):

    # The cleaner returns (cleaned_dataframe, employee_numbers).
    X_clean, employee_numbers = clean_prediction_data(
        X_test.copy()
    )

    X_transformed = preprocessor.transform(X_clean)

    predictions = model.predict(X_transformed)

    probabilities = model.predict_proba(
        X_transformed
    )[:, 1]

    metrics = {
        "accuracy": accuracy_score(
            y_test,
            predictions,
        ),

        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),

        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),

        "f1_score": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
    }

    # ROC-AUC requires both classes in the evaluation labels.
    if len(np.unique(y_test)) == 2:
        metrics["roc_auc"] = roc_auc_score(
            y_test,
            probabilities,
        )
    else:
        metrics["roc_auc"] = np.nan

    return metrics


# ---------------------------------------------------------
# MAIN FUNCTION
# ---------------------------------------------------------

def main():

    print("=" * 70)
    print("EMPLOYEE ATTRITION - MODEL V1 VS V2 COMPARISON")
    print("=" * 70)

    # Load held-out test data
    X_test = pd.read_csv(TEST_FEATURES_PATH)
    y_test = np.load(TEST_LABELS_PATH).astype(int)

    if len(X_test) != len(y_test):
        raise ValueError(
            "Test feature/label count mismatch."
        )

    if not set(np.unique(y_test)).issubset({0, 1}):
        raise ValueError("Invalid Attrition test labels.")

    print("\nEVALUATION DATASET")
    print("-" * 70)

    print(f"Features : {X_test.shape}")
    print(f"Labels   : {y_test.shape}")

    # -----------------------------------------------------
    # LOAD MODEL V1
    # -----------------------------------------------------

    print("\nLOADING MODEL V1")
    print("-" * 70)

    model_loader = ModelLoader()
    model_loader.load()

    v1_model = model_loader.model
    v1_preprocessor = model_loader.preprocessor

    print(f"Model name    : {model_loader.model_name}")
    print(f"Model version : {model_loader.model_version}")
    print(f"Run ID        : {model_loader.run_id}")

    # -----------------------------------------------------
    # LOAD MODEL V2
    # -----------------------------------------------------

    print("\nLOADING MODEL V2")
    print("-" * 70)

    if not V2_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"V2 model not found: {V2_MODEL_PATH}"
        )

    if not V2_PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            f"V2 preprocessor not found: {V2_PREPROCESSOR_PATH}"
        )

    v2_model = joblib.load(V2_MODEL_PATH)

    v2_preprocessor = joblib.load(
        V2_PREPROCESSOR_PATH
    )

    print(f"Model artifact        : {V2_MODEL_PATH}")
    print(f"Preprocessor artifact : {V2_PREPROCESSOR_PATH}")
    print("Model version         : V2")

    # -----------------------------------------------------
    # EVALUATE MODEL V1
    # -----------------------------------------------------

    print("\nEVALUATING MODEL V1")
    print("-" * 70)

    v1_metrics = evaluate(
        v1_model,
        v1_preprocessor,
        X_test,
        y_test,
    )

    for metric, value in v1_metrics.items():
        print(f"{metric.upper():10}: {value:.4f}")

    # -----------------------------------------------------
    # EVALUATE MODEL V2
    # -----------------------------------------------------

    print("\nEVALUATING MODEL V2")
    print("-" * 70)

    v2_metrics = evaluate(
        v2_model,
        v2_preprocessor,
        X_test,
        y_test,
    )

    for metric, value in v2_metrics.items():
        print(f"{metric.upper():10}: {value:.4f}")

    # -----------------------------------------------------
    # COMPARE MODELS
    # -----------------------------------------------------

    comparison = []

    for metric in v1_metrics:

        v1_value = float(v1_metrics[metric])
        v2_value = float(v2_metrics[metric])

        change = (
            v2_value - v1_value
            if not (
                np.isnan(v1_value) or np.isnan(v2_value)
            )
            else np.nan
        )

        comparison.append({
            "metric": metric,
            "model_v1": v1_value,
            "model_v2": v2_value,
            "change_v2_minus_v1": change,
        })

    comparison_df = pd.DataFrame(comparison)

    # -----------------------------------------------------
    # QUALITY GATE
    # -----------------------------------------------------

    v1_recall_pass = (
        v1_metrics["recall"] >= RECALL_THRESHOLD
    )

    v2_recall_pass = (
        v2_metrics["recall"] >= RECALL_THRESHOLD
    )

    v2_recall_improved = (
        v2_metrics["recall"] > v1_metrics["recall"]
    )

    print("\n" + "=" * 70)
    print("MODEL V1 VS V2 COMPARISON")
    print("=" * 70)

    print(comparison_df.to_string(index=False))

    print("\nQUALITY CHECKS")
    print("-" * 70)

    print(f"Recall threshold    : {RECALL_THRESHOLD:.2f}")

    print(
        f"V1 recall threshold : "
        f"{'PASS' if v1_recall_pass else 'FAIL'}"
    )

    print(
        f"V2 recall threshold : "
        f"{'PASS' if v2_recall_pass else 'FAIL'}"
    )

    print(
        f"V2 recall vs V1     : "
        f"{'IMPROVED' if v2_recall_improved else 'NOT IMPROVED'}"
    )

    # A candidate should meet the recall gate and improve on V1.
    v2_accepted = (
        v2_recall_pass and v2_recall_improved
    )

    print(
        f"V2 quality decision : "
        f"{'PASS' if v2_accepted else 'FAIL'}"
    )

    # -----------------------------------------------------
    # SAVE CSV REPORT
    # -----------------------------------------------------

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison_df.to_csv(
        REPORT_PATH,
        index=False,
    )

    # -----------------------------------------------------
    # SAVE JSON REPORT
    # -----------------------------------------------------

    json_report = {
        "stage": "Employee Attrition - Model Comparison",

        "evaluation_dataset": str(
            TEST_FEATURES_PATH
        ),

        "test_rows": int(len(X_test)),

        "model_v1": {
            "version": str(model_loader.model_version),
            "run_id": str(model_loader.run_id),
            "metrics": {
                key: (
                    None if np.isnan(value)
                    else round(float(value), 6)
                )
                for key, value in v1_metrics.items()
            },
        },

        "model_v2": {
            "version": "V2",
            "metrics": {
                key: (
                    None if np.isnan(value)
                    else round(float(value), 6)
                )
                for key, value in v2_metrics.items()
            },
        },

        "comparison": comparison,

        "quality_gate": {
            "metric": "recall",
            "threshold": RECALL_THRESHOLD,
            "v1_passed": bool(v1_recall_pass),
            "v2_passed": bool(v2_recall_pass),
            "v2_improved_over_v1": bool(v2_recall_improved),
            "v2_accepted": bool(v2_accepted),
        },
    }

    with open(
        JSON_REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            json_report,
            f,
            indent=4,
        )

    print("\n" + "=" * 70)
    print("COMPARISON REPORTS SAVED")
    print("=" * 70)

    print(f"CSV  : {REPORT_PATH}")
    print(f"JSON : {JSON_REPORT_PATH}")

    print("=" * 70)


if __name__ == "__main__":
    main()
