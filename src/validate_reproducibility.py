import os
import json
import numpy as np

from sklearn.svm import SVC
from sklearn.metrics import f1_score


def run_deterministic_test():

    print(
        "Running Reproducibility Validation..."
    )

    # ---------------------------------------------------------
    # 1. Load Processed Data
    # ---------------------------------------------------------

    X_train = np.load(
        'data/processed/X_train_final.npy'
    )

    X_test = np.load(
        'data/processed/X_test_final.npy'
    )

    y_train = np.load(
        'data/processed/y_train.npy'
    )

    y_test = np.load(
        'data/processed/y_test.npy'
    )

    # ---------------------------------------------------------
    # 2. Fixed Parameters
    # ---------------------------------------------------------

    params = {
        "C": 1.0,
        "kernel": "rbf",
        "class_weight": "balanced",
        "probability": True,
        "random_state": 42
    }

    # ---------------------------------------------------------
    # 3. Execution 1
    # ---------------------------------------------------------

    model_1 = SVC(
        **params
    )

    model_1.fit(
        X_train,
        y_train
    )

    predictions_1 = model_1.predict(
        X_test
    )

    score_1 = f1_score(
        y_test,
        predictions_1
    )

    # ---------------------------------------------------------
    # 4. Execution 2
    # ---------------------------------------------------------

    model_2 = SVC(
        **params
    )

    model_2.fit(
        X_train,
        y_train
    )

    predictions_2 = model_2.predict(
        X_test
    )

    score_2 = f1_score(
        y_test,
        predictions_2
    )

    # ---------------------------------------------------------
    # 5. Validate Consistency
    # ---------------------------------------------------------

    predictions_match = np.array_equal(
        predictions_1,
        predictions_2
    )

    scores_match = np.isclose(
        score_1,
        score_2
    )

    is_reproducible = (
        predictions_match and scores_match
    )

    # ---------------------------------------------------------
    # 6. Generate Report
    # ---------------------------------------------------------

    report = {

        "test_name":
            "Pipeline Reproducibility Validation",

        "model":
            "Balanced SVM",

        "parameters":
            params,

        "execution_1_f1":
            float(score_1),

        "execution_2_f1":
            float(score_2),

        "predictions_match":
            bool(predictions_match),

        "scores_match":
            bool(scores_match),

        "is_strictly_reproducible":
            bool(is_reproducible),

        "status":
            "PASSED"
            if is_reproducible
            else "FAILED"
    }

    # ---------------------------------------------------------
    # 7. Save Report
    # ---------------------------------------------------------

    os.makedirs(
        "artifacts",
        exist_ok=True
    )

    with open(
        'artifacts/reproducibility_report.json',
        'w'
    ) as f:

        json.dump(
            report,
            f,
            indent=4
        )

    # ---------------------------------------------------------
    # 8. Print Results
    # ---------------------------------------------------------

    print(
        f"Execution 1 F1: {score_1:.6f}"
    )

    print(
        f"Execution 2 F1: {score_2:.6f}"
    )

    print(
        f"Predictions Match: {predictions_match}"
    )

    if is_reproducible:

        print(
            "SUCCESS: Pipeline is reproducible."
        )

        print(
            "Report saved to "
            "artifacts/reproducibility_report.json"
        )

    else:

        print(
            "FAILED: Pipeline is not reproducible."
        )


if __name__ == "__main__":

    run_deterministic_test()