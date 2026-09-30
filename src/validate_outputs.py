import os
import json
import numpy as np


def validate_preprocessing_outputs():

    print(
        "[INFO] Validating Employee Attrition Preprocessing Outputs..."
    )

    # ---------------------------------------------------------
    # 1. Load Preprocessing Outputs
    # ---------------------------------------------------------

    try:

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

    except FileNotFoundError as e:

        print(
            f"[ERROR] Could not find processed data: {e}"
        )

        return False

    errors = []

    # ---------------------------------------------------------
    # 2. Check for Missing Values
    # ---------------------------------------------------------

    if np.isnan(X_train).sum() > 0:

        errors.append(
            "NaNs detected in X_train after preprocessing."
        )

    if np.isnan(X_test).sum() > 0:

        errors.append(
            "NaNs detected in X_test after preprocessing."
        )

    # ---------------------------------------------------------
    # 3. Check Dimensionality Consistency
    # ---------------------------------------------------------

    if X_train.shape[1] != X_test.shape[1]:

        errors.append(
            f"Feature mismatch: "
            f"X_train has {X_train.shape[1]} columns, "
            f"X_test has {X_test.shape[1]} columns."
        )

    if X_train.shape[0] != y_train.shape[0]:

        errors.append(
            "Row mismatch between X_train and y_train."
        )

    if X_test.shape[0] != y_test.shape[0]:

        errors.append(
            "Row mismatch between X_test and y_test."
        )

    # ---------------------------------------------------------
    # 4. Validate Target Values
    # ---------------------------------------------------------

    valid_targets = {0, 1}

    if not set(np.unique(y_train)).issubset(
        valid_targets
    ):

        errors.append(
            "Unexpected values detected in y_train."
        )

    if not set(np.unique(y_test)).issubset(
        valid_targets
    ):

        errors.append(
            "Unexpected values detected in y_test."
        )

    # ---------------------------------------------------------
    # 5. Generate Validation Report
    # ---------------------------------------------------------

    report = {

        "validation_status":
            "PASSED" if not errors else "FAILED",

        "matrix_dimensions": {

            "X_train_shape":
                list(X_train.shape),

            "X_test_shape":
                list(X_test.shape),

            "y_train_shape":
                list(y_train.shape),

            "y_test_shape":
                list(y_test.shape)
        },

        "data_quality": {

            "missing_values_X_train":
                int(np.isnan(X_train).sum()),

            "missing_values_X_test":
                int(np.isnan(X_test).sum())
        },

        "target_values": {

            "y_train_unique":
                np.unique(y_train).tolist(),

            "y_test_unique":
                np.unique(y_test).tolist()
        },

        "errors": errors
    }

    # ---------------------------------------------------------
    # 6. Save Validation Report
    # ---------------------------------------------------------

    os.makedirs(
        "artifacts",
        exist_ok=True
    )

    report_path = (
        'artifacts/preprocessing_summary_report.json'
    )

    with open(
        report_path,
        'w'
    ) as f:

        json.dump(
            report,
            f,
            indent=4
        )

    # ---------------------------------------------------------
    # 7. Print Validation Result
    # ---------------------------------------------------------

    if errors:

        print(
            "[ERROR] Output Validation FAILED!"
        )

        for error in errors:

            print(
                f"  - {error}"
            )

        return False

    else:

        print(
            "[SUCCESS] Output Validation PASSED."
        )

        print(
            "[INFO] Processed features are clean "
            "and dimensionally consistent."
        )

        print(
            "[INFO] Train/test features and targets "
            "have matching row counts."
        )

        print(
            "[INFO] Target values are valid: 0 and 1."
        )

        print(
            f"[INFO] Summary report saved to "
            f"{report_path}"
        )

        return True


if __name__ == "__main__":

    validate_preprocessing_outputs()