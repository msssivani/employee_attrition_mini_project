
import json
import os
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from api.preprocessing import clean_prediction_data


# Dataset paths
UPDATED_DATASET_PATH = Path(
    "data/processed/updated_training_data.csv"
)

TEST_FEATURES_PATH = Path(
    "data/processed/X_test_raw.csv"
)

TEST_LABELS_PATH = Path(
    "data/processed/y_test.npy"
)


# Model artifact paths
MODEL_V2_PATH = Path(
    "models/retrained_model_v2.pkl"
)

PREPROCESSOR_V2_PATH = Path(
    "models/retrained_preprocessor_v2.pkl"
)

REPORT_PATH = Path(
    "outputs/retraining/retraining_report.json"
)


# STEP 1: Configure MLflow
def configure_mlflow(config):

    tracking_uri = (
        os.getenv("MLFLOW_TRACKING_URI")
        or config["mlflow"].get("tracking_uri")
    )

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)


# STEP 2: Clean target data
def clean_target_data(series):

    mapping = {
        "yes": 1,
        "no": 0,
        "1": 1,
        "0": 0,
    }

    target = series.astype(str).str.strip().str.lower()

    encoded = target.map(mapping)

    if encoded.isna().any():
        raise ValueError(
            "Unexpected values found in Attrition target column."
        )

    return encoded.astype(int).to_numpy()


# STEP 3: Build preprocessing pipeline
def build_preprocessor(X_train):

    categorical_columns = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numerical_columns = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value=0
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "ohe",
                OneHotEncoder(
                    drop="first",
                    sparse_output=False,
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numerical_pipeline,
                numerical_columns,
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_columns,
            ),
        ]
    )

    return preprocessor


# STEP 4: Evaluate retrained model
def evaluate_model(
    model,
    preprocessor,
    X_test,
    y_test,
):

    X_test_clean,_ = clean_prediction_data(X_test)

    X_test_transformed = preprocessor.transform(
        X_test_clean
    )

    predictions = model.predict(
        X_test_transformed
    )

    probabilities = model.predict_proba(
        X_test_transformed
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

        "roc_auc": (
            roc_auc_score(y_test, probabilities)
            if len(np.unique(y_test)) == 2
            else np.nan
        ),
    }

    return metrics


# STEP 5: Main function
def main():

    print("=" * 70)
    print("LAB 12.13 - EMPLOYEE ATTRITION MODEL RETRAINING")
    print("=" * 70)

    # Load configuration
    with open(
        "config.yaml",
        "r",
        encoding="utf-8"
    ) as f:

        config = yaml.safe_load(f)

    configure_mlflow(config)

    # Check dataset availability
    for path in [
        UPDATED_DATASET_PATH,
        TEST_FEATURES_PATH,
        TEST_LABELS_PATH,
    ]:

        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    # Load datasets
    updated = pd.read_csv(
        UPDATED_DATASET_PATH
    )

    X_test = pd.read_csv(
        TEST_FEATURES_PATH
    )

    y_test = np.load(
        TEST_LABELS_PATH
    ).astype(int)

    print("\nDATASET INFORMATION")
    print("-" * 70)

    print(f"Updated training data : {updated.shape}")
    print(f"Held-out test data    : {X_test.shape}")
    print(f"Test labels           : {y_test.shape}")

    # Check target column
    if "Attrition" not in updated.columns:
        raise ValueError(
            "Updated dataset must contain Attrition column."
        )

    # Separate features and target
    X_train = updated.drop(
        columns=["Attrition"]
    )

    y_train = clean_target_data(
        updated["Attrition"]
    )

    # Validate dataset
    if len(X_train) != len(y_train):
        raise ValueError(
            "Training feature/label count mismatch."
        )

    if len(X_test) != len(y_test):
        raise ValueError(
            "Test feature/label count mismatch."
        )

    if set(X_train.columns) != set(X_test.columns):
        raise ValueError(
            "Training and test feature columns do not match."
        )

    # Maintain training feature order
    X_test = X_test[X_train.columns]

    # Validate target labels
    if not set(np.unique(y_train)).issubset({0, 1}):
        raise ValueError("Invalid training target labels.")

    if not set(np.unique(y_test)).issubset({0, 1}):
        raise ValueError("Invalid test target labels.")

    # STEP 6: Define SVM parameters
    params = {
        "C": 1.0,
        "kernel": "rbf",
        "class_weight": "balanced",
        "probability": True,
        "random_state": 42,
    }

    print("\nMODEL PARAMETERS")
    print("-" * 70)

    for name, value in params.items():
        print(f"{name:15}: {value}")

    # STEP 7: Fit preprocessing pipeline
    print("\nFITTING V2 PREPROCESSOR")
    print("-" * 70)

    X_train_clean,_ = clean_prediction_data(
        X_train
    )

    X_test_clean,_= clean_prediction_data(
        X_test
    )

    preprocessor = build_preprocessor(
        X_train_clean
    )

    X_train_transformed = (
        preprocessor.fit_transform(X_train_clean)
    )

    print(
        f"Original features    : {X_train.shape[1]}"
    )

    print(
        f"Transformed features : "
        f"{X_train_transformed.shape[1]}"
    )

    # STEP 8: Train SVM model
    print("\nTRAINING MODEL V2")
    print("-" * 70)

    model = SVC(**params)

    model.fit(
        X_train_transformed,
        y_train
    )

    print("SVM training completed.")

    # STEP 9: Evaluate model
    print("\nEVALUATING MODEL V2 ON ORIGINAL HELD-OUT TEST SET")
    print("-" * 70)

    metrics = evaluate_model(
        model,
        preprocessor,
        X_test,
        y_test,
    )

    for name, value in metrics.items():

        if pd.notna(value):
            print(f"{name.upper():10}: {value:.4f}")
        else:
            print(f"{name.upper():10}: N/A")

    # STEP 10: Save model artifacts
    MODEL_V2_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    PREPROCESSOR_V2_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_V2_PATH
    )

    joblib.dump(
        preprocessor,
        PREPROCESSOR_V2_PATH
    )

    print("\nARTIFACTS SAVED")
    print("-" * 70)

    print(f"Model V2        : {MODEL_V2_PATH}")
    print(f"Preprocessor V2 : {PREPROCESSOR_V2_PATH}")

    # STEP 11: MLflow experiment tracking
    mlflow.set_experiment(
        config["mlflow"]["experiment_name"]
    )

    with mlflow.start_run(
        run_name="Lab12_Retraining_Employee_Attrition_V2"
    ) as run:

        mlflow.log_params(params)

        mlflow.log_param(
            "model_family",
            "SVM",
        )

        mlflow.log_param(
            "lab",
            "Lab 12 - Continuous Training",
        )

        mlflow.log_param(
            "training_dataset",
            str(UPDATED_DATASET_PATH),
        )

        mlflow.log_param(
            "training_rows",
            len(X_train),
        )

        mlflow.log_param(
            "test_rows",
            len(X_test),
        )

        mlflow.log_param(
            "training_attrition_rate",
            float(y_train.mean()),
        )

        mlflow.log_param(
            "previous_production_model_version",
            "1",
        )

        # Log available metrics
        valid_metrics = {
            key: float(value)
            for key, value in metrics.items()
            if np.isfinite(value)
        }

        mlflow.log_metrics(valid_metrics)

        mlflow.log_artifact(
            str(UPDATED_DATASET_PATH),
            artifact_path="retraining_dataset",
        )

        mlflow.log_artifact(
            str(MODEL_V2_PATH),
            artifact_path="retrained_model",
        )

        mlflow.log_artifact(
            str(PREPROCESSOR_V2_PATH),
            artifact_path="preprocessing_pipeline_v2",
        )

        print("\nMLFLOW RETRAINING RUN")
        print("-" * 70)

        print(
            f"Experiment : "
            f"{config['mlflow']['experiment_name']}"
        )

        print(
            f"Run ID     : {run.info.run_id}"
        )

    # STEP 12: Generate retraining report
    report = {
        "stage": "Lab 12 - Retraining",

        "model_version": "V2",

        "previous_production_version": 1,

        "training_dataset": str(
            UPDATED_DATASET_PATH
        ),

        "training_rows": int(len(X_train)),

        "test_rows": int(len(X_test)),

        "training_attrition_rate": round(
            float(y_train.mean()),
            6
        ),

        "model_parameters": params,

        "metrics": {
            key: (
                round(float(value), 6)
                if np.isfinite(value)
                else None
            )
            for key, value in metrics.items()
        },

        "model_path": str(MODEL_V2_PATH),

        "preprocessor_path": str(
            PREPROCESSOR_V2_PATH
        ),

        "mlflow_run_id": run.info.run_id,
    }

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=4
        )

    print("\n" + "=" * 70)
    print("EMPLOYEE ATTRITION RETRAINING COMPLETE")
    print("=" * 70)

    print(f"Report : {REPORT_PATH}")

    print("=" * 70)


if __name__ == "__main__":
    main()
