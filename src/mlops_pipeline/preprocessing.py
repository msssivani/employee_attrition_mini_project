
import os
import sys
import json
import joblib
import yaml
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


def clean_dataframe(df, config):
    """
    Apply deterministic dataset cleaning shared by training and inference.
    """

    df = df.copy()

    target_column = config["data"]["target_column"]
    id_column = config["data"]["id_column"]

    # Remove employee ID because it is not a useful model feature.
    if id_column in df.columns:
        df = df.drop(columns=[id_column])

    # Convert target column from Yes/No to 1/0.
    if target_column in df.columns:

        target_mapping = {
            "yes": 1,
            "no": 0
        }

        mapped_target = (
            df[target_column]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(target_mapping)
        )

        if mapped_target.isna().any():
            raise ValueError(
                f"Unexpected values found in target column: {target_column}"
            )

        df[target_column] = mapped_target.astype(int)

    return df


def split_raw_data(df, config):
    """
    Clean the raw dataset and create the reproducible train/test split.
    """

    target_column = config["data"]["target_column"]

    cleaned = clean_dataframe(df, config)

    X = cleaned.drop(columns=[target_column])
    y = cleaned[target_column]

    return train_test_split(
        X,
        y,
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"],
        stratify=y
    )


def build_preprocessor(X_train):
    """
    Create the preprocessing pipeline for Employee Attrition.
    """

    # Numerical features
    num_cols = [
        "Age",
        "DailyRate",
        "DistanceFromHome",
        "HourlyRate",
        "MonthlyIncome",
        "MonthlyRate",
        "NumCompaniesWorked",
        "PercentSalaryHike",
        "TotalWorkingYears",
        "TrainingTimesLastYear",
        "YearsAtCompany",
        "YearsInCurrentRole",
        "YearsSinceLastPromotion",
        "YearsWithCurrManager"
    ]

    # Ordinal features
    ordinal_cols = [
        "Education",
        "EnvironmentSatisfaction",
        "JobInvolvement",
        "JobLevel",
        "JobSatisfaction",
        "RelationshipSatisfaction",
        "StockOptionLevel",
        "WorkLifeBalance",
        "PerformanceRating"
    ]

    # Categorical features
    cat_cols = [
        "BusinessTravel",
        "Department",
        "EducationField",
        "Gender",
        "JobRole",
        "MaritalStatus",
        "OverTime"
    ]

    # Check whether all expected features are available.
    expected_cols = num_cols + ordinal_cols + cat_cols

    missing_cols = [
        col for col in expected_cols
        if col not in X_train.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Missing expected feature columns: {missing_cols}"
        )

    # Numerical preprocessing
    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    # Ordinal preprocessing
    ordinal_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ]
    )

    # Categorical preprocessing
    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ohe",
                OneHotEncoder(
                    drop="first",
                    sparse_output=False,
                    handle_unknown="ignore"
                )
            )
        ]
    )

    # Combine all preprocessing pipelines.
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("ordinal", ordinal_pipeline, ordinal_cols),
            ("cat", cat_pipeline, cat_cols)
        ]
    )

    return preprocessor, cat_cols, num_cols, ordinal_cols


def run_preprocessing():

    print("[INFO] Starting Employee Attrition Preprocessing...")

    # Load configuration
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_path = config["data"]["raw_path"]
    processed_dir = config["data"]["processed_dir"]
    target_column = config["data"]["target_column"]

    # Check dataset availability
    if not os.path.exists(data_path):
        print(f"[ERROR] Raw dataset not found at: {data_path}")
        return False

    # Load dataset
    raw_df = pd.read_csv(data_path)

    print(f"[INFO] Dataset loaded: {raw_df.shape}")

    # Create train/test split
    X_train, X_test, y_train, y_test = split_raw_data(
        raw_df,
        config
    )

    print(f"[INFO] Training data shape: {X_train.shape}")
    print(f"[INFO] Testing data shape: {X_test.shape}")

    # Build preprocessing pipeline
    preprocessor, cat_cols, num_cols, ordinal_cols = build_preprocessor(
        X_train
    )

    # Fit only on training data.
    print("[INFO] Fitting preprocessing pipeline on training data...")

    X_train_final = preprocessor.fit_transform(X_train)

    # Apply learned transformations to test data.
    X_test_final = preprocessor.transform(X_test)

    # Create output directories.
    os.makedirs(processed_dir, exist_ok=True)

    preprocessor_path = config["paths"]["preprocessor"]

    os.makedirs(
        os.path.dirname(preprocessor_path) or ".",
        exist_ok=True
    )

    # Save transformed feature matrices.
    np.save(
        os.path.join(processed_dir, "X_train_final.npy"),
        X_train_final
    )

    np.save(
        os.path.join(processed_dir, "X_test_final.npy"),
        X_test_final
    )

    # Save target arrays.
    np.save(
        os.path.join(processed_dir, "y_train.npy"),
        y_train.to_numpy(dtype=np.int64)
    )

    np.save(
        os.path.join(processed_dir, "y_test.npy"),
        y_test.to_numpy(dtype=np.int64)
    )

    # Save raw split data.
    X_train.to_csv(
        os.path.join(processed_dir, "X_train_raw.csv"),
        index=False
    )

    X_test.to_csv(
        os.path.join(processed_dir, "X_test_raw.csv"),
        index=False
    )

    # Save fitted preprocessor.
    joblib.dump(preprocessor, preprocessor_path)

    # Create dataset metadata.
    metadata = {
        "dataset_name": "IBM HR Employee Attrition",
        "target_column": target_column,

        "train_shape_raw": list(X_train.shape),
        "test_shape_raw": list(X_test.shape),

        "train_shape_transformed": list(X_train_final.shape),
        "test_shape_transformed": list(X_test_final.shape),

        "numerical_features": num_cols,
        "ordinal_features": ordinal_cols,
        "categorical_features": cat_cols,

        "split_test_size": config["split"]["test_size"],
        "split_random_state": config["split"]["random_state"],

        "raw_train_path": os.path.join(
            processed_dir, "X_train_raw.csv"
        ),

        "raw_test_path": os.path.join(
            processed_dir, "X_test_raw.csv"
        )
    }

    with open(
        os.path.join(processed_dir, "dataset_metadata.json"),
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(metadata, f, indent=4)

    print("[SUCCESS] Preprocessing completed!")

    print(
        f"[INFO] Preprocessor saved to: {preprocessor_path}"
    )

    return True


if __name__ == "__main__":

    try:
        passed = run_preprocessing()
        sys.exit(0 if passed else 1)

    except Exception as e:
        print(f"[ERROR] Preprocessing failed: {e}")
        sys.exit(1)
