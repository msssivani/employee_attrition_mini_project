
import os
import sys

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml


def configure_mlflow(config):
    tracking_uri = (
        os.getenv("MLFLOW_TRACKING_URI")
        or config["mlflow"].get("tracking_uri")
    )

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)


def clean_dataframe(df, config):
    df = df.copy()

    target_column = config["data"]["target_column"]
    id_column = config["data"]["id_column"]

    # Remove employee ID because it is not a model feature.
    columns_to_drop = [
        id_column,
        target_column,
        "EmployeeCount",
        "Over18",
        "StandardHours",
    ]

    # Drop only columns that are present.
    columns_to_drop = [
        col for col in columns_to_drop if col in df.columns
    ]

    df = df.drop(columns=columns_to_drop)

    return df


def run_inference_demo():
    print("[INFO] Starting Employee Attrition Production Inference...")

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    configure_mlflow(config)

    model_name = config["registry"]["model_name"]
    model_uri = f"models:/{model_name}/Production"

    # 1. Load the Production model from MLflow.
    try:
        print(f"[INFO] Loading Production model: {model_uri}")
        model = mlflow.sklearn.load_model(model_uri)

    except Exception as exc:
        print(f"[ERROR] Could not load Production model: {exc}")
        print("[ERROR] Run the model lifecycle manager first.")
        return False

    # 2. Get file paths from config.
    preprocessor_path = config["paths"]["preprocessor"]
    raw_data_path = config["data"]["raw_path"]
    output_path = config["paths"]["inference_output"]

    # 3. Check whether required files exist.
    if not os.path.exists(preprocessor_path):
        print(f"[ERROR] Preprocessor not found: {preprocessor_path}")
        return False

    if not os.path.exists(raw_data_path):
        print(f"[ERROR] Raw dataset not found: {raw_data_path}")
        return False

    # 4. Load the fitted preprocessor and raw dataset.
    preprocessor = joblib.load(preprocessor_path)
    df = pd.read_csv(raw_data_path)

    # 5. Select five employee records.
    if len(df) < 5:
        print("[ERROR] Dataset contains fewer than 5 records.")
        return False

    sample = df.sample(n=5, random_state=7)

    # 6. Clean and transform the selected records.
    X_sample = clean_dataframe(sample, config)
    X_sample = preprocessor.transform(X_sample)

    # 7. Generate predictions and probabilities.
    predictions = model.predict(X_sample)
    probabilities = model.predict_proba(X_sample)[:, 1]

    # 8. Prepare the inference results.
    result = sample.copy()

    result["Predicted_Attrition"] = predictions
    result["Predicted_Attrition_Label"] = np.where(
        predictions == 1, "Yes", "No"
    )
    result["Attrition_Probability"] = np.round(probabilities, 4)

    # 9. Display the results.
    print("\n--- Employee Attrition Inference Results ---")

    display_columns = [
        "EmployeeNumber",
        "Predicted_Attrition_Label",
        "Attrition_Probability",
    ]

    # Include actual Attrition for comparison in this demo.
    if "Attrition" in result.columns:
        display_columns.append("Attrition")

    print(result[display_columns].to_string(index=False))
    print("--------------------------------------------\n")

    # 10. Save the results.
    output_dir = os.path.dirname(output_path)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    result.to_csv(output_path, index=False)

    print(f"[SUCCESS] Predictions saved to {output_path}")

    return True


if __name__ == "__main__":
    passed = run_inference_demo()
    sys.exit(0 if passed else 1)
