import json
import os
import shutil
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import yaml
from mlflow import MlflowClient


def load_config():
    # Find the project root directory
    project_root = Path(__file__).resolve().parents[2]

    # Load project configuration
    config_path = project_root / "config.yaml"

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return project_root, config


def get_production_model(client, model_name):
    # Retrieve all registered versions of the model
    versions = client.search_model_versions(
        f"name='{model_name}'"
    )

    # Select the version currently in Production
    production_versions = [
        version for version in versions
        if version.current_stage == "Production"
    ]

    if not production_versions:
        raise RuntimeError(
            f"No Production version found for '{model_name}'."
        )

    if len(production_versions) > 1:
        raise RuntimeError(
            f"Multiple Production versions found for '{model_name}'."
        )

    return production_versions[0]


def prepare_deployment():
    project_root, config = load_config()

    # Get model name from config.yaml
    model_name = config["registry"]["model_name"]

    # Configure MLflow tracking URI if specified
    tracking_uri = (
        os.getenv("MLFLOW_TRACKING_URI")
        or config.get("mlflow", {}).get("tracking_uri")
    )

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()

    # Create a fresh deployment directory
    deployment_dir = project_root / "deployment"

    # Create the deployment directory without deleting existing files
    deployment_dir.mkdir(parents=True, exist_ok=True)

    # Find the Production model
    model_version = get_production_model(client, model_name)

    version = model_version.version
    run_id = model_version.run_id
    model_source = model_version.source

    print("[INFO] Production model found")
    print(f"[INFO] Model name : {model_name}")
    print(f"[INFO] Version    : {version}")
    print(f"[INFO] Run ID     : {run_id}")

    # Download the Production model from MLflow
    model_uri = f"models:/{model_name}/{version}"

    model = mlflow.sklearn.load_model(model_uri)

    model_path = deployment_dir / "model.pkl"
    joblib.dump(model, model_path)

    print(f"[SUCCESS] Model saved to {model_path}")

    # Download the matching preprocessor from the same MLflow run
    preprocessor_path = client.download_artifacts(
        run_id,
        "preprocessing_pipeline/preprocessor.pkl",
        dst_path=str(deployment_dir),
    )

    downloaded_preprocessor = Path(preprocessor_path)
    final_preprocessor_path = deployment_dir / "preprocessor.pkl"

# Remove only the old preprocessor artifact, if it exists
    if final_preprocessor_path.exists():
        final_preprocessor_path.unlink()

    if downloaded_preprocessor != final_preprocessor_path:
        shutil.move(
        str(downloaded_preprocessor),
        str(final_preprocessor_path)
    )

    print(f"[SUCCESS] Preprocessor saved to {final_preprocessor_path}")

    # Verify that both artifacts can be loaded
    loaded_model = joblib.load(model_path)
    loaded_preprocessor = joblib.load(final_preprocessor_path)

    if loaded_model is None or loaded_preprocessor is None:
        raise RuntimeError("Deployment artifact verification failed.")

    # Save model metadata
    metadata = {
        "project": "Employee Attrition Prediction",
        "model_name": model_name,
        "model_version": str(version),
        "model_stage": model_version.current_stage,
        "run_id": run_id,
        "model_source": model_source,
        "preprocessor_artifact": "preprocessing_pipeline/preprocessor.pkl",
        "source": "MLflow"
    }

    metadata_path = deployment_dir / "model_metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print(f"[SUCCESS] Metadata saved to {metadata_path}")

    print("\n[INFO] Deployment bundle:")
    print("deployment/")
    print("├── model.pkl")
    print("├── preprocessor.pkl")
    print("└── model_metadata.json")

    print("\n[SUCCESS] Deployment preparation completed.")


if __name__ == "__main__":
    prepare_deployment()