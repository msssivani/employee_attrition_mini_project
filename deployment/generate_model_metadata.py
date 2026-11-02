
import json
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "Employee_Attrition_Production_Model"

client = MlflowClient()

# 1. Find the Production model
versions = client.search_model_versions(
    f"name='{MODEL_NAME}'"
)

production_versions = [
    version for version in versions
    if version.current_stage == "Production"
]

if not production_versions:
    raise RuntimeError("No Production model found!")

production_model = production_versions[0]

# 2. Prepare deployment paths
deployment_dir = Path("deployment")
deployment_dir.mkdir(parents=True, exist_ok=True)

preprocessor_path = deployment_dir / "preprocessor.pkl"

if not preprocessor_path.exists():
    raise FileNotFoundError(
        "Fitted preprocessor not found in deployment folder. "
        "Export the matching preprocessor before generating metadata."
    )

# 3. Get model ID if supported by the installed MLflow version
model_id = getattr(production_model, "model_id", None)

# 4. Create metadata
metadata = {
    "model_name": MODEL_NAME,
    "model_version": str(production_model.version),
    "model_stage": production_model.current_stage,
    "run_id": production_model.run_id,
    "model_source": production_model.source,
    "model_id": model_id,
    "preprocessor_artifact": "deployment/preprocessor.pkl",
    "source": "MLflow"
}

# 5. Save metadata
metadata_path = deployment_dir / "model_metadata.json"

with metadata_path.open("w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=4)

print("Metadata generated successfully!")
print(json.dumps(metadata, indent=4))


'''import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

print("MLflow version:", mlflow.__version__)

versions = client.search_model_versions(
    "name='Employee_Attrition_Production_Model'"
)

production_model = next(
    v for v in versions
    if v.current_stage == "Production"
)

print("Available attributes:")
print(production_model)

print("Model version:", production_model.version)
print("Run ID:", production_model.run_id)
print("Model ID:", getattr(production_model, "model_id", "Not available"))'''
