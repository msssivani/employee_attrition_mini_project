
import os

import mlflow
import yaml
from mlflow.tracking import MlflowClient


def main():

    # Step 1: Load configuration
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Step 2: Get tracking URI and registered model name
    tracking_uri = (
        os.getenv("MLFLOW_TRACKING_URI")
        or config["mlflow"].get("tracking_uri")
    )

    model_name = config["registry"]["model_name"]

    # Step 3: Set MLflow tracking URI
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    # Step 4: Create MLflow client
    client = MlflowClient()

    # Step 5: Get effective tracking URI
    effective_uri = mlflow.get_tracking_uri()

    # Step 6: Display model information
    print(f"Model: {model_name}")
    print(f"Tracking URI: {effective_uri}")
    print()
    print("VERSION | STAGE | RUN_ID")

    # Step 7: Retrieve all registered model versions
    versions = client.search_model_versions(
        f"name='{model_name}'"
    )

    # Step 8: Sort versions in descending order
    versions = sorted(
        versions,
        key=lambda v: int(v.version),
        reverse=True
    )

    # Step 9: Display version details
    for version in versions:
        print(
            f"{version.version} | "
            f"{version.current_stage} | "
            f"{version.run_id}"
        )


# Execute main function
if __name__ == "__main__":
    main()
