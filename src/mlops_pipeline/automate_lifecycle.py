import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import yaml
from mlflow.tracking import MlflowClient


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIFECYCLE_REPORT = PROJECT_ROOT / "reports" / "lifecycle_report.json"


def configure_mlflow(config):
    tracking_uri = (
        os.getenv("MLFLOW_TRACKING_URI")
        or config["mlflow"].get("tracking_uri")
    )
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)


def save_lifecycle_report(report):
    LIFECYCLE_REPORT.parent.mkdir(parents=True, exist_ok=True)
    report["timestamp"] = datetime.now(timezone.utc).isoformat()

    with LIFECYCLE_REPORT.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)


def automate_model_lifecycle():
    report = {
        "status": "failed",
        "promoted": False,
        "candidate_version": None,
        "production_version_after": None,
        "model_name": None,
        "decision_metric": None,
        "candidate_score": None,
        "production_score_before": None,
        "error": None,
    }

    try:
        print("[INFO] Starting Automated Model Lifecycle Manager...")

        with (PROJECT_ROOT / "config.yaml").open(
            "r", encoding="utf-8"
        ) as file:
            config = yaml.safe_load(file)

        configure_mlflow(config)

        client = MlflowClient()
        model_name = config["registry"]["model_name"]
        metric_to_optimize = config["registry"]["decision_metric"]

        report["model_name"] = model_name
        report["decision_metric"] = metric_to_optimize

        versions = client.search_model_versions(f"name='{model_name}'")
        new_models = [version for version in versions if version.current_stage == "None"]

        for version in new_models:
            client.transition_model_version_stage(
                name=model_name,
                version=version.version,
                stage="Staging",
                archive_existing_versions=False,
            )
            print(f"[INFO] Version {version.version} moved to Staging.")

        versions = client.search_model_versions(f"name='{model_name}'")
        staging_models = [
            version for version in versions if version.current_stage == "Staging"
        ]

        production_models = [
            version for version in versions if version.current_stage == "Production"
        ]
        production_model = production_models[0] if production_models else None

        if production_model is not None:
            production_run = client.get_run(production_model.run_id)
            report["production_score_before"] = production_run.data.metrics.get(
                metric_to_optimize
            )

        if not staging_models:
            print("[INFO] No models in Staging. Nothing to promote.")
            report["status"] = "no_staging_models"
            report["production_version_after"] = (
                str(production_model.version) if production_model else None
            )
            save_lifecycle_report(report)
            print(f"[INFO] Lifecycle report saved to {LIFECYCLE_REPORT}")
            return True

        best_model = None
        best_score = float("-inf")

        for version in staging_models:
            run = client.get_run(version.run_id)
            score = run.data.metrics.get(metric_to_optimize)

            if score is None:
                print(
                    f"[WARNING] Version {version.version} has no "
                    f"'{metric_to_optimize}' metric; skipping it."
                )
                continue

            print(
                f"[INFO] Staging Version {version.version} | "
                f"{metric_to_optimize}={score:.4f}"
            )

            if score > best_score:
                best_score = score
                best_model = version

        if best_model is None:
            raise ValueError(
                f"No Staging model has the '{metric_to_optimize}' metric."
            )

        report["candidate_version"] = str(best_model.version)
        report["candidate_score"] = best_score

        move_to_production = (
            production_model is None
            or best_score > report["production_score_before"]
        )

        if move_to_production:
            client.transition_model_version_stage(
                name=model_name,
                version=best_model.version,
                stage="Production",
                archive_existing_versions=False,
            )

            if production_model is not None:
                client.transition_model_version_stage(
                    name=model_name,
                    version=production_model.version,
                    stage="Archived",
                    archive_existing_versions=False,
                )

            report["promoted"] = True
            report["status"] = "promoted"
            print(f"[SUCCESS] Version {best_model.version} is now Production.")
        else:
            client.transition_model_version_stage(
                name=model_name,
                version=best_model.version,
                stage="Archived",
                archive_existing_versions=False,
            )
            report["status"] = "not_promoted"
            print("[INFO] Production model remains unchanged.")

        # Archive any other models left in Staging.
        for version in client.search_model_versions(f"name='{model_name}'"):
            if version.current_stage == "Staging":
                client.transition_model_version_stage(
                    name=model_name,
                    version=version.version,
                    stage="Archived",
                    archive_existing_versions=False,
                )

        final_versions = client.search_model_versions(f"name='{model_name}'")
        final_production = next(
            (
                version
                for version in final_versions
                if version.current_stage == "Production"
            ),
            None,
        )
        report["production_version_after"] = (
            str(final_production.version) if final_production else None
        )

        save_lifecycle_report(report)
        print(f"[INFO] Lifecycle report saved to {LIFECYCLE_REPORT}")
        print("[SUCCESS] Automated Model Lifecycle execution complete!")
        return True

    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        save_lifecycle_report(report)
        print(f"[ERROR] Failed to automate model lifecycle: {exc}")
        print(f"[INFO] Failure report saved to {LIFECYCLE_REPORT}")
        return False


if __name__ == "__main__":
    sys.exit(0 if automate_model_lifecycle() else 1)