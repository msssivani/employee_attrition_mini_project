
import json
import os
import subprocess
import sys
from pathlib import Path


# Assumes this orchestrator is inside src/mlops_pipeline/
PROJECT_ROOT = Path(__file__).resolve().parents[2]


STAGES = [
    ("Data Loading", "src/mlops_pipeline/data_loading.py"),
    ("Raw Data Validation", "src/mlops_pipeline/validate_data.py"),
    ("Preprocessing", "src/mlops_pipeline/preprocessing.py"),
    ("Processed Output Validation", "src/mlops_pipeline/validate_outputs.py"),
    ("Training + MLflow + Model Registry", "src/mlops_pipeline/train_registry.py"),
    ("Model Evaluation", "src/mlops_pipeline/model_evaluation.py"),
    ("Model Quality Gate", "src/mlops_pipeline/quality_gate.py"),
    ("Automated Model Lifecycle", "src/mlops_pipeline/automate_lifecycle.py"),
    ("Production Inference", "src/mlops_pipeline/inference.py"),
    ("Failure Injection and Recovery Testing", "src/mlops_pipeline/failure_injection.py"),
]


def run_pipeline():
    print("[INFO] ==================================================")
    print("[INFO] Starting Employee Attrition MLOps Pipeline")
    print("[INFO] ==================================================")

    run_log = {
        "pipeline": "Employee Attrition MLOps",
        "stages": []
    }

    for stage_name, script_path in STAGES:

        print("\n" + "=" * 70)
        print(f"RUNNING: {stage_name}")
        print("=" * 70)

        full_script_path = PROJECT_ROOT / script_path

        if not full_script_path.exists():
            print(f"[ERROR] Script not found: {full_script_path}")
            return_code = 1
        else:
            result = subprocess.run(
                [sys.executable, str(full_script_path)],
                cwd=PROJECT_ROOT
            )

            return_code = result.returncode

        status = "PASSED" if return_code == 0 else "FAILED"

        run_log["stages"].append(
            {
                "stage": stage_name,
                "status": status,
                "exit_code": return_code
            }
        )

        print(f"--> {stage_name}: {status}")

        if return_code != 0:
            print("[HALT] Pipeline stopped because this stage failed.")
            break

    all_passed = (
        len(run_log["stages"]) == len(STAGES)
        and all(
            stage["status"] == "PASSED"
            for stage in run_log["stages"]
        )
    )

    run_log["overall_status"] = (
        "PASSED" if all_passed else "FAILED"
    )

    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_path = logs_dir / "employee_attrition_pipeline_run_log.json"

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(run_log, f, indent=4)

    print("\n" + "=" * 70)

    if all_passed:
        print(
            "[SUCCESS] Employee Attrition MLOps Pipeline "
            "completed successfully."
        )
    else:
        print("[ERROR] Employee Attrition MLOps Pipeline failed.")

    print(f"[INFO] Run log: {log_path}")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    passed = run_pipeline()
    sys.exit(0 if passed else 1)
