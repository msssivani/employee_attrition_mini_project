
import json
import subprocess
import sys

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# FILE PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DRIFT_SCRIPT = PROJECT_ROOT / "src/mlops_pipeline/12_drift_analysis_report.py"

DRIFT_REPORT = PROJECT_ROOT / "outputs/drift/drift_analysis_report.csv"

NEW_DATA_PATH = PROJECT_ROOT / "data/raw/new_employee_attrition.csv"

PIPELINE_SCRIPT = (
    PROJECT_ROOT
    / "src/mlops_pipeline/employee_attrition_pipeline.py"
)

TRIGGER_REPORT = (
    PROJECT_ROOT
    / "outputs/monitoring/monitor_trigger_report.json"
)

DATA_LOAD_REPORT = (
    PROJECT_ROOT
    / "artifacts/data_load_report.json"
)


# ---------------------------------------------------------
# RUN DRIFT ANALYSIS
# ---------------------------------------------------------

def run_drift_analysis():

    print("[INFO] Running combined drift analysis...")

    result = subprocess.run(
        [sys.executable, str(DRIFT_SCRIPT)],
        cwd=PROJECT_ROOT,
    )

    return result.returncode == 0


# ---------------------------------------------------------
# DETECT DRIFT
# ---------------------------------------------------------

def detect_drift():

    if not DRIFT_REPORT.exists():
        raise FileNotFoundError(
            f"Drift report not found: {DRIFT_REPORT}"
        )

    df = pd.read_csv(DRIFT_REPORT)

    if "status" not in df.columns:
        raise ValueError(
            "Combined drift report does not contain a status column."
        )

    statuses = (
        df["status"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    drift_detected = statuses.eq("DRIFT").any()

    return bool(drift_detected)


# ---------------------------------------------------------
# GET DATA LOADING STATUS
# ---------------------------------------------------------

def get_data_load_status():

    if not DATA_LOAD_REPORT.exists():
        return None

    with DATA_LOAD_REPORT.open(
        "r",
        encoding="utf-8",
    ) as f:

        report = json.load(f)

    return report.get("status")


# ---------------------------------------------------------
# SAVE TRIGGER REPORT
# ---------------------------------------------------------

def save_trigger_report(report):

    TRIGGER_REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with TRIGGER_REPORT.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=4,
            default=str,
        )


# ---------------------------------------------------------
# MAIN MONITORING AND RETRAINING TRIGGER
# ---------------------------------------------------------

def run_monitor_trigger():

    print("=" * 70)
    print("EMPLOYEE ATTRITION - MONITORING + RETRAINING TRIGGER")
    print("=" * 70)

    # -----------------------------------------------------
    # STEP 1: RUN DRIFT ANALYSIS
    # -----------------------------------------------------

    drift_analysis_passed = run_drift_analysis()

    if not drift_analysis_passed:

        report = {
            "status": "drift_analysis_failed",
            "drift_detected": False,
            "new_data_present": NEW_DATA_PATH.exists(),
            "pipeline_called": False,
            "new_data_ingested": False,
            "retraining_executed": False,
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
        }

        save_trigger_report(report)

        print("[ERROR] Drift analysis failed.")

        return False

    # -----------------------------------------------------
    # STEP 2: CHECK WHETHER DRIFT EXISTS
    # -----------------------------------------------------

    drift_detected = detect_drift()

    new_data_present = NEW_DATA_PATH.exists()

    print(f"\n[INFO] Drift detected       : {drift_detected}")
    print(f"[INFO] New employee data    : {new_data_present}")

    # -----------------------------------------------------
    # STEP 3: NO DRIFT
    # -----------------------------------------------------

    if not drift_detected:

        report = {
            "status": "no_drift",
            "drift_detected": False,
            "new_data_present": new_data_present,
            "pipeline_called": False,
            "new_data_ingested": False,
            "retraining_executed": False,
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
        }

        save_trigger_report(report)

        print(
            "[INFO] No drift detected. "
            "Waiting for future monitoring cycles."
        )

        return True

    # -----------------------------------------------------
    # STEP 4: DRIFT EXISTS, BUT NO NEW DATA
    # -----------------------------------------------------

    if not new_data_present:

        report = {
            "status": "drift_waiting_for_data",
            "drift_detected": True,
            "new_data_present": False,
            "pipeline_called": False,
            "new_data_ingested": False,
            "retraining_executed": False,
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
        }

        save_trigger_report(report)

        print(
            "[INFO] Drift detected, "
            "but no new employee dataset is available."
        )

        print(
            "[INFO] Waiting for new_employee_attrition.csv."
        )

        return True

    # -----------------------------------------------------
    # STEP 5: DRIFT AND NEW DATA ARE AVAILABLE
    # -----------------------------------------------------

    print(
        "\n[INFO] Drift detected and new employee data is available."
    )

    if not PIPELINE_SCRIPT.exists():

        raise FileNotFoundError(
            f"Employee Attrition pipeline not found: "
            f"{PIPELINE_SCRIPT}"
        )

    print("[INFO] Triggering Employee Attrition MLOps pipeline...")

    pipeline_result = subprocess.run(
        [
            sys.executable,
            str(PIPELINE_SCRIPT),
        ],
        cwd=PROJECT_ROOT,
    )

    # -----------------------------------------------------
    # STEP 6: CHECK DATA INGESTION
    # -----------------------------------------------------

    data_load_status = get_data_load_status()

    new_data_ingested = (
        data_load_status == "ingested"
    )

    retraining_executed = (
        new_data_ingested
        and pipeline_result.returncode == 0
    )

    if retraining_executed:
        status = "retraining_completed"

    elif pipeline_result.returncode != 0:
        status = "pipeline_failed"

    else:
        status = "data_ingestion_not_confirmed"

    # -----------------------------------------------------
    # STEP 7: SAVE FINAL TRIGGER REPORT
    # -----------------------------------------------------

    report = {
        "status": status,

        "drift_detected": True,

        "new_data_present": True,

        "new_data_path": str(NEW_DATA_PATH),

        "pipeline_called": True,

        "pipeline_exit_code": pipeline_result.returncode,

        "pipeline_status": (
            "passed"
            if pipeline_result.returncode == 0
            else "failed"
        ),

        "data_load_status": data_load_status,

        "new_data_ingested": new_data_ingested,

        "retraining_executed": retraining_executed,

        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    save_trigger_report(report)

    # -----------------------------------------------------
    # STEP 8: PRINT RESULT
    # -----------------------------------------------------

    print("\n" + "=" * 70)

    if pipeline_result.returncode != 0:

        print("[ERROR] Employee Attrition pipeline failed.")

        return False

    if retraining_executed:

        print(
            "[SUCCESS] New employee data was ingested "
            "and the pipeline completed."
        )

    else:

        print(
            "[WARNING] Pipeline completed, but new data "
            "ingestion was not confirmed."
        )

        print(
            "[WARNING] Retraining completion cannot be confirmed."
        )

    print(f"\nTrigger status: {status}")

    print(f"Trigger report: {TRIGGER_REPORT}")

    print("=" * 70)

    return True


if __name__ == "__main__":

    passed = run_monitor_trigger()

    sys.exit(0 if passed else 1)
