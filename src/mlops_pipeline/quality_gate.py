
import json
import os
import sys

import yaml


def run_quality_gate():
    print("[INFO] Starting Employee Attrition Model Quality Gate...")

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    report_path = config["paths"]["evaluation_report"]
    gate_report_path = config["paths"]["quality_gate_report"]

    metric_name = config["quality_gate"]["metric"]
    minimum_value = float(config["quality_gate"]["minimum_value"])

    # Load evaluation report
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

    except FileNotFoundError:
        print(f"[ERROR] Evaluation report not found: {report_path}")
        return False

    # Check whether the required metric exists
    if metric_name not in report.get("metrics", {}):
        print(f"[ERROR] Metric '{metric_name}' not found in evaluation report.")
        return False

    actual_value = float(report["metrics"][metric_name])

    # Compare actual metric with required threshold
    passed = actual_value >= minimum_value

    print(
        f"[INFO] Quality Gate: {metric_name}={actual_value:.4f} | "
        f"required>={minimum_value:.4f}"
    )

    # Generate quality gate report
    gate_report = {
        "stage": "Employee Attrition - Model Quality Gate",
        "dataset": "HR Employee Attrition",
        "metric": metric_name,
        "minimum_value": minimum_value,
        "actual_value": actual_value,
        "status": "PASSED" if passed else "FAILED",
    }

    # Save report
    report_dir = os.path.dirname(gate_report_path)

    if report_dir:
        os.makedirs(report_dir, exist_ok=True)

    with open(gate_report_path, "w", encoding="utf-8") as f:
        json.dump(gate_report, f, indent=4)

    print(f"[INFO] Quality gate report saved to {gate_report_path}")

    if passed:
        print("[SUCCESS] Model Quality Gate PASSED.")
    else:
        print(
            "[ERROR] Model Quality Gate FAILED. "
            "Pipeline will stop before promotion."
        )

    return passed


if __name__ == "__main__":
    passed = run_quality_gate()
    sys.exit(0 if passed else 1)
