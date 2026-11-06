
import sys
import json
from pathlib import Path

import pandas as pd
import yaml


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_data():
    print("[INFO] Starting Data Loading...")

    # Load configuration
    config_path = PROJECT_ROOT / "config.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Get dataset path from config.yaml
    data_path = Path(config["data"]["raw_path"])

    # Resolve relative paths from project root
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path

    if not data_path.exists():
        print(f"[ERROR] Raw dataset not found at: {data_path}")
        return False

    try:
        df = pd.read_csv(data_path)

    except Exception as e:
        print(f"[ERROR] Unable to load dataset: {e}")
        return False

    print(
        f"[INFO] Loaded {df.shape[0]} rows, "
        f"{df.shape[1]} columns from {data_path}"
    )

    report = {
        "stage": "Employee Attrition - Data Loading",
        "source_file": str(data_path.relative_to(PROJECT_ROOT)),
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }

    # Save data loading report
    artifacts_dir = PROJECT_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    report_path = artifacts_dir / "data_load_report.json"

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print(
        "[SUCCESS] Data loaded successfully. "
        f"Report saved to {report_path}"
    )

    return True


if __name__ == "__main__":
    passed = load_data()
    sys.exit(0 if passed else 1)
