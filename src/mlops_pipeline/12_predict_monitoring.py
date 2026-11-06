
import json
from pathlib import Path

import pandas as pd


# Inference log path
LOG_PATH = Path("logs/inference/inference_log.jsonl")


# STEP 1: Load inference logs
def load_logs():

    if not LOG_PATH.exists():
        raise FileNotFoundError(
            f"Log file not found: {LOG_PATH}"
        )

    with LOG_PATH.open("r", encoding="utf-8") as f:

        records = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    if not records:
        raise ValueError("Inference log is empty")

    return pd.DataFrame(records)


# STEP 2: Analyze predictions
def analyze_predictions(df):

    required_columns = [
        "prediction",
        "attrition_probability",
        "model_name",
        "model_version",
        "run_id",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in inference logs: {missing_columns}"
        )

    total = len(df)

    # Count predicted classes
    prediction_counts = (
        df["prediction"].value_counts().sort_index()
    )

    prediction_percentages = (
        prediction_counts / total * 100
    ).round(2)

    # Probability statistics
    probability_stats = (
        df["attrition_probability"].describe()
    )

    print("\n" + "=" * 60)
    print("EMPLOYEE ATTRITION PREDICTION MONITORING")
    print("=" * 60)

    print(f"\nTotal predictions : {total}")

    print("\nPrediction counts:")
    print(prediction_counts)

    print("\nPrediction percentages:")
    print(prediction_percentages)

    print("\nAttrition probability statistics:")
    print(probability_stats)

    print("\nModel information:")

    print(f"Model name    : {df['model_name'].iloc[-1]}")
    print(f"Model version : {df['model_version'].iloc[-1]}")
    print(f"Run ID        : {df['run_id'].iloc[-1]}")


# STEP 3: Execute monitoring
def main():

    df = load_logs()

    analyze_predictions(df)


if __name__ == "__main__":
    main()
