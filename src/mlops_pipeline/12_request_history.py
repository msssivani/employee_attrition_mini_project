
import json
from pathlib import Path

import pandas as pd


# Inference log path
LOG_PATH = Path("logs/inference/inference_log.jsonl")


# STEP 1: Load request history
def load_history():

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
        raise ValueError("Inference history is empty")

    df = pd.DataFrame(records)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True
    )

    return df


# STEP 2: Display request summary
def show_summary(df):

    print("\n" + "=" * 60)
    print("EMPLOYEE ATTRITION REQUEST-RESPONSE HISTORY")
    print("=" * 60)

    print(f"\nTotal records : {len(df)}")

    print("\nRecent requests:")

    columns = [
        "request_id",
        "timestamp",
        "prediction",
        "attrition_probability",
        "latency_ms",
        "status",
        "model_version",
    ]

    print(
        df[columns]
        .tail(10)
        .to_string(index=False)
    )


# STEP 3: Find a specific request
def find_request(df, request_id):

    result = df[
        df["request_id"] == request_id
    ]

    if result.empty:

        print(f"\nRequest not found: {request_id}")
        return

    print("\n" + "=" * 60)
    print("REQUEST DETAILS")
    print("=" * 60)

    print(
        result.to_string(index=False)
    )


# STEP 4: Display failed requests
def show_failures(df):

    failures = df[
        df["status"] == "failure"
    ]

    print("\n" + "=" * 60)
    print("FAILED REQUESTS")
    print("=" * 60)

    if failures.empty:

        print("\nNo failed requests.")

    else:

        print(f"\nFailed requests: {len(failures)}")

        print(
            failures[
                [
                    "request_id",
                    "timestamp",
                    "status",
                    "error",
                    "model_version",
                ]
            ].to_string(index=False)
        )


# STEP 5: Main function
def main():

    df = load_history()

    # Display recent request history
    show_summary(df)

    # Display failed requests
    show_failures(df)

    # Find the first request
    request_id = df["request_id"].iloc[0]

    find_request(df, request_id)


# STEP 6: Execute program
if __name__ == "__main__":
    main()
