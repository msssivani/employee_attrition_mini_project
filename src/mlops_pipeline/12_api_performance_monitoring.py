
import json
from pathlib import Path

import pandas as pd


LOG_PATH = Path("logs/inference/inference_log.jsonl")


def load_logs():
    print("[INFO] Loading Employee Attrition API inference logs...")

    if not LOG_PATH.exists():
        raise FileNotFoundError(
            f"[ERROR] Inference log not found: {LOG_PATH}"
        )

    records = []

    with LOG_PATH.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(
                    f"[WARNING] Skipping invalid log at line "
                    f"{line_number}: {e}"
                )

    if not records:
        raise ValueError("[ERROR] No valid inference records found.")

    df = pd.DataFrame(records)

    required_columns = ["status", "latency_ms", "timestamp"]

    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"[ERROR] Missing required log fields: {missing_columns}"
        )

    print(f"[INFO] Loaded {len(df)} inference records.")

    return df


def analyze_performance(df):
    total_requests = len(df)

    success_count = (df["status"] == "success").sum()
    failure_count = (df["status"] == "failure").sum()
    other_count = total_requests - success_count - failure_count

    success_rate = (success_count / total_requests) * 100
    failure_rate = (failure_count / total_requests) * 100

    latency = pd.to_numeric(
        df["latency_ms"], errors="coerce"
    ).dropna()

    timestamps = pd.to_datetime(
        df["timestamp"], utc=True, errors="coerce"
    ).dropna()

    print("\n" + "=" * 65)
    print("EMPLOYEE ATTRITION API - PERFORMANCE MONITORING")
    print("=" * 65)

    print("\nREQUEST STATISTICS")
    print("-" * 40)

    print(f"Total requests       : {total_requests}")
    print(f"Successful requests : {success_count}")
    print(f"Failed requests     : {failure_count}")
    print(f"Other statuses      : {other_count}")

    print(f"Success rate        : {success_rate:.2f}%")
    print(f"Failure rate        : {failure_rate:.2f}%")

    if not latency.empty:
        print("\nLATENCY ANALYSIS")
        print("-" * 40)

        print(f"Average latency     : {latency.mean():.3f} ms")
        print(f"Median latency      : {latency.median():.3f} ms")
        print(f"P95 latency         : {latency.quantile(0.95):.3f} ms")
        print(f"Minimum latency     : {latency.min():.3f} ms")
        print(f"Maximum latency     : {latency.max():.3f} ms")
    else:
        print("\n[WARNING] No valid latency values available.")

    if len(timestamps) >= 2:
        elapsed_seconds = (
            timestamps.max() - timestamps.min()
        ).total_seconds()

        if elapsed_seconds > 0:
            throughput = total_requests / elapsed_seconds

            print("\nTHROUGHPUT ANALYSIS")
            print("-" * 40)
            print(f"Elapsed time        : {elapsed_seconds:.3f} seconds")
            print(f"Throughput          : {throughput:.3f} requests/sec")
        else:
            print("\nThroughput          : Cannot calculate")
    else:
        print("\nThroughput          : Insufficient valid timestamps")

    print("\nSTATUS DISTRIBUTION")
    print("-" * 40)
    print(df["status"].value_counts().to_string())

    print("\n" + "=" * 65)
    print("[SUCCESS] Employee Attrition API monitoring completed.")
    print("=" * 65)


def main():
    try:
        df = load_logs()
        analyze_performance(df)

    except (FileNotFoundError, ValueError) as e:
        print(e)


if __name__ == "__main__":
    main()
