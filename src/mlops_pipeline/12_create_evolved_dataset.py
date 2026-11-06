
import numpy as np
import pandas as pd
from pathlib import Path


SOURCE_PATH = Path(
    "data/processed/X_test_raw.csv"
)

OUTPUT_PATH = Path(
    "data/monitoring/drift/evolved_data.csv"
)

RANDOM_SEED = 42


def create_evolved_dataset():

    rng = np.random.default_rng(RANDOM_SEED)

    df = pd.read_csv(SOURCE_PATH).copy()

    # -------------------------------------------------
    # 1. MonthlyIncome distribution drift
    # -------------------------------------------------

    if "MonthlyIncome" in df.columns:

        df["MonthlyIncome"] = (
            df["MonthlyIncome"]
            * rng.normal(1.15, 0.08, len(df))
        ).round().clip(1000, 20000).astype(int)

    # -------------------------------------------------
    # 2. DistanceFromHome distribution drift
    # -------------------------------------------------

    if "DistanceFromHome" in df.columns:

        df["DistanceFromHome"] = (
            df["DistanceFromHome"]
            + rng.normal(3, 2, len(df))
        ).round().clip(1, 29).astype(int)

    # -------------------------------------------------
    # 3. TotalWorkingYears distribution drift
    # -------------------------------------------------

    if "TotalWorkingYears" in df.columns:

        df["TotalWorkingYears"] = (
            df["TotalWorkingYears"]
            + rng.normal(2, 3, len(df))
        ).round().clip(0, 40).astype(int)

    # -------------------------------------------------
    # 4. OverTime distribution drift
    # -------------------------------------------------

    if "OverTime" in df.columns:

        df["OverTime"] = rng.choice(
            ["Yes", "No"],
            size=len(df),
            p=[0.55, 0.45],
        )

    # -------------------------------------------------
    # 5. BusinessTravel distribution drift
    # -------------------------------------------------

    if "BusinessTravel" in df.columns:

        df["BusinessTravel"] = rng.choice(
            [
                "Travel_Rarely",
                "Travel_Frequently",
                "Non-Travel",
            ],
            size=len(df),
            p=[0.50, 0.35, 0.15],
        )

    # -------------------------------------------------
    # 6. JobSatisfaction distribution drift
    # -------------------------------------------------

    if "JobSatisfaction" in df.columns:

        df["JobSatisfaction"] = rng.choice(
            [1, 2, 3, 4],
            size=len(df),
            p=[0.35, 0.30, 0.20, 0.15],
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("=" * 65)
    print("EMPLOYEE ATTRITION - EVOLVED DATASET")
    print("=" * 65)

    print(f"Original dataset shape : {pd.read_csv(SOURCE_PATH).shape}")
    print(f"Evolved dataset shape  : {df.shape}")

    print("\nEvolved feature samples:")
    print(
        df[
            [
                col for col in [
                    "MonthlyIncome",
                    "DistanceFromHome",
                    "TotalWorkingYears",
                    "OverTime",
                    "BusinessTravel",
                    "JobSatisfaction",
                ]
                if col in df.columns
            ]
        ].head()
    )

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    create_evolved_dataset()
