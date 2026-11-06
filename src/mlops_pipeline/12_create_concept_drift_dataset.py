
import numpy as np
import pandas as pd
from pathlib import Path


FEATURES_PATH = Path(
    "data/monitoring/drift/evolved_data.csv"
)

LABELS_PATH = Path(
    "data/processed/y_test.npy"
)

OUTPUT_PATH = Path(
    "data/monitoring/drift/evolved_labeled_data.csv"
)

RANDOM_SEED = 42


def create_concept_drift_dataset():

    rng = np.random.default_rng(RANDOM_SEED)

    X = pd.read_csv(FEATURES_PATH)
    y = np.load(LABELS_PATH).astype(int)

    if len(X) != len(y):
        raise ValueError(
            f"Feature/label mismatch: X={len(X)}, y={len(y)}"
        )

    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("Invalid Attrition labels. Expected 0 and 1.")

    y_evolved = y.copy()

    changed_indices = set()

    # -------------------------------------------------
    # 1. Employees working overtime become more likely
    # to leave.
    # -------------------------------------------------

    if "OverTime" in X.columns:

        overtime_indices = np.where(
            (X["OverTime"] == "Yes") & (y_evolved == 0)
        )[0]

        flip_count = int(len(overtime_indices) * 0.25)

        if flip_count > 0:

            selected = rng.choice(
                overtime_indices,
                size=flip_count,
                replace=False,
            )

            y_evolved[selected] = 1
            changed_indices.update(selected.tolist())

    # -------------------------------------------------
    # 2. Lower job satisfaction becomes more strongly
    # associated with attrition.
    # -------------------------------------------------

    if "JobSatisfaction" in X.columns:

        satisfaction_indices = np.where(
            (X["JobSatisfaction"] <= 2) & (y_evolved == 0)
        )[0]

        flip_count = int(len(satisfaction_indices) * 0.15)

        if flip_count > 0:

            selected = rng.choice(
                satisfaction_indices,
                size=flip_count,
                replace=False,
            )

            y_evolved[selected] = 1
            changed_indices.update(selected.tolist())

    # -------------------------------------------------
    # 3. Lower monthly income becomes more strongly
    # associated with attrition.
    # -------------------------------------------------

    if "MonthlyIncome" in X.columns:

        income_threshold = X["MonthlyIncome"].quantile(0.25)

        income_indices = np.where(
            (X["MonthlyIncome"] <= income_threshold)
            & (y_evolved == 0)
        )[0]

        flip_count = int(len(income_indices) * 0.10)

        if flip_count > 0:

            selected = rng.choice(
                income_indices,
                size=flip_count,
                replace=False,
            )

            y_evolved[selected] = 1
            changed_indices.update(selected.tolist())

    # -------------------------------------------------
    # Save evolved labeled dataset
    # -------------------------------------------------

    result = X.copy()
    result["Attrition"] = y_evolved

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("=" * 65)
    print("EMPLOYEE ATTRITION - CONCEPT DRIFT DATASET")
    print("=" * 65)

    print(f"Total records       : {len(result)}")
    print(f"Original attrition  : {(y == 1).sum()}")
    print(f"New attrition       : {(y_evolved == 1).sum()}")
    print(f"Original non-attrition: {(y == 0).sum()}")
    print(f"New non-attrition   : {(y_evolved == 0).sum()}")
    print(f"Changed labels      : {len(changed_indices)}")

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    create_concept_drift_dataset()
