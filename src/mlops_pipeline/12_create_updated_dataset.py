
import numpy as np
import pandas as pd
from pathlib import Path


TRAIN_FEATURES_PATH = Path(
    "data/processed/X_train_raw.csv"
)

TRAIN_LABELS_PATH = Path(
    "data/processed/y_train.npy"
)

TEST_FEATURES_PATH = Path(
    "data/processed/X_test_raw.csv"
)

NEW_BATCH_PATH = Path(
    "data/monitoring/drift/post_drift_training_data.csv"
)

UPDATED_DATASET_PATH = Path(
    "data/processed/updated_training_data.csv"
)

SUMMARY_PATH = Path(
    "outputs/retraining/updated_dataset_summary.csv"
)

RANDOM_SEED = 42
NEW_BATCH_SIZE = 1409


def load_original_training_data():

    X_train = pd.read_csv(TRAIN_FEATURES_PATH)
    y_train = np.load(TRAIN_LABELS_PATH).astype(int)
    X_test = pd.read_csv(TEST_FEATURES_PATH)

    if len(X_train) != len(y_train):
        raise ValueError("Training feature/label mismatch.")

    return X_train, y_train, X_test


def create_post_drift_batch(X_train, y_train):

    rng = np.random.default_rng(RANDOM_SEED)

    sample_indices = rng.choice(
        len(X_train),
        size=NEW_BATCH_SIZE,
        replace=True,
    )

    X_new = (
        X_train.iloc[sample_indices]
        .copy()
        .reset_index(drop=True)
    )

    y_new = y_train[sample_indices].copy()

    # MonthlyIncome drift
    if "MonthlyIncome" in X_new.columns:

        X_new["MonthlyIncome"] = (
            X_new["MonthlyIncome"]
            * rng.normal(1.15, 0.08, len(X_new))
        ).round().clip(1000, 20000).astype(int)

    # DistanceFromHome drift
    if "DistanceFromHome" in X_new.columns:

        X_new["DistanceFromHome"] = (
            X_new["DistanceFromHome"]
            + rng.normal(3, 2, len(X_new))
        ).round().clip(1, 29).astype(int)

    # TotalWorkingYears drift
    if "TotalWorkingYears" in X_new.columns:

        X_new["TotalWorkingYears"] = (
            X_new["TotalWorkingYears"]
            + rng.normal(2, 3, len(X_new))
        ).round().clip(0, 40).astype(int)

    # OverTime drift
    if "OverTime" in X_new.columns:

        X_new["OverTime"] = rng.choice(
            ["Yes", "No"],
            size=len(X_new),
            p=[0.55, 0.45],
        )

    # BusinessTravel drift
    if "BusinessTravel" in X_new.columns:

        X_new["BusinessTravel"] = rng.choice(
            [
                "Travel_Rarely",
                "Travel_Frequently",
                "Non-Travel",
            ],
            size=len(X_new),
            p=[0.50, 0.35, 0.15],
        )

    # JobSatisfaction drift
    if "JobSatisfaction" in X_new.columns:

        X_new["JobSatisfaction"] = rng.choice(
            [1, 2, 3, 4],
            size=len(X_new),
            p=[0.35, 0.30, 0.20, 0.15],
        )

    # Simulate changed feature-target relationship:
    # OverTime = Yes increases simulated attrition.
    if "OverTime" in X_new.columns:

        candidates = np.where(
            (X_new["OverTime"] == "Yes") & (y_new == 0)
        )[0]

        flip_count = int(len(candidates) * 0.25)

        if flip_count > 0:

            selected = rng.choice(
                candidates,
                size=flip_count,
                replace=False,
            )

            y_new[selected] = 1

    # Low job satisfaction increases simulated attrition.
    if "JobSatisfaction" in X_new.columns:

        candidates = np.where(
            (X_new["JobSatisfaction"] <= 2) & (y_new == 0)
        )[0]

        flip_count = int(len(candidates) * 0.15)

        if flip_count > 0:

            selected = rng.choice(
                candidates,
                size=flip_count,
                replace=False,
            )

            y_new[selected] = 1

    # Low income increases simulated attrition.
    if "MonthlyIncome" in X_new.columns:

        threshold = X_new["MonthlyIncome"].quantile(0.25)

        candidates = np.where(
            (X_new["MonthlyIncome"] <= threshold) & (y_new == 0)
        )[0]

        flip_count = int(len(candidates) * 0.10)

        if flip_count > 0:

            selected = rng.choice(
                candidates,
                size=flip_count,
                replace=False,
            )

            y_new[selected] = 1

    return X_new, y_new


def validate_new_batch(X_new, y_new, X_test):

    if list(X_new.columns) != list(X_test.columns):
        raise ValueError("Feature schema mismatch.")

    if len(X_new) != len(y_new):
        raise ValueError("New batch feature/label mismatch.")

    if not set(np.unique(y_new)).issubset({0, 1}):
        raise ValueError("Invalid Attrition labels.")

    if X_new.isnull().all().any():
        raise ValueError("A column contains only null values.")

    # Check exact overlap with held-out test rows.
    test_hashes = set(
        pd.util.hash_pandas_object(
            X_test,
            index=False,
        ).astype("uint64")
    )

    new_hashes = pd.util.hash_pandas_object(
        X_new,
        index=False,
    ).astype("uint64")
    
    exact_overlap = sum(
        value in test_hashes for value in new_hashes
    )

    if exact_overlap > 0:
        raise ValueError(
            f"Detected {exact_overlap} exact feature-row overlaps "
            "with the test set."
        )

    return exact_overlap


def create_updated_dataset(X_train, y_train, X_new, y_new):

    original_train = X_train.copy()
    original_train["Attrition"] = y_train

    new_batch = X_new.copy()
    new_batch["Attrition"] = y_new

    updated = pd.concat(
        [original_train, new_batch],
        ignore_index=True,
    )

    return updated, original_train, new_batch


def save_summary(original_train, new_batch, updated):

    rows = []

    for name, dataset in [
        ("original_training", original_train),
        ("post_drift_batch", new_batch),
        ("updated_training", updated),
    ]:

        attrition_count = int(
            (dataset["Attrition"] == 1).sum()
        )

        non_attrition_count = int(
            (dataset["Attrition"] == 0).sum()
        )

        rows.append({
            "dataset": name,
            "rows": len(dataset),
            "attrition": attrition_count,
            "non_attrition": non_attrition_count,
            "attrition_rate": round(
                attrition_count / len(dataset),
                6,
            ),
        })

    summary = pd.DataFrame(rows)

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(SUMMARY_PATH, index=False)

    return summary


def main():

    print("=" * 65)
    print("EMPLOYEE ATTRITION - UPDATED TRAINING DATASET")
    print("=" * 65)

    X_train, y_train, X_test = load_original_training_data()

    print(f"\nOriginal training shape: {X_train.shape}")

    X_new, y_new = create_post_drift_batch(
        X_train,
        y_train,
    )

    exact_overlap = validate_new_batch(
        X_new,
        y_new,
        X_test,
    )

    print("\nPOST-DRIFT BATCH")
    print("-" * 65)

    print(f"Rows             : {len(X_new)}")
    print(f"Attrition        : {(y_new == 1).sum()}")
    print(f"Non-attrition    : {(y_new == 0).sum()}")
    print(f"Test-set overlap : {exact_overlap}")

    updated, original_train, new_batch = create_updated_dataset(
        X_train,
        y_train,
        X_new,
        y_new,
    )

    NEW_BATCH_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    UPDATED_DATASET_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    new_batch.to_csv(NEW_BATCH_PATH, index=False)
    updated.to_csv(UPDATED_DATASET_PATH, index=False)

    summary = save_summary(
        original_train,
        new_batch,
        updated,
    )

    print("\nUPDATED DATASET")
    print("-" * 65)

    print(f"Total rows    : {len(updated)}")
    print(f"Total columns : {len(updated.columns)}")

    print("\nDATASET SUMMARY")
    print(summary.to_string(index=False))

    print("\nFILES CREATED")
    print(f"Post-drift batch : {NEW_BATCH_PATH}")
    print(f"Updated dataset  : {UPDATED_DATASET_PATH}")
    print(f"Summary          : {SUMMARY_PATH}")

    print("\n[SUCCESS] Updated training dataset created.")


if __name__ == "__main__":
    main()
