
import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from api.model_loading import ModelLoader
from api.preprocessing import clean_prediction_data


# Test dataset paths
X_TEST_PATH = Path("data/processed/X_test_raw.csv")
Y_TEST_PATH = Path("data/processed/y_test.npy")


# STEP 1: Load test data
def load_test_data():

    X_test = pd.read_csv(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH)

    if len(X_test) != len(y_test):
        raise ValueError(
            f"X_test and y_test size mismatch: "
            f"{len(X_test)} != {len(y_test)}"
        )

    return X_test, y_test


# STEP 2: Evaluate model performance on one window
def evaluate_window(model_loader, X, y):

    X_clean,emp = clean_prediction_data(X)

    X_transformed = model_loader.preprocessor.transform(X_clean)

    predictions = model_loader.model.predict(X_transformed)

    probabilities = model_loader.model.predict_proba(
        X_transformed
    )[:, 1]

    # ROC-AUC requires both classes to be present
    auc = (
        roc_auc_score(y, probabilities)
        if len(np.unique(y)) == 2
        else np.nan
    )

    return {
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(
            y, predictions, zero_division=0
        ),
        "recall": recall_score(
            y, predictions, zero_division=0
        ),
        "f1": f1_score(
            y, predictions, zero_division=0
        ),
        "roc_auc": auc,
    }


# STEP 3: Divide test data into windows
def create_windows(X, y, number_of_windows=3):

    X_windows = np.array_split(X, number_of_windows)
    y_windows = np.array_split(y, number_of_windows)

    return list(zip(X_windows, y_windows))


# STEP 4: Main function
def main():

    # Load test data
    X_test, y_test = load_test_data()

    # Load production model
    model_loader = ModelLoader()
    model_loader.load()

    if not model_loader.is_ready():
        raise RuntimeError(
            "Production model or preprocessor is not available"
        )

    # Create three evaluation windows
    windows = create_windows(
        X_test,
        y_test,
        number_of_windows=3
    )

    results = []

    # Evaluate each window
    for index, (X_window, y_window) in enumerate(
        windows, start=1
    ):

        metrics = evaluate_window(
            model_loader,
            X_window,
            y_window
        )

        results.append({
            "window": f"Window {index}",
            "samples": len(X_window),
            **metrics,
        })

    # Convert results into DataFrame
    results_df = pd.DataFrame(results)

    # Display model information
    print("\n" + "=" * 80)
    print("EMPLOYEE ATTRITION MODEL PERFORMANCE TREND ANALYSIS")
    print("=" * 80)

    print(f"\nModel name    : {model_loader.model_name}")
    print(f"Model version : {model_loader.model_version}")
    print(f"Run ID        : {model_loader.run_id}")

    # Display performance metrics
    print("\nPerformance across evaluation windows:\n")

    print(
        results_df.to_string(
            index=False,
            formatters={
                "accuracy": "{:.4f}".format,
                "precision": "{:.4f}".format,
                "recall": "{:.4f}".format,
                "f1": "{:.4f}".format,
                "roc_auc": lambda x: (
                    f"{x:.4f}" if pd.notna(x) else "N/A"
                ),
            },
        )
    )

    # Display recall trend
    print("\nRecall trend:")

    for _, row in results_df.iterrows():
        print(
            f"{row['window']:<10}: {row['recall']:.4f}"
        )

    # Calculate recall change
    recall_change = (
        results_df["recall"].iloc[-1]
        - results_df["recall"].iloc[0]
    )

    print(
        f"\nRecall change from Window 1 to Window 3: "
        f"{recall_change:+.4f}"
    )

    # Interpret recall trend
    if recall_change < 0:
        print(
            "Observation: Recall decreased across "
            "the evaluation windows."
        )

    elif recall_change > 0:
        print(
            "Observation: Recall increased across "
            "the evaluation windows."
        )

    else:
        print(
            "Observation: Recall remained unchanged "
            "across the evaluation windows."
        )


# STEP 5: Execute program
if __name__ == "__main__":
    main()
