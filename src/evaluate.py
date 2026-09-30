import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def run_evaluation():

    print("Starting Model Evaluation...")

    # 1. Load test data and the trained final model
    X_test_final = np.load(
        'data/processed/X_test_final.npy'
    )

    y_test = np.load(
        'data/processed/y_test.npy'
    )

    model = joblib.load(
        'models/best_model.pkl'
    )

    print("Test data and final model loaded successfully.")

    # 2. Make predictions
    y_pred = model.predict(
        X_test_final
    )

    # 3. Calculate evaluation metrics
    acc = accuracy_score(
        y_test,
        y_pred
    )

    prec = precision_score(
        y_test,
        y_pred
    )

    rec = recall_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred
    )

    print("\n--- Model Evaluation Report ---")

    print(
        f"Accuracy : {acc:.4f}"
    )

    print(
        f"Precision: {prec:.4f}"
    )

    print(
        f"Recall   : {rec:.4f}"
    )

    print(
        f"F1-Score : {f1:.4f}"
    )

    print(
        "-------------------------------\n"
    )

    # 4. Error Analysis
    # Load the original dataset to recover
    # the original employee records
    df = pd.read_csv(
        'data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv'
    )

    # Recreate the same train-test split used during preprocessing
    _, X_test_raw = train_test_split(
        df.drop('Attrition', axis=1),
        test_size=0.2,
        random_state=42,
        stratify=df['Attrition']
    )

    # Create error analysis dataframe
    errors_df = X_test_raw.copy()

    errors_df['Actual_Attrition'] = y_test

    errors_df['Predicted_Attrition'] = y_pred

    # 5. False Negatives
    # Actual = 1 (employee left)
    # Predicted = 0 (model predicted employee would stay)
    false_negatives = errors_df[
        (errors_df['Actual_Attrition'] == 1) &
        (errors_df['Predicted_Attrition'] == 0)
    ]

    # 6. False Positives
    # Actual = 0 (employee stayed)
    # Predicted = 1 (model predicted employee would leave)
    false_positives = errors_df[
        (errors_df['Actual_Attrition'] == 0) &
        (errors_df['Predicted_Attrition'] == 1)
    ]

    # 7. Save error analysis results
    false_negatives.to_csv(
        'outputs/false_negatives.csv',
        index=False
    )

    false_positives.to_csv(
        'outputs/false_positives.csv',
        index=False
    )

    print(
        "False negatives saved to:",
        'outputs/false_negatives.csv'
    )

    print(
        "False positives saved to:",
        'outputs/false_positives.csv'
    )

    print(
        "\nEvaluation complete and error analysis files saved!"
    )


if __name__ == "__main__":
    run_evaluation()