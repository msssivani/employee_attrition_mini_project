
import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from scipy.stats import ks_2samp

from api.model_loading import ModelLoader
from api.preprocessing import clean_prediction_data


# Dataset paths
BASELINE_PATH = Path("data/processed/X_test_raw.csv")
EVOLVED_PATH = Path("data/monitoring/drift/evolved_data.csv")
REPORT_PATH = Path("outputs/drift/prediction_drift_report.csv")

# KS statistic threshold
KS_THRESHOLD = 0.10


# STEP 1: Load baseline and evolved datasets
def load_data():

    baseline = pd.read_csv(BASELINE_PATH)
    evolved = pd.read_csv(EVOLVED_PATH)

    if baseline.shape != evolved.shape:
        raise ValueError(
            f"Dataset shape mismatch: "
            f"baseline={baseline.shape}, evolved={evolved.shape}"
        )

    if baseline.empty or evolved.empty:
        raise ValueError("Baseline or evolved dataset is empty")

    return baseline, evolved


# STEP 2: Generate predictions
def predict(model_loader, data):

    clean_data,emp_no = clean_prediction_data(data)

    transformed_data = model_loader.preprocessor.transform(
        clean_data
    )

    predictions = model_loader.model.predict(
        transformed_data
    )

    probabilities = model_loader.model.predict_proba(
        transformed_data
    )[:, 1]

    return predictions, probabilities


# STEP 3: Summarize predictions
def summarize_predictions(name, predictions, probabilities):

    print()
    print(f"{name} PREDICTIONS")
    print("-" * 60)

    total = len(predictions)

    attrition_count = int((predictions == 1).sum())
    no_attrition_count = int((predictions == 0).sum())

    print(f"Total predictions  : {total}")

    print(
        f"Attrition (1)      : {attrition_count} "
        f"({attrition_count / total * 100:.2f}%)"
    )

    print(
        f"No Attrition (0)   : {no_attrition_count} "
        f"({no_attrition_count / total * 100:.2f}%)"
    )

    print(
        f"Mean probability   : {probabilities.mean():.6f}"
    )

    print(
        f"Median probability : {np.median(probabilities):.6f}"
    )

    print(
        f"Std probability    : {probabilities.std():.6f}"
    )

    print(
        f"Min probability    : {probabilities.min():.6f}"
    )

    print(
        f"Max probability    : {probabilities.max():.6f}"
    )


# STEP 4: Analyze prediction drift using KS test
def analyze_prediction_drift(
    baseline_predictions,
    evolved_predictions,
    baseline_probabilities,
    evolved_probabilities,
):

    # Compare predicted class distributions
    prediction_ks, prediction_p = ks_2samp(
        baseline_predictions,
        evolved_predictions,
    )

    # Compare predicted probability distributions
    probability_ks, probability_p = ks_2samp(
        baseline_probabilities,
        evolved_probabilities,
    )

    # Determine drift status
    prediction_drift = prediction_ks >= KS_THRESHOLD

    probability_drift = probability_ks >= KS_THRESHOLD

    return {
        "prediction_distribution_ks": float(prediction_ks),

        "prediction_distribution_p_value": float(prediction_p),

        "prediction_distribution_status":
            "DRIFT" if prediction_drift else "OK",

        "probability_distribution_ks": float(probability_ks),

        "probability_distribution_p_value": float(probability_p),

        "probability_distribution_status":
            "DRIFT" if probability_drift else "OK",
    }


# STEP 5: Save drift report
def save_report(results):

    report = pd.DataFrame([results])

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    report.to_csv(
        REPORT_PATH,
        index=False
    )

    return report


# STEP 6: Main function
def main():

    print("=" * 70)
    print("EMPLOYEE ATTRITION PREDICTION DRIFT ANALYSIS")
    print("=" * 70)

    # Load datasets
    baseline, evolved = load_data()

    print(f"Baseline dataset : {BASELINE_PATH}")
    print(f"Evolved dataset  : {EVOLVED_PATH}")
    print(f"Dataset shape    : {baseline.shape}")

    # Load production model
    model_loader = ModelLoader()
    model_loader.load()

    if not model_loader.is_ready():
        raise RuntimeError(
            "Production model or preprocessor is not available"
        )

    # Display model information
    print()
    print("Model information")
    print("-" * 60)

    print(f"Model name : {model_loader.model_name}")
    print(f"Version    : {model_loader.model_version}")
    print(f"Run ID     : {model_loader.run_id}")

    # Generate baseline predictions
    baseline_predictions, baseline_probabilities = predict(
        model_loader,
        baseline,
    )

    # Generate evolved predictions
    evolved_predictions, evolved_probabilities = predict(
        model_loader,
        evolved,
    )

    # Display baseline statistics
    summarize_predictions(
        "BASELINE",
        baseline_predictions,
        baseline_probabilities,
    )

    # Display evolved statistics
    summarize_predictions(
        "EVOLVED",
        evolved_predictions,
        evolved_probabilities,
    )

    # Analyze prediction drift
    results = analyze_prediction_drift(
        baseline_predictions,
        evolved_predictions,
        baseline_probabilities,
        evolved_probabilities,
    )

    # Display results
    print()
    print("=" * 70)
    print("PREDICTION DRIFT RESULTS")
    print("=" * 70)

    print(
        f"Prediction distribution KS : "
        f"{results['prediction_distribution_ks']:.6f}"
    )

    print(
        f"Prediction distribution p  : "
        f"{results['prediction_distribution_p_value']:.6f}"
    )

    print(
        f"Prediction distribution    : "
        f"{results['prediction_distribution_status']}"
    )

    print()

    print(
        f"Probability distribution KS : "
        f"{results['probability_distribution_ks']:.6f}"
    )

    print(
        f"Probability distribution p  : "
        f"{results['probability_distribution_p_value']:.6f}"
    )

    print(
        f"Probability distribution    : "
        f"{results['probability_distribution_status']}"
    )

    print()
    print(f"Threshold : KS >= {KS_THRESHOLD}")
    print(f"Report    : {REPORT_PATH}")

    # Save report
    save_report(results)

    print("\nDrift report saved successfully!")


# STEP 7: Execute program
if __name__ == "__main__":
    main()
