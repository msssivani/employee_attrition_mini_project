
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------
# FILE PATHS
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DRIFT_PATH = Path(
    PROJECT_ROOT / "outputs" / "drift" / "data_drift_report.csv"
)

PREDICTION_DRIFT_PATH = Path(
    PROJECT_ROOT / "outputs" / "drift" / "prediction_drift_report.csv"
)

CONCEPT_DRIFT_PATH = Path(
    PROJECT_ROOT / "outputs" / "drift" / "concept_drift_report.csv"
)

OUTPUT_PATH = Path(
    PROJECT_ROOT / "outputs" / "drift" / "drift_analysis_report.csv"
)


# ---------------------------------------------------------
# LOAD REPORTS
# ---------------------------------------------------------

def load_reports():

    required_paths = [
        DATA_DRIFT_PATH,
        PREDICTION_DRIFT_PATH,
        CONCEPT_DRIFT_PATH,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Required drift report not found: {path}"
            )

    data_drift = pd.read_csv(DATA_DRIFT_PATH)
    prediction_drift = pd.read_csv(PREDICTION_DRIFT_PATH)
    concept_drift = pd.read_csv(CONCEPT_DRIFT_PATH)

    return data_drift, prediction_drift, concept_drift


# ---------------------------------------------------------
# ANALYZE DATA DRIFT
# ---------------------------------------------------------

def analyze_data_drift(data_drift):

    if not {"feature", "status"}.issubset(data_drift.columns):
        raise ValueError(
            "Data drift report must contain feature and status columns."
        )

    drifted = data_drift[
        data_drift["status"].astype(str).str.upper() == "DRIFT"
    ]

    return {
        "features_analyzed": len(data_drift),
        "features_with_drift": len(drifted),
        "drifted_features": ", ".join(
            drifted["feature"].astype(str).tolist()
        ),
    }


# ---------------------------------------------------------
# ANALYZE PREDICTION DRIFT
# ---------------------------------------------------------

def analyze_prediction_drift(prediction_drift):

    required_columns = [
        "prediction_distribution_status",
        "prediction_distribution_ks",
        "probability_distribution_status",
        "probability_distribution_ks",
    ]

    missing = [
        column
        for column in required_columns
        if column not in prediction_drift.columns
    ]

    if missing:
        raise ValueError(
            f"Prediction drift report is missing columns: {missing}"
        )

    if prediction_drift.empty:
        raise ValueError("Prediction drift report is empty.")

    row = prediction_drift.iloc[0]

    return {
        "prediction_distribution_status":
            str(row["prediction_distribution_status"]).upper(),

        "prediction_distribution_ks":
            float(row["prediction_distribution_ks"]),

        "probability_distribution_status":
            str(row["probability_distribution_status"]).upper(),

        "probability_distribution_ks":
            float(row["probability_distribution_ks"]),
    }


# ---------------------------------------------------------
# ANALYZE CONCEPT DRIFT
# ---------------------------------------------------------

def analyze_concept_drift(concept_drift):

    if not {"metric", "change"}.issubset(concept_drift.columns):
        raise ValueError(
            "Concept drift report must contain metric and change columns."
        )

    report = concept_drift.copy()

    # Negative change means the metric decreased.
    degraded = report[
        pd.to_numeric(report["change"], errors="coerce") < 0
    ]

    return {
        "metrics_analyzed": len(report),
        "metrics_degraded": len(degraded),
        "degraded_metrics": ", ".join(
            degraded["metric"].astype(str).tolist()
        ),
    }


# ---------------------------------------------------------
# CREATE COMBINED SUMMARY
# ---------------------------------------------------------

def create_summary(
    data_summary,
    prediction_summary,
    concept_summary,
):

    return pd.DataFrame([

        {
            "category": "Data Drift",
            "metric": "Features with drift",
            "value": data_summary["features_with_drift"],
            "status": (
                "DRIFT"
                if data_summary["features_with_drift"] > 0
                else "OK"
            ),
            "details": data_summary["drifted_features"],
        },

        {
            "category": "Prediction Drift",
            "metric": "Prediction distribution",
            "value": prediction_summary[
                "prediction_distribution_ks"
            ],
            "status": prediction_summary[
                "prediction_distribution_status"
            ],
            "details": "KS statistic",
        },

        {
            "category": "Prediction Drift",
            "metric": "Probability distribution",
            "value": prediction_summary[
                "probability_distribution_ks"
            ],
            "status": prediction_summary[
                "probability_distribution_status"
            ],
            "details": "KS statistic",
        },

        {
            "category": "Concept Drift",
            "metric": "Degraded performance metrics",
            "value": concept_summary["metrics_degraded"],
            "status": (
                "DRIFT"
                if concept_summary["metrics_degraded"] > 0
                else "OK"
            ),
            "details": concept_summary["degraded_metrics"],
        },

    ])


# ---------------------------------------------------------
# PRINT COMBINED ANALYSIS
# ---------------------------------------------------------

def print_analysis(
    data_summary,
    prediction_summary,
    concept_summary,
):

    print("=" * 70)
    print("EMPLOYEE ATTRITION - COMBINED DRIFT ANALYSIS")
    print("=" * 70)

    print("\n1. DATA DRIFT")
    print("-" * 70)

    print(
        f"Features analyzed : "
        f"{data_summary['features_analyzed']}"
    )

    print(
        f"Features drifted  : "
        f"{data_summary['features_with_drift']}"
    )

    print(
        f"Drifted features  : "
        f"{data_summary['drifted_features'] or 'None'}"
    )

    print("\n2. PREDICTION DRIFT")
    print("-" * 70)

    print(
        f"Prediction distribution : "
        f"{prediction_summary['prediction_distribution_status']}"
    )

    print(
        f"Prediction KS           : "
        f"{prediction_summary['prediction_distribution_ks']:.6f}"
    )

    print(
        f"Probability distribution: "
        f"{prediction_summary['probability_distribution_status']}"
    )

    print(
        f"Probability KS          : "
        f"{prediction_summary['probability_distribution_ks']:.6f}"
    )

    print("\n3. CONCEPT DRIFT")
    print("-" * 70)

    print(
        f"Metrics analyzed : "
        f"{concept_summary['metrics_analyzed']}"
    )

    print(
        f"Metrics degraded : "
        f"{concept_summary['metrics_degraded']}"
    )

    print(
        f"Degraded metrics : "
        f"{concept_summary['degraded_metrics'] or 'None'}"
    )

    print("\n" + "=" * 70)

    drift_detected = (
        data_summary["features_with_drift"] > 0
        or prediction_summary["prediction_distribution_status"] == "DRIFT"
        or prediction_summary["probability_distribution_status"] == "DRIFT"
        or concept_summary["metrics_degraded"] > 0
    )

    if drift_detected:
        print(
            "OVERALL STATUS : DRIFT / MODEL BEHAVIOR CHANGE OBSERVED"
        )
    else:
        print("OVERALL STATUS : NO DRIFT FLAGS OBSERVED")

    print("=" * 70)


# ---------------------------------------------------------
# MAIN FUNCTION
# ---------------------------------------------------------

def main():

    data_drift, prediction_drift, concept_drift = load_reports()

    data_summary = analyze_data_drift(data_drift)

    prediction_summary = analyze_prediction_drift(
        prediction_drift
    )

    concept_summary = analyze_concept_drift(
        concept_drift
    )

    print_analysis(
        data_summary,
        prediction_summary,
        concept_summary,
    )

    summary = create_summary(
        data_summary,
        prediction_summary,
        concept_summary,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(f"\nCombined report saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
