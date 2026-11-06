
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import ks_2samp


# ---------------------------------------------------------
# FILE PATHS
# ---------------------------------------------------------

BASELINE_PATH = Path(
    "data/processed/X_test_raw.csv"
)

EVOLVED_PATH = Path(
    "data/monitoring/drift/evolved_data.csv"
)

REPORT_PATH = Path(
    "outputs/drift/data_drift_report.csv"
)


# ---------------------------------------------------------
# FEATURE COLUMNS
# ---------------------------------------------------------

# 14 numerical employee features
NUMERICAL_COLUMNS = [
    "Age",
    "DailyRate",
    "DistanceFromHome",
    "HourlyRate",
    "MonthlyIncome",
    "MonthlyRate",
    "NumCompaniesWorked",
    "PercentSalaryHike",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
]

# 9 ordinal employee features
ORDINAL_COLUMNS = [
    "Education",
    "EnvironmentSatisfaction",
    "JobInvolvement",
    "JobLevel",
    "JobSatisfaction",
    "RelationshipSatisfaction",
    "StockOptionLevel",
    "WorkLifeBalance",
    "PerformanceRating",
]

# 7 categorical employee features
CATEGORICAL_COLUMNS = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "OverTime",
]


KS_THRESHOLD = 0.10
JS_THRESHOLD = 0.10


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

def load_data():

    baseline = pd.read_csv(BASELINE_PATH)
    evolved = pd.read_csv(EVOLVED_PATH)

    if baseline.shape != evolved.shape:
        raise ValueError(
            f"Dataset shape mismatch: "
            f"baseline={baseline.shape}, "
            f"evolved={evolved.shape}"
        )

    if list(baseline.columns) != list(evolved.columns):
        raise ValueError(
            "Feature columns do not match between "
            "baseline and evolved datasets."
        )

    required_columns = (
        NUMERICAL_COLUMNS
        + ORDINAL_COLUMNS
        + CATEGORICAL_COLUMNS
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in baseline.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing expected feature columns: {missing_columns}"
        )

    return baseline, evolved


# ---------------------------------------------------------
# CALCULATE JENSEN-SHANNON DIVERGENCE
# ---------------------------------------------------------

def calculate_js_divergence(baseline, evolved):

    # Treat missing values as a separate category.
    baseline = baseline.fillna("__MISSING__").astype(str)
    evolved = evolved.fillna("__MISSING__").astype(str)

    categories = sorted(
        set(baseline.unique()) | set(evolved.unique())
    )

    baseline_counts = (
        baseline.value_counts(normalize=True)
        .reindex(categories, fill_value=0)
        .values
    )

    evolved_counts = (
        evolved.value_counts(normalize=True)
        .reindex(categories, fill_value=0)
        .values
    )

    baseline_counts = np.asarray(
        baseline_counts, dtype=float
    )

    evolved_counts = np.asarray(
        evolved_counts, dtype=float
    )

    midpoint = (
        baseline_counts + evolved_counts
    ) / 2

    baseline_mask = baseline_counts > 0
    evolved_mask = evolved_counts > 0

    kl_baseline = np.sum(
        baseline_counts[baseline_mask]
        * np.log2(
            baseline_counts[baseline_mask]
            / midpoint[baseline_mask]
        )
    )

    kl_evolved = np.sum(
        evolved_counts[evolved_mask]
        * np.log2(
            evolved_counts[evolved_mask]
            / midpoint[evolved_mask]
        )
    )

    js_score = (
        kl_baseline + kl_evolved
    ) / 2

    return float(js_score)


# ---------------------------------------------------------
# ANALYZE NUMERICAL AND ORDINAL DRIFT
# ---------------------------------------------------------

def analyze_numerical_drift(baseline, evolved):

    results = []

    columns_to_analyze = (
        NUMERICAL_COLUMNS + ORDINAL_COLUMNS
    )

    for column in columns_to_analyze:

        baseline_values = baseline[column].dropna()
        evolved_values = evolved[column].dropna()

        if len(baseline_values) == 0 or len(evolved_values) == 0:
            raise ValueError(
                f"Cannot analyze {column}: "
                "one dataset contains no valid values."
            )

        statistic, p_value = ks_2samp(
            baseline_values,
            evolved_values,
        )

        status = (
            "DRIFT"
            if statistic >= KS_THRESHOLD
            else "OK"
        )

        feature_type = (
            "ordinal"
            if column in ORDINAL_COLUMNS
            else "numerical"
        )

        results.append({
            "feature": column,
            "type": feature_type,
            "metric": "KS",
            "score": round(float(statistic), 6),
            "p_value": round(float(p_value), 6),
            "threshold": KS_THRESHOLD,
            "status": status,
        })

    return results


# ---------------------------------------------------------
# ANALYZE CATEGORICAL DRIFT
# ---------------------------------------------------------

def analyze_categorical_drift(baseline, evolved):

    results = []

    for column in CATEGORICAL_COLUMNS:

        score = calculate_js_divergence(
            baseline[column],
            evolved[column],
        )

        status = (
            "DRIFT"
            if score >= JS_THRESHOLD
            else "OK"
        )

        results.append({
            "feature": column,
            "type": "categorical",
            "metric": "JS",
            "score": round(float(score), 6),
            "p_value": None,
            "threshold": JS_THRESHOLD,
            "status": status,
        })

    return results


# ---------------------------------------------------------
# CREATE DATA DRIFT REPORT
# ---------------------------------------------------------

def create_drift_report(baseline, evolved):

    results = []

    results.extend(
        analyze_numerical_drift(baseline, evolved)
    )

    results.extend(
        analyze_categorical_drift(baseline, evolved)
    )

    report = pd.DataFrame(results)

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report.to_csv(
        REPORT_PATH,
        index=False,
    )

    return report


# ---------------------------------------------------------
# PRINT REPORT
# ---------------------------------------------------------

def print_report(report):

    print("=" * 70)
    print("EMPLOYEE ATTRITION - DATA DRIFT ANALYSIS")
    print("=" * 70)

    print(f"Baseline dataset : {BASELINE_PATH}")
    print(f"Evolved dataset  : {EVOLVED_PATH}")

    print("\nDRIFT ANALYSIS RESULTS")
    print("-" * 70)

    print(report.to_string(index=False))

    drift_count = (
        report["status"] == "DRIFT"
    ).sum()

    total_features = len(report)

    print("\n" + "=" * 70)

    print(f"Features analyzed : {total_features}")
    print(f"Drift detected    : {drift_count}")
    print(f"Features without drift : {total_features - drift_count}")

    print(f"Report saved      : {REPORT_PATH}")

    print("=" * 70)


# ---------------------------------------------------------
# MAIN FUNCTION
# ---------------------------------------------------------

def main():

    baseline, evolved = load_data()

    print(f"Baseline shape : {baseline.shape}")
    print(f"Evolved shape  : {evolved.shape}")
    print()

    report = create_drift_report(
        baseline,
        evolved,
    )

    print_report(report)


if __name__ == "__main__":
    main()
