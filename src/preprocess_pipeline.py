import pandas as pd
import numpy as np
import json
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


# ============================================================
# 1. Load dataset
# ============================================================

DATA_PATH = "data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv"

df = pd.read_csv(DATA_PATH)

print("[INFO] Dataset loaded successfully.")
print(f"[INFO] Dataset shape: {df.shape}")


# ============================================================
# 2. Drop unnecessary columns
# ============================================================

drop_cols = [
    "EmployeeCount",
    "EmployeeNumber",
    "Over18",
    "StandardHours"
]

df = df.drop(columns=drop_cols)

print("[INFO] Dropped unnecessary columns.")


# ============================================================
# 3. Separate target
# ============================================================

df["Attrition"] = df["Attrition"].map({
    "Yes": 1,
    "No": 0
})

X = df.drop(columns=["Attrition"])
y = df["Attrition"]


# ============================================================
# 4. Define feature groups
# ============================================================

numerical_cols = [
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
    "YearsWithCurrManager"
]

ordinal_cols = [
    "Education",
    "EnvironmentSatisfaction",
    "JobInvolvement",
    "JobLevel",
    "JobSatisfaction",
    "RelationshipSatisfaction",
    "StockOptionLevel",
    "WorkLifeBalance",
    "PerformanceRating"
]

categorical_cols = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "OverTime"
]


# ============================================================
# 5. Train-test split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
    shuffle=True
)

print(f"[INFO] Training data: {X_train.shape}")
print(f"[INFO] Testing data: {X_test.shape}")


# ============================================================
# 6. Create preprocessing pipelines
# ============================================================

# Numerical:
# Missing values → median
# Then → StandardScaler

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# Ordinal:
# Missing values → median
# Then → keep ordinal values unchanged

ordinal_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])


# Categorical:
# Missing values → most frequent
# Then → OneHotEncoder

categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(
            drop="first",
            handle_unknown="ignore",
            sparse_output=False
        )
    )
])


# ============================================================
# 7. Combine preprocessing pipelines
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numerical_pipeline,
            numerical_cols
        ),
        (
            "ordinal",
            ordinal_pipeline,
            ordinal_cols
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_cols
        )
    ]
)


# ============================================================
# 8. Fit preprocessing only on training data
# ============================================================

X_train_final = preprocessor.fit_transform(X_train)

X_test_final = preprocessor.transform(X_test)

print("[INFO] Preprocessing completed.")

print(
    f"[INFO] Final training shape: "
    f"{X_train_final.shape}"
)

print(
    f"[INFO] Final testing shape: "
    f"{X_test_final.shape}"
)


# ============================================================
# 9. Save processed data
# ============================================================

np.save(
    "data/processed/X_train_final.npy",
    X_train_final
)

np.save(
    "data/processed/X_test_final.npy",
    X_test_final
)

np.save(
    "data/processed/y_train.npy",
    y_train.to_numpy()
)

np.save(
    "data/processed/y_test.npy",
    y_test.to_numpy()
)


# ============================================================
# 10. Save complete preprocessing pipeline
# ============================================================

joblib.dump(
    preprocessor,
    "models/preprocessor.pkl"
)

print(
    "[INFO] Complete preprocessing pipeline "
    "saved to models/preprocessor.pkl"
)


# ============================================================
# 11. Save metadata
# ============================================================

metadata = {
    "dropped_columns": drop_cols,

    "numerical_columns": numerical_cols,

    "ordinal_columns": ordinal_cols,

    "categorical_columns": categorical_cols,

    "missing_value_handling": {
        "numerical": "median",
        "ordinal": "median",
        "categorical": "most_frequent"
    },

    "numerical_handling": (
        "median imputation + StandardScaler"
    ),

    "ordinal_handling": (
        "median imputation + unchanged ordinal values"
    ),

    "categorical_handling": (
        "most frequent imputation + "
        "OneHotEncoder(drop='first', "
        "handle_unknown='ignore')"
    ),

    "train_test_split": {
        "test_size": 0.2,
        "random_state": 42,
        "stratify": True,
        "shuffle": True
    }
}

with open(
    "data/processed/dataset_metadata.json",
    "w"
) as f:

    json.dump(
        metadata,
        f,
        indent=4
    )


print("[SUCCESS] Pipeline preprocessing completed successfully!")