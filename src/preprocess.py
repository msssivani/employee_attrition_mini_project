import pandas as pd
import numpy as np
import json
import joblib

from sklearn.model_selection import train_test_split
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
# 6. Handle missing values
# ============================================================

# Numerical features → median imputation
numerical_imputer = SimpleImputer(strategy="median")

X_train_num = numerical_imputer.fit_transform(
    X_train[numerical_cols]
)

X_test_num = numerical_imputer.transform(
    X_test[numerical_cols]
)


# Ordinal features → median imputation
ordinal_imputer = SimpleImputer(strategy="median")

X_train_ord = ordinal_imputer.fit_transform(
    X_train[ordinal_cols]
)

X_test_ord = ordinal_imputer.transform(
    X_test[ordinal_cols]
)


# Categorical features → most frequent value
categorical_imputer = SimpleImputer(
    strategy="most_frequent"
)

X_train_cat = categorical_imputer.fit_transform(
    X_train[categorical_cols]
)

X_test_cat = categorical_imputer.transform(
    X_test[categorical_cols]
)


print("[INFO] Missing-value handling completed.")


# ============================================================
# 7. Scale numerical features
# ============================================================

scaler = StandardScaler()

X_train_num = scaler.fit_transform(X_train_num)
X_test_num = scaler.transform(X_test_num)

print("[INFO] Numerical features scaled.")


# ============================================================
# 8. Encode categorical features
# ============================================================

ohe = OneHotEncoder(
    drop="first",
    handle_unknown="ignore",
    sparse_output=False
)

X_train_cat = ohe.fit_transform(X_train_cat)
X_test_cat = ohe.transform(X_test_cat)

print("[INFO] Categorical features encoded.")


# ============================================================
# 9. Combine all features
# ============================================================

X_train_final = np.hstack([
    X_train_num,
    X_train_ord,
    X_train_cat
])

X_test_final = np.hstack([
    X_test_num,
    X_test_ord,
    X_test_cat
])

print(f"[INFO] Final training shape: {X_train_final.shape}")
print(f"[INFO] Final testing shape: {X_test_final.shape}")


# ============================================================
# 10. Save processed data
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
# 11. Save preprocessing objects
# ============================================================

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

joblib.dump(
    ohe,
    "models/ohe.pkl"
)

joblib.dump(
    numerical_imputer,
    "models/numerical_imputer.pkl"
)

joblib.dump(
    ordinal_imputer,
    "models/ordinal_imputer.pkl"
)

joblib.dump(
    categorical_imputer,
    "models/categorical_imputer.pkl"
)

print("[INFO] Preprocessing objects saved.")


# ============================================================
# 12. Save metadata
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

    "numerical_handling": "median imputation + StandardScaler",

    "ordinal_handling": "median imputation + unchanged ordinal values",

    "categorical_handling": (
        "most frequent imputation + "
        "OneHotEncoder(drop='first', handle_unknown='ignore')"
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


print("[SUCCESS] Preprocessing completed successfully!")