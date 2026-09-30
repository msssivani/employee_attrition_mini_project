import os
import json
import joblib
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    GridSearchCV
)

from imblearn.over_sampling import SMOTE


def train_models():

    print("Starting Model Training...")

    # 1. Load processed training data
    X_train = np.load(
        'data/processed/X_train_final.npy'
    )

    y_train = np.load(
        'data/processed/y_train.npy'
    )

    print("Training data loaded successfully.")
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)

    # ---------------------------------------------------------
    # 2. Define baseline models
    # ---------------------------------------------------------

    lr = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    lr_balanced = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )

    dt = DecisionTreeClassifier(
        random_state=42
    )

    rf = RandomForestClassifier(
        random_state=42
    )

    svm = SVC(
        random_state=42
    )

    svm_balanced = SVC(
        random_state=42,
        class_weight='balanced'
    )

    # ---------------------------------------------------------
    # 3. Train baseline models
    # ---------------------------------------------------------

    print("\nTraining Logistic Regression...")
    lr.fit(X_train, y_train)

    print("Training Balanced Logistic Regression...")
    lr_balanced.fit(X_train, y_train)

    print("Training Decision Tree...")
    dt.fit(X_train, y_train)

    print("Training Random Forest...")
    rf.fit(X_train, y_train)

    print("Training SVM...")
    svm.fit(X_train, y_train)

    print("Training Balanced SVM...")
    svm_balanced.fit(X_train, y_train)

    # ---------------------------------------------------------
    # 4. Handle class imbalance using SMOTE
    # ---------------------------------------------------------

    print("\nApplying SMOTE...")

    print("Original class distribution:")
    unique, counts = np.unique(
        y_train,
        return_counts=True
    )

    for label, count in zip(unique, counts):
        print(f"Class {label}: {count}")

    smote = SMOTE(
        random_state=42
    )

    X_train_smote, y_train_smote = smote.fit_resample(
        X_train,
        y_train
    )

    print("\nClass distribution after SMOTE:")
    unique, counts = np.unique(
        y_train_smote,
        return_counts=True
    )

    for label, count in zip(unique, counts):
        print(f"Class {label}: {count}")

    # ---------------------------------------------------------
    # 5. Train models using SMOTE data
    # ---------------------------------------------------------

    print("\nTraining Logistic Regression with SMOTE...")
    lr_smote = LogisticRegression(
        max_iter=1000,
        random_state=42
    )
    lr_smote.fit(
        X_train_smote,
        y_train_smote
    )

    print("Training Decision Tree with SMOTE...")
    dt_smote = DecisionTreeClassifier(
        random_state=42
    )
    dt_smote.fit(
        X_train_smote,
        y_train_smote
    )

    print("Training Random Forest with SMOTE...")
    rf_smote = RandomForestClassifier(
        random_state=42
    )
    rf_smote.fit(
        X_train_smote,
        y_train_smote
    )

    print("Training SVM with SMOTE...")
    svm_smote = SVC(
        random_state=42
    )
    svm_smote.fit(
        X_train_smote,
        y_train_smote
    )

    # ---------------------------------------------------------
    # 6. Stratified K-Fold Cross Validation
    # ---------------------------------------------------------

    print("\nStarting Stratified K-Fold Cross Validation...")

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1"
    }

    cv_models = {
        "Logistic Regression": lr,
        "Balanced Logistic Regression": lr_balanced,
        "Balanced SVM": svm_balanced
    }

    cv_results = {}

    for name, model in cv_models.items():

        print(f"\n{name}")

        scores = cross_validate(
            model,
            X_train,
            y_train,
            scoring=scoring,
            cv=skf
        )

        cv_results[name] = {}

        for metric in scoring:

            values = scores[f"test_{metric}"]

            mean_score = values.mean()
            std_score = values.std()

            cv_results[name][metric] = {
                "scores": values.tolist(),
                "mean": float(mean_score),
                "std": float(std_score)
            }

            print(
                f"{metric}: "
                f"{mean_score:.4f} "
                f"(± {std_score:.4f})"
            )

    # ---------------------------------------------------------
    # 7. Hyperparameter tuning - Balanced Logistic Regression
    # ---------------------------------------------------------

    print("\nTuning Balanced Logistic Regression...")

    lr_param_grid = {
        "C": [0.01, 0.1, 1, 10, 100]
    }

    grid_lr = GridSearchCV(
        estimator=lr_balanced,
        param_grid=lr_param_grid,
        cv=skf,
        scoring="f1",
        n_jobs=-1
    )

    grid_lr.fit(
        X_train,
        y_train
    )

    print(
        "Best Logistic Regression parameters:",
        grid_lr.best_params_
    )

    print(
        "Best Logistic Regression F1:",
        grid_lr.best_score_
    )

    # ---------------------------------------------------------
    # 8. Hyperparameter tuning - Balanced SVM
    # ---------------------------------------------------------

    print("\nTuning Balanced SVM...")

    svm_param_grid = {
        "C": [0.01, 0.1, 1, 10, 100]
    }

    grid_svm = GridSearchCV(
        estimator=svm_balanced,
        param_grid=svm_param_grid,
        cv=skf,
        scoring="f1",
        n_jobs=-1
    )

    grid_svm.fit(
        X_train,
        y_train
    )

    print(
        "Best SVM parameters:",
        grid_svm.best_params_
    )

    print(
        "Best SVM F1:",
        grid_svm.best_score_
    )

    # ---------------------------------------------------------
    # 9. Select final model
    # ---------------------------------------------------------

    best_lr = grid_lr.best_estimator_
    best_svm = grid_svm.best_estimator_

    print("\nFinal model selected:")
    print("Balanced SVM")
    print("Parameters:", grid_svm.best_params_)

    # ---------------------------------------------------------
    # 10. Save final model
    # ---------------------------------------------------------

    os.makedirs(
        'models',
        exist_ok=True
    )

    joblib.dump(
        best_svm,
        'models/best_model.pkl'
    )

    print(
        "\nFinal model saved to:",
        'models/best_model.pkl'
    )

    # ---------------------------------------------------------
    # 11. Save training results
    # ---------------------------------------------------------

    os.makedirs(
        'reports',
        exist_ok=True
    )

    training_results = {
        "cross_validation": cv_results,

        "balanced_logistic_regression": {
            "best_parameters": grid_lr.best_params_,
            "best_f1": float(grid_lr.best_score_)
        },

        "balanced_svm": {
            "best_parameters": grid_svm.best_params_,
            "best_f1": float(grid_svm.best_score_)
        },

        "selected_model": "Balanced SVM",
        "selected_model_parameters": grid_svm.best_params_
    }

    with open(
        'reports/model_training_results.json',
        'w'
    ) as f:

        json.dump(
            training_results,
            f,
            indent=4
        )

    print(
        "Training results saved to:",
        'reports/model_training_results.json'
    )

    print("\nModel training completed successfully!")


if __name__ == "__main__":
    train_models()