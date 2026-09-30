import os
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def train_and_register_model():

    print(
        "[INFO] --- Starting MLflow Run "
        "with Model Registry ---"
    )

    # ---------------------------------------------------------
    # 1. Set up MLflow
    # ---------------------------------------------------------

    mlflow.set_experiment(
        "Employee_Attrition_Prediction"
    )

    # ---------------------------------------------------------
    # 2. Load Processed Data
    # ---------------------------------------------------------

    try:

        X_train = np.load(
            'data/processed/X_train_final.npy'
        )

        X_test = np.load(
            'data/processed/X_test_final.npy'
        )

        y_train = np.load(
            'data/processed/y_train.npy'
        )

        y_test = np.load(
            'data/processed/y_test.npy'
        )

    except FileNotFoundError:

        print(
            "[ERROR] Processed data not found. "
            "Please run the Lab 5 pipeline first."
        )

        return

    # ---------------------------------------------------------
    # 3. Define Balanced SVM Parameters
    # ---------------------------------------------------------

    params = {
        "C": 1.0,
        "kernel": "rbf",
        "class_weight": "balanced",
        "probability": True,
        "random_state": 42
    }

    # ---------------------------------------------------------
    # 4. Start MLflow Run
    # ---------------------------------------------------------

    with mlflow.start_run(
        run_name="Balanced_SVM_Registry_V1"
    ) as run:

        # -----------------------------------------------------
        # 5. Log Parameters
        # -----------------------------------------------------

        mlflow.log_params(
            params
        )

        mlflow.log_param(
            "model_family",
            "SVM"
        )

        mlflow.log_param(
            "class_imbalance_handling",
            "class_weight='balanced'"
        )

        mlflow.log_param(
            "problem_type",
            "Employee Attrition Classification"
        )

        # -----------------------------------------------------
        # 6. Train Model
        # -----------------------------------------------------

        print(
            "[INFO] Training Balanced SVM model..."
        )

        svm_model = SVC(
            **params
        )

        svm_model.fit(
            X_train,
            y_train
        )

        # -----------------------------------------------------
        # 7. Evaluate Model
        # -----------------------------------------------------

        print(
            "[INFO] Evaluating model..."
        )

        y_pred = svm_model.predict(
            X_test
        )

        y_proba = svm_model.predict_proba(
            X_test
        )[:, 1]

        metrics = {

            "accuracy":
                accuracy_score(
                    y_test,
                    y_pred
                ),

            "precision":
                precision_score(
                    y_test,
                    y_pred
                ),

            "recall":
                recall_score(
                    y_test,
                    y_pred
                ),

            "f1_score":
                f1_score(
                    y_test,
                    y_pred
                ),

            "roc_auc":
                roc_auc_score(
                    y_test,
                    y_proba
                )
        }

        mlflow.log_metrics(
            metrics
        )

        # -----------------------------------------------------
        # 8. Log Preprocessing Pipeline
        # -----------------------------------------------------

        preprocessor_path = (
            "models/preprocessor.pkl"
        )

        if os.path.exists(
            preprocessor_path
        ):

            print(
                "[INFO] Attaching preprocessing pipeline "
                "to model artifacts..."
            )

            mlflow.log_artifact(
                preprocessor_path,
                artifact_path="preprocessing_pipeline"
            )

        else:

            print(
                "[WARNING] Preprocessor artifact not found. "
                "Model will still be registered."
            )

        # -----------------------------------------------------
        # 9. Log and Register Model
        # -----------------------------------------------------

        print(
            "[INFO] Pushing model to MLflow Registry..."
        )

        mlflow.sklearn.log_model(

            sk_model=svm_model,

            name="balanced_svm_model",

            registered_model_name=
                "Employee_Attrition_Production_Model"
        )

        # -----------------------------------------------------
        # 10. Print Results
        # -----------------------------------------------------

        print(
            f"[SUCCESS] Metrics logged: "
            f"F1 = {metrics['f1_score']:.4f} | "
            f"ROC-AUC = {metrics['roc_auc']:.4f}"
        )

        print(
            "[SUCCESS] Model successfully registered under "
            "name: 'Employee_Attrition_Production_Model'"
        )


if __name__ == "__main__":

    train_and_register_model()