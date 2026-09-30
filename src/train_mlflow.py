import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)


def train_and_track(
    run_name="Balanced_SVM_Baseline",
    params=None
):

    if params is None:
        params = {
            "C": 1.0,
            "kernel": "rbf",
            "class_weight": "balanced",
            "probability": True,
            "random_state": 42
        }

    print(
        f"\n--- Starting MLflow Run: {run_name} ---"
    )

    # ---------------------------------------------------------
    # 1. Load Processed Data
    # ---------------------------------------------------------

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

    print("Processed data loaded successfully.")

    # ---------------------------------------------------------
    # 2. Set MLflow Experiment
    # ---------------------------------------------------------

    mlflow.set_experiment(
        "Employee_Attrition_Prediction"
    )

    with mlflow.start_run(
        run_name=run_name
    ):

        # -----------------------------------------------------
        # 3. Log Parameters
        # --------------------------------------------- --------

        mlflow.log_params(params)

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
        # 4. Train Balanced SVM
        # -----------------------------------------------------

        model = SVC(
            **params
        )

        model.fit(
            X_train,
            y_train
        )

        print(
            "Balanced SVM training completed."
        )

        # -----------------------------------------------------
        # 5. Make Predictions
        # -----------------------------------------------------

        y_pred = model.predict(
            X_test
        )

        y_prob = model.predict_proba(
            X_test
        )[:, 1]

        # -----------------------------------------------------
        # 6. Calculate Metrics
        # -----------------------------------------------------

        metrics = {
            "accuracy": accuracy_score(
                y_test,
                y_pred
            ),

            "precision": precision_score(
                y_test,
                y_pred
            ),

            "recall": recall_score(
                y_test,
                y_pred
            ),

            "f1_score": f1_score(
                y_test,
                y_pred
            ),

            "roc_auc": roc_auc_score(
                y_test,
                y_prob
            )
        }

        # -----------------------------------------------------
        # 7. Log Metrics
        # -----------------------------------------------------

        mlflow.log_metrics(
            metrics
        )    

        print(
            f"Metrics logged: "
            f"F1 = {metrics['f1_score']:.4f} | "
            f"ROC-AUC = {metrics['roc_auc']:.4f}"
        )

        # -----------------------------------------------------
        # 8. Generate Diagnostic Plots
        # -----------------------------------------------------

        os.makedirs(
            "artifacts",
            exist_ok=True
        )

        # -----------------------------------------------------
        # Confusion Matrix
        # -----------------------------------------------------

        fig_cm, ax_cm = plt.subplots(
            figsize=(6, 5)
        )

        ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            ax=ax_cm
        )

        ax_cm.set_title(
            f"Confusion Matrix - {run_name}"
        )

        cm_path = (
            "artifacts/confusion_matrix.png"
        )

        fig_cm.savefig(
            cm_path,
            bbox_inches="tight"
        )

        plt.close(fig_cm)

        mlflow.log_artifact(
            cm_path,
            artifact_path="plots"
        )

        # -----------------------------------------------------
        # ROC Curve
        # -----------------------------------------------------

        fig_roc, ax_roc = plt.subplots(
            figsize=(6, 5)
        )

        RocCurveDisplay.from_predictions(
            y_test,
            y_prob,
            ax=ax_roc
        )

        ax_roc.set_title(
            f"ROC Curve - {run_name}"
        )

        roc_path = (
            "artifacts/roc_curve.png"
        )

        fig_roc.savefig(
            roc_path,
            bbox_inches="tight"
        )

        plt.close(fig_roc)

        mlflow.log_artifact(
            roc_path,
            artifact_path="plots"
        )

        # -----------------------------------------------------
        # 9. Log Preprocessing Metadata
        # -----------------------------------------------------

        metadata_path = (
            "data/processed/dataset_metadata.json"
        )

        if os.path.exists(
            metadata_path
        ):

            mlflow.log_artifact(
                metadata_path,
                artifact_path="metadata"
            )

        # -----------------------------------------------------
        # 10. Log Model to MLflow
        # -----------------------------------------------------

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model"
        )

        # -----------------------------------------------------
        # 11. Save Local Model Backup
        # -----------------------------------------------------

        os.makedirs(
            "models",
            exist_ok=True
        )

        joblib.dump(
            model,
            "models/balanced_svm_model.pkl"
        )

        print(
            "Local model saved to:",
            "models/balanced_svm_model.pkl"
        )

        print(
            f"Run '{run_name}' successfully tracked!"
        )


if __name__ == "__main__":

    train_and_track(
        run_name="Balanced_SVM_Baseline"
    )