import os
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from src.utils.logger import get_logger

logger = get_logger("model_training")

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

def train_and_evaluate(
    train_path: str = "data/processed/train.csv",
    test_path: str = "data/processed/test.csv",
    model_output_path: str = "data/processed/model.joblib",
    model_type: str = "random_forest",
    random_state: int = 42
):
    """
    Trains a classification model on processed training data, logs parameters/metrics/artifacts
    to MLflow, and saves the final model artifact to disk.
    """
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        err = "Processed train/test files missing! Run data transformation first."
        logger.error(err)
        raise FileNotFoundError(err)

    logger.info(f"Loading processed training and testing datasets from {train_path} and {test_path}...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=['target'])
    y_train = train_df['target']
    X_test = test_df.drop(columns=['target'])
    y_test = test_df['target']

    # Set sqlite tracking URI
    db_path = os.path.abspath("mlflow.db")
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    experiment_name = "Loan_Approval_Prediction"
    mlflow.set_experiment(experiment_name)

    logger.info(f"Starting MLflow experiment run for model_type='{model_type}'...")

    with mlflow.start_run(run_name=f"Run_{model_type}"):
        if model_type == "random_forest":
            params = {
                "n_estimators": 100,
                "max_depth": 6,
                "min_samples_split": 4,
                "random_state": random_state
            }
            model = RandomForestClassifier(**params)
        elif model_type == "logistic_regression":
            params = {
                "C": 1.0,
                "max_iter": 500,
                "random_state": random_state
            }
            model = LogisticRegression(**params)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        logger.info(f"Training {model.__class__.__name__} with hyperparams: {params}...")
        model.fit(X_train, y_train)

        # Predict on Test Set
        y_pred = model.predict(X_test)

        # Compute Metrics
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        cm = confusion_matrix(y_test, y_pred)

        logger.info(f"--- Model Evaluation Results ---")
        logger.info(f"Accuracy : {acc:.4f}")
        logger.info(f"Precision: {prec:.4f}")
        logger.info(f"Recall   : {rec:.4f}")
        logger.info(f"F1-Score : {f1:.4f}")
        logger.info(f"Confusion Matrix:\n{cm}")

        # MLflow Logging
        mlflow.log_param("model_type", model_type)
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # Log Model Artifact in MLflow with trusted skops types or cloudpickle
        mlflow.sklearn.log_model(
            model,
            name="model",
            serialization_format="cloudpickle"
        )

        # Save model locally for production FastAPI service
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        joblib.dump(model, model_output_path)
        logger.info(f"Saved trained model artifact to {model_output_path}")

    return model, {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm.tolist()
    }

if __name__ == "__main__":
    train_and_evaluate(model_type="random_forest")
