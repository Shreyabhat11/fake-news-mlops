"""
training_pipeline/train.py
===========================
Trains a fake news classifier on top of sentence embeddings.

Supports:
    - Logistic Regression (fast, interpretable baseline)
    - XGBoost (improved model)

MLflow tracking:
    - parameters, metrics, confusion matrix, model artifact
    - registers best model in the MLflow Model Registry

Usage:
    python training_pipeline/train.py --classifier logistic_regression
    python training_pipeline/train.py --classifier xgboost
"""

import argparse
import json
import logging
import os
import pickle
from pathlib import Path
from contextlib import nullcontext

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── MLflow Config ─────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "fake_news_detection")
REGISTERED_MODEL_NAME = "fake_news_classifier"

MLFLOW_ENABLED = os.getenv("MLFLOW_ENABLED", "false").lower() == "true"

def load_split(split: str) -> tuple[np.ndarray, np.ndarray]:
    """Load embeddings and labels for a given split."""
    emb_path = PROCESSED_DIR / f"{split}_embeddings.npy"
    lbl_path = PROCESSED_DIR / f"{split}_labels.npy"

    if not emb_path.exists():
        raise FileNotFoundError(
            f"Embeddings not found at {emb_path}. "
            "Run the feature pipeline first: python feature_pipeline/embeddings.py"
        )

    X = np.load(emb_path)
    y = np.load(lbl_path)
    logger.info(f"Loaded {split}: X={X.shape}, y={y.shape}, "
                f"positive_rate={y.mean():.2%}")
    return X, y


def build_model(classifier: str, **kwargs):
    """
    Factory function — returns a scikit-learn compatible classifier.
    """
    if classifier == "logistic_regression":
        return LogisticRegression(
            C=kwargs.get("C", 1.0),
            max_iter=kwargs.get("max_iter", 1000),
            solver="lbfgs",
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )
    elif classifier == "xgboost":
        if not XGBOOST_AVAILABLE:
            raise ImportError("xgboost not installed. pip install xgboost")
        return xgb.XGBClassifier(
            n_estimators=kwargs.get("n_estimators", 200),
            max_depth=kwargs.get("max_depth", 6),
            learning_rate=kwargs.get("learning_rate", 0.1),
            subsample=kwargs.get("subsample", 0.8),
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown classifier: {classifier}. "
                         "Options: logistic_regression, xgboost")


def evaluate(model, X: np.ndarray, y: np.ndarray, split: str) -> dict:
    """
    Compute classification metrics and return as a dict.
    """
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        f"{split}_accuracy": round(accuracy_score(y, y_pred), 4),
        f"{split}_f1": round(f1_score(y, y_pred, average="weighted"), 4),
        f"{split}_precision": round(precision_score(y, y_pred, average="weighted"), 4),
        f"{split}_recall": round(recall_score(y, y_pred, average="weighted"), 4),
    }
    if y_proba is not None:
        metrics[f"{split}_roc_auc"] = round(roc_auc_score(y, y_proba), 4)

    report = classification_report(y, y_pred, target_names=["REAL", "FAKE"])
    cm = confusion_matrix(y, y_pred)

    logger.info(f"\n{split.upper()} metrics:\n{report}")
    logger.info(f"Confusion matrix:\n{cm}")

    return metrics, cm


def save_confusion_matrix_artifact(cm: np.ndarray, split: str) -> str:
    """Save confusion matrix as JSON for MLflow artifact logging."""
    path = MODEL_DIR / f"confusion_matrix_{split}.json"
    data = {
        "split": split,
        "matrix": cm.tolist(),
        "labels": ["REAL", "FAKE"],
        "true_negatives": int(cm[0, 0]),
        "false_positives": int(cm[0, 1]),
        "false_negatives": int(cm[1, 0]),
        "true_positives": int(cm[1, 1]),
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return str(path)


def train_and_track(
    classifier: str = "logistic_regression",
    register: bool = True,
    **kwargs,
) -> str:
    """
    Main training function.

    Steps:
        1. Load train/val/test embeddings
        2. Fit model
        3. Evaluate on val + test
        4. Log everything to MLflow
        5. Register model if register=True
        6. Save model.pkl locally

    Returns:
        MLflow run_id
    """
        # ── MLflow setup ─────────────────────────────────────────────────────────
    # ── MLflow setup ─────────────────────────────────────────────────────────────
    if MLFLOW_ENABLED:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        logger.info(f"MLflow tracking enabled: {MLFLOW_TRACKING_URI}")
    else:
        logger.info("MLflow tracking disabled. Training locally.")

    # ── Load data ─────────────────────────────────────────────────────────────
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")

    # ── Run ───────────────────────────────────────────────────────────────────
    if MLFLOW_ENABLED:
        run_context = mlflow.start_run(run_name=f"{classifier}_run")

    else:
        run_context = nullcontext()

    with run_context as run:
        run_id = run.info.run_id if MLFLOW_ENABLED else "local_run"
        logger.info(f"Starting run {run_id} with classifier={classifier}")

        # Log parameters
        params = {"classifier": classifier, **kwargs}
        if MLFLOW_ENABLED:
            mlflow.log_params(params)

        # Build & train model
        model = build_model(classifier, **kwargs)
        logger.info(f"Training {classifier} on {len(X_train)} samples...")
        model.fit(X_train, y_train)
        logger.info("Training complete.")

        # Evaluate
        val_metrics, val_cm = evaluate(model, X_val, y_val, split="val")
        test_metrics, test_cm = evaluate(model, X_test, y_test, split="test")

        all_metrics = {**val_metrics, **test_metrics}
        if MLFLOW_ENABLED:
            mlflow.log_metrics(all_metrics)

        # Artifact: confusion matrices
        val_cm_path = save_confusion_matrix_artifact(val_cm, "val")
        test_cm_path = save_confusion_matrix_artifact(test_cm, "test")
        if MLFLOW_ENABLED:
            mlflow.log_artifact(val_cm_path)
            mlflow.log_artifact(test_cm_path)

        # Artifact: model pkl
        model_path = MODEL_DIR / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        if MLFLOW_ENABLED:
            mlflow.log_artifact(str(model_path))

        # Log model to MLflow model registry
        if MLFLOW_ENABLED:
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=REGISTERED_MODEL_NAME if register else None,
            )

            # Tag the run with metadata
            mlflow.set_tags({
                "classifier": classifier,
                "val_f1": val_metrics["val_f1"],
                "test_f1": test_metrics["test_f1"],
                "model_version": "v1.0",
            })

        logger.info(f"Run {run_id} complete. Val F1={val_metrics['val_f1']}, "
                    f"Test F1={test_metrics['test_f1']}")

    # ── Save metadata for API ─────────────────────────────────────────────────
    metadata = {
        "run_id": run_id,
        "classifier": classifier,
        "registered_model_name": REGISTERED_MODEL_NAME,
        "val_f1": val_metrics["val_f1"],
        "test_f1": test_metrics["test_f1"],
        "test_roc_auc": test_metrics.get("test_roc_auc"),
        "model_path": str(model_path),
    }
    with open(MODEL_DIR / "latest_model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return run_id


def load_latest_model():
    """Load the model.pkl saved by the last training run."""
    model_path = MODEL_DIR / "model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            "No model found. Run: python training_pipeline/train.py"
        )
    with open(model_path, "rb") as f:
        return pickle.load(f)


def rollback_model(run_id: str) -> None:
    """
    Rollback to a specific MLflow run by downloading and replacing model.pkl.
    Useful for production incidents.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()
    run = client.get_run(run_id)
    artifact_uri = run.info.artifact_uri
    logger.info(f"Rolling back to run {run_id}, artifact: {artifact_uri}")

    # Download model artifact
    local_path = mlflow.artifacts.download_artifacts(
        artifact_uri=f"{artifact_uri}/model.pkl"
    )
    import shutil
    shutil.copy(local_path, MODEL_DIR / "model.pkl")
    logger.info("Model rolled back successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fake news classifier")
    parser.add_argument(
        "--classifier",
        type=str,
        default="logistic_regression",
        choices=["logistic_regression", "xgboost"],
        help="Classifier to train",
    )
    parser.add_argument("--register", action="store_true", default=True,
                        help="Register model in MLflow Model Registry")
    parser.add_argument("--C", type=float, default=1.0,
                        help="Regularization for logistic regression")
    parser.add_argument("--n_estimators", type=int, default=200,
                        help="Number of trees for XGBoost")
    args = parser.parse_args()

    run_id = train_and_track(
        classifier=args.classifier,
        register=args.register,
        C=args.C,
        n_estimators=args.n_estimators,
    )
    print(f"\n✅ Training complete! MLflow Run ID: {run_id}")

    if MLFLOW_ENABLED:
        print(f"View at: {MLFLOW_TRACKING_URI}/#/experiments")
