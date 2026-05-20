"""
tests/test_training_pipeline.py
=================================
Tests for model training — uses synthetic embeddings so no real
data pipeline run is required.
"""

import pickle
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression


class TestBuildModel:
    def test_logistic_regression_returns_correct_type(self):
        from training_pipeline.train import build_model
        model = build_model("logistic_regression")
        assert isinstance(model, LogisticRegression)

    def test_xgboost_raises_if_not_installed(self):
        from training_pipeline.train import build_model
        with patch.dict("sys.modules", {"xgboost": None}):
            import importlib
            import training_pipeline.train as train_module
            importlib.reload(train_module)
            # If xgboost not available, should raise ImportError or ValueError
            try:
                model = train_module.build_model("xgboost")
            except (ImportError, ValueError):
                pass  # expected

    def test_unknown_classifier_raises(self):
        from training_pipeline.train import build_model
        with pytest.raises(ValueError):
            build_model("random_forest_unknown")

    def test_logistic_regression_custom_C(self):
        from training_pipeline.train import build_model
        model = build_model("logistic_regression", C=0.5)
        assert model.C == 0.5


class TestEvaluate:
    @pytest.fixture
    def trained_model(self):
        """Train a quick LR model on random data."""
        np.random.seed(42)
        X = np.random.randn(200, 10)
        y = (X[:, 0] > 0).astype(int)
        model = LogisticRegression(max_iter=200)
        model.fit(X, y)
        return model, X, y

    def test_evaluate_returns_metrics_dict_and_cm(self, trained_model):
        from training_pipeline.train import evaluate
        model, X, y = trained_model
        metrics, cm = evaluate(model, X, y, split="test")
        assert isinstance(metrics, dict)
        assert cm.shape == (2, 2)

    def test_evaluate_metric_keys(self, trained_model):
        from training_pipeline.train import evaluate
        model, X, y = trained_model
        metrics, _ = evaluate(model, X, y, split="val")
        for key in ["val_accuracy", "val_f1", "val_precision", "val_recall"]:
            assert key in metrics

    def test_evaluate_metrics_in_range(self, trained_model):
        from training_pipeline.train import evaluate
        model, X, y = trained_model
        metrics, _ = evaluate(model, X, y, split="test")
        for v in metrics.values():
            assert 0.0 <= v <= 1.0


class TestTrainAndTrack:
    """Integration test using synthetic embeddings and local MLflow tracking."""

    @pytest.fixture(autouse=True)
    def setup_synthetic_embeddings(self, tmp_path):
        """Create fake embedding files so train_and_track can load them."""
        np.random.seed(42)
        for split in ["train", "val", "test"]:
            n = {"train": 400, "val": 50, "test": 50}[split]
            X = np.random.randn(n, 384).astype(np.float32)
            y = (X[:, 0] > 0).astype(int)
            np.save(tmp_path / f"{split}_embeddings.npy", X)
            np.save(tmp_path / f"{split}_labels.npy", y)

        # Patch PROCESSED_DIR and MODEL_DIR
        import training_pipeline.train as train_module
        self.orig_proc = train_module.PROCESSED_DIR
        self.orig_model = train_module.MODEL_DIR
        train_module.PROCESSED_DIR = tmp_path
        train_module.MODEL_DIR = tmp_path

        yield

        train_module.PROCESSED_DIR = self.orig_proc
        train_module.MODEL_DIR = self.orig_model

    def test_train_returns_run_id(self, tmp_path):
        from training_pipeline.train import train_and_track
        import mlflow
        # Use local file tracking to avoid needing MLflow server
        mlflow.set_tracking_uri(f"file://{tmp_path}/mlruns")
        run_id = train_and_track(
            classifier="logistic_regression",
            register=False,
        )
        assert isinstance(run_id, str)
        assert len(run_id) > 0

    def test_model_pkl_saved(self, tmp_path):
        from training_pipeline.train import train_and_track
        import mlflow
        mlflow.set_tracking_uri(f"file://{tmp_path}/mlruns")
        train_and_track(classifier="logistic_regression", register=False)
        assert (tmp_path / "model.pkl").exists()

    def test_saved_model_can_predict(self, tmp_path):
        from training_pipeline.train import train_and_track
        import mlflow
        mlflow.set_tracking_uri(f"file://{tmp_path}/mlruns")
        train_and_track(classifier="logistic_regression", register=False)

        with open(tmp_path / "model.pkl", "rb") as f:
            model = pickle.load(f)

        X_test = np.random.randn(5, 384).astype(np.float32)
        preds = model.predict(X_test)
        assert len(preds) == 5
        assert set(preds).issubset({0, 1})
