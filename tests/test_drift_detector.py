"""
tests/test_drift_detector.py
==============================
Unit tests for the drift detection module.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from monitoring.drift_detector import (
    compute_confidence_drift,
    compute_embedding_drift,
    compute_prediction_drift,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def reference_embeddings():
    np.random.seed(42)
    return np.random.randn(500, 384).astype(np.float32)


@pytest.fixture
def similar_embeddings(reference_embeddings):
    """Embeddings from same distribution — no drift expected."""
    np.random.seed(99)
    noise = np.random.randn(*reference_embeddings.shape) * 0.01
    idx = np.random.choice(len(reference_embeddings), 100, replace=True)
    return reference_embeddings[idx] + noise[:100]


@pytest.fixture
def drifted_embeddings():
    """Embeddings from a different distribution — drift expected."""
    np.random.seed(7)
    return (np.random.randn(100, 384) * 2.0 + 3.0).astype(np.float32)


@pytest.fixture
def reference_df():
    np.random.seed(42)
    n = 500
    return pd.DataFrame({
        "label": np.random.choice([0, 1], n, p=[0.5, 0.5]),
        "confidence": np.random.uniform(0.85, 0.99, n),
    })


@pytest.fixture
def similar_df():
    np.random.seed(99)
    n = 100
    return pd.DataFrame({
        "prediction": np.random.choice(["FAKE", "REAL"], n, p=[0.50, 0.50]),
        "confidence": np.random.uniform(0.83, 0.97, n),
    })


@pytest.fixture
def drifted_df():
    np.random.seed(7)
    n = 100
    return pd.DataFrame({
        "prediction": np.random.choice(["FAKE", "REAL"], n, p=[0.75, 0.25]),  # shifted
        "confidence": np.random.uniform(0.50, 0.68, n),  # much lower
    })


# ── compute_embedding_drift ───────────────────────────────────────────────────

class TestEmbeddingDrift:
    def test_returns_dict(self, reference_embeddings, similar_embeddings):
        result = compute_embedding_drift(reference_embeddings, similar_embeddings)
        assert isinstance(result, dict)

    def test_required_keys(self, reference_embeddings, similar_embeddings):
        result = compute_embedding_drift(reference_embeddings, similar_embeddings)
        for key in ["drift_score", "is_drifted", "mean_cosine_distance",
                    "drift_fraction_pca", "avg_wasserstein_distance"]:
            assert key in result

    def test_drift_score_in_range(self, reference_embeddings, similar_embeddings):
        result = compute_embedding_drift(reference_embeddings, similar_embeddings)
        assert 0.0 <= result["drift_score"] <= 1.0

    def test_no_drift_on_similar_data(self, reference_embeddings, similar_embeddings):
        result = compute_embedding_drift(reference_embeddings, similar_embeddings)
        assert result["is_drifted"] is False

    def test_drift_detected_on_different_distribution(self, reference_embeddings, drifted_embeddings):
        result = compute_embedding_drift(reference_embeddings, drifted_embeddings)
        assert result["is_drifted"] is True

    def test_drift_score_higher_for_drifted(self, reference_embeddings, similar_embeddings, drifted_embeddings):
        result_similar = compute_embedding_drift(reference_embeddings, similar_embeddings)
        result_drifted = compute_embedding_drift(reference_embeddings, drifted_embeddings)
        assert result_drifted["drift_score"] > result_similar["drift_score"]


# ── compute_prediction_drift ──────────────────────────────────────────────────

class TestPredictionDrift:
    def test_returns_dict(self, reference_df, similar_df):
        result = compute_prediction_drift(
            reference_df["label"],
            similar_df["prediction"].map({"FAKE": 1, "REAL": 0})
        )
        assert isinstance(result, dict)

    def test_required_keys(self, reference_df, similar_df):
        result = compute_prediction_drift(
            reference_df["label"],
            similar_df["prediction"].map({"FAKE": 1, "REAL": 0})
        )
        for key in ["reference_fake_pct", "current_fake_pct", "prediction_shift", "is_drifted"]:
            assert key in result

    def test_no_drift_on_similar_distribution(self, reference_df, similar_df):
        result = compute_prediction_drift(
            reference_df["label"],
            similar_df["prediction"].map({"FAKE": 1, "REAL": 0})
        )
        # Both ~50% fake — shift should be small
        assert result["prediction_shift"] < 0.15

    def test_drift_on_shifted_distribution(self, reference_df, drifted_df):
        result = compute_prediction_drift(
            reference_df["label"],
            drifted_df["prediction"].map({"FAKE": 1, "REAL": 0})
        )
        # Reference ~50%, current ~75% — large shift
        assert result["prediction_shift"] > 0.15

    def test_prediction_percentages_in_range(self, reference_df, similar_df):
        result = compute_prediction_drift(
            reference_df["label"],
            similar_df["prediction"].map({"FAKE": 1, "REAL": 0})
        )
        assert 0.0 <= result["reference_fake_pct"] <= 1.0
        assert 0.0 <= result["current_fake_pct"] <= 1.0


# ── compute_confidence_drift ──────────────────────────────────────────────────

class TestConfidenceDrift:
    def test_returns_dict(self, reference_df, similar_df):
        result = compute_confidence_drift(reference_df["confidence"], similar_df["confidence"])
        assert isinstance(result, dict)

    def test_required_keys(self, reference_df, similar_df):
        result = compute_confidence_drift(reference_df["confidence"], similar_df["confidence"])
        for key in ["reference_mean_confidence", "current_mean_confidence",
                    "confidence_drop", "is_drifted"]:
            assert key in result

    def test_no_drift_on_similar_confidence(self, reference_df, similar_df):
        result = compute_confidence_drift(reference_df["confidence"], similar_df["confidence"])
        # Both ~0.90 range — drop should be small
        assert result["is_drifted"] is False

    def test_drift_on_low_confidence(self, reference_df, drifted_df):
        result = compute_confidence_drift(reference_df["confidence"], drifted_df["confidence"])
        # Reference ~0.92, drifted ~0.59 — large drop
        assert result["confidence_drop"] > 0.10
        assert result["is_drifted"] is True

    def test_confidence_drop_calculation(self, reference_df, drifted_df):
        result = compute_confidence_drift(reference_df["confidence"], drifted_df["confidence"])
        expected_drop = reference_df["confidence"].mean() - drifted_df["confidence"].mean()
        assert abs(result["confidence_drop"] - expected_drop) < 0.001
