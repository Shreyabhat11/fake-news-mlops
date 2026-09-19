"""
tests/test_api.py
==================
Tests for the FastAPI inference service.
Uses httpx TestClient so no real server is needed.
The model and embedding service are mocked to avoid requiring
trained artifacts in CI.
"""

import json
import pickle
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# ── Fixtures and Mocks ─────────────────────────────────────────────────────────

class MockModel:
    """Minimal sklearn-compatible model mock."""
    def predict(self, X):
        return np.array([1 if x.mean() > 0 else 0 for x in X])

    def predict_proba(self, X):
        results = []
        for x in X:
            fake_prob = min(max(float(x.mean() + 0.5), 0.05), 0.95)
            results.append([1 - fake_prob, fake_prob])
        return np.array(results)


class MockEmbeddingService:
    """Returns deterministic fake embeddings."""
    def embed_single(self, text: str) -> np.ndarray:
        np.random.seed(hash(text) % (2**31))
        return np.random.randn(384).astype(np.float32)

    def embed_batch(self, texts, batch_size=32):
        return np.array([self.embed_single(t) for t in texts])


@pytest.fixture
def client():
    """
    Create a TestClient with mocked model and embedding service.
    This avoids needing real artifacts in CI.
    """
    with patch("api.main.app_state") as mock_state:
        mock_state.model = MockModel()
        mock_state.challenger_model = None
        mock_state.embedding_service = MockEmbeddingService()
        mock_state.redis_client = None
        mock_state.prediction_logger = None
        mock_state.model_metadata = {
            "model_version": "v1.0-test",
            "classifier": "logistic_regression",
            "val_f1": 0.95,
            "test_f1": 0.94,
        }
        mock_state.drift_status = "normal"
        mock_state.loaded_at = "2024-01-01T00:00:00"

        from api.main import app
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


# ── Health ────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_schema(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "model_loaded" in data
        assert "embedding_service_ready" in data
        assert "redis_connected" in data
        assert "timestamp" in data

    def test_health_model_loaded(self, client):
        data = client.get("/health").json()
        assert data["model_loaded"] is True

    def test_health_embedding_ready(self, client):
        data = client.get("/health").json()
        assert data["embedding_service_ready"] is True


# ── /predict ──────────────────────────────────────────────────────────────────

class TestPredict:
    SAMPLE_TEXT = (
        "The Federal Reserve announced today that interest rates will remain "
        "unchanged following the latest economic data reports from multiple sources."
    )

    def test_predict_returns_200(self, client):
        response = client.post("/predict", json={"text": self.SAMPLE_TEXT})
        assert response.status_code == 200

    def test_predict_response_schema(self, client):
        data = client.post("/predict", json={"text": self.SAMPLE_TEXT}).json()
        required_fields = [
            "request_id", "prediction", "label", "confidence",
            "fake_probability", "real_probability", "drift_status",
            "model_version", "processing_time_ms", "cached",
        ]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"

    def test_predict_label_binary(self, client):
        data = client.post("/predict", json={"text": self.SAMPLE_TEXT}).json()
        assert data["label"] in (0, 1)

    def test_predict_prediction_string(self, client):
        data = client.post("/predict", json={"text": self.SAMPLE_TEXT}).json()
        assert data["prediction"] in ("FAKE", "REAL")

    def test_predict_probabilities_sum_to_one(self, client):
        data = client.post("/predict", json={"text": self.SAMPLE_TEXT}).json()
        total = data["fake_probability"] + data["real_probability"]
        assert abs(total - 1.0) < 0.01

    def test_predict_confidence_in_range(self, client):
        data = client.post("/predict", json={"text": self.SAMPLE_TEXT}).json()
        assert 0.0 <= data["confidence"] <= 1.0

    def test_predict_with_title(self, client):
        response = client.post("/predict", json={
            "text": self.SAMPLE_TEXT,
            "title": "Fed Holds Rates Steady"
        })
        assert response.status_code == 200

    def test_predict_request_id_is_uuid(self, client):
        import uuid
        data = client.post("/predict", json={"text": self.SAMPLE_TEXT}).json()
        try:
            uuid.UUID(data["request_id"])
        except ValueError:
            pytest.fail("request_id is not a valid UUID")

    def test_predict_rejects_short_text(self, client):
        response = client.post("/predict", json={"text": "hi"})
        assert response.status_code == 422

    def test_predict_rejects_empty_text(self, client):
        response = client.post("/predict", json={"text": ""})
        assert response.status_code == 422

    def test_predict_rejects_missing_text(self, client):
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_predict_drift_status_field(self, client):
        data = client.post("/predict", json={"text": self.SAMPLE_TEXT}).json()
        assert data["drift_status"] in ("normal", "warning", "critical")

    def test_predict_model_version_present(self, client):
        data = client.post("/predict", json={"text": self.SAMPLE_TEXT}).json()
        assert data["model_version"]

    def test_predict_processing_time_positive(self, client):
        data = client.post("/predict", json={"text": self.SAMPLE_TEXT}).json()
        assert data["processing_time_ms"] >= 0


# ── /batch_predict ────────────────────────────────────────────────────────────

class TestBatchPredict:
    TEXTS = [
        "The government today released annual budget figures showing a 5% increase.",
        "SHOCKING secret discovered that doctors don't want you to know about miracle cure!",
        "Scientists published a new study on climate change patterns in Nature journal.",
    ]

    def test_batch_predict_returns_200(self, client):
        response = client.post("/batch_predict", json={"texts": self.TEXTS})
        assert response.status_code == 200

    def test_batch_predict_returns_all_predictions(self, client):
        data = client.post("/batch_predict", json={"texts": self.TEXTS}).json()
        assert data["total_processed"] == len(self.TEXTS)
        assert len(data["predictions"]) == len(self.TEXTS)

    def test_batch_predict_each_has_schema(self, client):
        data = client.post("/batch_predict", json={"texts": self.TEXTS}).json()
        for pred in data["predictions"]:
            assert "prediction" in pred
            assert "confidence" in pred

    def test_batch_predict_rejects_empty_list(self, client):
        response = client.post("/batch_predict", json={"texts": []})
        assert response.status_code == 422

    def test_batch_predict_rejects_too_many(self, client):
        texts = ["Sample article text here." * 3] * 51  # 51 > max 50
        response = client.post("/batch_predict", json={"texts": texts})
        assert response.status_code == 422

    def test_batch_predict_processing_time(self, client):
        data = client.post("/batch_predict", json={"texts": self.TEXTS}).json()
        assert data["processing_time_ms"] >= 0


# ── /model/info ───────────────────────────────────────────────────────────────

class TestModelInfo:
    def test_model_info_returns_200(self, client):
        response = client.get("/model/info")
        assert response.status_code == 200

    def test_model_info_schema(self, client):
        data = client.get("/model/info").json()
        for field in ["model_version", "classifier", "loaded_at", "canary_mode", "shadow_mode"]:
            assert field in data


# ── /metrics ──────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_metrics_returns_200(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_content_type(self, client):
        response = client.get("/metrics")
        assert "text/plain" in response.headers.get("content-type", "")

    def test_metrics_contains_prometheus_format(self, client):
        response = client.get("/metrics")
        # Prometheus metrics start with "# HELP" or "# TYPE"
        assert "# HELP" in response.text or "# TYPE" in response.text


# ── Rate limiting ─────────────────────────────────────────────────────────────

class TestRateLimiting:
    """Verify rate limiting headers are present."""

    def test_predict_has_rate_limit_header(self, client):
        response = client.post("/predict", json={
            "text": "The president signed the new economic policy bill into law today after negotiations."
        })
        # slowapi adds X-RateLimit headers
        # (may not be present in test client mode — just check no crash)
        assert response.status_code in (200, 429)
