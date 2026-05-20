"""
tests/conftest.py
==================
Shared pytest fixtures available across all test modules.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def sample_texts():
    return [
        "The Federal Reserve raised interest rates by 25 basis points as expected.",
        "SHOCKING: Secret government program found to be controlling the weather!",
        "Scientists discover new species of deep-sea fish in the Pacific Ocean.",
        "Miracle cure for all diseases hidden by pharmaceutical companies, insiders claim.",
        "European Union reaches agreement on new trade policy with Asia-Pacific nations.",
    ]


@pytest.fixture(scope="session")
def sample_labels():
    return [0, 1, 0, 1, 0]  # REAL=0, FAKE=1


@pytest.fixture(scope="session")
def synthetic_embeddings():
    np.random.seed(42)
    return np.random.randn(100, 384).astype(np.float32)


@pytest.fixture(scope="session")
def synthetic_labels():
    np.random.seed(42)
    return np.random.choice([0, 1], 100)


@pytest.fixture
def tmp_model_dir(tmp_path):
    """Temporary directory for model artifacts in tests."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Temporary directory for data artifacts."""
    data_dir = tmp_path / "data" / "processed"
    data_dir.mkdir(parents=True)
    return data_dir
