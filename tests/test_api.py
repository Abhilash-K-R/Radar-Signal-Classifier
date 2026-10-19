"""
Integration and Unit Tests for FastAPI Radar Signal Classifier API
"""

import sys
import os
import io
import pytest
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "api"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from fastapi.testclient import TestClient
from api.main import app
from src.config import N_SAMPLES


@pytest.fixture(scope="module")
def client():
    """Create a FastAPI TestClient instance."""
    with TestClient(app) as test_client:
        yield test_client


class TestAPIEndpoints:
    """Test standard API endpoints."""

    def test_root_endpoint(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert "endpoints" in data

    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "models_available" in data
        assert data["models_available"]["rf"] is True
        assert data["sample_rate"] == 500
        assert data["signal_length"] == 500

    def test_models_list(self, client):
        response = client.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 2
        names = [m["name"] for m in data]
        assert "Random Forest" in names
        assert any("CNN" in name for name in names)


class TestAPIPredictions:
    """Test prediction endpoints."""

    def test_predict_generated_sine(self, client):
        payload = {
            "signal_type": "sine",
            "frequency": 5.0,
            "amplitude": 1.0,
            "model": "rf",
        }
        response = client.post("/predict/generate", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["label"] == "sine"
        assert "confidence" in data
        assert "probabilities" in data
        assert "sine" in data["probabilities"]

    def test_predict_generated_chirp(self, client):
        payload = {
            "signal_type": "chirp",
            "frequency": 2.0,
            "f1": 50.0,
            "amplitude": 1.0,
            "model": "rf",
        }
        response = client.post("/predict/generate", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["label"] == "chirp"

    def test_predict_array_valid(self, client):
        # 500-sample sine wave
        t = np.linspace(0, 0.5, N_SAMPLES, endpoint=False)
        sig = (np.sin(2 * np.pi * 5.0 * t)).tolist()

        payload = {"signal": sig, "model": "rf"}
        response = client.post("/predict/array", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["label"] == "sine"
        assert data["confidence"] > 50.0

    def test_predict_array_invalid_length(self, client):
        # Incorrect sample length (100 samples instead of 500)
        payload = {"signal": [0.0] * 100, "model": "rf"}
        response = client.post("/predict/array", json=payload)
        assert response.status_code == 422

    def test_predict_upload_csv(self, client):
        # Create in-memory CSV
        t = np.linspace(0, 0.5, N_SAMPLES, endpoint=False)
        sig = np.sin(2 * np.pi * 5.0 * t)
        df = pd.DataFrame({"signal": sig})

        csv_buffer = io.BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        response = client.post(
            "/predict/upload?model=rf",
            files={"file": ("test_signal.csv", csv_buffer, "text/csv")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["label"] == "sine"
        assert data["signal_info"]["source"] == "test_signal.csv"

    def test_predict_upload_invalid_extension(self, client):
        buffer = io.BytesIO(b"dummy binary data")
        response = client.post(
            "/predict/upload?model=rf",
            files={"file": ("test.txt", buffer, "text/plain")},
        )
        assert response.status_code == 422
