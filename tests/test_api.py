"""
Integration tests for FastAPI endpoints, input validation, and sanity predictions.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.artifact_loader import artifact_manager
from ml.config import MODEL_VERSION, FEATURES


@pytest.fixture(scope="module")
def client():
    # Ensure artifacts are loaded
    artifact_manager.load()
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    """Verify /health returns 200, status ok, and model_loaded true."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["model_version"] == MODEL_VERSION


def test_model_info_endpoint(client):
    """Verify /api/model-info returns features and metrics."""
    resp = client.get("/api/model-info")
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_version"] == MODEL_VERSION
    assert data["features"] == list(FEATURES)
    assert "metrics" in data
    assert "accuracy" in data["metrics"]


def test_scaler_endpoint(client):
    """Verify /api/scaler returns feature names, mean, and scale."""
    resp = client.get("/api/scaler")
    assert resp.status_code == 200
    data = resp.json()
    assert "feature_names" in data
    assert "mean" in data
    assert "scale" in data
    assert len(data["mean"]) == 5


def test_model_config_endpoint(client):
    """Verify /api/model-config returns architecture and version."""
    resp = client.get("/api/model-config")
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_version"] == MODEL_VERSION
    assert "architecture" in data
    assert data["architecture"]["activation"] == "ReLU"
    assert data["architecture"]["input_dim"] == 5


def test_predict_ideal_center(client):
    """Test input A — ideal center parameters."""
    payload = {
        "suhu": 157.5,
        "tekanan": 20.0,
        "waktu_curing": 11.5,
        "rasio_karet": 0.60,
        "kekerasan_karet": 65.0,
    }
    resp = client.post("/api/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["label"] in [0, 1]
    assert data["label_name"] in ["Baik", "Perlu Perbaikan"]
    assert 0.0 <= data["probability_good"] <= 1.0
    assert 0.0 <= data["probability_bad"] <= 1.0
    assert pytest.approx(data["probability_good"] + data["probability_bad"], abs=0.01) == 1.0
    assert "feature_checks" in data
    assert data["feature_checks"]["suhu"]["in_ideal_range"] is True
    assert data["feature_checks"]["kekerasan_karet"]["in_ideal_range"] is True


def test_predict_lower_temp_boundary(client):
    """Test input B — lower temperature boundary."""
    payload = {
        "suhu": 140.0,
        "tekanan": 20.0,
        "waktu_curing": 11.5,
        "rasio_karet": 0.60,
        "kekerasan_karet": 62.0,
    }
    resp = client.post("/api/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["label"] in [0, 1]
    assert 0.0 <= data["probability_bad"] <= 1.0


def test_predict_upper_temp_boundary(client):
    """Test input C — upper temperature boundary."""
    payload = {
        "suhu": 180.0,
        "tekanan": 20.0,
        "waktu_curing": 11.5,
        "rasio_karet": 0.60,
        "kekerasan_karet": 71.0,
    }
    resp = client.post("/api/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["label"] in [0, 1]
    assert 0.0 <= data["probability_bad"] <= 1.0


def test_predict_batch_endpoint(client):
    """Verify batch prediction endpoint."""
    payload = {
        "items": [
            {"suhu": 155.0, "tekanan": 19.0, "waktu_curing": 11.0, "rasio_karet": 0.58, "kekerasan_karet": 64.0},
            {"suhu": 165.0, "tekanan": 21.0, "waktu_curing": 12.0, "rasio_karet": 0.62, "kekerasan_karet": 68.0},
        ]
    }
    resp = client.post("/api/predict/batch", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["results"]) == 2


def test_predict_validation_missing_field(client):
    """Verify 422 Unprocessable Entity when required field is missing."""
    payload = {
        "suhu": 160.0,
        "tekanan": 20.0,
        # missing waktu_curing and rasio_karet
    }
    resp = client.post("/api/predict", json=payload)
    assert resp.status_code == 422


def test_predict_validation_domain_exceeded(client):
    """Verify 422 Unprocessable Entity when value is beyond physical domain limit."""
    payload = {
        "suhu": 999.0,  # Exceeds maximum physical domain limit
        "tekanan": 20.0,
        "waktu_curing": 11.5,
        "rasio_karet": 0.60,
        "kekerasan_karet": 65.0,
    }
    resp = client.post("/api/predict", json=payload)
    assert resp.status_code == 422


def test_predict_validation_kekerasan_exceeded(client):
    """Verify 422 Unprocessable Entity when kekerasan_karet is outside domain limit."""
    payload = {
        "suhu": 160.0,
        "tekanan": 20.0,
        "waktu_curing": 11.5,
        "rasio_karet": 0.60,
        "kekerasan_karet": 150.0,  # Exceeds domain [30, 100]
    }
    resp = client.post("/api/predict", json=payload)
    assert resp.status_code == 422


def test_dashboard_page(client):
    """Verify GET / returns 200 and HTML content."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Ban Quality ML" in resp.text


def test_model_weights_endpoint(client):
    """Verify GET /api/model/weights returns layer weights and biases."""
    resp = client.get("/api/model/weights")
    assert resp.status_code == 200
    data = resp.json()
    assert "layers" in data
    assert len(data["layers"]) == 3  # 3 Linear layers
    l0 = data["layers"][0]
    assert l0["in_features"] == 5
    assert l0["out_features"] == 32
    assert len(l0["weights"]) == 32
    assert len(l0["weights"][0]) == 5


def test_neuron_activations_endpoint(client):
    """Verify POST /api/model/activations computes layer-by-layer neuron activations."""
    payload = {
        "suhu": 160.0,
        "tekanan": 20.0,
        "waktu_curing": 11.5,
        "rasio_karet": 0.60,
        "kekerasan_karet": 65.0,
    }
    resp = client.post("/api/model/activations", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "layers" in data
    assert len(data["layers"]) == 3
    # Hidden 1 has 32 units
    h1 = data["layers"][0]
    assert h1["units"] == 32
    assert len(h1["post_activations"]) == 32
    assert 0 <= h1["active_count"] <= 32
    # Hidden 2 has 16 units
    h2 = data["layers"][1]
    assert h2["units"] == 16
    assert len(h2["post_activations"]) == 16
    assert 0 <= h2["active_count"] <= 16
    # Output layer
    out = data["layers"][2]
    assert out["units"] == 1
    assert "probability_bad" in data

