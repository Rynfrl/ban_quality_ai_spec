"""
Unit tests for PyTorch BanQualityMLP model architecture and threshold logic.
"""

import pytest
import torch
import numpy as np

from ml.model import BanQualityMLP, build_model


def test_model_forward_shape():
    """Verify input shape (batch, 5) produces output shape (batch, 1)."""
    model = BanQualityMLP(input_dim=5, hidden_layers=[32, 16], dropout=0.10)
    model.eval()

    batch_x = torch.randn(8, 5, dtype=torch.float32)
    with torch.no_grad():
        out = model(batch_x)

    assert out.shape == (8, 1)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_model_predict_proba():
    """Verify predict_proba returns probabilities in [0, 1]."""
    model = BanQualityMLP(input_dim=5, hidden_layers=[32, 16], dropout=0.10)
    batch_x = torch.randn(5, 5, dtype=torch.float32)
    probs = model.predict_proba(batch_x)

    assert probs.shape == (5, 1)
    assert (probs >= 0.0).all()
    assert (probs <= 1.0).all()


def test_model_factory_from_config():
    """Verify model can be dynamically constructed from config dictionary."""
    config = {
        "architecture": {
            "input_dim": 5,
            "hidden_layers": [32, 16],
            "dropout": 0.15,
            "activation": "ReLU",
            "output_dim": 1,
        }
    }
    model = build_model(config)
    assert model.input_dim == 5
    assert model.hidden_layers == [32, 16]
    assert model.dropout == 0.15


def test_threshold_decision_logic():
    """Verify threshold boundary behavior."""
    threshold = 0.50
    test_probs = np.array([0.49, 0.50, 0.51, 0.0, 1.0])
    preds = (test_probs >= threshold).astype(int)

    assert preds[0] == 0  # < 0.50 => 0
    assert preds[1] == 1  # == 0.50 => 1
    assert preds[2] == 1  # > 0.50 => 1
    assert preds[3] == 0  # 0.0 => 0
    assert preds[4] == 1  # 1.0 => 1


def test_model_compute_activations():
    """Verify compute_activations produces correct shapes and ReLU non-negativity."""
    model = BanQualityMLP(input_dim=5, hidden_layers=[32, 16], dropout=0.10)
    x = torch.randn(5)
    act = model.compute_activations(x)

    assert "layers" in act
    assert len(act["layers"]) == 3
    # Hidden 1
    h1 = act["layers"][0]
    assert h1["units"] == 32
    assert all(val >= 0.0 for val in h1["post_activations"])  # ReLU ensures >= 0
    # Hidden 2
    h2 = act["layers"][1]
    assert h2["units"] == 16
    assert all(val >= 0.0 for val in h2["post_activations"])  # ReLU ensures >= 0
    # Output
    out = act["layers"][2]
    assert out["units"] == 1
    assert 0.0 <= act["probability_bad"] <= 1.0


def test_model_get_weights_summary():
    """Verify get_weights_summary extracts all linear layers and biases."""
    model = BanQualityMLP(input_dim=5, hidden_layers=[32, 16], dropout=0.10)
    weights = model.get_weights_summary()

    assert "layers" in weights
    assert len(weights["layers"]) == 3
    assert weights["layers"][0]["in_features"] == 5
    assert weights["layers"][0]["out_features"] == 32
    assert weights["layers"][1]["in_features"] == 32
    assert weights["layers"][1]["out_features"] == 16
    assert weights["layers"][2]["in_features"] == 16
    assert weights["layers"][2]["out_features"] == 1

