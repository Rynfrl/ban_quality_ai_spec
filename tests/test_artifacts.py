"""
Integration tests for model artifact persistence, schema integrity, and reloading.
"""

from pathlib import Path
import json
import pytest

from ml.config import ARTIFACTS_DIR, FEATURES, MODEL_VERSION
from ml.artifacts import load_artifacts, ArtifactValidationError


def test_artifacts_exist():
    """Verify all 5 required artifact files exist on disk."""
    required = [
        "model.pt",
        "model_config.json",
        "scaler.json",
        "hasil_evaluasi.json",
        "training_history.json",
    ]
    for fname in required:
        fpath = ARTIFACTS_DIR / fname
        assert fpath.is_file(), f"Expected artifact file {fname} not found in {ARTIFACTS_DIR}"


def test_artifacts_reload_and_smoke_test():
    """Verify artifacts reload in fresh object with successful smoke test."""
    bundle = load_artifacts(ARTIFACTS_DIR)
    assert bundle["model"] is not None
    assert bundle["scaler"] is not None
    assert bundle["threshold"] > 0.0
    assert bundle["model_version"] == MODEL_VERSION


def test_hasil_evaluasi_schema():
    """Verify schema and non-null metrics in hasil_evaluasi.json."""
    eval_path = ARTIFACTS_DIR / "hasil_evaluasi.json"
    with open(eval_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["model_version"] == MODEL_VERSION
    assert data["features"] == list(FEATURES)
    assert "metrics" in data

    metrics = data["metrics"]
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["precision_class_1"] <= 1.0
    assert 0.0 <= metrics["recall_class_1"] <= 1.0
    assert 0.0 <= metrics["f1_class_1"] <= 1.0
    if metrics["roc_auc"] is not None:
        assert 0.0 <= metrics["roc_auc"] <= 1.0

    cm = data["confusion_matrix"]
    assert len(cm) == 2 and len(cm[0]) == 2 and len(cm[1]) == 2


def test_missing_artifact_raises_error(tmp_path):
    """Verify load_artifacts raises ArtifactValidationError if an artifact is missing."""
    with pytest.raises(ArtifactValidationError):
        load_artifacts(tmp_path)
