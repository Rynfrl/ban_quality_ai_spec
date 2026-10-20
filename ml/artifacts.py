"""
Artifact management and serialization module for Ban Quality ML.
Handles saving, loading, validation, and smoke-testing of model artifacts.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import torch
import numpy as np
import sklearn

from ml.config import ARTIFACTS_DIR, FEATURES, TARGET, LABEL_MAPPING, MODEL_VERSION
from ml.model import BanQualityMLP, build_model
from ml.preprocessing import FittedScaler, serialize_scaler, deserialize_scaler


class ArtifactValidationError(Exception):
    """Raised when artifact validation or loading fails."""
    pass


def save_all_artifacts(
    model: BanQualityMLP,
    scaler_dict: Dict[str, Any],
    eval_dict: Dict[str, Any],
    history_list: List[Dict[str, Any]],
    threshold: float,
    artifacts_dir: Path | str = ARTIFACTS_DIR,
    model_version: str = MODEL_VERSION,
) -> Dict[str, Path]:
    """
    Persist all required artifact files into the target directory.
    """
    out_dir = Path(artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. model.pt
    model_pt_path = out_dir / "model.pt"
    torch.save(model.state_dict(), model_pt_path)

    # 2. model_config.json
    model_config = {
        "model_version": model_version,
        "architecture": {
            "input_dim": model.input_dim,
            "hidden_layers": model.hidden_layers,
            "dropout": model.dropout,
            "activation": "ReLU",
            "output_dim": 1,
        },
        "features": list(FEATURES),
        "target": TARGET,
        "label_mapping": LABEL_MAPPING,
        "threshold": float(threshold),
        "pytorch_version": torch.__version__,
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
    }
    model_config_path = out_dir / "model_config.json"
    with open(model_config_path, "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2)

    # 3. scaler.json
    scaler_path = out_dir / "scaler.json"
    with open(scaler_path, "w", encoding="utf-8") as f:
        json.dump(scaler_dict, f, indent=2)

    # 4. hasil_evaluasi.json
    eval_path = out_dir / "hasil_evaluasi.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_dict, f, indent=2)

    # 5. training_history.json
    history_path = out_dir / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history_list, f, indent=2)

    return {
        "model_pt": model_pt_path,
        "model_config": model_config_path,
        "scaler": scaler_path,
        "hasil_evaluasi": eval_path,
        "training_history": history_path,
    }


def load_artifacts(artifacts_dir: Path | str = ARTIFACTS_DIR) -> Dict[str, Any]:
    """
    Validate and load all persisted artifacts from disk.
    Performs smoke testing before returning.
    
    Raises:
        ArtifactValidationError: If files are missing, corrupted, or smoke test fails.
    """
    dir_path = Path(artifacts_dir)
    if not dir_path.is_dir():
        raise ArtifactValidationError(f"Artifacts directory not found: {dir_path}")

    required_files = [
        "model.pt",
        "model_config.json",
        "scaler.json",
        "hasil_evaluasi.json",
        "training_history.json",
    ]

    for fname in required_files:
        fpath = dir_path / fname
        if not fpath.is_file():
            raise ArtifactValidationError(f"Missing required artifact file: {fpath}")

    try:
        # Load config
        with open(dir_path / "model_config.json", "r", encoding="utf-8") as f:
            model_config = json.load(f)

        # Validate feature list matches
        features_in_config = model_config.get("features", [])
        if list(features_in_config) != list(FEATURES):
            raise ArtifactValidationError(
                f"Feature mismatch in model_config: expected {list(FEATURES)}, got {features_in_config}"
            )

        # Load scaler
        with open(dir_path / "scaler.json", "r", encoding="utf-8") as f:
            scaler_data = json.load(f)
        scaler = deserialize_scaler(scaler_data)

        # Construct and load model
        model = build_model(model_config)
        state_dict = torch.load(dir_path / "model.pt", map_location=torch.device("cpu"), weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()

        # Load evaluation and history
        with open(dir_path / "hasil_evaluasi.json", "r", encoding="utf-8") as f:
            evaluation = json.load(f)

        with open(dir_path / "training_history.json", "r", encoding="utf-8") as f:
            training_history = json.load(f)

        threshold = float(model_config.get("threshold", 0.50))

        # Smoke-test inference
        sample_input = np.array([[160.0, 20.0, 11.5, 0.60, 65.0]])
        scaled_sample = scaler.transform(sample_input)
        sample_tensor = torch.tensor(scaled_sample, dtype=torch.float32)
        with torch.no_grad():
            logit = model(sample_tensor)
            prob = torch.sigmoid(logit).item()

        if np.isnan(prob) or np.isinf(prob) or not (0.0 <= prob <= 1.0):
            raise ArtifactValidationError(f"Smoke test inference returned invalid probability: {prob}")

        return {
            "model": model,
            "scaler": scaler,
            "scaler_data": scaler_data,
            "model_config": model_config,
            "evaluation": evaluation,
            "training_history": training_history,
            "threshold": threshold,
            "model_version": model_config.get("model_version", MODEL_VERSION),
        }

    except Exception as e:
        if isinstance(e, ArtifactValidationError):
            raise e
        raise ArtifactValidationError(f"Failed to load or validate artifacts: {str(e)}") from e
