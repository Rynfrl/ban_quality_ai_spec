"""
Prediction service combining preprocessor transform, neural network inference,
and business range checks.
"""

from typing import Dict, Any, List
import numpy as np
import torch

from ml.config import FEATURES, LABEL_MAPPING
from app.schemas import PredictRequest, PredictResponse, FeatureCheck
from app.services.artifact_loader import artifact_manager
from app.services.range_checker import check_feature_ranges


class ModelUnavailableError(Exception):
    """Raised when inference is attempted but model artifacts are not loaded."""
    pass


def predict_single(req: PredictRequest) -> PredictResponse:
    """
    Run full end-to-end inference on a single production input vector.
    """
    if not artifact_manager.is_loaded or artifact_manager.model is None or artifact_manager.scaler is None:
        raise ModelUnavailableError(
            f"Model artifacts unavailable. {artifact_manager.load_error or 'Please train the model first.'}"
        )

    # 1. Prepare ordered feature array
    inputs_dict = {
        "suhu": req.suhu,
        "tekanan": req.tekanan,
        "waktu_curing": req.waktu_curing,
        "rasio_karet": req.rasio_karet,
        "kekerasan_karet": req.kekerasan_karet,
    }
    raw_vector = np.array([[inputs_dict[f] for f in FEATURES]], dtype=np.float64)

    # 2. Scale features using fitted training scaler
    scaled_vector = artifact_manager.scaler.transform(raw_vector)
    input_tensor = torch.tensor(scaled_vector, dtype=torch.float32)

    # 3. Model forward pass & layer-by-layer neuron activations
    with torch.no_grad():
        neuron_act = artifact_manager.model.compute_activations(input_tensor)
        prob_bad = neuron_act["probability_bad"]

    prob_good = float(max(0.0, min(1.0, 1.0 - prob_bad)))
    prob_bad = float(max(0.0, min(1.0, prob_bad)))

    # 4. Decision thresholding
    threshold = artifact_manager.threshold
    label = 1 if prob_bad >= threshold else 0
    label_name = LABEL_MAPPING.get(label, "Unknown")

    # 5. Range checks and warning annotations
    feature_checks, warnings = check_feature_ranges(inputs_dict)

    return PredictResponse(
        label=label,
        label_name=label_name,
        probability_good=round(prob_good, 4),
        probability_bad=round(prob_bad, 4),
        threshold=round(threshold, 4),
        model_version=artifact_manager.model_version or "unknown",
        feature_checks=feature_checks,
        warnings=warnings,
        neuron_activations=neuron_act,
    )


def compute_single_activations(req: PredictRequest) -> Dict[str, Any]:
    """
    Dedicated calculation of neuron activations for interactive visualizers.
    """
    if not artifact_manager.is_loaded or artifact_manager.model is None or artifact_manager.scaler is None:
        raise ModelUnavailableError("Model belum dimuat. Tidak dapat menghitung aktivasi neuron.")

    inputs_dict = {
        "suhu": req.suhu,
        "tekanan": req.tekanan,
        "waktu_curing": req.waktu_curing,
        "rasio_karet": req.rasio_karet,
        "kekerasan_karet": req.kekerasan_karet,
    }
    raw_vector = np.array([[inputs_dict[f] for f in FEATURES]], dtype=np.float64)
    scaled_vector = artifact_manager.scaler.transform(raw_vector)
    input_tensor = torch.tensor(scaled_vector, dtype=torch.float32)

    activations = artifact_manager.model.compute_activations(input_tensor)
    prob_bad = activations["probability_bad"]
    threshold = artifact_manager.threshold
    label = 1 if prob_bad >= threshold else 0

    return {
        "input_features": inputs_dict,
        "scaled_values": [round(float(v), 4) for v in scaled_vector[0].tolist()],
        "layers": activations["layers"],
        "output_logit": activations["output_logit"],
        "probability_bad": prob_bad,
        "probability_good": activations["probability_good"],
        "threshold": round(threshold, 4),
        "decision_label": LABEL_MAPPING.get(label, "Unknown"),
        "model_version": artifact_manager.model_version or "unknown",
    }
