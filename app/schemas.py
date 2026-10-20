"""
Pydantic schemas for Ban Quality ML REST API.
"""

from typing import Dict, List, Any, Optional
import math
from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    suhu: float = Field(..., description="Suhu vulkanisasi/curing (°C)")
    tekanan: float = Field(..., description="Tekanan cetakan ban (bar)")
    waktu_curing: float = Field(..., description="Durasi proses curing (menit)")
    rasio_karet: float = Field(..., description="Rasio campuran karet (-)")
    kekerasan_karet: float = Field(..., description="Kekerasan fisik karet (Shore A)")

    @field_validator("suhu")
    @classmethod
    def validate_suhu(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("Suhu harus berupa angka berhingga (finite).")
        if v < 50.0 or v > 300.0:
            raise ValueError("Suhu di luar batas fisik ekstrem domain [50, 300] °C.")
        return v

    @field_validator("tekanan")
    @classmethod
    def validate_tekanan(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("Tekanan harus berupa angka berhingga (finite).")
        if v < 1.0 or v > 100.0:
            raise ValueError("Tekanan di luar batas fisik ekstrem domain [1, 100] bar.")
        return v

    @field_validator("waktu_curing")
    @classmethod
    def validate_waktu_curing(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("Waktu curing harus berupa angka berhingga (finite).")
        if v < 0.5 or v > 120.0:
            raise ValueError("Waktu curing di luar batas fisik ekstrem domain [0.5, 120] menit.")
        return v

    @field_validator("rasio_karet")
    @classmethod
    def validate_rasio_karet(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("Rasio karet harus berupa angka berhingga (finite).")
        if v <= 0.0 or v > 1.0:
            raise ValueError("Rasio karet harus di antara (0.0, 1.0].")
        return v

    @field_validator("kekerasan_karet")
    @classmethod
    def validate_kekerasan_karet(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("Kekerasan karet harus berupa angka berhingga (finite).")
        if v < 30.0 or v > 100.0:
            raise ValueError("Kekerasan karet di luar batas fisik ekstrem domain [30, 100] Shore A.")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "suhu": 160.0,
                "tekanan": 20.0,
                "waktu_curing": 11.0,
                "rasio_karet": 0.60,
                "kekerasan_karet": 65.0,
            }
        }
    }


class FeatureCheck(BaseModel):
    value: float
    in_dataset_range: bool
    in_ideal_range: bool
    ideal_min: float
    ideal_max: float
    unit: str


class PredictResponse(BaseModel):
    label: int
    label_name: str
    probability_good: float
    probability_bad: float
    threshold: float
    model_version: str
    feature_checks: Dict[str, FeatureCheck]
    warnings: List[str]
    neuron_activations: Optional[Dict[str, Any]] = None


class NeuronActivationsResponse(BaseModel):
    input_features: Dict[str, float]
    scaled_values: List[float]
    layers: List[Dict[str, Any]]
    output_logit: float
    probability_bad: float
    probability_good: float
    threshold: float
    decision_label: str
    model_version: str


class BatchPredictRequest(BaseModel):
    items: List[PredictRequest]


class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str] = None


class ModelInfoResponse(BaseModel):
    model_version: str
    features: List[str]
    threshold: float
    label_mapping: Dict[str, str]
    metrics: Dict[str, Any]
