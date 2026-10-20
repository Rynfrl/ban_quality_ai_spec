"""
Artifact loader service for FastAPI application.
Ensures atomic loading, validation, and status tracking.
"""

from typing import Dict, Any, Optional
import logging
from pathlib import Path

from ml.artifacts import load_artifacts, ArtifactValidationError
from app.config import ARTIFACTS_PATH

logger = logging.getLogger("ban_quality_ml.artifact_loader")


class ModelArtifactManager:
    """
    Singleton-style manager holding loaded model, scaler, and metadata.
    """
    def __init__(self, artifacts_dir: Path = ARTIFACTS_PATH):
        self.artifacts_dir = artifacts_dir
        self.artifacts: Optional[Dict[str, Any]] = None
        self.is_loaded: bool = False
        self.load_error: Optional[str] = None

    def load(self) -> bool:
        try:
            logger.info("Loading model artifacts from %s...", self.artifacts_dir)
            self.artifacts = load_artifacts(self.artifacts_dir)
            self.is_loaded = True
            self.load_error = None
            logger.info("Successfully loaded model version %s", self.artifacts["model_version"])
            return True
        except Exception as e:
            self.is_loaded = False
            self.artifacts = None
            self.load_error = str(e)
            logger.error("Failed to load artifacts: %s", self.load_error)
            return False

    @property
    def model(self):
        if not self.is_loaded or not self.artifacts:
            return None
        return self.artifacts["model"]

    @property
    def scaler(self):
        if not self.is_loaded or not self.artifacts:
            return None
        return self.artifacts["scaler"]

    @property
    def model_version(self) -> Optional[str]:
        if not self.is_loaded or not self.artifacts:
            return None
        return self.artifacts["model_version"]

    @property
    def threshold(self) -> float:
        if not self.is_loaded or not self.artifacts:
            return 0.50
        return self.artifacts["threshold"]

    @property
    def evaluation(self) -> Optional[Dict[str, Any]]:
        if not self.is_loaded or not self.artifacts:
            return None
        return self.artifacts["evaluation"]

    @property
    def training_history(self) -> Optional[list]:
        if not self.is_loaded or not self.artifacts:
            return None
        return self.artifacts["training_history"]

    @property
    def scaler_data(self) -> Optional[Dict[str, Any]]:
        if not self.is_loaded or not self.artifacts:
            return None
        return self.artifacts.get("scaler_data")

    @property
    def model_config(self) -> Optional[Dict[str, Any]]:
        if not self.is_loaded or not self.artifacts:
            return None
        return self.artifacts.get("model_config")


artifact_manager = ModelArtifactManager()
