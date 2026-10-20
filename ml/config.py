"""
Shared constants and configurations for the Ban Quality ML project.
This serves as the single source of truth for features, target, units, and ranges.
"""

from pathlib import Path
from typing import Dict, Any

# Root directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# Dataset defaults
DEFAULT_DATA_FILE = DATA_DIR / "data_ban.xlsx"
DEFAULT_DATA_SHEET = "data_ban"

# Single source of truth for features and target
FEATURES = (
    "suhu",
    "tekanan",
    "waktu_curing",
    "rasio_karet",
    "kekerasan_karet",
)

OFFLINE_FEATURES = ()

TARGET = "label"

# Label mappings
LABEL_MAPPING = {
    0: "Baik",
    1: "Perlu Perbaikan",
}

# Model version
MODEL_VERSION = "ban-mlp-v1"

# Feature specifications: units, domain valid ranges, observed ranges, and ideal ranges
FEATURE_SPECS: Dict[str, Dict[str, Any]] = {
    "suhu": {
        "unit": "°C",
        "domain_min": 100.0,
        "domain_max": 250.0,
        "dataset_min": 140.0,
        "dataset_max": 180.0,
        "ideal_min": 150.0,
        "ideal_max": 165.0,
        "description": "Suhu vulkanisasi/curing",
    },
    "tekanan": {
        "unit": "bar",
        "domain_min": 5.0,
        "domain_max": 50.0,
        "dataset_min": 15.0,
        "dataset_max": 25.0,
        "ideal_min": 18.0,
        "ideal_max": 22.0,
        "description": "Tekanan cetakan ban",
    },
    "waktu_curing": {
        "unit": "menit",
        "domain_min": 1.0,
        "domain_max": 60.0,
        "dataset_min": 8.0,
        "dataset_max": 15.0,
        "ideal_min": 10.0,
        "ideal_max": 13.0,
        "description": "Durasi proses curing",
    },
    "rasio_karet": {
        "unit": "-",
        "domain_min": 0.1,
        "domain_max": 1.0,
        "dataset_min": 0.4,
        "dataset_max": 0.8,
        "ideal_min": 0.5,
        "ideal_max": 0.7,
        "description": "Rasio campuran karet",
    },
    "kekerasan_karet": {
        "unit": "Shore A",
        "domain_min": 30.0,
        "domain_max": 100.0,
        "dataset_min": 55.0,
        "dataset_max": 75.0,
        "ideal_min": 60.0,
        "ideal_max": 70.0,
        "description": "Kekerasan fisik karet (pengukuran live)",
    },
}

OFFLINE_FEATURE_SPECS: Dict[str, Dict[str, Any]] = {}
