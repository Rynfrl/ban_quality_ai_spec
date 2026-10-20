"""
Application settings and configuration.
"""

import os
from pathlib import Path
from ml.config import (
    BASE_DIR,
    ARTIFACTS_DIR,
    DATA_DIR,
    DEFAULT_DATA_FILE,
    DEFAULT_DATA_SHEET,
    FEATURES,
    TARGET,
    LABEL_MAPPING,
    MODEL_VERSION,
    FEATURE_SPECS,
    OFFLINE_FEATURE_SPECS,
)

HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))
ARTIFACTS_PATH = Path(os.getenv("ARTIFACTS_DIR", str(ARTIFACTS_DIR)))
DATA_FILE_PATH = Path(os.getenv("DATA_PATH", str(DEFAULT_DATA_FILE)))
DATA_SHEET_NAME = os.getenv("DATA_SHEET", DEFAULT_DATA_SHEET)
