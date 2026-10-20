"""
Unit tests for data ingestion, schema validation, and domain checks.
"""

import pytest
import pandas as pd
import numpy as np

from ml.data import load_dataset, validate_dataset, DataValidationError
from ml.config import FEATURES, TARGET, DEFAULT_DATA_FILE, DEFAULT_DATA_SHEET


def test_load_dataset_success():
    """Verify loading real dataset returns 500 rows and correct columns."""
    df = load_dataset(DEFAULT_DATA_FILE, sheet_name=DEFAULT_DATA_SHEET)
    assert not df.empty
    assert len(df) == 500
    for col in FEATURES:
        assert col in df.columns
    assert TARGET in df.columns


def test_load_dataset_file_not_found():
    """Verify loading non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_dataset("non_existent_file.xlsx")


def test_load_dataset_invalid_sheet():
    """Verify invalid sheet name raises ValueError."""
    with pytest.raises(ValueError):
        load_dataset(DEFAULT_DATA_FILE, sheet_name="invalid_sheet_xyz")


def test_validate_dataset_valid():
    """Verify valid dataset passes validation and produces audit report."""
    valid_df = pd.DataFrame({
        "suhu": [150.0, 160.0, 170.0],
        "tekanan": [18.0, 20.0, 22.0],
        "waktu_curing": [10.0, 11.5, 12.0],
        "rasio_karet": [0.55, 0.60, 0.65],
        "kekerasan_karet": [62.0, 65.0, 68.0],
        "label": [0, 1, 0]
    })
    cleaned, audit = validate_dataset(valid_df, is_training=True)
    assert len(cleaned) == 3
    assert audit["status"] == "valid"
    assert audit["label_distribution"] == {"0": 2, "1": 1}


def test_validate_dataset_missing_column():
    """Verify missing required column raises DataValidationError."""
    invalid_df = pd.DataFrame({
        "suhu": [160.0],
        "tekanan": [20.0],
        "waktu_curing": [11.0],
        "rasio_karet": [0.60],
        # 'kekerasan_karet' is missing
        "label": [0]
    })
    with pytest.raises(DataValidationError, match="Missing required columns"):
        validate_dataset(invalid_df, is_training=True)


def test_validate_dataset_invalid_target_label():
    """Verify non-binary target label raises DataValidationError."""
    invalid_df = pd.DataFrame({
        "suhu": [160.0],
        "tekanan": [20.0],
        "waktu_curing": [11.0],
        "rasio_karet": [0.60],
        "kekerasan_karet": [65.0],
        "label": [2]  # Invalid, only 0 or 1 allowed
    })
    with pytest.raises(DataValidationError, match="Invalid target labels"):
        validate_dataset(invalid_df, is_training=True)


def test_validate_dataset_non_numeric_feature():
    """Verify non-numeric values in features raise DataValidationError."""
    invalid_df = pd.DataFrame({
        "suhu": ["abc"],
        "tekanan": [20.0],
        "waktu_curing": [11.0],
        "rasio_karet": [0.60],
        "kekerasan_karet": [65.0],
        "label": [0]
    })
    with pytest.raises(DataValidationError, match="non-numeric"):
        validate_dataset(invalid_df, is_training=True)


def test_validate_dataset_domain_bounds():
    """Verify values exceeding extreme physical domain bounds raise error."""
    invalid_df = pd.DataFrame({
        "suhu": [999.0],  # Out of domain bounds
        "tekanan": [20.0],
        "waktu_curing": [11.0],
        "rasio_karet": [0.60],
        "kekerasan_karet": [65.0],
        "label": [0]
    })
    with pytest.raises(DataValidationError, match="outside domain bounds"):
        validate_dataset(invalid_df, is_training=True)
