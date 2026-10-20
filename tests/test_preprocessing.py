"""
Unit tests for data preprocessing, stratified split, and StandardScaler serialization.
"""

import pytest
import numpy as np
import pandas as pd

from ml.preprocessing import split_dataset, fit_scaler, serialize_scaler, deserialize_scaler
from ml.config import FEATURES, TARGET


def test_stratified_split_ratios():
    """Verify 70% train, 15% validation, and 15% test split on 500 samples."""
    # Create 500 mock rows with 65% label 0 and 35% label 1
    n_rows = 500
    y = np.zeros(n_rows, dtype=int)
    y[:175] = 1  # 35% label 1

    df = pd.DataFrame({
        "suhu": np.linspace(140, 180, n_rows),
        "tekanan": np.linspace(15, 25, n_rows),
        "waktu_curing": np.linspace(8, 15, n_rows),
        "rasio_karet": np.linspace(0.4, 0.8, n_rows),
        "kekerasan_karet": np.linspace(55, 75, n_rows),
        "label": y
    })

    train_df, val_df, test_df = split_dataset(df, test_size=0.30, val_ratio=0.50, random_state=42)

    assert len(train_df) == 350  # 70%
    assert len(val_df) == 75    # 15%
    assert len(test_df) == 75   # 15%

    # Verify stratified label proportions are closely matching
    train_prop = train_df[TARGET].mean()
    val_prop = val_df[TARGET].mean()
    test_prop = test_df[TARGET].mean()

    assert pytest.approx(train_prop, abs=0.03) == 0.35
    assert pytest.approx(val_prop, abs=0.03) == 0.35
    assert pytest.approx(test_prop, abs=0.03) == 0.35


def test_scaler_fit_and_json_reproducibility():
    """Verify serialized scaler reproduces StandardScaler transform identically."""
    X_train = np.array([
        [150.0, 18.0, 10.0, 0.50, 60.0],
        [160.0, 20.0, 11.5, 0.60, 65.0],
        [170.0, 22.0, 13.0, 0.70, 70.0],
    ])

    scaler = fit_scaler(X_train)
    scaler_dict = serialize_scaler(scaler, FEATURES)

    # Reconstruct scaler from dict
    restored_scaler = deserialize_scaler(scaler_dict)

    # Test sample
    test_input = np.array([[155.0, 19.0, 11.0, 0.55, 62.5]])
    orig_scaled = scaler.transform(test_input)
    restored_scaled = restored_scaler.transform(test_input)

    np.testing.assert_allclose(orig_scaled, restored_scaled, rtol=1e-5, atol=1e-5)


def test_scaler_feature_dimension_mismatch():
    """Verify error when input dimension does not match feature list."""
    scaler_dict = {
        "feature_names": ["suhu", "tekanan", "waktu_curing", "rasio_karet", "kekerasan_karet"],
        "mean": [160.0, 20.0, 11.0, 0.60, 65.0],
        "scale": [10.0, 2.0, 1.5, 0.10, 5.0],
        "var": [100.0, 4.0, 2.25, 0.01, 25.0],
        "n_samples_seen": 350
    }
    scaler = deserialize_scaler(scaler_dict)

    with pytest.raises(ValueError, match="Expected input with 5 features"):
        scaler.transform(np.array([[160.0, 20.0]]))  # Only 2 features passed
