"""
Data preprocessing module for Ban Quality ML.
Handles stratified splitting, StandardScaler fitting only on train set,
and scaler JSON serialization/deserialization.
"""

from typing import Tuple, Dict, Any, List
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ml.config import FEATURES, TARGET


class FittedScaler:
    """
    Lightweight scaler wrapper reconstructed from JSON parameters,
    guaranteeing exact same transform without dependency on pickle.
    """
    def __init__(self, feature_names: List[str], mean: List[float], scale: List[float], var: List[float], n_samples_seen: int):
        self.feature_names = list(feature_names)
        self.mean_ = np.array(mean, dtype=np.float64)
        self.scale_ = np.array(scale, dtype=np.float64)
        self.var_ = np.array(var, dtype=np.float64)
        self.n_samples_seen_ = int(n_samples_seen)

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_names].to_numpy()
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Expected input with {len(self.feature_names)} features, got {arr.shape[1]}"
            )
        return (arr - self.mean_) / self.scale_


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.30,
    val_ratio: float = 0.50,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train, validation, and test subsets with stratification.
    
    Default ratio: 70% train, 15% validation, 15% test.
    """
    y = df[TARGET]
    
    # 1. Split train (70%) and temporary holdout (30%)
    train_df, holdout_df = train_test_split(
        df,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    
    # 2. Split holdout into validation (50% of 30% = 15%) and test (50% of 30% = 15%)
    val_df, test_df = train_test_split(
        holdout_df,
        test_size=val_ratio,
        stratify=holdout_df[TARGET],
        random_state=random_state,
    )
    
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def fit_scaler(X_train: pd.DataFrame | np.ndarray) -> StandardScaler:
    """
    Fit StandardScaler strictly on training features only.
    """
    if isinstance(X_train, pd.DataFrame):
        X_arr = X_train[list(FEATURES)].to_numpy()
    else:
        X_arr = np.asarray(X_train)
        
    scaler = StandardScaler()
    scaler.fit(X_arr)
    return scaler


def serialize_scaler(scaler: StandardScaler | FittedScaler, feature_names: List[str] | tuple = FEATURES) -> Dict[str, Any]:
    """
    Convert fitted scaler parameters to JSON-serializable dictionary.
    """
    return {
        "feature_names": list(feature_names),
        "mean": [float(m) for m in scaler.mean_],
        "scale": [float(s) for s in scaler.scale_],
        "var": [float(v) for v in scaler.var_],
        "n_samples_seen": int(scaler.n_samples_seen_ if hasattr(scaler, "n_samples_seen_") else 0),
    }


def deserialize_scaler(data: Dict[str, Any]) -> FittedScaler:
    """
    Reconstruct FittedScaler from dictionary.
    """
    return FittedScaler(
        feature_names=data["feature_names"],
        mean=data["mean"],
        scale=data["scale"],
        var=data["var"],
        n_samples_seen=data.get("n_samples_seen", 0),
    )
