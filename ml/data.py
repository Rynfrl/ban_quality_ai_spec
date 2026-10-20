"""
Data ingestion and validation module for Ban Quality ML.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np

from ml.config import FEATURES, TARGET, FEATURE_SPECS


class DataValidationError(Exception):
    """Raised when data validation fails."""
    pass


def load_dataset(
    filepath: str | Path = "data/data_ban.xlsx",
    sheet_name: str = "data_ban",
) -> pd.DataFrame:
    """
    Load dataset from Excel file.
    
    Args:
        filepath: Path to the Excel file.
        sheet_name: Sheet name to read from.
        
    Returns:
        pd.DataFrame: Loaded raw DataFrame.
        
    Raises:
        FileNotFoundError: If Excel file does not exist.
        ValueError: If sheet is not found.
    """
    path = Path(filepath)
    if not path.is_file():
        # Fallback to alternate location only if looking for default data_ban.xlsx
        if path.name == "data_ban.xlsx" and Path("data_ban.xlsx").is_file():
            path = Path("data_ban.xlsx")
        elif path.name == "data_ban.xlsx" and (Path("data") / "data_ban.xlsx").is_file():
            path = Path("data") / "data_ban.xlsx"
        else:
            raise FileNotFoundError(f"Dataset file not found at: {filepath}")

    try:
        excel_file = pd.ExcelFile(path)
        if sheet_name not in excel_file.sheet_names:
            raise ValueError(
                f"Sheet '{sheet_name}' not found in {path}. Available sheets: {excel_file.sheet_names}"
            )
        df = pd.read_excel(path, sheet_name=sheet_name)
    except Exception as e:
        if isinstance(e, (FileNotFoundError, ValueError)):
            raise e
        raise DataValidationError(f"Failed to read Excel file {path}: {str(e)}") from e

    return df


def validate_dataset(
    df: pd.DataFrame,
    is_training: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Validate DataFrame against schema and business domain requirements.
    
    Args:
        df: Input DataFrame.
        is_training: Whether dataset is expected to include target labels.
        
    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: Cleaned DataFrame and audit report dictionary.
        
    Raises:
        DataValidationError: If critical validation errors are found.
    """
    if df.empty:
        raise DataValidationError("Dataset is empty.")

    cleaned_df = df.copy()
    audit: Dict[str, Any] = {
        "total_rows_initial": len(cleaned_df),
        "total_columns_initial": len(cleaned_df.columns),
        "columns_present": list(cleaned_df.columns),
        "missing_values": {},
        "duplicate_rows": 0,
        "domain_warnings": [],
        "label_distribution": {},
        "status": "valid",
    }

    # 1. Required column validation
    required_cols = list(FEATURES)
    if is_training:
        required_cols.append(TARGET)

    missing_cols = [col for col in required_cols if col not in cleaned_df.columns]
    if missing_cols:
        raise DataValidationError(f"Missing required columns: {missing_cols}")

    # 2. Check complete duplicate rows
    audit["duplicate_rows"] = int(cleaned_df.duplicated().sum())

    # 3. Numeric coercion & NaN checking for features
    for col in FEATURES:
        # Coerce to numeric
        orig_nulls = int(cleaned_df[col].isna().sum())
        cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors="coerce")
        coerced_nulls = int(cleaned_df[col].isna().sum())
        new_nans = coerced_nulls - orig_nulls
        
        audit["missing_values"][col] = coerced_nulls
        if new_nans > 0:
            raise DataValidationError(f"Column '{col}' contains {new_nans} non-numeric values.")

        if coerced_nulls > 0:
            raise DataValidationError(
                f"Column '{col}' contains {coerced_nulls} missing values. Missing values are not permitted."
            )

        # Domain bounds check (hard bounds)
        spec = FEATURE_SPECS.get(col, {})
        d_min = spec.get("domain_min", -np.inf)
        d_max = spec.get("domain_max", np.inf)
        out_of_bounds = cleaned_df[(cleaned_df[col] < d_min) | (cleaned_df[col] > d_max)]
        if not out_of_bounds.empty:
            raise DataValidationError(
                f"Column '{col}' has {len(out_of_bounds)} values outside domain bounds [{d_min}, {d_max}]."
            )

        # Observed range warning (soft audit)
        obs_min = spec.get("dataset_min", d_min)
        obs_max = spec.get("dataset_max", d_max)
        outside_observed = cleaned_df[(cleaned_df[col] < obs_min) | (cleaned_df[col] > obs_max)]
        if not outside_observed.empty:
            audit["domain_warnings"].append(
                f"Column '{col}' has {len(outside_observed)} rows outside typical dataset range [{obs_min}, {obs_max}]."
            )

    # 4. Target validation
    if is_training:
        orig_nulls = int(cleaned_df[TARGET].isna().sum())
        cleaned_df[TARGET] = pd.to_numeric(cleaned_df[TARGET], errors="coerce")
        if cleaned_df[TARGET].isna().sum() > orig_nulls:
            raise DataValidationError("Target column contains non-numeric values.")
        if cleaned_df[TARGET].isna().sum() > 0:
            raise DataValidationError("Target column contains missing values.")

        cleaned_df[TARGET] = cleaned_df[TARGET].astype(int)
        unique_labels = set(cleaned_df[TARGET].unique())
        if not unique_labels.issubset({0, 1}):
            raise DataValidationError(
                f"Invalid target labels detected: {unique_labels}. Target must only contain 0 and 1."
            )
        
        val_counts = cleaned_df[TARGET].value_counts().to_dict()
        audit["label_distribution"] = {str(k): int(v) for k, v in val_counts.items()}

    audit["total_rows_cleaned"] = len(cleaned_df)
    return cleaned_df, audit
