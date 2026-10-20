"""
Range checking service for process parameters.
Evaluates observed dataset ranges and engineering ideal ranges.
"""

from typing import Dict, Tuple, List, Any
from ml.config import FEATURE_SPECS
from app.schemas import FeatureCheck


def check_feature_ranges(inputs: Dict[str, float]) -> Tuple[Dict[str, FeatureCheck], List[str]]:
    """
    Evaluate feature inputs against business specifications.
    
    Returns:
        feature_checks: dictionary mapping feature name to FeatureCheck
        warnings: list of human-readable warnings
    """
    feature_checks: Dict[str, FeatureCheck] = {}
    warnings: List[str] = []

    for name, value in inputs.items():
        spec = FEATURE_SPECS.get(name)
        if not spec:
            continue

        val = float(value)
        unit = spec["unit"]
        d_min = spec["dataset_min"]
        d_max = spec["dataset_max"]
        i_min = spec["ideal_min"]
        i_max = spec["ideal_max"]

        in_dataset_range = bool(d_min <= val <= d_max)
        in_ideal_range = bool(i_min <= val <= i_max)

        feature_checks[name] = FeatureCheck(
            value=val,
            in_dataset_range=in_dataset_range,
            in_ideal_range=in_ideal_range,
            ideal_min=i_min,
            ideal_max=i_max,
            unit=unit,
        )

        if not in_dataset_range:
            warnings.append(
                f"Nilai '{name}' ({val} {unit}) berada di luar rentang observasi dataset [{d_min}, {d_max}]."
            )
        elif not in_ideal_range:
            warnings.append(
                f"Nilai '{name}' ({val} {unit}) berada di luar rentang ideal proses [{i_min}, {i_max}]."
            )

    return feature_checks, warnings
