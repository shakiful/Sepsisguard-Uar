"""Leakage-safe first-stage patient time series transformations.

All future transforms must use parameters fitted on training patients only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from .schema import FEATURES, LABEL


def training_medians(training_patients: list[pd.DataFrame], features: list[str] = FEATURES) -> pd.Series:
    """For Week 1 tests/small samples. Fit ONLY on train patients, never validation/B."""
    if not training_patients:
        raise ValueError("No training patients")
    frame = pd.concat([p[features] for p in training_patients], ignore_index=True)
    # A completely unobserved training feature is filled with 0 after masking.
    return frame.median(numeric_only=True).reindex(features).fillna(0.0)


def causal_impute(patient: pd.DataFrame, medians: pd.Series,
                  features: list[str] = FEATURES) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Forward-fill ONLY past measurements, then training-only median fallback.

    Returns (filled features, observation mask where 1=originally observed).
    """
    original = patient[features].copy()
    mask = original.notna().astype("int8")
    filled = original.ffill().fillna(medians.reindex(features))
    if filled.isna().any().any():
        raise ValueError("Failed to impute all features")
    return filled, mask


def causal_window(patient: pd.DataFrame, end_index: int, window: int,
                  features: list[str] = FEATURES) -> tuple[np.ndarray, int]:
    """Unimputed real observations from [end-window+1, end] only.

    Values remain NaN for unmeasured features (Week 2 will transform them).
    """
    if not isinstance(end_index, int) or not 0 <= end_index < len(patient):
        raise IndexError("end_index outside patient record")
    if window <= 0:
        raise ValueError("Window must be positive")
    start = max(0, end_index - window + 1)
    x = patient.iloc[start:end_index + 1][features].to_numpy(dtype=np.float32, copy=True)
    y = int(patient.iloc[end_index][LABEL])
    return x, y
