"""Validated streaming access to challenge patient data. Does not join patient files."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator
import numpy as np
import pandas as pd

from .schema import ALL_COLUMNS, LABEL


def patient_files(data_root: str | Path, sources: tuple[str, ...] = ("A", "B")) -> Iterator[tuple[str, Path]]:
    root = Path(data_root)
    for source in sources:
        if source not in ("A", "B"):
            raise ValueError(f"Unknown hospital dataset: {source}")
        folder = root / f"training_set{source}"
        if not folder.is_dir():
            raise FileNotFoundError(f"Expected {folder}; download/extract the official training data first")
        paths = sorted(folder.glob("*.psv"))
        if not paths:
            raise FileNotFoundError(f"No *.psv files in {folder}")
        for path in paths:
            yield source, path


def read_patient(path: str | Path, source: str) -> pd.DataFrame:
    """Return patient rows with globally unique IDs and strictly ordered time index."""
    path = Path(path)
    if source not in ("A", "B"):
        raise ValueError("Source must be A or B")
    df = pd.read_csv(path, sep="|", na_values=["NaN", "nan", ""], low_memory=False)
    if list(df.columns) != ALL_COLUMNS:
        raise ValueError(f"Incorrect columns in {path}: expected 40 predictors + SepsisLabel in official order")
    if df.empty:
        raise ValueError(f"Empty patient record: {path}")
    df = df.apply(pd.to_numeric, errors="raise")
    label = df[LABEL]
    if label.isna().any() or not label.isin([0, 1]).all():
        raise ValueError(f"Non-binary / missing label in {path}")
    if not np.isfinite(df["ICULOS"].to_numpy()).all():
        raise ValueError(f"Invalid or missing ICU time in {path}")
    if len(df) > 1 and not np.all(np.diff(df["ICULOS"].to_numpy()) > 0):
        raise ValueError(f"ICULOS not strictly increasing in {path}")
    if np.isinf(df[ALL_COLUMNS].to_numpy(dtype=float)).any():
        raise ValueError(f"Infinite values in {path}")
    # Original labels are already shifted ~6 hours ahead of documented onset.
    # DO NOT shift them a second time.
    df["patient_id"] = f"{source}:{path.stem}"
    df["source"] = source
    df["hour_index"] = np.arange(len(df), dtype=np.int32)
    return df


def iter_patients(data_root: str | Path, sources: tuple[str, ...] = ("A", "B"),
                  max_per_source: int | None = None) -> Iterator[pd.DataFrame]:
    counts = {s: 0 for s in sources}
    for source, path in patient_files(data_root, sources):
        if max_per_source is not None and counts[source] >= max_per_source:
            continue
        counts[source] += 1
        yield read_patient(path, source)
