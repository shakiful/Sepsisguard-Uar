"""Exact train-only medians, causal filling, and streaming standardization.

Fits read A/train paths only. Memory is bounded by one patient plus one
feature's training observations; the raw fitting matrix is disk-backed.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .load import read_patient
from .schema import FEATURES
from .splits import check_split_integrity


def manifest_digest(manifest: pd.DataFrame) -> str:
    columns = ["patient_id", "source", "split", "n_hours"]
    data = manifest[columns].sort_values("patient_id").to_csv(index=False)
    return hashlib.sha256(data.encode()).hexdigest()


def record_path(root: str | Path, patient_id: str) -> Path:
    source, stem = patient_id.split(":", 1)
    if source not in ("A", "B") or not stem or Path(stem).name != stem or "\\" in stem:
        raise ValueError("Unsafe or invalid patient ID")
    return Path(root) / f"training_set{source}" / f"{stem}.psv"


def validate_hourly(patient: pd.DataFrame) -> None:
    times = patient["ICULOS"].to_numpy(dtype=float)
    if not np.isfinite(times).all() or not np.allclose(times, np.round(times), atol=1e-8, rtol=0):
        raise ValueError("Expected integer ICU hours")
    if len(times) > 1 and not np.allclose(np.diff(times), 1, atol=1e-8, rtol=0):
        raise ValueError("Preprocessing requires contiguous hourly rows; irregular gaps must be resolved explicitly")


@dataclass
class TrainOnlyPreprocessor:
    """Unsupervised feature filtering is fixed before viewing held-out data.

    Default keeps every feature observed at least once and nonconstant after
    causal imputation. No outcome-dependent selection or B fitting occurs.
    Statistics weight hourly rows equally, rather than weighting patients equally.
    """
    features: list[str]
    medians: list[float]
    means: list[float]
    scales: list[float]
    provenance: dict

    @classmethod
    def fit(cls, data_root: str | Path, manifest: pd.DataFrame,
            *, min_observed_fraction: float = 0.0,
            features: list[str] | None = None, scratch_dir: str | Path | None = None):
        check_split_integrity(manifest)
        candidates = list(FEATURES if features is None else features)
        if not candidates or len(candidates) != len(set(candidates)) or not set(candidates) <= set(FEATURES):
            raise ValueError("Features must be a nonempty unique subset of the predictor schema")
        if not 0 <= min_observed_fraction <= 1:
            raise ValueError("min_observed_fraction must lie in [0, 1]")
        train = manifest.loc[manifest.split.eq("train")].sort_values("patient_id")
        if train.empty or not train.source.eq("A").all():
            raise ValueError("Fit requires A training patients only")
        if not (pd.to_numeric(train.n_hours) > 0).all() or not (train.n_hours % 1 == 0).all():
            raise ValueError("Invalid manifest lengths")
        n_hours = int(train.n_hours.sum())
        counts = np.zeros(len(candidates), dtype=np.int64)
        hashes = hashlib.sha256()
        if scratch_dir is not None:
            Path(scratch_dir).mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch_dir) as temp:
            raw = np.memmap(Path(temp) / "train.dat", mode="w+", dtype="float64",
                            shape=(n_hours, len(candidates)))
            offset = 0
            for row in train.itertuples():
                path = record_path(data_root, row.patient_id)
                hashes.update(row.patient_id.encode() + b"\0" + hashlib.sha256(path.read_bytes()).digest())
                patient = read_patient(path, "A")
                validate_hourly(patient)
                if len(patient) != row.n_hours:
                    raise ValueError("Manifest length differs from patient file")
                values = patient[candidates].to_numpy(dtype=float)
                raw[offset:offset + len(patient)] = values
                counts += np.isfinite(values).sum(axis=0)
                offset += len(patient)
            raw.flush()
            keep = (counts > 0) & (counts / n_hours >= min_observed_fraction)
            indices = np.flatnonzero(keep)
            if not len(indices):
                raise ValueError("No observed training features survive filtering")
            selected = [candidates[i] for i in indices]
            medians = np.array([np.nanmedian(raw[:, i]) for i in indices], dtype=float)
            del raw
        # Second pass resets forward fill at each patient boundary.
        scaler = StandardScaler()
        for row in train.itertuples():
            patient = read_patient(record_path(data_root, row.patient_id), "A")
            values = patient[selected].ffill().fillna(pd.Series(medians, index=selected)).to_numpy(float)
            scaler.partial_fit(values)
        variable = scaler.var_ > 1e-12
        if not variable.any():
            raise ValueError("All training features are constant")
        dropped = {f: "unobserved_or_below_fixed_threshold" for f, k in zip(candidates, keep) if not k}
        dropped.update({f: "constant_after_causal_imputation" for f, k in zip(selected, variable) if not k})
        provenance = {
            "format_version": 1, "fit_source": "A", "fit_split": "train",
            "train_patients": len(train), "train_hours": n_hours,
            "manifest_sha256": manifest_digest(manifest),
            "train_ids_sha256": hashlib.sha256("\n".join(train.patient_id).encode()).hexdigest(),
            "train_content_sha256": hashes.hexdigest(),
            "min_observed_fraction": min_observed_fraction,
            "candidate_features": candidates, "dropped_features": dropped,
            "observed_fractions": dict(zip(candidates, (counts / n_hours).tolist())),
            "imputation": "within-patient forward fill then raw A/train median",
            "scaling": "hour-weighted A/train mean and population std after causal imputation",
            "selection": "A/train missingness and constant filter; no supervised selection",
            "label": "original SepsisLabel at window endpoint; no shift",
        }
        return cls([f for f, k in zip(selected, variable) if k], medians[variable].tolist(),
                   scaler.mean_[variable].tolist(), scaler.scale_[variable].tolist(), provenance)

    def transform(self, patient: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        validate_hourly(patient)
        original = patient[self.features]
        mask = original.notna().to_numpy(dtype=np.float32)
        filled = original.ffill().fillna(pd.Series(self.medians, index=self.features)).to_numpy(float)
        scaled = (filled - np.asarray(self.means)) / np.asarray(self.scales)
        if not np.isfinite(scaled).all():
            raise ValueError("Nonfinite transformed values")
        return scaled.astype(np.float32), mask

    def save(self, path: str | Path) -> None:
        from dataclasses import asdict
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".partial")
        tmp.write_text(json.dumps(asdict(self), indent=2, allow_nan=False) + "\n")
        tmp.replace(path)

    @classmethod
    def load(cls, path: str | Path):
        data = json.loads(Path(path).read_text())
        if data["provenance"]["format_version"] != 1:
            raise ValueError("Unsupported preprocessing state")
        return cls(**data)
