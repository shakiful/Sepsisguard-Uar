"""Lazy reproducible PyTorch windows; no expanded (hour x 12) dataset on disk."""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .load import read_patient
from .preprocessing import TrainOnlyPreprocessor, manifest_digest, record_path
from .splits import check_split_integrity


class CausalWindowDataset(Dataset):
    """One inclusive trailing 12-hour window per original row, left zero padded.

    x contains standardized values followed by original-observation indicators.
    time_mask distinguishes real rows from padding. Labels are unchanged.
    The full preceding history may supply forward-fill values within a window.
    B access is denied unless explicitly unlocked for a final frozen evaluation.
    """
    def __init__(self, data_root: str | Path, manifest: pd.DataFrame,
                 preprocessor: TrainOnlyPreprocessor, split: str,
                 window: int = 12, cache_size: int = 8, allow_test_B: bool = False):
        check_split_integrity(manifest)
        if split not in ("train", "validation", "calibration", "test_B"):
            raise ValueError("Unknown split")
        if split == "test_B" and not allow_test_B:
            raise ValueError("B holdout locked; unlock only for frozen final evaluation")
        if window <= 0 or not isinstance(window, int) or cache_size <= 0:
            raise ValueError("Window and cache size must be positive integers")
        if manifest_digest(manifest) != preprocessor.provenance["manifest_sha256"]:
            raise ValueError("Preprocessor belongs to a different split manifest")
        self.rows = manifest.loc[manifest.split.eq(split)].sort_values("patient_id").reset_index(drop=True)
        if self.rows.empty:
            raise ValueError("Empty split")
        lengths = self.rows.n_hours.to_numpy(dtype=np.int64)
        if (lengths <= 0).any() or not np.array_equal(lengths, self.rows.n_hours.to_numpy()):
            raise ValueError("Invalid manifest lengths")
        self.ends = np.cumsum(lengths)
        self.data_root = Path(data_root)
        self.preprocessor = preprocessor
        self.window = window
        self.cache_size = cache_size
        self.cache = OrderedDict()

    def __len__(self):
        return int(self.ends[-1])

    def _patient(self, index):
        if index not in self.cache:
            row = self.rows.iloc[index]
            patient = read_patient(record_path(self.data_root, row.patient_id), row.source)
            if len(patient) != row.n_hours:
                raise ValueError("Manifest length differs from patient file")
            values, mask = self.preprocessor.transform(patient)
            self.cache[index] = (np.concatenate([values, mask], axis=1), patient.SepsisLabel.to_numpy(np.int64))
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        self.cache.move_to_end(index)
        return self.cache[index]

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        patient_index = int(np.searchsorted(self.ends, index, side="right"))
        hour = int(index - (self.ends[patient_index - 1] if patient_index else 0))
        values, labels = self._patient(patient_index)
        length = min(self.window, hour + 1)
        x = np.zeros((self.window, values.shape[1]), dtype=np.float32)
        x[-length:] = values[hour - length + 1:hour + 1]
        time_mask = np.zeros(self.window, dtype=bool)
        time_mask[-length:] = True
        return {"x": torch.from_numpy(x), "y": torch.tensor(labels[hour], dtype=torch.long),
                "time_mask": torch.from_numpy(time_mask), "length": length,
                "patient_id": self.rows.patient_id.iloc[patient_index], "hour_index": hour}


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def seeded_generator(seed: int = 42):
    return torch.Generator().manual_seed(seed)
