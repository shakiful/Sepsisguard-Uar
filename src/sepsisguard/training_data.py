"""Compact A/train and A/validation cache; no calibration/B file reads."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .load import read_patient
from .preprocessing import manifest_digest, record_path
from .splits import check_split_integrity


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def cache_split(data_root, manifest, preprocessor, inventory, split, output, *, allow_calibration=False):
    allowed = ('train', 'validation', 'calibration') if allow_calibration else ('train', 'validation')
    if split not in allowed:
        raise ValueError('Training cache permits A/train and A/validation only')
    check_split_integrity(manifest)
    if manifest_digest(manifest) != preprocessor.provenance['manifest_sha256']:
        raise ValueError('Preprocessing and manifest do not match')
    rows = manifest.loc[manifest.split.eq(split)].sort_values('patient_id').reset_index(drop=True)
    if rows.empty or not rows.source.eq('A').all():
        raise ValueError('Expected A patients only')
    inventory = inventory.set_index('patient_id', verify_integrity=True)
    content = hashlib.sha256()
    for row in rows.itertuples():
        path = record_path(data_root, row.patient_id)
        observed = file_hash(path)
        if row.patient_id not in inventory.index or observed != inventory.loc[row.patient_id, 'sha256']:
            raise ValueError('Patient contents changed since preparation')
        content.update(row.patient_id.encode() + b'\0' + bytes.fromhex(observed))
    if split == 'train' and content.hexdigest() != preprocessor.provenance['train_content_sha256']:
        raise ValueError('Training contents differ from the fitted preprocessing input')
    key = {'split': split, 'manifest': manifest_digest(manifest), 'content': content.hexdigest(),
           'preprocessor': hashlib.sha256(json.dumps(preprocessor.__dict__, sort_keys=True).encode()).hexdigest()}
    output = Path(output)
    marker = output / 'cache.json'
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved['input'] == key and all((output / p).is_file() and file_hash(output / p) == h for p, h in saved['files'].items()):
            return CompactWindows(output)
    output.mkdir(parents=True, exist_ok=True)
    marker.unlink(missing_ok=True)
    total = int(rows.n_hours.sum())
    values = np.lib.format.open_memmap(output / 'values.npy', mode='w+', dtype=np.float32,
                                      shape=(total, len(preprocessor.features) * 2))
    labels = np.lib.format.open_memmap(output / 'labels.npy', mode='w+', dtype=np.int64, shape=(total,))
    starts = np.lib.format.open_memmap(output / 'starts.npy', mode='w+', dtype=np.int64, shape=(total,))
    offset = 0
    for i, row in enumerate(rows.itertuples()):
        patient = read_patient(record_path(data_root, row.patient_id), 'A')
        if len(patient) != row.n_hours:
            raise ValueError('Patient length differs from manifest')
        x, mask = preprocessor.transform(patient)
        n = len(patient)
        values[offset:offset+n] = np.concatenate([x, mask], axis=1)
        labels[offset:offset+n] = patient.SepsisLabel.to_numpy(np.int64)
        starts[offset:offset+n] = offset
        offset += n
        if (i + 1) % 2000 == 0:
            print(f'Cached {split}: {i+1:,}/{len(rows):,} patients', flush=True)
    values.flush(); labels.flush(); starts.flush()
    del values, labels, starts
    saved = {'input': key, 'patients': len(rows), 'hours': total,
             'files': {p: file_hash(output / p) for p in ('values.npy', 'labels.npy', 'starts.npy')}}
    temp = marker.with_suffix('.partial')
    temp.write_text(json.dumps(saved, indent=2) + '\n')
    temp.replace(marker)
    return CompactWindows(output)


class CompactWindows(Dataset):
    """One trailing window per endpoint; valid rows first, zero padding last."""
    def __init__(self, root, window=12):
        self.root = str(root)
        self.window = window
        self.values = np.load(Path(root) / 'values.npy', mmap_mode='r')
        self.labels = np.load(Path(root) / 'labels.npy', mmap_mode='r')
        self.starts = np.load(Path(root) / 'starts.npy', mmap_mode='r')
        self.channels = self.values.shape[1]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        start = max(int(self.starts[index]), index - self.window + 1)
        length = index - start + 1
        x = np.zeros((self.window, self.channels), dtype=np.float32)
        x[:length] = self.values[start:index+1]
        return {'x': torch.from_numpy(x), 'y': int(self.labels[index]), 'length': length}
