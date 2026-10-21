import numpy as np
import shutil
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader
from sepsisguard.dataset import CausalWindowDataset, seeded_generator
from sepsisguard.eda import run_eda
from sepsisguard.load import read_patient
from sepsisguard.preprocessing import TrainOnlyPreprocessor, record_path
from sepsisguard.splits import make_splits


@pytest.fixture
def prepared(dataset, tmp_path):
    isolated = tmp_path / "data"
    shutil.copytree(dataset, isolated)
    dataset = isolated
    run_eda(dataset, tmp_path / "eda")
    manifest = make_splits(pd.read_csv(tmp_path / "eda/patient_summary.csv"))
    pre = TrainOnlyPreprocessor.fit(dataset, manifest)
    return dataset, manifest, pre


def test_windows_padding_original_target_and_boundaries(prepared):
    root, manifest, pre = prepared
    data = CausalWindowDataset(root, manifest, pre, "train")
    first = data[0]
    assert first["x"].shape == (12, 2 * len(pre.features))
    assert first["time_mask"].tolist() == [False] * 11 + [True]
    assert not first["x"][:11].any()
    row = data.rows.iloc[0]
    patient = read_patient(record_path(root, row.patient_id), "A")
    values, mask = pre.transform(patient)
    np.testing.assert_array_equal(data[11]["x"].numpy(), np.concatenate([values, mask], axis=1)[:12])
    assert data[11]["y"].item() == int(patient.SepsisLabel.iloc[11])
    next_patient = data[int(data.ends[0])]
    assert next_patient["hour_index"] == 0 and next_patient["length"] == 1
    assert next_patient["patient_id"] != first["patient_id"]
    assert not next_patient["x"][:11].any()
    assert data[-1]["hour_index"] == int(data.rows.n_hours.iloc[-1]) - 1
    with pytest.raises(IndexError):
        data[len(data)]


def test_seeded_shuffling_is_reproducible(prepared):
    root, manifest, pre = prepared
    data = CausalWindowDataset(root, manifest, pre, "validation")
    def batch():
        return next(iter(DataLoader(data, batch_size=8, shuffle=True, generator=seeded_generator(42))))
    a, b = batch(), batch()
    assert a["patient_id"] == b["patient_id"]
    assert torch.equal(a["x"], b["x"])
    assert torch.equal(a["hour_index"], b["hour_index"])


def test_holdout_locked_and_manifest_mismatch_rejected(prepared):
    root, manifest, pre = prepared
    with pytest.raises(ValueError, match="locked"):
        CausalWindowDataset(root, manifest, pre, "test_B")
    changed = manifest.copy()
    i = changed.index[changed.split.eq("validation")][0]
    changed.loc[i, "split"] = "calibration"
    with pytest.raises(ValueError, match="different split"):
        CausalWindowDataset(root, changed, pre, "train")


def test_future_mutation_preserves_entire_earlier_sample(prepared):
    root, manifest, pre = prepared
    a = CausalWindowDataset(root, manifest, pre, "validation")
    before = a[5]
    path = record_path(root, before["patient_id"])
    raw = pd.read_csv(path, sep="|")
    raw.loc[6:, "HR"] = 1e8
    raw.loc[6:, "SepsisLabel"] = 1
    raw.to_csv(path, sep="|", index=False)
    b = CausalWindowDataset(root, manifest, pre, "validation")
    after = b[5]
    assert torch.equal(before["x"], after["x"])
    assert torch.equal(before["y"], after["y"])
