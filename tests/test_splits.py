import pandas as pd
import pytest
from sepsisguard.eda import run_eda
from sepsisguard.splits import make_splits, check_split_integrity


def test_disjoint_reproducible_splits(dataset, tmp_path):
    run_eda(dataset, tmp_path / "report")
    summary = pd.read_csv(tmp_path / "report" / "patient_summary.csv")
    m1 = make_splits(summary, seed=42)
    m2 = make_splits(summary, seed=42)
    pd.testing.assert_frame_equal(m1, m2)
    assert len(m1) == len(summary)
    assert (m1[m1.source == "B"].split == "test_B").all()
    assert set(m1[m1.source == "A"].split) == {"train", "validation", "calibration"}
    assert m1.patient_id.is_unique
    check_split_integrity(m1)
    for split in ("train", "validation", "calibration"):
        assert set(m1[m1.split == split].ever_sepsis) == {0, 1}


def test_overlap_is_rejected(dataset, tmp_path):
    run_eda(dataset, tmp_path / "report")
    manifest = make_splits(pd.read_csv(tmp_path / "report" / "patient_summary.csv"))
    bad = pd.concat([manifest, manifest.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="overlap"):
        check_split_integrity(bad)


def test_siteB_cannot_be_training(dataset, tmp_path):
    run_eda(dataset, tmp_path / "report")
    manifest = make_splits(pd.read_csv(tmp_path / "report" / "patient_summary.csv"))
    bidx = manifest.index[manifest.source == "B"][0]
    manifest.loc[bidx, "split"] = "train"
    with pytest.raises(ValueError, match="B patients"):
        check_split_integrity(manifest)
