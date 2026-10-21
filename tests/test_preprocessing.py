import json
import shutil

import numpy as np
import pandas as pd
import pytest

from sepsisguard.eda import run_eda
from sepsisguard.load import read_patient
from sepsisguard.preprocessing import TrainOnlyPreprocessor, record_path
from sepsisguard.splits import make_splits, check_split_integrity


@pytest.fixture
def fitted(dataset, tmp_path):
    isolated = tmp_path / "data"
    shutil.copytree(dataset, isolated)
    dataset = isolated
    run_eda(dataset, tmp_path / "eda")
    manifest = make_splits(pd.read_csv(tmp_path / "eda/patient_summary.csv"))
    return dataset, manifest, TrainOnlyPreprocessor.fit(dataset, manifest)


def test_fit_never_reads_nontraining_data_and_is_invariant(fitted):
    root, manifest, pre = fitted
    # Delete every held-out file. A refit must still succeed with identical state.
    for row in manifest.loc[manifest.split.ne("train")].itertuples():
        record_path(root, row.patient_id).unlink()
    again = TrainOnlyPreprocessor.fit(root, manifest)
    assert pre == again
    assert pre.provenance["fit_source"] == "A"


def test_preprocessing_matches_independent_train_calculation(fitted):
    root, manifest, pre = fitted
    patients = [read_patient(record_path(root, row.patient_id), "A")
                for row in manifest.loc[manifest.split.eq("train")].itertuples()]
    medians = pd.concat([p[pre.features] for p in patients]).median()
    transformed = pd.concat([p[pre.features].ffill().fillna(medians) for p in patients])
    np.testing.assert_allclose(pre.medians, medians.to_numpy())
    np.testing.assert_allclose(pre.means, transformed.mean().to_numpy(), atol=1e-10)
    np.testing.assert_allclose(pre.scales, transformed.std(ddof=0).to_numpy(), atol=1e-10)


def test_future_features_and_labels_cannot_affect_past_transforms(fitted):
    root, manifest, pre = fitted
    row = manifest.loc[manifest.split.eq("validation")].iloc[0]
    patient = read_patient(record_path(root, row.patient_id), "A")
    before, mask = pre.transform(patient)
    changed = patient.copy()
    changed.loc[6:, [f for f in pre.features if f != "ICULOS"]] = 1e8
    changed.loc[6:, "SepsisLabel"] = 1
    after, after_mask = pre.transform(changed)
    np.testing.assert_array_equal(before[:6], after[:6])
    np.testing.assert_array_equal(mask[:6], after_mask[:6])
    prefix, _ = pre.transform(patient.iloc[:6])
    np.testing.assert_array_equal(prefix, before[:6])


def test_serialization_and_forbidden_features(fitted, tmp_path):
    root, manifest, pre = fitted
    pre.save(tmp_path / "pre.json")
    assert pre == TrainOnlyPreprocessor.load(tmp_path / "pre.json")
    for bad in (["SepsisLabel"], ["n_hours"], ["ever_sepsis"], ["HR", "HR"]):
        with pytest.raises(ValueError, match="Features"):
            TrainOnlyPreprocessor.fit(root, manifest, features=bad)


def test_unknown_source_rejected(fitted):
    _, manifest, _ = fitted
    manifest.loc[0, "source"] = "C"
    with pytest.raises(ValueError, match="source"):
        check_split_integrity(manifest)


def test_all_missing_and_constant_features_are_removed(fitted):
    root, manifest, _ = fitted
    for row in manifest.loc[manifest.split.eq("train")].itertuples():
        path = record_path(root, row.patient_id)
        df = pd.read_csv(path, sep="|")
        df["EtCO2"] = np.nan
        df["Gender"] = 1.0
        df.to_csv(path, sep="|", index=False)
    pre = TrainOnlyPreprocessor.fit(root, manifest)
    assert "EtCO2" not in pre.features and "Gender" not in pre.features
    assert pre.provenance["dropped_features"]["Gender"] == "constant_after_causal_imputation"


def test_forward_fill_resets_at_patient_boundary(fitted):
    root, manifest, pre = fitted
    row = manifest.iloc[0]
    patient = read_patient(record_path(root, row.patient_id), row.source)
    patient.loc[:, "HR"] = np.nan
    values, mask = pre.transform(patient)
    i = pre.features.index("HR")
    expected = (pre.medians[i] - pre.means[i]) / pre.scales[i]
    np.testing.assert_allclose(values[:, i], expected, rtol=1e-6)
    assert not mask[:, i].any()


def test_irregular_hour_gaps_rejected(fitted):
    root, manifest, pre = fitted
    row = manifest.iloc[0]
    patient = read_patient(record_path(root, row.patient_id), row.source)
    patient.loc[1:, "ICULOS"] += 1
    with pytest.raises(ValueError, match="contiguous"):
        pre.transform(patient)
