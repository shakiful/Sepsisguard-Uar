import pandas as pd
import pytest
from sepsisguard.load import read_patient, patient_files, iter_patients
from sepsisguard.schema import FEATURES, ALL_COLUMNS


def test_patient_count_and_schema(dataset):
    paths = list(patient_files(dataset))
    assert len(paths) == 160
    a = read_patient(paths[0][1], paths[0][0])
    assert len(FEATURES) == 40
    assert list(a.columns[:41]) == ALL_COLUMNS
    assert a["hour_index"].tolist() == list(range(len(a)))
    assert a["patient_id"].iloc[0].startswith("A:")


def test_missing_target_is_rejected(dataset, tmp_path):
    patient = next(iter_patients(dataset))
    patient.loc[0, "SepsisLabel"] = None
    p = tmp_path / "corrupted.psv"
    patient[ALL_COLUMNS].to_csv(p, index=False, sep="|", na_rep="NaN")
    with pytest.raises(ValueError, match="Non-binary"):
        read_patient(p, "A")


def test_non_monotonic_iculos_is_rejected(dataset, tmp_path):
    patient = next(iter_patients(dataset))
    patient.loc[1, "ICULOS"] = patient.loc[0, "ICULOS"]
    p = tmp_path / "bad_time.psv"
    patient[ALL_COLUMNS].to_csv(p, index=False, sep="|", na_rep="NaN")
    with pytest.raises(ValueError, match="strictly increasing"):
        read_patient(p, "A")
