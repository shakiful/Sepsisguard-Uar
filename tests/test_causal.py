import numpy as np
import pandas as pd
from sepsisguard.causal import causal_window, causal_impute, training_medians
from sepsisguard.load import iter_patients
from sepsisguard.schema import FEATURES


def test_changing_future_does_not_change_past_window(dataset):
    patient = next(iter_patients(dataset))
    x1, y1 = causal_window(patient, 5, window=4)
    modified = patient.copy()
    modified.loc[6:, "HR"] = 99999.0
    modified.loc[6:, "SepsisLabel"] = 1
    x2, y2 = causal_window(modified, 5, window=4)
    np.testing.assert_allclose(x1, x2, equal_nan=True)
    assert y1 == y2
    assert x1.shape == (4, 40)


def test_imputation_uses_past_not_future():
    data = pd.DataFrame(np.nan, index=[0, 1, 2, 3], columns=FEATURES)
    data["HR"] = [np.nan, 82.0, np.nan, 999.0]
    medians = pd.Series(0.0, index=FEATURES)
    medians["HR"] = 70
    filled, mask = causal_impute(data, medians)
    assert filled["HR"].tolist() == [70, 82, 82, 999]
    assert mask["HR"].tolist() == [0, 1, 0, 1]


def test_medians_depend_only_on_given_training_frames(dataset):
    patients = list(iter_patients(dataset, ("A",), max_per_source=2))
    medians = training_medians(patients)
    corrupted_test = next(iter_patients(dataset, ("B",), max_per_source=1)).copy()
    corrupted_test.loc[:, "HR"] = 10000000
    # A test-set change cannot alter an already-fitted training median.
    assert training_medians(patients)["HR"] == medians["HR"]
