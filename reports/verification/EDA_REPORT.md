# Week 1 Exploratory Data Analysis

**Real PhysioNet records; cohort completeness is reported below.**

Dataset: PhysioNet/CinC Challenge 2019 v1.0.0. Data are retrospective and the target is the **already shifted** challenge label.

- Patients examined: **40,336**
- Hourly observations: **1,552,210**
- Positive hourly observations: **27,916** (1.80%)
- Patients with any positive hour: **2,932** (7.27%)
- Source A: **20,336** / 20,336 expected
- Source B: **20,000** / 20,000 expected
- Full expected training cohort represented: **yes**
- Limited smoke-test mode: **no**

## Five features with highest missing fractions

- Bilirubin_direct: 99.8%
- Fibrinogen: 99.3%
- TroponinI: 99.0%
- Bilirubin_total: 98.5%
- Alkalinephos: 98.4%

## Outputs

- `patient_summary.csv`: one row per patient, including labels and source.
- `dataset_summary.csv`: aggregated patient/hour class distributions.
- `feature_missingness.csv`: missing observation fractions per predictor.
- `figures/*.png`: aggregate visualizations.

## Methodological notes

- `SepsisLabel` was shifted in the original challenge. Never shift it again when using original evaluation.
- Patient-level prevalence and hourly class balance are **different** statistics.
- No train/validation/test preprocessing was fitted during this descriptive analysis.
- EDA on the complete source B is **descriptive only**; avoid using it to choose models, features, or thresholds when preserving B as a test set.
- This report does **not** claim predictive accuracy or clinical validation.
