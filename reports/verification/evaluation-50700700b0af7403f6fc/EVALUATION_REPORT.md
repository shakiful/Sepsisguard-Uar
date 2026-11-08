# SepsisGuard retrospective evaluation

Protocol: `50700700b0af7403f6fc66f19d27ea93b82d3e969696c26fb484d4bf65bd7133`

Thresholds and all scoring rules were fixed using A/validation and frozen before predictive evaluation on B. Earlier descriptive auditing was not used for model or threshold selection. Calibration used A/calibration. B is the held-out source from the public development data, not the hidden Challenge leaderboard test set or a prospective clinical trial.

| Model | B AUROC (95% patient-bootstrap CI) | B AP (95% CI) | Brier | ECE10 |
|---|---:|---:|---:|---:|
| lstm | 0.7470 (0.7318–0.7605) | 0.0540 (0.0490–0.0603) | 0.0137 | 0.0026 |
| logistic | 0.7210 (0.7057–0.7356) | 0.0543 (0.0481–0.0625) | 0.0139 | 0.0044 |

| Model / calibrated rule | Hourly sensitivity | PPV | Nonseptic patients with any alarm | Alarm episodes /100 nonseptic hours | Normalized challenge utility |
|---|---:|---:|---:|---:|---:|
| lstm / calibrated_0_5 | 0.0000 | undefined | 0.0000 | 0.0000 | 0.0000 |
| lstm / validation_recall_80 | 0.6442 | 0.0317 | 0.9988 | 6.2754 | 0.0635 |
| lstm / validation_utility | 0.3352 | 0.0677 | 0.1179 | 0.9311 | 0.2110 |
| logistic / calibrated_0_5 | 0.0024 | 0.1116 | 0.0004 | 0.0079 | 0.0019 |
| logistic / validation_recall_80 | 0.6599 | 0.0286 | 0.6390 | 7.0889 | 0.0112 |
| logistic / validation_utility | 0.4717 | 0.0472 | 0.3302 | 3.0153 | 0.2025 |

Explanation sample: 12 distinct A/validation patients, stratified by risk and missingness. Output cases 12; abstentions 0. Explicit patient-observation grounding: 0/12. Unique queries 3/12. Rechecked citation/numeric claims 24/24.

These are automated proxies. Citation matching and a same-model verifier do not establish independent clinical correctness. A blinded clinician review packet with blank rating fields is saved privately; no expert ratings are claimed. Input ablation measures sensitivity to removed channels, not causal clinical attribution.

The original hourly targets are already shifted six hours. Label-relative detection and lead-time summaries are not measured clinical-onset lead times. Challenge utility uses the official onset convention. Alarm episodes use a predeclared six-hour refractory period. Thresholds were not retuned on B. Further predictor selection using these B results would consume the holdout.
