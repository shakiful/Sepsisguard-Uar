# First predictive training result — 2026-11-04

Verified from the user-supplied Colab console output; metrics were checked against the logged histories and the source/input fingerprints. Private predictions and saved weights have not been independently opened by the reviewer.

Run: `training-05ed8fa8a5f01f4f92a8`. Preparation: `pipeline-20261104T190221Z-22906fa2`. Source: `7df0a07e2e2e2a489245164081aae293d887acb4`.

## Validation results

| Model | Best epoch | Epochs completed | Average precision | AUROC | Brier | Precision at 0.5 | Recall at 0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic | 6 | 8 | 0.087473 | 0.775901 | 0.180779 | 5.9799% | 66.0944% |
| LSTM (selected) | 1 | 5 | 0.098821 | 0.772333 | 0.157566 | 5.9049% | 64.3387% |

Selection used A/validation average precision, as specified before training. The LSTM improves AP by 0.011348 (12.97% relative), while logistic has slightly higher AUROC. The LSTM's first-epoch weights were correctly restored after early stopping at epoch 5. Its falling training loss and declining validation AP suggest overfitting; additional epochs alone did not improve this run.

The validation set contains 118,499 hourly endpoints and 2,563 positive endpoints (2.16%). At the fixed 0.5 threshold, selected LSTM precision is 5.90% and recall is 64.34%; approximately 94.10% of its positive predictions are false positives. This is a limitation of the present operating point. Scores were optimized with an A/train positive-class weight of 45.0593 and remain uncalibrated. These values must not be presented as independent test performance or clinical validation.

## Execution and artifacts

- All 37 tests passed in 74.55 seconds.
- Actual training allocation: NVIDIA A100-SXM4-40GB, 39.49 GiB reported VRAM, 83.47 GiB system RAM and 12 CPUs. This differs from the earlier preparation allocation of A100 80GB / 167.05 GiB RAM.
- Both candidates ran on CUDA with bfloat16; no AMP fallback was needed.
- A/train supplied 552,665 endpoints. A/validation supplied checkpoint/architecture selection. Calibration and B raw files remained unopened by training; independent B evaluation and probability calibration are pending.
- The run reports successful model and metadata checkpoint readback under `MyDrive/SepsisGuard-UAR/training/training-05ed8fa8a5f01f4f92a8`.
- Selected weights: `lstm_best.pt`, SHA256 `266d203773981dce01f2b604c29895e4588b0a2568aa16a333d6a6ffa62764bc`. Keep the weights and per-endpoint predictions private.
- Detailed aggregate evidence: [TRAINING_COLAB_EVIDENCE.json](TRAINING_COLAB_EVIDENCE.json).

## Next experiment

Preserve this baseline run and selected weight hash. Calibration and operating-threshold analysis remain to be performed using the reserved development/calibration data, with B locked. Any further architecture or optimization comparisons belong on A/train and A/validation and require a new recorded training result. No independent holdout score or statistically significant improvement is claimed.
