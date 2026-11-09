# Completed Qwen-only prediction experiment

The real Qwen LoRA pilot completed and passed independent readback. **This run did not demonstrate improved average precision from fine-tuning Qwen.** Its calibrated AP point estimate was lower than the preserved LSTM and logistic implementations on the matched pilot. The conditional paired interval was below zero against logistic; intervals against base Qwen and LSTM crossed zero.

## What ran

Official `Qwen/Qwen3.8-27B`, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, ran locally in Colab on an NVIDIA A100-SXM4-80GB with 79.25 GiB VRAM and 167.05 GiB system RAM. The base model stayed frozen in BF16. Rank-8 LoRA optimized **5,940,224 parameters** in 76 attention projections across the final 16 language layers. This is parameter-efficient fine-tuning, not updating all 27 billion parameters.

Qwen's existing vocabulary logits for the tokens `0` and `1` supplied the binary prediction scores. No LSTM/logistic scores or auxiliary classifier head supplied Qwen predictions. Published PhysioNet labels supplied evaluation outcomes; another LLM judge was not used.

Training used **512 A/train hours from 487 distinct patient records**, balanced by the original hourly label. Both epochs completed, with 256 optimizer updates. Weighted A/validation AP rose from 0.03668 at epoch one to 0.06165 at epoch two; epoch two was selected. This selection result does not establish test improvement.

A separate 512-hour A/calibration sample fitted weighted isotonic score calibration. A 256-hour A/validation sample selected cutoffs targeting at least 80% weighted recall. The selected adapter, calibration identities and cutoffs were frozen before B prompt construction. Labels retain the original benchmark's six-hour shift; they were not shifted again.

## Matched test results

All four implementations were evaluated on the **same 1,024 B hours from 972 patient records**. The sample contains 768 negative and 256 positive hours, drawn uniformly within label classes. Inverse-inclusion weights recover the original hourly class mass for these pilot estimates: B has 761,995 hours, including 10,780 positive hours, a 1.4147% positive rate. AP below is **average precision**, not a trapezoidal PR-area calculation. AUROC and AP are better when larger; Brier and log loss are better when smaller.

| Implementation | Weighted AUROC | Weighted AP | Brier | Log loss | Hourly sensitivity | Hourly PPV |
|---|---:|---:|---:|---:|---:|---:|
| Preserved logistic | 0.75720 | 0.11141 | 0.01343 | 0.06695 | 69.53% | 3.14% |
| Preserved LSTM | 0.76911 | 0.06246 | 0.01360 | 0.06664 | 65.23% | 3.33% |
| Pretrained Qwen | 0.68283 | 0.03265 | 0.01597 | 0.14170 | 88.28% | 1.87% |
| Fine-tuned Qwen LoRA | 0.69082 | 0.03192 | 0.01536 | 0.10756 | 89.06% | 1.86% |

The fine-tuned Qwen cutoff yields high estimated hourly sensitivity (89.06%) but only 1.86% PPV. That means roughly 1.86% of its flagged hours carry the benchmark-positive label under these pilot weights. This is weak precision. These sparse endpoint results do not estimate the number of continuous alarm episodes or their clinical burden. The threshold metrics use different A/validation sample sizes for Qwen versus the saved older predictors, so they should not be interpreted as a comparison at an identical test operating point.

Calibration introduced ties and altered ranking metrics. The raw Qwen scores are also reported, preventing selective presentation of only the calibrated result:

| Qwen scores before calibration | Weighted AUROC | Weighted AP | Brier | Log loss |
|---|---:|---:|---:|---:|
| Pretrained Qwen | 0.70037 | 0.03973 | 0.15172 | 0.46695 |
| Fine-tuned Qwen LoRA | 0.70173 | 0.04019 | 0.20158 | 0.58418 |

Raw AP changed only from 0.03973 to 0.04019. This small point increase does not demonstrate a useful fine-tuning benefit. The prespecified comparison below uses calibrated scores consistently with the frozen evaluation and saved comparators.

## Uncertainty in AP differences

Five hundred paired bootstrap resamples used whole **sampled** patient records and fixed inclusion weights. These are conditional pilot intervals, not full stratified sampling-design intervals and not training-uncertainty estimates.

| Fine-tuned Qwen minus comparator | AP difference | Conditional paired 95% interval | Interpretation |
|---|---:|---|---|
| Preserved logistic | -0.07949 | [-0.13491, -0.02426] | negative conditional pilot AP difference |
| Preserved LSTM | -0.03054 | [-0.08849, 0.01670] | no clear conditional pilot AP improvement |
| Pretrained Qwen | -0.00073 | [-0.00861, 0.01323] | no clear conditional pilot AP improvement |

## What this supports

The implementation successfully trained and resumed a real 27B foundation model, and the completed pilot on source B can be reproduced from the preserved checkpoints. It does **not** support replacing the earlier predictors with this Qwen configuration. Logistic has the highest AP point estimate in this pilot; LSTM has the highest AUROC point estimate.

This does not establish that Qwen is intrinsically worse than either architecture. Qwen saw only 512 training hours and uses numeric history summaries; the earlier predictors trained on their larger original training data and use a different representation. The complete implementations are being compared, rather than architecture alone. The experiment does not isolate the effects of sample size, input representation, optimization or hospital shift.

B was already evaluated in the first experiment. This is an exploratory retrospective pilot, not a new untouched independent test. Weighted nonlinear ranking metrics from the sparse sample can differ substantially from full-cohort values. The earlier full-B results remain authoritative for that earlier evaluation and were not replaced by these pilot estimates. No B-driven retuning was performed.

No clinical readiness, treatment benefit, early-warning lead time, official Challenge utility, or patient-specific explanation quality is established here. The new experiment produces numerical predictions; it does not replace or repair the earlier explanation-agent audit.

## Execution recovery and preservation

The first browser environment disconnected during the run. Direct Drive inspection later established that both epochs, calibration and 680 fine-tuned B scores had persisted. A subsequent Colab provisioning issue temporarily returned a 22 GB L4, on which no new training or model predictions were launched. When A100 80GB returned, the exact source/software identity was restored and the remaining 344 predictions completed. The frozen protocol hash remained unchanged.

The audit ran in a fresh Python process because the live notebook kernel did not pick up the editable package's source path after installation. This verification correction did not alter model source, sampling, training, calibration or predictions.

Independent checks regenerated all 2,304 sampled prompts and verified the original SHA256 of their 2,138 distinct patient records, unchanged original labels, disjoint patient partitions, and exact patient-hour alignment. All four sets of reported point metrics were recomputed. The selected adapter differs from initialization; the completed optimizer checkpoint was verified, and the portable adapter files were present and matched their runtime copies. Fifteen files totaling 43,925,644 bytes were compared between runtime and private Drive, with the optimizer checkpoint verified separately. The original result and private prediction hashes remain unchanged.

Source `5bb4b1dc52d726596af8144357e4c9235e0e64e9` passed 73 original-plus-new tests in [GitHub CI](https://github.com/shakiful/Sepsisguard-Uar/actions/runs/42561250624), 86.53 seconds. Software tests use a tiny randomly initialized native Qwen topology; the real pretrained model results are the separate Colab evidence above.

The original `main` remains `f9b78d471ae98edc58921e7028210c609569bccd`. Qwen source and aggregate evidence stay on `experiment/qwen-prediction-lora-v1`. Patient inputs, adapters, optimizer snapshots, fitted calibration mappings and individual predictions remain private in Drive. Original notebooks and experiments were not overwritten.

- [Executed Colab notebook](https://colab.research.google.com/drive/1KTxQ3rddV9CA_G--sZ4PnkxxiO0eM7q8)
- [Exact runner-generated report](QWEN_REPORT.md)
- [Aggregate measurements and intervals](QWEN_RESULT.json)
- [Independent readback evidence](POST_RUN_READBACK.json)
- [Weighted ROC and precision–recall curves](AGGREGATE_CURVES.svg)
- [Reusable separate source notebook](../../SepsisGuard_Qwen_Only_LoRA.ipynb)

The four runner/audit aggregate files above are byte-for-byte copies of verified Drive files. This interpretation is separate from those immutable exported measurements.
