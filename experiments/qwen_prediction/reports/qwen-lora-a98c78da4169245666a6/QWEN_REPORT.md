# Qwen-only LoRA prediction pilot

Qwen itself supplies the binary prediction scores. LoRA adapters were optimized on A/train before B prediction. The previous LSTM/logistic experiment was not modified.

**Exploratory sampled comparison.** B was previously evaluated. This is neither a fresh independent test nor the original full-cohort evaluation. Samples were drawn uniformly within original hourly label classes; inverse-inclusion weights restore the original hourly class prevalence for point estimates. Confidence intervals are conditional pilot intervals.

Training examples: 512; optimizer updates: 256; selected fine-tuned epoch: 2. Test sample: 1024 hours from 972 records.

| Model | Weighted AUROC | Weighted AP | Brier | Hourly sensitivity | Hourly PPV |
|---|---:|---:|---:|---:|---:|
| qwen_base | 0.68283 | 0.03265 | 0.01597 | 0.88281 | 0.01868 |
| qwen_lora | 0.69082 | 0.03192 | 0.01536 | 0.89062 | 0.01856 |
| lstm | 0.76911 | 0.06246 | 0.01360 | 0.65234 | 0.03326 |
| logistic | 0.75720 | 0.11141 | 0.01343 | 0.69531 | 0.03145 |

Qwen cutoffs target 80% weighted A/validation recall after A/calibration-only isotonic calibration. Saved LSTM/logistic cutoffs retain their original full A/validation selection. AP/AUROC do not depend on those cutoffs. Raw Qwen metrics and all confidence intervals are in the aggregate JSON.

- Qwen LoRA versus qwen_base: AP difference -0.000731; paired 95% interval {'lower': -0.008613740830529747, 'upper': 0.013225545973879491, 'valid_draws': 500}; no clear conditional pilot AP improvement.
- Qwen LoRA versus lstm: AP difference -0.030538; paired 95% interval {'lower': -0.08848864967881417, 'upper': 0.016700867775789136, 'valid_draws': 500}; no clear conditional pilot AP improvement.
- Qwen LoRA versus logistic: AP difference -0.079491; paired 95% interval {'lower': -0.1349114778951612, 'upper': -0.0242598627543657, 'valid_draws': 500}; negative conditional pilot AP difference.

Sparse endpoints do not support continuous alarm-burden, early-detection lead-time or official Challenge-utility estimates. Do not compare this sample directly with the older full-B headline metrics. A new independent cohort is needed for confirmatory evaluation. No LLM judge supplies performance labels; published dataset labels supply numerical evaluation.

Base Qwen weights stay frozen, and only declared LoRA adapters in the final 16 language-model layers are trained. This pilot does not claim exhaustive fine-tuning on every training hour, patient-specific explanatory validity, clinical readiness, or improved clinical outcomes. Protected inputs, scores, adapters and optimizer checkpoints remain private.
