# Separate Qwen-only sepsis prediction experiment

This experiment is on branch `experiment/qwen-prediction-lora-v1`, starting from the preserved original result commit `f9b78d471ae98edc58921e7028210c609569bccd`. The original notebooks, predictor code, weights and results are unchanged. The `main` branch remains the first experiment. This folder and a new branch-specific test workflow are the only additions.

Open `SepsisGuard_Qwen_Only_LoRA.ipynb` in Colab, select A100 80GB / high RAM and Run all. Use the same private Drive containing the completed preparation checkpoint. The notebook uses a separate checkout at `/content/sepsisguard-qwen-prediction`, base-weight cache at `/content/sepsisguard-qwen-model-cache`, and private outputs under `MyDrive/SepsisGuard-UAR/experiments/qwen_prediction/<run-id>`. Original raw records may be reused read-only if present. Missing records are restored into the new checkout; new archive copies also use the new experiment namespace.

## What is actually trained and predicted

- Official pretrained `Qwen/Qwen3.8-27B`, pinned revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, native Transformers 5.22.1, PEFT 0.21.0, BF16 base weights. No hosted inference, remote model code, credentials or external tracing are used.
- LoRA rank 8, alpha 16, dropout 0.05, applied only to declared attention projections in the final 16 language-model layers. Vision tower, language head and original base weights remain frozen. This is parameter-efficient fine-tuning, not full-parameter training of all 27 billion weights.
- The existing Qwen vocabulary head provides next-token logits for the single tokens `0` and `1`. Binary cross-entropy on those two logits trains the adapters. Predictions are their renormalized softmax scores. There is no LSTM, logistic predictor, new linear classifier head or risk from another model inside Qwen inputs.
- Original PhysioNet hourly labels are preserved, with no second six-hour shift. Each numeric prompt contains only observations available at that endpoint: the latest available raw value and its age, plus the change and observation count within the trailing 12 hours, for each of the original 39 A/train-retained features. Missing values remain explicit `NA`. Patient IDs, hospital source, labels, future data and baseline risk scores are excluded from model inputs. The prompt explains the benchmark target and requests the class token; no free-form diagnosis or treatment output is generated.

## Pilot scope, fixed before testing

The first profile is deliberately bounded: 512 A/train endpoints (256 per label), two epochs, batch size 1 with four-example gradient accumulation, AdamW learning rate 1e-4. This uses a real but small subset of the training cohort. It does not claim fine-tuning on every patient-hour. The best completed fine-tuned epoch is selected by weighted A/validation AP on 256 endpoints. The optimizer, adapters, selected checkpoint, shuffle position and RNG states resume at optimizer-step boundaries; immutable private checkpoint generations preserve earlier states.

A/calibration uses 512 endpoints. Weighted isotonic calibration fits only Qwen scores and A/calibration labels. It is separate from the fine-tuning objective and does not run a logistic/LSTM predictor. A/validation selects a threshold targeting at least 80% weighted hourly recall. Both base Qwen and fine-tuned Qwen receive the same calibration/validation treatment. A frozen gate records selected adapter, calibration and threshold identities before B raw inputs can be constructed.

The B pilot has 1,024 endpoints (768 negative, 256 positive). Each split is sampled uniformly without replacement within its original hourly label classes, seed 42. Outside the deliberately balanced training set, inverse-inclusion weights (`population class count / sampled class count`) restore the original hourly class prevalence for estimated metrics. Reporting naive precision from the enriched sample would be misleading and is prohibited. Labels are available to sampling/evaluation code, but never to prediction inputs. Counts, private selected endpoints and weights are recorded.

B was already scored and its results inspected in the first experiment. This is an **exploratory retrospective comparison**, not a new untouched test. Sampled weighted AUROC/AP/Brier/log loss and threshold metrics are estimates from the pilot; they are not measured full-cohort performance. Sparse sampling cannot support continuous alarm episodes, official timing utility or early-detection lead-time claims, so those metrics are intentionally absent.

## Fair comparison

After fine-tuning, score both the selected LoRA adapter and the same pretrained Qwen with adapters disabled. Then read the original private `private_predictions.npz` read-only and extract LSTM/logistic predictions for exactly the same selected B hours. Verify original result SHA256, patient ordering and labels. No older predictor is loaded, run, retrained or used to produce Qwen predictions. The original full-B headline scores are not substituted for matched-sample scores. If saved private predictions are absent, that comparison is explicitly unavailable.

Use 500 paired bootstrap resamples of whole sampled patient records, retaining fixed inclusion weights. These are conditional pilot intervals, not a complete reproduction of the stratified sampling design and not training-uncertainty estimates. Report the Qwen-LoRA-minus-comparator AP difference and interval. Even a positive interval is exploratory evidence, not clinical effectiveness or confirmatory independent performance.

Numerical accuracy is evaluated against published labels. An extra LLM judge is not needed to assign those outcomes. This prediction experiment does not mix in the earlier evidence-writing agent or claim patient-specific explanation quality.

## Execution and preservation

```bash
python -m pip install -e '.[dev,download,agents]' -r experiments/qwen_prediction/requirements.txt
python -m experiments.qwen_prediction.run --data-root data/raw \
  --storage-root /path/to/private/SepsisGuard-UAR --max-minutes 90
```

The 90-minute per-invocation compute budget starts after base-model loading; it bounds training/inference and records an explicit paused status if exhausted. Run all again resumes verified optimizer/prediction checkpoints. A changed source, configuration or runtime identity creates a separate run ID. No inference results or improvement claims are fabricated when execution is incomplete. Input sequences exceeding 2,048 tokens fail explicitly instead of silently dropping features.

Some Colab images include unused `torchao==0.10.0`, which PEFT 0.21 rejects even for ordinary BF16 adapters. The notebook runs `python -m experiments.qwen_prediction.prepare_runtime` after dependency installation. It removes only an installed torchao version below 0.16 from this separate runtime; BF16 LoRA does not need torchao. It does not change Torch, quantize weights, modify private Drive results or alter the earlier notebook.

Private artifacts include `RUN_INPUT.json`, sampled numeric inputs, immutable optimizer checkpoints, portable selected PEFT adapter, calibration mapping, frozen evaluation gate, cached predictions, `QWEN_RESULT.json`, `QWEN_REPORT.md` and aggregate plots. Only aggregate reports and code may be committed publicly. Adapters trained on medical records stay private, along with all per-patient derivatives. Base foundation-model weights stay on runtime disk and can be redownloaded.

## Validation and sources

CPU tests use a tiny, randomly initialized native Qwen3.5 architecture with the same Transformers/PEFT code path. They verify real LoRA gradients through linear/full attention, unchanged base parameters, exact optimizer/RNG resume, corruption rejection, left-padding invariance, weighted metrics, matched baseline alignment, future-value mutation, labels excluded from prompts and pre-test gating. These tests do not use the 27B pretrained checkpoint and are not clinical performance evidence.

Primary implementation references: [official Qwen model card](https://huggingface.co/Qwen/Qwen3.8-27B), [PEFT LoRA](https://huggingface.co/docs/peft/v0.21.0/package_reference/lora), [Transformers native Qwen3.5 architecture](https://huggingface.co/docs/transformers/model_doc/qwen3_5), [PhysioNet 2019 task and original shifted labels](https://physionet.org/content/challenge-2019/1.0.0/).
