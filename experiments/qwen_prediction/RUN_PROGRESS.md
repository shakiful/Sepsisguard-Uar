# Qwen experiment: verified progress and connection blocker

Real Qwen LoRA fine-tuning was launched in the separate Colab notebook and observed through **252 of 256 planned optimizer updates**. The execution environment then became unavailable. Completion, epoch-two validation, calibration and B predictions are **not verified**. The Colab job may have continued independently; no final comparison or improvement is claimed.

## What was actually observed

- Signed-in Colab; private Drive mounted; NVIDIA A100-SXM4-80GB (79.25 GiB VRAM), 167.05 GiB RAM.
- Official PhysioNet records downloaded into the new experiment namespace and validated: A 20,336 records, B 20,000 records. Historical ZIP URLs returned HTTP 404; the downloader used the official anonymous S3 fallback.
- Official pinned Qwen/Qwen3.8-27B checkpoint loaded in BF16 with frozen base weights. Real cross-entropy gradients and AdamW updates trained 5,940,224 LoRA parameters in the final 16 language layers. Predictions use Qwen's own vocabulary logits for 0/1; no old predictor supplies Qwen scores.
- Fixed pilot: 512 A/train hours, two planned epochs, four-example gradient accumulation, 256 planned updates. The final observed log reached update 252 in epoch two.
- Checkpoints were copied to private Drive and read back with SHA256 verification. Checkpoint step 240 had necessarily completed before the later observed updates. A later checkpoint may now exist, but it could not be inspected.
- Epoch one completed and scored 256 A/validation hours. Raw weighted AUROC **0.5688069661**, AP **0.0366775243**, Brier **0.3139166369**; weighted positive rate **0.0216288745**. This is modest discrimination and poor raw probability calibration. It is not a test result or proof of improvement over pretrained Qwen or earlier models.
- Local original-plus-new tests: 73 passed in 61.63s. Actual Colab Qwen tests: 9 passed in 12.01s. Executed source GitHub CI: 73 passed in 86.53s, [run 42561250624](https://github.com/shakiful/Sepsisguard-Uar/actions/runs/42561250624).
- Preinstalled torchao 0.10.0 conflicted with PEFT 0.21.0. The unused incompatible optional package was removed before real training; the canonical notebook includes this correction.

## Preserved work

Executed source commit: `5bb4b1dc52d726596af8144357e4c9235e0e64e9`.

Original `main` remains `f9b78d471ae98edc58921e7028210c609569bccd`. The Qwen source is on `experiment/qwen-prediction-lora-v1`. No original notebook, model weights, data or results were overwritten. New checkpoints, numeric inputs, adapters and per-patient scores remain private and are not committed here.

Executed notebook: [SepsisGuard_Qwen_Only_LoRA.ipynb](https://colab.research.google.com/drive/1KTxQ3rddV9CA_G--sZ4PnkxxiO0eM7q8).

Private checkpoint directory: `MyDrive/SepsisGuard-UAR/experiments/qwen_prediction/qwen-lora-a98c78da4169245666a6`.

## Continue from the actual state

First inspect the existing execution; do not launch a second GPU job while it is active. If it completed, read `QWEN_RESULT.json` and `QWEN_REPORT.md` in the private run directory. If it paused or stopped, resume the existing runner using the same source/configuration and runtime software; optimizer, adapter, RNG and prediction checkpoints are available. The canonical GitHub notebook supports this workflow. The live executed copy contains diagnostic setup history, so use its existing runner cell or the canonical notebook rather than blindly rerunning the earlier diagnostic cells.

Do not change kernels, sampling, model, thresholds or training settings in this run. Before publishing a final result, verify selected adapter change and completed updates, frozen A-only calibration/threshold gate, all saved file hashes, unchanged original result/prediction hashes, and aligned baseline scores for the same B hours.

The intended B comparison is an exploratory 1,024-hour weighted pilot, not a fresh independent or full-cohort test. Qwen also has a smaller training sample and a different input representation than the original predictors. Conditional patient-bootstrap intervals do not capture full sampling-design or training uncertainty. Sparse endpoints do not support continuous alarm burden, lead time, official timing utility, or clinical readiness claims.
