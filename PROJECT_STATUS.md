# SepsisGuard-UAR progress — 2026-11-08

Current status: native Qwen3.8-27B BF16 agent executed in Colab; three-case full-graph functional check and private artifact readback passed. Clinical quality and independent B evaluation remain pending.

Historical training milestone: predictive training completed in the user's Colab run `training-05ed8fa8a5f01f4f92a8`. All 37 tests passed. The selected LSTM has A/validation AP 0.098821 and AUROC 0.772333; its best weights are from epoch 1. CUDA bfloat16 training and Drive readback are confirmed in the supplied output. Scores are uncalibrated and B remains reserved.

## Completed and observed locally

- Repository imported into local Git; implementation and execution history retained.
- Downloaded official v1.0.0 PhysioNet 2019 A/B data via anonymous S3.
- A=20,336; B=20,000; 1,552,210 hourly rows. Sepsis-labeled patients: 2,932 (7.27%); positive hours: 27,916 (1.80%).
- Full schema, label, time-continuity, manifest/data and exact-byte-duplicate audit passed. Zero exact duplicate files. Subject linkage beyond the published file IDs cannot be ruled out by this audit.
- Deterministic patient splits: A train 14,235, validation 3,050, calibration 3,051; B test 20,000. All B exclusively test_B.
- Original validation: 23 tests passed, including future-value mutation, held-out-file deletion, independent train-statistic calculations, patient-boundary resets, original endpoint labels, and deterministic DataLoader ordering.
- Week 2 fitted on 552,665 A/train rows only. Retained 39 predictors (EtCO2 entirely unobserved in A/train). Windows: train 552,665, validation 118,499, calibration 119,051. Verified 128 spread-out windows per development split plus deterministic batches of shape 8 x 12 x 78. B not opened for fitting or window checks.
- Both notebook Python-cell sequences completed using the explicitly identified in-process local fallback. This host denied Jupyter kernel sockets; neither a local Jupyter session nor a Colab result is being claimed for these executions.
- Local validation source commit: 28f1aef005338af437ea565660213aed83b0ac66. Later source changes add first-run source-ZIP upload and this documentation.

## Import verification (2026-10-21)

- Updated code passes 25 local tests, including fresh-subprocess Colab detection checks.
- Notebook code cells pass syntax validation.
- A CPU-only GitHub Actions workflow runs the synthetic and leakage regression tests.

## Cloud and version-control status

- Source import completed on 2026-10-21: commit 4755ee5c26f231e0520808ae34b3fedcb3fe9d9d, 47 project files. Remote tree inspection found no protected patient-level files.
- GitHub Actions run 39648233169 succeeded: 25 tests passed in 42.99 seconds on the CPU runner.
- The user completed Week 1 in Colab and supplied its executed notebook on 2026-10-21. All seven code cells have saved execution counts (9–15), no saved error outputs, and the final milestone. The assistant could not operate the browser directly (Cannot attach to this target).
- Canonical repository: https://github.com/shakiful/Sepsisguard-Uar, supplied by the user on 2026-10-21. The repository is public. Source, tests, notebooks, and aggregate verification reports are suitable for this remote; protected derivatives remain private.
- The original 2026-10-18 CPU Colab attempt did not complete the data pipeline or confirm Drive report persistence. Its old imported notebook is not evidence of successful cloud computation.
- Both notebooks now clone the canonical GitHub repository directly, record its actual commit, and save milestones to private Drive. A starter ZIP upload is no longer required.
- A compute-reporting bug was corrected so a fresh Colab subprocess can be identified before importing google.colab.
- The completed notebook provides evidence that its report and run-log copies to private Drive returned successfully. A separate Drive readback and archive-cache inspection have not been performed. No predictive model, Week 2 cloud run, or GPU training run is claimed.

## Completed Week 1 Colab evidence (2026-10-21)

- Run ID: `20261021T183143Z-82ed151a`; source commit: `4fc9c4a19cefe94d2ffcca724368f0d70d989c2c`.
- Runtime: Python 3.13.16, CPU only, 2 CPUs, 12.67 GiB reported memory, PyTorch 2.11.0+cpu. A Pro subscription or high-RAM allocation is not inferred from these observations.
- Counts and EDA: 40,336 patients, 1,552,210 hours, 2,932 positive patients, 27,916 positive hours; all match local validation.
- Schema, binary targets, contiguous times, manifest/data matching and exact-file-duplicate audit passed; zero exact duplicates. Cohort SHA256 `e7d81a6326a49965e5bba9e4415c33251fd8122051795654e058bd76adcaeee1` matches the independently processed local cohort.
- All 25 tests passed before the real-data audit (62.30 s) and afterward (60.74 s).
- Patient partition: A train 14,235 / validation 3,050 / calibration 3,051; all 20,000 B patients remain test_B.
- The final cell copied reports to `MyDrive/SepsisGuard-UAR/reports/20261021T183143Z-82ed151a/week1` and logs to `MyDrive/SepsisGuard-UAR/runs/20261021T183143Z-82ed151a`, then printed the saved milestone. This is notebook-output evidence of completed copies, not an independent connected-Drive readback.
- The download cell reused already extracted complete datasets; the original transfer and Drive cached archives are not separately verified by this notebook.
- Evidence summary: [WEEK1_COLAB_EVIDENCE.json](reports/verification/WEEK1_COLAB_EVIDENCE.json). The full uploaded notebook remains private; only aggregate evidence is committed.

## Single-workflow entrypoint (2026-11-04)

The user requested one Run-all workflow instead of week-by-week handoffs. The canonical entrypoint is now `notebooks/SepsisGuard_UAR.ipynb`, backed by `scripts/run_pipeline.py`.

- One execution covers source setup, data download/restoration, EDA, the existing fixed partition, complete audit, regression/leakage tests, causal preprocessing and development Dataset checks.
- The completed data run `20261021T183143Z-82ed151a` is imported automatically when present in Drive. Missing old checkpoints lead to a fresh full data pipeline; an existing incomplete or inconsistent checkpoint is rejected.
- Source/configuration-matched snapshots resume automatically. Every run revalidates raw data; EDA and preprocessing reuse require matching input and artifact hashes. Immutable generations preserve the previous Drive checkpoint if a new save is interrupted.
- Results, protected artifacts and logs are kept together under the private Drive run folder with one `PIPELINE_RESULT.json` and `PIPELINE_REPORT.md`.
- Original dated notebooks and results are retained as historical evidence. Future predictive training, calibration and retrieval components should extend the same workflow rather than add calendar-based handoffs.
- Local validation: all 29 tests passed in 44.72 s. The four added integration tests cover runtime-loss restoration, interrupted fitting, saved-artifact corruption, changed raw data, interrupted snapshot publication and exact legacy-manifest import. The unified notebook passes schema and code-cell syntax checks.
- No new unified Colab run, predictive model or retrieval experiment is claimed from this source change.

## Hardware preference update (2026-11-04)

- The user reports access to an A100 and high-RAM Colab Pro runtime. The unified notebook and runner accept GPU-attached runtimes and record the observed GPU/VRAM/system RAM rather than requiring CPU-only mode.
- At source-change validation time the A100 compatibility check used a simulated hardware report. Actual A100/high-RAM allocation is subsequently confirmed by the completed Colab run below. Preprocessing remains on CPU and no CUDA model training or model speedup is claimed.
- Hardware-compatibility validation: all 30 local tests passed in 38.50 s, including a simulated A100/high-RAM report that confirms the runtime is accepted and its hardware metadata reaches the verified snapshot.

## Completed unified Colab run (2026-11-04)

Verified from the user-supplied console output, with source fingerprint matching the reviewed code:

- Run: `pipeline-20261104T190221Z-22906fa2`; source commit: `c7230fe49ccda9a1a6498cb7ef8fc64bd6e256a1`.
- Status: `data_pipeline_passed`; 30 tests passed in 66.66 s.
- Observed allocation: NVIDIA A100-SXM4-80GB, 79.25 GiB reported VRAM, 167.05 GiB system RAM, 12 CPUs; Python 3.13.16 and PyTorch 2.11.0+cu130. Preprocessing executed on CPU; no CUDA model was trained.
- Cohort: 40,336 patients, 1,552,210 hours, all schema/time/manifest checks passing, zero exact-file duplicates. Cohort fingerprint matches the original local and earlier Colab results.
- Imported the exact original audited patient manifest from `20261021T183143Z-82ed151a`. Manifest digest: `9cff1efb61496519205620836df527a23e9ba24a0dad3c763f3b6d023f154ca7`.
- Fitted on A/train only: 14,235 patients and 552,665 hours. Retained 39 predictors; EtCO2 was unobserved in A/train and dropped. Input shape: 12 hours x 78 value/mask channels.
- Development windows: train 552,665 / validation 118,499 / calibration 119,051. Checked 128 spread-out windows per split and 8 x 12 x 78 batches. B was not used for fitting or development Dataset checks; B was included in descriptive full-cohort auditing.
- The output confirms checkpoint publication and readback at `MyDrive/SepsisGuard-UAR/runs/pipeline-20261104T190221Z-22906fa2`. A separate connected-Drive readback by the reviewer has not been performed.
- Aggregate evidence: [UNIFIED_COLAB_EVIDENCE.json](reports/verification/UNIFIED_COLAB_EVIDENCE.json). Predictive models, independent final holdout evaluation, calibration and retrieval experiments remain unimplemented.

## Predictive training implementation (2026-11-04)

- Added current-hour logistic and packed 12-hour LSTM classifiers. Their scores remain uncalibrated.
- A/train alone supplies optimization and positive-class weight. A/validation average precision selects the best epoch and architecture. Calibration and B raw files stay unopened by training.
- The same notebook now calls `scripts.run_training` and restores the exact completed preparation run above. Compact local caches avoid repeated per-window CSV reads; no preprocessing refit is required.
- CUDA training tests bfloat16 forward/backward and uses FP32 when unsupported. Defaults: 1,024-hour batches, 64 LSTM hidden units, 8 logistic / 12 LSTM maximum epochs, patience 4, AdamW lr=0.001, weight_decay=0.0001 and a 90-minute per-invocation budget.
- Epoch checkpoints preserve optimizer/current/best weights and Torch RNG state. Immutable private weight files and hash-verified resume pointers allow epoch-boundary continuation after runtime loss. Predictions and weights remain private.
- Validation: all 37 local tests passed in 87.77 s. The seven new training tests verify actual weight updates, padding invariance, equality with uninterrupted CPU training after resume, model-checkpoint corruption rejection, matching original causal windows and successful training after calibration/B files are removed. Notebook format and code-cell syntax checks also passed.
- Browser control was retried against the user's current Colab tab; both DOM methods still returned `Cannot attach to this target`. No actual A100 training job has been started by the assistant and no predictive research metrics are claimed.

## Completed predictive Colab training (2026-11-04)

- Run `training-05ed8fa8a5f01f4f92a8`, source `7df0a07e2e2e2a489245164081aae293d887acb4`; restored preparation `pipeline-20261104T190221Z-22906fa2` with matching cohort and manifest.
- All 37 tests passed in 74.55 s. Both models trained on CUDA bfloat16 on A100-SXM4-40GB / 83.47 GiB system RAM.
- Logistic: best epoch 6 of 8, validation AP 0.087473, AUROC 0.775901.
- LSTM: best epoch 1, early-stopped after epoch 5, validation AP 0.098821, AUROC 0.772333. LSTM was correctly selected by the predeclared average-precision criterion; its later declining AP suggests overfitting.
- LSTM precision/recall at the predeclared 0.5 threshold: 5.90% / 64.34%. Scores remain uncalibrated and performance is not an independent test result.
- The output reports verified artifacts under `MyDrive/SepsisGuard-UAR/training/training-05ed8fa8a5f01f4f92a8`. Best LSTM weights hash: `266d203773981dce01f2b604c29895e4588b0a2568aa16a333d6a6ffa62764bc`. No separate connected-Drive readback or private-prediction metric recomputation was performed by the reviewer.
- Calibration and B raw files were not opened during training. Independent calibration, final B evaluation and retrieval experiments remain pending.
- Evidence: [TRAINING_COLAB_EVIDENCE.json](reports/verification/TRAINING_COLAB_EVIDENCE.json); interpretation: [TRAINING_RESULTS.md](reports/verification/TRAINING_RESULTS.md).

## Continue from the predictive checkpoint (historical)

1. Preserve the completed training run and selected `lstm_best.pt` hash above. A routine training rerun is unnecessary; the same unchanged configuration resumes this completed result.
2. Use the reserved A/calibration data for probability-calibration work and examine operating thresholds on development data. Keep B locked until the model and evaluation protocol are frozen.
3. If comparing revised training settings, fit on A/train, select on A/validation, preserve this baseline and record the new run separately. Do not infer that more epochs will improve the selected LSTM.

## Deliverables

The source ZIP contains shareable source, two Colab notebooks, tests, aggregate verification reports, and a Git history bundle. A separate private checkpoint ZIP contains protected patient summaries, split manifest, per-file hashes, fitted preprocessing state, notebook execution outputs, environment snapshots, and logs. It excludes raw patient files. Original data can be downloaded again from the documented official S3 mirror.

Real-cohort predictive training is verified from the supplied A100 output. Independent test performance and clinical validity have not been established.

## Proposed agent extension (2026-11-04)

The user requested agentic AI with Qwen 3.5/latest at about 27B and permitted consideration of alternatives. Official source checks identify Qwen3.8-27B as a newer verified candidate; it is proposed as the primary local backbone, with Qwen3.5-27B and Gemma 4 31B as comparisons. The user-reported future allocation is 80GB VRAM / 160GB RAM, so BF16, one shared instance and an initial 8k context cap are proposed. Actual agent memory use, compatibility and explanation quality have not been measured. The design and profile are saved under `docs/AGENTIC_DESIGN.md` and `configs/agent_profile.json`; no LLM download, agent execution or model fine-tuning has occurred.

## Implemented evidence agent (2026-11-04; current)

The single notebook now calls `scripts.run_agents` and preserves the completed predictor. It fits reserved A/calibration sigmoid calibration and runs a bounded local Qwen3.8-27B / LangGraph retrieval, drafting, verification and abstention workflow over three A/validation endpoints. B remains locked. Model revision and Transformers version are pinned; text-only tokenizer/native class compatibility passed.

Actual local execution used scripted responses with a tiny synthetic predictor, not Qwen. It verified calibration, graph execution, checkpoint readback/resume and integrity rejection. Live CDC/NIH retrieval produced 19 passages. Real Qwen generation, real-cohort calibration diagnostics and peak GPU memory are pending: local CUDA is unavailable and browser control cannot attach to Colab. Current results and regression counts are in `reports/verification/AGENT_IMPLEMENTATION_VALIDATION.json`.

Open the latest GitHub notebook on A100 80GB/high RAM and Run all with the same Drive. Existing Drive notebook copies retain old code. Review actual private `AGENT_RESULT.json`, report and case traces before making research claims. The same-model verifier is not independent factual ground truth. Controlled comparisons and blinded explanation annotation remain pending.

## Verified real Qwen agent execution (2026-11-08; current)

- Authentication and private Drive mount completed in the connected browser, including the user-approved Google Drive for desktop OAuth connection. A100-SXM4-80GB (79.25 GiB) / 167.05 GiB RAM was directly observed; Torch 2.14.1+cu130, Transformers 5.22.1.
- Frozen LSTM/preparation restored without retraining. Sigmoid calibration fitted on 119,051 reserved A/calibration hours, 2,574 positive. Non-flat fit; Brier 0.020253 / log loss 0.090618 are calibration-fit diagnostics only. B remained excluded from calibration and inference.
- Live reference HTTP 403 fixed by bundled CDC/NIH text snapshots: 19 passages, both source SHA256 hashes exactly match 2026-11-04. Offline and corruption tests passed; wheel assets verified.
- First real native BF16 Qwen run `agents-b5667526ece3287af39f` (source 69ae1bc) generated three valid planner abstentions. This run remains saved.
- Follow-up `agents-dc179bf37a3ea479c35b` (source cd6c601) clarified reference lookup before evidence sufficiency. Same sample, predictor, calibration, corpus and budgets. Three cases completed planning, local retrieval, drafting and verification: nine Qwen calls, three retrievals, six accepted claims, no abstentions.
- Generation calls totaled 62.77 seconds and 483 output tokens; this excludes checkpoint download/load and other work. Model-load peak allocated memory was 50.96 GiB. Native reference kernels were used; generation peak memory and optimized-kernel performance were not measured.
- Runner artifact copy/readback and a separate checksum read of all three saved case files passed. Private outputs are under `MyDrive/SepsisGuard-UAR/agents/agents-dc179bf37a3ea479c35b`. Both source changes passed all 56 tests in CI.
- Outputs were general educational sepsis context, with identical retrieval queries/passages. This does not establish patient-specific explanation quality, uncertainty-routing benefit, improved AP, clinical validity or independent factual accuracy. Controlled comparisons and blinded annotation remain next research work.
- Aggregate evidence: [AGENT_COLAB_EVIDENCE.json](reports/verification/AGENT_COLAB_EVIDENCE.json).


## Frozen evaluation implementation (2026-11-08)

The canonical notebook now invokes `scripts.run_evaluation` for frozen LSTM/logistic clinical performance and 12-patient explanation grounding. An immutable, verified protocol gates B predictive access. A/calibration supplies calibration; A/validation selects thresholds and the outcome-blind explanation sample. Official challenge utility is parity-tested against the pinned BSD-licensed scorer. Whole-patient bootstrap intervals, alarm burden, automated explanation proxies and a private blank blinded-review packet are implemented. All 64 local tests passed, including synthetic end-to-end execution. Real evaluation results are pending execution; independent expert ratings are not fabricated. See `docs/EVALUATION_PROTOCOL.md`.
