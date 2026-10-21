# SepsisGuard-UAR — ChatGPT Work handoff

**Project type:** Solo research project; course milestone followed by publication-quality research extension.
**Compute preference:** Google Colab Pro is the **main compute environment**. Windows RTX 4060 and an Ubuntu server are fallbacks. Store durable outputs on Google Drive. Do not assume any GPU model is guaranteed, and don't waste Colab GPU compute units on the Week 1 CPU-only EDA.
**Safety and validity:** Public retrospective PhysioNet 2019 sepsis data, not a medical device, and not suitable for individual care decisions. Never provide clinical treatment instructions or upload identifiable data to external AI services.

## User intent

Take ownership of practical implementation in ChatGPT Work. Work through the implementation steps, interact with signed-in Google Colab/Drive and GitHub through available authorized UI/connectors, and leave artifacts plus concise progress summaries. Do not claim a Colab job or cloud execution succeeded unless observed. Never silently use paid cloud resources for lengthy runs without checking the session/runtime and expected costs.

## Current checkpoint (2026-10-21)

Canonical repository: https://github.com/shakiful/Sepsisguard-Uar (user-supplied public source repository). See PROJECT_STATUS.md for measured local results and pending cloud verification. Week 1 full-cohort local validation and Week 2 train-only preprocessing have completed. The notebooks now clone this repository directly and record the actual source commit; no starter ZIP is required.

## Immediate Work sequence (do in order)

1. Check authorized connected tabs/apps for Colab, Drive and GitHub. If login or connection is missing, stop on that item and give precise sign-in steps; continue other available steps.
2. Use the existing user-identified repository `shakiful/Sepsisguard-Uar`. Preserve its history and unrelated edits; never push to an unrelated repository.
3. Import this package's content to the repository, preserving `.gitignore`. Push only source and aggregated de-identified results. Do not commit raw `.psv` records or patient-level summaries.
4. Open the Week 1 notebook from this GitHub repository in Colab. Use CPU. Its bootstrap creates `MyDrive/SepsisGuard-UAR/starter`, `dataset-archives`, `reports`, `models`, and `runs` after Drive access is authorized.
5. In Colab run the synthetic smoke test and all tests, then retrieve both official PhysioNet v1.0.0 training sets. Use official ZIP URLs if available; otherwise use the PhysioNet `aws s3 sync --no-sign-request` documented mirror. Validate counts before EDA.
6. Perform real EDA: missingness per feature, class prevalence by patient and time, record lengths. Generate and preserve aggregate reports.
7. Generate deterministic patient-level splits: A=training 70% / validation 15% / calibration 15%, B=unseen holdout. Run leakage tests. Keep B untouched for model selection and preprocessing fit.
8. Save reports and split manifest to private Drive. Document actual results; never substitute synthetic numbers.
9. Next begin Week 2: train-only imputation/scaling, causal 12-hour windows, feature selection, a reproducible PyTorch Dataset, and unit tests. Prefer CPU for preprocessing; activate a Colab Pro GPU only for neural-network training and save frequent model checkpoints to Drive.

## Colab Pro execution policy

- On each runtime, run `python scripts/compute_check.py --output runs/compute.json` and record the observed GPU, available memory, Python/PyTorch versions.
- Process large datasets on `/content` local runtime storage where practical; avoid training directly against mounted Drive files because repeated Drive I/O is slow. Copy checkpoint/report artifacts to Drive at milestones.
- If a GPU is available for weeks 3+, prefer batched PyTorch training, pinned-memory DataLoaders where beneficial, `torch.autocast(device_type='cuda')`/mixed precision only when tested, and checkpoint/resume support. Do not assume mixed precision improves all tasks; benchmark and retain FP32 fallback.
- Preserve seeds, dataset split IDs, configs, commit SHA, feature schema, model metrics, and calibration details.
- Colab compute availability and runtime length are dynamic and not guaranteed even with Pro.
- GPU use should maximize **research productivity**, not merely GPU hours. Week 1 is CPU-oriented.

## Acceptance criteria for Week 1

- Real patient counts A=20,336 / B=20,000 observed in Colab.
- `pytest` succeeds in the Colab environment.
- EDA report and figures saved to Drive.
- `split_manifest.csv` saved privately to Drive; source B exclusively `test_B`.
- No patient or future-hour leakage.
- GitHub repository contains only shareable source, configs, documentation, tests, and aggregate reports.

## Research scope

After Week 1, compare logistic regression, MLP, LSTM, and TCN. Calibrate the best predictor on an independent calibration subset. Main novel research question: whether uncertainty-conditioned retrieval, evidence verification, and selective abstention reduce unsupported explanations relative to budget-matched standard RAG. Implement LangChain retrieval and LangGraph routing in a later stage. Do not claim medical validity or publication readiness before experimentation and independent review.
