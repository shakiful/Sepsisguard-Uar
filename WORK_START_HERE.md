# SepsisGuard-UAR — ChatGPT Work handoff

**Project type:** Solo research project; course milestone followed by publication-quality research extension.
**Compute preference:** Google Colab Pro is the **main compute environment**. Windows RTX 4060 and an Ubuntu server are fallbacks. Store durable outputs on Google Drive. Do not assume any GPU model is guaranteed, and don't waste Colab GPU compute units on the Week 1 CPU-only EDA.
**Safety and validity:** Public retrospective PhysioNet 2019 sepsis data, not a medical device, and not suitable for individual care decisions. Never provide clinical treatment instructions or upload identifiable data to external AI services.

## User intent

Take ownership of practical implementation in ChatGPT Work. Work through the implementation steps, interact with signed-in Google Colab/Drive and GitHub through available authorized UI/connectors, and leave artifacts plus concise progress summaries. Do not claim a Colab job or cloud execution succeeded unless observed. Never silently use paid cloud resources for lengthy runs without checking the session/runtime and expected costs.

## Current checkpoint (2026-10-21)

Canonical repository: https://github.com/shakiful/Sepsisguard-Uar (user-supplied public source repository). See PROJECT_STATUS.md for measured results and the remaining Week 2 cloud verification. Week 1 is verified from the user-supplied completed Colab notebook (run `20261021T183143Z-82ed151a`). Week 2 train-only preprocessing is implemented and locally validated; its Colab execution is next. The notebooks now clone this repository directly and record the actual source commit; no starter ZIP is required.

## Immediate Work sequence (do in order)

1. Review `PROJECT_STATUS.md` and `reports/verification/WEEK1_COLAB_EVIDENCE.json`. Week 1 is complete; preserve its exact run `20261021T183143Z-82ed151a` and private split manifest.
2. Use the existing user-identified repository `shakiful/Sepsisguard-Uar`. Preserve its history and unrelated edits. Source import is complete; push only source and aggregate results, never raw records or patient-level derivatives.
3. Check authorized connected Colab, Drive and GitHub access. If the browser cannot attach, keep progressing on source/review work and use executed notebook uploads as evidence. Do not claim direct cloud operation or independent Drive verification without observing it.
4. Open `notebooks/02_week2_colab.ipynb` in Colab with the same Google account/Drive. Use CPU. Its default restores the completed Week 1 run above, including the exact private manifest; a fresh runtime can restore or download the same official A/B files.
5. Validate the restored manifest and full-cohort fingerprint against the saved Week 1 audit. Run the regression tests before fitting. Preserve A=train 70% / validation 15% / calibration 15% and all B=test_B.
6. Execute train-only imputation/scaling, causal 12-hour windows, unsupervised training-only feature filtering and reproducible PyTorch Dataset checks. B must remain unopened for fit and development window checks.
7. Save fitted state, reports and provenance privately to Drive. The final cell reads back key saved files before reporting success. Review the actual output before marking Week 2 cloud execution complete.
8. Later compare the planned predictive baselines on A only; activate a Colab Pro GPU when neural-network training benefits, with frequent Drive checkpoints. Calibrate independently and unlock B only for frozen final evaluation.

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
