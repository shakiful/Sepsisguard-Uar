# SepsisGuard-UAR — Working handoff

User preference: one runnable workflow, without a week-by-week schedule or manual notebook handoffs. Use `notebooks/SepsisGuard_UAR.ipynb` and the single runner `python -m scripts.run_pipeline` for ongoing work.

Google Colab Pro is the main compute environment, private Google Drive holds durable data/checkpoints, and https://github.com/shakiful/Sepsisguard-Uar is the canonical version-controlled source. The completed Colab output confirms an A100-SXM4-80GB allocation (79.25 GiB reported VRAM) and 167.05 GiB system RAM on 2026-11-04. Accept that runtime and record the actual GPU, VRAM and system RAM each run. The current data/preprocessing pipeline executes on CPU within either CPU-only or GPU-attached runtimes. Use GPU resources when later neural training benefits; first inspect the allocated runtime and expected cost, and checkpoint regularly. A subscription does not guarantee any GPU or runtime duration.

## Current implementation

The unified workflow downloads/restores official PhysioNet 2019 A/B data, runs EDA, preserves deterministic patient partitions, audits the complete cohort, runs regression and leakage tests, fits causal train-only preprocessing, verifies the development Dataset and saves one final result plus a private snapshot. It imports the existing completed data run `20261021T183143Z-82ed151a` when present. Otherwise it performs the same data pipeline from scratch. Matching pipeline checkpoints resume automatically, with all raw files re-audited before fitted state can be reused.

The earlier data run was verified from the user's uploaded executed notebook: 40,336 patients, 1,552,210 hours, all 25 original tests passing, and completed Drive copy calls. Local preprocessing validation retained 39 features and produced 12 x 78 development windows. The unified run `pipeline-20261104T190221Z-22906fa2` is now verified from supplied Colab output: all 30 tests passed, the exact earlier manifest was imported, preprocessing passed on A/train only, and the persistent checkpoint readback passed. See `reports/verification/UNIFIED_COLAB_EVIDENCE.json`.

## Continue the work

1. Read `PROJECT_STATUS.md` and the aggregate verification evidence. Preserve the original patient partition and dataset fingerprints.
2. Data preparation is complete. Preserve private run `pipeline-20261104T190221Z-22906fa2` and its fitted state/manifest. Continue model implementation in the same notebook/runner; do not ask for a redundant preparation rerun. Resume the same Drive checkpoint when computation is needed.
3. Record actual results and blockers. Browser control previously returned `Cannot attach to this target`; do not claim cloud execution from source creation or local tests. If direct operation remains blocked, review the executed notebook supplied by the user.
4. Preserve all protected outputs privately. The pipeline writes immutable snapshots under `MyDrive/SepsisGuard-UAR/runs/<run-id>` and read-verifies files before atomically replacing the checkpoint pointer. Never commit raw records, split manifests, patient summaries, per-file hashes or fitted artifacts.
5. Extend the same workflow with the planned predictive baselines (logistic regression, MLP, LSTM and TCN), independent calibration and retrieval experiments. These components are not implemented or trained yet. Do not represent successful data preparation as a completed predictive system.

## Research invariants

A/train alone determines preprocessing and model fitting; A/validation supports model selection; A/calibration is independent; B remains locked until frozen final evaluation. Original PhysioNet challenge labels are already shifted and must not be shifted again. Causal filling and window creation reset at patient boundaries. Fixed preprocessing rules must not be selected using B's descriptive EDA. Future supervised feature selection belongs inside patient-level training folds.

Main research question: whether uncertainty-conditioned retrieval, evidence verification and selective abstention reduce unsupported explanations relative to budget-matched standard RAG. Compare controlled baselines before making novelty or publication-readiness claims. This retrospective public-data study is not a medical device and must not produce individual treatment instructions or send identifiable data to external AI services.
