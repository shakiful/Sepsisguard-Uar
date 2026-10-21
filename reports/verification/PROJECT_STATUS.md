# SepsisGuard-UAR progress — 2026-10-21

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

## Continue from this checkpoint

1. Week 1 is complete; do not rerun it merely to recover its run ID.
2. Open notebooks/02_week2_colab.ipynb in CPU Colab using the same Google account/Drive. Its default WEEK1_RUN_ID is `20261021T183143Z-82ed151a`. It restores the exact manifest, checks the cohort fingerprint, then fits preprocessing using A/train only.
3. Preserve the Week 2 fitted state, provenance, reports and run logs privately in Drive. The final cell reads back the saved key files before reporting success.
4. Review the executed Week 2 result before marking cloud preprocessing complete. Browser control remains blocked; the user may run the prepared notebook and supply its executed outputs.
5. Later train and compare the planned baselines on A only, calibrate independently, and unlock B only for frozen final evaluation.

## Deliverables

The source ZIP contains shareable source, two Colab notebooks, tests, aggregate verification reports, and a Git history bundle. A separate private checkpoint ZIP contains protected patient summaries, split manifest, per-file hashes, fitted preprocessing state, notebook execution outputs, environment snapshots, and logs. It excludes raw patient files. Original data can be downloaded again from the documented official S3 mirror.

No model training or clinical validation has been performed. No Colab GPU compute was requested.
