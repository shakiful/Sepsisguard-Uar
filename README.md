# SepsisGuard-UAR

Run the implemented research data pipeline in one Colab session: download/restore → EDA → patient partitions → data and leakage checks → causal preprocessing → PyTorch Dataset checks → verified private Drive checkpoint.

[![Run in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shakiful/Sepsisguard-Uar/blob/main/notebooks/SepsisGuard_UAR.ipynb)

Select your available **A100 / high-RAM runtime** (or CPU-only), then **Runtime → Run all**. Authorize your private Drive mount when prompted. There is no calendar schedule or notebook handoff. The notebook updates a clean dedicated source checkout and reuses earlier downloaded records when available. GPU-attached runtimes are accepted. Every run records the actual GPU model, VRAM and system RAM; current data processing executes on CPU. The GPU becomes useful when predictive model training is implemented.

The runner automatically imports the completed data milestone `20261021T183143Z-82ed151a` if it is present in the same Drive. Otherwise it builds the data reports and deterministic partitions itself. Subsequent runs resume the latest checkpoint matching the current source and configuration. Every run re-audits raw files and runs the regression/leakage tests; previously validated EDA and preprocessing are reused only when their hashes and inputs match. An incomplete or changed checkpoint stops with an error instead of silently changing the experiment.

All private outputs are under `MyDrive/SepsisGuard-UAR/runs/<run-id>`. The checkpoint marker identifies an immutable snapshot containing `PIPELINE_RESULT.json`, `PIPELINE_REPORT.md`, EDA, the exact patient manifest, fitted preprocessing, source/runtime records and logs. Each snapshot is read back before its completion pointer is replaced, so an interrupted save preserves the previous checkpoint. Cached data archives live in `dataset-archives`; working raw data stays on `/content`. After a Colab reset, open the same notebook and select Run all again.

This implementation prepares and verifies data. Predictive model training, calibration and uncertainty-conditioned retrieval experiments are not implemented yet. Their eventual execution belongs in this same workflow. No predictive accuracy or clinical validation is claimed.

## Measured results

The unified Colab pipeline completed on 2026-11-04: **30 tests passed**, preprocessing and Dataset checks passed, and the private checkpoint passed readback. The recorded allocation was **A100-SXM4-80GB / 167.05 GiB system RAM**. Run ID: `pipeline-20261104T190221Z-22906fa2`. See [execution evidence](reports/verification/UNIFIED_COLAB_EVIDENCE.json). GPU model training has not occurred.

The completed data notebook was verified on 2026-10-21: 40,336 patients, 1,552,210 hours, 2,932 patients with positive labels and 27,916 positive hours. All 25 existing tests passed before and after the data audit. The cohort fingerprint matches independent local validation.

| Source | Patients | Hours | Patients with positive labels | Positive hours |
|---|---:|---:|---:|---:|
| A | 20,336 | 790,215 | 1,790 | 17,136 |
| B | 20,000 | 761,995 | 1,142 | 10,780 |
| Total | 40,336 | 1,552,210 | 2,932 | 27,916 |

A is partitioned into 14,235 train / 3,050 validation / 3,051 calibration patients (seed 42). All 20,000 B patients are reserved for frozen final evaluation. Descriptive B EDA must not guide model, feature or threshold selection. Audit checks cover the schema, binary labels, contiguous hours, patient IDs, manifest correspondence and exact-file duplicates; they cannot establish undocumented subject linkage.

## Local command

```bash
python -m pip install -e '.[dev,download,torch]'
python -m scripts.run_pipeline --data-root data/raw --storage-root /path/to/private/persistent/storage
```

Install PyTorch from its official CPU wheel index first when a local CPU install is desired. Colab normally includes it. A local storage path is not a Google Drive claim; actual runtime/storage locations are recorded.

For a synthetic functional check only:

```bash
python -m scripts.run_pipeline --demo --data-root data/synthetic-pipeline
```

Synthetic results are explicitly labelled and are not research estimates. The real-data pipeline always runs the full regression suite. `--run-id <new-name>` starts an isolated run; the default `auto` resumes the latest source/configuration match. A source change creates a new run, preserving older snapshots and the imported patient partition.

## Causal preprocessing

- Fit reads A/train patients only. Exact medians use a disk-backed raw matrix; streaming scaling uses hour-weighted population statistics after causal imputation.
- Forward fill uses only earlier observations within each patient; initial gaps use training medians. The fixed unsupervised training-only missing/constant filter retains 39 predictors on the measured cohort (EtCO2 is entirely unobserved in A/train).
- Windows contain 12 contiguous hours including the endpoint, with left zero padding, 39 observation-mask channels and a separate time mask. Original challenge labels are preserved without another shift.
- The Dataset loads patients lazily with a bounded cache, rejects a different manifest and locks B unless explicitly unlocked for final evaluation.
- Tests cover future-value mutation, held-out-file deletion before fitting, independent statistic calculations, patient boundaries, padding, deterministic loading and checkpoint resumption/integrity.

## Data, version control and evidence

Official source: https://physionet.org/content/challenge-2019/1.0.0/ (DOI 10.13026/v64v-d857). The downloader uses official archives when available and the official anonymous S3 mirror otherwise, caching successful transfers for runtime resets. Counts must equal A=20,336 / B=20,000 before research results are accepted.

The public repository contains source, tests and aggregate reports. Raw PSV records, patient summaries, patient manifests, per-file hashes, fitted artifacts and runtime snapshots remain private and excluded by `.gitignore`. Original dated notebooks remain as historical execution records; use the single notebook linked above for ongoing work.

See [PROJECT_STATUS.md](PROJECT_STATUS.md), [WORK_START_HERE.md](WORK_START_HERE.md) and [completed Colab evidence](reports/verification/WEEK1_COLAB_EVIDENCE.json). This is retrospective benchmark research, not a diagnostic or treatment system.
