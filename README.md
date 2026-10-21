# Sepsisguard-Uar

Solo retrospective PhysioNet/CinC 2019 research. Week 1 ingestion, EDA, splits, and auditing plus Week 2 train-only preprocessing and PyTorch windows are implemented. No predictive model has been trained or clinically validated.

[![Week 1 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shakiful/Sepsisguard-Uar/blob/main/notebooks/01_week1_colab.ipynb)

See [PROJECT_STATUS.md](PROJECT_STATUS.md) for measured results, execution evidence, and remaining cloud access blockers. See [WORK_START_HERE.md](WORK_START_HERE.md) for the research handoff.

## Measured local results (2026-10-18)

| Source | Patients | Hours | Patients with positive labels | Positive hours |
|---|---:|---:|---:|---:|
| A | 20,336 | 790,215 | 1,790 | 17,136 |
| B | 20,000 | 761,995 | 1,142 | 10,780 |
| Total | 40,336 | 1,552,210 | 2,932 | 27,916 |

Full schema/time validation passed; zero exact-byte duplicate records. The split manifest matches all data files. Source B descriptions are for documentation only and must not influence model selection or preprocessing.

A is split into train 14,235 / validation 3,050 / calibration 3,051 patients (seed 42). All B patients are reserved as `test_B`.

## Install and verify

```bash
python -m pip install -e '.[dev,download,torch]'
python -m pytest -q
python scripts/compute_check.py --output runs/compute.json
```

For a CPU-only local PyTorch install, install PyTorch from its official CPU wheel index before installing the package extras. Colab normally includes PyTorch. GPU is unnecessary in Weeks 1 and 2.

## Colab Pro and Drive

1. Open `notebooks/01_week1_colab.ipynb` in Colab; select a CPU runtime.
2. Run the first cell to clone this repository and record its commit. Mount your private Drive when Colab requests access; no starter ZIP upload is needed.
3. Run cells in order. `/content` holds working data; Drive holds cached archives, reports, models, and runs.
4. Continue with `notebooks/02_week2_colab.ipynb`. After a runtime reset, specify the exact previous Week 1 run ID to restore its manifest.

If Drive mounting is unavailable, outputs must be explicitly staged, downloaded, and uploaded to private Drive using the browser. A staging path is not proof of a Drive save. The existing cloud notebook was imported successfully, but its data pipeline has not run successfully in Colab yet.

## Data and Week 1

Official dataset: https://physionet.org/content/challenge-2019/1.0.0/ (v1.0.0, DOI 10.13026/v64v-d857). The historical root ZIP endpoint returned 404 on 2026-10-18. The official anonymous S3 training prefix was downloaded and validated locally.

```bash
python -m sepsisguard.download --data-root data/raw --archives-dir data/downloads --transport s3
python -m sepsisguard.eda --data-root data/raw --output-dir reports/week1
python -m sepsisguard.splits --summary reports/week1/patient_summary.csv --output reports/week1/split_manifest.csv --seed 42
python -m sepsisguard.audit --data-root data/raw --output-dir reports/week1 --manifest reports/week1/split_manifest.csv
```

The downloader caches successful S3 downloads as ZIPs for reuse. Counts must match A=20,336 and B=20,000. The audit validates headers, values, labels, hourly continuity, manifest/data correspondence, and exact-byte duplication; it also saves private SHA256 file inventories. Hashes provide local integrity fingerprints, not independent certification of subject identity or official publisher checksums.

The original `SepsisLabel` is already shifted for the challenge's early-warning objective. Preserve it without another shift.

## Week 2

```bash
python -m scripts.run_week2 --data-root data/raw --manifest reports/week1/split_manifest.csv --output reports/week2
```

- Only A/train files are read when fitting. Raw training medians use a disk-backed matrix. Scaling uses streaming, hour-weighted population statistics after causal imputation.
- Each patient's forward fill uses only earlier observations and resets at patient boundaries; initial gaps use training medians.
- A fixed, unsupervised training-only filter drops unobserved or constant predictors. On the observed A/train split, only `EtCO2` is removed because it is entirely unobserved there. No B patterns informed this rule.
- There are 39 retained features and 39 observation-mask channels. Each window is 12 contiguous hours, including the current row, with left zero padding and a separate time mask. Original endpoint labels are unchanged.
- Irregular time gaps are rejected rather than silently interpreted as hours. Full previous history may provide forward-fill values within a trailing window.
- `CausalWindowDataset` loads patients lazily with a bounded cache. Source B requires an explicit final-evaluation unlock. A preprocessor cannot be paired with a different split manifest.
- Deterministic DataLoader helpers are supplied. Tests mutate future data, delete held-out files before refitting, check statistics independently, and verify padding and patient boundaries.

The current feature filter is unsupervised. Future supervised feature selection must be fitted inside patient-level A/train cross-validation folds. No trained model, accuracy metric, retrieval system, calibration model, or clinical evaluation is included yet.

## Version control and protected outputs

Local Git history is preserved. Canonical repository: https://github.com/shakiful/Sepsisguard-Uar. The user supplied this public repository on 2026-10-21. It contains source and aggregate results. Keep raw records, split manifests, fitted artifacts, and private checkpoints outside Git.

Only source and approved aggregate reports belong in Git. `.gitignore` excludes raw data, patient summaries and split manifests, per-patient hashes, fitted artifacts, runtime logs, and archives. Inspect staged paths before any push. The separate private checkpoint archive contains patient-level derivatives; keep it out of GitHub.

Week 2 notebook: [Open in Colab](https://colab.research.google.com/github/shakiful/Sepsisguard-Uar/blob/main/notebooks/02_week2_colab.ipynb).

## Citation and use

Reyna et al. (2019), PhysioNet dataset DOI 10.13026/v64v-d857; original challenge paper DOI 10.1097/CCM.0000000000004145. Consult the official page for licensing and citations. This is retrospective benchmark research, not a diagnostic or treatment system.
