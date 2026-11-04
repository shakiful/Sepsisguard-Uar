"""Continue the same Colab workflow from a frozen preparation checkpoint."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pandas as pd
import torch

from sepsisguard.download import run_download
from sepsisguard.preprocessing import TrainOnlyPreprocessor, manifest_digest
from sepsisguard.training import TrainingConfig, TrainingPaused, atomic_json, train_one
from sepsisguard.training_data import cache_split, file_hash
from scripts.compute_check import inspect_compute
from scripts.run_pipeline import command_log, run_pipeline, safe_id, snapshot_root, source_fingerprint, verify_checkpoint

PREPARED_RUN = 'pipeline-20261104T190221Z-22906fa2'


def prepared_checkpoint(project, storage, run_id, *, demo=False):
    safe_id(run_id)
    roots = ([Path(storage) / 'runs' / run_id] if storage else []) + [Path(project) / 'runs' / run_id]
    for root in roots:
        marker = root / 'checkpoint.json'
        if not marker.is_file():
            continue
        state = json.loads(marker.read_text())
        verify_checkpoint(root, state)
        source = snapshot_root(root, state)
        result = json.loads((source / 'PIPELINE_RESULT.json').read_text())
        expected = 'synthetic_pipeline_passed' if demo else 'data_pipeline_passed'
        if state['stage'] != 'complete' or result['status'] != expected or result['run_id'] != run_id:
            raise ValueError('A completed preparation checkpoint with matching real/demo mode is required')
        if not demo and (result['tests'] != 'passed' or result['audit']['patient_counts'] != {'A': 20336, 'B': 20000}):
            raise ValueError('Preparation lacks the required real-cohort validation')
        return source, result
    raise FileNotFoundError(f'Prepared run {run_id} is not available in this Drive/runtime')


def train_prepared(project, data_root, source, preparation, storage=None, *, config=None, device='cuda', demo=False, run_checks=True):
    project, data_root, source = Path(project), Path(data_root), Path(source)
    storage = Path(storage) if storage else None
    config = config or TrainingConfig()
    config.validate()
    device = torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('Select a GPU runtime for training, or explicitly choose --device cpu')
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Training supports CPU or CUDA')
    if not demo and not run_checks:
        raise ValueError('Real training requires the regression/leakage suite')
    # Bound CPU worker threading; an A100 workload is fed through pinned batches.
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    manifest = pd.read_csv(source / 'data/split_manifest.csv')
    pre = TrainOnlyPreprocessor.load(source / 'preprocessing/preprocessor.json')
    if manifest_digest(manifest) != preparation['input']['manifest_sha256'] or pre.provenance['manifest_sha256'] != manifest_digest(manifest):
        raise ValueError('Prepared partition and fitted state disagree')
    if pre.provenance['fit_source'] != 'A' or pre.provenance['fit_split'] != 'train':
        raise ValueError('Expected frozen A/train-only preprocessing')
    if (data_root / 'SYNTHETIC_DATA.txt').exists() != demo:
        raise ValueError('Real/demo data marker mismatch')
    inventory = pd.read_csv(source / 'data/patient_file_hashes.csv')
    identity = {'prepared_run': preparation['run_id'], 'cohort': preparation['audit']['cohort_sha256'],
                'manifest': manifest_digest(manifest), 'preprocessor_sha256': file_hash(source / 'preprocessing/preprocessor.json'),
                'training_source_fingerprint': source_fingerprint(project), 'config': asdict(config),
                'device': device.type, 'torch_version': str(torch.__version__), 'demo': demo}
    input_signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    run_id = 'training-' + input_signature[:20]
    output = project / 'runs' / run_id
    output.mkdir(parents=True, exist_ok=True)
    persistent = storage / 'training' / run_id if storage else None
    if persistent is not None:
        persistent.mkdir(parents=True, exist_ok=True)
    atomic_json(output / 'TRAINING_INPUT.json', identity)
    atomic_json(output / 'compute.json', inspect_compute())
    command_log([sys.executable, '-m', 'pip', 'freeze'], output / 'environment.log', project, echo=False)
    if run_checks:
        command_log([sys.executable, '-m', 'pytest', '-q'], output / 'tests.log', project)
    cache = project / 'data/processed/training' / input_signature[:20]
    print('Building/verifying A/train and A/validation caches; calibration and B raw files are not read', flush=True)
    train = cache_split(data_root, manifest, pre, inventory, 'train', cache / 'train')
    val = cache_split(data_root, manifest, pre, inventory, 'validation', cache / 'validation')
    print(f'Training {len(train):,} endpoints; validation {len(val):,}; {train.channels} channels on {device}', flush=True)
    print(f'Budget: {config.max_minutes:g} minutes per invocation; epoch-boundary Drive resume is enabled', flush=True)
    models = {}
    deadline = time.monotonic() + config.max_minutes * 60
    status = 'training_complete'
    try:
        for name in ('logistic', 'lstm'):
            models[name] = train_one(name, train, val, output, persistent, config, device, input_signature, deadline=deadline)
    except TrainingPaused as error:
        status = 'training_paused'
        print(str(error), flush=True)
    result = {'status': status, 'mode': 'synthetic_demo' if demo else 'real',
              'run_id': run_id, 'prepared_run_id': preparation['run_id'], 'input': identity,
              'models': models, 'selected_model': max(models, key=lambda n: models[n]['validation']['average_precision']) if status == 'training_complete' else None,
              'selection_metric': 'A/validation hourly average precision (not trapezoidal PR area)',
              'model_fit_split': 'A/train', 'model_selection_split': 'A/validation',
              'calibration_raw_files_opened': False, 'test_B_raw_files_opened': False,
              'independent_holdout_evaluated': False, 'probability_calibration_fitted': False,
              'evaluation_scope': 'Validation used for epoch and architecture selection; these metrics are not independent test performance or official challenge utility',
              'persistent_checkpoint': str(persistent) if persistent else None,
              'source_commit': subprocess.check_output(['git','rev-parse','HEAD'], cwd=project, text=True).strip(),
              'compute': inspect_compute(), 'completed_at': datetime.now(timezone.utc).isoformat()}
    atomic_json(output / 'TRAINING_RESULT.json', result)
    report = '# SepsisGuard predictive training\n\n'
    if demo:
        report += '**SYNTHETIC TEST ONLY — not research performance.**\n\n'
    report += f'Status: {status}\n\nPrepared run: {preparation["run_id"]}\n\n'
    for name, model in models.items():
        report += f'- {name}: best epoch {model["best_epoch"]}; validation AP {model["validation"]["average_precision"]:.6f}; AUROC {model["validation"]["roc_auc"]:.6f}.\n'
    report += '\nModel selection uses A/validation. Calibration and B are unopened by training. Scores are uncalibrated; no independent test performance or clinical validity is established.\n'
    (output / 'TRAINING_REPORT.md').write_text(report)
    if persistent is not None:
        for name in ('TRAINING_INPUT.json', 'TRAINING_RESULT.json', 'TRAINING_REPORT.md', 'compute.json', 'environment.log', 'tests.log'):
            path = output / name
            if not path.is_file():
                continue
            dest = persistent / name
            partial = dest.with_suffix(dest.suffix + '.partial')
            shutil.copyfile(path, partial)
            if file_hash(partial) != file_hash(path):
                raise ValueError('Training metadata readback failed')
            partial.replace(dest)
    print(json.dumps(result, indent=2), flush=True)
    print('Training result and model artifact readback verified:', persistent or output, flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', default='data/raw')
    p.add_argument('--storage-root')
    p.add_argument('--prepared-run-id', default=PREPARED_RUN)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--batch-size', type=int, default=1024)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--baseline-epochs', type=int, default=8)
    p.add_argument('--lstm-epochs', type=int, default=12)
    p.add_argument('--patience', type=int, default=4)
    p.add_argument('--max-minutes', type=float, default=90)
    p.add_argument('--precision', choices=['auto', 'fp32'], default='auto')
    p.add_argument('--demo', action='store_true')
    a = p.parse_args()
    project, data = Path.cwd(), Path(a.data_root)
    storage = Path(a.storage_root) if a.storage_root else None
    try:
        source, preparation = prepared_checkpoint(project, storage, a.prepared_run_id, demo=a.demo)
    except FileNotFoundError:
        print('No completed preparation is available; executing preparation once before training', flush=True)
        preparation = run_pipeline(project, data, storage, mode='synthetic' if a.demo else 'real')
        source, preparation = prepared_checkpoint(project, storage, preparation['run_id'], demo=a.demo)
    if not a.demo:
        run_download(data, storage / 'dataset-archives' if storage else project / 'data/downloads')
    config = TrainingConfig(batch_size=a.batch_size, workers=a.workers, baseline_epochs=a.baseline_epochs,
                            lstm_epochs=a.lstm_epochs, patience=a.patience, max_minutes=a.max_minutes, precision=a.precision)
    result = train_prepared(project, data, source, preparation, storage, config=config, device=a.device, demo=a.demo)
    if result['status'] != 'training_complete':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
