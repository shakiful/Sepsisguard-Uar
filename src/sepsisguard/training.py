"""Train-only optimization, validation-only selection, epoch-boundary resume."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, log_loss, precision_score, recall_score, roc_auc_score
import torch
from torch.utils.data import DataLoader

from .dataset import seed_worker
from .models import build_model
from .training_data import file_hash


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    batch_size: int = 1024
    hidden_size: int = 64
    baseline_epochs: int = 8
    lstm_epochs: int = 12
    patience: int = 4
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    workers: int = 2
    precision: str = 'auto'
    max_minutes: float = 90.0

    def validate(self):
        if min(self.batch_size, self.hidden_size, self.baseline_epochs, self.lstm_epochs, self.patience) <= 0:
            raise ValueError('Positive training sizes, epochs and patience are required')
        if self.workers < 0 or self.learning_rate <= 0 or self.weight_decay < 0 or self.max_minutes <= 0:
            raise ValueError('Invalid training configuration')
        if self.precision not in ('auto', 'fp32'):
            raise ValueError('precision must be auto or fp32')


def seed_all(seed):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + '.partial')
    partial.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    partial.replace(path)


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def save_epoch(output, persistent, name, payload):
    output.mkdir(parents=True, exist_ok=True)
    path = output / f'{name}_last.pt'
    partial = path.with_suffix('.partial')
    torch.save(payload, partial)
    partial.replace(path)
    sha = file_hash(path)
    pointer = {'sha256': sha, 'file': f'weights/{sha}.pt', 'signature': payload['signature'], 'epoch': payload['epoch']}
    # Content-addressed immutable weights keep the previous epoch valid if copying fails.
    if persistent is not None:
        saved = persistent / pointer['file']
        saved.parent.mkdir(parents=True, exist_ok=True)
        temp = saved.with_suffix('.partial')
        shutil.copyfile(path, temp)
        if file_hash(temp) != sha:
            raise ValueError('Training checkpoint readback failed')
        temp.replace(saved)
        atomic_json(persistent / f'{name}_resume.json', pointer)
    atomic_json(output / f'{name}_resume.json', pointer)


def load_epoch(output, persistent, name, signature):
    if persistent is not None and (persistent / f'{name}_resume.json').is_file():
        pointer = json.loads((persistent / f'{name}_resume.json').read_text())
        rel = Path(pointer['file'])
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('Unsafe model checkpoint path')
        source = persistent / rel
    elif (output / f'{name}_resume.json').is_file():
        pointer = json.loads((output / f'{name}_resume.json').read_text())
        source = output / f'{name}_last.pt'
    else:
        return None
    if pointer['signature'] != signature or not source.is_file() or file_hash(source) != pointer['sha256']:
        raise ValueError('Model checkpoint configuration or integrity mismatch')
    return torch.load(source, map_location='cpu', weights_only=True)


def metrics(y, scores):
    y, scores = np.asarray(y, dtype=np.int64), np.asarray(scores, dtype=np.float64)
    if set(np.unique(y)) != {0, 1} or not np.isfinite(scores).all():
        raise ValueError('Evaluation requires finite scores and both classes')
    hard = scores >= 0.5
    return {'average_precision': float(average_precision_score(y, scores)),
            'roc_auc': float(roc_auc_score(y, scores)), 'brier': float(brier_score_loss(y, scores)),
            'log_loss': float(log_loss(y, scores, labels=[0, 1])),
            'precision_at_0_5': float(precision_score(y, hard, zero_division=0)),
            'recall_at_0_5': float(recall_score(y, hard, zero_division=0)),
            'f1_at_0_5': float(f1_score(y, hard, zero_division=0)),
            'positive_fraction': float(y.mean()), 'hours': len(y), 'positive_hours': int(y.sum())}


def autocast(device, precision):
    return torch.autocast(device_type='cuda', dtype=torch.bfloat16) if precision == 'bf16' else nullcontext()


def choose_precision(model, device, requested, channels):
    if device.type != 'cuda' or requested == 'fp32' or not torch.cuda.is_bf16_supported():
        return 'fp32', None
    # Test the actual recurrent operator before selecting AMP; no optimizer step.
    try:
        with autocast(device, 'bf16'):
            logits = model(torch.zeros(2, 12, channels, device=device), torch.tensor([12, 4]))
            logits.float().square().mean().backward()
        model.zero_grad(set_to_none=True)
        return 'bf16', None
    except RuntimeError as error:
        model.zero_grad(set_to_none=True)
        return 'fp32', str(error)[:400]


def loader(dataset, config, *, epoch, shuffle, cuda):
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle, num_workers=config.workers,
                      pin_memory=cuda, worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(config.seed + epoch), drop_last=False)


@torch.no_grad()
def evaluate(model, dataset, config, device, precision):
    model.eval()
    ys, scores = [], []
    for batch in loader(dataset, config, epoch=0, shuffle=False, cuda=device.type == 'cuda'):
        x = batch['x'].to(device, non_blocking=True)
        with autocast(device, precision):
            logits = model(x, batch['length'])
        scores.append(logits.float().sigmoid().cpu().numpy())
        ys.append(batch['y'].numpy())
    y, score = np.concatenate(ys), np.concatenate(scores)
    return metrics(y, score), y, score


class TrainingPaused(Exception):
    pass


def train_one(name, train, validation, output, persistent, config, device, input_signature, *, deadline=None, stop_after_epoch=None):
    config.validate()
    output = Path(output)
    persistent = Path(persistent) if persistent is not None else None
    output.mkdir(parents=True, exist_ok=True)
    seed_all(config.seed)
    positives = int(np.asarray(train.labels).sum())
    if not 0 < positives < len(train):
        raise ValueError('Training must contain both classes')
    model = build_model(name, train.channels, config.hidden_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    precision, fallback = choose_precision(model, device, config.precision, train.channels)
    # Runtime/precision are part of resume identity; no cross-platform exact-resume claim.
    identity = {'input': input_signature, 'model': name, 'config': asdict(config),
                'device': device.type, 'torch': str(torch.__version__), 'precision': precision}
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    saved = load_epoch(output, persistent, name, signature)
    epochs = config.baseline_epochs if name == 'logistic' else config.lstm_epochs
    if saved:
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['cpu_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        start, best, best_state, best_epoch, stale = saved['epoch'], saved['best_ap'], saved['best_model'], saved['best_epoch'], saved['stale']
        history = saved['history']
    else:
        start, best, best_state, best_epoch, stale, history = 0, -1.0, None, 0, 0, []
    weight = (len(train) - positives) / positives
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight, device=device))
    for epoch in range(start + 1, epochs + 1):
        if stale >= config.patience:
            break
        model.train()
        began = time.monotonic()
        total_loss = 0.0
        for batch in loader(train, config, epoch=epoch, shuffle=True, cuda=device.type == 'cuda'):
            if deadline is not None and time.monotonic() >= deadline:
                raise TrainingPaused('Time budget reached; resume from the last completed epoch')
            x = batch['x'].to(device, non_blocking=True)
            y = batch['y'].to(device, dtype=torch.float32, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device, precision):
                logits = model(x, batch['length'])
            loss = criterion(logits.float(), y)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss; retry with fp32 after investigation')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            total_loss += float(loss.detach()) * len(y)
        val, _, _ = evaluate(model, validation, config, device, precision)
        improved = val['average_precision'] > best + 1e-5
        if improved:
            best, best_state, best_epoch, stale = val['average_precision'], cpu_state(model), epoch, 0
        else:
            stale += 1
        record = {'epoch': epoch, 'weighted_train_loss': total_loss / len(train),
                  'validation': val, 'seconds': time.monotonic() - began}
        history.append(record)
        payload = {'signature': signature, 'epoch': epoch, 'model': cpu_state(model),
                   'optimizer': optimizer.state_dict(), 'best_model': best_state, 'best_ap': best,
                   'best_epoch': best_epoch, 'stale': stale, 'history': history,
                   'cpu_rng': torch.get_rng_state(),
                   'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []}
        save_epoch(output, persistent, name, payload)
        print(f'{name} epoch {epoch}/{epochs}: val AP={val["average_precision"]:.6f}, AUROC={val["roc_auc"]:.6f}, seconds={record["seconds"]:.1f}; checkpoint verified', flush=True)
        if stop_after_epoch is not None and epoch >= stop_after_epoch:
            raise TrainingPaused('Simulated interruption at an epoch boundary')
    if best_state is None:
        raise TrainingPaused('No completed training epoch; rerun to begin')
    model.load_state_dict(best_state)
    val, y, scores = evaluate(model, validation, config, device, precision)
    # Private endpoint-level arrays, never committed to the public source repo.
    np.savez_compressed(output / f'{name}_validation_predictions.npz', labels=y, scores=scores)
    weights_path = output / f'{name}_best.pt'
    torch.save({'architecture': name, 'channels': train.channels, 'hidden': config.hidden_size,
                'state_dict': best_state, 'input_signature': input_signature}, weights_path)
    if persistent is not None:
        persistent.mkdir(parents=True, exist_ok=True)
        for path in (weights_path, output / f'{name}_validation_predictions.npz'):
            destination = persistent / path.name
            temp = destination.with_suffix(destination.suffix + '.partial')
            shutil.copyfile(path, temp)
            if file_hash(temp) != file_hash(path):
                raise ValueError('Final model artifact readback failed')
            temp.replace(destination)
    return {'architecture': name, 'best_epoch': best_epoch, 'epochs_completed': len(history),
            'validation': val, 'history': history, 'positive_weight_from_train': weight,
            'precision': precision, 'amp_fallback_reason': fallback, 'device': device.type,
            'best_weights_sha256': file_hash(weights_path), 'checkpoint_signature': signature,
            'probability_calibration': 'not fitted; scores are uncalibrated',
            'threshold': 0.5, 'threshold_origin': 'fixed before training'}
