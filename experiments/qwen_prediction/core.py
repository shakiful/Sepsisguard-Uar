"""Past-only inputs, design weights and deterministic scoring; no predictor imports."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

SYSTEM_PROMPT = '''Predict the ORIGINAL PhysioNet 2019 hourly SepsisLabel from the supplied numeric history. Output exactly 0 or 1. Label 1 begins six hours before the benchmark-defined sepsis onset and remains 1 afterward; otherwise label 0. Twelve hours is the history window, not the forecast horizon. Each row is feature|latest known raw value|age of that value in hours|last minus first actually observed value in the trailing 12 hours|number of actual observations in that window. NA means unavailable. Use only this past/current history. No identifiers, future measurements, other-model predictions or outcome labels are supplied. This is retrospective label prediction, not treatment advice.'''


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    tmp.replace(path)


def immutable_json(path, payload):
    path = Path(path)
    if path.exists():
        prior = json.loads(path.read_text())
        if prior != payload: raise ValueError('Refusing to replace an immutable experiment artifact')
    else: save_json(path, payload)
    return payload


def source_digest(project):
    project = Path(project); h = hashlib.sha256()
    paths = sorted((project/'experiments/qwen_prediction').rglob('*.py'))
    paths += [project/'experiments/qwen_prediction/protocol.json', project/'experiments/qwen_prediction/requirements.txt']
    # Also bind the reused, read-only data-loader/checkpoint infrastructure.
    paths += sorted((project/'src/sepsisguard').rglob('*.py')) + [project/'scripts/run_training.py', project/'scripts/run_pipeline.py']
    for path in paths: h.update(path.relative_to(project).as_posix().encode()+b'\0'+path.read_bytes())
    return h.hexdigest()


def past_only_text(frame, endpoint, features, window=12):
    from sepsisguard.schema import FEATURES
    if not isinstance(endpoint, (int, np.integer)) or not 0 <= endpoint < len(frame): raise ValueError('Invalid endpoint')
    if not features or len(set(features)) != len(features) or not set(features) <= set(FEATURES): raise ValueError('Only official predictor columns may enter the prompt')
    prefix = frame.iloc[:int(endpoint)+1]
    def number(x): return 'NA' if x is None else format(float(x), '.4g')
    lines = ['feature|last|age_h|delta12|n12']
    for feature in features:
        values = prefix[feature].to_numpy(dtype=float)
        known = np.flatnonzero(np.isfinite(values)); start = max(0, len(values)-window)
        recent = values[start:]; observed = recent[np.isfinite(recent)]
        last = float(values[known[-1]]) if len(known) else None
        age = int(endpoint-known[-1]) if len(known) else None
        change = float(observed[-1]-observed[0]) if len(observed)>1 else None
        lines.append('|'.join([feature, number(last), number(age), number(change), str(len(observed))]))
    return '\n'.join(lines)


def choose_endpoints(labels, sizes, seed):
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.isin(labels, [0,1]).all(): raise ValueError('Original binary endpoint labels required')
    rng = np.random.default_rng(seed); selected = []; weights = []
    for label in (0,1):
        pool = np.flatnonzero(labels == label); count = int(sizes[label])
        if count <= 0 or count > len(pool): raise ValueError('Not enough endpoints for the declared sample; do not silently change the protocol')
        indices = rng.choice(pool, count, replace=False)
        selected.extend(indices.tolist()); weights.extend([len(pool)/count]*count)
    order = np.argsort(selected)
    return np.asarray(selected)[order], np.asarray(weights, dtype=float)[order]


def require_test_gate(path, experiment_id):
    if path is None: raise ValueError('B input construction requires completed fine-tuning and a frozen evaluation gate')
    gate = json.loads(Path(path).read_text())
    payload = gate['protocol']
    if gate['sha256'] != digest(payload) or payload.get('experiment_id') != experiment_id or payload.get('training_complete') is not True:
        raise ValueError('Frozen gate is missing, changed or belongs to another experiment')
    return payload


def build_sample(data_root, manifest, inventory, features, split, sizes, seed, *, gate=None, experiment_id=None):
    if split == 'test_B': require_test_gate(gate, experiment_id)
    elif split not in ('train','validation','calibration'): raise ValueError('Unknown split')
    from sepsisguard.load import read_patient
    from sepsisguard.preprocessing import record_path
    rows = manifest.loc[manifest.split.eq(split)].sort_values('patient_id').reset_index(drop=True)
    source = 'B' if split == 'test_B' else 'A'
    if rows.empty or not rows.source.eq(source).all(): raise ValueError('Unexpected patient source')
    hashes = inventory.set_index('patient_id', verify_integrity=True).sha256
    labels = []; lengths = []; content = hashlib.sha256()
    for j, row in enumerate(rows.itertuples()):
        path = record_path(data_root, row.patient_id); actual = sha_file(path)
        if actual != hashes.loc[row.patient_id]: raise ValueError('Raw patient contents differ from the verified preparation')
        df = read_patient(path, source)
        if len(df) != row.n_hours: raise ValueError('Patient duration changed')
        labels.append(df.SepsisLabel.to_numpy(dtype=np.int8)); lengths.append(len(df))
        content.update(row.patient_id.encode()+bytes.fromhex(actual))
        if (j+1)%4000 == 0: print(f'Indexed {split}: {j+1:,}/{len(rows):,} patients', flush=True)
    labels = np.concatenate(labels); lengths = np.asarray(lengths); ends = np.cumsum(lengths)
    indices, weights = choose_endpoints(labels, sizes, seed); records = []; previous = None; df = None
    for index, weight in zip(indices, weights):
        j = int(np.searchsorted(ends, index, side='right')); row = rows.iloc[j]
        if previous != j: df = read_patient(record_path(data_root, row.patient_id), source); previous = j
        hour = int(index-(ends[j-1] if j else 0))
        records.append({'global_index':int(index), 'private_patient_id':row.patient_id, 'endpoint':hour,
                        'label':int(labels[index]), 'weight':float(weight), 'text':past_only_text(df, hour, features)})
    return {'split':split, 'population_hours':int(len(labels)), 'population_positive_hours':int(labels.sum()),
            'population_patients':int(len(rows)), 'sample_hours':len(records), 'content_sha256':content.hexdigest(), 'records':records}


def calibrate(scores, labels, weights):
    scores = np.asarray(scores, dtype=float); labels = np.asarray(labels); weights = np.asarray(weights)
    if len(np.unique(labels)) != 2 or not np.isfinite(scores).all(): raise ValueError('Both calibration classes and finite scores required')
    fit = IsotonicRegression(y_min=0., y_max=1., out_of_bounds='clip').fit(scores, labels, sample_weight=weights)
    return {'method':'weighted_isotonic_A_calibration_only','x':fit.X_thresholds_.tolist(),'y':fit.y_thresholds_.tolist()}


def apply_calibration(calibrator, scores):
    return np.interp(np.asarray(scores), calibrator['x'], calibrator['y'])


def recall_threshold(y, p, w, target=.8):
    y = np.asarray(y); p = np.asarray(p); w = np.asarray(w)
    order = np.argsort(-p[y == 1], kind='stable'); positive = p[y == 1][order]; mass = w[y == 1][order]
    if not len(positive): raise ValueError('Validation positives required')
    return float(positive[np.searchsorted(np.cumsum(mass), target*mass.sum(), side='left')])


def metrics(y, p, w, threshold=.5):
    y = np.asarray(y); p = np.asarray(p, dtype=float); w = np.asarray(w, dtype=float)
    if not np.isin(y,[0,1]).all() or p.shape != y.shape or w.shape != y.shape or np.any(w<0) or not np.isfinite(p).all() or np.any((p<0)|(p>1)) or w.sum() <= 0: raise ValueError('Invalid aligned labels, probabilities or weights')
    both = len(np.unique(y[w > 0])) == 2
    alert = p >= threshold; tp = float(np.sum(w*alert*y)); fp = float(np.sum(w*alert*(1-y))); positive = float(w@y)
    return {'roc_auc':float(roc_auc_score(y,p,sample_weight=w)) if both else None,
            'average_precision':float(average_precision_score(y,p,sample_weight=w)) if both else None,
            'brier':float(brier_score_loss(y,p,sample_weight=w)),
            'log_loss':float(log_loss(y,np.clip(p,1e-12,1-1e-12),sample_weight=w,labels=[0,1])),
            'sensitivity':tp/positive if positive else None,'precision':tp/(tp+fp) if tp+fp else None,
            'threshold':float(threshold),'weighted_positive_fraction':positive/float(w.sum()),
            'sample_hours':len(y),'weighted_population_hours':float(w.sum())}


def paired_bootstrap(y, predictions, weights, patient_ids, thresholds, repetitions=500, seed=42):
    _, group = np.unique(patient_ids, return_inverse=True); n = int(group.max()+1); rng = np.random.default_rng(seed)
    draws = {name:[] for name in predictions}; contrasts = {name:[] for name in predictions if name != 'qwen_lora'}
    for _ in range(repetitions):
        counts = np.bincount(rng.integers(0,n,n), minlength=n); w = weights*counts[group]
        current = {}
        for name, p in predictions.items():
            current[name] = metrics(y,p,w,thresholds[name]); draws[name].append(current[name])
        for name in contrasts:
            a = current['qwen_lora']['average_precision']; b = current[name]['average_precision']
            if a is not None and b is not None: contrasts[name].append(a-b)
    def interval(values):
        values = [v for v in values if v is not None and np.isfinite(v)]
        return {'lower':float(np.quantile(values,.025)), 'upper':float(np.quantile(values,.975)), 'valid_draws':len(values)} if values else None
    intervals = {name:{key:interval([d[key] for d in rows]) for key in ('roc_auc','average_precision','brier','log_loss','sensitivity','precision')} for name,rows in draws.items()}
    return {'intervals_95':intervals,'paired_AP_difference_qwen_lora_minus':{name:interval(v) for name,v in contrasts.items()},
            'unit':'whole sampled patient records; fixed inclusion weights','scope':'Conditional pilot bootstrap; does not reproduce the full stratified sampling design or training uncertainty'}
