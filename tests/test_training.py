"""Padding, input isolation, real optimization and exact epoch-resume checks."""
from dataclasses import replace
import json
from pathlib import Path
import shutil
import uuid

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.run_pipeline import run_pipeline
from scripts.run_training import train_prepared
from sepsisguard.dataset import CausalWindowDataset
from sepsisguard.models import build_model
from sepsisguard.preprocessing import TrainOnlyPreprocessor, record_path
from sepsisguard.training import TrainingConfig, TrainingPaused, load_epoch, metrics, train_one
from sepsisguard.training_data import cache_split


@pytest.fixture(scope='module')
def prepared(dataset, tmp_path_factory):
    torch.set_num_threads(2)
    work = tmp_path_factory.mktemp('training')
    data = work / 'synthetic'
    shutil.copytree(dataset, data)
    project = Path(__file__).resolve().parents[1]
    run_id = 'pipeline-training-test-' + uuid.uuid4().hex
    result = run_pipeline(project, data, run_id=run_id, mode='synthetic', run_checks=False)
    source = project / 'runs' / run_id
    manifest = pd.read_csv(source / 'data/split_manifest.csv')
    pre = TrainOnlyPreprocessor.load(source / 'preprocessing/preprocessor.json')
    inventory = pd.read_csv(source / 'data/patient_file_hashes.csv')
    train = cache_split(data, manifest, pre, inventory, 'train', work / 'cache/train')
    val = cache_split(data, manifest, pre, inventory, 'validation', work / 'cache/validation')
    yield project, data, source, result, manifest, pre, inventory, train, val
    shutil.rmtree(source, ignore_errors=True)


def test_compact_windows_match_original_causal_windows(prepared):
    _, data, _, _, manifest, pre, _, train, val = prepared
    for name, cached in [('train', train), ('validation', val)]:
        reference = CausalWindowDataset(data, manifest, pre, name)
        indices = np.unique(np.concatenate([np.linspace(0,len(cached)-1,20,dtype=int), reference.ends - 1, np.r_[0,reference.ends[:-1]]]))
        for i in indices:
            actual, expected = cached[int(i)], reference[int(i)]
            length = actual['length']
            torch.testing.assert_close(actual['x'][:length], expected['x'][-length:], rtol=0, atol=0)
            assert torch.count_nonzero(actual['x'][length:]) == 0
            assert actual['y'] == expected['y'].item()


@pytest.mark.parametrize('architecture', ['logistic','lstm'])
def test_padding_cannot_affect_prediction(architecture):
    torch.manual_seed(42)
    model = build_model(architecture, 4, hidden=8).eval()
    x = torch.randn(3,12,4)
    lengths = torch.tensor([1,5,12])
    changed = x.clone()
    changed[0,1:] = 1000
    changed[1,5:] = -1000
    with torch.no_grad():
        torch.testing.assert_close(model(x,lengths), model(changed,lengths), rtol=0, atol=0)


@pytest.mark.parametrize('architecture', ['logistic','lstm'])
def test_optimizer_updates_and_epoch_resume_matches_uninterrupted(prepared, tmp_path, architecture):
    *_, train, val = prepared
    config = TrainingConfig(batch_size=256, hidden_size=8, baseline_epochs=3,lstm_epochs=3,patience=5,workers=0,precision='fp32')
    full = train_one(architecture,train,val,tmp_path/'full',None,config,torch.device('cpu'),'frozen-input')
    with pytest.raises(TrainingPaused):
        train_one(architecture,train,val,tmp_path/'restart',tmp_path/'drive',config,torch.device('cpu'),'frozen-input',stop_after_epoch=1)
    # Remove local optimizer/weight files: restore solely from private persisted weights.
    shutil.rmtree(tmp_path/'restart')
    resumed = train_one(architecture,train,val,tmp_path/'restart',tmp_path/'drive',config,torch.device('cpu'),'frozen-input')
    assert full['best_epoch'] == resumed['best_epoch']
    assert full['validation'] == resumed['validation']
    assert [h['weighted_train_loss'] for h in full['history']] == [h['weighted_train_loss'] for h in resumed['history']]
    state = torch.load(tmp_path/'full'/f'{architecture}_last.pt',weights_only=True)
    seed = torch.manual_seed(config.seed)
    initial = build_model(architecture,train.channels,8).state_dict()
    assert any(not torch.equal(v,initial[k]) for k,v in state['model'].items())
    pointer = json.loads((tmp_path/'drive'/f'{architecture}_resume.json').read_text())
    (tmp_path/'drive'/pointer['file']).write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='integrity mismatch'):
        load_epoch(tmp_path/'restart',tmp_path/'drive',architecture,state['signature'])


def test_no_calibration_or_B_inputs_and_end_to_end_result(prepared, tmp_path):
    project,data,source,preparation,manifest,pre,inventory,_,_ = prepared
    # Strong isolation check: neither split's raw records exist during training.
    for row in manifest.loc[manifest.split.isin(['calibration','test_B'])].itertuples():
        record_path(data,row.patient_id).unlink(missing_ok=True)
    with pytest.raises(ValueError,match='only'):
        cache_split(data,manifest,pre,inventory,'calibration',tmp_path/'forbidden')
    config = TrainingConfig(batch_size=256,hidden_size=8,baseline_epochs=1,lstm_epochs=1,workers=0,precision='fp32')
    result = train_prepared(project,data,source,preparation,tmp_path/'drive',config=config,device='cpu',demo=True,run_checks=False)
    try:
        assert result['status']=='training_complete' and result['mode']=='synthetic_demo'
        assert set(result['models'])=={'logistic','lstm'}
        assert result['calibration_raw_files_opened'] is False and result['test_B_raw_files_opened'] is False
        saved = Path(result['persistent_checkpoint'])
        assert json.loads((saved/'TRAINING_RESULT.json').read_text())['selected_model']==result['selected_model']
        assert (saved/'lstm_best.pt').is_file()
    finally:
        shutil.rmtree(project/'runs'/result['run_id'],ignore_errors=True)


def test_metric_requires_both_classes():
    with pytest.raises(ValueError,match='both classes'):
        metrics(np.zeros(4),np.full(4,0.1))
