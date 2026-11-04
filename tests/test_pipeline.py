"""Integration coverage for persisted resume, interruption and input integrity."""
import json
from pathlib import Path
import shutil
import uuid

import pytest

from scripts import run_pipeline as pipeline


@pytest.fixture
def pipeline_workspace(tmp_path):
    project = Path(__file__).resolve().parents[1]
    run_id = 'pipeline-test-' + uuid.uuid4().hex
    yield project, run_id, tmp_path / 'private-storage'
    shutil.rmtree(project / 'runs' / run_id, ignore_errors=True)


def test_resume_restores_private_checkpoint_and_rejects_changed_data(dataset, pipeline_workspace, monkeypatch, tmp_path):
    project, run_id, storage = pipeline_workspace
    data = tmp_path / 'synthetic'
    shutil.copytree(dataset, data)
    first = pipeline.run_pipeline(project, data, storage, run_id=run_id, mode='synthetic', run_checks=False)
    assert first['status'] == 'synthetic_pipeline_passed'
    assert first['preprocessing']['test_B_opened'] is False
    # Simulate loss of the entire runtime's working checkpoint.
    shutil.rmtree(project / 'runs' / run_id)
    monkeypatch.setattr(pipeline, 'run_eda', lambda *a, **k: pytest.fail('EDA should resume'))
    monkeypatch.setattr(pipeline, 'run_preprocessing', lambda *a, **k: pytest.fail('fit should resume'))
    second = pipeline.run_pipeline(project, data, storage, run_id=run_id, mode='synthetic', run_checks=False)
    assert second['preprocessing_reused'] is True
    assert first['input'] == second['input']
    record = sorted((data / 'training_setA').glob('*.psv'))[0]
    # Same numeric contents but different raw bytes must invalidate the cohort fingerprint.
    record.write_bytes(record.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='Cohort changed'):
        pipeline.run_pipeline(project, data, storage, run_id=run_id, mode='synthetic', run_checks=False)


def test_interrupted_fit_resumes_audited_data(dataset, pipeline_workspace, monkeypatch):
    project, run_id, storage = pipeline_workspace
    fit = pipeline.run_preprocessing
    def stop(*a, **k):
        raise RuntimeError('simulated runtime loss')
    monkeypatch.setattr(pipeline, 'run_preprocessing', stop)
    with pytest.raises(RuntimeError, match='simulated runtime loss'):
        pipeline.run_pipeline(project, dataset, storage, run_id=run_id, mode='synthetic', run_checks=False)
    marker = storage / 'runs' / run_id / 'checkpoint.json'
    assert json.loads(marker.read_text())['stage'] == 'data_ready'
    shutil.rmtree(project / 'runs' / run_id)
    monkeypatch.setattr(pipeline, 'run_preprocessing', fit)
    monkeypatch.setattr(pipeline, 'run_eda', lambda *a, **k: pytest.fail('audited EDA should resume'))
    result = pipeline.run_pipeline(project, dataset, storage, run_id=run_id, mode='synthetic', run_checks=False)
    assert result['preprocessing_reused'] is False
    assert json.loads(marker.read_text())['stage'] == 'complete'
    # Saved fitted-state corruption must be caught before it can be reused.
    shutil.rmtree(project / 'runs' / run_id)
    state = json.loads(marker.read_text())
    (storage / 'runs' / run_id / state['snapshot'] / 'preprocessing/preprocessor.json').write_text('{}')
    with pytest.raises(ValueError, match='Checkpoint integrity failure'):
        pipeline.run_pipeline(project, dataset, storage, run_id=run_id, mode='synthetic', run_checks=False)


def test_interrupted_snapshot_save_preserves_previous_generation(tmp_path, monkeypatch):
    run_dir = tmp_path / 'working'
    (run_dir / 'data').mkdir(parents=True)
    artifact = run_dir / 'data/EDA_REPORT.md'
    artifact.write_text('previous complete report')
    storage = tmp_path / 'persistent'
    state = {'format': pipeline.FORMAT, 'run_id': 'pipeline-save-test', 'config': {}, 'created_at': 'example'}
    pipeline.checkpoint(run_dir, storage, state, 'data_ready')
    root = storage / 'runs/pipeline-save-test'
    previous = json.loads((root / 'checkpoint.json').read_text())
    artifact.write_text('new report')
    def interrupted(*a, **k):
        raise OSError('simulated disconnect during copy')
    monkeypatch.setattr(pipeline.shutil, 'copyfile', interrupted)
    with pytest.raises(OSError, match='simulated disconnect'):
        pipeline.checkpoint(run_dir, storage, state, 'data_ready')
    assert json.loads((root / 'checkpoint.json').read_text()) == previous
    pipeline.verify_checkpoint(root, previous)
    assert (pipeline.snapshot_root(root, previous) / 'data/EDA_REPORT.md').read_text() == 'previous complete report'


def test_import_previous_milestone_preserves_manifest_and_checks_evidence(tmp_path):
    storage = tmp_path / 'private'
    old = storage / 'reports/previous/week1'
    old.mkdir(parents=True)
    audit = {'cohort_sha256': 'example'}
    (old / 'DATA_AUDIT.json').write_text(json.dumps(audit))
    manifest = b'patient_id,source,split,n_hours,ever_sepsis\nA:p1,A,train,2,0\n'
    (old / 'split_manifest.csv').write_bytes(manifest)
    (old / 'patient_summary.csv').write_text('example summary')
    (old / 'EDA_REPORT.md').write_text('# Week 1 Exploratory Data Analysis')
    result = storage / 'runs/previous/WEEK1_RESULT.json'
    result.parent.mkdir(parents=True)
    result.write_text(json.dumps({'run_id': 'previous', 'status': 'week1_data_pipeline_passed', 'audit': audit}))
    target = tmp_path / 'restored'
    imported = pipeline.import_legacy(storage, 'previous', target)
    assert (target / 'split_manifest.csv').read_bytes() == manifest
    assert imported['manifest_file_sha256'] == pipeline.digest(target / 'split_manifest.csv')
    result.write_text(json.dumps({'run_id': 'previous', 'status': 'week1_data_pipeline_passed', 'audit': {}}))
    with pytest.raises(ValueError, match='disagree'):
        pipeline.import_legacy(storage, 'previous', tmp_path / 'bad')
