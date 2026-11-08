import copy
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,log_loss

from sepsisguard.evaluation import (PatientMetrics,hourly_utility_weights,cohort_utility_weights,
                                    validation_thresholds,freeze_protocol,read_protocol,evaluate_predictions)
from sepsisguard.vendor.physionet2019 import compute_prediction_utility
from sepsisguard.explanation_evaluation import audit_cases,grounded_features,sample_validation_endpoints,write_annotation_pack


def test_vectorized_utility_matches_official_code_and_shifted_label_example():
    y=np.array([0,0,0,0,1,1]);pred=np.array([0,0,1,1,1,1])
    yes,no=hourly_utility_weights(y)
    assert np.sum(no+pred*(yes-no))==pytest.approx(3.388888888888889)
    rng=np.random.default_rng(4)
    for _ in range(100):
        n=int(rng.integers(2,150));y=np.zeros(n,dtype=int)
        if rng.random()>.3:y[int(rng.integers(n)):]=1
        pred=rng.integers(0,2,n);yes,no=hourly_utility_weights(y)
        assert np.sum(no+pred*(yes-no))==pytest.approx(compute_prediction_utility(y,pred),abs=1e-12)


def test_patient_weighted_metrics_match_explicit_patient_resampling_with_ties():
    y=np.array([0,1,1,0,0,1]);p=np.array([.1,.5,.5,.5,.8,.8]);lengths=np.array([2,1,3]);counts=np.array([2,0,1])
    m=PatientMetrics(y,p,lengths);actual=m.weighted(counts)
    groups=[np.arange(0,2),np.arange(2,3),np.arange(3,6)]
    idx=np.concatenate([groups[0],groups[0],groups[2]])
    assert actual['roc_auc']==pytest.approx(roc_auc_score(y[idx],p[idx]),abs=1e-12)
    assert actual['average_precision']==pytest.approx(average_precision_score(y[idx],p[idx]),abs=1e-12)
    assert actual['brier']==pytest.approx(brier_score_loss(y[idx],p[idx]),abs=1e-12)
    assert actual['log_loss']==pytest.approx(log_loss(y[idx],p[idx]),abs=1e-12)


def test_utility_oracle_inaction_and_patient_boundary_alarm_accounting():
    y=np.r_[np.zeros(8),np.ones(5),np.zeros(10)].astype(int);lengths=np.array([13,10]);yes,no=cohort_utility_weights(y,lengths)
    oracle=(yes>no).astype(float);m=PatientMetrics(y,oracle,lengths)
    assert m.utility(m.utility_array(.5))==pytest.approx(1)
    assert m.utility(m.utility_array(1.01))==pytest.approx(0)
    all_alerts=PatientMetrics(y,np.ones(len(y)),lengths)
    op=all_alerts.operating(all_alerts.operating_arrays(.5))
    assert op['nonseptic_patients_with_any_alarm_fraction']==1
    assert op['positive_flags_per_100_nonseptic_hours']==100
    assert op['alarm_episodes_per_100_nonseptic_hours_6h_refractory']==20
    assert op['detected_by_first_positive_label_fraction']==1


def test_thresholds_are_validation_only_and_bootstrap_is_reproducible():
    y=np.array([0,1,0,1,1,0,0,0]);p=np.array([.1,.4,.2,.9,.6,.3,.1,.2]);lengths=[2,2,2,2]
    thresholds=validation_thresholds(y,p,lengths)
    assert np.mean(p[y==1]>=thresholds['validation_recall_80'])>=.8
    m=PatientMetrics(y,p,lengths);assert m.utility(m.utility_array(thresholds['validation_utility']))>=-1e-12
    first,draws=m.bootstrap(thresholds,repetitions=20,seed=5)
    second,again=m.bootstrap(thresholds,repetitions=20,seed=5)
    assert first==second and draws==again
    assert 0<=first['roc_auc']['lower']<=first['roc_auc']['upper']<=1


def test_B_cache_requires_a_matching_untampered_frozen_protocol_before_any_byte_read(dataset,tmp_path,monkeypatch):
    import sepsisguard.training_data as td
    from sepsisguard.load import iter_patients
    from sepsisguard.splits import make_splits
    from sepsisguard.preprocessing import TrainOnlyPreprocessor,manifest_digest,record_path
    summary=pd.DataFrame([{'patient_id':p.patient_id.iat[0],'source':p.source.iat[0],
                           'n_hours':len(p),'ever_sepsis':int(p.SepsisLabel.max())} for p in iter_patients(dataset)])
    manifest=make_splits(summary);pre=TrainOnlyPreprocessor.fit(dataset,manifest)
    inventory=pd.DataFrame([{'patient_id':r.patient_id,'sha256':td.file_hash(record_path(dataset,r.patient_id))} for r in manifest.itertuples()])
    b_paths={record_path(dataset,r.patient_id) for r in manifest.loc[manifest.source.eq('B')].itertuples()};opened=[]
    original_hash=td.file_hash
    def tracked(path):
        if Path(path) in b_paths:opened.append(Path(path))
        return original_hash(path)
    monkeypatch.setattr(td,'file_hash',tracked)
    with pytest.raises(ValueError):td.cache_split(dataset,manifest,pre,inventory,'test_B',tmp_path/'unlocked')
    path=tmp_path/'protocol.json';payload={'stage':'frozen_before_test_B','test_split':'test_B','manifest_sha256':manifest_digest(manifest),'thresholds':{'frozen':.4}}
    frozen=freeze_protocol(path,payload);assert not opened
    broken=json.loads(path.read_text());broken['protocol']['thresholds']['frozen']=.1;path.write_text(json.dumps(broken))
    with pytest.raises(ValueError):td.cache_split(dataset,manifest,pre,inventory,'test_B',tmp_path/'bad',frozen_test_protocol=path)
    assert not opened
    path.unlink();freeze_protocol(path,payload)
    state=copy.deepcopy(pre.__dict__)
    test=td.cache_split(dataset,manifest,pre,inventory,'test_B',tmp_path/'test',frozen_test_protocol=path)
    assert len(test)==int(manifest.loc[manifest.split.eq('test_B'),'n_hours'].sum()) and len(opened)==80
    assert pre.__dict__==st…3764 tokens truncated…nd code-cell syntax checks also passed.
- Browser control was retried against the user's current Colab tab; both DOM methods still returned `Cannot attach to this target`. No actual A100 training job has been started by the assistant and no predictive research metrics are claimed.

## Completed predictive Colab training (2026-11-04)

- Run `training-05ed8fa8a5f01f4f92a8`, source `7df0a07e2e2e2a489245164081aae293d887acb4`; restored preparation `pipeline-20261104T190221Z-22906fa2` with matching cohort and manifest.
- All 37 tests passed in 74.55 s. Both models trained on CUDA bfloat16 on A100-SXM4-40GB / 83.47 GiB system RAM.
- Logistic: best epoch 6 of 8, validation AP 0.087473, AUROC 0.775901.
- LSTM: best epoch 1, early-stopped after epoch 5, validation AP 0.098821, AUROC 0.772333. LSTM was correctly selected by the predeclared average-precision criterion; its later declining AP suggests overfitting.
- LSTM precision/recall at the predeclared 0.5 threshold: 5.90% / 64.34%. Scores remain uncalibrated and performance is not an independent test result.
- The output reports verified artifacts under `MyDrive/SepsisGuard-UAR/training/training-05ed8fa8a5f01f4f92a8`. Best LSTM weights hash: `266d203773981dce01f2b604c29895e4588b0a2568aa16a333d6a6ffa62764bc`. No separate connected-Drive readback or private-prediction metric recomputation was performed by the reviewer.
- Calibration and B raw files were not opened during training. Independent calibration, final B evaluation and retrieval experiments remain pending.
- Evidence: [TRAINING_COLAB_EVIDENCE.json](reports/verification/TRAINING_COLAB_EVIDENCE.json); interpretation: [TRAINING_RESULTS.md](reports/verification/TRAINING_RESULTS.md).

## Continue from the predictive checkpoint (historical)

1. Preserve the completed training run and selected `lstm_best.pt` hash above. A routine training rerun is unnecessary; the same unchanged configuration resumes this completed result.
2. Use the reserved A/calibration data for probability-calibration work and examine operating thresholds on development data. Keep B locked until the model and evaluation protocol are frozen.
3. If comparing revised training settings, fit on A/train, select on A/validation, preserve this baseline and record the new run separately. Do not infer that more epochs will improve the selected LSTM.

## Deliverables

The source ZIP contains shareable source, two Colab notebooks, tests, aggregate verification reports, and a Git history bundle. A separate private checkpoint ZIP contains protected patient summaries, split manifest, per-file hashes, fitted preprocessing state, notebook execution outputs, environment snapshots, and logs. It excludes raw patient files. Original data can be downloaded again from the documented official S3 mirror.

Real-cohort predictive training is verified from the supplied A100 output. Independent test performance and clinical validity have not been established.

## Proposed agent extension (2026-11-04)

The user requested agentic AI with Qwen 3.5/latest at about 27B and permitted consideration of alternatives. Official source checks identify Qwen3.8-27B as a newer verified candidate; it is proposed as the primary local backbone, with Qwen3.5-27B and Gemma 4 31B as comparisons. The user-reported future allocation is 80GB VRAM / 160GB RAM, so BF16, one shared instance and an initial 8k context cap are proposed. Actual agent memory use, compatibility and explanation quality have not been measured. The design and profile are saved under `docs/AGENTIC_DESIGN.md` and `configs/agent_profile.json`; no LLM download, agent execution or model fine-tuning has occurred.

## Implemented evidence agent (2026-11-04; current)

The single notebook now calls `scripts.run_agents` and preserves the completed predictor. It fits reserved A/calibration sigmoid calibration and runs a bounded local Qwen3.8-27B / LangGraph retrieval, drafting, verification and abstention workflow over three A/validation endpoints. B remains locked. Model revision and Transformers version are pinned; text-only tokenizer/native class compatibility passed.

Actual local execution used scripted responses with a tiny synthetic predictor, not Qwen. It verified calibration, graph execution, checkpoint readback/resume and integrity rejection. Live CDC/NIH retrieval produced 19 passages. Real Qwen generation, real-cohort calibration diagnostics and peak GPU memory are pending: local CUDA is unavailable and browser control cannot attach to Colab. Current results and regression counts are in `reports/verification/AGENT_IMPLEMENTATION_VALIDATION.json`.

Open the latest GitHub notebook on A100 80GB/high RAM and Run all with the same Drive. Existing Drive notebook copies retain old code. Review actual private `AGENT_RESULT.json`, report and case traces before making research claims. The same-model verifier is not independent factual ground truth. Controlled comparisons and blinded explanation annotation remain pending.

## Verified real Qwen agent execution (2026-11-08; current)

- Authentication and private Drive mount completed in the connected browser, including the user-approved Google Drive for desktop OAuth connection. A100-SXM4-80GB (79.25 GiB) / 167.05 GiB RAM was directly observed; Torch 2.14.1+cu130, Transformers 5.22.1.
- Frozen LSTM/preparation restored without retraining. Sigmoid calibration fitted on 119,051 reserved A/calibration hours, 2,574 positive. Non-flat fit; Brier 0.020253 / log loss 0.090618 are calibration-fit diagnostics only. B remained excluded from calibration and inference.
- Live reference HTTP 403 fixed by bundled CDC/NIH text snapshots: 19 passages, both source SHA256 hashes exactly match 2026-11-04. Offline and corruption tests passed; wheel assets verified.
- First real native BF16 Qwen run `agents-b5667526ece3287af39f` (source 69ae1bc) generated three valid planner abstentions. This run remains saved.
- Follow-up `agents-dc179bf37a3ea479c35b` (source cd6c601) clarified reference lookup before evidence sufficiency. Same sample, predictor, calibration, corpus and budgets. Three cases completed planning, local retrieval, drafting and verification: nine Qwen calls, three retrievals, six accepted claims, no abstentions.
- Generation calls totaled 62.77 seconds and 483 output tokens; this excludes checkpoint download/load and other work. Model-load peak allocated memory was 50.96 GiB. Native reference kernels were used; generation peak memory and optimized-kernel performance were not measured.
- Runner artifact copy/readback and a separate checksum read of all three saved case files passed. Private outputs are under `MyDrive/SepsisGuard-UAR/agents/agents-dc179bf37a3ea479c35b`. Both source changes passed all 56 tests in CI.
- Outputs were general educational sepsis context, with identical retrieval queries/passages. This does not establish patient-specific explanation quality, uncertainty-routing benefit, improved AP, clinical validity or independent factual accuracy. Controlled comparisons and blinded annotation remain next research work.
- Aggregate evidence: [AGENT_COLAB_EVIDENCE.json](reports/verification/AGENT_COLAB_EVIDENCE.json).


## Frozen evaluation implementation (2026-11-08)

The canonical notebook now invokes `scripts.run_evaluation` for frozen LSTM/logistic clinical performance and 12-patient explanation grounding. An immutable, verified protocol gates B predictive access. A/calibration supplies calibration; A/validation selects thresholds and the outcome-blind explanation sample. Official challenge utility is parity-tested against the pinned BSD-licensed scorer. Whole-patient bootstrap intervals, alarm burden, automated explanation proxies and a private blank blinded-review packet are implemented. All 64 local tests passed, including synthetic end-to-end execution. Real evaluation results are pending execution; independent expert ratings are not fabricated. See `docs/EVALUATION_PROTOCOL.md`.
