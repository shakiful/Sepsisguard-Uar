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
    assert pre.__dict__==state and read_protocol(path)['sha256']==frozen['sha256']
    with pytest.raises(ValueError):freeze_protocol(path,{**payload,'thresholds':{'changed':.8}})


def test_explanation_audit_separates_generic_citations_from_patient_grounding(tmp_path):
    p={'source_id':'fixture','passage_id':'fixture:0','text':'Sepsis is a response to infection.','url':'urn:fixture','title':'Fixture'}
    claim={'text':'Sepsis is a response to infection.','citations':[{'passage_id':'fixture:0','quote':'response to infection'}]}
    case={'status':'supported_research_context','claims':[claim],'case':{'case_id':'test','observations':{'HR':110.}},'passages':[p],
          'trace':[{'role':'planner','response':{'action':'search','query':'sepsis infection'}}],'answer':claim['text'],'reason':'checks passed'}
    generic=audit_cases([case,copy.deepcopy(case)])
    assert generic['cases_with_explicit_patient_observation_grounding']==0
    assert generic['claims_passing_rechecked_citation_numeric_checks']==2
    assert generic['exact_duplicate_output_fraction']==.5
    personalized=copy.deepcopy(case);personalized['claims'][0]['text']='The recorded heart rate is 110.'
    assert grounded_features(personalized['claims'][0]['text'],{'HR':110.})==['HR']
    assert audit_cases([personalized])['cases_with_explicit_patient_observation_grounding']==1
    write_annotation_pack(tmp_path,[case],{'test':[]})
    assert 'pending_independent_review'==generic['clinical_expert_ratings']['status']
    ratings=(tmp_path/'clinician_ratings.csv').read_text();assert 'review_001,,,,,,,' in ratings
    packet=json.loads((tmp_path/'blinded_cases.json').read_text())
    assert 'SepsisLabel' not in json.dumps(packet) and 'model_id' not in json.dumps(packet)


def test_explanation_sampling_uses_unique_patients_and_ignores_outcomes():
    class Data:
        channels=4;window=12;values=np.ones((48,4))
    manifest=pd.DataFrame({'patient_id':[f'A:{i}' for i in range(12)],'split':['validation']*12,'n_hours':[4]*12,'ever_sepsis':[i%2 for i in range(12)]})
    p=np.linspace(0,1,48);first=sample_validation_endpoints(Data(),manifest,p)
    changed=manifest.copy();changed['ever_sepsis']=1-changed.ever_sepsis
    assert first==sample_validation_endpoints(Data(),changed,p) and len(first)==12
    assert len(set(i//4 for i in first))==12


def test_full_synthetic_evaluation_freezes_before_test_and_prepares_blank_review(tmp_path):
    from scripts.run_agents import create_demo
    from scripts.run_evaluation import run_evaluation
    project=Path(__file__).resolve().parents[1];(project/'runs').mkdir(exist_ok=True);before=set((project/'runs').iterdir())
    try:
        data,profile=create_demo(project,tmp_path/'private')
        config=json.loads((project/'configs/evaluation_protocol.json').read_text());config['bootstrap_repetitions']=10
        result=run_evaluation(project,data,tmp_path/'private',profile,config,demo=True)
        assert result['mode']=='synthetic_demo' and result['status']=='retrospective_evaluation_complete'
        assert result['test_B_used_only_after_protocol_freeze'] and not result['test_B_is_now_consumed_for_this_frozen_comparison']
        assert result['explanations']['evaluation_sample']['cases']==12
        assert result['explanations']['evaluation_sample']['clinical_expert_ratings']['completed']==0
        root=Path(result['persistent_checkpoint'])
        assert (root/'FROZEN_PROTOCOL.json').is_file() and (root/'clinical_performance.svg').is_file()
        assert (root/'independent_review/clinician_ratings.csv').is_file()
    finally:
        for directory in set((project/'runs').iterdir())-before:
            if directory.is_dir():shutil.rmtree(directory)
