"""CPU software checks. Tiny random native Qwen is not research performance."""
import copy
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from experiments.qwen_prediction import core, engine
from experiments.qwen_prediction.run import original_comparators, validate_config


def configuration():
    return json.loads((Path(__file__).parents[1]/'protocol.json').read_text())


def tiny_qwen():
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    torch.set_num_threads(1); torch.manual_seed(123)
    c = Qwen3_5Config(text_config={'vocab_size':64,'hidden_size':32,'intermediate_size':64,
        'num_hidden_layers':2,'num_attention_heads':2,'num_key_value_heads':1,'head_dim':16,
        'linear_key_head_dim':8,'linear_value_head_dim':8,'linear_num_key_heads':2,'linear_num_value_heads':2,
        'layer_types':['linear_attention','full_attention'],'max_position_embeddings':128,
        'rope_parameters':{'rope_type':'default','rope_theta':10000.,'partial_rotary_factor':.5}},
        vision_config={'depth':1,'hidden_size':32,'intermediate_size':64,'num_heads':2,'out_hidden_size':32,
        'num_position_embeddings':16,'patch_size':2,'spatial_merge_size':1,'temporal_patch_size':1},
        image_token_id=60,video_token_id=61,vision_start_token_id=62,vision_end_token_id=63)
    base = Qwen3_5ForConditionalGeneration(c); base.config.use_cache = False
    cfg = configuration(); cfg.update(lora_last_layers=2,lora_rank=2,lora_dropout=0.,lora_alpha=4,
        epochs=2,gradient_accumulation=2,checkpoint_every_updates=2,inference_batch_size=2,learning_rate=.01)
    model,targets,trainable = engine.attach_lora(base,cfg)
    return model,cfg,targets


def test_only_past_numeric_feature_history_enters_prompt():
    df = pd.DataFrame({'HR':[80.,np.nan,90.,999.],'Age':[60.,np.nan,np.nan,np.nan],
                       'SepsisLabel':[0,0,1,1],'patient_id':['secret']*4,'source':['B']*4})
    before = core.past_only_text(df,2,['HR','Age'])
    df.loc[3,['HR','Age','SepsisLabel']] = [1e10,1e10,0]
    assert core.past_only_text(df,2,['HR','Age']) == before
    df['SepsisLabel'] = 0; assert core.past_only_text(df,2,['HR','Age']) == before
    assert 'HR|90|0|10|2' in before and 'Age|60|2|NA|1' in before
    assert all(x not in before for x in ('secret','SepsisLabel','patient_id','source'))
    with pytest.raises(ValueError): core.past_only_text(df,2,['SepsisLabel'])


def test_sampling_weights_restore_population_class_mass_and_seed():
    y = np.r_[np.zeros(90),np.ones(10)].astype(int)
    ix,w = core.choose_endpoints(y,[9,5],42); again,ww = core.choose_endpoints(y,[9,5],42)
    assert np.array_equal(ix,again) and np.array_equal(w,ww) and len(set(ix)) == 14
    assert np.sum(w*y[ix]) == 10 and np.sum(w*(1-y[ix])) == 90
    assert np.average(y[ix],weights=w) == pytest.approx(.1)
    with pytest.raises(ValueError): core.choose_endpoints(y,[9,11],42)


def test_B_gate_rejects_tampering_before_raw_reads(tmp_path,monkeypatch):
    opened=[]; monkeypatch.setattr(core,'sha_file',lambda p:opened.append(p))
    with pytest.raises(ValueError): core.build_sample(None,None,None,None,'test_B',[1,1],42,experiment_id='x')
    assert not opened
    payload={'experiment_id':'x','training_complete':True}
    path=tmp_path/'gate.json'; core.save_json(path,{'protocol':payload,'sha256':core.digest(payload)})
    assert core.require_test_gate(path,'x') == payload
    payload['training_complete']=False; core.save_json(path,{'protocol':payload,'sha256':'bad'})
    with pytest.raises(ValueError): core.require_test_gate(path,'x')


def test_real_sample_builder_preserves_original_hour_labels_and_partition(tmp_path):
    from sepsisguard.synthetic import make_demo
    from sepsisguard.load import iter_patients,read_patient
    from sepsisguard.splits import make_splits
    from sepsisguard.preprocessing import record_path
    data=tmp_path/'data';make_demo(data,n_per_source=80)
    summary=pd.DataFrame([{'patient_id':p.patient_id.iat[0],'source':p.source.iat[0],'n_hours':len(p),
                           'ever_sepsis':int(p.SepsisLabel.max())} for p in iter_patients(data)])
    manifest=make_splits(summary)
    inventory=pd.DataFrame([{'patient_id':r.patient_id,'sha256':core.sha_file(record_path(data,r.patient_id))} for r in manifest.itertuples()])
    a=core.build_sample(data,manifest,inventory,['HR','Age'],'train',[2,2],42)
    train_ids=set(manifest.loc[manifest.split.eq('train'),'patient_id'])
    assert len(a['records'])==4
    for r in a['records']:
        assert r['private_patient_id'] in train_ids
        raw=read_patient(record_path(data,r['private_patient_id']),'A')
        assert r['label']==int(raw.SepsisLabel.iat[r['endpoint']])
        assert 'SepsisLabel' not in r['text'] and r['private_patient_id'] not in r['text']
    assert sum(r['weight']*r['label'] for r in a['records'])==a['population_positive_hours']


def test_weighted_metrics_calibration_and_paired_bootstrap():
    y=np.array([0,0,1,1]);p=np.array([.1,.4,.3,.8]);w=np.array([4,4,1,1]);ix=np.repeat(np.arange(4),w)
    r=core.metrics(y,p,w)
    assert r['average_precision'] == pytest.approx(average_precision_score(y[ix],p[ix]))
    assert r['roc_auc'] == pytest.approx(roc_auc_score(y[ix],p[ix]))
    cal=core.calibrate(p,y,w); score=core.apply_calibration(cal,p)
    assert np.isfinite(score).all() and np.all(np.diff(core.apply_calibration(cal,np.linspace(0,1,50)))>=0)
    threshold=core.recall_threshold(y,score,w,.8)
    assert core.metrics(y,score,w,threshold)['sensitivity']>=.8
    args=(y,{'qwen_lora':p,'qwen_base':p},w,np.array(['a','a','b','c']),{'qwen_lora':.5,'qwen_base':.5})
    a=core.paired_bootstrap(*args,repetitions=20);b=core.paired_bootstrap(*args,repetitions=20)
    assert a==b
    assert a['paired_AP_difference_qwen_lora_minus']['qwen_base']['lower']==0
    assert a['paired_AP_difference_qwen_lora_minus']['qwen_base']['upper']==0


def test_native_qwen_lora_updates_and_exact_resume(tmp_path,monkeypatch):
    model,cfg,targets=tiny_qwen()
    assert any('linear_attn' in x for x in targets) and any('self_attn' in x for x in targets)
    assert all('visual' not in x and 'lm_head' not in x for x in targets)
    tokens=[[3,4,5],[8,9,10],[3,4,6],[8,9,11]]*2
    rows=[{'label':j%2,'weight':1.} for j in range(len(tokens))]
    frozen={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
    initial=engine.adapter_digest(engine.adapter_state(model))
    full=engine.Store(tmp_path/'full',tmp_path/'persistent_full')
    a=engine.train_adapter(model,tokens,rows,tokens,rows,[1,2],0,cfg,full,'identity',time.monotonic()+120)
    assert a['optimizer_updates']==8 and a['selected_adapter_sha256']!=initial
    assert all(torch.equal(p,frozen[n]) for n,p in model.named_parameters() if not p.requires_grad)
    resumed,_,_=tiny_qwen(); store=engine.Store(tmp_path/'resume',tmp_path/'persistent_resume')
    real_clock=engine.time.monotonic; ticks=iter([0.,1.,2.,3.]); monkeypatch.setattr(engine.time,'monotonic',lambda:next(ticks))
    with pytest.raises(engine.Paused): engine.train_adapter(resumed,tokens,rows,tokens,rows,[1,2],0,cfg,store,'identity',1.5)
    monkeypatch.setattr(engine.time,'monotonic',real_clock)
    b=engine.train_adapter(resumed,tokens,rows,tokens,rows,[1,2],0,cfg,store,'identity',time.monotonic()+120)
    assert a==b
    pointer=store.read_json('LATEST_TRAINING_CHECKPOINT.json');path=store.persistent/pointer['file']
    path.write_bytes(path.read_bytes()+b'corrupt')
    with pytest.raises(ValueError): store.restore('identity')
    with pytest.raises(ValueError): store.restore('wrong')


def test_padded_scoring_resumes_and_uses_qwen_token_head(tmp_path):
    model,cfg,_=tiny_qwen();model.eval();tokens=[[3,4,5],[8,9],[3,7,6,4]]
    a=engine.score_records(model,tokens,[1,2],0,batch_size=1)
    store=engine.Store(tmp_path/'local',tmp_path/'persistent')
    b=engine.score_records(model,tokens,[1,2],0,batch_size=2,store=store,name='qwen')
    c=engine.score_records(model,tokens,[1,2],0,batch_size=2,store=store,name='qwen')
    assert np.allclose(a,b,atol=1e-6) and np.array_equal(b,c)
    changed=copy.deepcopy(tokens);changed[0][0]=10
    with pytest.raises(ValueError):engine.score_records(model,changed,[1,2],0,store=store,name='qwen')
    with pytest.raises(ValueError):store.relative('../original-results')


def test_original_comparison_reads_exact_matching_scores_without_mutation(tmp_path):
    cfg=configuration();root=tmp_path/'evaluations'/cfg['original_evaluation'];root.mkdir(parents=True)
    result={'clinical':{'models':{n:{'calibrated':{'operating_points':{'validation_recall_80':{'threshold':.2}}}} for n in ('lstm','logistic')}}}
    core.save_json(root/'EVALUATION_RESULT.json',result);cfg['original_result_sha256']=core.sha_file(root/'EVALUATION_RESULT.json')
    path=root/'private_predictions.npz'; np.savez(path,labels=np.array([0,1,0]),lengths=np.array([1,2]),lstm_calibrated=np.array([.1,.8,.2]),logistic_calibrated=np.array([.2,.7,.3]))
    manifest=pd.DataFrame({'patient_id':['B:a','B:b'],'split':['test_B']*2,'n_hours':[1,2]})
    sample={'records':[{'global_index':0,'label':0},{'global_index':1,'label':1}]};before=core.sha_file(path)
    scores,thresholds,meta=original_comparators(tmp_path,cfg,manifest,sample)
    assert np.array_equal(scores['lstm'],[.1,.8]) and thresholds['lstm']==.2
    assert core.sha_file(path)==before and meta['predictors_loaded_or_run'] is False
    sample['records'][0]['label']=1
    with pytest.raises(ValueError):original_comparators(tmp_path,cfg,manifest,sample)


def test_fixed_protocol_and_no_silent_token_truncation():
    cfg=configuration();validate_config(cfg)
    cfg['sample_sizes']['test_B']=[0,2]
    with pytest.raises(ValueError):validate_config(cfg)
    class Tokenizer:
        def apply_chat_template(self,*a,**k):return [1]*12
    with pytest.raises(ValueError):engine.encode_records(Tokenizer(),[{'text':'observations'}],10)
