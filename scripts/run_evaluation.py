"""Frozen retrospective B evaluation and A/validation explanation-grounding audit."""
from __future__ import annotations
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import subprocess
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from sklearn.metrics import roc_curve,precision_recall_curve

from scripts.run_agents import load_frozen,frozen_calibrator,run_agent_cases,create_demo,persist_file
from scripts.run_pipeline import source_fingerprint
from scripts.compute_check import inspect_compute
from sepsisguard.calibration import collect_logits
from sepsisguard.evaluation import (digest_json,evaluation_fingerprint,freeze_protocol,read_protocol,
                                  validation_thresholds,evaluate_predictions,PatientMetrics)
from sepsisguard.explanation_evaluation import sample_validation_endpoints,audit_cases,write_annotation_pack
from sepsisguard.training import atomic_json
from sepsisguard.training_data import cache_split,file_hash
from sepsisguard.preprocessing import manifest_digest,record_path
from sepsisguard.load import read_patient
from sepsisguard.evidence import build_corpus
from sepsisguard.local_llm import gpu_preflight


def candidate_profiles(project,storage,profile,demo):
    primary=profile['frozen_predictor']['architecture']
    other='logistic' if primary=='lstm' else 'lstm'
    run_id=profile['frozen_predictor']['run_id']
    roots=([Path(storage)/'training'/run_id] if storage else [])+[Path(project)/'runs'/run_id]
    root=next((r for r in roots if (r/'TRAINING_RESULT.json').exists()),None)
    if root is None:raise FileNotFoundError('Completed frozen training result required')
    result=json.loads((root/'TRAINING_RESULT.json').read_text())
    profiles={primary:copy.deepcopy(profile)}
    for name in (other,):
        spec={'run_id':run_id,'architecture':name,'weights_file':name+'_best.pt',
              'weights_sha256':result['models'][name]['best_weights_sha256']}
        p=copy.deepcopy(profile);p['frozen_predictor']=spec;profiles[name]=p
    return profiles


@torch.no_grad()
def ablate_features(model,dataset,indices,features,calibrator,device):
    output={};model=model.to(device).eval();n=len(features)
    for index in indices:
        item=dataset[index];x=item['x'].unsqueeze(0).to(device);length=torch.tensor([item['length']])
        base=float(calibrator.transform(model(x,length).float().cpu().numpy())[0])
        changed=x.repeat(n,1,1)
        for j in range(n):changed[j,:,j]=0.;changed[j,:,j+n]=0.
        predicted=calibrator.transform(model(changed,length.repeat(n)).float().cpu().numpy())
        effects=[{'feature':name,'risk_change_when_removed':float(base-score)} for name,score in zip(features,predicted)]
        output[f'validation_endpoint_{index}']=sorted(effects,key=lambda x:-abs(x['risk_change_when_removed']))
    return output


def causal_contexts(manifest,data_root,indices):
    rows=manifest.loc[manifest.split.eq('validation')].sort_values('patient_id').reset_index(drop=True)
    ends=np.cumsum(rows.n_hours.to_numpy(dtype=int));result={}
    for index in indices:
        j=int(np.searchsorted(ends,index,side='right'));hour=int(index-(ends[j-1] if j else 0));row=rows.iloc[j]
        # Prefix-only access for reviewer context; outcomes and original IDs are excluded.
        df=read_patient(record_path(data_root,row.patient_id),'A').iloc[:hour+1]
        context=[]
        for t in range(max(0,hour-11),hour+1):
            observed={k:float(df[k].iat[t]) for k in ('HR','Resp','Temp','MAP','Lactate','O2Sat','SBP','DBP') if np.isfinite(df[k].iat[t])}
            context.append({'hours_relative_to_endpoint':t-hour,'observed':observed})
        result[f'validation_endpoint_{index}']=context
    return result


def read_case_files(root):
    cases=[]
    for p in sorted(Path(root).glob('case_[0-9][0-9][0-9].json')):
        expected=json.loads(p.with_suffix('.checksum.json').read_text())['sha256']
        if file_hash(p)!=expected:raise ValueError('Explanation case checksum mismatch')
        cases.append(json.loads(p.read_text()))
    return cases


def save_plots(output,clinical,predictions,labels):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for name,p in predictions.items():
        fpr,tpr,_=roc_curve(labels,p);precision,recall,_=precision_recall_curve(labels,p)
        # Bounded aggregate curves; no individual labels/probabilities exported.
        ix=np.linspace(0,len(fpr)-1,min(300,len(fpr)),dtype=int)
        axes[0].plot(fpr[ix],tpr[ix],label=name)
        ix=np.linspace(0,len(recall)-1,min(300,len(recall)),dtype=int)
        axes[1].plot(recall[ix],precision[ix],label=name)
        bins=clinical[name]['calibrated']['reliability_bins'];nonempty=[b for b in bins if b['hours']]
        axes[2].plot([b['mean_risk'] for b in nonempty],[b['observed_fraction'] for b in nonempty],'o-',label=name)
    axes[0].plot([0,1],[0,1],'k--',alpha=.4);axes[2].plot([0,1],[0,1],'k--',alpha=.4)
    axes[1].axhline(float(np.mean(labels)),color='k',linestyle='--',alpha=.4,label='prevalence')
    for ax,title,x,y in zip(axes,['B: ROC','B: Precision–recall','B: Reliability'],['False positive rate','Recall','Mean calibrated risk'],['Sensitivity','Precision','Observed positive fraction']):
        ax.set(title=title,xlabel=x,ylabel=y);ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(output/'clinical_performance.svg');fig.savefig(output/'clinical_performance.png',dpi=160);plt.close(fig)


def build_report(result):
    text='# SepsisGuard retrospective evaluation\n\n'
    if result['mode']=='synthetic_demo':text+='**SYNTHETIC SOFTWARE CHECK — no research performance or Qwen inference.**\n\n'
    text+=f"Protocol: `{result['protocol_sha256']}`\n\n"
    text+='Thresholds and all scoring rules were fixed using A/validation and frozen before predictive evaluation on B. Earlier descriptive auditing was not used for model or threshold selection. Calibration used A/calibration. B is the held-out source from the public development data, not the hidden Challenge leaderboard test set or a prospective clinical trial.\n\n'
    text+='| Model | B AUROC (95% patient-bootstrap CI) | B AP (95% CI) | Brier | ECE10 |\n|---|---:|---:|---:|---:|\n'
    for name,m in result['clinical']['models'].items():
        r=m['calibrated'];ci=m['confidence_intervals_95']
        fmt=lambda k:f"{r[k]:.4f} ({ci[k]['lower']:.4f}–{ci[k]['upper']:.4f})"
        text+=f"| {name} | {fmt('roc_auc')} | {fmt('average_precision')} | {r['brier']:.4f} | {r['ece_10_equal_width']:.4f} |\n"
    text+='\n| Model / calibrated rule | Hourly sensitivity | PPV | Nonseptic patients with any alarm | Alarm episodes /100 nonseptic hours | Normalized challenge utility |\n|---|---:|---:|---:|---:|---:|\n'
    fmt=lambda x:'undefined' if x is None else f'{x:.4f}'
    for name,m in result['clinical']['models'].items():
        for rule,r in m['calibrated']['operating_points'].items():
            text+='| '+name+' / '+rule+' | '+' | '.join(fmt(r[k]) for k in ('sensitivity','precision','nonseptic_patients_with_any_alarm_fraction','alarm_episodes_per_100_nonseptic_hours_6h_refractory','normalized_challenge_utility'))+' |\n'
    q=result['explanations']['evaluation_sample']
    text+=f"\nExplanation sample: {q['cases']} distinct A/validation patients, stratified by risk and missingness. Output cases {q['output_cases']}; abstentions {q['abstained_cases']}. Explicit patient-observation grounding: {q['cases_with_explicit_patient_observation_grounding']}/{q['cases']}. Unique queries {q['unique_queries']}/{q['queries']}. Rechecked citation/numeric claims {q['claims_passing_rechecked_citation_numeric_checks']}/{q['claims']}.\n\n"
    text+='These are automated proxies. Citation matching and a same-model verifier do not establish independent clinical correctness. A blinded clinician review packet with blank rating fields is saved privately; no expert ratings are claimed. Input ablation measures sensitivity to removed channels, not causal clinical attribution.\n\n'
    text+='The original hourly targets are already shifted six hours. Label-relative detection and lead-time summaries are not measured clinical-onset lead times. Challenge utility uses the official onset convention. Alarm episodes use a predeclared six-hour refractory period. Thresholds were not retuned on B. Further predictor selection using these B results would consume the holdout.\n'
    return text


def run_evaluation(project,data_root,storage,profile,config,*,demo=False,run_explanations=True):
    project=Path(project);data_root=Path(data_root);storage=Path(storage) if storage else None
    device=torch.device('cpu' if demo else 'cuda');torch.set_num_threads(min(torch.get_num_threads(),4))
    profiles=candidate_profiles(project,storage,profile,demo);primary=next(iter(profiles))
    if not demo and primary!=config['selected_model']:raise ValueError('Selected predictor must remain the frozen LSTM')
    models={};calibrators={};cal_paths={};validation={};raw_val={};thresholds={};data=None
    for name,spec in profiles.items():
        model,pre,manifest,inventory,prepared=load_frozen(project,storage,spec,demo=demo,require_selected=name==primary)
        model=model.to(device);models[name]=model
        if data is None:
            data=cache_split(data_root,manifest,pre,inventory,'validation',project/'data/processed/evaluation/validation')
        cal,path=frozen_calibrator(project,data_root,storage,spec,model,pre,manifest,inventory,device)
        calibrators[name]=cal;cal_paths[name]=path
        logits,y=collect_logits(model,data,device)
        raw_val[name]=expit(np.asarray(logits,dtype=float));validation[name]=cal.transform(logits)
        lengths=manifest.loc[manifest.split.eq('validation')].sort_values('patient_id').n_hours.to_numpy(dtype=int)
        thresholds[name]=validation_thresholds(y,validation[name],lengths,config['target_hourly_recall'],config['utility_threshold_validation_quantiles'])
    indices=sample_validation_endpoints(data,manifest,validation[primary],seed=config['explanation_sampling_seed'])
    corpus=build_corpus(project/'data/processed/agent-references')
    protocol={'stage':'frozen_before_test_B','test_split':'test_B','configuration':config,
              'manifest_sha256':manifest_digest(manifest),'preprocessor_sha256':prepared['input']['preprocessor_sha256'] if 'preprocessor_sha256' in prepared['input'] else hashlib.sha256(json.dumps(pre.__dict__,sort_keys=True).encode()).hexdigest(),
              'predictor_weights':{n:p['frozen_predictor']['weights_sha256'] for n,p in profiles.items()},
              'calibrator_sha256':{n:file_hash(p) for n,p in cal_paths.items()},'thresholds':thresholds,
              'explanation_indices_before_test':indices,'agent_profile':profile,'corpus_sha256':digest_json(corpus),
              'evaluation_source_fingerprint':evaluation_fingerprint(project),'torch':str(torch.__version__),'demo':demo}
    protocol_sha=digest_json(protocol);run_id='evaluation-'+protocol_sha[:20]
    output=project/'runs'/run_id;output.mkdir(parents=True,exist_ok=True)
    persistent=storage/'evaluations'/run_id if storage else None
    protocol_path=output/'FROZEN_PROTOCOL.json';freeze_protocol(protocol_path,protocol)
    if persistent:
        persist_file(protocol_path,persistent/'FROZEN_PROTOCOL.json')
        if read_protocol(persistent/'FROZEN_PROTOCOL.json')['sha256']!=protocol_sha:raise ValueError('Persistent protocol readback failed')
    print('PROTOCOL FROZEN before B access:',protocol_sha,flush=True)
    print('Fixed rules:',json.dumps(config),flush=True)
    if evaluation_fingerprint(project)!=protocol['evaluation_source_fingerprint']:raise ValueError('Evaluation source changed after protocol freeze')
    ablations=ablate_features(models[primary],data,indices,pre.features,calibrators[primary],device)
    contexts=causal_contexts(manifest,data_root,indices)
    validation_results={n:evaluate_predictions(y,validation[n],lengths,thresholds[n]) for n in models}
    print('Opening B only under the frozen evaluation protocol; no fitting or threshold tuning follows',flush=True)
    test=cache_split(data_root,manifest,pre,inventory,'test_B',project/'data/processed/evaluation/test_B',frozen_test_protocol=protocol_path)
    test_lengths=manifest.loc[manifest.split.eq('test_B')].sort_values('patient_id').n_hours.to_numpy(dtype=int)
    clinical={};predictions={};draws={};private_scores={}
    for name,model in models.items():
        logits,labels=collect_logits(model,test,device);raw=expit(np.asarray(logits,dtype=float));p=calibrators[name].transform(logits)
        predictions[name]=p;private_scores[name+'_raw']=raw;private_scores[name+'_calibrated']=p
        print(f'Scoring frozen {name} on B: {len(test_lengths):,} patients / {len(labels):,} hours',flush=True)
        m=PatientMetrics(labels,p,test_lengths)
        ci,bootstrap=m.bootstrap(thresholds[name],config['bootstrap_repetitions'],config['bootstrap_seed']);draws[name]=bootstrap
        clinical[name]={'raw':evaluate_predictions(labels,raw,test_lengths,{'raw_0_5':.5}),
                        'calibrated':evaluate_predictions(labels,p,test_lengths,thresholds[name]),'confidence_intervals_95':ci}
    names=list(models);differences=[a['average_precision']-b['average_precision'] for a,b in zip(draws[names[0]],draws[names[1]]) if a['average_precision'] is not None and b['average_precision'] is not None]
    comparison={'contrast':names[0]+' minus '+names[1],'average_precision_difference':clinical[names[0]]['calibrated']['average_precision']-clinical[names[1]]['calibrated']['average_precision'],
                'paired_patient_bootstrap_ci95':[float(np.quantile(differences,.025)),float(np.quantile(differences,.975))]}
    np.savez_compressed(output/'private_predictions.npz',labels=labels,lengths=test_lengths,**private_scores)
    save_plots(output,clinical,predictions,labels)
    del test
    for model in models.values():model.cpu()
    models.clear();gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()
    if not run_explanations:raise ValueError('This protocol requires its predeclared explanation evaluation')
    agent=run_agent_cases(project,data_root,storage,profile,cases=len(indices),demo=demo,
                          max_minutes=config['explanation_max_inference_minutes'],endpoint_indices=indices)
    case_root=(storage/'agents'/agent['run_id']) if storage else project/'runs'/agent['run_id']
    cases=read_case_files(case_root);quality=audit_cases(cases,ablations)
    old=[]
    if storage:
        for rid in ('agents-b5667526ece3287af39f','agents-dc179bf37a3ea479c35b'):
            r=storage/'agents'/rid
            if (r/'AGENT_RESULT.json').exists():old.append({'run_id':rid,'audit':audit_cases(read_case_files(r))})
    annotation=output/'independent_review';write_annotation_pack(annotation,cases,contexts,seed=config['explanation_sampling_seed'])
    atomic_json(output/'private_feature_ablations.json',ablations)
    result={'status':'retrospective_evaluation_complete','mode':'synthetic_demo' if demo else 'real',
            'run_id':run_id,'protocol_sha256':protocol_sha,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=project,text=True).strip(),
            'evaluation_source_fingerprint':evaluation_fingerprint(project),'compute':inspect_compute(),
            'test_B_used_only_after_protocol_freeze':True,'test_B_is_now_consumed_for_this_frozen_comparison':not demo,
            'model_weights_unchanged':True,'clinical':{'split':'test_B','patients':len(test_lengths),'hours':len(labels),
            'positive_hours':int(labels.sum()),'positive_patients':int(sum(np.any(labels[start:start+n]) for start,n in zip(np.r_[0,np.cumsum(test_lengths)[:-1]],test_lengths))),
            'models':clinical,'paired_comparison':comparison,'bootstrap_repetitions':config['bootstrap_repetitions'],'bootstrap_unit':'whole_patient'},
            'validation_selection_diagnostics':validation_results,
            'explanations':{'split':'A/validation','agent_run_id':agent['run_id'],'evaluation_sample':quality,'prior_runs':old,
                            'clinical_expert_review':'pending; blank private annotation packet prepared'},
            'persistent_checkpoint':str(persistent) if persistent else None,'completed_at':datetime.now(timezone.utc).isoformat()}
    atomic_json(output/'EVALUATION_RESULT.json',result);(output/'EVALUATION_REPORT.md').write_text(build_report(result))
    atomic_json(output/'private_bootstrap_draws.json',draws)
    if persistent:
        for p in output.rglob('*'):
            if p.is_file():persist_file(p,persistent/p.relative_to(output))
    print('EVALUATION_RESULT_JSON',json.dumps(result),flush=True)
    print('Evaluation and private reviewer packet readback verified:',persistent or output,flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',default='data/raw');p.add_argument('--storage-root')
    p.add_argument('--profile',default='configs/agent_profile.json');p.add_argument('--protocol',default='configs/evaluation_protocol.json');p.add_argument('--demo',action='store_true')
    a=p.parse_args();project=Path.cwd();storage=Path(a.storage_root) if a.storage_root else None
    profile=json.loads(Path(a.profile).read_text());config=json.loads(Path(a.protocol).read_text())
    if a.demo:
        data,profile=create_demo(project,storage);config['bootstrap_repetitions']=20
    else:
        print('Evaluation GPU preflight:',json.dumps(gpu_preflight()),flush=True)
        data=Path(a.data_root)
        from sepsisguard.download import run_download
        run_download(data,storage/'dataset-archives' if storage else project/'data/downloads')
    run_evaluation(project,data,storage,profile,config,demo=a.demo)


if __name__=='__main__':main()
