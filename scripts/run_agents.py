"""Frozen predictor + held-out calibration + local Qwen evidence agents."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

import numpy as np
import pandas as pd
import torch

from sepsisguard.agents import ResearchAgent
from sepsisguard.calibration import SigmoidCalibrator,collect_logits,predictive_entropy
from sepsisguard.evidence import LocalRetriever,build_corpus
from sepsisguard.local_llm import LocalQwen,gpu_preflight
from sepsisguard.load import read_patient
from sepsisguard.models import build_model
from sepsisguard.preprocessing import TrainOnlyPreprocessor,manifest_digest,record_path
from sepsisguard.training import TrainingConfig,atomic_json
from sepsisguard.training_data import cache_split,file_hash
from scripts.compute_check import inspect_compute
from scripts.run_pipeline import run_pipeline,safe_id,source_fingerprint
from scripts.run_training import prepared_checkpoint,train_prepared


def persist_file(path,destination):
    destination.parent.mkdir(parents=True,exist_ok=True)
    partial=destination.with_suffix(destination.suffix+'.partial')
    shutil.copyfile(path,partial)
    if file_hash(path)!=file_hash(partial):raise ValueError('Agent artifact readback mismatch')
    partial.replace(destination)


def load_frozen(project,storage,profile,*,demo=False):
    spec=profile['frozen_predictor'];safe_id(spec['run_id'])
    roots=([Path(storage)/'training'/spec['run_id']] if storage else [])+[Path(project)/'runs'/spec['run_id']]
    for root in roots:
        result_path=root/'TRAINING_RESULT.json';weights=root/spec['weights_file']
        if not result_path.is_file() or not weights.is_file():continue
        result=json.loads(result_path.read_text())
        if result['status']!='training_complete' or result['mode']!=('synthetic_demo' if demo else 'real'):
            raise ValueError('Completed real/demo training result required')
        if result['selected_model']!=spec['architecture'] or file_hash(weights)!=spec['weights_sha256']:
            raise ValueError('Frozen model hash/selection differs from the reviewed profile')
        if result['models'][spec['architecture']]['best_weights_sha256']!=spec['weights_sha256']:
            raise ValueError('Weights disagree with training evidence')
        source,prepared=prepared_checkpoint(project,storage,result['prepared_run_id'],demo=demo)
        pre=TrainOnlyPreprocessor.load(source/'preprocessing/preprocessor.json')
        manifest=pd.read_csv(source/'data/split_manifest.csv')
        if manifest_digest(manifest)!=pre.provenance['manifest_sha256'] or manifest_digest(manifest)!=result['input']['manifest']:
            raise ValueError('Frozen preprocessing/partition does not match the model')
        payload=torch.load(weights,map_location='cpu',weights_only=True)
        identity=hashlib.sha256(json.dumps(result['input'],sort_keys=True).encode()).hexdigest()
        if payload['input_signature']!=identity or payload['channels']!=2*len(pre.features) or payload['architecture']!=spec['architecture']:
            raise ValueError('Weights input signature/architecture mismatch')
        model=build_model(payload['architecture'],payload['channels'],payload['hidden'])
        model.load_state_dict(payload['state_dict']);model.eval()
        return model,pre,manifest,pd.read_csv(source/'data/patient_file_hashes.csv'),prepared
    raise FileNotFoundError('Frozen trained model is missing; mount the Drive containing the completed training run')


def make_case(model,pre,manifest,inventory,data_root,index,calibrator,device):
    rows=manifest.loc[manifest.split.eq('validation')].sort_values('patient_id').reset_index(drop=True)
    ends=np.cumsum(rows.n_hours.to_numpy(dtype=np.int64));total=int(ends[-1])
    if not 0<=index<total:raise ValueError('Validation endpoint outside the fixed split')
    p=int(np.searchsorted(ends,index,side='right'));hour=int(index-(ends[p-1] if p else 0));row=rows.iloc[p]
    path=record_path(data_root,row.patient_id)
    expected=inventory.set_index('patient_id').loc[row.patient_id,'sha256']
    if file_hash(path)!=expected:raise ValueError('Validation record changed since preparation')
    # Slice BEFORE transform/prompt construction; labels and future rows are never exposed.
    patient=read_patient(path,'A').iloc[:hour+1].copy()
    values,mask=pre.transform(patient);length=min(12,len(patient))
    x=np.zeros((1,12,2*len(pre.features)),dtype=np.float32)
    x[0,:length]=np.concatenate([values[-length:],mask[-length:]],axis=1)
    with torch.no_grad():logit=float(model(torch.from_numpy(x).to(device),torch.tensor([length])).float().cpu()[0])
    raw=float(1/(1+np.exp(-np.clip(logit,-50,50))));risk=float(calibrator.transform([logit])[0])
    observations={k:float(patient[k].iat[-1]) for k in ('HR','Resp','Temp','MAP','Lactate','O2Sat','SBP','DBP') if np.isfinite(patient[k].iat[-1])}
    return {'case_id':f'validation_endpoint_{index}','raw_score':round(raw,6),'calibrated_risk':round(risk,6),
            'predictive_entropy_bits':round(predictive_entropy(risk),6),'observation_missing_fraction':round(float(1-mask[-length:].mean()),6),
            'observations':observations,'calibration_fitted':True}


class ScriptedDemo:
    backend='scripted_fixture_not_qwen'
    metadata={'model_id':None,'revision':None,'real_qwen_inference':False}
    def __init__(self):self.calls=[]
    def json(self,role,payload):
        self.calls.append({'role':role})
        if role=='planner':return {'action':'search','query':'sepsis infection response'}
        if role=='writer':
            p=payload['evidence'][0]
            return {'claims':[{'text':'This synthetic reference describes an infection response.',
                               'citations':[{'passage_id':p['passage_id'],'quote':'response to infection'}]}]}
        return {'decisions':[{'claim_index':i,'supported':True} for i in range(len(payload['claims']))]}


def run_agent_cases(project,data_root,storage,profile,*,cases=3,demo=False,max_minutes=20,llm=None):
    project=Path(project);storage=Path(storage) if storage else None;data_root=Path(data_root)
    model,pre,manifest,inventory,preparation=load_frozen(project,storage,profile,demo=demo)
    device=torch.device('cpu' if demo else 'cuda');model=model.to(device)
    torch.set_num_threads(min(torch.get_num_threads(),4))
    cache=project/'data/processed/agent-calibration'/profile['frozen_predictor']['weights_sha256'][:16]
    calset=cache_split(data_root,manifest,pre,inventory,'calibration',cache,allow_calibration=True)
    provenance={'model_sha256':profile['frozen_predictor']['weights_sha256'],'manifest_sha256':manifest_digest(manifest),
                'preprocessor_sha256':hashlib.sha256(json.dumps(pre.__dict__,sort_keys=True).encode()).hexdigest(),
                'calibration_input':json.loads((cache/'cache.json').read_text())['input'],'inference_precision':'fp32'}
    calibration_key=hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest()
    cal_path=project/'runs'/('calibration-'+calibration_key[:20])/'calibrator.json'
    saved_cal=storage/'calibration'/calibration_key/'calibrator.json' if storage else None
    if saved_cal is not None and saved_cal.is_file() and not cal_path.exists():
        marker=json.loads(saved_cal.with_suffix('.checksum.json').read_text())
        if file_hash(saved_cal)!=marker['sha256']:raise ValueError('Saved calibration integrity mismatch')
        cal_path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(saved_cal,cal_path)
        shutil.copyfile(saved_cal.with_suffix('.checksum.json'),cal_path.with_suffix('.checksum.json'))
    if cal_path.exists():
        if file_hash(cal_path)!=json.loads(cal_path.with_suffix('.checksum.json').read_text())['sha256']:
            raise ValueError('Local calibration integrity mismatch')
        calibrator=SigmoidCalibrator.load(cal_path)
        if any(calibrator.provenance.get(k)!=v for k,v in provenance.items()):raise ValueError('Calibrator belongs to different frozen inputs')
    else:
        print(f'Calibrating frozen predictor on {len(calset):,} A/calibration endpoints; B is not read',flush=True)
        logits,labels=collect_logits(model,calset,device)
        calibrator=SigmoidCalibrator.fit(logits,labels,fit_split='A/calibration',provenance=provenance);calibrator.save(cal_path)
        atomic_json(cal_path.with_suffix('.checksum.json'),{'sha256':file_hash(cal_path)})
    if saved_cal is not None:
        persist_file(cal_path,saved_cal);atomic_json(saved_cal.with_suffix('.checksum.json'),{'sha256':file_hash(saved_cal)})
    references=project/'data/processed/agent-references'
    if demo:
        corpus={'sources':[{'source_id':'fixture','scope':'Synthetic test fixture, not clinical evidence'}],
                'passages':[{'source_id':'fixture','passage_id':'fixture:0','url':'urn:synthetic:test',
                             'title':'Synthetic evidence fixture','text':'A synthetic description of sepsis as a response to infection for software testing only.'}]}
    else:
        saved_refs=storage/'agent-reference-corpus' if storage else None
        if saved_refs and (saved_refs/'corpus.json').is_file() and not (references/'corpus.json').exists():
            shutil.copytree(saved_refs,references,dirs_exist_ok=True)
        corpus=build_corpus(references)
        if saved_refs:
            for p in references.iterdir():
                if p.is_file():persist_file(p,saved_refs/p.name)
    total=int(manifest.loc[manifest.split.eq('validation'),'n_hours'].sum())
    indices=np.unique(np.linspace(0,total-1,min(cases,total),dtype=int)).tolist()
    case_inputs=[make_case(model,pre,manifest,inventory,data_root,i,calibrator,device) for i in indices]
    # Release the predictor GPU allocation before loading the single 27B backbone.
    model=model.cpu();del model
    if device.type=='cuda':torch.cuda.empty_cache()
    llm=llm or (ScriptedDemo() if demo else LocalQwen(profile,project/'model-cache'))
    if hasattr(llm,'deadline'):llm.deadline=time.monotonic()+max_minutes*60
    identity={'training_run':profile['frozen_predictor']['run_id'],'weights':profile['frozen_predictor']['weights_sha256'],
              'calibrator_sha256':file_hash(cal_path),'profile':profile,'case_indices':indices,
              'corpus_sha256':hashlib.sha256(json.dumps(corpus,sort_keys=True).encode()).hexdigest(),
              'agent_source_fingerprint':source_fingerprint(project),'backend':llm.backend,
              'model_metadata':{k:v for k,v in llm.metadata.items() if k in ('model_id','revision','dtype','transformers_version','torch_version')},'demo':demo}
    run_id='agents-'+hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:20]
    output=project/'runs'/run_id;output.mkdir(parents=True,exist_ok=True)
    persistent=storage/'agents'/run_id if storage else None
    atomic_json(output/'AGENT_INPUT.json',identity);atomic_json(output/'corpus.json',corpus)
    calibrator.save(output/'calibrator.json');atomic_json(output/'compute.json',inspect_compute())
    completed=[];agent=ResearchAgent(llm,LocalRetriever(corpus['passages']),max_calls=profile['max_tool_decisions'],max_attempts=profile['max_retrieval_attempts'])
    for i,case in enumerate(case_inputs):
        path=output/f'case_{i:03d}.json'
        remote_pointer=persistent/f'case_{i:03d}_pointer.json' if persistent else None
        if not path.exists() and remote_pointer and remote_pointer.is_file():
            pointer=json.loads(remote_pointer.read_text());relative=Path(pointer['file'])
            if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe saved agent path')
            remote=persistent/relative
            if file_hash(remote)!=pointer['sha256']:raise ValueError('Saved agent case changed')
            shutil.copyfile(remote,path)
        if path.exists():
            result=json.loads(path.read_text())
            checksum=path.with_suffix('.checksum.json')
            if checksum.exists() and file_hash(path)!=json.loads(checksum.read_text())['sha256']:
                raise ValueError('Local case changed')
            if result['case']!=case:raise ValueError('Agent case cache does not match frozen input')
        else:
            print(f'Running local agent case {i+1}/{len(case_inputs)}',flush=True)
            before=len(llm.calls)
            result=agent.run(case)
            result['generation_provenance']={'backend':llm.backend,'model':identity['model_metadata'],'backend_calls':len(llm.calls)-before}
            atomic_json(path,result)
        digest=file_hash(path)
        atomic_json(path.with_suffix('.checksum.json'),{'sha256':digest})
        completed.append(result)
        if remote_pointer:
            relative='cases/'+digest+'.json'
            persist_file(path,persistent/relative)
            atomic_json(remote_pointer,{'file':relative,'sha256':digest})
        print(f"Case {i+1}: {result['status']}; calls={result['calls']}; retrievals={result['attempts']}",flush=True)
    inference_calls=len(getattr(llm,'calls',[]))
    result={'status':'agent_workflow_completed','mode':'scripted_demo' if demo else 'real_qwen',
            'run_id':run_id,'backend':llm.backend,'model':llm.metadata,'new_backend_calls':inference_calls,
            'real_qwen_inference_executed_this_invocation':not demo and inference_calls>0,
            'cases':len(completed),'supported_context_cases':sum(x['status']=='supported_research_context' for x in completed),
            'abstained_cases':sum(x['status']=='abstained' for x in completed),'calibration_fitted':True,
            'calibration_fit_split':'A/calibration','calibration_flat':calibrator.provenance.get('flat_calibrator',False),'test_B_raw_files_opened':False,
            'frozen_predictor_sha256':profile['frozen_predictor']['weights_sha256'],
            'persistent_checkpoint':str(persistent) if persistent else None,
            'limits':['Small functional run, not a research benchmark','Verifier is the same backbone, not independent factual ground truth',
                      'Predictive entropy is a routing heuristic, not epistemic certainty','No diagnosis or treatment recommendation is established']}
    report='# SepsisGuard agent functional run\n\n'
    report+=('**SCRIPTED SYNTHETIC DEMO — no Qwen inference.**\n\n' if demo else 'Local Qwen inference workflow; a small functional run, not a clinical benchmark.\n\n')
    for item in completed:
        report+=f"## {item['case']['case_id']}\n\nStatus: {item['status']}\n\n"
        report+=f"Calibrated score: {item['case']['calibrated_risk']:.6f}; predictive entropy: {item['case']['predictive_entropy_bits']:.6f} bits.\n\n"
        report+=(item['answer'] or item['reason'])+'\n\n'
        if item['answer']:
            for passage in item['passages']:
                report+=f"- [{passage['passage_id']}]({passage['url']}) — {passage['title']}\n"
            report+='\n'
    report+='The verifier uses the same backbone; citation checks and its decisions do not establish independent factual correctness. No diagnosis or treatment recommendation is produced.\n'
    (output/'AGENT_REPORT.md').write_text(report)
    atomic_json(output/'AGENT_RESULT.json',result)
    atomic_json(output/'backend_calls.json',getattr(llm,'calls',[]))
    if persistent:
        for p in output.iterdir():
            if p.is_file():persist_file(p,persistent/p.name)
    print(json.dumps(result,indent=2),flush=True)
    print('Agent artifacts and case traces readback verified:',persistent or output,flush=True)
    return result


def create_demo(project,storage):
    run_id='pipeline-agent-demo-'+uuid.uuid4().hex[:10]
    data=project/'data/synthetic-agent-demo'
    prep=run_pipeline(project,data,storage,run_id=run_id,mode='synthetic',run_checks=False)
    source,prep=prepared_checkpoint(project,storage,prep['run_id'],demo=True)
    config=TrainingConfig(batch_size=256,hidden_size=8,baseline_epochs=1,lstm_epochs=1,workers=0,precision='fp32')
    trained=train_prepared(project,data,source,prep,storage,config=config,device='cpu',demo=True,run_checks=False)
    selected=trained['selected_model']
    profile=json.loads((project/'configs/agent_profile.json').read_text())
    profile['frozen_predictor']={'run_id':trained['run_id'],'architecture':selected,'weights_file':selected+'_best.pt',
                                 'weights_sha256':trained['models'][selected]['best_weights_sha256']}
    return data,profile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',default='data/raw');p.add_argument('--storage-root')
    p.add_argument('--profile',default='configs/agent_profile.json');p.add_argument('--cases',type=int,default=3)
    p.add_argument('--max-minutes',type=float,default=20);p.add_argument('--demo',action='store_true');p.add_argument('--preflight',action='store_true')
    a=p.parse_args();project=Path.cwd();storage=Path(a.storage_root) if a.storage_root else None
    if not 1<=a.cases<=10 or not 0<a.max_minutes<=60:raise ValueError('Initial agent run is bounded to 1–10 cases and at most 60 inference minutes')
    profile=json.loads(Path(a.profile).read_text())
    if not a.demo:
        print('GPU preflight:',json.dumps(gpu_preflight()),flush=True)
    if a.preflight:return
    if a.demo:data,profile=create_demo(project,storage)
    else:
        data=Path(a.data_root)
        from sepsisguard.download import run_download
        run_download(data,storage/'dataset-archives' if storage else project/'data/downloads')
    run_agent_cases(project,data,storage,profile,cases=a.cases,demo=a.demo,max_minutes=a.max_minutes)


if __name__=='__main__':main()
