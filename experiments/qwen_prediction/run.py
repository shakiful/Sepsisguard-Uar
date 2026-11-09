"""Fine-tune Qwen itself, freeze the adapter, then run an exploratory paired pilot."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import pandas as pd
import torch

from .core import (SYSTEM_PROMPT, digest, sha_file, source_digest, build_sample, calibrate,
                   apply_calibration, recall_threshold, metrics, paired_bootstrap)
from .engine import Store, Paused, load_qwen, encode_records, score_records, train_adapter, adapter_state, adapter_digest


def validate_config(c):
    if c['model_id'] != 'Qwen/Qwen3.8-27B' or len(c['revision']) != 40: raise ValueError('Use the pinned, separate Qwen protocol')
    for split in ('train','validation','calibration','test_B'):
        if len(c['sample_sizes'][split]) != 2 or any(type(n) is not int or n <= 0 for n in c['sample_sizes'][split]): raise ValueError('Positive fixed class sample sizes required')
    for k in ('epochs','gradient_accumulation','lora_rank','lora_last_layers','max_tokens','inference_batch_size','checkpoint_every_updates','bootstrap_repetitions'):
        if type(c[k]) is not int or c[k] <= 0: raise ValueError('Positive integer required: '+k)
    if c['window_hours'] != 12 or not 0 < c['target_recall'] <= 1 or not 0 <= c['lora_dropout'] < 1: raise ValueError('Unsupported history, recall or dropout')


def original_comparators(storage, config, manifest, sample):
    root = Path(storage)/'evaluations'/config['original_evaluation']; meta = root/'EVALUATION_RESULT.json'
    path = root/'private_predictions.npz'
    if not meta.is_file() or not path.is_file():
        return {}, {}, {'status':'missing_original_predictions','limitation':'No matched LSTM/logistic superiority claim can be made'}
    if sha_file(meta) != config['original_result_sha256']: raise ValueError('Original result differs from the preserved verified evidence')
    result = json.loads(meta.read_text())
    rows = manifest.loc[manifest.split.eq('test_B')].sort_values('patient_id')
    indices = np.array([r['global_index'] for r in sample['records']],dtype=int)
    with np.load(path,allow_pickle=False) as saved:
        if not np.array_equal(saved['lengths'],rows.n_hours.to_numpy(dtype=int)): raise ValueError('Original predictions use a different patient ordering')
        if not np.array_equal(saved['labels'][indices],np.array([r['label'] for r in sample['records']])): raise ValueError('Matched original prediction labels disagree')
        predictions = {n:np.asarray(saved[n+'_calibrated'][indices],dtype=float) for n in ('lstm','logistic')}
    thresholds = {n:result['clinical']['models'][n]['calibrated']['operating_points']['validation_recall_80']['threshold'] for n in predictions}
    return predictions, thresholds, {'status':'matched_saved_predictions_loaded_read_only','original_run':config['original_evaluation'],
               'original_predictions_sha256':sha_file(path),'original_result_sha256':sha_file(meta),'predictors_loaded_or_run':False}


def report_markdown(result):
    text = '# Qwen-only LoRA prediction pilot\n\n'
    text += 'Qwen itself supplies the binary prediction scores. LoRA adapters were optimized on A/train before B prediction. The previous LSTM/logistic experiment was not modified.\n\n'
    text += '**Exploratory sampled comparison.** B was previously evaluated. This is neither a fresh independent test nor the original full-cohort evaluation. Samples were drawn uniformly within original hourly label classes; inverse-inclusion weights restore the original hourly class prevalence for point estimates. Confidence intervals are conditional pilot intervals.\n\n'
    text += f"Training examples: {result['training']['train_examples']}; optimizer updates: {result['training']['optimizer_updates']}; selected fine-tuned epoch: {result['training']['best_epoch']}. Test sample: {result['test_sample']['sample_hours']} hours from {result['test_sample']['distinct_patients']} records.\n\n"
    text += '| Model | Weighted AUROC | Weighted AP | Brier | Hourly sensitivity | Hourly PPV |\n|---|---:|---:|---:|---:|---:|\n'
    fmt = lambda x:'undefined' if x is None else f'{x:.5f}'
    for name,row in result['models'].items():
        text += '| '+name+' | '+' | '.join(fmt(row['calibrated'][k]) for k in ('roc_auc','average_precision','brier','sensitivity','precision'))+' |\n'
    text += '\nQwen cutoffs target 80% weighted A/validation recall after A/calibration-only isotonic calibration. Saved LSTM/logistic cutoffs retain their original full A/validation selection. AP/AUROC do not depend on those cutoffs. Raw Qwen metrics and all confidence intervals are in the aggregate JSON.\n\n'
    for name,row in result['improvement_assessment'].items():
        text += f"- Qwen LoRA versus {name}: AP difference {row['AP_difference']:.6f}; paired 95% interval {row['paired_interval_95']}; {row['interpretation']}.\n"
    text += '\nSparse endpoints do not support continuous alarm-burden, early-detection lead-time or official Challenge-utility estimates. Do not compare this sample directly with the older full-B headline metrics. A new independent cohort is needed for confirmatory evaluation. No LLM judge supplies performance labels; published dataset labels supply numerical evaluation.\n\n'
    text += 'Base Qwen weights stay frozen, and only declared LoRA adapters in the final 16 language-model layers are trained. This pilot does not claim exhaustive fine-tuning on every training hour, patient-specific explanatory validity, clinical readiness, or improved clinical outcomes. Protected inputs, scores, adapters and optimizer checkpoints remain private.\n'
    return text


def save_plot(store, y, predictions, weights):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, precision_recall_curve
    fig,axes = plt.subplots(1,2,figsize=(10,4))
    for name,p in predictions.items():
        fpr,tpr,_ = roc_curve(y,p,sample_weight=weights)
        prec,rec,_ = precision_recall_curve(y,p,sample_weight=weights)
        axes[0].plot(fpr,tpr,label=name); axes[1].plot(rec,prec,label=name)
    axes[0].plot([0,1],[0,1],'k--',alpha=.3)
    axes[1].axhline(np.average(y,weights=weights),color='k',linestyle='--',alpha=.4)
    for ax,title,x,yy in zip(axes,['Exploratory weighted ROC','Exploratory weighted precision–recall'],['False-positive rate','Recall'],['Sensitivity','Precision']):
        ax.set(title=title,xlabel=x,ylabel=yy); ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(store.local/'AGGREGATE_CURVES.svg'); plt.close(fig); store.publish('AGGREGATE_CURVES.svg')


def run(project, data_root, storage, config, *, max_minutes=90, model_cache=None):
    from scripts.run_training import prepared_checkpoint
    from sepsisguard.preprocessing import TrainOnlyPreprocessor, manifest_digest, record_path
    from sepsisguard.download import run_download
    from scripts.compute_check import inspect_compute
    project,data_root,storage = Path(project),Path(data_root),Path(storage)
    validate_config(config)
    if not torch.cuda.is_available(): raise RuntimeError('Real Qwen fine-tuning requires the authorized Colab GPU; CPU tests are separate')
    torch.manual_seed(config['seed']); torch.cuda.manual_seed_all(config['seed']); torch.set_num_threads(min(4,torch.get_num_threads()))
    source, preparation = prepared_checkpoint(project,storage,config['prepared_run'])
    manifest = pd.read_csv(source/'data/split_manifest.csv'); inventory = pd.read_csv(source/'data/patient_file_hashes.csv')
    pre = TrainOnlyPreprocessor.load(source/'preprocessing/preprocessor.json')
    if manifest_digest(manifest) != pre.provenance['manifest_sha256']: raise ValueError('Original prepared partition differs')
    software = {n:importlib.metadata.version(n) for n in ('torch','transformers','peft','numpy','scikit-learn')}
    identity = {'config':config,'source_sha256':source_digest(project),'manifest_sha256':manifest_digest(manifest),
                'inventory_sha256':sha_file(source/'data/patient_file_hashes.csv'),'features':pre.features,'prompt_sha256':hashlib_sha(SYSTEM_PROMPT),'software':software}
    key = digest(identity); run_id = 'qwen-lora-'+key[:20]
    store = Store(project/'runs/qwen_prediction'/run_id,storage/'experiments/qwen_prediction'/run_id)
    complete = store.read_json('QWEN_RESULT.json')
    if complete and complete.get('status') == 'exploratory_pilot_complete':
        print('Existing completed immutable experiment:',run_id,flush=True); print('QWEN_RESULT_JSON',json.dumps(complete),flush=True); return complete
    store.json('RUN_INPUT.json',identity)
    print('Isolated run:',run_id,'private checkpoint:',store.persistent,flush=True)
    run_download(data_root,storage/'experiments/qwen_prediction/dataset-archives')
    def sample(split, gate=None):
        if split == 'test_B':
            from .core import require_test_gate
            require_test_gate(gate,run_id)
        name = '_private/'+split+'_sample.json'; cached = store.read_json(name)
        if cached:
            if cached['identity'] != key: raise ValueError('Sample cache belongs to different inputs')
            expected = inventory.set_index('patient_id').sha256
            for pid in set(r['private_patient_id'] for r in cached['data']['records']):
                if sha_file(record_path(data_root,pid)) != expected.loc[pid]: raise ValueError('Consumed raw record changed since sampling')
            return cached['data']
        value = build_sample(data_root,manifest,inventory,pre.features,split,config['sample_sizes'][split],config['seed'],gate=gate,experiment_id=run_id)
        store.json(name,{'identity':key,'data':value}); return value
    # All gradient and checkpoint-selection data are A-only. No B raw reads here.
    train = sample('train'); val = sample('validation')
    model,tokenizer,digits,metadata = load_qwen(config,model_cache or project/'model-cache')
    store.json('MODEL_METADATA.json',metadata)
    encoded_train = encode_records(tokenizer,train['records'],config['max_tokens'])
    encoded_val = encode_records(tokenizer,val['records'],config['max_tokens'])
    print('Input tokens max/mean:',max(map(len,encoded_train)),float(np.mean(list(map(len,encoded_train)))),flush=True)
    deadline = time.monotonic()+max_minutes*60
    try:
        training = train_adapter(model,encoded_train,train['records'],encoded_val,val['records'],digits,tokenizer.pad_token_id,config,store,key,deadline)
        store.json('TRAINING_COMPLETE.json',training)
        # Portable selected adapter; full 27B base weights are never duplicated or overwritten.
        adapter_dir = store.local/'selected_adapter'; model.save_pretrained(adapter_dir,safe_serialization=True)
        tokenizer.save_pretrained(adapter_dir)
        for path in adapter_dir.iterdir():
            if path.is_file(): store.publish(path.relative_to(store.local))
        cal = sample('calibration'); encoded_cal = encode_records(tokenizer,cal['records'],config['max_tokens'])
        ycal = np.array([r['label'] for r in cal['records']]); wcal = np.array([r['weight'] for r in cal['records']])
        yval = np.array([r['label'] for r in val['records']]); wval = np.array([r['weight'] for r in val['records']])
        calibrators = {}; thresholds = {}; validation = {}
        for name,disabled in (('qwen_base',True),('qwen_lora',False)):
            raw_cal = score_records(model,encoded_cal,digits,tokenizer.pad_token_id,batch_size=config['inference_batch_size'],disable_adapter=disabled,store=store,name=name+'_calibration',deadline=deadline)
            raw_val = score_records(model,encoded_val,digits,tokenizer.pad_token_id,batch_size=config['inference_batch_size'],disable_adapter=disabled,store=store,name=name+'_validation',deadline=deadline)
            calibrators[name] = calibrate(raw_cal,ycal,wcal)
            pval = apply_calibration(calibrators[name],raw_val)
            thresholds[name] = recall_threshold(yval,pval,wval,config['target_recall'])
            validation[name] = {'raw':metrics(yval,raw_val,wval),'calibrated':metrics(yval,pval,wval,thresholds[name])}
        store.json('_private/calibration.json',calibrators)
        gate_payload = {'experiment_id':run_id,'training_complete':True,'selected_adapter_sha256':training['selected_adapter_sha256'],
                        'calibrator_sha256':digest(calibrators),'thresholds':thresholds,'identity_sha256':key,
                        'manifest_sha256':manifest_digest(manifest),'test_sampling_seed':config['seed'],'test_sample_sizes':config['sample_sizes']['test_B'],
                        'scope':config['test_scope']}
        gate = {'protocol':gate_payload,'sha256':digest(gate_payload)}
        previous_gate = store.read_json('FROZEN_EVALUATION.json')
        if previous_gate and previous_gate != gate: raise ValueError('Refusing a changed frozen evaluation gate')
        store.json('FROZEN_EVALUATION.json',gate); print('FROZEN before B prediction:',gate['sha256'],flush=True)
        if source_digest(project) != identity['source_sha256']: raise ValueError('Source changed during the experiment')
        test = sample('test_B',store.persistent/'FROZEN_EVALUATION.json'); encoded_test = encode_records(tokenizer,test['records'],config['max_tokens'])
        y = np.array([r['label'] for r in test['records']]); w = np.array([r['weight'] for r in test['records']])
        predictions = {}; results = {}
        for name,disabled in (('qwen_base',True),('qwen_lora',False)):
            raw = score_records(model,encoded_test,digits,tokenizer.pad_token_id,batch_size=config['inference_batch_size'],disable_adapter=disabled,store=store,name=name+'_test_B',deadline=deadline)
            predictions[name] = apply_calibration(calibrators[name],raw)
            results[name] = {'raw':metrics(y,raw,w),'calibrated':metrics(y,predictions[name],w,thresholds[name])}
        baseline, baseline_thresholds, comparison = original_comparators(storage,config,manifest,test)
        predictions.update(baseline); thresholds.update(baseline_thresholds)
        for name,p in baseline.items(): results[name] = {'calibrated':metrics(y,p,w,thresholds[name])}
        bootstrap = paired_bootstrap(y,predictions,w,np.array([r['private_patient_id'] for r in test['records']]),thresholds,config['bootstrap_repetitions'],config['seed'])
        assessments = {}
        for name,ci in bootstrap['paired_AP_difference_qwen_lora_minus'].items():
            delta = results['qwen_lora']['calibrated']['average_precision']-results[name]['calibrated']['average_precision']
            verdict = 'positive conditional pilot AP difference' if ci and ci['lower']>0 else ('negative conditional pilot AP difference' if ci and ci['upper']<0 else 'no clear conditional pilot AP improvement')
            assessments[name] = {'AP_difference':delta,'paired_interval_95':ci,'interpretation':verdict}
        save_plot(store,y,predictions,w)
        result = {'status':'exploratory_pilot_complete','mode':'real_qwen_lora','run_id':run_id,'identity_sha256':key,
                  'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=project,text=True).strip(),
                  'source_sha256':identity['source_sha256'],'protocol_sha256':gate['sha256'],'model':metadata,
                  'training':training,'validation':validation,'models':results,'uncertainty':bootstrap,'improvement_assessment':assessments,
                  'test_sample':{k:v for k,v in test.items() if k!='records'},
                  'original_comparators':comparison,'scope':config['test_scope'],'compute':inspect_compute(),
                  'persistent_checkpoint':str(store.persistent),'completed_at':datetime.now(timezone.utc).isoformat(),
                  'original_notebook_and_results_modified':False,'published_labels_used_for_evaluation':True,
                  'clinical_effectiveness_established':False,'full_cohort_performance_claimed':False}
        result['test_sample']['distinct_patients'] = len(set(r['private_patient_id'] for r in test['records']))
        store.json('QWEN_RESULT.json',result)
        (store.local/'QWEN_REPORT.md').write_text(report_markdown(result)); store.publish('QWEN_REPORT.md')
        print('QWEN_RESULT_JSON',json.dumps(result),flush=True)
        print('Fine-tuning, predictions, aggregate report and private Drive readback complete:',store.persistent,flush=True)
        return result
    except Paused as error:
        progress = {'status':'paused_resume_available','run_id':run_id,'reason':str(error),'persistent_checkpoint':str(store.persistent)}
        store.json('PROGRESS.json',progress); print('QWEN_PROGRESS_JSON',json.dumps(progress),flush=True); return progress


def hashlib_sha(text):
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',default='data/raw'); p.add_argument('--storage-root',required=True)
    p.add_argument('--config',default='experiments/qwen_prediction/protocol.json'); p.add_argument('--max-minutes',type=float,default=90)
    p.add_argument('--model-cache')
    a = p.parse_args()
    if not 0 < a.max_minutes <= 180: raise ValueError('Per-invocation compute budget must be in (0,180] minutes')
    result = run(Path.cwd(),a.data_root,a.storage_root,json.loads(Path(a.config).read_text()),max_minutes=a.max_minutes,model_cache=a.model_cache)
    if result['status'] != 'exploratory_pilot_complete': raise SystemExit(75)


if __name__ == '__main__': main()
