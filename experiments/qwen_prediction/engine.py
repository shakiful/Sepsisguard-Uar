"""Qwen's own token head supplies all predictions; LoRA training is resumable."""
from __future__ import annotations
import contextlib
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
import uuid
import numpy as np
import torch
from .core import SYSTEM_PROMPT, digest, save_json, sha_file, metrics


class Paused(RuntimeError): pass


class Store:
    def __init__(self, local, persistent):
        self.local = Path(local); self.persistent = Path(persistent)
        self.local.mkdir(parents=True, exist_ok=True); self.persistent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def relative(name):
        p = Path(name)
        if p.is_absolute() or '..' in p.parts: raise ValueError('Unsafe artifact path')
        return p

    def publish(self, name):
        name = self.relative(name); src = self.local/name; dest = self.persistent/name
        dest.parent.mkdir(parents=True, exist_ok=True); temp = dest.with_suffix(dest.suffix+'.partial')
        shutil.copyfile(src, temp)
        if sha_file(src) != sha_file(temp): raise ValueError('Private Drive readback failed')
        temp.replace(dest)
        if sha_file(src) != sha_file(dest): raise ValueError('Private Drive final readback failed')

    def json(self, name, value):
        name = self.relative(name); save_json(self.local/name, value); self.publish(name)

    def read_json(self, name):
        name = self.relative(name); remote = self.persistent/name
        return json.loads(remote.read_text()) if remote.exists() else None

    def checkpoint(self, state):
        name = f'checkpoints/step-{state["updates"]:06d}-{uuid.uuid4().hex[:10]}.pt'
        path = self.local/name; path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix('.partial'); torch.save(state, temp); temp.replace(path); self.publish(name)
        pointer = {'file':name,'sha256':sha_file(path),'identity':state['identity']}
        self.json('LATEST_TRAINING_CHECKPOINT.json', pointer)

    def restore(self, identity):
        pointer = self.read_json('LATEST_TRAINING_CHECKPOINT.json')
        if pointer is None: return None
        if pointer['identity'] != identity: raise ValueError('Checkpoint belongs to a different experiment')
        name = self.relative(pointer['file']); remote = self.persistent/name
        if sha_file(remote) != pointer['sha256']: raise ValueError('Training checkpoint is corrupt')
        state = torch.load(remote, map_location='cpu', weights_only=True)
        if state['identity'] != identity: raise ValueError('Checkpoint identity differs from pointer')
        return state


def adapter_state(model):
    return {n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}


def set_adapter(model, state):
    names = {n:p for n,p in model.named_parameters() if p.requires_grad}
    if names.keys() != state.keys(): raise ValueError('Adapter parameter names changed')
    with torch.no_grad():
        for name,p in names.items():
            if p.shape != state[name].shape: raise ValueError('Adapter shape changed')
            p.copy_(state[name].to(device=p.device, dtype=p.dtype))


def adapter_digest(state):
    h = hashlib.sha256()
    for name,t in sorted(state.items()):
        h.update(name.encode()); h.update(t.contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def attach_lora(base, config):
    from peft import LoraConfig, get_peft_model
    import re
    total_layers = base.config.text_config.num_hidden_layers; start = total_layers-config['lora_last_layers']
    if start < 0: raise ValueError('Requested LoRA layer range is invalid')
    leaves = {'q_proj','k_proj','v_proj','o_proj','in_proj_qkv','in_proj_z','in_proj_b','in_proj_a','out_proj'}
    targets = []
    for name,module in base.named_modules():
        found = re.search(r'(?:^|\.)language_model\.layers\.(\d+)\.', name)
        if found and int(found.group(1)) >= start and name.rsplit('.',1)[-1] in leaves and isinstance(module,torch.nn.Linear): targets.append(name)
    if not targets: raise RuntimeError('Native Qwen language attention names differ; refusing broad/vision LoRA injection')
    peft = LoraConfig(r=config['lora_rank'],lora_alpha=config['lora_alpha'],lora_dropout=config['lora_dropout'],
                      bias='none',target_modules=targets,task_type='CAUSAL_LM')
    model = get_peft_model(base, peft)
    if config['gradient_checkpointing']:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    trainable = [(n,p.numel()) for n,p in model.named_parameters() if p.requires_grad]
    if not trainable or any('lora_' not in n or 'language_model.layers.' not in n for n,_ in trainable): raise ValueError('Only declared language-model LoRA adapters may be trained')
    return model, targets, trainable


def load_qwen(config, cache):
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    if not torch.cuda.is_available(): raise RuntimeError('The real experiment requires the Colab A100 GPU')
    free,total = torch.cuda.mem_get_info()
    if total/2**30 < 70 or free/2**30 < 64: raise RuntimeError('Select A100 80GB with at least 64 GiB free before loading 27B weights')
    kwargs = {'revision':config['revision'],'cache_dir':str(cache),'trust_remote_code':False,'token':False}
    tokenizer = AutoTokenizer.from_pretrained(config['model_id'], **kwargs)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    digits = [tokenizer.encode(str(i), add_special_tokens=False) for i in (0,1)]
    if any(len(t) != 1 for t in digits) or digits[0] == digits[1]: raise ValueError('Binary class strings must have distinct single-token encodings')
    torch.cuda.reset_peak_memory_stats()
    base = Qwen3_5ForConditionalGeneration.from_pretrained(config['model_id'], dtype=torch.bfloat16,
            device_map={'':0}, use_safetensors=True, attn_implementation='sdpa', **kwargs)
    base.config.use_cache = False
    model, targets, trainable = attach_lora(base, config)
    metadata = {'model_id':config['model_id'],'revision':config['revision'],'dtype':'bfloat16','quantized':False,
                'base_weights_frozen':True,'lora_layers':config['lora_last_layers'],'lora_targets':len(targets),
                'trainable_parameters':sum(n for _,n in trainable),'total_parameters':sum(p.numel() for p in model.parameters()),
                'loaded_peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
                'gpu':torch.cuda.get_device_name(0),'torch_version':str(torch.__version__)}
    print('QWEN_MODEL_METADATA',json.dumps(metadata),flush=True)
    return model, tokenizer, [t[0] for t in digits], metadata


def encode_records(tokenizer, records, cap):
    encoded = []
    for record in records:
        ids = tokenizer.apply_chat_template([{'role':'system','content':SYSTEM_PROMPT},{'role':'user','content':record['text']}],
                tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
        if not ids or len(ids) > cap: raise ValueError(f'Input has {len(ids)} tokens, exceeds {cap}; silent truncation is forbidden')
        encoded.append(ids)
    return encoded


def collate(sequences, pad_id, device):
    longest = max(map(len,sequences)); ids = torch.full((len(sequences),longest),pad_id,dtype=torch.long,device=device)
    mask = torch.zeros_like(ids)
    for j,seq in enumerate(sequences): ids[j,-len(seq):] = torch.tensor(seq,device=device); mask[j,-len(seq):] = 1
    return {'input_ids':ids,'attention_mask':mask}


def binary_logits(model, batch, digits):
    # Keeping only the final-position vocabulary logits avoids a huge sequence x vocab allocation.
    outputs = model(**batch,use_cache=False,logits_to_keep=1)
    return outputs.logits[:,-1,digits].float()


@torch.no_grad()
def score_records(model, encoded, digits, pad_id, *, batch_size=4, disable_adapter=False, store=None, name=None, deadline=None):
    identity = digest({'tokens':encoded,'digits':digits,'disable_adapter':disable_adapter,
                       'adapter_sha256':None if disable_adapter else adapter_digest(adapter_state(model))})
    previous = store.read_json('predictions/'+name+'.json') if store and name else None
    if previous and previous['identity'] != identity: raise ValueError('Cached predictions have different inputs')
    scores = list(previous['scores']) if previous else []; model.eval(); device = next(model.parameters()).device
    if len(scores) > len(encoded): raise ValueError('Prediction cache is too long')
    ctx = model.disable_adapter() if disable_adapter else contextlib.nullcontext()
    with ctx:
        for start in range(len(scores),len(encoded),batch_size):
            if deadline and time.monotonic() >= deadline: raise Paused('Inference budget reached; saved predictions resume')
            batch = collate(encoded[start:start+batch_size],pad_id,device)
            p = torch.softmax(binary_logits(model,batch,digits),dim=-1)[:,1].cpu().numpy()
            if not np.isfinite(p).all(): raise ValueError('Qwen produced nonfinite prediction scores')
            scores.extend(map(float,p))
            if store and name: store.json('predictions/'+name+'.json',{'identity':identity,'scores':scores})
            if start%128 == 0: print(f'Qwen {name or "validation"}: {len(scores)}/{len(encoded)} scored',flush=True)
    return np.asarray(scores)


def train_adapter(model, train_tokens, train_records, val_tokens, val_records, digits, pad_id, config, store, identity, deadline):
    device = next(model.parameters()).device
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters,lr=config['learning_rate'],weight_decay=config['weight_decay'])
    initial = adapter_state(model)
    state = store.restore(identity)
    if state:
        set_adapter(model,state['adapter']); optimizer.load_state_dict(state['optimizer'])
        for values in optimizer.state.values():
            for key,v in values.items():
                if isinstance(v,torch.Tensor) and key != 'step': values[key] = v.to(device)
        torch.set_rng_state(state['torch_rng'])
        if device.type == 'cuda': torch.cuda.set_rng_state_all(state['cuda_rng'])
    else:
        state = {'identity':identity,'epoch':0,'next_example':0,'updates':0,'best_ap':-1.,'best_epoch':None,
                 'best_adapter':{},'history':[],'initial_adapter_sha256':adapter_digest(initial)}
    def checkpoint():
        state.update(adapter=adapter_state(model),optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else [])
        store.checkpoint(state)
    while state['epoch'] < config['epochs']:
        epoch = state['epoch']; order = np.random.default_rng(config['seed']+epoch).permutation(len(train_tokens))
        for start in range(state['next_example'],len(order),config['gradient_accumulation']):
            if time.monotonic() >= deadline: checkpoint(); raise Paused('Fine-tuning budget reached; optimizer/adapter/RNG checkpoint saved')
            group = order[start:start+config['gradient_accumulation']]; model.train(); optimizer.zero_grad(set_to_none=True); loss_sum = 0.
            for index in group:
                logits = binary_logits(model,collate([train_tokens[index]],pad_id,device),digits)
                target = torch.tensor([train_records[index]['label']],device=device)
                loss = torch.nn.functional.cross_entropy(logits,target)/len(group)
                if not torch.isfinite(loss): raise ValueError('Nonfinite fine-tuning loss')
                loss.backward(); loss_sum += float(loss.detach())
            norm = torch.nn.utils.clip_grad_norm_(parameters,1.)
            if not torch.isfinite(norm): raise ValueError('Nonfinite adapter gradient')
            optimizer.step(); state['updates'] += 1; state['next_example'] = min(start+len(group),len(order))
            if state['updates']%4 == 0: print(f'Fine-tune epoch {epoch+1}, update {state["updates"]}, loss {loss_sum:.6f}',flush=True)
            if state['updates']%config['checkpoint_every_updates'] == 0: checkpoint()
        scores = score_records(model,val_tokens,digits,pad_id,batch_size=config['inference_batch_size'])
        y = np.array([r['label'] for r in val_records]); w = np.array([r['weight'] for r in val_records])
        result = metrics(y,scores,w); state['history'].append({'epoch':epoch+1,'validation':result})
        if result['average_precision'] > state['best_ap']:
            state['best_ap'] = result['average_precision']; state['best_adapter'] = adapter_state(model); state['best_epoch'] = epoch+1
        state['epoch'] += 1; state['next_example'] = 0; checkpoint()
        print('COMPLETED_FINE_TUNED_EPOCH',json.dumps(state['history'][-1]),flush=True)
    if not state['best_adapter'] or not state['updates']: raise ValueError('A genuinely optimized adapter is required before testing')
    set_adapter(model,state['best_adapter'])
    final_digest = adapter_digest(adapter_state(model))
    if final_digest == state['initial_adapter_sha256']: raise ValueError('Fine-tuning failed to change the adapter')
    return {'status':'fine_tuning_complete','optimizer_updates':state['updates'],'best_epoch':state['best_epoch'],
            'best_validation_ap':state['best_ap'],'history':state['history'],'initial_adapter_sha256':state['initial_adapter_sha256'],
            'selected_adapter_sha256':final_digest,'train_examples':len(train_records),'epochs_completed':state['epoch']}
