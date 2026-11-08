"""Observable patient grounding and blinded-review preparation, not clinician ratings."""
from __future__ import annotations
import hashlib
import json
import re
import csv
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .agents import citation_checks

ALIASES={'HR':['HR','heart rate','pulse'],'Resp':['Resp','respiratory rate','respiration'],
         'Temp':['Temp','temperature'],'MAP':['MAP','mean arterial pressure'],
         'O2Sat':['O2Sat','SpO2','oxygen saturation'],'SBP':['SBP','systolic blood pressure','systolic'],
         'DBP':['DBP','diastolic blood pressure','diastolic'],'Lactate':['Lactate','lactic acid']}
ANCHOR=re.compile(r'\b(patient|case|observed|measured|recorded|provided|individual|current)\b',re.I)


def sample_validation_endpoints(dataset,manifest,probabilities,seed=42):
    rows=manifest.loc[manifest.split.eq('validation')].sort_values('patient_id')
    candidates=[];offset=0;features=dataset.channels//2
    for row in rows.itertuples():
        n=int(row.n_hours)
        hashed=int(hashlib.sha256(f'{seed}:{row.patient_id}'.encode()).hexdigest()[:16],16)
        idx=offset+hashed%n;start=max(offset,idx-dataset.window+1)
        missing=1-float(dataset.values[start:idx+1,features:].mean())
        candidates.append((idx,float(probabilities[idx]),missing));offset+=n
    ordered=sorted(candidates,key=lambda x:(x[1],x[0]));selected=[]
    for group in np.array_split(np.array(ordered,dtype=float),4):
        if not len(group):continue
        group=sorted(group.tolist(),key=lambda x:(x[2],x[0]))
        for j in np.unique(np.round(np.linspace(0,len(group)-1,min(3,len(group)))).astype(int)):
            selected.append(int(group[j][0]))
    return selected


def grounded_features(text,observations):
    matched=[]
    for feature,value in observations.items():
        aliases=ALIASES.get(feature,[feature])
        for alias in aliases:
            positions=list(re.finditer(r'\b'+re.escape(alias)+r'\b',text,re.I))
            for hit in positions:
                span=text[max(0,hit.start()-35):min(len(text),hit.end()+60)]
                span=re.sub(r'(?<=\d),(?=\d{3}(?:\D|$))','',span)
                nums=[float(x) for x in re.findall(r'(?<![A-Za-z])[-+]?(?:\d+(?:\.\d*)?|\.\d+)',span)]
                if any(abs(x-float(value))<=max(.05,abs(float(value))*.005)+1e-8 for x in nums):
                    matched.append(feature);break
            if feature in matched:break
    return sorted(set(matched))


def audit_cases(cases,ablations=None):
    ablations=ablations or {};supported=[];patient_grounded=0;numeric_match=0;claims_n=0;valid_claims=0;queries=[];align=[];visible=[]
    eligible=sum(bool(case['case']['observations']) for case in cases)
    for case in cases:
        for item in case['trace']:
            if item.get('role')=='planner' and item['response'].get('action')=='search':queries.append(item['response']['query'])
        if case['status']!='supported_research_context':continue
        claims=case['claims'];text=' '.join(x['text'] for x in claims);supported.append(text)
        grounded=grounded_features(text,case['case']['observations'])
        numeric_match+=bool(grounded);patient_grounded+=bool(grounded and ANCHOR.search(text))
        for claim in claims:
            claims_n+=1;valid_claims+=not bool(citation_checks([claim],case['passages'],case['case']))
        top={x['feature'] for x in ablations.get(case['case']['case_id'],[])[:5]}
        if top:
            align.append(len(top.intersection(grounded))/len(top))
            visible.append(len(top.intersection(case['case']['observations']))/len(top))
    normalized=[' '.join(re.findall(r'[a-z0-9]+',x.lower())) for x in supported]
    similarity=None
    if len(supported)>1:
        try:
            matrix=TfidfVectorizer(ngram_range=(1,2),stop_words='english').fit_transform(supported)
            sims=cosine_similarity(matrix);similarity=float(sims[np.triu_indices(len(supported),1)].mean())
        except ValueError:similarity=None
    return {'cases':len(cases),'output_cases':len(supported),'abstained_cases':len(cases)-len(supported),
            'claims':claims_n,'claims_passing_rechecked_citation_numeric_checks':int(valid_claims),
            'cases_with_exact_or_rounded_observation_value_and_feature':int(numeric_match),
            'cases_with_explicit_patient_observation_grounding':int(patient_grounded),
            'cases_with_available_current_observations':int(eligible),
            'patient_grounding_fraction_over_all_cases':patient_grounded/len(cases) if cases else None,
            'patient_grounding_fraction_among_returned_outputs':patient_grounded/len(supported) if supported else None,
            'patient_grounding_fraction_among_cases_with_observations':patient_grounded/eligible if eligible else None,
            'exact_duplicate_output_fraction':(len(normalized)-len(set(normalized)))/len(normalized) if normalized else None,
            'mean_pairwise_output_tfidf_cosine':similarity,'unique_queries':len(set(queries)),'queries':len(queries),
            'mean_grounded_top5_ablation_feature_coverage':float(np.mean(align)) if align else None,
            'mean_top5_ablation_features_visible_to_generator':float(np.mean(visible)) if visible else None,
            'clinical_expert_ratings':{'status':'pending_independent_review','completed':0},
            'scope':'Automated lexical/numeric grounding, redundancy and ablation-alignment proxies; not clinical relevance, causal fidelity, or independent entailment truth',
            'grounding_rule':'Feature alias and matching observed value within a local text span (0.05 absolute or 0.5% rounding tolerance), plus explicit patient/observed-context language for grounded-case count',
            'sampling_limit':'Risk/missingness-stratified development sample, not representative population coverage'}


def write_annotation_pack(output,cases,contexts,seed=42):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(seed);order=rng.permutation(len(cases))
    packet=[];key=[];rating_rows=[]
    for j,k in enumerate(order):
        item=cases[int(k)];rid=f'review_{j+1:03d}'
        packet.append({'review_id':rid,'current_observations':item['case']['observations'],
                       'causal_history':contexts.get(item['case']['case_id'],[]),
                       'explanation':item['answer'],'abstention_reason':item['reason'] if item['answer'] is None else None,
                       'cited_reference_passages':item['passages']})
        key.append({'review_id':rid,'private_case_id':item['case']['case_id']})
        rating_rows.append({'review_id':rid,'reviewer_id':'','clinical_relevance_0_2':'','patient_specificity_0_2':'',
                            'observation_accuracy_0_2':'','evidence_support_0_2':'','unsafe_or_misleading_yes_no':'','comments':''})
    (output/'blinded_cases.json').write_text(json.dumps(packet,indent=2)+'\n')
    (output/'review_key_private.json').write_text(json.dumps(key,indent=2)+'\n')
    with (output/'clinician_ratings.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rating_rows[0]));writer.writeheader();writer.writerows(rating_rows)
    (output/'REVIEW_INSTRUCTIONS.md').write_text('''# Independent explanation review

Cases are randomized; model identity, outcomes, labels and original patient IDs are hidden. Only observations available at the endpoint and cited reference text are provided. Keep review_key_private.json away from reviewers until ratings are locked. Use two independent clinicians when possible; report disagreements and inter-rater agreement after actual ratings exist.

For each ordinal item use 0 = absent/incorrect, 1 = partly satisfied or uncertain, 2 = clearly satisfied. Clinical relevance: does the explanation address this case? Patient specificity: does it meaningfully use its observations rather than generic background? Observation accuracy: are case claims consistent with supplied observed values and time availability? Evidence support: does the cited passage entail each claim? Mark unsafe_or_misleading_yes_no separately and explain concerns. For abstention, assess whether it appropriately acknowledges insufficient information; leave inapplicable accuracy/support cells blank and explain.

Do not infer a diagnosis from the model score. These ratings assess retrospective explanations, not treatment effectiveness. No clinician ratings have been prefilled by software or by another LLM.
''')
