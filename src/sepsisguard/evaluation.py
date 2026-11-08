"""Frozen retrospective scoring with whole-patient bootstrap and explicit B gate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss,log_loss
from .training import atomic_json


def digest_json(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def evaluation_fingerprint(project):
    project=Path(project);paths=[project/'pyproject.toml',project/'configs/evaluation_protocol.json']
    paths+=sorted((project/'src/sepsisguard').rglob('*.py'))+sorted((project/'scripts').glob('*.py'))
    h=hashlib.sha256()
    for p in paths:
        h.update(p.relative_to(project).as_posix().encode()+b'\0'+p.read_bytes())
    return h.hexdigest()


def freeze_protocol(path,payload):
    if payload.get('stage')!='frozen_before_test_B' or payload.get('test_split')!='test_B':
        raise ValueError('A pre-test frozen protocol is required')
    document={'protocol':payload,'sha256':digest_json(payload),'frozen_at':datetime.now(timezone.utc).isoformat()}
    path=Path(path)
    if path.exists():
        prior=read_protocol(path)
        if prior['sha256']!=document['sha256']:raise ValueError('Cannot overwrite a different frozen protocol')
        return prior
    atomic_json(path,document)
    return read_protocol(path)


def read_protocol(path,manifest_sha256=None):
    document=json.loads(Path(path).read_text());p=document['protocol']
    if document['sha256']!=digest_json(p) or p.get('stage')!='frozen_before_test_B' or p.get('test_split')!='test_B':
        raise ValueError('Frozen evaluation protocol is missing or changed')
    if manifest_sha256 is not None and p.get('manifest_sha256')!=manifest_sha256:
        raise ValueError('Evaluation protocol belongs to a different patient partition')
    return document


def validate_vectors(y,p,lengths):
    y=np.asarray(y);p=np.asarray(p,dtype=float);lengths=np.asarray(lengths,dtype=int)
    if y.ndim!=1 or p.shape!=y.shape or not len(y) or not np.isin(y,[0,1]).all():raise ValueError('Binary aligned endpoint labels required')
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('Finite probabilities required')
    if lengths.ndim!=1 or np.any(lengths<=0) or lengths.sum()!=len(y):raise ValueError('Patient lengths must partition every endpoint exactly once')
    return y.astype(np.int8),p,lengths


def hourly_utility_weights(labels):
    """Vectorized hourly yes/no utility, tested against the unmodified official code.

    Labels are ALREADY shifted. argmax(label)+6 recovers the scorer's onset
    convention; no further target shift is made. No clinical onset is inferred.
    """
    labels=np.asarray(labels);n=len(labels);yes=np.zeros(n);no=np.zeros(n)
    if not np.any(labels):yes[:]=-.05;return yes,no
    onset=int(np.argmax(labels))+6;t=np.arange(n)-onset
    early=t<=-6;active=t<=3
    yes[active&early]=np.maximum((t[active&early]+12)/6,-.05)
    yes[active&~early]=(3-t[active&~early])/9
    no[active&~early]=-2*(t[active&~early]+6)/9
    return yes,no


def cohort_utility_weights(y,lengths):
    yes=np.zeros(len(y));no=np.zeros(len(y));offset=0
    for n in lengths:
        yes[offset:offset+n],no[offset:offset+n]=hourly_utility_weights(y[offset:offset+n]);offset+=n
    return yes,no


def validation_thresholds(y,p,lengths,target_recall=.8,utility_quantiles=51):
    y,p,lengths=validate_vectors(y,p,lengths)
    positive=np.sort(p[y==1])[::-1]
    if not len(positive):raise ValueError('Validation positives required to select operating points')
    recall_cut=float(positive[max(0,int(np.ceil(target_recall*len(positive)))-1)])
    yes,no=cohort_utility_weights(y,lengths);gain=yes-no
    cuts=np.unique(np.r_[0.,np.quantile(p,np.linspace(0,1,utility_quantiles)),np.nextafter(1.,2.)])
    utility=np.array([np.sum(no)+np.sum(gain[p>=cut]) for cut in cuts])
    best=float(cuts[np.flatnonzero(np.isclose(utility,utility.max(),rtol=0,atol=1e-10))[-1]])
    return {'calibrated_0_5':.5,'validation_recall_80':recall_cut,'validation_utility':best}


def calibration_bins(y,p,bins=10):
    edges=np.linspace(0,1,bins+1);idx=np.minimum((p*bins).astype(int),bins-1);rows=[];ece=0.
    for k in range(bins):
        mask=idx==k;n=int(mask.sum())
        mean=float(np.mean(p[mask])) if n else None;observed=float(np.mean(y[mask])) if n else None
        if n:ece+=n*abs(mean-observed)/len(y)
        rows.append({'lower':float(edges[k]),'upper':float(edges[k+1]),'hours':n,'mean_risk':mean,'observed_fraction':observed})
    return float(ece),rows


class PatientMetrics:
    def __init__(self,y,p,lengths):
        self.y,self.p,self.lengths=validate_vectors(y,p,lengths);self.n=len(self.lengths)
        self.starts=np.r_[0,np.cumsum(self.lengths)[:-1]];self.groups=np.repeat(np.arange(self.n),self.lengths)
        order=np.argsort(-self.p,kind='stable');self.sy=self.y[order];self.sg=self.groups[order]
        ordered=self.p[order];self.ends=np.r_[np.flatnonzero(np.diff(ordered)!=0),len(ordered)-1]
        self.pos=np.bincount(self.groups,weights=self.y,minlength=self.n);self.neg=self.lengths-self.pos
        self.brier=np.bincount(self.groups,weights=(self.p-self.y)**2,minlength=self.n)
        eps=np.finfo(float).eps;clipped=np.clip(self.p,eps,1-eps)
        self.loss=np.bincount(self.groups,weights=-(self.y*np.log(clipped)+(1-self.y)*np.log1p(-clipped)),minlength=self.n)
        self.yes,self.no=cohort_utility_weights(self.y,self.lengths)
        self.inaction=np.add.reduceat(self.no,self.starts)
        self.oracle=np.add.reduceat(np.maximum(self.yes,self.no),self.starts)

    def weighted(self,counts):
        counts=np.asarray(counts,dtype=float)
        if counts.shape!=(self.n,) or np.any(counts<0):raise ValueError('One nonnegative multiplicity per patient required')
        positive=float(counts@self.pos);negative=float(counts@self.neg);total=positive+negative
        w=counts[self.sg];tp=np.cumsum(w*self.sy)[self.ends];fp=np.cumsum(w*(1-self.sy))[self.ends]
        if positive>0 and negative>0:
            tpr=np.r_[0.,tp/positive];fpr=np.r_[0.,fp/negative]
            auc=float(np.sum(np.diff(fpr)*(tpr[1:]+tpr[:-1])/2))
            precision=np.divide(tp,tp+fp,out=np.zeros_like(tp),where=tp+fp>0)
            ap=float(np.sum(np.diff(np.r_[0.,tp])*precision)/positive)
        else:auc=ap=None
        return {'roc_auc':auc,'average_precision':ap,'brier':float(counts@self.brier/total),'log_loss':float(counts@self.loss/total)}

    def point(self):
        r=self.weighted(np.ones(self.n));ece,bins=calibration_bins(self.y,self.p)
        r.update({'ece_10_equal_width':ece,'reliability_bins':bins,'positive_fraction':float(self.y.mean()),'hours':len(self.y),'patients':self.n})
        return r

    def operating_arrays(self,threshold,refractory=6):
        alert=self.p>=threshold;g=self.groups
        tp=np.bincount(g,weights=alert*self.y,minlength=self.n);fp=np.bincount(g,weights=alert*(1-self.y),minlength=self.n)
        patient_alert=np.zeros(self.n);by_label=np.zeros(self.n);episodes=np.zeros(self.n);lead=[];left=0
        for j,(start,n) in enumerate(zip(self.starts,self.lengths)):
            y=self.y[start:start+n];alarm=np.flatnonzero(alert[start:start+n]);positive=np.flatnonzero(y)
            patient_alert[j]=bool(len(alarm))
            last=-refractory
            for t in alarm:
                if t-last>=refractory:episodes[j]+=1;last=t
            if len(positive):
                first=int(positive[0]);left+=first==0
                if len(alarm):by_label[j]=alarm[0]<=first;lead.append(first-int(alarm[0]))
        return {'tp':tp,'fp':fp,'fn':self.pos-tp,'tn':self.neg-fp,'patient_alert':patient_alert,'by_label':by_label,'episodes':episodes,
                'septic':(self.pos>0).astype(float),'nonseptic':(self.pos==0).astype(float),'lead':lead,'left_censored':left}

    def operating(self,arrays,counts=None):
        c=np.ones(self.n) if counts is None else np.asarray(counts,dtype=float)
        tp,fp,fn,tn=(float(c@arrays[k]) for k in ('tp','fp','fn','tn'))
        def ratio(a,b):return float(a/b) if b else None
        septic=arrays['septic'];non=arrays['nonseptic'];patient_alert=arrays['patient_alert']
        return {'tp':tp,'fp':fp,'fn':fn,'tn':tn,'sensitivity':ratio(tp,tp+fn),'precision':ratio(tp,tp+fp),'specificity':ratio(tn,tn+fp),
                'f1':ratio(2*tp,2*tp+fp+fn),'patient_ever_sepsis_sensitivity':ratio(c@(septic*patient_alert),c@septic),
                'detected_by_first_positive_label_fraction':ratio(c@arrays['by_label'],c@septic),
                'nonseptic_patients_with_any_alarm_fraction':ratio(c@(non*patient_alert),c@non),
                'positive_flags_per_100_nonseptic_hours':ratio(100*(c@(non*arrays['fp'])),c@(non*self.lengths)),
                'alarm_episodes_per_100_nonseptic_hours_6h_refractory':ratio(100*(c@(non*arrays['episodes'])),c@(non*self.lengths))}

    def utility_array(self,threshold):
        return np.add.reduceat(self.no+(self.p>=threshold)*(self.yes-self.no),self.starts)

    def utility(self,observed,counts=None):
        c=np.ones(self.n) if counts is None else counts;den=float(c@(self.oracle-self.inaction))
        return float(c@(observed-self.inaction)/den) if den else None

    def bootstrap(self,thresholds,repetitions=500,seed=42):
        rng=np.random.default_rng(seed);draws=[]
        op=self.operating_arrays(thresholds['validation_recall_80']);u=self.utility_array(thresholds['validation_utility'])
        for _ in range(repetitions):
            counts=np.bincount(rng.integers(0,self.n,size=self.n),minlength=self.n)
            r=self.weighted(counts);a=self.operating(op,counts)
            r.update({'sensitivity_at_validation_recall_80':a['sensitivity'],'precision_at_validation_recall_80':a['precision'],
                      'normalized_utility_at_validation_utility':self.utility(u,counts)})
            draws.append(r)
        intervals={}
        for key in draws[0]:
            values=[r[key] for r in draws if r[key] is not None and np.isfinite(r[key])]
            intervals[key]={'lower':float(np.quantile(values,.025)),'upper':float(np.quantile(values,.975)),'valid_draws':len(values)} if values else None
        return intervals,draws


def evaluate_predictions(y,p,lengths,thresholds):
    m=PatientMetrics(y,p,lengths);r=m.point();operating={}
    for name,threshold in thresholds.items():
        arrays=m.operating_arrays(threshold);q=m.operating(arrays);q['normalized_challenge_utility']=m.utility(m.utility_array(threshold))
        q['threshold']=threshold;q['positive_label_at_first_hour_patients']=arrays['left_censored']
        q['median_hours_before_first_positive_label_among_detected_septic_patients']=float(np.median(arrays['lead'])) if arrays['lead'] else None
        operating[name]=q
    r['operating_points']=operating
    return r
