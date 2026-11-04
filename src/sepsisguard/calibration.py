"""Frozen-model sigmoid calibration using A/calibration only."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
from sklearn.metrics import brier_score_loss, log_loss
import torch
from torch.utils.data import DataLoader


@dataclass
class SigmoidCalibrator:
    slope: float
    intercept: float
    provenance: dict

    @classmethod
    def fit(cls, logits, labels, *, fit_split, provenance):
        if fit_split != 'A/calibration':
            raise ValueError('Calibration fit must use A/calibration only')
        x, y = np.asarray(logits, dtype=float), np.asarray(labels, dtype=int)
        if x.ndim != 1 or len(x) != len(y) or not np.isfinite(x).all() or set(np.unique(y)) != {0, 1}:
            raise ValueError('Finite logits and both calibration classes are required')
        def objective(theta):
            z=theta[0]*x+theta[1]
            residual=expit(z)-y
            return float(np.mean(np.logaddexp(0,z)-y*z)), np.array([np.mean(residual*x),np.mean(residual)])
        prevalence=float(y.mean())
        initial=[1.0,float(np.log(prevalence/(1-prevalence))-x.mean())]
        fitted=minimize(objective,initial,jac=True,method='L-BFGS-B',bounds=[(0.0,100.0),(-30.0,30.0)],
                        options={'maxiter':1000,'ftol':1e-12,'gtol':1e-9})
        if not fitted.success:
            raise ValueError('Calibration optimizer did not converge')
        slope,intercept=map(float,fitted.x)
        calibrated = expit(slope*x+intercept)
        return cls(slope, intercept, {**provenance, 'method':'nonnegative_sigmoid_LBFGSB','flat_calibrator':slope<=1e-8, 'fit_split':fit_split,
                   'hours':len(y),'positive_hours':int(y.sum()),
                   'fit_brier':float(brier_score_loss(y,calibrated)),
                   'fit_log_loss':float(log_loss(y,calibrated,labels=[0,1])),
                   'metric_scope':'Calibration-fit diagnostics, not independent evaluation'})

    def transform(self, logits):
        return expit(self.slope*np.asarray(logits,dtype=float)+self.intercept)

    def save(self, path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        temp=path.with_suffix('.partial')
        temp.write_text(json.dumps(self.__dict__,indent=2,allow_nan=False)+'\n');temp.replace(path)

    @classmethod
    def load(cls,path):
        return cls(**json.loads(Path(path).read_text()))


@torch.no_grad()
def collect_logits(model,dataset,device,batch_size=1024):
    model.eval();values=[];labels=[]
    for batch in DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=0):
        values.append(model(batch['x'].to(device),batch['length']).float().cpu().numpy())
        labels.append(batch['y'].numpy())
    return np.concatenate(values),np.concatenate(labels)


def predictive_entropy(probability):
    p=float(np.clip(probability,1e-12,1-1e-12))
    return float(-(p*np.log2(p)+(1-p)*np.log2(1-p)))
