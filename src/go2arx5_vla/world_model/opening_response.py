"""Small stationary-opening baseline; internal reference memory is causal.

This is not a closing/contact model, a new Actor, or an M4 parameter update.
The observed initial gap is a floor for this fixed supported-release probe.
"""
import numpy as np
from .actuator_observer import SourceFingerTarget

def reference_curve(initial_target,planned_raw):
    raw=np.asarray(planned_raw,dtype=float)
    if raw.shape!=(8,7)or not np.isfinite(raw).all():raise ValueError('Original planned8x7 candidate required')
    initial=np.asarray(initial_target,dtype=float)
    if initial.shape!=(2,)or not np.isfinite(initial).all()or np.any(initial<0)or np.any(initial>.044):raise ValueError('Causally reconstructed two-finger target required')
    memory=SourceFingerTarget();memory.lower=initial.copy();memory.upper=initial.copy();curve=[]
    for opening in np.r_[raw[:,6],np.repeat(raw[-1,6],8)]:
        for _ in range(25):memory.consume(float(opening))
        curve.append(float(memory.interval()[0].mean()/.044))
    return np.asarray(curve)

def predict_opening(initial_gap,reference,lag):
    if not np.isfinite(initial_gap)or not 0<=initial_gap<=1 or not np.isfinite(lag)or not 0<=lag<=.1:raise ValueError('Physical opening fraction and finite stationary lag required')
    reference=np.asarray(reference,dtype=float)
    if not np.isfinite(reference).all()or np.any(reference<0)or np.any(reference>1+1e-12):raise ValueError('Physical causal limiter references required')
    return np.clip(np.maximum(initial_gap,reference-lag),0,1)

def fit_lag(examples):
    residual=[]
    for initial,reference,truth in examples:
        reference=np.asarray(reference);truth=np.asarray(truth)
        if reference.shape!=(16,)or truth.shape!=(16,)or not np.isfinite(truth).all():raise ValueError('Complete FIT-only16-step examples required')
        # Fixed before evaluation: fit where opening clears the initial gap
        # by .05 fraction, not where contact prevents actual finger movement.
        residual.extend((reference-truth)[reference>initial+.05].tolist())
    if not residual:raise ValueError('No FIT moving-opening observations')
    lag=float(np.median(residual))
    if not 0<=lag<=.1:raise ValueError('Fitted stationary response outside declared model range')
    return lag,len(residual)
