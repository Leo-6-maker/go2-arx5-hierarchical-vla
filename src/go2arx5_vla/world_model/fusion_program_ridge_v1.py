"""Strong TRAIN-only linear program-response controls in the same causal ABI."""
import numpy as np
from .fusion_program_response_data_v1 import validate,normalizers,FUTURE_SLICE


def fit(data,train,*,no_future=False,ridge=1.):
    validate(data)
    train=np.asarray(train,dtype=int)
    if (not len(train) or np.any(data['split'][train]!='TRAIN')
            or ridge!=1. or type(no_future)is not bool):
        raise ValueError('FIXED_RIDGE_AND_ORIGINAL_TRAIN_ONLY')
    stats=normalizers(data,train)
    z=(data['X'][train]-stats['x_mean'])/stats['x_scale']
    if no_future:z=z.copy();z[:,FUTURE_SLICE]=0.
    centers=[];weights=[]
    for h in range(16):
        observed=data['valid'][train,h]
        if not observed.any():raise ValueError('NO_TRAIN_OBSERVATIONS_AT_DECLARED_HORIZON')
        a=z[observed];center=a.mean(0);a=a-center
        y=(data['Y'][train[observed],h]-stats['y_mean'][h])/stats['y_scale'][h]
        coefficient=a.T@np.linalg.solve(a@a.T+ridge*np.eye(len(a)),y)
        centers.append(center);weights.append(coefficient)
    return dict(**{k:stats[k] for k in ('x_mean','x_scale','y_mean','y_scale')},
        centers=np.stack(centers),weights=np.stack(weights),mask_future=np.array(no_future),
        ridge=np.array(ridge))


def predict(arrays,x):
    z=(np.asarray(x,dtype=float)-arrays['x_mean'])/arrays['x_scale']
    if bool(np.asarray(arrays['mask_future']).item()):z=z.copy();z[:,FUTURE_SLICE]=0.
    delta=np.stack([(z-arrays['centers'][h])@arrays['weights'][h] for h in range(16)],axis=1)
    result=arrays['y_mean']+arrays['y_scale']*delta
    if not np.isfinite(result).all():raise FloatingPointError('FINITE_TRAIN_RIDGE_PREDICTIONS_REQUIRED')
    return result
