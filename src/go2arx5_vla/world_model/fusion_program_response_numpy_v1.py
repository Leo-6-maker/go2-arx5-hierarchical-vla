"""Portable factual TCP/rotation-vector/finger forecast, with no Torch import."""
import numpy as np
from .fusion_program_response_data_v1 import FEATURE_DIM,FUTURE_SLICE


def predict(arrays,x):
    x=np.asarray(x,dtype=np.float32)
    if x.ndim!=2 or x.shape[1]!=FEATURE_DIM:raise ValueError('CAUSAL_PROGRAM_FEATURE_LAYOUT_REQUIRED')
    z=(x-arrays['x_mean'])/arrays['x_scale']
    if bool(np.asarray(arrays['mask_future']).item()):z=z.copy();z[:,FUTURE_SLICE]=0.
    hidden=np.tanh(z@arrays['w0'].T+arrays['b0'])
    result=arrays['y_mean']+arrays['y_scale']*(hidden@arrays['w1'].T+arrays['b1']).reshape(-1,16,8)
    if not np.isfinite(result).all():raise FloatingPointError('NONFINITE_PROGRAM_PREDICTION')
    return result
