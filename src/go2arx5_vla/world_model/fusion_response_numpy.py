"""NumPy-only local TCP response residual; no Torch or native controller imports."""
import numpy as np


def predict(arrays, x, base):
    x=np.asarray(x,dtype=np.float32);base=np.asarray(base,dtype=np.float32)
    if x.ndim!=2 or base.shape!=(len(x),16,3) or arrays['mean'].shape!=(x.shape[1],):
        raise ValueError('CAUSAL_LOCAL_H16_TCP_FEATURES_REQUIRED')
    value=(x-arrays['mean'])/arrays['scale']
    hidden=np.tanh(value@arrays['w0'].T+arrays['b0'])
    result=base+(hidden@arrays['w1'].T+arrays['b1']).reshape(-1,16,3)*np.float32(.001)
    if not np.isfinite(result).all():raise FloatingPointError('NONFINITE_RESPONSE_NO_CLIPPING')
    return result


def candidate_forecasts(arrays,data,meta,arx_forecasts,layout,arm):
    """Keep the past fixed; blind control removes the candidate from both paths."""
    from .fusion_contract import OFFSETS
    from .pilot1_numpy import actor_features
    if arm not in ('ACTION_CONDITIONED','NO_FUTURE_ACTION')or arx_forecasts.shape!=(4,16,106):
        raise ValueError('MATCHED_CAUSAL_H16_FOUR_ACTION_RESPONSE_REQUIRED')
    selected=np.zeros(4,dtype=int)if arm=='NO_FUTURE_ACTION'else np.arange(4)
    x=np.concatenate([actor_features(data,meta,np.array([0]),OFFSETS[a:a+1,:2])for a in selected])
    result=arx_forecasts.copy();sl=layout['tcp_position_m']
    result[:,:,sl]=predict(arrays,x,arx_forecasts[selected,:,sl])
    return result
