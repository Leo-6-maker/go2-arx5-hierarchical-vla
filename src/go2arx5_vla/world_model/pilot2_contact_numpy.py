"""NumPy inference for the declared partial direct contact heads, no rollout."""
import numpy as np


def predict(arrays, inputs):
    x = np.asarray(inputs, np.float64)
    mean, scale = np.asarray(arrays['mean']), np.asarray(arrays['scale'])
    if x.ndim != 2 or x.shape[1:] != mean.shape or scale.shape != mean.shape or np.any(scale <= 0):
        raise ValueError('DIRECT_CONTACT_NUMPY_LAYOUT_REQUIRED')
    x = (x-mean)/scale
    for index in (0, 2, 4):
        x = x@np.asarray(arrays[f'network.{index}.weight']).T + arrays[f'network.{index}.bias']
        if index != 4: x = np.tanh(x)
    count = int(np.asarray(arrays['horizon_count']).item())
    if x.shape[1] != count*10 or not np.isfinite(x).all():
        raise ValueError('FINITE_PARTIAL_CONTACT_HEADS_REQUIRED')
    return x.reshape(len(x), count, 10)
