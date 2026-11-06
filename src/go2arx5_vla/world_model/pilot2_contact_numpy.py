"""NumPy inference for the declared partial direct contact heads, no rollout."""
import numpy as np


def export_arrays(state_arrays, horizon_count):
    prefix = 'prefix_mask' in state_arrays
    # A legacy native reader must fail on the new representation, rather than
    # silently evaluating masked-model weights as a full-plan model.
    result = {('prefix_'+k if prefix and k.startswith('network.') else k): v
        for k, v in state_arrays.items()}
    result['horizon_count'] = np.array(horizon_count, dtype=np.int64)
    if prefix: result['architecture_code'] = np.array(1, dtype=np.int64)
    return result


def predict(arrays, inputs):
    x = np.asarray(inputs, np.float64)
    mean, scale = np.asarray(arrays['mean']), np.asarray(arrays['scale'])
    if x.ndim != 2 or x.shape[1:] != mean.shape or scale.shape != mean.shape or np.any(scale <= 0):
        raise ValueError('DIRECT_CONTACT_NUMPY_LAYOUT_REQUIRED')
    count = int(np.asarray(arrays['horizon_count']).item())
    prefix = 'prefix_mask' in arrays
    if prefix:
        mask = np.asarray(arrays['prefix_mask'])
        horizons = np.asarray(arrays['forecast_horizons'])
        if (int(np.asarray(arrays.get('architecture_code', -1)).item()) != 1
                or mask.ndim != 2 or mask.shape[0] != count or not 0 < mask.shape[1] < len(mean)
                or horizons.shape != (count,) or not np.issubdtype(horizons.dtype, np.integer)
                or list(horizons) != sorted(set(horizons.tolist()))
                or np.any(horizons <= 0) or np.any(horizons > mask.shape[1])
                or not np.array_equal(mask, np.arange(mask.shape[1])[None] < horizons[:, None])
                or any(k.startswith('network.') for k in arrays)):
            raise ValueError('EXACT_FORECAST_PREFIX_MASK_AND_DISTINCT_PORTABLE_ARCHITECTURE_REQUIRED')
    elif 'architecture_code' in arrays and int(np.asarray(arrays['architecture_code']).item()) != 0:
        raise ValueError('UNKNOWN_DIRECT_CONTACT_ARCHITECTURE')
    x = (x-mean)/scale
    if prefix:
        x = np.concatenate((np.broadcast_to(x[:, None, :-mask.shape[1]],
            (len(x), count, len(mean)-mask.shape[1])), x[:, None, -mask.shape[1]:]*mask[None]), -1)
    network = 'prefix_network' if prefix else 'network'
    for index in (0, 2, 4):
        x = x@np.asarray(arrays[f'{network}.{index}.weight']).T + arrays[f'{network}.{index}.bias']
        if index != 4: x = np.tanh(x)
    expected = (count, count*10) if prefix else (count*10,)
    if x.shape[1:] != expected or not np.isfinite(x).all():
        raise ValueError('FINITE_PARTIAL_CONTACT_HEADS_REQUIRED')
    if prefix:
        return x.reshape(len(x), count, count, 10)[:, np.arange(count), np.arange(count)]
    return x.reshape(len(x), count, 10)
