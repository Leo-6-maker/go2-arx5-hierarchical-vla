"""EEF10 bridge through the original processor; no LIBERO or BASE conversion."""
from dataclasses import dataclass

import numpy as np

from ..trajectory.so3 import rot6d_rows_to_matrix, geodesic_angle


def eef10(value, *, horizon=40):
    out = np.asarray(value, dtype=np.float64)
    shape = (10,) if horizon is None else (horizon, 10)
    if out.shape != shape or not np.isfinite(out).all():
        raise ValueError('EXACT_FINITE_EEF10_H40_NO_BASE_OR_TAIL_PADDING')
    if np.any(out[..., 9] < 0) or np.any(out[..., 9] > 1):
        raise ValueError('GRIPPER_0_CLOSED_1_OPEN_NO_CLIP')
    rot6d_rows_to_matrix(out[..., 3:9])
    return out.copy()


@dataclass(frozen=True)
class ModelSpace:
    horizon: int
    padded_dim: int
    inference_steps: int
    active_dim: int = 10

    def __post_init__(self):
        if (type(self.horizon) is not int or self.horizon != 40
                or type(self.padded_dim) is not int or self.padded_dim < 10
                or self.active_dim != 10 or type(self.inference_steps) is not int or self.inference_steps != 4):
            raise ValueError('ORIGINAL_M2048_EEF10_FOUR_STEP_MODEL_SPACE_REQUIRED')

    @classmethod
    def from_metadata(cls, model_config, processor_config):
        p = processor_config['processor_kwargs']
        spec = cls(model_config['action_horizon'], model_config['max_action_dim'],
                   model_config['num_inference_timesteps'])
        if (p['max_action_dim'] != spec.padded_dim or p['max_action_horizon'] != spec.horizon
                or p['use_relative_action'] is not True or p['clip_outliers'] is not False
                or p['use_percentiles'] is not False):
            raise ValueError('ORIGINAL_PROCESSOR_SEMANTICS_REQUIRED')
        return spec


class EEF10Codec:
    def __init__(self, processor, embodiment_tag, space):
        self.processor, self.tag, self.space = processor, embodiment_tag, space
        config = processor.modality_configs[embodiment_tag.value]['action']
        dims = processor.state_action_processor.norm_params[embodiment_tag.value]['action']
        def value(x):return getattr(x, 'value', x)
        formats = [(value(c.rep), value(c.type), value(c.format), c.state_key) for c in config.action_configs or []]
        if (embodiment_tag.value != 'new_embodiment'
                or list(config.modality_keys) != ['eef_9d', 'gripper_position']
                or list(config.delta_indices) != list(range(40))
                or [int(dims[k]['dim']) for k in config.modality_keys] != [9, 1]
                or formats != [('relative', 'eef', 'xyz+rot6d', 'eef_9d'),
                               ('absolute', 'non_eef', 'default', 'gripper_position')]
                or processor.max_action_dim != space.padded_dim
                or processor.max_action_horizon != space.horizon
                or processor.use_relative_action is not True or processor.clip_outliers is not False):
            raise ValueError('EXACT_ORIGINAL_EEF10_PROCESSOR_ROUTE_REQUIRED')

    @staticmethod
    def state(anchor, *, batched=False):
        anchor = eef10(anchor, horizon=None)
        values = dict(eef_9d=anchor[None, :9], gripper_position=anchor[None, 9:])
        return {k: v[None] for k, v in values.items()} if batched else values

    def encode(self, absolute_h40, anchor):
        absolute = eef10(absolute_h40)
        # The original SAP performs the sole absolute->relative conversion.
        normalized = self.processor.state_action_processor.apply_action(
            dict(eef_9d=absolute[:, :9], gripper_position=absolute[:, 9:]),
            self.tag.value, state=self.state(anchor))
        value = np.concatenate([normalized[k] for k in ('eef_9d', 'gripper_position')], axis=-1)
        if value.shape != (40, 10) or not np.isfinite(value).all():
            raise ValueError('INVALID_ORIGINAL_PROCESSOR_OUTPUT')
        padded = np.zeros((1, 40, self.space.padded_dim), dtype=np.float32)
        padded[0, :, :10] = value
        return padded

    def decode(self, model_action, anchor):
        value = np.asarray(model_action)
        if value.shape != (1, 40, self.space.padded_dim) or not np.isfinite(value).all():
            raise ValueError('EXACT_FINITE_MODEL_TENSOR_REQUIRED')
        decoded = self.processor.decode_action(value.copy(), self.tag, self.state(anchor, batched=True))
        return eef10(np.concatenate([decoded[k] for k in ('eef_9d', 'gripper_position')], axis=-1)[0])


def physical_metrics(reconstructed, expert, anchor):
    """Report all H40/prefix8 errors; thresholds belong to a future task card."""
    pred, target, state = eef10(reconstructed), eef10(expert), eef10(anchor, horizon=None)
    xyz = np.linalg.norm(pred[:, :3]-target[:, :3], axis=-1)
    angles = geodesic_angle(rot6d_rows_to_matrix(pred[:, 3:9]), rot6d_rows_to_matrix(target[:, 3:9]))
    grip = np.abs(pred[:, 9]-target[:, 9])
    result = {}
    for view, count in [('H40', 40), ('prefix8', 8)]:
        result[view] = {name: dict(mean=float(np.mean(v[:count])), maximum=float(np.max(v[:count])))
            for name, v in [('XYZ_m', xyz), ('SO3_rad', angles), ('gripper_absolute', grip)]}
    for label, closing in [('close', True), ('open', False)]:
        def events(action):
            values = np.r_[state[9], action[:, 9]]
            hit = ((values[1:] < .5) & (values[:-1] >= .5)) if closing else ((values[1:] >= .5) & (values[:-1] < .5))
            return np.flatnonzero(hit).tolist()
        expected, actual = events(target), events(pred)
        result[label] = dict(expert_indices=expected, reconstructed_indices=actual,
            first_timing_error_s=(actual[0]-expected[0])*.05 if actual and expected else None,
            missing_count=max(0, len(expected)-len(actual)), spurious_count=max(0, len(actual)-len(expected)))
    return result
