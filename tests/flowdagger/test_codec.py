from types import SimpleNamespace

import numpy as np
import pytest

from go2arx5_vla.flowdagger.codec import EEF10Codec, ModelSpace, eef10, physical_metrics
from go2arx5_vla.task_data.codec import pose10, relative_action, decode


def action():
    return np.tile([.3, .1, .5, 1, 0, 0, 0, 1, 0, .8], (40, 1)).astype(float)


class ToySAP:
    norm_params = {'new_embodiment': {'action': {'eef_9d': {'dim': 9}, 'gripper_position': {'dim': 1}}}}
    calls = 0

    def apply_action(self, values, tag, state):
        self.calls += 1
        anchor = np.eye(4)
        from go2arx5_vla.trajectory.so3 import rot6d_rows_to_matrix
        anchor[:3, 3] = state['eef_9d'][0, :3]
        anchor[:3, :3] = rot6d_rows_to_matrix(state['eef_9d'][0, 3:])
        targets = []
        for row, grip in zip(values['eef_9d'], values['gripper_position'][:, 0]):
            matrix = np.eye(4);matrix[:3, 3] = row[:3];matrix[:3, :3] = rot6d_rows_to_matrix(row[3:])
            targets.append((matrix, grip))
        relative = relative_action(targets, anchor)
        return dict(eef_9d=relative[:, :9], gripper_position=relative[:, 9:])


class ToyProcessor:
    max_action_dim, max_action_horizon = 12, 40
    use_relative_action, clip_outliers = True, False
    modality_configs = {'new_embodiment': {'action': SimpleNamespace(
        modality_keys=['eef_9d', 'gripper_position'], delta_indices=list(range(40)),
        action_configs=[SimpleNamespace(rep='relative',type='eef',format='xyz+rot6d',state_key='eef_9d'),
                        SimpleNamespace(rep='absolute',type='non_eef',format='default',state_key='gripper_position')])}}

    def __init__(self):
        self.state_action_processor = ToySAP()

    def decode_action(self, value, tag, state):
        from go2arx5_vla.trajectory.so3 import rot6d_rows_to_matrix
        anchor = np.eye(4)
        anchor[:3, 3] = state['eef_9d'][0, 0, :3]
        anchor[:3, :3] = rot6d_rows_to_matrix(state['eef_9d'][0, 0, 3:])
        physical = np.stack([pose10(*decode(row, anchor)) for row in value[0, :, :10]])
        return dict(eef_9d=physical[None, :, :9], gripper_position=physical[None, :, 9:])


def test_actual_metadata_schema_not_libero_dimension_assumption():
    model = dict(action_horizon=40, max_action_dim=132, num_inference_timesteps=4)
    p = dict(max_action_dim=132, max_action_horizon=40, use_relative_action=True,
             clip_outliers=False, use_percentiles=False)
    assert ModelSpace.from_metadata(model, dict(processor_kwargs=p)).active_dim == 10
    for changed in (dict(max_action_dim=10), dict(max_action_horizon=16), dict(clip_outliers=True)):
        with pytest.raises(ValueError):
            ModelSpace.from_metadata(model, dict(processor_kwargs=dict(p, **changed)))
    with pytest.raises(ValueError):ModelSpace(40, 132, 8)


def test_single_relative_transform_rotated_nonzero_anchor():
    processor = ToyProcessor()
    codec = EEF10Codec(processor, SimpleNamespace(value='new_embodiment'), ModelSpace(40, 12, 4))
    anchor = np.array([.2, -.1, .4, 0, -1, 0, 1, 0, 0, .8])
    target = action();saved = target.copy()
    encoded = codec.encode(target, anchor)
    assert processor.state_action_processor.calls == 1
    assert encoded.shape == (1, 40, 12)
    assert (encoded[..., 10:] == 0).all()
    np.testing.assert_allclose(codec.decode(encoded, anchor), target, atol=1e-7)
    np.testing.assert_array_equal(target, saved)


@pytest.mark.parametrize('bad', [np.zeros((8, 10)), np.zeros((40, 13)), np.full((40, 10), np.nan)])
def test_no_short_tail_base_padding_or_nonfinite(bad):
    with pytest.raises(ValueError):eef10(bad)


def test_physical_errors_include_rotation_gripper_and_timing():
    target = action();target[10:, 9] = .1
    pred = target.copy();pred[:, 0] += .02;pred[10:12, 9] = .8
    pred[:, 3:9] = [0, -1, 0, 1, 0, 0]
    result = physical_metrics(pred, target, action()[0])
    assert result['H40']['XYZ_m']['mean'] == pytest.approx(.02)
    assert result['H40']['SO3_rad']['mean'] == pytest.approx(np.pi/2)
    assert result['close']['first_timing_error_s'] == pytest.approx(.1)
    assert result['prefix8']['gripper_absolute']['maximum'] == 0


def test_no_gripper_clipping_or_libero_sign_conversion():
    bad = action();bad[:, 9] = -1
    with pytest.raises(ValueError):eef10(bad)
    np.testing.assert_array_equal(eef10(action())[:, 9], action()[:, 9])


def test_processor_representation_cannot_silently_switch():
    import copy
    p=ToyProcessor();p.modality_configs=copy.deepcopy(p.modality_configs)
    p.modality_configs['new_embodiment']['action'].action_configs[0].rep='absolute'
    with pytest.raises(ValueError):EEF10Codec(p,SimpleNamespace(value='new_embodiment'),ModelSpace(40,12,4))
