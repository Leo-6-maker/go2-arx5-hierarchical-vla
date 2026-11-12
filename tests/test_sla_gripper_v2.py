import numpy as np
import pytest

from go2arx5_vla.sla_adapter.codec_v1 import Codec, pad32
from go2arx5_vla.sla_adapter.gripper_v2 import EPSILON, GripLogitCodec, decode_latent, label_latent, range_audit


def test_logit_labels_preserve_original_and_declared_endpoint_bias():
    labels = np.array([0., 1e-5, .2, .5, .9, 1.])
    before = labels.copy()
    decoded = decode_latent(label_latent(labels))
    np.testing.assert_array_equal(labels, before)
    np.testing.assert_allclose(decoded, np.clip(labels, EPSILON, 1 - EPSILON), rtol=0, atol=1e-15)
    assert np.abs(decoded - labels).max() <= EPSILON + 1e-15


def test_extreme_finite_inference_is_bounded_without_overflow():
    with np.errstate(over='raise', invalid='raise'):
        decoded = decode_latent(np.array([-1e6, -1000., 0., 1000., 1e6]))
    assert np.all((decoded >= 0) & (decoded <= 1))
    assert decoded[2] == .5


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_nonfinite_predictions_still_rejected(bad):
    with pytest.raises(ValueError):
        decode_latent([bad])


def test_invalid_original_labels_are_not_clipped_into_valid_labels():
    with pytest.raises(ValueError):
        label_latent([-1e-6])
    with pytest.raises(ValueError):
        label_latent([1 + 1e-6])


def test_codec_changes_only_action_grip_and_preserves_actual_state():
    state = np.array([.4, 0, .2, 1, 0, 0, 0, 1, 0, 1.0001])
    actions = np.tile(state, (40, 1)); actions[:, 9] = np.linspace(0, 1, 40)
    original = actions.copy()
    s1, a1 = Codec(np.eye(4)).encode(state, actions)
    codec = GripLogitCodec(np.eye(4)); s2, a2 = codec.encode(state, actions)
    np.testing.assert_array_equal(s1, s2)
    np.testing.assert_array_equal(a1[:, :6], a2[:, :6])
    decoded = codec.decode(state, pad32(a2))
    np.testing.assert_allclose(decoded[:, :9], actions[:, :9], rtol=0, atol=1e-12)
    np.testing.assert_allclose(decoded[:, 9], np.clip(actions[:, 9], EPSILON, 1-EPSILON), rtol=0, atol=1e-15)
    np.testing.assert_array_equal(actions, original)


def test_middle_violation_keeps_legal_last_point_separate():
    grip = np.ones(40); grip[13] = 1.0006
    row = range_audit(grip)
    assert row['last_legal'] and row['endpoints_legal'] and not row['full_H40_legal']
    assert row['interior_outside'] == row['above_one'] == 1
    assert row['first_outside'] == row['last_outside'] == row['below_zero'] == 0
