import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from go2arx5_vla.sla_adapter.codec_v1 import Codec, Quantiles, pad32, roundtrip_error, TOLERANCES, LABEL_TIMES_S
from go2arx5_vla.sla_adapter.dataset_v1 import images224
from go2arx5_vla.sla_adapter.config_v1 import learning_rates, trainable
from scripts.pi0291_arx5_data.native_dataset import pose10


def poses(rotations, positions):
    a = np.broadcast_to(np.eye(4), (len(positions), 4, 4)).copy()
    a[:, :3, :3] = Rotation.from_rotvec(rotations).as_matrix()
    a[:, :3, 3] = positions
    return a


@pytest.mark.parametrize('dtype', [np.float64, np.float32])
def test_noncommuting_frame_near_pi_absolute_nominal_and_normalization(dtype):
    rng = np.random.default_rng(53)
    mount = poses([[.3, -.8, .4]], [[.1, .2, -.3]])[0]
    state = poses([[1.1, .7, -.3]], [[.4, .6, .9]])[0]
    rotations = Rotation.random(40, random_state=rng).as_rotvec()
    rotations[0] = [np.pi-1e-7, 0, 0]
    action = poses(rotations, rng.normal(size=(40, 3)))
    s10 = pose10(state, .37).astype(dtype)
    a10 = pose10(action, np.linspace(0, 1, 40)).astype(dtype)
    codec = Codec(mount)
    s7, a7 = codec.encode(s10, a10)
    assert s7[6] == s10[9]
    np.testing.assert_array_equal(a7[:, 6], a10[:, 9])
    stats = Quantiles.fit(a7)
    normalized = stats.normalize(a7)
    assert (np.abs(normalized) > 1).any()  # Quantile outliers are not clipped.
    decoded = codec.decode(s10, pad32(stats.unnormalize(normalized)))
    assert all(v <= TOLERANCES[k] for k, v in roundtrip_error(a10, decoded).items())
    # Independent formula in the calibrated mount frame, including LEFT SO(3).
    relative = np.linalg.inv(mount) @ action
    initial = np.linalg.inv(mount) @ state
    np.testing.assert_allclose(a7[:, :3], relative[:, :3, 3]-initial[:3, 3], atol=2e-7)
    np.testing.assert_allclose(Rotation.from_rotvec(a7[:, 3:6]).as_matrix(),
                               relative[:, :3, :3] @ initial[:3, :3].T, atol=2e-7)


def test_padding_has_no_physical_effect_and_model_grip_is_not_clipped():
    s = pose10(np.eye(4), .2)
    c = Codec(np.eye(4))
    a = np.zeros((40, 32))
    a[:, 6] = 1.2
    expected = c.decode(s, a)
    a[:, 7:] = 123
    np.testing.assert_array_equal(expected, c.decode(s, a))
    assert (expected[:, 9] == 1.2).all()
    with pytest.raises(ValueError):
        c.decode(s, np.zeros((50, 32)))
    with pytest.raises(ValueError):
        Codec(None)
    with pytest.raises(ValueError):
        c.encode(np.zeros(10), np.zeros((40, 10)))


def test_constant_quantiles_preserve_values():
    q = Quantiles.fit(np.ones((40, 7)))
    x = np.arange(7.)
    np.testing.assert_allclose(q.unnormalize(q.normalize(x)), x, atol=1e-10)
    assert q.summary(np.ones((2, 7)))['constant_dimensions'] == list(range(7))


def test_mask_and_upstream_resize_contract():
    # A resize spy verifies delegation and missing-camera semantics; production
    # equivalence is checked against the exact pinned client on real images.
    calls = []
    def resize(a, h, w, *, return_mask):
        calls.append((a.copy(), h, w, return_mask))
        return a[:h, :w].copy(), np.ones((h, w), bool)
    body = np.full((256, 256, 3), 27, np.uint8)
    wrist = np.full_like(body, 81)
    out = images224([body, wrist], resize)
    assert len(calls) == 2 and all(c[1:] == (224, 224, True) for c in calls)
    assert list(out['image_mask'].values()) == [True, True, False]
    assert not out['image']['right_wrist_0_rgb'].any()
    assert not out['image_padding_mask']['right_wrist_0_rgb'].any()
    assert out['image']['left_wrist_0_rgb'][0, 0, 0] == 81


def test_schedule_clock_and_positive_allowlist():
    lr = learning_rates()
    np.testing.assert_allclose([lr[0], lr[24], lr[-1]], [2e-6, 5e-5, 1e-5], rtol=0, atol=1e-20)
    assert len(lr) == 500 and len(LABEL_TIMES_S) == 40
    assert LABEL_TIMES_S[-1] == pytest.approx(1.95, abs=1e-15)
    assert trainable(('PaliGemma', 'llm', 'layers', 'attn', 'lora_a'))
    assert trainable(('action_in_proj', 'kernel'))
    assert not trainable(('PaliGemma', 'img', 'lora_a'))
    assert not trainable(('PaliGemma', 'llm', 'layers', 'attn', 'kernel'))
    assert not trainable(('align_projector', 'kernel'))
    assert trainable(('align_projector', 'kernel'), align=True)
