import numpy as np
import pytest

from go2arx5_vla.adapters import GR00TToWBCAdapter, GR00TToWBCAdapterV2, ReferenceAdapterConfig
from go2arx5_vla.trajectory.so3 import rot6d_rows_to_matrix, rotation_matrix_to_rot6d


def adapter():
    return GR00TToWBCAdapter(
        ReferenceAdapterConfig(
            expected_source_frame="wbc_world",
            wbc_frame="wbc_world",
            tcp_semantic_id="arx5_tcp_v0",
            stale_timeout_s=0.1,
        )
    )


def chunk_inputs():
    ts = np.linspace(0.0, 2.0, 40)
    xyz = np.c_[0.2 * ts, 0.03 * np.sin(ts * 5), np.zeros_like(ts)]
    rot = np.repeat(np.eye(3)[None], len(ts), axis=0)
    return ts, xyz, rotation_matrix_to_rot6d(rot)


@pytest.mark.parametrize("age_ms", [0, 20, 50, 100, 200])
def test_latency_ages_are_explicit_and_finite(age_ms):
    ts, xyz, rot6d = chunk_inputs()
    preview, grip = adapter().adapt_xyz_rot6d(
        chunk_id="latency",
        observation_timestamp=0.0,
        point_timestamps=ts,
        eef_xyz=xyz,
        eef_rot6d=rot6d,
        controller_timestamp=age_ms / 1000.0,
    )
    assert preview.xyz_rot6d.shape == (9, 9)
    assert np.isfinite(preview.xyz_rot6d).all()
    assert preview.provenance["command_age_s"] == pytest.approx(age_ms / 1000.0)
    assert preview.provenance["stale"] is (age_ms / 1000.0 > 0.1)
    assert grip is None


def test_gripper_is_returned_as_parallel_channel():
    ts, xyz, rot6d = chunk_inputs()
    preview, gripper = adapter().adapt_xyz_rot6d(
        chunk_id="grip",
        observation_timestamp=0.0,
        point_timestamps=ts,
        eef_xyz=xyz,
        eef_rot6d=rot6d,
        gripper=np.linspace(0.0, 1.0, len(ts)),
        controller_timestamp=0.05,
    )
    assert preview.source_frame == "wbc_world"
    assert gripper is not None
    assert gripper.normalized_command == pytest.approx(0.0)


def test_frame_mismatch_requires_explicit_transform():
    ts, xyz, rot6d = chunk_inputs()
    with pytest.raises(ValueError, match="frame mismatch"):
        GR00TToWBCAdapter(
            ReferenceAdapterConfig("groot_frame", "wbc_world", "arx5_tcp_v0")
        ).adapt_xyz_rot6d(
            chunk_id="bad-frame",
            observation_timestamp=0.0,
            point_timestamps=ts,
            eef_xyz=xyz,
            eef_rot6d=rot6d,
        )


@pytest.mark.parametrize("rotation_rad", [0.0, np.pi / 2])
def test_expected_source_frame_still_applies_explicit_wbc_transform(rotation_rad):
    ts, xyz, rot6d = chunk_inputs()
    c, s = np.cos(rotation_rad), np.sin(rotation_rad)
    transform = np.eye(4)
    transform[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    transform[:3, 3] = [0.3, -0.2, 0.1]
    kwargs = dict(chunk_id="frame", observation_timestamp=0.0,
                  point_timestamps=ts, eef_xyz=xyz, eef_rot6d=rot6d,
                  controller_timestamp=0.05, gripper=np.linspace(0, 1, len(ts)))
    source_preview, source_grip = adapter().adapt_xyz_rot6d(**kwargs)
    mapped, grip = GR00TToWBCAdapterV2(ReferenceAdapterConfig(
        "source_world", "wbc_world", "arx5_tcp_v0", frame_transform=transform
    )).adapt_xyz_rot6d(**kwargs)
    assert mapped.source_frame == "wbc_world"
    assert np.allclose(mapped.eef_position,
                       source_preview.eef_position @ transform[:3, :3].T + transform[:3, 3])
    assert np.allclose(rot6d_rows_to_matrix(mapped.eef_rot6d),
                       transform[:3, :3] @ rot6d_rows_to_matrix(source_preview.eef_rot6d))
    assert grip.normalized_command == source_grip.normalized_command
    assert mapped.provenance['reference_adapter_version'] == 'FRAME_CORRECTED_REFERENCE_V2'


def test_v2_same_frame_keeps_legacy_preview_time_and_gripper_values():
    ts, xyz, rot6d = chunk_inputs()
    kwargs = dict(chunk_id='same-frame', observation_timestamp=0.0,
                  point_timestamps=ts, eef_xyz=xyz, eef_rot6d=rot6d,
                  controller_timestamp=0.05, gripper=np.linspace(0, 1, len(ts)))
    old_preview, old_grip = adapter().adapt_xyz_rot6d(**kwargs)
    new_preview, new_grip = GR00TToWBCAdapterV2(adapter().config).adapt_xyz_rot6d(**kwargs)
    assert np.array_equal(old_preview.xyz_rot6d, new_preview.xyz_rot6d)
    assert old_preview.anchor_timestamp == new_preview.anchor_timestamp
    assert old_preview.source_frame == new_preview.source_frame
    assert old_grip == new_grip


def test_rotation_is_not_linear_rot6d_interpolation():
    ts = np.array([0.0, 1.0])
    xyz = np.zeros((2, 3))
    c, s = np.cos(np.pi / 2), np.sin(np.pi / 2)
    r1 = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
    rot = rotation_matrix_to_rot6d(np.asarray([np.eye(3), r1]))
    preview, _ = adapter().adapt_xyz_rot6d(
        chunk_id="slerp",
        observation_timestamp=0.0,
        point_timestamps=ts,
        eef_xyz=xyz,
        eef_rot6d=rot,
    )
    # Preview starts at the first positive spatial offset rather than the
    # anchor. Every decoded point must nevertheless remain a proper rotation;
    # this guards against linear rot6d interpolation.
    assert np.isfinite(preview.eef_rot6d).all()
    assert np.allclose(np.linalg.det(rot6d_rows_to_matrix(preview.eef_rot6d)), 1.0, atol=1e-10)
