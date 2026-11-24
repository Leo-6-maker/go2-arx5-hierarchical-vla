import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/sla_go2_v18"))
from task_space_intent import direct_fit, geometric_control, propose


def scene():
    actual = np.eye(4)
    ref = np.eye(4)
    ref[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    ref[:3, 3] = [0.03, -0.02, 0.01]
    rgb = dict(
        status="measured",
        center_world_m=[0.0, 0.14, -0.08],
        separation_mm=1.0,
        views={v: dict(bbox_xyxy=[45, 43, 72, 65]) for v in ("body_rgb", "wrist_rgb")},
    )
    world = np.repeat(actual[None], 50, axis=0)
    world[:, :3, 3] = np.arange(50)[:, None] * np.array([0.003, 0.014, 0.0])
    return actual, ref, rgb, world


def test_reanchor_is_right_increment_and_preserves_rigid_motion():
    actual, ref, rgb, world = scene()
    p, _ = propose("REANCHOR", world, actual, ref, rgb)
    assert np.allclose(p[0], ref)
    assert np.allclose(
        np.linalg.solve(p[0], p[-1]), np.linalg.solve(world[0], world[-1])
    )


def test_direct_projection_consumed_frame_and_null_control():
    actual, ref, rgb, world = scene()
    path, _ = direct_fit(world, actual, ref, rgb)
    d = path[:, :3, 3] - ref[:3, 3]
    assert np.allclose(path[0], ref)
    assert np.max(np.linalg.norm(d[:, [0, 2]], axis=1)) <= 0.0025 + 1e-10
    assert np.all(d[:, 1] >= -1e-12) and np.all(
        d[:, 1] <= np.arange(16) * 0.003 + 1e-10
    )
    assert np.isclose(d[-1, 1], 0.02)
    assert (
        np.linalg.norm(path[-1, :3, 3] - geometric_control(actual, ref, rgb)[-1, :3, 3])
        > 1e-4
    )
    for factor in (0.0, -1.0):
        world[:, :3, 3] *= 0.0
        world[:, 0, 3] = factor * np.arange(50) * 0.01
        with pytest.raises(ValueError, match="no forward component"):
            direct_fit(world, actual, ref, rgb)


@pytest.mark.parametrize(
    "status,separation,bbox",
    [
        ("missing_body_rgb", 0.0, [40, 40, 60, 60]),
        ("measured", 6.0, [40, 40, 60, 60]),
        ("measured", 1.0, [0, 30, 55, 80]),
    ],
)
def test_visual_refusal_for_both_planners(status, separation, bbox):
    actual, ref, rgb, world = scene()
    rgb.update(status=status, separation_mm=separation)
    rgb["views"]["body_rgb"]["bbox_xyxy"] = bbox
    with pytest.raises(ValueError, match="vision"):
        direct_fit(world, actual, ref, rgb)
    with pytest.raises(ValueError, match="vision"):
        geometric_control(actual, ref, rgb)


@pytest.mark.parametrize("method", ["DIRECT", "GEOMETRY"])
def test_requery_uses_remaining_episode_displacement_and_fails_closed(method):
    from task_space_intent import ReachEpisode

    actual, ref, rgb, world = scene()
    episode = ReachEpisode("entry", actual, ref, rgb)
    halfway = actual.copy()
    halfway[1, 3] += 0.011
    consumed = ref.copy()
    consumed[1, 3] += 0.012
    raw = world.copy()
    raw[:, 1, 3] += 0.011
    data = {"method": method, "model_world": raw if method == "DIRECT" else None}
    p = episode.plan("entry", halfway, consumed, rgb, **data)
    assert np.allclose(p[0], consumed)
    assert p[-1, 1, 3] == pytest.approx(ref[1, 3] + 0.020)
    with pytest.raises(ValueError, match="reset"):
        episode.plan("different", halfway, consumed, rgb, **data)
    off_axis = halfway.copy()
    off_axis[0, 3] += 0.03
    with pytest.raises(ValueError, match="outside original"):
        episode.plan("entry", off_axis, consumed, rgb, **data)
    lost_object = dict(rgb, center_world_m=[0.03, 0.14, -0.08])
    with pytest.raises(ValueError, match="inconsistent"):
        episode.plan("entry", halfway, consumed, lost_object, **data)
    with pytest.raises(ValueError, match="missing model"):
        episode.plan("entry", halfway, consumed, rgb, method="DIRECT")


def test_world_offset_preserves_world_translation_and_space_rotation():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts / "sla_hierarchical_v16"))
    from world_intent_clock import world_offset

    actual = np.eye(4)
    reference = np.eye(4)
    reference[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    reference[:3, 3] = [0, 0, 0.025]
    proposal = actual.copy()
    proposal[:3, :3] = Rotation.from_euler("x", 45, degrees=True).as_matrix()
    proposal[:3, 3] = [0.02, 0, 0]
    transformed = world_offset(np.stack([actual, proposal]), actual, reference)
    assert np.allclose(transformed[0], reference)
    assert np.allclose(transformed[1, :3, 3], [0.02, 0, 0.025])
    assert np.allclose(transformed[1, :3, :3] @ reference[:3, :3].T, proposal[:3, :3])


def test_grip_units_preserve_their_distinct_domains():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts / "sla_hierarchical_v16"))
    from world_intent_clock import (
        action_fraction_to_width,
        command_width_to_action,
        h40_fraction_to_width,
        sensed_width_to_state,
    )

    assert sensed_width_to_state(0.042) == pytest.approx(42 / 44)
    assert h40_fraction_to_width(42 / 44) == pytest.approx(0.042)
    assert command_width_to_action(0.042) == pytest.approx(1.0)
    assert action_fraction_to_width(42 / 44) == pytest.approx(0.040181818)
    with pytest.raises(ValueError, match="working interval"):
        command_width_to_action(0.044)
