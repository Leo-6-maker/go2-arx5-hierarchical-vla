from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "scripts/sla_hierarchical_v16")
)
from visual_objects import (
    VisualLiftProgress,
    estimate_objects,
    optical_ray,
    triangulate,
)


def scene(offset=0):
    cal = {}
    pair = []
    for i, name in enumerate(("body_rgb", "wrist_rgb")):
        frame = np.eye(4)
        frame[0, 3] = 0.1 * i
        cal[name] = dict(
            K=[[200, 0, 128], [0, 200, 128], [0, 0, 1]],
            T_world_optical=frame,
            timestamp_s=1.0,
        )
        image = np.zeros((256, 256, 3), np.uint8)
        x = 128 - 20 * i + offset * i
        image[124:132, x - 4 : x + 4, 0] = 210
        image[144:152, x - 4 : x + 4, 2] = 210
        pair.append(image)
    return pair, cal


def test_optical_coordinates_recover_both_colors():
    pair, cal = scene()
    got = estimate_objects(*pair, cal, 1.0)
    np.testing.assert_allclose(got["red"]["center_world_m"], [0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(got["blue"]["center_world_m"], [0, 0.1, 1], atol=1e-12)
    assert got["red"]["views"]["body_rgb"]["pixels"] == 64
    frame = np.eye(4)
    frame[:3, :3] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    p, d = optical_ray((frame, np.eye(3)), [0, 0])
    np.testing.assert_allclose(d, [1, 0, 0])


def test_missing_colors_calibration_and_invalid_ray_pairs():
    pair, cal = scene()
    pair[0][:, :, 0] = 0
    pair[0][:2, :2, 0] = 255
    got = estimate_objects(*pair, cal, 1.0)
    assert (
        got["red"]["status"] == "missing_body_rgb"
        and got["blue"]["status"] == "measured"
    )
    with pytest.raises(ValueError, match="asynchronous"):
        estimate_objects(*pair, cal, 1.02)
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])
    assert triangulate((a, up), (b, up))["status"] == "near_parallel"
    assert triangulate((a, up), (b, (up + b) / np.sqrt(2)))["status"] == "behind_camera"
    assert (
        triangulate((a, up), (b + [0, 0.05, 0], (up - b) / np.sqrt(2)))["status"]
        == "ray_separation"
    )


def test_first_anchor_cannot_shift_after_a_missing_first_color():
    state = VisualLiftProgress()

    def obs(r, b):
        return {
            "red": {"center_world_m": None if r is None else [0, 0, r]},
            "blue": {"center_world_m": [0, 0, b]},
        }

    assert state.observe(1, obs(None, 0))["red"]["sampled_height20"] is None
    result = state.observe(1.5, obs(2, 0.020))
    assert (
        result["red"]["sampled_height20"] is None
        and result["blue"]["sampled_height20"] is True
    )
    with pytest.raises(ValueError, match="chronological"):
        state.observe(1.5, obs(None, 0))


def test_opposite_borders_rejected_while_single_edge_remains_visible():
    from visual_objects import color_component

    image = np.zeros((256, 256, 3), dtype=np.int16)
    image[:20, :, 2] = 255
    image[80:96, :30, 2] = 180
    assert color_component(image, 2, reject_spanning=False)["bbox_xyxy"] == [
        0,
        0,
        255,
        19,
    ]
    assert color_component(image, 2, reject_spanning=True)["bbox_xyxy"] == [
        0,
        80,
        29,
        95,
    ]


def test_bbox_iou_distinguishes_missing_from_disjoint():
    from visual_lift_panel import bbox_iou

    assert bbox_iou(None, [0, 0, 1, 1]) is None
    assert bbox_iou([0, 0, 0, 0], [1, 1, 1, 1]) == 0
    assert bbox_iou([0, 0, 1, 1], [1, 1, 1, 1]) == 0.25
