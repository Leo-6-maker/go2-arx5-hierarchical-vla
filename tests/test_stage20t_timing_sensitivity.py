import numpy as np

from scripts.stage20t_timing_sensitivity import (
    _build_run_plan,
    _trajectory_audit,
    scale_timestamps,
    spatial_geometry_hash,
)
from scripts.stage20t_analyze_timing import analyze_trace


def test_timing_scale_changes_only_timestamp_grid():
    timestamps = np.arange(40, dtype=np.float64) * 0.05
    scaled = scale_timestamps(timestamps, 1.5)

    np.testing.assert_allclose(scaled[0], timestamps[0])
    np.testing.assert_allclose(scaled[-1] - scaled[0], 1.5 * (timestamps[-1] - timestamps[0]))
    np.testing.assert_allclose(np.diff(scaled), 1.5 * np.diff(timestamps))
    assert np.all(np.diff(scaled) > 0.0)


def test_spatial_hash_is_invariant_to_time_scale_and_quaternion_sign():
    position = np.stack([np.linspace(0.2, 0.3, 4), np.zeros(4), np.full(4, 0.4)], axis=-1)
    quaternion = np.repeat(np.asarray([[0.0, 0.0, 0.0, 1.0]]), 4, axis=0)
    signed = quaternion.copy()
    signed[1::2] *= -1.0

    assert spatial_geometry_hash(position, quaternion) == spatial_geometry_hash(position, signed)
    audit_a = _trajectory_audit(position, quaternion, np.arange(4) * 0.05, position[0], quaternion[0])
    audit_b = _trajectory_audit(position, quaternion, scale_timestamps(np.arange(4) * 0.05, 2.0), position[0], quaternion[0])
    assert audit_a["spatial_geometry_sha256"] == audit_b["spatial_geometry_sha256"]
    assert audit_b["linear_speed_mps"]["mean"] == audit_a["linear_speed_mps"]["mean"] / 2.0


def test_stage20t_plan_has_three_seeds_for_a_b_and_control():
    plan = _build_run_plan((0.75, 1.0, 1.25, 1.5, 2.0), (13, 14, 15), (13, 14, 15))
    assert len(plan) == 33
    assert sum(item["condition"] == "A_MODEL" and item["scale"] == 1.0 for item in plan) == 3
    assert sum(item["condition"] == "B_GT" and item["scale"] == 1.0 for item in plan) == 3
    assert sum(item["condition"] == "C_STAGE16_CONTROL" for item in plan) == 3
    assert all(not item["derived_timing_diagnostic"] for item in plan if item["scale"] == 1.0)
    assert all(item["derived_timing_diagnostic"] for item in plan if item["scale"] != 1.0)


def test_trace_analysis_separates_first_reset_jump_from_timing_trace():
    rows = [
        {
            "step": 0,
            "sim_time": 0.04,
            "traj_s": 0.06,
            "traj_s_ref": 0.0,
            "traj_timing_err": 0.06,
            "traj_L": 0.06,
            "traj_d_lat": 0.15,
            "desired_pos": [0.3, 0.0, 0.6],
            "desired_quat": [0.0, 0.0, 0.0, 1.0],
            "achieved_pos": [0.2, 0.0, 0.6],
            "achieved_quat": [0.0, 0.0, 0.70710678, 0.70710678],
            "termination_code": "NONE",
            "episode_done": False,
            "reference_complete": False,
        },
        {
            "step": 1,
            "sim_time": 0.06,
            "traj_s": 0.06,
            "traj_s_ref": 0.06,
            "traj_timing_err": 0.0,
            "traj_L": 0.06,
            "traj_d_lat": 0.01,
            "desired_pos": [0.3, 0.0, 0.6],
            "desired_quat": [0.0, 0.0, 0.0, 1.0],
            "achieved_pos": [0.3, 0.0, 0.6],
            "achieved_quat": [0.0, 0.0, 0.0, 1.0],
            "termination_code": "REFERENCE_COMPLETE",
            "episode_done": True,
            "reference_complete": True,
        },
    ]
    result = analyze_trace(rows)
    np.testing.assert_allclose(
        result["initial_runtime_continuity"]["position_discontinuity_m"], 0.1
    )
    assert result["initial_runtime_continuity"]["rotation_discontinuity_rad"] > 1.5
    assert result["termination_code"] == "REFERENCE_COMPLETE"
