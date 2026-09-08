"""CPU-only checks for Stage41 L0 characterization helpers."""

from __future__ import annotations

import unittest

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import stage41_m2_baseline_profile as PROFILE  # noqa: E402


def _row(offset: float, step: int = 0):
    return {
        "achieved_pos": [offset, 0.0, 0.0],
        "desired_pos": [0.0, 0.0, 0.0],
        "achieved_quat": [0.0, 0.0, 0.0, 1.0],
        "desired_quat": [0.0, 0.0, 0.0, 1.0],
        "sim_time": 0.02 * (step + 1),
        "traj_s": 0.1 + step,
        "traj_s_ref": 0.2 + step,
        "traj_L": 2.0,
        "traj_d_lat": 0.01,
        "traj_timing_err": 0.02,
        "done": False,
        "reset_buf": False,
        "invalid_state": False,
        "joint_limit_violation": False,
        "joint_limit_margin": [0.1, 0.2],
        "termination_code": "NONE",
    }


class Stage41BaselineProfileTest(unittest.TestCase):
    def test_row_metrics_are_descriptive_and_finite(self):
        metrics = PROFILE._row_metrics([_row(0.1), _row(0.2, step=1)], 0.02, "fixture")
        self.assertEqual(metrics["native_tick_count"], 2)
        self.assertAlmostEqual(metrics["eef_position_tracking_m"]["rms"], (0.025**0.5), places=12)
        self.assertEqual(metrics["lifecycle_observations"]["joint_limit_violation_count"], 0)
        self.assertEqual(metrics["reference_progress"]["traj_s_last"], 1.1)

    def test_state_profile_rejects_reordered_observations(self):
        observations = []
        for index, request_id in enumerate(PROFILE.M2_REQUEST_IDS):
            observations.append({
                "request_id": request_id,
                "observation_timestamp": float(index),
                "body_shape": [96, 96, 3],
                "body_dtype": "uint8",
                "body_sha256": str(index) * 64,
                "wrist_shape": [96, 96, 3],
                "wrist_dtype": "uint8",
                "wrist_sha256": str(index + 4) * 64,
                "state_v1": [float(index)] * 10,
                "state_v1_sha256": str(index + 8) * 64,
                "raw_sha256": str(index + 12) * 64,
                "decoded_sha256": str(index + 16) * 64,
            })
        profile = PROFILE._state_profile({"observations": observations})
        self.assertTrue(profile["all_state_hashes_distinct"])
        self.assertEqual(len(profile["state_v1_deltas"]), 3)
        observations[2]["observation_timestamp"] = 0.5
        with self.assertRaises(ValueError):
            PROFILE._state_profile({"observations": observations})

    def test_handoff_profile_keeps_native_and_action_clocks_separate(self):
        initial = {"events": [{"native_handoff_steps": 49, "executor_prefix_steps": 20, "action_sample_dt_s": 0.05,
                                "physics_root_state_unchanged": True, "physics_dof_state_unchanged": True,
                                "episode_length_unchanged": True, "arm_policy_history_unchanged": True,
                                "dog_policy_history_unchanged": True, "reset_buf_unchanged": True,
                                "physics_reset": False, "physics_teleport": False, "policy_history_reset": False}]}
        event = {"native_handoff_steps": 49, "executor_prefix_steps": 20, "action_sample_dt_s": 0.05,
                 "physics_root_state_unchanged": True, "physics_dof_state_unchanged": True,
                 "episode_length_unchanged": True, "arm_policy_history_unchanged": True,
                 "dog_policy_history_unchanged": True, "reset_buf_unchanged": True,
                 "physics_reset": False, "physics_teleport": False, "policy_history_reset": False}
        result = PROFILE._handoff_profile(initial, {"events": [dict(event), dict(event), dict(event)]})
        self.assertEqual(result["native_handoff_steps"], [49, 49, 49])
        self.assertEqual(result["executor_prefix_steps"], [20, 20, 20])
        bad = dict(event)
        bad["executor_prefix_steps"] = 49
        with self.assertRaises(ValueError):
            PROFILE._handoff_profile(initial, {"events": [bad, dict(event), dict(event)]})


if __name__ == "__main__":
    unittest.main()
