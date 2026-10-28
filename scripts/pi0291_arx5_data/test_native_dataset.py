"""Contract tests only: synthetic rows do not claim physical data qualification."""
import copy
import unittest

import numpy as np

from native_dataset import matrix_from10, pose10, validate_samples


def synthetic_rows():
    rows = []
    for index in range(641):
        snapshot = "contract-test:" + str(index * 25)
        obs = dict(instruction="Contract fixture", actual_tcp=np.eye(4).tolist(), actual_gripper=.5,
                   actual_gripper_speed=0., timestamp_s=index / 20, valid=True)
        for camera in ("body_rgb", "wrist_rgb"):
            obs[camera] = dict(path=camera + "/frame.png", sha256="fixture-not-a-real-hash", frame_index=index,
                               timestamp_s=index / 20, snapshot_id=snapshot, valid=True)
        command = None if index == 640 else dict(actually_committed=True, valid=True, sequence=index,
                                                command_time_s=index / 20, committed=np.eye(4).tolist(), committed_gripper=.5)
        rows.append(dict(policy_obs=obs, snapshot_id=snapshot, supervision=command,
                         causal_history=dict(episode_id="contract-test", segment_id=0, previous_samples=index,
                                             previous_committed=index, reset=False, abort=False, takeover=False)))
    return rows


class ContractTest(unittest.TestCase):
    def test_641_observations_640_commands(self):
        validate_samples(synthetic_rows())

    def test_no_fake_terminal_command(self):
        rows = synthetic_rows()
        rows[-1]["supervision"] = rows[0]["supervision"]
        with self.assertRaises(ValueError):
            validate_samples(rows)

    def test_oracle_and_future_history_and_clock_rejected(self):
        original = synthetic_rows()
        for field in ("object_pose", "teacher_stage", "success", "joint_qpos"):
            rows = copy.deepcopy(original)
            rows[0]["policy_obs"][field] = 1
            with self.assertRaises(ValueError):
                validate_samples(rows)
        rows = copy.deepcopy(original)
        rows[0]["causal_history"]["previous_committed"] = 40
        with self.assertRaises(ValueError):
            validate_samples(rows)
        rows = copy.deepcopy(original)
        rows[10]["policy_obs"]["timestamp_s"] = .1
        with self.assertRaises(ValueError):
            validate_samples(rows)

    def test_segment_boundary_and_invalid_gripper_rejected(self):
        rows = synthetic_rows()
        rows[80]["causal_history"]["segment_id"] = 1
        with self.assertRaises(ValueError):
            validate_samples(rows)
        rows = synthetic_rows()
        rows[80]["policy_obs"]["actual_gripper"] = 1.03
        with self.assertRaises(ValueError):
            validate_samples(rows)

    def test_measured_physical_compliance_not_clipped(self):
        rows = synthetic_rows()
        rows[80]["policy_obs"]["actual_gripper"] = 1.0001
        validate_samples(rows)
        self.assertEqual(rows[80]["policy_obs"]["actual_gripper"], 1.0001)

    def test_row_rot6d_se3_not_translation_only(self):
        anchor = np.eye(4)
        anchor[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        anchor[:3, 3] = [1, 2, 3]
        target = anchor.copy()
        target[:3, 3] += [1, 0, 0]
        relative = np.linalg.inv(anchor) @ target
        value = pose10(relative, .7)
        self.assertTrue(np.allclose(value[:3], [0, -1, 0]))
        self.assertTrue(np.allclose(anchor @ matrix_from10(value), target))
        self.assertEqual(value[-1], .7)


if __name__ == "__main__":
    unittest.main()
