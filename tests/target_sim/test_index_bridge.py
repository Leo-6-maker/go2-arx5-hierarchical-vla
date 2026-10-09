"""CPU contracts, including the actual private Gamma/TimeLaw when configured.

Set PI0287_WBC_ROOT to the audited private source; these are no-physics tests.
"""

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from go2arx5_vla.contracts.trajectory import TrajectoryChunkV1
from go2arx5_vla.target_sim.bridge import ExternalEEFReferenceBridge
from go2arx5_vla.target_sim.indexing import TargetActorIndexMap
from go2arx5_vla.trajectory.so3 import quat_to_matrix, rotation_matrix_to_rot6d_rows


def index_map(**changes):
    fields = dict(actor_indices={"robot": [1, 5], "table": [0, 4], "object": [2, 6]},
                  robot_body_indices=np.array([[1, 3, 4], [9, 11, 12]]),
                  robot_dof_indices=np.array([[1, 2, 4], [7, 8, 10]]),
                  body_names=("foot", "arm", "tcp"), dof_names=("leg", "arm", "gripper"),
                  feet_names=("foot",), tcp_name="tcp", leg_dof_names=("leg",),
                  arm_dof_names=("arm",), gripper_dof_names=("gripper",))
    fields.update(changes)
    return TargetActorIndexMap(**fields)


class IndexTests(unittest.TestCase):
    def test_fresh_gather_and_partial_scatter(self):
        indices = index_map()
        indices.validate_counts(8, 14, 12)
        root = np.arange(8 * 13).reshape(8, 13).astype(float)
        dof = np.arange(12 * 2).reshape(12, 2).astype(float)
        body = np.arange(14 * 13).reshape(14, 13).astype(float)
        old = indices.gather_actor(root)
        root[5, 0] = 999
        self.assertNotEqual(old[1, 0], indices.gather_actor(root)[1, 0])
        other_root, other_dof = root[[5, 6]].copy(), dof[[7, 8, 10]].copy()
        indices.scatter_actor(root, np.zeros((1, 13)), env_ids=[0])
        indices.scatter_actor(root, np.zeros((1, 13)), actor="object", env_ids=[0])
        indices.scatter_robot_dofs(dof, np.zeros((1, 3, 2)), env_ids=[0])
        np.testing.assert_array_equal(root[[5, 6]], other_root)
        np.testing.assert_array_equal(dof[[7, 8, 10]], other_dof)
        np.testing.assert_array_equal(indices.gather_robot_bodies(body), body[[[1, 3, 4], [9, 11, 12]]])
        np.testing.assert_array_equal(indices.tcp_indices, [4, 12])
        for changes in (dict(domain="DOMAIN_ENV"), dict(robot_dof_indices=np.array([[1, 2, 4], [1, 8, 10]])),
                        dict(gripper_dof_names=("arm",))):
            with self.assertRaises(ValueError):
                index_map(**changes)
        with self.assertRaises(ValueError):
            indices.validate_counts(6, 14, 12)

    def test_torch_scatter_is_global_not_an_indexing_copy(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        indices = index_map()
        dof = torch.ones(12, 2)
        cached = indices.gather_robot_dofs(dof)
        cached[0].zero_()
        self.assertTrue(bool(dof[1].all()))
        indices.scatter_robot_dofs(dof, torch.zeros(1, 3, 2), env_ids=[0])
        self.assertFalse(bool(dof[1].any()))
        self.assertTrue(bool(dof[7].all()))


@unittest.skipUnless(os.environ.get("PI0287_WBC_ROOT"), "actual private source not configured")
class NativeBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        spec = importlib.util.spec_from_file_location(
            "pi0287_private_trajectory", Path(os.environ["PI0287_WBC_ROOT"]) / "modules" / "trajectory.py")
        cls.native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.native)

    def setUp(self):
        torch = self.torch
        self.wbc = SimpleNamespace(num_envs=2, dt=0.02,
                                   traj_batch=self.native.TrajectoryBatch(2, 128, 64),
                                   arm_goal_pos_world=torch.zeros(2, 3),
                                   arm_goal_quat_world=torch.tensor([[0., 0., 0., 1.]] * 2),
                                   end_effector_state=torch.tensor([[0., 0., 0., 0., 0., 0., 1.]] * 2),
                                   traj_samples=torch.tensor([55., 77.]),
                                   traj_early_term=torch.tensor([True, False]))
        for name in ExternalEEFReferenceBridge._progress_buffers:
            setattr(self.wbc, name, torch.tensor([0., 0.3]))
        self.bridge = ExternalEEFReferenceBridge(self.wbc, self.native, tcp_semantic_id="x5_tcp")

    def chunk(self, name="test", **changes):
        ts = np.arange(40) * 0.05 + 10
        fields = dict(chunk_id=name, source="teacher", observation_timestamp=10., point_timestamps=ts,
                      eef_position=np.column_stack([np.linspace(0, 0.05, 40), np.zeros((40, 2))]),
                      eef_orientation=np.tile([0, 0, np.sin(0.05), np.cos(0.05)], (40, 1)),
                      source_frame="world", tcp_semantic_id="x5_tcp", validity_start=10.,
                      validity_end=12., gripper=np.linspace(0, 1, 40))
        fields.update(changes)
        return TrajectoryChunkV1(**fields)

    def test_native_install_latency_progress_and_hotswap(self):
        receipt = self.bridge.submit(0, self.chunk(), issue_timestamp=10.01, install_timestamp=10.02)
        self.assertEqual(receipt["status"], "INSTALLED")
        self.assertAlmostEqual(float(self.wbc.traj_sim_time[0]), 0.02, places=6)
        self.assertEqual(float(self.wbc.traj_s[0]), 0.)
        self.assertAlmostEqual(float(self.wbc.traj_s[1]), 0.3, places=6)
        self.assertTrue(bool(self.wbc.traj_early_term[0]))
        np.testing.assert_array_equal(self.wbc.traj_samples.numpy(), [55, 77])
        np.testing.assert_allclose(receipt["canonical"]["position"], receipt["installed"]["position"])
        self.assertAlmostEqual(self.bridge.gripper_at(0, 10.4), 8 / 39)
        other = self.wbc.traj_batch.gamma_p[1].clone()
        shifted = self.chunk("hotswap", observation_timestamp=10.4, point_timestamps=np.arange(40)*.05+10.4,
                             validity_start=10.4, validity_end=12.4, eef_position=np.tile([2, 3, 4], (40, 1)))
        self.bridge.submit(0, shifted, issue_timestamp=10.4, install_timestamp=10.4)
        self.assertTrue(self.torch.equal(other, self.wbc.traj_batch.gamma_p[1]))
        self.assertAlmostEqual(float(self.wbc.traj_batch.gamma_p[0, 0, 0]), 2.)

    def test_frames_capacity_rot6d_expiration_cancel_and_partial_reset(self):
        transform = np.eye(4)
        transform[0, 3] = 2
        receipt = self.bridge.submit(0, self.chunk(source_frame="observed_task"), issue_timestamp=10.,
                                     install_timestamp=10., source_to_world=transform)
        self.assertEqual(receipt["status"], "INSTALLED")
        self.assertAlmostEqual(float(self.wbc.traj_batch.gamma_p[0, 0, 0]), 2.)
        bad = self.chunk(eef_rot6d=rotation_matrix_to_rot6d_rows(quat_to_matrix(self.chunk().eef_orientation)))
        receipt = self.bridge.submit(0, bad, issue_timestamp=10., install_timestamp=10.)
        self.assertEqual(receipt["status"], "REJECTED")
        self.assertIsNotNone(receipt["hold_ref_id"])
        np.testing.assert_allclose(self.wbc.traj_batch.gamma_p[0].numpy(), 0)
        self.wbc.traj_batch.max_tl_points = 3
        self.assertEqual(self.bridge.submit(1, self.chunk(), issue_timestamp=10., install_timestamp=10.)["status"], "REJECTED")
        self.wbc.traj_batch.max_tl_points = 64
        self.bridge.submit(1, self.chunk(), issue_timestamp=10., install_timestamp=10.)
        self.assertEqual(self.bridge.boundary(12.1)[0]["status"], "EXPIRED_HOLD")
        self.assertEqual(self.bridge.cancel(0, timestamp=12.1)["status"], "CANCELLED")
        other = self.bridge.active[1]
        self.bridge.reset([0])
        self.assertNotIn(0, self.bridge.active)
        self.assertIs(self.bridge.active[1], other)

    def test_validation_and_command_snapshot(self):
        chunk = self.chunk()
        self.bridge.submit(0, chunk, issue_timestamp=10., install_timestamp=10.)
        chunk.gripper[:] = 1.
        self.assertEqual(self.bridge.gripper_at(0, 10.), 0.)
        for changes in (dict(source_frame="unbound"), dict(tcp_semantic_id="other_tcp"),
                        dict(gripper=np.full(40, 2.)), dict(gripper=None),
                        dict(point_timestamps=np.zeros(40) + 10),
                        dict(point_timestamps=np.arange(40)**2*.001+10)):
            receipt = self.bridge.submit(1, self.chunk(**changes), issue_timestamp=10., install_timestamp=10.)
            self.assertEqual(receipt["status"], "REJECTED", changes)
            self.assertNotIn(1, self.bridge.active)
        self.assertEqual(self.bridge.submit(1, self.chunk(), issue_timestamp=9., install_timestamp=10.)["status"], "REJECTED")
        self.assertEqual(self.bridge.submit(1, self.chunk(), issue_timestamp=10., install_timestamp=13.)["status"], "REJECTED")
        self.wbc.traj_batch.max_gamma_points = 3
        self.assertEqual(self.bridge.submit(1, self.chunk(), issue_timestamp=10., install_timestamp=10.)["status"], "REJECTED")

    def test_pure_rotation_uses_native_SE3_length(self):
        angles = np.linspace(0, .1, 40)
        q = np.column_stack([np.zeros((40, 2)), np.sin(angles/2), np.cos(angles/2)])
        receipt = self.bridge.submit(0, self.chunk(eef_position=np.zeros((40, 3)), eef_orientation=q),
                                     issue_timestamp=10., install_timestamp=10.)
        self.assertEqual(receipt["status"], "INSTALLED")
        self.assertGreater(float(self.wbc.traj_batch.L[0]), .01)
        np.testing.assert_allclose(self.wbc.traj_batch.gamma_p[0].numpy(), 0)


if __name__ == "__main__":
    unittest.main()
