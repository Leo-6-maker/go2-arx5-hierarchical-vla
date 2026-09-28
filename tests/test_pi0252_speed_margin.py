import unittest
import copy

import numpy as np
import torch

from go2arx5_vla.contracts.trajectory import TrajectoryChunkV1
from go2arx5_vla.safety.trajectory_gate import TrajectorySafetyGateV1
from pi0252_speed_margin import LAMBDA, MARGIN_Z, margin_from_peak_z, peak_speed_margin
from pi0251_ce_train import _verify_paired_identity
import pi0252_step_rng


def _mask():
    mask = torch.zeros((1, 40, 132), dtype=torch.float32)
    mask[..., :10] = 1.0
    return mask


def _positions(speed_mps=1.0):
    t = torch.arange(40, dtype=torch.float32) * 0.05
    xyz = torch.zeros((1, 40, 3), dtype=torch.float32)
    xyz[0, :, 0] = t * speed_mps
    return xyz


class PeakSpeedMarginTests(unittest.TestCase):
    def test_pair_mismatch_is_rejected_before_backward_or_optimizer(self):
        expected = {
            "source": {"episode_index": 11, "observation_index": 0},
            "key": "PI0252|22301|stage=M64_AG|optimizer_step=1",
            "native_tau_identity": {"sha256": "tau", "dtype": "torch.float32", "device": "cuda:0"},
            "native_epsilon_sha256": "epsilon",
            "stock_initial_noise": {"sha256": "noise", "dtype": "torch.float32", "device": "cuda:0"},
        }
        for field in expected:
            counters = {"backward": 0, "optimizer": 0}
            actual = copy.deepcopy(expected)
            actual[field] = {"mismatch": True}

            def execute_step():
                _verify_paired_identity(pi0252_step_rng, actual, expected)
                counters["backward"] += 1
                counters["optimizer"] += 1

            with self.assertRaises(RuntimeError):
                execute_step()
            self.assertEqual(counters, {"backward": 0, "optimizer": 0})

    def test_speed_matches_frozen_gate_calculation(self):
        dt = torch.full((39,), 0.05)
        xyz = _positions(2.5)
        margin, _, _, speed = peak_speed_margin(xyz, dt, _mask())
        timestamps = np.arange(40, dtype=np.float64) * 0.05
        quaternion = np.tile([0.0, 0.0, 0.0, 1.0], (40, 1))
        chunk = TrajectoryChunkV1(
            chunk_id="cpu-fixture", source="PI0252", observation_timestamp=0.0,
            point_timestamps=timestamps, eef_position=xyz[0].numpy(),
            eef_orientation=quaternion, source_frame="world", tcp_semantic_id="fixture",
            validity_start=0.0, validity_end=float(timestamps[-1]),
        )
        gate = TrajectorySafetyGateV1().evaluate(chunk, now=float(timestamps[-1]))
        gate_speed = gate.metrics["linear_speed_mps"]["max"]
        self.assertAlmostEqual(float(speed), gate_speed, delta=1e-5)
        self.assertAlmostEqual(float(margin), max((gate_speed / 2) ** 2 - MARGIN_Z, 0.0), delta=1e-5)

    def test_margin_sides_boundary_and_first_tied_segment(self):
        dt = torch.full((39,), 0.05)
        below, *_ = peak_speed_margin(_positions(1.79), dt, _mask())
        above, *_ = peak_speed_margin(_positions(1.81), dt, _mask())
        exact_dt = torch.full((39,), 0.25)
        exact_xyz = torch.zeros((1, 40, 3), dtype=torch.float32)
        exact_xyz[0, :, 0] = torch.arange(40, dtype=torch.float32) * 0.5
        tied, _, segment, _ = peak_speed_margin(exact_xyz, exact_dt, _mask())
        self.assertEqual(float(below), 0.0)
        self.assertEqual(int(segment), 0)
        self.assertGreater(float(above), 0.0)
        self.assertGreater(float(tied), 0.0)
        exact_peak_z = torch.tensor(MARGIN_Z, requires_grad=True)
        boundary = margin_from_peak_z(exact_peak_z)
        grad = torch.autograd.grad(boundary, exact_peak_z)[0]
        self.assertEqual(float(boundary), 0.0)
        self.assertEqual(float(grad), 0.0)

    def test_actual_dt_denominator_mask_and_zero_lambda(self):
        dt = torch.full((39,), 0.05)
        xyz = _positions(1.0).requires_grad_()
        _, z, _, speed = peak_speed_margin(xyz, dt, _mask())
        self.assertAlmostEqual(float(speed), 1.0, places=5)
        self.assertAlmostEqual(float(z), 0.25, places=5)
        base = xyz.square().sum()
        margin, *_ = peak_speed_margin(xyz, dt, _mask())
        actual_grad = torch.autograd.grad(base + 0.0 * LAMBDA * margin, xyz, retain_graph=True)[0]
        expected_grad = torch.autograd.grad(base, xyz)[0]
        self.assertTrue(torch.equal(actual_grad, expected_grad))
        invalid_mask = _mask()
        invalid_mask[0, 12, 0] = 0.0
        with self.assertRaisesRegex(ValueError, "all 40 XYZ"):
            peak_speed_margin(xyz, dt, invalid_mask)

    def test_speed_margin_gradient_matches_finite_difference(self):
        dt = torch.full((39,), 0.05)
        xyz = _positions(2.2)
        xyz[0, 8, 1] = 0.007
        xyz.requires_grad_()
        margin, _, _, _ = peak_speed_margin(xyz, dt, _mask())
        gradient = torch.autograd.grad(LAMBDA * margin, xyz)[0][0, 8, 1].item()
        delta = 1e-4
        direction = torch.zeros_like(xyz)
        direction[0, 8, 1] = delta
        plus, *_ = peak_speed_margin(xyz.detach() + direction, dt, _mask())
        minus, *_ = peak_speed_margin(xyz.detach() - direction, dt, _mask())
        finite_difference = LAMBDA * (plus.item() - minus.item()) / (2.0 * delta)
        self.assertAlmostEqual(gradient, finite_difference, delta=2e-3)


if __name__ == "__main__":
    unittest.main()
