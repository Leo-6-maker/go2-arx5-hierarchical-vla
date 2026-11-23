import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from acceptance import handover, once_new, physical_grip_kinematics
from retime import DT, END, NAMES, SourcePath, crossing, geometric_progress, rates, select_timing


def fixture():
    a = np.zeros((50, 8))
    a[:, 3] = 1
    a[:, 0] = np.arange(50) / 30 * 0.01
    a[:, 7] = 0.5
    return a


class ClockAndPath(unittest.TestCase):
    def test_30hz_exact_with_separate_hold(self):
        path = SourcePath(fixture(), np.eye(4), np.eye(4))
        times = np.minimum(np.arange(40) * DT, END)
        self.assertEqual(np.flatnonzero(np.arange(40) * DT > END).tolist(), list(range(33, 40)))
        self.assertAlmostEqual(path.at(times)[20, 0, 3], 0.01)
        self.assertNotAlmostEqual(path.at(path.t)[20, 0, 3], 0.01)
        with self.assertRaises(ValueError):
            path.at([2.0])

    def test_no_extra_base_midpoint_and_slerp(self):
        a = fixture()
        a[:, 0] = 0.0
        a[:, 3:7] = Rotation.from_euler("z", np.pi / 2).as_quat()[[3, 0, 1, 2]]
        origin = np.eye(4)
        origin[0, 3] = 0.2
        tool = np.eye(4)
        tool[0, 3] = 0.1
        p = SourcePath(a, origin, tool)
        np.testing.assert_allclose(p.at([0])[0, :3, 3], [0.2, 0.1, 0], atol=1e-12)

    def test_nonzero_previous_velocity_is_binding(self):
        path = SourcePath(fixture(), np.eye(4), np.eye(4))
        lim = dict(zip(NAMES, [0.05, 0.25, 0.25, 1.0]))
        still = (np.zeros(3), np.zeros(3))
        self.assertIsNotNone(select_timing(path, lim, still, np.eye(3), max_ticks=80))
        sideways = (np.array([0.0, 0.025, 0.0]), np.zeros(3))
        self.assertIsNone(select_timing(path, lim, sideways, np.eye(3), max_ticks=80))

    def test_suffix_does_not_remove_internal_peak(self):
        a = fixture()
        a[12, 1] = 0.003
        p = SourcePath(a, np.eye(4), np.eye(4))
        m = p.at(np.minimum(np.arange(40) * DT, END))
        acc = np.linalg.norm(rates(m, DT)["linear_acceleration"], axis=1)
        self.assertGreater(max(acc[1:32]), 0.25)
        self.assertTrue((acc[34:] < 1e-12).all())

    def test_zero_motion_cannot_qualify(self):
        a = fixture()
        a[:, 0] = 0.0
        p = SourcePath(a, np.eye(4), np.eye(4))
        t = np.minimum(np.arange(40) * DT, END)
        m = p.at(t)
        r = geometric_progress(p, t, m, m, 0.02, "HOLD", np.zeros(40))
        self.assertFalse(r["motion_pass"])

    def test_event_phase_matters(self):
        a = fixture()
        a[:, 7] = np.linspace(0.9, 0.2, 50)
        p = SourcePath(a, np.eye(4), np.eye(4))
        t = np.arange(40) * 0.01
        g = np.interp(np.minimum(np.arange(40) * DT, END), p.t, a[:, 7])
        self.assertEqual(len(crossing(g)), 1)
        self.assertFalse(geometric_progress(p, t, p.at(t), p.at(t), 0.02, "CLOSE", g)["event_pass"])


class ControlReceipts(unittest.TestCase):
    def test_real_previous_command_not_measured(self):
        r = physical_grip_kinematics(np.repeat((0.01919 - 0.002) / 0.040, 40), 0.002)
        self.assertAlmostEqual(r["entry"] * 1000, 343.8)

    def test_contact_rejects_even_with_an_owner_token(self):
        ledger = dict(
            contact=True,
            grasp=True,
            contact_valid=True,
            prohibited_contact=False,
            student_owner_transfer_receipt="test",
            actual_takeover_flag=True,
        )
        self.assertEqual(handover(ledger), "HANDOVER_BLOCKED_CONTACT")

    def test_missing_owner_prevents_transition(self):
        self.assertEqual(
            handover(
                dict(
                    contact_valid=True,
                    prohibited_contact=False,
                    contact=False,
                    grasp=False,
                    student_owner_transfer_receipt=None,
                    actual_takeover_flag=False,
                )
            ),
            "HANDOVER_BLOCKED_NO_TRANSFER_ACK",
        )

    def test_unknown_run_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "run"
            once_new(p)
            (p / "UNKNOWN").touch()
            with self.assertRaises(FileExistsError):
                once_new(p)
            self.assertTrue((p / "UNKNOWN").is_file())


if __name__ == "__main__":
    unittest.main()
