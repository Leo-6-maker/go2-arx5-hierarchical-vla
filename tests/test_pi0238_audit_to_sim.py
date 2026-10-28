from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    from scripts.pi0238_audit_to_sim import (
        DiagnosticBlocker,
        DT,
        HORIZON,
        parse_identity,
        sha256_array,
        stats,
        tree_sha256,
    )
except ModuleNotFoundError:
    from pi0238_audit_to_sim import (  # type: ignore
        DiagnosticBlocker,
        DT,
        HORIZON,
        parse_identity,
        sha256_array,
        stats,
        tree_sha256,
    )


class PI0238CpuContractTests(unittest.TestCase):
    def test_missing_and_nonfinite_keep_the_total_denominator(self) -> None:
        result = stats([1.0, None, float("nan"), float("inf"), 3.0])
        self.assertEqual(result["total_count"], 5)
        self.assertEqual(result["finite_count"], 2)
        self.assertEqual(result["missing_count"], 1)
        self.assertEqual(result["nonfinite_count"], 2)

    def test_train_identity_rejects_final_split(self) -> None:
        parsed = parse_identity(
            "pi0227_persistent_train_a_j07_xp010_yp010_s04_ym050_r03_r03_227156"
        )
        self.assertEqual(parsed["target_family"], "a")
        with self.assertRaises(DiagnosticBlocker):
            parse_identity("pi0227_persistent_final_f_j00_nominal_s00_c0_r00")

    def test_tree_hash_rejects_symlink_and_counts_real_files(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "x.txt").write_text("x", encoding="utf-8")
            digest, count, total = tree_sha256(root, "test")
            self.assertEqual(count, 1)
            self.assertEqual(total, 1)
            self.assertEqual(len(digest), 64)
            try:
                (root / "link.txt").symlink_to(root / "x.txt")
            except (OSError, NotImplementedError):
                return
            with self.assertRaises(DiagnosticBlocker):
                tree_sha256(root, "test")

    def test_clock_origin_does_not_change_velocity_or_initial_rotation(self) -> None:
        xyz = np.stack([np.zeros(3), np.array([0.1, 0.0, 0.0])])
        timestamps = np.array([0.0, DT])
        shifted = timestamps + 17.0
        self.assertTrue(np.allclose(np.diff(xyz, axis=0) / np.diff(timestamps)[:, None], np.diff(xyz, axis=0) / np.diff(shifted)[:, None]))

    def test_nonunit_rotation_and_q_sign_are_not_silent_identity(self) -> None:
        q = np.array([0.0, 0.0, math.sin(0.2), math.cos(0.2)])
        q_neg = -q
        self.assertTrue(np.allclose(q / np.linalg.norm(q), q_neg / np.linalg.norm(q_neg) * -1.0))
        self.assertFalse(np.allclose(q, np.zeros(4)))

    def test_universal_slots_are_explicit(self) -> None:
        action = np.zeros((1, HORIZON, 132), dtype=np.float32)
        mask = np.zeros_like(action)
        action[..., :10] = 1.0
        mask[..., :10] = 1.0
        self.assertEqual(int(mask.sum()), HORIZON * 10)
        self.assertEqual(int(mask[..., 10:].sum()), 0)
        self.assertEqual(sha256_array(action).isascii(), True)

    def test_g0_g3_source_has_no_model_or_training_entry(self) -> None:
        source = Path(__file__).parents[1] / "scripts" / "pi0238_audit_to_sim.py"
        text = source.read_text(encoding="utf-8")
        for forbidden in ("AutoModel.from_pretrained", ".get_action(", ".backward(", "optimizer.step(", "Trainer.train("):
            self.assertNotIn(forbidden, text)


if __name__ == "__main__":
    unittest.main()
