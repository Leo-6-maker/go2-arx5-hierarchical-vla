from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, "scripts")

import stage45f_vanda_validation_e_admission as admission
import stage45f_vanda_validation_e_gen1 as runner


def _snapshot_payload() -> dict[str, object]:
    state = [0.1, -0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5]
    return {
        "example_id": "stage22_expert_red_target_holdout_e:expert-0000",
        "state_v1": {
            "schema": "StateV1",
            "layout": ["eef_xyz[3]", "eef_rot6d_rows[6]", "gripper[1]"],
            "values": state,
        },
        "raw_telemetry": {
            "achieved_position_local_m": state[:3],
            "achieved_position_world_m": [0.4, 0.0, 0.9],
        },
    }


class ValidationEFrameContractTests(unittest.TestCase):
    def test_checkpoint_tree_prefix_is_canonicalized(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "file.bin").write_bytes(b"pi0154")

            digest_with_slash, entries_with_slash = runner.tree_sha256(root, "arm/")
            digest_without_slash, entries_without_slash = runner.tree_sha256(root, "arm")

            self.assertEqual(digest_with_slash, digest_without_slash)
            self.assertEqual(entries_with_slash, entries_without_slash)
            self.assertEqual(entries_with_slash[0]["path"], "arm/file.bin")

    def test_frame_origin_uses_world_minus_relative_without_target(self) -> None:
        payload = _snapshot_payload()

        np.testing.assert_allclose(runner._anchor_from_snapshot(payload), [0.3, 0.2, 0.6])
        audit = admission._frame_origin_audit(payload, admission.REPO / "fixture.json")
        self.assertEqual(audit["source"], "raw_telemetry.achieved_position_world_m-minus-achieved_position_local_m")
        self.assertFalse(audit["target_recentered"])
        self.assertFalse(audit["measured_start_recentered"])

    def test_frame_origin_rejects_state_telemetry_mismatch(self) -> None:
        payload = _snapshot_payload()
        payload["raw_telemetry"]["achieved_position_local_m"] = [0.11, -0.2, 0.3]  # type: ignore[index]

        with self.assertRaises((admission.AdmissionError, runner.ValidationExecutionError)):
            admission._frame_origin_audit(payload, admission.REPO / "fixture.json")
        with self.assertRaises(runner.ValidationExecutionError):
            runner._anchor_from_snapshot(payload)

    def test_pre_model_import_guard_rejects_model_modules(self) -> None:
        self.assertEqual(admission.model_imports_present({"torch": object(), "torch.nn": object(), "json": object()}), ["torch", "torch.nn"])
        self.assertEqual(admission.model_imports_present({"json": object()}), [])


if __name__ == "__main__":
    unittest.main()
