"""Unit tests for read-only V1.6 science gating, with synthetic evidence fixtures."""
import copy
import unittest

from scripts.sla_hierarchical_v16.preflight_public import assess, EvidenceMismatch


def fixture():
    o1 = {
        "denominator": 16, "windows": [{"id": f"T{i:02d}"} for i in range(1, 17)],
        "labels": {
            "DIRECT_OFFICIAL_RAW": {"TASK_LEGAL": 0},
            "OFFICIAL_WITH_CONTINUOUS_REF": {
                "TASK_LEGAL": 0, "DYNAMICS_PASS": 0, "NONTRIVIAL_PROGRESS": 0
            },
        },
        "gripper_commands_issued": 0, "heldout_episodes": 0,
        "new_gpu_forwards": 0, "new_physics_ticks": 0, "training_updates": 0,
    }
    o3 = {"archived_student_owner_receipts": 0, "blocked_windows": 16, "new_native_runs": 0}
    f2 = {
        "decision": "FROZEN_WBC_KNOWN_GOAL_DWELL_QUALIFIED",
        "matched_comparisons": [{} for _ in range(4)],
        "native_student_grasps_attempted": 0,
        "student_grasp_success_rate": None,
        "SLA_forwards": 0,
        "training_updates": 0,
        "physical_stops": 0,
    }
    reg = {
        "status": "PLAN_ONLY_NOT_AUTHORIZED",
        "base_parent_sha": "d6639cb03af1ff98519b2566cee2c9ee29627ba8",
        "proposed_resources": {
            k: 0 for k in (
                "new_gpu_forwards_authorized", "new_native_requests_authorized",
                "new_physics_ticks_authorized", "training_updates_authorized",
                "real_hardware_runs_authorized",
            )
        },
    }
    return o1, o3, f2, reg


class V16PreflightTests(unittest.TestCase):
    def test_valid_public_baseline_still_blocks_student_native(self):
        result = assess(*fixture())
        self.assertEqual(result["status"], "PUBLIC_EVIDENCE_CONSISTENT")
        self.assertIsNone(result["real_student_grasp_success_rate"])
        self.assertFalse(result["native_or_GPU_permission_granted"])
        self.assertIn("BLOCKED", result["native_student_control"])

    def test_tampered_legal_denominator_is_rejected(self):
        items = list(fixture())
        items[0] = copy.deepcopy(items[0])
        items[0]["labels"]["DIRECT_OFFICIAL_RAW"]["TASK_LEGAL"] = 1
        with self.assertRaises(EvidenceMismatch):
            assess(*items)

    def test_fabricated_gripper_owner_is_rejected(self):
        items = list(fixture())
        items[1] = dict(items[1], archived_student_owner_receipts=16)
        with self.assertRaises(EvidenceMismatch):
            assess(*items)

    def test_static_dwell_cannot_be_misreported_as_task_success(self):
        items = list(fixture())
        items[2] = dict(items[2], student_grasp_success_rate=0.0)
        with self.assertRaises(EvidenceMismatch):
            assess(*items)

    def test_resource_budget_cannot_be_silently_granted(self):
        items = list(fixture())
        items[3] = copy.deepcopy(items[3])
        items[3]["proposed_resources"]["new_native_requests_authorized"] = 1
        with self.assertRaises(EvidenceMismatch):
            assess(*items)

    def test_duplicate_student_windows_are_rejected(self):
        items = list(fixture())
        items[0] = copy.deepcopy(items[0])
        items[0]["windows"][1]["id"] = "T01"
        with self.assertRaises(EvidenceMismatch):
            assess(*items)


if __name__ == "__main__":
    unittest.main()
