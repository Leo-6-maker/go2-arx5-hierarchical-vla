import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "stage45f_pi0179_train24_canary_admission.py"
SPEC = importlib.util.spec_from_file_location("stage45f_pi0179_train24_canary_admission", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _snapshot(uuid: str = "GPU-test", total: int = 32760, free: int = 20000):
    return {"gpus": [{"index": 0, "name": "NVIDIA GeForce RTX 4080 SUPER", "uuid": uuid, "memory_total_mib": total, "memory_used_mib": total - free, "memory_free_mib": free}]}


class PI0179CanaryAdmissionTests(unittest.TestCase):
    def test_plan_freezes_two_snapshot_shared_gpu_policy(self):
        plan = MODULE.plan()
        self.assertEqual(plan["gpu_policy"]["occupancy_snapshots"], 2)
        self.assertTrue(plan["gpu_policy"]["same_uuid"])
        self.assertEqual(plan["canary_ceiling"]["model_side_total"], 5)
        self.assertTrue(plan["gpu_policy"]["foreign_processes"], "allowed")

    def test_admission_accepts_foreign_process_tolerant_4080(self):
        result = MODULE.validate_admission([_snapshot(), _snapshot()], 0)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["snapshot_count"], 2)

    def test_admission_rejects_uuid_or_capacity_drift(self):
        with self.assertRaisesRegex(MODULE.AdmissionError, "UUID"):
            MODULE.validate_admission([_snapshot("GPU-a"), _snapshot("GPU-b")], 0)
        with self.assertRaisesRegex(MODULE.AdmissionError, "free VRAM"):
            MODULE.validate_admission([_snapshot(free=11000), _snapshot(free=11000)], 0)
        with self.assertRaisesRegex(MODULE.AdmissionError, "total VRAM"):
            MODULE.validate_admission([_snapshot(total=29000, free=12000), _snapshot(total=29000, free=12000)], 0)


if __name__ == "__main__":
    unittest.main()
