from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import stage45f_pi0177_validation_e_hopper_resume as runner


class PI0177RunnerTests(unittest.TestCase):
    def test_plan_is_static_and_budgeted(self) -> None:
        plan = runner.plan()
        self.assertEqual(plan["status"], "PASS_STATIC_NOT_EXECUTED")
        self.assertTrue(plan["fresh_child_per_arm"])
        self.assertTrue(plan["raw_before_decode"])
        self.assertEqual(plan["new"]["raw"], 30)
        self.assertEqual(plan["final"]["raw"], 36)
        self.assertEqual(plan["h200_submissions_by_runner"], 0)

    def test_ledger_is_append_only_and_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "execution_ledger.jsonl"
            first = runner.ExecutionLedger(path, generation="Validation-E-gen1", gpu_uuid="GPU-test")
            token = first.begin("MODEL_LOAD", arm="Legacy(seed2223)@2000")
            first.end(token, arm="Legacy(seed2223)@2000")
            before = path.read_bytes()
            second = runner.ExecutionLedger(path, generation="Validation-E-gen1", gpu_uuid="GPU-test")
            second.emit("RAW_SEALED", phase="RAW_SEAL", arm="Legacy(seed2223)@2000", window_id=runner.EXPECTED_WINDOWS[0])
            after = path.read_bytes()
            self.assertTrue(after.startswith(before))
            records = [json.loads(line) for line in after.decode().splitlines()]
            self.assertEqual([item["sequence"] for item in records], list(range(1, len(records) + 1)))
            self.assertEqual(second.count_begin("MODEL_LOAD"), 1)

    def test_static_guard_has_no_model_import_or_scheduler_submit(self) -> None:
        guard = runner.static_guard()
        self.assertEqual(guard["status"], "PASS")
        self.assertFalse(guard["model_imported"])
        self.assertEqual(guard["pbs_submissions"], 0)
        self.assertTrue(guard["markers"]["no_h200_submit"])


if __name__ == "__main__":
    unittest.main()
