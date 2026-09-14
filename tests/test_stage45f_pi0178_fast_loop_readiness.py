from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import stage45f_pi0178_fast_loop_readiness as readiness


class PI0178ReadinessTests(unittest.TestCase):
    def test_schedule_is_deterministic_and_exposure_matched(self) -> None:
        schedule = readiness.exposure_schedule()
        summary = readiness.validate_exposure_schedule(schedule)
        self.assertEqual(summary["sequence_sha256"], readiness.Q0_SCHEDULE_SHA)
        self.assertEqual(summary["E_full24_slots"], 256)
        self.assertEqual(summary["S_full24_slots"], 128)
        self.assertEqual(summary["S_q0_slots"], 128)

    def test_runspec_late_binds_base_and_freezes_recipe(self) -> None:
        spec = readiness.training_runspec()
        self.assertEqual(spec["status"], "PREPARED_NOT_EXECUTED")
        self.assertIsNone(spec["base_checkpoint"]["identity"])
        self.assertEqual(spec["arms"]["E"]["optimizer_updates"], 256)
        self.assertEqual(spec["arms"]["S"]["q0_slots"], 128)
        self.assertEqual(spec["recipe"]["trainable_modules"], ["projector", "diffusion"])
        self.assertEqual(spec["recipe"]["frozen_modules"], ["visual", "llm", "vlln"])
        self.assertEqual(spec["execution_counters"]["optimizer_steps"], 0)

    def test_dev_cases_and_hard_zeroes_are_locked(self) -> None:
        plan = readiness.readiness_plan(ROOT)
        self.assertEqual(plan["authority"]["source_head"], readiness.SOURCE_HEAD)
        self.assertEqual(plan["candidate_promotion"]["status"], "READY_NOT_EXECUTED")
        self.assertEqual(plan["candidate_promotion"]["fixed_dev_initial_conditions"][1]["condition_id"], "C1_X_PLUS_5CM")
        self.assertFalse(plan["candidate_promotion"]["closed_loop_scorecard"]["wbc_executed"])
        self.assertEqual(plan["hard_zeroes"]["validation_e_model_calls"], 0)
        self.assertEqual(plan["hard_zeroes"]["simulator_steps"], 0)

    def test_q0_audit_requires_exact_identity_constants(self) -> None:
        self.assertEqual(len(readiness.Q0_NPZ_SHA), 64)
        self.assertEqual(len(readiness.Q0_JSON_SHA), 64)
        self.assertEqual(len(readiness.Q0_EXPERT_CHUNK_SHA), 64)
        self.assertEqual(readiness.Q0_CONDITION, "C1_X_PLUS_5CM")
        self.assertEqual(readiness.Q0_TARGET, "red_target_train_a")


if __name__ == "__main__":
    unittest.main()
