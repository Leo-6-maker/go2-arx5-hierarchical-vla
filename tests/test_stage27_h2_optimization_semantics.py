"""Deterministic CPU-only tests for the Stage27 H2 semantics audit."""

from __future__ import annotations

import ast
import importlib.util
import math
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "stage27_h2_optimization_semantics_audit.py"
SPEC = importlib.util.spec_from_file_location("stage27_h2_semantics_under_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Stage27H2OptimizationSemanticsTests(unittest.TestCase):
    def test_plan_is_static_and_keeps_hopper_hold(self) -> None:
        plan = MODULE.protocol_plan()
        self.assertEqual(plan["schema"], MODULE.STAGE27_SCHEMA)
        self.assertEqual(plan["stage27_base_sha"], MODULE.STAGE27_BASE_SHA)
        self.assertFalse(plan["execution_started"])
        self.assertTrue(all(value is False for value in plan["audit_scope"].values() if value is not True))
        self.assertEqual(
            plan["h2_protocol_decision_aid"]["recommended_stock_path_candidate"],
            "H2_FRESH_BASE_2000_FIXED_WARMUP10",
        )

    def test_warmup_ratio_uses_ceil_and_explicit_steps_override(self) -> None:
        self.assertEqual(MODULE.resolve_warmup_steps(200), 10)
        self.assertEqual(MODULE.resolve_warmup_steps(2000), 100)
        self.assertEqual(MODULE.resolve_warmup_steps(2000, warmup_steps=10), 10)

    def test_pinned_logged_schedule_matches_sealed_values(self) -> None:
        expected = {
            1: 0.0,
            2: 1e-5,
            10: 9e-5,
            11: 1e-4,
            12: 9.999316524962345e-5,
            200: 6.834750376549792e-9,
        }
        for step, value in expected.items():
            self.assertEqual(MODULE.logged_cosine_learning_rate(step, 200), value)

    def test_fresh_2000_schedule_diverges_before_step_200(self) -> None:
        self.assertEqual(MODULE.logged_cosine_learning_rate(1, 200), 0.0)
        self.assertNotEqual(
            MODULE.logged_cosine_learning_rate(2, 200),
            MODULE.logged_cosine_learning_rate(2, 2000),
        )
        self.assertTrue(
            math.isclose(
                MODULE.logged_cosine_learning_rate(200, 2000),
                9.93316053564e-5,
                rel_tol=1e-10,
            )
        )

    def test_schedule_reconciliation_requires_every_log_row(self) -> None:
        history = [
            {"step": step, "learning_rate": MODULE.logged_cosine_learning_rate(step, 200)}
            for step in range(1, 201)
        ]
        result = MODULE.compare_logged_schedule(history, 200)
        self.assertEqual(result["observed_rows"], 200)
        self.assertEqual(result["matched_rows"], 200)
        self.assertTrue(result["all_rows_match"])
        changed = list(history)
        changed[100] = dict(changed[100], learning_rate=0.123)
        changed_result = MODULE.compare_logged_schedule(changed, 200)
        self.assertFalse(changed_result["all_rows_match"])

    def test_state_artifact_detector_is_strict_and_excludes_trainer_metadata(self) -> None:
        self.assertEqual(
            MODULE._state_artifact_paths(
                [
                    "trainer_state.json",
                    "training_args.bin",
                    "optimizer.pt",
                    "scheduler.pt",
                    "rng_state.pth",
                    "scaler.pt",
                    "zero/optimizer_state.pt",
                ]
            ),
            ["optimizer.pt", "rng_state.pth", "scaler.pt", "scheduler.pt"],
        )

    def test_checkpoint_state_summary_rejects_state_file(self) -> None:
        manifest = {
            "artifacts": {
                "checkpoint-200/trainer_state.json": {},
                "checkpoint-200/optimizer.pt": {},
            }
        }
        trainer_state = {
            key: None for key in MODULE.EXPECTED_TRAINER_STATE_TOP_LEVEL_KEYS
        }
        summary = MODULE.checkpoint_state_summary(manifest, trainer_state)
        self.assertEqual(summary["all_state_artifact_files"], ["optimizer.pt"])

    def test_source_has_no_runtime_model_or_gpu_imports(self) -> None:
        tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
        self.assertNotIn("torch", imported)
        self.assertNotIn("transformers", imported)
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertNotIn("AutoModel", source)


if __name__ == "__main__":
    unittest.main()
