"""Deterministic CPU/static tests for the Stage28 H2 preparation contract."""

from __future__ import annotations

import ast
import importlib.util
import json
import shutil
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "stage28_h2_fresh_base_2000_fixed_warmup10.py"
SPEC = importlib.util.spec_from_file_location("stage28_h2_under_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Stage28H2PreparationTests(unittest.TestCase):
    def test_plan_freezes_new_recipe_and_single_downstream_run(self) -> None:
        plan = MODULE.protocol_plan()
        self.assertEqual(plan["schema"], MODULE.STAGE28_SCHEMA)
        self.assertEqual(plan["stage27_accepted_sha"], "9d0cf7c33b2d89ffee86168befe65978115564aa")
        contract = plan["training_contract"]
        self.assertEqual(contract["max_steps"], 2000)
        self.assertEqual(contract["warmup_steps"], 10)
        self.assertEqual(contract["warmup_ratio"], 0.0)
        self.assertEqual(contract["save_steps"], 1000)
        self.assertFalse(contract["save_only_model"])
        self.assertFalse(contract["resume_from_checkpoint"])
        self.assertEqual(plan["execution_budget"]["training_runs"], 1)
        self.assertEqual(plan["execution_budget"]["maximum_direct_flow_evaluations"], 18)
        self.assertFalse(plan["preparation_execution_boundary"]["training"])
        self.assertTrue(plan["conditional_handoff"]["h2_execution_ready"])
        self.assertEqual(
            plan["conditional_handoff"]["storage_preflight"],
            "HOPPER_RUNTIME_REQUIRED",
        )

    def test_schedule_has_ten_step_warmup_and_1990_cosine_denominator(self) -> None:
        self.assertEqual(MODULE._cosine_learning_rate(1), 0.0)
        self.assertAlmostEqual(MODULE._cosine_learning_rate(10), 9e-5, places=16)
        self.assertAlmostEqual(MODULE._cosine_learning_rate(11), 1e-4, places=16)
        self.assertEqual(
            MODULE.protocol_plan()["schedule"]["cosine_denominator_steps"], 1990
        )
        self.assertLess(MODULE._cosine_learning_rate(2000), 1e-9)

    def test_storage_reservation_is_deterministic_and_conservative(self) -> None:
        reserve = MODULE.storage_reservation(7_000_000_000)
        self.assertEqual(
            reserve["retained_model_checkpoint_bytes"], 14_000_000_000
        )
        self.assertEqual(reserve["minimum_required_free_bytes"], max(
            MODULE.MIN_STORAGE_RESERVE_BYTES,
            reserve["calculated_required_free_bytes"],
        ))
        self.assertGreater(reserve["adam_full_state_reserve_bytes"], 0)

    def test_storage_preflight_passes_and_fails_without_touching_inputs(self) -> None:
        with self.subTest("pass"):
            with unittest.mock.patch.object(
                MODULE.shutil,
                "disk_usage",
                return_value=shutil._ntuple_diskusage(
                    200 * MODULE.GIB, 20 * MODULE.GIB, 180 * MODULE.GIB
                ),
            ):
                # Use a real temporary directory only for this test's output
                # root; no model/data input is created or modified.
                import tempfile

                with tempfile.TemporaryDirectory() as temp:
                    root = Path(temp) / "run"
                    result = MODULE._storage_preflight(
                        root,
                        base_tree_bytes=7_000_000_000,
                        immutable_inputs=(Path(temp) / "base", Path(temp) / "data"),
                    )
                    self.assertEqual(result["status"], "PASS")

        with self.subTest("fail"):
            with mock.patch.object(
                MODULE.shutil,
                "disk_usage",
                return_value=shutil._ntuple_diskusage(
                    200 * MODULE.GIB, 199 * MODULE.GIB, 1 * MODULE.GIB
                ),
            ):
                import tempfile

                with tempfile.TemporaryDirectory() as temp:
                    with self.assertRaises(MODULE.Stage28Error):
                        MODULE._storage_preflight(
                            Path(temp) / "run",
                            base_tree_bytes=7_000_000_000,
                            immutable_inputs=(Path(temp) / "base",),
                        )

    def test_checkpoint_inventory_requires_all_full_state_files(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "checkpoint-2000"
            root.mkdir()
            (root / MODULE.MODEL_INDEX_NAME).write_text(
                json.dumps({"weight_map": {"x": "model.safetensors"}}),
                encoding="utf-8",
            )
            (root / "model.safetensors").write_bytes(b"fixture")
            (root / "config.json").write_text("{}", encoding="utf-8")
            (root / "trainer_state.json").write_text(
                json.dumps({"global_step": 2000, "max_steps": 2000}),
                encoding="utf-8",
            )
            for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
                (root / name).write_bytes(b"state")
            info = MODULE._inspect_checkpoint(root, expected_step=2000)
            self.assertEqual(info["state_files"], sorted(MODULE.STATE_FILES_REQUIRED))
            (root / "optimizer.pt").unlink()
            with self.assertRaises(MODULE.Stage28Error):
                MODULE._inspect_checkpoint(root, expected_step=2000)

    def test_checkpoint_inventory_rejects_symlink(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "checkpoint-1000"
            root.mkdir()
            (root / "target").write_bytes(b"x")
            (root / "link").symlink_to("target")
            with self.assertRaises(MODULE.Stage28Error):
                MODULE._regular_entries(root)

    def test_source_has_no_eager_runtime_ml_imports_or_sampler_calls(self) -> None:
        tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
        self.assertNotIn("torch", imported)
        self.assertNotIn("transformers", imported)
        self.assertNotIn("safetensors", imported)
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertNotIn("get_action(", source)
        self.assertNotIn("Euler", source)

    def test_pbs_and_evaluation_scope_are_bounded(self) -> None:
        plan = MODULE.protocol_plan()
        self.assertFalse(plan["execution_budget"]["free_running_sampler"])
        self.assertEqual(
            plan["checkpoint_policy"]["required_checkpoint_directories"],
            ["checkpoint-1000", "checkpoint-2000"],
        )
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertIn("same_fixed_probes_reused_for", source)
        self.assertIn('checkpoint_label="BASE"', source)
        self.assertIn('checkpoint_label="H2@2000"', source)
        self.assertIn("maximum_flow_evaluation_count", source)


if __name__ == "__main__":
    unittest.main()
