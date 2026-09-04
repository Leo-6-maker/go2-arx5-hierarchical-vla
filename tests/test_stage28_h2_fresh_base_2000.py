"""Deterministic CPU/static tests for the Stage28 H2 preparation contract."""

from __future__ import annotations

import ast
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "stage28_h2_fresh_base_2000_fixed_warmup10.py"
EVALUATION_SCRIPT_PATH = REPO_ROOT / "scripts" / "stage28_h2_evaluation_only.py"
EVALUATION_PBS_PATH = REPO_ROOT / "hpc/pbs/stage28_h2_evaluation_only.pbs"
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

    def test_recovery_contract_binds_current_gate_and_historical_artifacts(self) -> None:
        recovery = MODULE.protocol_plan()["evaluation_recovery"]
        self.assertEqual(recovery["pi_gate_id"], "PI_GATE_STATE_0008")
        self.assertEqual(recovery["token"], "PI_GATE_STATE_0008")
        self.assertTrue(recovery["evaluation_only"])
        self.assertFalse(recovery["training"])
        self.assertFalse(recovery["backward"])
        self.assertFalse(recovery["optimizer_step"])
        self.assertFalse(recovery["scheduler_step"])
        self.assertFalse(recovery["resume"])
        self.assertFalse(recovery["checkpoint_write"])
        self.assertEqual(
            recovery["h2_checkpoint_tree_sha256"],
            "3106f9b5993b390438b62594d044781d11b393452a49f80b5c8e9e910606d0ba",
        )
        self.assertEqual(
            recovery["h2_checkpoint_seal_sha256"],
            "1349896f7b31a930764dbd9d849c5368e48372d0dc6273449f42f19f82a69a82",
        )
        self.assertEqual(
            recovery["historical_training_source_sha"],
            "a361e5cb8a4be33f40a21c9bbf7ad742e25c7fd0",
        )
        self.assertEqual(
            recovery["historical_training_runner_sha256"],
            "d61c429a2b8d53c25df63d9b5b9e2464b33c36e9e3c0a610b109b10a6bd63fe4",
        )

    def test_schedule_has_ten_step_warmup_and_1990_cosine_denominator(self) -> None:
        self.assertEqual(MODULE._cosine_learning_rate(1), 0.0)
        self.assertAlmostEqual(MODULE._cosine_learning_rate(10), 9e-5, places=16)
        self.assertAlmostEqual(MODULE._cosine_learning_rate(11), 1e-4, places=16)
        self.assertEqual(
            MODULE.protocol_plan()["schedule"]["cosine_denominator_steps"], 1990
        )
        self.assertLess(MODULE._cosine_learning_rate(2000), 1e-9)

    def test_stage26_loader_filters_18_rows_and_returns_exact_base_nine(self) -> None:
        digest_base = "a" * 64
        digest_o1 = "b" * 64
        base_rows = [
            {
                "checkpoint": "BASE",
                "probe_id": probe_id,
                "raw_predicted_velocity_sha256": digest_base,
            }
            for probe_id in MODULE.EXPECTED_STAGE26_PROBE_IDS
        ]
        o1_rows = [
            {
                "checkpoint": "O1@200",
                "probe_id": probe_id,
                "raw_predicted_velocity_sha256": digest_o1,
            }
            for probe_id in MODULE.EXPECTED_STAGE26_PROBE_IDS
        ]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "stage26_hash_manifest.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": "STAGE26_H1_C1_FIXED_NOISE_LOCAL_FLOW_V1",
                        "raw_velocity_hashes": [*base_rows, *o1_rows],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            observed_sha = MODULE._sha256_file(path)
            with mock.patch.object(
                MODULE, "EXPECTED_STAGE26_HASH_MANIFEST_SHA256", observed_sha
            ):
                result = MODULE._load_stage26_hash_manifest(path)
            self.assertEqual(
                set(result["raw_velocity_hashes"]),
                set(MODULE.EXPECTED_STAGE26_PROBE_IDS),
            )
            self.assertEqual(
                set(result["raw_velocity_hashes"].values()), {digest_base}
            )

    def test_stage26_loader_rejects_count_id_hash_and_sha_failures(self) -> None:
        digest = "a" * 64

        def rows_for(
            checkpoint: str, probe_ids: tuple[str, ...], digest: str = digest
        ) -> list[dict[str, str]]:
            return [
                {
                    "checkpoint": checkpoint,
                    "probe_id": probe_id,
                    "raw_predicted_velocity_sha256": digest,
                }
                for probe_id in probe_ids
            ]

        def assert_rows_fail(rows: list[dict[str, str]], message: str) -> None:
            with self.subTest(message=message):
                with tempfile.TemporaryDirectory() as temp:
                    path = Path(temp) / "stage26_hash_manifest.json"
                    path.write_text(
                        json.dumps(
                            {
                                "schema": "STAGE26_H1_C1_FIXED_NOISE_LOCAL_FLOW_V1",
                                "raw_velocity_hashes": rows,
                            }
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    observed_sha = MODULE._sha256_file(path)
                    with mock.patch.object(
                        MODULE, "EXPECTED_STAGE26_HASH_MANIFEST_SHA256", observed_sha
                    ):
                        with self.assertRaises(MODULE.Stage28Error):
                            MODULE._load_stage26_hash_manifest(path)

        ids = MODULE.EXPECTED_STAGE26_PROBE_IDS
        o1_rows = rows_for("O1@200", ids, "b" * 64)
        assert_rows_fail(
            [*rows_for("BASE", ids[:8]), *o1_rows[:10]], "BASE count 8"
        )
        assert_rows_fail(
            [*rows_for("BASE", ids), *o1_rows[:8], *rows_for("BASE", ("foreign",))],
            "BASE count 10",
        )
        duplicate = [
            *rows_for("BASE", ids[:-1]),
            {**rows_for("BASE", (ids[-2],))[0]},
            *o1_rows,
        ]
        assert_rows_fail(duplicate, "duplicate BASE probe ID")
        missing = [
            *rows_for("BASE", ids[:-1]),
            {**rows_for("BASE", ("foreign",))[0]},
            *o1_rows,
        ]
        assert_rows_fail(missing, "missing expected and foreign BASE ID")
        malformed = [*rows_for("BASE", ids), *o1_rows]
        malformed[0]["raw_predicted_velocity_sha256"] = "not-a-sha"
        assert_rows_fail(malformed, "malformed BASE hash")
        unknown_checkpoint = [*rows_for("BASE", ids), *o1_rows]
        unknown_checkpoint[-1]["checkpoint"] = "foreign-checkpoint"
        assert_rows_fail(unknown_checkpoint, "unknown checkpoint")
        assert_rows_fail(
            [*rows_for("BASE", ids), *o1_rows[:-1]], "seventeen total rows"
        )

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "stage26_hash_manifest.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": "STAGE26_H1_C1_FIXED_NOISE_LOCAL_FLOW_V1",
                        "raw_velocity_hashes": [*rows_for("BASE", ids), *o1_rows],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(
                MODULE, "EXPECTED_STAGE26_HASH_MANIFEST_SHA256", "b" * 64
            ):
                with self.assertRaises(MODULE.Stage28Error):
                    MODULE._load_stage26_hash_manifest(path)

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

    def test_h2_checkpoint_external_tree_and_seal_bindings_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output_root = Path(temp) / "stage28-output"
            training_root = output_root / "h2_training"
            training_root.mkdir(parents=True)

            for step in (1000, 2000):
                checkpoint = training_root / f"checkpoint-{step}"
                checkpoint.mkdir()
                (checkpoint / MODULE.MODEL_INDEX_NAME).write_text(
                    json.dumps({"weight_map": {"x": "model.safetensors"}}),
                    encoding="utf-8",
                )
                (checkpoint / "model.safetensors").write_bytes(
                    f"fixture-{step}".encode("utf-8")
                )
                (checkpoint / "config.json").write_text("{}", encoding="utf-8")
                (checkpoint / "trainer_state.json").write_text(
                    json.dumps({"global_step": step, "max_steps": 2000}),
                    encoding="utf-8",
                )
                for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
                    (checkpoint / name).write_bytes(b"state")

            checkpoint_2000 = MODULE._inspect_checkpoint(
                training_root / "checkpoint-2000", expected_step=2000
            )
            seal_path = output_root / "stage28_h2_checkpoint_2000_seal.json"
            seal_path.write_text(
                json.dumps(
                    {
                        "schema": "STAGE28_H2_FINAL_CHECKPOINT_SEAL_V1",
                        "status": "PASS",
                        "checkpoint_name": "checkpoint-2000",
                        "checkpoint_path": str(training_root / "checkpoint-2000"),
                        "tree_sha256": checkpoint_2000["tree_sha256"],
                        "file_count": checkpoint_2000["file_count"],
                        "total_bytes": checkpoint_2000["total_bytes"],
                        "entries": checkpoint_2000["entries"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            seal_sha256 = MODULE._sha256_file(seal_path)
            result = MODULE._verify_final_checkpoint_artifacts(
                output_root,
                expected_checkpoint_tree_sha256=checkpoint_2000["tree_sha256"],
                expected_seal_sha256=seal_sha256,
            )
            self.assertEqual(
                result["checkpoint-2000"]["tree_sha256"],
                checkpoint_2000["tree_sha256"],
            )
            with self.assertRaisesRegex(
                MODULE.Stage28Error, "external tree SHA mismatch"
            ):
                MODULE._verify_final_checkpoint_artifacts(
                    output_root,
                    expected_checkpoint_tree_sha256="0" * 64,
                    expected_seal_sha256=seal_sha256,
                )
            with self.assertRaisesRegex(
                MODULE.Stage28Error, "external seal SHA mismatch"
            ):
                MODULE._verify_final_checkpoint_artifacts(
                    output_root,
                    expected_checkpoint_tree_sha256=checkpoint_2000["tree_sha256"],
                    expected_seal_sha256="0" * 64,
                )

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

    def test_evaluation_only_launcher_has_no_training_or_optimizer_path(self) -> None:
        launcher_source = EVALUATION_SCRIPT_PATH.read_text(encoding="utf-8")
        # ``--training-*`` provenance arguments are allowed; only a training
        # mode flag itself must be absent from the evaluation-only launcher.
        self.assertNotIn('add_argument("--train"', launcher_source)
        self.assertNotIn("execute_training", launcher_source)
        self.assertNotIn("optimizer.step", launcher_source)
        launcher_tree = ast.parse(launcher_source)
        launcher_calls = {
            node.func.attr
            for node in ast.walk(launcher_tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertIn("evaluate_fixed_probes", launcher_calls)
        self.assertNotIn("run", launcher_calls)

        evaluator_tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
        evaluate_function = next(
            node
            for node in ast.walk(evaluator_tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "evaluate_fixed_probes"
        )
        evaluate_calls = {
            node.func.id
            for node in ast.walk(evaluate_function)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertNotIn("execute_training", evaluate_calls)
        self.assertNotIn("run", evaluate_calls)
        self.assertNotIn("_build_training_config", evaluate_calls)
        self.assertNotIn("_load_and_revalidate_preflight", evaluate_calls)

        evaluator_source = SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertLess(
            evaluator_source.index(
                "training_artifacts = _verify_final_checkpoint_artifacts("
            ),
            evaluator_source.index(
                "processor, embodiment, modality_configs = "
                "stage26._build_runtime_processor("
            ),
        )

        pbs_source = EVALUATION_PBS_PATH.read_text(encoding="utf-8")
        self.assertIn("stage28_h2_evaluation_only.py", pbs_source)
        self.assertNotIn(" --train ", pbs_source)
        self.assertNotIn("execute_training", pbs_source)
        self.assertNotIn("optimizer.step", pbs_source)

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
