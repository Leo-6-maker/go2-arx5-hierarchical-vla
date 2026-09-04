"""Deterministic tests for the Stage26 H1 C1 preparation boundary."""

from __future__ import annotations

import ast
import importlib.util
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "stage26_h1_c1_fixed_noise_local_flow.py"
SPEC = importlib.util.spec_from_file_location("stage26_h1_c1_under_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Stage26H1C1Tests(unittest.TestCase):
    def test_static_plan_reuses_exact_stage23_grid(self) -> None:
        plan = MODULE.protocol_plan()
        MODULE.assert_protocol_static(plan)
        self.assertEqual(plan["schema"], MODULE.STAGE26_SCHEMA)
        self.assertTrue(plan["stage23_c1_protocol_reused"])
        self.assertEqual(plan["probe_contract"]["probe_count"], 9)
        self.assertEqual(plan["execution_budget"]["maximum_c1_flow_evaluations"], 18)
        self.assertTrue(plan["readiness"]["h1_c1_execution_ready"])
        self.assertTrue(plan["readiness"]["runtime_identity_revalidation_required"])
        self.assertFalse(plan["readiness"]["execution_started"])
        self.assertEqual(
            [(item["noise_seed"], item["t_discretized"]) for item in plan["probe_contract"]["probes"]],
            [
                (2223, 67),
                (2223, 369),
                (2223, 783),
                (2224, 67),
                (2224, 369),
                (2224, 783),
                (2225, 67),
                (2225, 369),
                (2225, 783),
            ],
        )

    def test_static_plan_has_no_enabled_execution_boundary(self) -> None:
        plan = MODULE.protocol_plan()
        self.assertTrue(all(value is False for value in plan["execution_boundary"].values()))
        self.assertTrue(plan["output_schema"]["descriptive_only"])
        self.assertTrue(plan["output_schema"]["no_new_scientific_pass_threshold"])
        self.assertEqual(
            plan["runtime_precision"]["device_cast"],
            "model.to(device=args.device, dtype=torch.bfloat16)",
        )

    def test_source_map_and_direct_runner_are_pre_sampler(self) -> None:
        source_map = MODULE._source_map_static()
        self.assertEqual(source_map["status"], "PASS")
        self.assertTrue(source_map["pre_sampler"])
        self.assertFalse(source_map["complete_model_forward_called"])
        self.assertFalse(source_map["free_running_sampler_called"])

        tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
        function = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "_teacher_forced_local_flow"
        )
        calls = [
            node.func.attr
            for node in ast.walk(function)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        ]
        self.assertNotIn("get_action", calls)
        self.assertNotIn("get_action_with_features", calls)
        self.assertNotIn("forward", calls)

    def test_canonical_probes_are_created_once_and_reused(self) -> None:
        tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
        run_function = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "_run_one_checkpoint"
        )
        run_canonical_calls = [
            node
            for node in ast.walk(run_function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_canonical_probe_inputs"
        ]
        self.assertEqual(run_canonical_calls, [])

        execute_function = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "execute"
        )
        execute_canonical_calls = [
            node
            for node in ast.walk(execute_function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_canonical_probe_inputs"
        ]
        self.assertEqual(len(execute_canonical_calls), 1)

        run_fixed_probe_keywords = [
            keyword.arg
            for node in ast.walk(execute_function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_run_one_checkpoint"
            for keyword in node.keywords
        ]
        self.assertEqual(run_fixed_probe_keywords.count("fixed_probes"), 2)

    def test_tree_digest_is_order_independent_but_bytes_sensitive(self) -> None:
        entries = [
            {"path": "b", "bytes": 2, "sha256": "bb"},
            {"path": "a", "bytes": 1, "sha256": "aa"},
        ]
        reversed_entries = list(reversed(entries))
        self.assertEqual(
            MODULE._tiny_tree_digest(entries),
            MODULE._tiny_tree_digest(reversed_entries),
        )
        changed = [dict(item) for item in entries]
        changed[0]["bytes"] = 3
        self.assertNotEqual(
            MODULE._tiny_tree_digest(entries),
            MODULE._tiny_tree_digest(changed),
        )

    def test_regular_file_manifest_rejects_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "regular").write_bytes(b"ok")
            link = root / "link"
            try:
                link.symlink_to(root / "regular")
            except (NotImplementedError, OSError):
                self.skipTest("symlinks are unavailable in this test environment")
            with self.assertRaises(ValueError):
                MODULE._regular_file_entries(root)

    def test_checkpoint_preflight_rejects_wrong_index_without_tensor_read(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary)
            (checkpoint / "model.safetensors.index.json").write_text(
                '{"weight_map": {}}\\n', encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                MODULE._checkpoint_preflight("BASE", checkpoint)


if __name__ == "__main__":
    unittest.main()
