from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, "scripts")
import stage45f_pi0172_rtx4080_validation_e as execution


class PI0172RTX4080ValidationETests(unittest.TestCase):
    def test_plan_binds_six_arms_full_matrix_and_recovery_ceiling(self) -> None:
        plan = execution.plan()
        self.assertEqual(plan["pi_gate"], "PI_GATE_STATE_0172")
        self.assertEqual(len(plan["checkpoint_arms"]), 6)
        self.assertEqual(plan["qualification"]["model_side_total"], 30)
        self.assertEqual(plan["full_generation_nominal"], {"model_loads": 6, "conditioning": 36, "action_head": 144, "model_side_total": 180, "raw_outputs": 36, "derived_outputs": 36})
        self.assertEqual(plan["recovery_ceiling"], {"model_load_attempts": 8, "conditioning_attempts": 40, "action_head_attempts": 160, "model_side_attempts": 200})
        self.assertEqual(plan["forbidden"]["remaining_30_rtx_cells"], 0)
        self.assertEqual(plan["h200_portability"]["allocation"], 8)
        self.assertTrue(plan["h200_portability"]["descriptive_only"])
        self.assertFalse(plan["h200_portability"]["submission"])

    def test_manifest_matches_code_identity_and_ledger_contract(self) -> None:
        manifest = json.loads(Path("manifests/STAGE45F_PI0172_RTX4080_VALIDATION_E_V1.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["authority"]["source_head"], execution.SOURCE_HEAD)
        self.assertEqual(manifest["authority"]["pi_gate"], execution.PI_GATE)
        self.assertEqual([arm["label"] for arm in manifest["scope"]["checkpoint_arms"]], [arm["label"] for arm in execution.ARM_SPECS])
        self.assertEqual([arm["tree_sha256"] for arm in manifest["scope"]["checkpoint_arms"]], [arm["tree_sha256"] for arm in execution.ARM_SPECS])
        self.assertEqual(manifest["frozen_runtime"]["groot_manifest_sha256"], execution.GROOT_MANIFEST_SHA)
        self.assertEqual(manifest["budgets"]["model_side_attempts_max"], execution.MODEL_SIDE_ATTEMPTS_MAX)
        self.assertIn("MODEL_LOAD BEGIN/END", manifest["durable_execution_ledger"]["required_events"])

    def test_zero_model_static_risk_audit_passes(self) -> None:
        audit = execution.static_risk_audit()
        self.assertEqual(audit["status"], "PASS", audit)
        self.assertEqual(audit["top_level_forbidden_imports"], [])
        self.assertEqual(audit["forbidden_call_hits"], [])
        self.assertFalse(audit["model_imported"])
        self.assertEqual(audit["h200_submissions"], 0)
        self.assertFalse(audit["holdout_f_read"])

    def test_top_level_imports_are_standard_library_only(self) -> None:
        source = Path("scripts/stage45f_pi0172_rtx4080_validation_e.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        top_imports = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                top_imports.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                top_imports.add(node.module.split(".", 1)[0])
        self.assertFalse(top_imports.intersection(execution.FORBIDDEN_TOP_LEVEL_IMPORTS))

    def test_checkpoint_namespaces_are_explicit(self) -> None:
        checkpoint = Path("/tmp/checkpoint-2000")
        self.assertEqual(execution.checkpoint_hash_prefix(execution.ARM_SPECS[0], checkpoint), "checkpoint-2000")
        self.assertEqual(execution.checkpoint_hash_prefix(execution.ARM_SPECS[1], checkpoint), "stage45e_legacy/checkpoint-2000")
        self.assertEqual(execution.checkpoint_hash_prefix(execution.ARM_SPECS[3], checkpoint), "stage45e_l1/checkpoint-2000")

    def test_tree_digest_is_deterministic_and_rejects_symlinked_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.txt").write_text("a", encoding="utf-8")
            first = execution.tree_digest(execution.tree_entries(root, "root"))
            second = execution.tree_digest(execution.tree_entries(root, "root"))
            self.assertEqual(first, second)
            (root / "link").symlink_to("a.txt")
            with self.assertRaises(execution.PI0172Error):
                execution.tree_entries(root, "root")

    def test_ledger_is_append_only_and_records_begin_end(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "execution_ledger.jsonl"
            ledger = execution.ExecutionLedger(path, "GPU-test")
            token = ledger.begin("MODEL_LOAD", phase="MODEL_LOAD", checkpoint="Legacy(seed2222)@2000", free_vram_mib=[14000, 14000])
            ledger.end(token, phase="MODEL_LOAD", checkpoint="Legacy(seed2222)@2000", load_status="COMPLETED")
            records = execution._jsonl(path)
            self.assertEqual([record["sequence"] for record in records], list(range(1, len(records) + 1)))
            self.assertEqual(ledger.count_begin("MODEL_LOAD"), 1)
            self.assertEqual(records[-1]["status"], "END")
            self.assertEqual(records[-1]["torch_memory"]["torch_max_memory_allocated"], None)


if __name__ == "__main__":
    unittest.main()
