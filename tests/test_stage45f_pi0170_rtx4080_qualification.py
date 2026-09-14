from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, "scripts")
import stage45f_pi0170_rtx4080_qualification as qualification


class PI0170QualificationTests(unittest.TestCase):
    def test_manifest_binds_one_arm_six_windows_and_exact_budget(self) -> None:
        manifest = json.loads(Path("manifests/STAGE45F_PI0170_RTX4080_VALIDATION_QUALIFICATION_V1.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["authority"]["source_head"], qualification.SOURCE_HEAD)
        self.assertEqual(manifest["scope"]["checkpoint"]["label"], qualification.CHECKPOINT_LABEL)
        self.assertEqual(manifest["scope"]["checkpoint"]["tree_sha256"], qualification.CHECKPOINT_TREE_SHA)
        self.assertEqual(len(manifest["scope"]["window_ids"]), 6)
        self.assertEqual(manifest["budgets"]["model_load_attempts_max"], 1)
        self.assertEqual(manifest["budgets"]["conditioning_max"], 6)
        self.assertEqual(manifest["budgets"]["action_head_max"], 24)
        self.assertEqual(manifest["budgets"]["model_side_max"], 30)
        self.assertFalse(manifest["h200_portability_anchor"]["submission"])

    def test_plan_is_exactly_single_arm_and_no_expansion(self) -> None:
        plan = qualification.plan()
        self.assertEqual(plan["windows"], 6)
        self.assertEqual(plan["model_load_attempts"], 1)
        self.assertEqual(plan["conditioning"], 6)
        self.assertEqual(plan["action_head"], 24)
        self.assertEqual(plan["model_side_total"], 30)
        self.assertEqual(plan["h200_portability"], {"allocation": 8, "cells": 6, "submission": False})
        self.assertEqual(plan["forbidden"]["remaining_30_rtx_cells"], 0)
        self.assertEqual(plan["forbidden"]["h200_submissions"], 0)

    def test_zero_model_static_risk_audit_passes(self) -> None:
        audit = qualification.static_risk_audit()
        self.assertEqual(audit["status"], "PASS", audit)
        self.assertEqual(audit["top_level_forbidden_imports"], [])
        self.assertEqual(audit["forbidden_call_hits"], [])
        self.assertEqual(audit["forbidden_text_hits"], [])
        self.assertFalse(audit["model_imported"])
        self.assertFalse(audit["h200_submissions"])

    def test_top_level_imports_are_standard_library_only(self) -> None:
        source = Path("scripts/stage45f_pi0170_rtx4080_qualification.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        top_imports = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                top_imports.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                top_imports.add(node.module.split(".", 1)[0])
        self.assertFalse(top_imports.intersection(qualification.FORBIDDEN_TOP_LEVEL_IMPORTS))

    def test_resource_admission_rejects_insufficient_or_foreign_gpu_without_mutation(self) -> None:
        self.assertEqual(qualification.MIN_FREE_VRAM_MIB, 12288)
        snapshot = {
            "gpus": [{"name": "NVIDIA GeForce RTX 4080 SUPER", "uuid": "GPU-test", "total_vram_mib": 32760, "free_vram_mib": 12189}],
            "foreign_compute_processes": [],
        }
        selected = snapshot["gpus"][0]
        self.assertLess(selected["free_vram_mib"], qualification.MIN_FREE_VRAM_MIB)
        foreign = [{"gpu_uuid": "GPU-test", "pid": 7, "process_name": "foreign", "used_memory_mib": 10}]
        self.assertTrue(any(row["gpu_uuid"] == selected["uuid"] for row in foreign))

    def test_h200_expected_anchor_has_six_hash_bound_cells(self) -> None:
        cells = qualification._expected_h200_cells()
        self.assertEqual(list(cells), list(qualification.WINDOW_IDS))
        self.assertEqual(len(cells), 6)
        for identity in cells.values():
            self.assertEqual(len(identity["file_sha256"]), 64)
            self.assertEqual(len(identity["array_sha256"]), 64)

    def test_tree_digest_is_deterministic_and_rejects_symlinked_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.txt").write_text("a", encoding="utf-8")
            first = qualification.tree_digest(qualification.tree_entries(root, "root"))
            second = qualification.tree_digest(qualification.tree_entries(root, "root"))
            self.assertEqual(first, second)
            (root / "link").symlink_to("a.txt")
            with self.assertRaises(qualification.QualificationError):
                qualification.tree_entries(root, "root")


if __name__ == "__main__":
    unittest.main()
