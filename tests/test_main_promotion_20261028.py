"""CPU-only repository navigation/retirement scope checks; no network or science.

Added with the consolidation. Execution was unavailable in that connector session;
this file's existence is not a test PASS.
"""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts/maintenance/retire_merged_branches_20261028.py"
PLAN_PATH = ROOT / "docs/maintenance/2026-10-28/BRANCH_RETIREMENT.json"


def load_helper():
    spec = importlib.util.spec_from_file_location("retire_exact_oct28", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MainPromotionTest(unittest.TestCase):
    def test_current_navigation_targets_exist(self):
        for name in ("START_HERE.md", "AGENTS.md", "CURRENT_SCIENCE_STATUS.md",
                     "docs/EXPERIMENT_ROADMAP.md", "docs/CODE_MAP.md", "docs/REPOSITORY_WORKFLOW.md",
                     "docs/maintenance/2026-10-28/CONSOLIDATION_REPORT.md",
                     "infrastructure/workspace/INTEGRATION_REF_CHANGE_20261028.json"):
            with self.subTest(path=name):
                self.assertTrue((ROOT / name).is_file())

    def test_main_ref_promotion_does_not_claim_server_adoption(self):
        record = json.loads((ROOT / "infrastructure/workspace/INTEGRATION_REF_CHANGE_20261028.json").read_text())
        self.assertEqual(record["new_integration_branch"], "main")
        self.assertFalse(record["new_server_deployment"])
        self.assertFalse(record["new_scientific_authority"])
        self.assertFalse(record["force_push_or_reset"])

    def test_retirement_is_explicitly_pending_and_exact(self):
        module = load_helper()
        plan = json.loads(PLAN_PATH.read_text())
        self.assertEqual(module.validate_plan(plan), module.EXPECTED)
        self.assertEqual(plan["deleted_count"], 0)
        self.assertFalse(plan["archive_tags_created_by_this_session"])
        self.assertEqual(set(plan["keep"]), {"main", "pi0267/executor-decoupled-recovery"})

    def test_script_defaults_to_dry_run(self):
        module = load_helper()
        self.assertFalse(module.parser().parse_args([]).apply)
        self.assertTrue(module.parser().parse_args(["--apply"]).apply)

    def test_candidate_head_or_scope_change_rejected(self):
        module = load_helper()
        original = json.loads(PLAN_PATH.read_text())
        changed = copy.deepcopy(original)
        changed["candidates"][0]["sha"] = "0" * 40
        with self.assertRaises(RuntimeError):
            module.validate_plan(changed)
        changed = copy.deepcopy(original)
        changed["candidates"][0]["name"] = "main"
        with self.assertRaises(RuntimeError):
            module.validate_plan(changed)

    def test_archived_entries_and_scientific_history_remain(self):
        archive = ROOT / "docs/archive/2026-10-28/pre-consolidation"
        for name in ("README.experimental.md", "README.main.md", "cpu.experimental.yml",
                     "START_HERE.md", "CURRENT_SCIENCE_STATUS.md", "AGENTS.md",
                     "PROJECT_STATUS.md", "RESEARCH_INDEX.md", "REPOSITORY_WORKFLOW.md"):
            with self.subTest(path=name):
                self.assertTrue((archive / name).is_file())
        self.assertTrue((ROOT / "docs/handover/2026-10-25/AUTONOMOUS_PROGRESS.md").is_file())
        self.assertFalse((ROOT / ".github/workflows/consolidation-snapshot-20261028.yml").exists())

    def test_cpu_workflow_keeps_both_isolated_suites(self):
        workflow = (ROOT / ".github/workflows/cpu.yml").read_text()
        self.assertIn("python -m pytest -q scripts/pi0292", workflow)
        self.assertIn("python -m pytest -q scripts/pi0294", workflow)


if __name__ == "__main__":
    unittest.main()
