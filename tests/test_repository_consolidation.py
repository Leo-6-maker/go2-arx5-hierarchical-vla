"""CPU-only regression checks for the repository consolidation layer."""

from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT_JSON = ROOT / "reports" / "REPOSITORY_CONSOLIDATION_AUDIT.json"


class RepositoryConsolidationTest(unittest.TestCase):
    def test_navigation_documents_are_present(self) -> None:
        required = (
            ROOT / "README.md",
            ROOT / "docs" / "RESEARCH_TIMELINE.md",
            ROOT / "docs" / "ACCEPTED_BASELINES.md",
            ROOT / "docs" / "REPOSITORY_GUIDE.md",
            ROOT / "docs" / "HISTORICAL_ARCHIVAL_RECOMMENDATION.md",
            ROOT / "scripts" / "README.md",
            ROOT / "reports" / "README.md",
            ROOT / "reports" / "REPOSITORY_CONSOLIDATION_AUDIT.md",
            ROOT / "hpc" / "README.md",
        )
        for path in required:
            with self.subTest(path=path):
                self.assertTrue(path.is_file())

    def test_readme_exposes_the_frozen_architecture_and_hold(self) -> None:
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        for marker in (
            "Vision + Language + Proprioception",
            "GR00T N1.7",
            "StructuredActionV1",
            "SafetyGate",
            "reference-only receding-horizon handoff",
            "general trajectory-tracking WBC",
            "bd0f02934e680789cf38adacb838e1ea555c707d",
            "Stage45",
            "H200",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, text)

    def test_audit_json_is_machine_readable_and_resource_closed(self) -> None:
        payload = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema"], "REPOSITORY_CONSOLIDATION_AUDIT_V1")
        self.assertEqual(payload["status"], "PASS")
        self.assertEqual(payload["source_baseline"]["accepted_head"], "bd0f02934e680789cf38adacb838e1ea555c707d")
        self.assertEqual(payload["main_comparison"]["ahead_by"], 183)
        self.assertEqual(payload["main_comparison"]["behind_by"], 0)
        self.assertTrue(payload["main_comparison"]["forward_only"])
        boundary = payload["resource_boundary"]
        self.assertEqual(boundary["gr00t_forwards"], 0)
        self.assertEqual(boundary["wbc_steps"], 0)
        self.assertEqual(boundary["reference_updates"], 0)
        self.assertEqual(boundary["training_backward_optimizer_steps"], 0)
        self.assertEqual(boundary["new_data_generation"], 0)
        self.assertFalse(boundary["gpu_or_simulator_execution"])
        self.assertEqual(boundary["pbs_submissions"], 0)

    def test_historical_stage_paths_and_branches_are_preserved(self) -> None:
        timeline = (ROOT / "docs" / "RESEARCH_TIMELINE.md").read_text(encoding="utf-8")
        for path in (
            "scripts/stage40_corelab_evidence.py",
            "scripts/stage41_corelab_long_horizon.py",
            "scripts/stage42_feedback_causality.py",
            "scripts/stage43_corelab_initial_condition.py",
            "scripts/stage44r_corelab_attribution_repair.py",
        ):
            with self.subTest(path=path):
                self.assertTrue((ROOT / path).is_file())
        for stage in ("40", "41", "42", "43", "44", "44R"):
            with self.subTest(stage=stage):
                self.assertIn(f"| {stage} |", timeline)
        audit = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
        self.assertTrue(audit["tree_inventory"]["historical_paths_preserved"])
        self.assertEqual(
            set(audit["github_inventory"]["diverged_stage_branches"]),
            {"stage39-vis/offline-mp4", "stage40-dev/a40-bringup"},
        )

    def test_no_binary_like_artifacts_are_added_to_the_checkout(self) -> None:
        output = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True)
        forbidden = (".pt", ".pth", ".ckpt", ".pkl", ".jit", ".npz", ".npy", ".parquet", ".mp4", ".usd")
        paths = [line for line in output.splitlines() if line.endswith(forbidden)]
        self.assertEqual(paths, [])


if __name__ == "__main__":
    unittest.main()
