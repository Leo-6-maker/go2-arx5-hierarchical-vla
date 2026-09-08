from __future__ import annotations

import ast
import importlib.util
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "stage33_bottleneck_map.py"
PBS = REPO / "hpc/pbs/stage33_bottleneck_map.pbs"
SPEC = importlib.util.spec_from_file_location("stage33_bottleneck_map_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Stage33ContractTests(unittest.TestCase):
    def test_protocol_freezes_dual_track_budgets_and_boundaries(self) -> None:
        plan = MODULE.protocol_plan()
        MODULE.assert_protocol_static(plan)
        self.assertEqual(plan["ma"]["new_seeds"], list(range(2226, 2239)))
        self.assertEqual(plan["ma"]["maximum_new_sampler_calls"], 26)
        self.assertEqual(plan["mb"]["maximum_direct_calls"], 90)
        self.assertEqual(plan["mb"]["conditions"], list(MODULE.MB_CONDITIONS))
        self.assertTrue(all(value is False for value in plan["execution_boundary"].values()))

    def test_mb_intervention_plan_is_exact_and_deterministic(self) -> None:
        first = MODULE.mb_intervention_plan()
        second = MODULE.mb_intervention_plan()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 90)
        self.assertEqual(
            {(row["checkpoint"], row["probe_id"], row["condition"]) for row in first},
            {
                (checkpoint, probe, condition)
                for checkpoint in MODULE.MB_CHECKPOINTS
                for probe in MODULE.STAGE26_PROBE_IDS
                for condition in MODULE.MB_CONDITIONS
            },
        )
        seeds = [row["cpu_generator_seed"] for row in first if row["condition"].startswith(("B_", "C_", "D_"))]
        self.assertEqual(len(seeds), len(set(seeds)))

    def test_stage34_branch_priority_is_deterministic(self) -> None:
        ma = {"aggregates": {"exposure_robust_positive": True, "classification": "EXPOSURE_ROBUST_POSITIVE"}}
        mb = {"aggregates": {"inactive_coupling_strong": True}}
        mc = {"mapping_status": "QUALIFIED"}
        wa = {"classification": "NO_VALID_SUPPORT_AWARE_CONTINUOUS_COHORT"}
        self.assertEqual(MODULE.choose_stage34_branch(ma, mb, mc, wa)["selected_branch"], "M")
        mb["aggregates"]["inactive_coupling_strong"] = False
        self.assertEqual(MODULE.choose_stage34_branch(ma, mb, mc, wa)["selected_branch"], "S")
        ma["aggregates"]["exposure_robust_positive"] = False
        ma["aggregates"]["classification"] = "EXPOSURE_ROBUST_MIXED"
        self.assertEqual(MODULE.choose_stage34_branch(ma, mb, mc, wa)["selected_branch"], "D")

    def test_hex_validator_is_fail_closed(self) -> None:
        self.assertEqual(MODULE._validate_hex("a" * 64, "x"), "a" * 64)
        with self.assertRaises(MODULE.Stage33Error):
            MODULE._validate_hex("g" * 64, "x")
        with self.assertRaises(MODULE.Stage33Error):
            MODULE._validate_hex("a" * 63, "x")

    def test_no_eager_ml_imports(self) -> None:
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        top_imports = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                top_imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                top_imports.append(node.module or "")
        self.assertNotIn("torch", top_imports)
        self.assertNotIn("transformers", top_imports)

    def test_pbs_is_single_h200_and_has_no_replacement_submission(self) -> None:
        text = PBS.read_text(encoding="utf-8")
        self.assertIn("#PBS -P CFP03-CF-004", text)
        self.assertIn("#PBS -l select=1:ncpus=12:ngpus=1", text)
        self.assertIn("--mode ma", text)
        self.assertIn("--mode mb", text)
        self.assertIn("--mode mc", text)
        self.assertIn("--mode wa", text)
        self.assertNotIn("qsub", text)

    def test_output_overlap_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            immutable = root / "immutable"
            immutable.mkdir()
            with self.assertRaises(MODULE.Stage33Error):
                MODULE._ensure_external_output(immutable / "nested", (immutable,))


if __name__ == "__main__":
    unittest.main()
