import copy
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO = Path(__file__).parents[1]
SCRIPT = REPO / "scripts" / "stage45f_pi0194_candidate_eval.py"
sys.path.insert(0, str(REPO / "src"))
SPEC = importlib.util.spec_from_file_location("stage45f_pi0196_corelab_stage43_eval", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PI0196CorelabStage43ReuseTests(unittest.TestCase):
    def test_plan_delegates_runtime_and_keeps_frozen_budget(self):
        plan = MODULE.plan()
        self.assertEqual(plan["status"], "READY_NOT_EXECUTED")
        self.assertTrue(plan["ownership"]["stage43_stage40_runtime"] == "delegated")
        self.assertNotIn("independent_native_loop", plan["ownership"])
        self.assertEqual(plan["matrix"]["cells"], 15)
        self.assertEqual(plan["matrix"]["gr00t_forwards"], 60)
        self.assertEqual(plan["matrix"]["reference_updates"], 45)
        self.assertEqual(plan["matrix"]["native_ticks"], 3675)
        self.assertEqual(plan["cell_contract"]["separate_model_only_inference"], 0)

    def test_static_contract_and_metric_regression_are_behavioral_admissions(self):
        static = MODULE.static_contract()
        self.assertEqual(static["status"], "PASS_STATIC_NOT_EXECUTED")
        self.assertEqual(static["top_level_forbidden_imports"], [])
        self.assertTrue(static["stage43_runner_reused"])
        metrics = MODULE.metric_regression()
        self.assertEqual(metrics["status"], "PASS")
        self.assertTrue(metrics["path_length_not_tracking_rms"])
        self.assertNotEqual(metrics["achieved_path_length_m"], metrics["tracking_rms_m"])

    def test_candidate_inventory_rejects_extra_and_symlink_entries(self):
        with tempfile.TemporaryDirectory(prefix="pi0196_inventory_") as directory:
            root = Path(directory) / "checkpoint"
            root.mkdir()
            for relative in MODULE.CHECKPOINT_FILES:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(relative.encode("utf-8"))
            stats = root / "experiment_cfg" / "structured_action_statistics.json"
            candidate = copy.deepcopy(MODULE.CANDIDATES["PARENT"])
            entries = MODULE._tree_entries(root, candidate["tree_prefix"])
            candidate["tree_sha256"] = MODULE._tree_digest(entries)
            candidate["stats_sha256"] = MODULE.sha256_file(stats)
            with mock.patch.dict(MODULE.CANDIDATES, {"PARENT": candidate}, clear=False):
                verified = MODULE._verify_candidate_tree(root, "PARENT", {"arms": {}})
                self.assertEqual(verified["file_count"], 19)
                (root / "extra.bin").write_bytes(b"extra")
                with self.assertRaises(MODULE.AdapterError):
                    MODULE._verify_candidate_tree(root, "PARENT", {"arms": {}})
                (root / "extra.bin").unlink()
                os.symlink(root / "config.json", root / "symlink.bin")
                with self.assertRaisesRegex(MODULE.AdapterError, "symlink"):
                    MODULE._verify_candidate_tree(root, "PARENT", {"arms": {}})

    def test_achieved_path_and_tracking_are_derived_from_distinct_rows(self):
        with tempfile.TemporaryDirectory(prefix="pi0196_metrics_") as directory:
            root = Path(directory)
            MODULE.write_json(
                root / "condition_setup.json",
                {"event": {"s0": {"eef_world_m": [0.0, 0.0, 0.0]}}},
            )
            rows = [
                {"achieved_pos": [0.5, 0.0, 0.0], "desired_pos": [0.4, 0.0, 0.0]},
                {"achieved_pos": [0.5, 0.5, 0.0], "desired_pos": [0.4, 0.4, 0.0]},
            ]
            MODULE.write_json(root / "rows_before_handoff.json", {"segments": [rows]})
            MODULE.write_json(root / "rows_after_handoff.json", {"segments": []})
            result = MODULE._derive_achieved_metrics(root)
            self.assertAlmostEqual(result["achieved_path_length_m"], 1.0)
            self.assertAlmostEqual(result["tracking_rms_m"], (0.015) ** 0.5)
            self.assertNotEqual(result["achieved_path_length_m"], result["tracking_rms_m"])

    def test_scorecard_is_fail_closed_for_incomplete_or_rejected_cells(self):
        def aggregate(progress, finals, status="PASS"):
            return {
                "candidate": "E",
                "complete_cell_count": 5 if status == "PASS" else 4,
                "scientific_reject_count": 0 if status == "PASS" else 1,
                "scientific_abort_count": 0,
                "engineering_blocked_count": 0,
                "mean_progress_m": {"mean": progress},
                "mean_final_distance_m": {"mean": sum(finals) / len(finals)},
                "per_condition": {
                    condition: {
                        "status": status,
                        "final_target_distance_m": finals[index],
                        "absolute_progress_m": progress,
                    }
                    for index, condition in enumerate(MODULE.CONDITION_ORDER)
                },
            }

        parent = aggregate(0.1, [0.5] * 5)
        e = aggregate(0.2, [0.4, 0.41, 0.42, 0.43, 0.44])
        s20 = aggregate(0.25, [0.3, 0.31, 0.45, 0.33, 0.34])
        result = MODULE._classify_scorecard({"PARENT": parent, "E": e, "S20": s20})
        self.assertEqual(result["classification"], "STATE_SUPPORT_SUPPORTED")
        self.assertEqual(result["matched_condition_s20_final_distance_wins"], 4)
        rejected = aggregate(0.25, [0.3, 0.31, 0.45, 0.33, 0.34], status="SCIENTIFIC_REJECT")
        result = MODULE._classify_scorecard({"PARENT": parent, "E": e, "S20": rejected})
        self.assertEqual(result["classification"], "MIXED_FOR_PI_REVIEW")


if __name__ == "__main__":
    unittest.main()
