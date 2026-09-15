import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "stage45f_pi0194_candidate_eval.py"
sys.path.insert(0, str(SCRIPT.parents[1] / "src"))
SPEC = importlib.util.spec_from_file_location("stage45f_pi0194_candidate_eval", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PI0194CandidateEvalTests(unittest.TestCase):
    def test_plan_is_exact_fixed_matrix_and_execution_free(self):
        plan = MODULE.plan()
        self.assertEqual(plan["status"], "READY_NOT_EXECUTED")
        self.assertEqual(plan["matrix"]["candidate_order"], ["PARENT", "E", "S20"])
        self.assertEqual(plan["matrix"]["condition_order"], list(MODULE.CONDITION_ORDER))
        self.assertEqual(plan["matrix"]["cells"], 15)
        self.assertEqual(plan["matrix"]["gr00t_forwards"], 60)
        self.assertEqual(plan["matrix"]["reference_updates"], 45)
        self.assertEqual(plan["matrix"]["native_ticks"], 3675)
        self.assertEqual(plan["cell_contract"]["separate_model_only_inference"], 0)
        self.assertEqual(plan["hard_zeroes"]["training"], 0)

    def test_static_contract_proves_frozen_files_and_no_runtime_imports(self):
        result = MODULE.static_contract()
        self.assertEqual(result["status"], "PASS_STATIC_NOT_EXECUTED")
        self.assertEqual(result["top_level_forbidden_imports"], [])
        self.assertEqual(result["forbidden_call_hits"], [])
        self.assertEqual(result["model_calls_in_plan"], 0)
        self.assertEqual(result["simulator_steps_in_plan"], 0)
        self.assertEqual(result["training_in_plan"], 0)
        self.assertEqual(set(result["frozen_files"]), set(MODULE.FROZEN_FILES))

    def test_tree_closure_rejects_symlink_and_extra_entries(self):
        with tempfile.TemporaryDirectory(prefix="pi0194_tree_") as directory:
            root = Path(directory) / "checkpoint"
            root.mkdir()
            (root / "one").write_bytes(b"one")
            entries = MODULE._tree_entries(root, "test/checkpoint")
            self.assertEqual([item["path"] for item in entries], ["test/checkpoint/one"])
            (root / "extra").write_bytes(b"extra")
            self.assertEqual(len(MODULE._tree_entries(root, "test/checkpoint")), 2)
            os.symlink(root / "one", root / "symlink")
            with self.assertRaisesRegex(MODULE.AdapterError, "symlink"):
                MODULE._tree_entries(root, "test/checkpoint")

    def test_model_metrics_are_derived_from_one_raw_and_decoded_query(self):
        import numpy as np

        class _Tensor:
            def __init__(self, value):
                self.value = value

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self.value

            def reshape(self, *shape):
                return _Tensor(self.value.reshape(*shape))

        class _Native:
            stage21_anchor_world = _Tensor(np.asarray([0.1, 0.2, 0.3], dtype=np.float32))

        with tempfile.TemporaryDirectory(prefix="pi0194_metric_") as directory:
            artifact = Path(directory) / "q0.npz"
            raw = np.zeros((1, 40, 132), dtype=np.float32)
            decoded = np.zeros((40, 10), dtype=np.float64)
            decoded[:, 0] = np.linspace(0.0, 0.2, 40)
            np.savez_compressed(artifact, raw_universal_action=raw)
            cases = [{
                "request_id": "C0_NOMINAL-q0",
                "response": {"artifact": str(artifact)},
                "decoded": decoded,
                "safety_gate": {"decision": "ACCEPT", "issues": []},
                "body_sha256": "body",
                "wrist_sha256": "wrist",
                "state_v1_sha256": "state",
            }]
            from go2arx5_vla.evaluation.trajectory_metrics import trajectory_metrics

            result = MODULE._model_metrics(cases, _Native(), np, trajectory_metrics)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["raw_shape"], [1, 40, 132])
            self.assertEqual(result[0]["decoded_shape"], [40, 10])
            self.assertFalse(result[0]["target_xyz_in_model_input"])
            self.assertGreater(result[0]["predicted_path_length_m"], 0.0)
            self.assertEqual(result[0]["safety_gate"]["decision"], "ACCEPT")

    def test_scorecard_uses_three_matched_conditions_and_fail_closed_science_failures(self):
        with tempfile.TemporaryDirectory(prefix="pi0194_scorecard_") as directory:
            root = Path(directory)
            for candidate in MODULE.CANDIDATE_ORDER:
                for condition in MODULE.CONDITION_ORDER:
                    cell_root = root / candidate / condition
                    cell_root.mkdir(parents=True)
                    if candidate == "PARENT":
                        final = 0.50
                        progress = 0.10
                    elif candidate == "E":
                        final = {"C0_NOMINAL": 0.40, "C1_X_PLUS_5CM": 0.41, "C2_X_MINUS_5CM": 0.42, "C3_Y_PLUS_5CM": 0.43, "C4_Y_MINUS_5CM": 0.44}[condition]
                        progress = 0.20
                    else:
                        final = {"C0_NOMINAL": 0.30, "C1_X_PLUS_5CM": 0.31, "C2_X_MINUS_5CM": 0.45, "C3_Y_PLUS_5CM": 0.33, "C4_Y_MINUS_5CM": 0.34}[condition]
                        progress = 0.25
                    payload = {
                        "schema": MODULE.CELL_SCHEMA,
                        "status": "PASS",
                        "candidate": candidate,
                        "condition_id": condition,
                        "gr00t_forward_count": 4,
                        "reference_update_count": 3,
                        "wbc_step_count": 245,
                        "metrics": {
                            "absolute_progress_m": progress,
                            "final_target_error_m": final,
                            "minimum_target_error_m": final,
                            "tracking_rms_m": 0.01,
                        },
                        "model_metrics": [],
                    }
                    (cell_root / "cell_result.json").write_text(json.dumps(payload), encoding="utf-8")
            aggregates = {candidate: MODULE._aggregate_candidate(root / candidate, candidate) for candidate in MODULE.CANDIDATE_ORDER}
            scorecard = MODULE._classify_scorecard(aggregates)
            self.assertEqual(scorecard["classification"], "STATE_SUPPORT_SUPPORTED")
            self.assertEqual(scorecard["matched_condition_s20_final_distance_wins"], 4)

            reject = json.loads((root / "S20" / "C2_X_MINUS_5CM" / "cell_result.json").read_text())
            reject["status"] = "SCIENTIFIC_REJECT"
            reject["metrics"] = None
            reject["scientific_failure"] = {"classification": "SAFETY_GATE_REJECT"}
            (root / "S20" / "C2_X_MINUS_5CM" / "cell_result.json").write_text(json.dumps(reject), encoding="utf-8")
            aggregates["S20"] = MODULE._aggregate_candidate(root / "S20", "S20")
            scorecard = MODULE._classify_scorecard(aggregates)
            self.assertEqual(scorecard["classification"], "MIXED_FOR_PI_REVIEW")
            self.assertEqual(scorecard["S20"]["scientific_reject_count"], 1)


if __name__ == "__main__":
    unittest.main()
