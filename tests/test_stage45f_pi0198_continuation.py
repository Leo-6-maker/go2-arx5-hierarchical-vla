import importlib.util
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).parents[1]
SCRIPT = REPO / "scripts" / "stage45f_pi0194_candidate_eval.py"
SPEC = importlib.util.spec_from_file_location("stage45f_pi0198_continuation", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class _ManifestStub:
    @staticmethod
    def verify_condition_manifest(manifest, root):
        assert manifest["schema"] == MODULE.CONTINUATION_SCHEMA + "_TEST_MANIFEST"


def _cell_fixture(root: Path, *, decision: str, error: dict, aborted: bool = False) -> Path:
    import numpy as np

    root.mkdir()
    MODULE.write_json(root / "condition_manifest.json", {"schema": MODULE.CONTINUATION_SCHEMA + "_TEST_MANIFEST"})
    MODULE.write_json(
        root / "condition_result.json",
        {
            "schema": "CORELAB_STAGE43_CONDITION_RESULT_V1",
            "status": "BLOCKED",
            "cases": 1,
            "gr00t_forward_count": 1,
            "reference_update_count": 0,
            "wbc_step_count": 0,
            "wbc_segment_lengths": [],
            "safety_gate_decisions": [decision],
            "error": error,
        },
    )
    MODULE.write_json(
        root / "safety_gate.json",
        {
            "schema": "CORELAB_STAGE43_SAFETYGATE_EVIDENCE_V1",
            "decisions": [decision],
            "cases": [{"decision": decision, "issues": [] if decision != "REJECT" else [{"code": "LINEAR_SPEED_REJECT"}], "metrics": {}, "provenance": {}}],
        },
    )
    MODULE.write_json(root / "counter_observations.json", {"events": [], "final": {"gr00t_forward_count": 1, "reference_update_count": 0, "wbc_step_count": 0}})
    np.savez_compressed(root / "raw_action_d132.npz", q0=np.zeros((1, 40, 132), dtype=np.float32))
    np.savez_compressed(root / "decoded_action_d10_h40.npz", q0=np.zeros((40, 10), dtype=np.float64))
    if aborted:
        MODULE.write_json(root / "rows_before_handoff.json", {"segments": [[{"done": True, "reset_buf": False, "termination_code": "NATIVE_TASK_TERMINATION"}]]})
    return root


class PI0198ContinuationTests(unittest.TestCase):
    def test_fixed_order_excludes_immutable_parent_c0(self):
        self.assertEqual(len(MODULE._continuation_order()), 14)
        self.assertNotIn(MODULE.PRIOR_CELL, MODULE._continuation_order())
        self.assertEqual(MODULE._continuation_order()[0], ("PARENT", "C1_X_PLUS_5CM"))

    def test_reject_is_persisted_as_terminal_cell(self):
        with tempfile.TemporaryDirectory() as directory:
            root = _cell_fixture(Path(directory) / "cell", decision="REJECT", error={"message": "SAFETYGATE_REJECT_C0", "type": "RuntimeError"})
            record = MODULE._persist_cell_result(root, "E", "C0_NOMINAL", 139, stage43_evidence=_ManifestStub())
            self.assertEqual(record["status"], "SCIENTIFIC_REJECT")
            self.assertTrue((root / "pi0198_cell_seal.json").is_file())
            self.assertTrue(record["post_persistence_native_teardown"])

    def test_abort_is_persisted_and_not_engineering_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = _cell_fixture(Path(directory) / "cell", decision="WARN", error={"message": "NATIVE_TASK_TERMINATION", "type": "RuntimeError"}, aborted=True)
            record = MODULE._persist_cell_result(root, "S20", "C1_X_PLUS_5CM", 2, stage43_evidence=_ManifestStub())
            self.assertEqual(record["status"], "SCIENTIFIC_ABORT")

    def test_incomplete_rc139_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = _cell_fixture(Path(directory) / "cell", decision="REJECT", error={"message": "SAFETYGATE_REJECT_C0", "type": "RuntimeError"})
            (root / "condition_manifest.json").unlink()
            with self.assertRaises(MODULE.AdapterError):
                MODULE._persist_cell_result(root, "E", "C0_NOMINAL", 139, stage43_evidence=_ManifestStub())

    def test_engineering_record_is_terminal_stop_state(self):
        with tempfile.TemporaryDirectory() as directory:
            record = MODULE._persist_engineering_cell(Path(directory) / "cell", "E", "C0_NOMINAL", runner_code=7, message="launch failed")
            self.assertEqual(record["status"], "ENGINEERING_BLOCKED")

    def test_budget_helper_enforces_original_ceiling(self):
        totals = MODULE._budget_totals({"PARENT": {"gr00t_forward_count": 4, "reference_update_count": 2, "wbc_step_count": 147}, "E": {"gr00t_forward_count": 56, "reference_update_count": 42, "wbc_step_count": 3430}, "S20": {"gr00t_forward_count": 0, "reference_update_count": 0, "wbc_step_count": 0}})
        self.assertEqual(totals, {"gr00t_forwards": 60, "reference_updates": 44, "native_ticks": 3577})
        self.assertLessEqual(totals["gr00t_forwards"], MODULE.MAX_FORWARDS)
        self.assertLessEqual(totals["reference_updates"], MODULE.MAX_UPDATES)
        self.assertLessEqual(totals["native_ticks"], MODULE.MAX_TICKS)

    def test_child_uses_mature_runner_in_a_fresh_process(self):
        base = {"repository": {"head": "d042ecd51af437040103892d9b2e2a6acdd00a9b"}, "artifacts": {key: {"path": "/tmp/" + key} for key in ("wbc_source", "trajectory", "wbc_arm_checkpoint", "wbc_dog_checkpoint", "stage10_root", "base_model_path", "dataset_dir", "modality_config", "cosmos_snapshot_path")}}
        binding = {"candidate_paths": {name: "/tmp/" + name for name in MODULE.CANDIDATE_ORDER}, "paths": {"model_python": "/tmp/python"}}
        command = MODULE._child_command(base=base, overlay_path=Path("/tmp/overlay.json"), candidate="E", condition="C0_NOMINAL", condition_root=Path("/tmp/cell"), binding=binding, expected_head="head", gpu_uuid="GPU-test", sim_device="cuda:0", seed=4242)
        self.assertEqual(command[:2], [MODULE.sys.executable, "-B"])
        self.assertIn("runpy.run_path", command[3])
        self.assertIn(str(MODULE.RUNNER_SCRIPT), command[3])


if __name__ == "__main__":
    unittest.main()
