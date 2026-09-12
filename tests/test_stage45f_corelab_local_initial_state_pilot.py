import ast
import json
from pathlib import Path
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/stage45f_corelab_local_initial_state_pilot.py"
MANIFEST = ROOT / "manifests/STAGE45F_CORELAB_LOCAL_INITIAL_STATE_PILOT_V1.json"


class LocalInitialStatePilotStaticTests(unittest.TestCase):
    def test_manifest_contract(self):
        value = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(value["authority"]["pi_gate"], "PI_GATE_STATE_0152")
        self.assertEqual(value["authority"]["issue_comment"], 5647237688)
        self.assertEqual(value["authority"]["pr_pointer_comment"], 5647244579)
        self.assertEqual(value["repository"]["base_head"], "9fef9064979192b66161e91ed0e7d0ab505b0364")
        self.assertEqual(value["local_reference"]["seed"], 2223)
        self.assertEqual(value["budget"]["conditioning_passes"], 6)
        self.assertEqual(value["budget"]["action_head_calls"], 24)
        self.assertFalse(value["stage43_input"]["target_xyz_in_model_input"])
        self.assertEqual(value["authority"]["return_token"], "CORELAB_PI0152_POLICY_CHARACTERIZATION_FOR_PI_REVIEW")
        self.assertEqual(value["recovery"]["max_new_worker_attempts"], 2)

    def test_no_top_level_runtime_imports(self):
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT))
        forbidden = {"torch", "transformers", "gr00t"}
        for node in tree.body:
            if isinstance(node, ast.Import):
                names = {alias.name.split(".")[0] for alias in node.names}
                self.assertTrue(forbidden.isdisjoint(names), names)
            elif isinstance(node, ast.ImportFrom):
                self.assertNotIn((node.module or "").split(".")[0], forbidden)

    def test_required_modes_and_fail_closed_boundary_are_source_bound(self):
        text = SCRIPT.read_text(encoding="utf-8")
        for token in ("--mode", "audit", "preflight", "execute", "failure_seal.json", "target_xyz_in_model_input", "--ffmpeg-runtime", "train_pixel_qualification"):
            self.assertIn(token, text)
        self.assertIn("action_head_calls\"] == 24", text)
        self.assertIn("conditioning_passes\"] == 6", text)

    def test_uuid_allowlist_accepts_nondefault_live_device_and_prefers_32gb(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("pilot", SCRIPT)
        pilot = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pilot)
        rows = [
            {"index": 0, "name": "NVIDIA GeForce RTX 4080 SUPER", "uuid": "GPU-fef928e1-b12e-2f20-4b2e-a868e6e0faad", "memory_total_mib": 32760, "memory_free_mib": 15000},
            {"index": 1, "name": "NVIDIA GeForce RTX 4080 SUPER", "uuid": "GPU-845348f1-1511-dc19-0877-cc97658e3edd", "memory_total_mib": 16376, "memory_free_mib": 15900},
        ]
        selected = pilot._select_gpu_row(rows, "corelab-4080s", None)
        self.assertEqual(selected["uuid"], "GPU-fef928e1-b12e-2f20-4b2e-a868e6e0faad")
        self.assertEqual(pilot._select_gpu_row(rows, "corelab-4080s", rows[1]["uuid"])["uuid"], rows[1]["uuid"])

    def test_attempt_lock_is_atomic_and_second_attempt_is_rejected(self):
        import importlib.util
        import tempfile
        spec = importlib.util.spec_from_file_location("pilot", SCRIPT)
        pilot = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pilot)
        with tempfile.TemporaryDirectory() as root:
            control = Path(root) / "control"
            gpu = {"uuid": "GPU-test", "index": 0}
            lock, attempt = pilot._attempt_lock(control, repository_head="abc", output_root=Path(root) / "out", gpu=gpu, noise_sha256="noise")
            self.assertTrue(lock.is_file())
            with self.assertRaises(pilot.PilotError):
                pilot._attempt_lock(control, repository_head="abc", output_root=Path(root) / "out2", gpu=gpu, noise_sha256="noise")
            pilot._finish_attempt(lock, attempt, status="FAILED_OR_PARTIAL", counters={"exceptions": 1})
            value = json.loads(lock.read_text(encoding="utf-8"))
            self.assertEqual(value["status"], "FAILED_OR_PARTIAL")

    def test_raw_analysis_phase_is_separate_from_model_loop(self):
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("_analyze_raw_records(raw_records", text)
        self.assertLess(text.index("raw_records.append"), text.rindex("_analyze_raw_records(raw_records"))
        self.assertIn("CPU-only raw->decoded->metrics phase", text)

    def test_pi0152_reuses_exact_noise_and_separate_attempt_names(self):
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("EXPECTED_NOISE_ARRAY_SHA", text)
        self.assertIn("EXPECTED_NOISE_FILE_SHA", text)
        self.assertIn("stage45f_pi0152_", text)
        self.assertIn("accepted_stage44_input_npz", text)


if __name__ == "__main__":
    unittest.main()
