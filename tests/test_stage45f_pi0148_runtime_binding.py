import ast
import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/stage45f_corelab_pi0148_runtime_binding.py"


class FakeResult:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class RuntimeBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("pi0148_runtime", SCRIPT)
        cls.mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(cls.mod)

    def test_cpu_control_has_no_top_level_model_import(self):
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".", 1)[0])
        self.assertTrue(imported.isdisjoint({"torch", "transformers", "gr00t", "isaacgym", "isaac"}))

    def test_process_and_gpu_parsers_are_explicit(self):
        rows = self.mod._parse_gpu_rows("0, NVIDIA GeForce RTX 4080 SUPER, GPU-a, 32760, 10, 32750, 580.1\n")
        self.assertEqual(rows[0]["uuid"], "GPU-a")
        self.assertEqual(self.mod._parse_process_rows("GPU-a, 42, python, 100\n")[0]["pid"], 42)

    def test_busy_uuid_is_blocked_without_attempt(self):
        devices = {"gpus": [{"uuid": "GPU-a"}], "processes": [{"uuid": "GPU-a", "pid": 42}]}
        result = self.mod.run_toy_smoke("/does/not/run", "GPU-a", "4080S", devices=devices)
        self.assertEqual(result["status"], "BLOCKED_DEVICE_BUSY")
        self.assertEqual(result["attempted_calls"], 0)

    def test_runtime_probe_missing_candidate_is_explicit(self):
        result = self.mod.probe_interpreter({"path": "/definitely/not/a/python", "source": "test", "role": "test"})
        self.assertEqual(result["status"], "ABSENT_AT_CHECKED_PATH")

    def test_budget_and_request_are_frozen(self):
        self.assertEqual(self.mod.MAX_CALLS_PER_DEVICE, 4)
        self.assertEqual(self.mod.MAX_CALLS_TOTAL, 8)
        report = self.mod.build_report(role="4080S", uuid=None, interpreter=None, execute_smoke=False)
        self.assertEqual(report["toy_cuda_smoke"]["status"], "NOT_ATTEMPTED")
        self.assertEqual(report["next_model_request"]["model_forward"], 0)
        self.assertFalse(report["next_model_request"]["sampler"]["target_xyz_in_model_input"])

    def test_schema_is_json_serializable(self):
        self.assertEqual(json.loads(json.dumps({"schema": self.mod.SCHEMA}))["schema"], self.mod.SCHEMA)


if __name__ == "__main__":
    unittest.main()
