import ast
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "stage45f_corelab_pi0146_local_device_runtime_smoke.py"


class FakeResult:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def fake_runner(argv, **_kwargs):
    if argv[0] != "nvidia-smi":
        raise AssertionError(argv)
    if "--query-gpu=index,name,uuid,memory.total,memory.used,memory.free,driver_version" in argv:
        return FakeResult(0, "0, NVIDIA GeForce RTX 4080 SUPER, GPU-abc, 32768 MiB, 0 MiB, 32768 MiB, 555.42\n")
    return FakeResult(0, "")


class LocalDeviceRuntimeSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import importlib.util

        spec = importlib.util.spec_from_file_location("local_smoke", SCRIPT)
        cls.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.mod)

    def test_device_census_is_read_only_and_parses_uuid(self):
        result = self.mod.device_census(fake_runner, hostname="synthetic-corelab")
        self.assertEqual(result["status"], "READY_READ_ONLY")
        self.assertEqual(result["gpu_rows"][0]["uuid"], "GPU-abc")
        self.assertEqual(result["cuda_compute_calls"], 0)
        self.assertEqual(result["model_forward_calls"], 0)

    def test_missing_nvidia_smi_is_blocked_not_fabricated(self):
        def missing(argv, **_kwargs):
            return FakeResult(None, "", "nvidia-smi missing")

        result = self.mod.device_census(missing, hostname="synthetic-corelab")
        self.assertEqual(result["status"], "BLOCKED_DEVICE_UNAVAILABLE")
        self.assertEqual(result["gpu_rows"], [])

    def test_plan_binds_exact_checkpoint_and_zero_boundary(self):
        artifacts = self.mod.artifact_inventory(ROOT)
        plan = self.mod.fixed_checkpoint_smoke_plan(artifacts)
        self.assertEqual(plan["checkpoint"]["tree_sha256"], self.mod.CHECKPOINT_TREE)
        self.assertEqual(plan["planned_call_ceiling"]["total_model_side_calls"], 5)
        self.assertFalse(plan["cohort"]["target_xyz_in_model_input"])
        self.assertEqual(plan["execution_boundary"]["model_forward"], 0)
        self.assertEqual(plan["execution_boundary"]["cuda_compute"], 0)

    def test_authorized_endpoint_observations_are_explicit_and_cuda_smoke_is_bounded(self):
        observations = self.mod.AUTHORIZED_ENDPOINT_OBSERVATIONS
        self.assertEqual(set(observations), {"gpu4080s", "gpu5080-1", "gpu5080-2"})
        self.assertEqual(observations["gpu4080s"]["devices"][0]["occupancy"], "BUSY")
        self.assertEqual(observations["gpu5080-1"]["devices"][0]["occupancy"], "FREE_NO_COMPUTE_PROCESS_REPORTED")
        admission = self.mod.non_model_cuda_smoke_admission(observations)
        self.assertEqual(admission["status"], "BLOCKED_AUTHORIZED_GPU_LAUNCHER_REQUIRED")
        self.assertEqual(admission["actual_calls"], 0)
        self.assertEqual(admission["scope"]["max_calls_total"], 8)

    def test_script_has_no_forbidden_top_level_runtime_import(self):
        source = SCRIPT.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = {
            alias.name.split(".", 1)[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imported.update(
            node.module.split(".", 1)[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        )
        self.assertTrue(imported.isdisjoint({"torch", "transformers", "gr00t", "isaacgym", "isaac"}))

    def test_manifest_is_valid_json_and_zero_science(self):
        manifest = json.loads((ROOT / "manifests/STAGE45F_CORELAB_PI0146_LOCAL_DEVICE_RUNTIME_SMOKE_V1.json").read_text())
        self.assertEqual(manifest["status"], "Q0_READ_ONLY_ENDPOINT_CENSUS_AND_B2_PLAN_ONLY")
        self.assertFalse(manifest["future_smoke"]["engineering_reference_only"] is False)
        self.assertEqual(manifest["execution_boundary"]["model_forward"], 0)
        self.assertEqual(manifest["execution_boundary"]["cuda_compute"], 0)
        self.assertEqual(manifest["non_model_cuda_smoke"]["max_calls_total"], 8)


if __name__ == "__main__":
    unittest.main()
