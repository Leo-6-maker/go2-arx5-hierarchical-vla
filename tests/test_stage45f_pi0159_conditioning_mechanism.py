import ast
import json
from pathlib import Path
import tempfile
import unittest

try:
    import numpy as np
except ImportError:  # The login-node system Python is intentionally minimal.
    np = None


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "stage45f_pi0159_conditioning_mechanism.py"
MANIFEST = ROOT / "manifests" / "STAGE45F_PI0159_CONDITIONING_MECHANISM_V1.json"


class PI0159StaticTests(unittest.TestCase):
    def test_manifest_freezes_gate_flow_and_boundaries(self):
        value = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(value["authority"]["pi_gate"], "PI_GATE_STATE_0159")
        self.assertEqual(value["authority"]["issue_comment"], 5651409238)
        self.assertEqual(value["authority"]["pr_comment"], 5651409688)
        self.assertEqual(value["fixed_model"]["checkpoint_tree_sha256"], "09bb76cb2b74a48644c120201042671fb7eaa42da3603ffb84c62f49a35d6152")
        self.assertEqual(value["fixed_model"]["active_scalars"], 190)
        self.assertEqual(value["flow"]["t_buckets"], [67, 369, 783])
        self.assertEqual(value["budget"]["model_side_calls"], 16)
        self.assertEqual(value["conditions"]["order"], ["TRAIN_REFERENCE", "STATE_C0_VISION_TRAIN", "VISION_C0_STATE_TRAIN", "C0_FULL"])
        self.assertFalse(value["fixed_model"]["target_xyz_in_model_input"])
        self.assertFalse(value["execution_boundary"]["training"])
        self.assertFalse(value["execution_boundary"]["wbc"])
        self.assertFalse(value["execution_boundary"]["validation_e_read"])
        self.assertFalse(value["execution_boundary"]["holdout_f_read"])

    def test_no_runtime_model_import_at_module_top_level(self):
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT))
        forbidden = {"torch", "transformers", "gr00t", "isaacgym"}
        for node in tree.body:
            if isinstance(node, ast.Import):
                self.assertTrue(forbidden.isdisjoint({alias.name.split(".")[0] for alias in node.names}))
            elif isinstance(node, ast.ImportFrom):
                self.assertNotIn((node.module or "").split(".")[0], forbidden)

    @unittest.skipIf(np is None, "numpy is available in the qualified runtime, not system Python")
    def test_flow_oracle_and_raw_bundle_are_deterministic(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location("pi0159", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        rng = np.random.default_rng(159)
        x0 = rng.normal(size=(1, 40, 132)).astype(np.float32)
        x1 = rng.normal(size=(1, 40, 132)).astype(np.float32)
        t = module.T_VALUES[1]
        x_t = ((1.0 - t) * x0 + t * x1).astype(np.float32)
        v_target = (x1 - x0).astype(np.float32)
        x1_hat = (x_t + (1.0 - t) * v_target).astype(np.float32)
        self.assertTrue(np.array_equal(x1_hat, x1))
        mask = np.ones((40, 132), dtype=bool)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.npz"
            digest = module._save_raw(path, x0=x0, x_t=x_t, v_target=v_target, v_hat=v_target, x1_hat=x1_hat, active_mask=mask, state=np.zeros(10, dtype=np.float32), t_value=t, t_bucket=369)
            self.assertEqual(digest, module.sha256_file(path))
            with np.load(path, allow_pickle=False) as data:
                self.assertEqual(tuple(data["x_t_float32"].shape), (1, 40, 132))
                self.assertTrue(np.array_equal(data["x1_hat_float32"], x1))
                self.assertEqual(tuple(data["active_mask_float64"].shape), (40, 132))

    def test_source_has_exact_flow_and_forbidden_guards(self):
        source = SCRIPT.read_text(encoding="utf-8")
        for token in ("x_t=(1-t)x0+t*x1", "v_target=x1-x0", "x1_hat=x_t+(1-t)*v_hat", "raw_first", "target_xyz_in_model_input", "torch.inference_mode", "free_running_inference", "STATE_C0_VISION_TRAIN", "VISION_C0_STATE_TRAIN"):
            self.assertIn(token, source)
        self.assertIn("public_sampler", source)
        self.assertIn("No public sampler", source)


if __name__ == "__main__":
    unittest.main()
