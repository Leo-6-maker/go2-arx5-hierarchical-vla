import ast
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "stage45f_corelab_pi0149_single_window_smoke.py"
MANIFEST = ROOT / "manifests" / "STAGE45F_CORELAB_PI0149_SINGLE_WINDOW_SMOKE_V1.json"


class PI0149SingleWindowSmokeTests(unittest.TestCase):
    def test_manifest_is_pathless_and_frozen(self):
        payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(payload["authority"]["issue_11_comment"], 5646608846)
        self.assertEqual(payload["checkpoint"]["tree_sha256"], "2113a6a3b2a91233a42214dbb017c0136acde282fda2a418066f9317e78981a5")
        self.assertEqual(payload["train"]["tree_sha256"], "2f3badd2019a68c6549a66f85b5486faf626078a3e438342f67aacc4edc95789")
        text = MANIFEST.read_text(encoding="utf-8")
        self.assertNotIn("/scratch/", text)
        self.assertNotIn("/home/", text)
        self.assertNotIn("/var/cache/", text)

    def test_static_guard_has_no_model_imports(self):
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        top = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                top.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                top.add(node.module.split(".", 1)[0])
        self.assertFalse(top.intersection({"torch", "transformers", "gr00t", "cuda", "numpy"}))

    def test_budget_and_target_exclusion_are_literal(self):
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("MAX_CONDITIONING = 1", text)
        self.assertIn("MAX_ACTION_HEAD = 4", text)
        self.assertIn('target_xyz_in_model_input") is False', text)
        self.assertIn('public_sampler_calls") == 0', text)


if __name__ == "__main__":
    unittest.main()
