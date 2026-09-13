import ast
import importlib.util
import json
from pathlib import Path
import unittest


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "stage45f_pi0164_controlled_supervision.py"
MANIFEST = REPO / "manifests" / "STAGE45F_PI0164_CONTROLLED_SUPERVISION_PILOT_V1.json"


def load_module():
    spec = importlib.util.spec_from_file_location("pi0164_runner", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PI0164StaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def test_manifest_contract(self):
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(manifest["authority"]["pi_gate"], "PI_GATE_STATE_0164")
        self.assertEqual(manifest["arms"]["E"]["updates"], 256)
        self.assertEqual(manifest["arms"]["S"]["q_slots"], 128)
        self.assertEqual(manifest["recipe"]["active_scalars"], 190)
        self.assertEqual(manifest["evaluation"]["train_world_metrics"], "UNVERIFIED")

    def test_sequence_is_deterministic_and_paired(self):
        first = self.module._make_slot_sequence()
        second = self.module._make_slot_sequence()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 256)
        self.assertTrue(all(0 <= value < 24 for value in first))
        self.assertEqual(len(set(first[:24])), 24)
        self.assertEqual(sum(1 for index in range(256) if index % 2 == 0), 128)

    def test_no_model_import_at_module_scope(self):
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        module_imports = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                module_imports.extend(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                module_imports.append(node.module.split(".")[0])
        self.assertNotIn("torch", module_imports)
        self.assertNotIn("gr00t", module_imports)
        self.assertNotIn("transformers", module_imports)

    def test_static_self_check(self):
        value = self.module.static_self_check()
        self.assertEqual(value["status"], "PASS")
        self.assertEqual(value["model_forward"], 0)
        self.assertFalse(value["gpu"])


if __name__ == "__main__":
    unittest.main()
