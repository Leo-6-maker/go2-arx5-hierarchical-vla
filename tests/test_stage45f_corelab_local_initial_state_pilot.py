import ast
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/stage45f_corelab_local_initial_state_pilot.py"
MANIFEST = ROOT / "manifests/STAGE45F_CORELAB_LOCAL_INITIAL_STATE_PILOT_V1.json"


class LocalInitialStatePilotStaticTests(unittest.TestCase):
    def test_manifest_contract(self):
        value = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(value["authority"]["pi_gate"], "PI_GATE_STATE_0150")
        self.assertEqual(value["repository"]["base_head"], "d762e731d2c11ceea3c001c37ac27890fe6f4e70")
        self.assertEqual(value["local_reference"]["seed"], 2223)
        self.assertEqual(value["budget"]["conditioning_passes"], 6)
        self.assertEqual(value["budget"]["action_head_calls"], 24)
        self.assertFalse(value["stage43_input"]["target_xyz_in_model_input"])

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
        for token in ("--mode", "audit", "preflight", "execute", "failure_seal.json", "target_xyz_in_model_input"):
            self.assertIn(token, text)
        self.assertIn("action_head_calls\"] == 24", text)
        self.assertIn("conditioning_passes\"] == 6", text)


if __name__ == "__main__":
    unittest.main()
