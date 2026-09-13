import ast
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "stage45f_pi0157_conditioning_dependence.py"
MANIFEST = ROOT / "manifests" / "STAGE45F_PI0157_CONDITIONING_DEPENDENCE_V1.json"


class PI0157ConditioningDependenceStaticTests(unittest.TestCase):
    def test_manifest_freezes_authority_inputs_and_budget(self):
        value = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(value["authority"]["pi_gate"], "PI_GATE_STATE_0157")
        self.assertEqual(value["authority"]["issue_comment"], 5651062080)
        self.assertEqual(value["authority"]["pr_pointer_comment"], 5651064217)
        self.assertEqual(value["authority"]["starting_head"], "f4fdb01e5071b530d6997ec2d0a17666cdeacddb")
        self.assertEqual(value["fixed_model"]["checkpoint_tree_sha256"], "09bb76cb2b74a48644c120201042671fb7eaa42da3603ffb84c62f49a35d6152")
        self.assertEqual(value["fixed_model"]["active_scalars"], 190)
        self.assertEqual(value["fixed_noise"]["shape"], [1, 40, 132])
        self.assertEqual(len(value["conditions"]["order"]), 8)
        self.assertEqual(value["budget"]["nominal_model_side_calls"], 40)
        self.assertEqual(value["budget"]["hard_model_side_cap"], 50)
        self.assertFalse(value["fixed_model"]["target_xyz_in_model_input"])
        self.assertFalse(value["execution_boundary"]["validation_e_read"])
        self.assertFalse(value["execution_boundary"]["holdout_f_read"])

    def test_no_forbidden_runtime_imports_at_module_top_level(self):
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT))
        forbidden = {"torch", "transformers", "gr00t"}
        for node in tree.body:
            if isinstance(node, ast.Import):
                names = {alias.name.split(".")[0] for alias in node.names}
                self.assertTrue(forbidden.isdisjoint(names), names)
            elif isinstance(node, ast.ImportFrom):
                self.assertNotIn((node.module or "").split(".")[0], forbidden)

    def test_source_binds_raw_first_state_audit_and_single_recommendation(self):
        text = SCRIPT.read_text(encoding="utf-8")
        for token in ("_state_audit", "_raw_blocks", "_decode", "raw_first", "standardized_T", "N1_NATIVE_a", "X1_VT_SC", "Z1_CONST128_ST", "torch.inference_mode", "failure_seal.json", "CORELAB_PI0157_CONDITIONING_ADMISSION_RECORDED"):
            self.assertIn(token, text)
        self.assertLess(text.index("np.save(raw_path"), text.index("# CPU-only decoding starts"))
        self.assertEqual(text.count("Single recommendation"), 1)

    def test_source_does_not_route_target_or_labels_into_model_inputs(self):
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("target_xyz_in_model_input", text)
        self.assertIn("future_desired_eef_9d", text)
        self.assertIn("target/metadata/label leaked", text)
        self.assertIn('"target_xyz_in_model_input": False', text)


if __name__ == "__main__":
    unittest.main()
