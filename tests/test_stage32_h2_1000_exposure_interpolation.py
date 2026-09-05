import ast
import importlib.util
from pathlib import Path
import unittest

REPO = Path(__file__).resolve().parents[1] if (Path(__file__).resolve().parents[1] / "scripts").exists() else Path.cwd()
SCRIPT = REPO / "scripts" / "stage32_h2_1000_exposure_interpolation.py"
SPEC = importlib.util.spec_from_file_location("stage32", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Stage32ContractTests(unittest.TestCase):
    def test_frozen_contract(self):
        self.assertEqual(MODULE.PI_GATE, "PI_GATE_STATE_0013")
        self.assertEqual(MODULE.PROJECT, "CFP03-CF-004")
        self.assertEqual(MODULE.SEEDS, (2223, 2224, 2225))

    def test_tree_digest_is_order_independent(self):
        rows = [
            {"path": "b", "bytes": 2, "sha256": "b" * 64},
            {"path": "a", "bytes": 1, "sha256": "a" * 64},
        ]
        self.assertEqual(MODULE.tree_digest(rows, "x"), MODULE.tree_digest(list(reversed(rows)), "x"))

    def test_only_stock_sampler_call_is_present(self):
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        calls = [node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        self.assertEqual(calls.count("get_action"), 1)
        self.assertNotIn("backward", calls)
        self.assertNotIn("save_pretrained", calls)


if __name__ == "__main__":
    unittest.main()
