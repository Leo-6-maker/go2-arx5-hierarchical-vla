import ast
import importlib.util
from pathlib import Path
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1] if (Path(__file__).resolve().parents[1] / "scripts").exists() else Path.cwd()
SCRIPT = REPO / "scripts" / "stage32_h2_1000_exposure_interpolation.py"
SPEC = importlib.util.spec_from_file_location("stage32", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Stage32ContractTests(unittest.TestCase):
    def test_frozen_contract(self):
        self.assertEqual(MODULE.PI_GATE, "PI_GATE_STATE_0014")
        self.assertEqual(MODULE.PROJECT, "CFP03-CF-004")
        self.assertEqual(MODULE.SEEDS, (2223, 2224, 2225))
        self.assertEqual(MODULE.CONTINUATION_SAMPLER_SEEDS, (2224, 2225))

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

    def test_teacher_record_persistence_is_atomic_and_keyed(self):
        with tempfile.TemporaryDirectory(prefix="stage32_test_") as temporary:
            root = Path(temporary)
            row = {"checkpoint": "H2@1000", "probe_id": "seed2223_tq10", "finite_flags": {"all": True}}
            self.assertEqual(MODULE.persist_teacher_record(root, row), 1)
            self.assertEqual(MODULE.load_teacher_records(root), [row])
            with self.assertRaises(MODULE.Stage32Error):
                MODULE.persist_teacher_record(root, row)

    def test_sampler_metric_packaging_uses_exact_profile_provider(self):
        class ProfileProvider:
            def __init__(self):
                self.calls = []

            def _profile(self, rows, period):
                self.calls.append((rows, period))
                return {"profile": "phase-b", "rows": len(rows)}

        class FakeStage29:
            def _trajectory_metrics(self, decoded, expert, provider, stage26, processor, embodiment, snapshot, checkpoint, label):
                self.assert_profile = provider._profile
                return {"predicted": provider._profile(decoded, 0.05), "expert": provider._profile(expert, 0.05)}

        provider = ProfileProvider()
        packaged = MODULE.package_sampler_metrics(
            FakeStage29(), object(), [[1]], [[2]], provider,
            object(), object(), object(), Path("/tmp/checkpoint"), "H2@1000",
        )
        self.assertEqual(len(provider.calls), 2)
        self.assertEqual(packaged["predicted"]["profile"], "phase-b")

    def test_continuation_seed_guard_rejects_historical_sampler(self):
        with self.assertRaises(MODULE.Stage32Error):
            MODULE.assert_continuation_sampler_seed(2223)
        for seed in MODULE.CONTINUATION_SAMPLER_SEEDS:
            MODULE.assert_continuation_sampler_seed(seed)

    def test_execution_counters_never_regress(self):
        with tempfile.TemporaryDirectory(prefix="stage32_test_") as temporary:
            path = Path(temporary) / "state.json"
            MODULE.write_json(path, {"teacher_forced_probes": 0, "sampler_calls": 0})
            MODULE.update_execution_state(path, {"teacher_forced_probes": 9, "sampler_calls": 1})
            MODULE.update_execution_state(path, {"teacher_forced_probes": 3, "sampler_calls": 0})
            state = MODULE.read_json(path)
            self.assertEqual(state["teacher_forced_probes"], 9)
            self.assertEqual(state["sampler_calls"], 1)

    def test_runner_is_continuation_only(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("PI_GATE_STATE_0014", source)
        self.assertIn("601973_SALVAGED", source)
        self.assertIn("CONTINUATION", source)
        self.assertNotIn("for seed in SEEDS:", source)
        self.assertIn("state_error_mse", source)

    def test_packaging_self_test(self):
        result = MODULE.run_packaging_self_test()
        self.assertEqual(result["status"], "PASS_PACKAGING_SELF_TEST")
        self.assertTrue(result["seed2223_rejected"])


if __name__ == "__main__":
    unittest.main()
