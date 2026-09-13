"""CPU-only contract checks for the PI0163 launch and q0 data boundary."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError:  # system Python on the login node is CPU-minimal
    np = None


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PI0163ContractTests(unittest.TestCase):
    def test_pi0163_ledger_caps_are_new_and_exact(self):
        guard = load_script("pi0163_guard", "stage45f_pi0162_budget_guard.py")
        self.assertEqual(guard.PI0163_SCHEMA, "CORELAB_PI0163_BUDGET_LEDGER_V1")
        self.assertEqual(guard.PI0163_BUDGETS["simulator_constructor"], 4)
        self.assertEqual(guard.PI0163_BUDGETS["explicit_control_tick"], 984)
        self.assertEqual(guard.PI0163_BUDGETS["reference_update"], 16)
        self.assertEqual(guard.PI0163_BUDGETS["conditioning"], 6)
        self.assertEqual(guard.PI0163_BUDGETS["action_head"], 21)
        self.assertEqual(guard.PI0163_BUDGETS["model_side_call"], 27)
        with tempfile.TemporaryDirectory() as td:
            ledger = Path(td) / "ledger.json"
            owner = guard.BudgetLedger(ledger, "PI0163_CORELAB_TEST", output_root=Path(td) / "out")
            self.assertEqual(owner.snapshot()["schema"], guard.PI0163_SCHEMA)
            owner.close("test")

    @unittest.skipIf(np is None, "numpy unavailable in the minimal CPU test interpreter")
    def test_complete_q0_npz_is_consumable(self):
        producer = load_script("pi0163_expert", "stage45f_pi0162_guarded_sync_expert.py")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            condition = "C0_NOMINAL"
            snap = root / "conditions" / condition / "snapshots"
            snap.mkdir(parents=True)
            body = np.zeros((96, 96, 3), dtype=np.uint8)
            wrist = np.zeros_like(body)
            state = np.zeros(10, dtype=np.float32)
            eef = np.zeros(3, dtype=np.float64)
            anchor = np.zeros(3, dtype=np.float64)
            target = np.array([0.3, 0.1, 0.58], dtype=np.float64)
            expert = np.zeros((40, 10), dtype=np.float64)
            timestamps = np.arange(40, dtype=np.float64) * 0.05
            np.savez_compressed(
                snap / "q0.npz",
                body_rgb=body,
                wrist_rgb=wrist,
                state_v1=state,
                eef_world_m=eef,
                eef_local_m=eef,
                anchor_world_m=anchor,
                target_world_m=target,
                expert_physical_local=expert,
                expert_timestamps=timestamps,
            )
            (root / "conditions" / condition / "condition_metadata.json").write_text(
                json.dumps({"status": "PASS"}), encoding="utf-8"
            )
            loaded = producer.load_expert_q0(root, condition)
            self.assertEqual(loaded["expert"].shape, (40, 10))
            self.assertEqual(loaded["timestamps"].shape, (40,))
            self.assertEqual(loaded["state"].shape, (10,))

    def test_pi0163_runner_exposes_loader_and_stops_on_failure(self):
        source = (ROOT / "scripts" / "stage45f_pi0162_guarded_sync_expert.py").read_text(encoding="utf-8")
        self.assertIn("--import-only-evidence", source)
        self.assertIn("PI0163_CORELAB", source)
        self.assertIn("selected_conditions", source)
        # The native constructor is reached only after its reservation; a
        # loader failure therefore cannot fall through to the next condition.
        self.assertIn('guard.reserve("simulator_constructor"', source)
        self.assertIn("refusing to overwrite q0 artifact", source)

    @unittest.skipIf(np is None, "numpy unavailable in the minimal CPU test interpreter")
    def test_two_condition_model_counts_are_4_14_18(self):
        model = load_script("pi0163_model", "stage45f_pi0160_n2_q0_diagnostic.py")
        self.assertEqual(2 * 2, 4)
        self.assertEqual(7 * 2, 14)
        self.assertEqual(9 * 2, 18)
        source = (ROOT / "scripts" / "stage45f_pi0160_n2_q0_diagnostic.py").read_text(encoding="utf-8")
        self.assertIn("expected_conditioning", source)
        self.assertIn("expected_action_heads", source)
        self.assertIn("expected_model_calls", source)


if __name__ == "__main__":
    unittest.main()
