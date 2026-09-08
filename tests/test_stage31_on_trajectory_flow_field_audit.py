from __future__ import annotations

import ast
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/stage31_on_trajectory_flow_field_audit.py"
SPEC = importlib.util.spec_from_file_location("stage31_trace", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class _Handle:
    def __init__(self) -> None:
        self.removed = False

    def remove(self) -> None:
        self.removed = True


class _HookModule:
    def __init__(self) -> None:
        self.pre_hook = None
        self.forward_hook = None
        self.handles = []

    def register_forward_pre_hook(self, hook):
        self.pre_hook = hook
        handle = _Handle()
        self.handles.append(handle)
        return handle

    def register_forward_hook(self, hook):
        self.forward_hook = hook
        handle = _Handle()
        self.handles.append(handle)
        return handle


class _ActionHead:
    def __init__(self) -> None:
        self.action_encoder = _HookModule()
        self.action_decoder = _HookModule()


class Stage31ContractTests(unittest.TestCase):
    def test_protocol_is_exact_six_call_stock_sampler_audit(self) -> None:
        plan = MODULE.protocol_plan()
        MODULE.assert_protocol_static(plan)
        self.assertEqual(plan["experiment"]["seeds"], [2223, 2224, 2225])
        self.assertEqual(plan["experiment"]["checkpoints"], ["BASE", "H2@2000"])
        self.assertEqual(plan["experiment"]["maximum_sampler_calls"], 6)
        self.assertEqual(plan["sampler"]["steps"], 4)
        self.assertEqual(plan["sampler"]["timestep_ids"], [0, 250, 500, 750])

    def test_forbidden_boundaries_are_false(self) -> None:
        execution = MODULE.protocol_plan()["execution"]
        for key in (
            "training", "backward", "optimizer_step", "scheduler_step", "resume",
            "checkpoint_write", "sampler_step_change", "alternative_sampler", "wbc",
            "hardware", "validation_e", "holdout_f",
        ):
            self.assertIs(execution[key], False)

    def test_hooks_only_observe_and_are_removed(self) -> None:
        head = _ActionHead()
        x = object()
        timestep = object()
        velocity = object()
        with MODULE.StockSamplerTrace(head) as observer:
            self.assertIsNone(head.action_encoder.pre_hook(None, (x, timestep, object())))
            self.assertIsNone(head.action_decoder.forward_hook(None, (object(),), velocity))
            self.assertEqual(observer.states, [x])
            self.assertEqual(observer.timesteps, [timestep])
            self.assertEqual(observer.decoder_outputs, [velocity])
        self.assertTrue(all(handle.removed for module in (head.action_encoder, head.action_decoder) for handle in module.handles))

    def test_real_tensor_trace_reproduces_four_euler_updates(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is only present in the pinned execution environment")
        head = _ActionHead()
        observer = MODULE.StockSamplerTrace(head)
        current = torch.zeros((1, 40, 132), dtype=torch.bfloat16)
        observer.states = []
        observer.timesteps = []
        observer.decoder_outputs = []
        for timestep_id in (0, 250, 500, 750):
            velocity = torch.full((1, 40, 132), 0.25, dtype=torch.bfloat16)
            observer.states.append(current)
            observer.timesteps.append(torch.tensor([timestep_id]))
            observer.decoder_outputs.append(velocity)
            current = current + 0.25 * velocity
        result = MODULE._finalize_trace(observer, current, torch)
        self.assertEqual(result["x_trajectory_float32"].shape, (5, 1, 40, 132))
        self.assertEqual(result["v_trajectory_float32"].shape, (4, 1, 40, 132))
        self.assertTrue(all(row["exact_tensor_equality"] for row in result["euler_recurrence"]))

    def test_any_stage29_reproduction_mismatch_stops(self) -> None:
        reproduction = {
            "status": "MATCH",
            "noise_raw_exact": True,
            "noise_float32_exact": True,
            "trace_x0_equals_captured_noise_raw": True,
            "final_raw_universal_exact": True,
            "conditioning_exact": True,
        }
        MODULE.assert_stage29_reproduction(reproduction, 1)
        reproduction["final_raw_universal_exact"] = False
        reproduction["status"] = "MISMATCH"
        with self.assertRaisesRegex(MODULE.Stage31Error, "scientific call 1"):
            MODULE.assert_stage29_reproduction(reproduction, 1)

    def test_analytic_reference_perfect_velocity_contracts_exactly(self) -> None:
        target = np.zeros((40, 132), dtype=np.float32)
        current = np.ones((40, 132), dtype=np.float32)
        reference_velocity = -current
        following = current + 0.25 * reference_velocity
        metrics = MODULE.step_metrics(current, following, reference_velocity, target, 0.0)
        for scope in metrics["scopes"].values():
            self.assertEqual(scope["velocity_reference_error_mse"], 0.0)
            self.assertEqual(scope["reference_next_deviation_mse"], 0.0)
            self.assertAlmostEqual(scope["actual_error_contraction"], 0.5625)
            self.assertAlmostEqual(scope["velocity_alignment"]["cosine"], 1.0)

    def test_late_time_reference_reaches_target_in_one_euler_step(self) -> None:
        target = np.full((40, 132), 0.5, dtype=np.float32)
        current = np.full((40, 132), -0.5, dtype=np.float32)
        reference_velocity = (target - current) / 0.25
        metrics = MODULE.step_metrics(current, target, reference_velocity, target, 0.75)
        for scope in metrics["scopes"].values():
            self.assertEqual(scope["reference_next_deviation_mse"], 0.0)
            self.assertEqual(scope["actual_error_contraction"], 0.0)

    def test_cross_seed_decomposition_identity(self) -> None:
        rows = np.stack([
            np.ones((40, 3), dtype=np.float64),
            np.full((40, 3), 2.0, dtype=np.float64),
            np.full((40, 3), 3.0, dtype=np.float64),
        ])
        result = MODULE._cross_seed_decomposition(rows)
        self.assertLess(result["identity_abs_delta"], 1e-12)
        self.assertAlmostEqual(result["common_mode_mse"], 4.0)
        self.assertAlmostEqual(result["seed_variance_mse"], 2.0 / 3.0)

    def test_retention_projection_known_case(self) -> None:
        initial = np.arange(1, 7, dtype=np.float32)
        final = 2.0 * initial
        result = MODULE._retention_metrics(initial, final)
        self.assertAlmostEqual(result["cosine"], 1.0)
        self.assertAlmostEqual(result["centered_pearson"], 1.0)
        self.assertAlmostEqual(result["projection_coefficient_final_on_initial"], 2.0)
        self.assertAlmostEqual(result["projected_output_energy_fraction"], 1.0)

    def test_output_must_not_overlap_historical_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            immutable = root / "stage29"
            immutable.mkdir()
            with self.assertRaisesRegex(MODULE.Stage31Error, "overlaps immutable"):
                MODULE._ensure_external_output(immutable / "new", (immutable,))

    def test_runner_has_only_public_sampler_call_and_no_training_calls(self) -> None:
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        called = [
            node.func.attr for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        ]
        self.assertEqual(called.count("get_action"), 1)
        for forbidden in ("backward", "save_pretrained", "train"):
            self.assertNotIn(forbidden, called)

    def test_no_eager_torch_transformers_or_numpy_import(self) -> None:
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        imports = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
        self.assertNotIn("torch", imports)
        self.assertNotIn("transformers", imports)
        self.assertNotIn("numpy", imports)

    def test_source_audit_and_h200_wrapper_are_bound(self) -> None:
        audit = (REPO / "reports/STAGE31_ON_TRAJECTORY_TRACE_AUDIT.md").read_text(encoding="utf-8")
        wrapper = (REPO / "hpc/pbs/stage31_on_trajectory_flow_field_audit.pbs").read_text(encoding="utf-8")
        self.assertIn("51d4c89f72fda44cbf77285c6a8114b52676b8a1", audit)
        self.assertIn("Both hooks return `None`", audit)
        self.assertIn("#PBS -P CFP03-CF-004", wrapper)
        self.assertIn("#PBS -l select=1:ngpus=1", wrapper)
        self.assertIn("PI_GATE_STATE_0011", wrapper)
        self.assertNotIn("qsub", wrapper)


if __name__ == "__main__":
    unittest.main()
