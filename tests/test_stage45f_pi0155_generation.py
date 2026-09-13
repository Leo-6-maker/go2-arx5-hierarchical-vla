"""CPU behaviour checks for the PI0155 generation-path repairs.

These tests run with the already-qualified CPU Torch environment.  They use a
tiny action-head stub and never construct GR00T, read a checkpoint, or touch
CUDA.
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = ROOT / "scripts" / "stage45f_pi0153_generation_probe.py"


def load_probe():
    spec = importlib.util.spec_from_file_location("stage45f_pi0155_probe_test", PROBE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load generation probe")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PI0155GenerationTests(unittest.TestCase):
    def test_tuple_sampler_is_unpacked_and_observed(self):
        import torch

        probe = load_probe()

        class Action(torch.nn.Module):
            def forward(self, value):
                return value

        class Model:
            def __init__(self):
                self.parameter = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
                self.action_head = type("Head", (), {"model": Action()})()

            def parameters(self):
                return iter((self.parameter,))

            def prepare_input(self, value):
                return ({}, {})

        model = Model()

        def sampler(current_model, _inputs, _noise, torch_module):
            for _ in range(4):
                current_model.action_head.model(torch_module.zeros(1))
            return torch_module.zeros(probe.RAW_SHAPE), 4

        raw, details, calls = probe._c1_call(model, {}, __import__("numpy").zeros(probe.RAW_SHAPE, dtype="float32"), torch, sampler)
        self.assertEqual(tuple(raw.shape), probe.RAW_SHAPE)
        self.assertEqual(calls, 4)
        self.assertEqual(details["observed_sampler_calls"], 4)
        self.assertEqual(details["noise_injection_status"], "EXACTLY_ONE_DIRECT_ARGUMENT")

    def test_bfloat16_summary_is_byte_safe_and_float32_numeric(self):
        import torch

        probe = load_probe()
        value = torch.tensor([1.0, -2.0], dtype=torch.bfloat16)
        summary = probe._array_summary(value)
        self.assertEqual(summary["dtype"], "torch.bfloat16")
        self.assertEqual(summary["numeric_summary_dtype"], "float32")
        self.assertIn("raw_dtype_sha256", summary)
        self.assertIn("numeric_float32_sha256", summary)
        self.assertAlmostEqual(summary["mean"], -0.5, places=5)

    def test_first_difference_is_deterministic(self):
        probe = load_probe()
        self.assertEqual(probe._first_summary_difference({"a": {"b": 1}}, {"a": {"b": 2}}), "a.b")


if __name__ == "__main__":
    unittest.main()
