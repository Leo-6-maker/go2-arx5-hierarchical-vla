"""CPU regression for the PI0156 complete-query inference boundary.

The failure being guarded is specifically an inference feature produced by an
inner helper and then consumed by a parameterized operator after autograd has
been re-enabled.  ``eval()`` and cloning do not establish the required
contract; the real query wrapper must keep the entire operation in
``torch.inference_mode()``.
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = ROOT / "scripts" / "stage45f_pi0153_generation_probe.py"


def load_probe():
    spec = importlib.util.spec_from_file_location("stage45f_pi0156_probe_test", PROBE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load generation probe")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PI0156InferenceBoundaryTests(unittest.TestCase):
    def test_real_wrapper_covers_inference_feature_and_parameterized_operator(self):
        import torch

        probe = load_probe()
        feature_producer = torch.nn.Linear(3, 3, bias=False)
        parameterized_operator = torch.nn.Linear(3, 2, bias=False)
        source = torch.tensor([[0.25, -0.5, 1.0]], dtype=torch.float32)

        # This is the old split boundary: an inner helper creates an
        # inference tensor, then the parameterized action operator runs with
        # autograd enabled.  Current Torch correctly rejects that use.
        with torch.inference_mode():
            inference_feature = feature_producer(source)
        with self.assertRaisesRegex(RuntimeError, "Inference tensors"):
            parameterized_operator(inference_feature)

        # The actual probe wrapper keeps both producer and consumer inside one
        # complete inference query.  No clone or eval-only workaround is used.
        output = probe._inference_query(
            torch,
            lambda: parameterized_operator(feature_producer(source)),
        )
        self.assertFalse(output.requires_grad)
        self.assertIsNone(output.grad_fn)
        self.assertTrue(torch.isfinite(output).all().item())
        self.assertFalse(feature_producer.weight.grad is not None)
        self.assertFalse(parameterized_operator.weight.grad is not None)


if __name__ == "__main__":
    unittest.main()
