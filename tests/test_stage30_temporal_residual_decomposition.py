from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/stage30_temporal_residual_decomposition.py"
SPEC = importlib.util.spec_from_file_location("stage30_temporal", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TemporalResidualTests(unittest.TestCase):
    def test_constant_position_bias_has_zero_derivative_error(self) -> None:
        residual = np.ones((MODULE.HORIZON, 3), dtype=np.float64)
        self.assertEqual(MODULE.error_metrics(residual)["mse"], 1.0)
        self.assertEqual(MODULE.error_metrics(np.diff(residual, axis=0))["mse"], 0.0)
        self.assertEqual(MODULE.error_metrics(np.diff(residual, n=2, axis=0))["mse"], 0.0)

    def test_frequency_bands_separate_dc_and_alternation(self) -> None:
        constant = np.ones((40, 3), dtype=np.float64)
        alternating = np.tile(np.array([1.0, -1.0]), 20)[:, None] * np.ones((1, 3))
        dc = MODULE.spectral_decomposition(constant)["aggregate_xyz"]
        high = MODULE.spectral_decomposition(alternating)["aggregate_xyz"]
        self.assertAlmostEqual(dc["dc_fraction"], 1.0)
        self.assertAlmostEqual(dc["high_fraction"], 0.0)
        self.assertAlmostEqual(high["high_fraction"], 1.0)

    def test_common_seed_decomposition_identity(self) -> None:
        expert = np.zeros((40, 3), dtype=np.float64)
        predictions = np.stack([
            np.full((40, 3), -1.0),
            np.zeros((40, 3)),
            np.full((40, 3), 1.0),
        ])
        result = MODULE.common_seed_decomposition(predictions, expert)
        self.assertAlmostEqual(result["common_mode_residual_mse"], 0.0)
        self.assertAlmostEqual(result["seed_dependent_variance"], 2.0 / 3.0)
        self.assertLess(result["decomposition_absolute_error"], 1e-15)

    def test_clipping_audit_uses_raw_active_values(self) -> None:
        raw = np.zeros((1, 40, 10), dtype=np.float64)
        raw[:, :, 0] = 2.0
        result = MODULE.clipping_stats(raw)
        self.assertAlmostEqual(result["raw_fraction_abs_gt_1"], 0.1)
        self.assertAlmostEqual(result["xyz"]["raw_fraction_abs_gt_1"], 1.0 / 3.0)
        self.assertAlmostEqual(result["after_official_clip_fraction_at_bound"], 0.1)

    def test_alternating_residual_direction_is_detected(self) -> None:
        expert = np.zeros((40, 3), dtype=np.float64)
        predicted = np.tile(np.array([1.0, -1.0]), 20)[:, None] * np.ones((1, 3))
        result = MODULE.direction_structure(predicted, expert)
        self.assertAlmostEqual(result["residual_sign_flip_rate_all_axes"], 1.0)
        self.assertAlmostEqual(result["residual_lag1_autocorrelation_axis_mean"], -1.0)
        self.assertAlmostEqual(result["adjacent_residual_vector"]["negative_fraction"], 1.0)

    def test_analysis_has_no_model_or_sampler_execution_path(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        for forbidden in (
            "import torch", "import transformers", "AutoModel", ".from_pretrained(",
            ".get_action(", "torch.load", "qsub", "backward(",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn("posthoc_model_calls", source)
        self.assertIn("new_sampler_calls", source)

    def test_stage29_metric_reproduction_tolerance_is_strict(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('"tolerance": 1e-12', source)
        self.assertIn('"status": "MATCH"', source)


if __name__ == "__main__":
    unittest.main()
