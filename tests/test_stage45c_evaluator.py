"""CPU-only Stage45C evaluator contract tests."""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from go2arx5_vla.evaluation.checkpoint_eval import (
    EXPECTED_WINDOWS,
    EvaluationManifestError,
    aggregate_evaluation_rows,
    load_evaluation_manifest,
    runtime_is_bound,
)
from go2arx5_vla.evaluation.structured_action_eval import (
    validate_model_input_contract,
    validate_observation_arrays,
)
from go2arx5_vla.evaluation.trajectory_metrics import (
    EvaluationContractError,
    comparison_metrics,
    trajectory_metrics,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests" / "STAGE45C_TRAIN24_EVALUATION_PROTOCOL_V1.json"


class TestStage45CEvaluator(unittest.TestCase):
    def test_stage44r_progress_sign_and_cosine_semantics(self) -> None:
        target = np.asarray([1.0, 0.0, 0.0])
        positive = trajectory_metrics([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], target)
        zero = trajectory_metrics([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], target)
        negative = trajectory_metrics([[0.0, 0.0, 0.0], [-0.5, 0.0, 0.0]], target)
        self.assertEqual(positive["progress_sign"], "POSITIVE")
        self.assertAlmostEqual(positive["absolute_progress_m"], 0.5)
        self.assertAlmostEqual(positive["endpoint_target_vector_cosine"], 1.0)
        self.assertEqual(zero["progress_sign"], "ZERO")
        self.assertIsNone(zero["endpoint_target_vector_cosine"])
        self.assertEqual(negative["progress_sign"], "NEGATIVE")
        self.assertAlmostEqual(negative["endpoint_target_vector_cosine"], -1.0)
        rough = trajectory_metrics(
            [[0.0, 0.0, 0.0], [0.5, 0.1, 0.0], [1.0, 0.0, 0.0]], target
        )
        self.assertGreater(rough["roughness_position_second_difference_m"]["mean"], 0.0)

    def test_comparison_preserves_h40_d10_metrics(self) -> None:
        expert = np.zeros((40, 10), dtype=np.float64)
        decoded = np.zeros((40, 10), dtype=np.float64)
        expert[:, 0] = np.linspace(0.0, 0.50, 40)
        decoded[:, 0] = np.linspace(0.0, 0.40, 40)
        decoded[:, 9] = 0.25
        result = comparison_metrics(decoded, expert, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        self.assertEqual(result["target"]["progress_sign"], "POSITIVE")
        self.assertEqual(result["target"]["sample_count"], 40)
        self.assertAlmostEqual(result["endpoint_error_m"], 0.10)
        self.assertAlmostEqual(result["gripper_rmse"], 0.25)
        self.assertAlmostEqual(result["prediction_path_length_m"], 0.40)
        self.assertAlmostEqual(result["expert_path_length_m"], 0.50)
        self.assertAlmostEqual(result["prediction_roughness_position_second_difference_m"]["mean"], 0.0)

    def test_shape_and_finiteness_fail_closed(self) -> None:
        with self.assertRaises(EvaluationContractError):
            trajectory_metrics(np.zeros((40, 2)), [1.0, 0.0, 0.0])
        with self.assertRaises(EvaluationContractError):
            trajectory_metrics([[0.0, 0.0, np.nan]], [1.0, 0.0, 0.0])
        with self.assertRaises(EvaluationContractError):
            comparison_metrics(np.zeros((39, 10)), np.zeros((40, 10)), [0.0] * 3, [1.0, 0.0, 0.0])

    def test_manifest_is_prepared_and_runtime_is_not_guessed(self) -> None:
        manifest = load_evaluation_manifest(MANIFEST)
        self.assertEqual(manifest["status"], "PREPARED_NOT_EXECUTED")
        self.assertEqual(manifest["cohort"]["window_ids"], list(EXPECTED_WINDOWS))
        self.assertFalse(runtime_is_bound(manifest))
        broken = dict(manifest)
        broken["source_golden_head"] = "wrong"
        with self.assertRaises(EvaluationManifestError):
            from go2arx5_vla.evaluation.checkpoint_eval import validate_evaluation_manifest

            validate_evaluation_manifest(broken)

    def test_observation_and_model_input_boundaries(self) -> None:
        body = np.zeros((96, 96, 3), dtype=np.uint8)
        wrist = np.zeros((96, 96, 3), dtype=np.uint8)
        state = np.zeros(10, dtype=np.float32)
        validate_model_input_contract(("body_rgb", "wrist_rgb", "state"), target_xyz_in_model_input=False)
        validate_observation_arrays(body_rgb=body, wrist_rgb=wrist, state_v1=state)
        with self.assertRaises(EvaluationContractError):
            validate_model_input_contract(("body_rgb", "wrist_rgb", "state"), target_xyz_in_model_input=True)
        with self.assertRaises(EvaluationContractError):
            validate_observation_arrays(body_rgb=body.astype(np.float32), wrist_rgb=wrist, state_v1=state)

    def test_aggregate_preserves_order_and_raw_hashes(self) -> None:
        rows = []
        expert = np.zeros((40, 10), dtype=np.float64)
        decoded = np.zeros((40, 10), dtype=np.float64)
        expert[:, 0] = np.linspace(0.0, 0.50, 40)
        decoded[:, 0] = np.linspace(0.0, 0.40, 40)
        metrics = comparison_metrics(decoded, expert, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        for index, window_id in enumerate(EXPECTED_WINDOWS):
            rows.append(
                {
                    "window_id": window_id,
                    "raw_prediction_artifact": {"array_sha256": f"raw-{index}"},
                    "decoded_action": {"array_sha256": f"decoded-{index}"},
                    "checkpoint": {"label": "checkpoint-2000", "sha256": "a" * 64},
                    "model": {"label": "golden-full24", "sha256": "b" * 64},
                    "metrics": metrics,
                    "safety_gate_decision": "ACCEPT",
                }
            )
        aggregate = aggregate_evaluation_rows(rows)
        self.assertEqual(aggregate["window_count"], 24)
        self.assertEqual(aggregate["target_metrics"]["count"], 24)
        self.assertEqual(aggregate["safety_gate_decision_counts"]["ACCEPT"], 24)
        self.assertEqual(aggregate["safety_gate_decision_counts"]["null"], 0)
        self.assertTrue(aggregate["raw_predictions_preserved_separately"])
        rows[1], rows[2] = rows[2], rows[1]
        with self.assertRaises(EvaluationContractError):
            aggregate_evaluation_rows(rows)


if __name__ == "__main__":
    unittest.main()
