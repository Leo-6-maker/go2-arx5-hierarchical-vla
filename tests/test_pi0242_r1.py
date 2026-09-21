import unittest

import numpy as np

from scripts.pi0242_train_infer_bridge import (
    _feature_delta,
    _feature_transform_provenance,
    _physical_profile,
    _snapshot_tree,
)


def _rot6d(angle: float, count: int = 40) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    matrix = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    columns = np.concatenate([matrix[:, 0], matrix[:, 1]])
    return np.repeat(columns[None, :], count, axis=0).astype(np.float32)


class PI0242R1Tests(unittest.TestCase):
    def test_snapshot_is_independent_from_later_mutation(self):
        raw = {"backbone_features": np.array([[1.0, 2.0]], dtype=np.float32)}
        snapshot = _snapshot_tree(raw)
        raw["backbone_features"][0, 0] = 99.0
        self.assertEqual(float(snapshot["backbone_features"][0, 0]), 1.0)

    def test_feature_transform_records_single_and_double_passes(self):
        raw = {"backbone_features": np.array([[1.0, 2.0]], dtype=np.float32)}
        once = {"backbone_features": np.array([[3.0, 5.0]], dtype=np.float32)}
        twice = {"backbone_features": np.array([[7.0, 11.0]], dtype=np.float32)}
        result = _feature_transform_provenance(raw, once, twice)
        self.assertFalse(result["feature_transform_deltas"]["raw_to_processed_once"]["exact"])
        self.assertEqual(result["feature_transform_deltas"]["processed_once_to_processed_twice"]["max_abs"], 6.0)
        self.assertEqual(_feature_delta(once["backbone_features"], once["backbone_features"])["max_abs"], 0.0)

    def test_angular_speed_is_adjacent_prediction_speed(self):
        xyz = np.zeros((40, 3), dtype=np.float32)
        predicted = np.concatenate([xyz, _rot6d(0.2), np.zeros((40, 1), dtype=np.float32)], axis=1)
        expert = np.concatenate([xyz, _rot6d(0.0), np.zeros((40, 1), dtype=np.float32)], axis=1)
        result = _physical_profile(predicted[None], expert, np.zeros(9, dtype=np.float32), np.zeros(3, dtype=np.float32))
        self.assertAlmostEqual(result["max_angular_speed_radps"], 0.0, places=6)
        self.assertAlmostEqual(result["expert_max_angular_speed_radps"], 0.0, places=6)
        self.assertAlmostEqual(result["so3_rmse_rad"], 0.2, places=3)

    def test_angular_speed_matches_known_constant_increment(self):
        xyz = np.zeros((40, 3), dtype=np.float32)
        rotations = np.concatenate([_rot6d(i * 0.01, 1) for i in range(40)], axis=0)
        predicted = np.concatenate([xyz, rotations, np.zeros((40, 1), dtype=np.float32)], axis=1)
        result = _physical_profile(predicted[None], predicted, np.zeros(9, dtype=np.float32), np.zeros(3, dtype=np.float32))
        self.assertAlmostEqual(result["max_angular_speed_radps"], 0.2, places=4)


if __name__ == "__main__":
    unittest.main()
