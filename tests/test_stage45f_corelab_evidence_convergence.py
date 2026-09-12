from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import stage45f_corelab_evidence_convergence as convergence


class CorelabEvidenceConvergenceTest(unittest.TestCase):
    def test_static_and_future_plans_are_locked(self) -> None:
        static = convergence.static_audit()
        self.assertEqual(static["status"], "PASS")
        self.assertEqual(static["model_forward"], 0)
        self.assertEqual(static["gpu"], 0)
        self.assertFalse(convergence.recipe_freeze_plan()["execute_supported"])
        self.assertFalse(convergence.validation_plan()["read_authorized"])

    def _row(self, index: int, label: str, checkpoint_sha: str) -> dict:
        target = convergence.TARGETS[index // 6]
        return {
            "schema": "CORELAB_STAGE45C_TRAIN24_EVALUATION_ROW_V1",
            "window_id": convergence.WINDOW_IDS[index],
            "target_id": target,
            "checkpoint": {"label": label, "sha256": checkpoint_sha},
            "model": {"label": label, "sha256": checkpoint_sha},
            "metrics": {
                "target": {
                    "final_target_distance_m": 0.2,
                    "absolute_progress_m": 0.1,
                    "minimum_target_distance_m": 0.15,
                    "endpoint_target_vector_cosine": 0.8,
                    "path_length_m": 0.3,
                    "roughness_position_second_difference_m": {"mean": 0.01},
                    "progress_sign": "POSITIVE",
                }
            },
            "normalized_metrics": {
                "active_rmse": 0.2,
                "active_mae": 0.1,
                "active_max_abs": 0.4,
                "raw_active_outside_neg1_pos1_fraction": 0.0,
            },
            "sampler": {"steps_observed": 4, "official_public_sampler_called": False},
            "safety_gate_decision": None,
            "target_xyz_in_model_input": False,
            "finite": True,
        }

    def _index(self, root: Path) -> Path:
        arms: dict[str, dict] = {}
        zero = "0" * 64
        manifest = convergence.manifest()
        for label in manifest["matrix"]["arms"]:
            route, seed, step = convergence._arm_axes(label)
            rows = [self._row(index, label, zero) for index in range(24)]
            rows_path = root / (label.replace("(", "_").replace(")", "").replace("@", "_") + ".json")
            rows_path.write_text(json.dumps(rows, sort_keys=True) + "\n")
            route_cfg = manifest["h200_campaign_surfaces"]["routes"][route]
            frozen = manifest["h200_campaign_surfaces"]["frozen_sources"]
            arms[label] = {
                "rows_json": str(rows_path),
                "rows_sha256": convergence.sha256_file(rows_path),
                "provenance": {
                    "route": route,
                    "seed": seed,
                    "step": step,
                    "row_checkpoint_label": label,
                    "checkpoint_tree_sha256": zero,
                    "evidence_seal_sha256": zero,
                    "lineage_artifact_sha256": zero,
                    "route_branch": route_cfg["branch"],
                    "route_head": route_cfg["head"],
                    "normalization_identity": route_cfg["normalization"],
                    "stats_sha256": route_cfg["stats_sha256"],
                    "stats_payload_sha256": route_cfg["stats_payload_sha256"],
                    "hardware": "H200",
                    "decoder_identity": "accepted Stage45C decoder",
                    "metric_identity": "accepted Stage44R metrics",
                    "sampler_identity": "accepted H200 E4 row semantics",
                    "raw_before_derived": True,
                    "full_state_checkpoint": True,
                    "completed_step": step,
                    "target_xyz_in_model_input": False,
                    "base_tree_sha256": frozen["base_tree_sha256"],
                    "full24_tree_sha256": frozen["full24_tree_sha256"],
                    "training_result_sha256": zero,
                    "trainer_state_sha256": zero,
                    "hardware_runtime_sha256": zero,
                    "training_recipe_sha256": zero,
                    "validation_e_read": False,
                    "holdout_f_read": False,
                },
            }
        index_path = root / "index.json"
        index_path.write_text(json.dumps({
            "schema": "STAGE45F_H200_TRAIN24_EVIDENCE_INDEX_V1",
            "split": "TRAIN24",
            "validation_e_read": False,
            "holdout_f_read": False,
            "arms": arms,
        }, sort_keys=True) + "\n")
        return index_path

    def test_full_matrix_packet_is_paired_and_neutral(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stage45f-convergence-test-") as directory:
            root = Path(directory)
            index = self._index(root)
            output = root / "packet.json"
            packet = convergence.build_packet(index, output)
            self.assertEqual(packet["status"], "PASS_DESCRIPTIVE_H200_TRAIN24_ONLY")
            self.assertEqual(packet["decision_state"], "PI_DECISION_REQUIRED")
            self.assertFalse(packet["winner_declared"])
            self.assertEqual(len(packet["arm_summaries"]), 18)
            self.assertEqual(len(packet["paired_comparisons"]), 21)
            self.assertIn("N2", packet["three_seed_descriptive"]["route_minus_legacy"])
            self.assertFalse(packet["three_seed_descriptive"]["window_iid_inference"])
            first_pair = packet["paired_comparisons"]["n2_minus_legacy_seed2222_at_1000"]
            self.assertEqual(first_pair["physical_delta_exact_sign_counts"]["absolute_progress_m"]["zero"], 24)
            self.assertFalse(packet["validation_e_read"])
            self.assertEqual(packet["execution_boundary"]["model_forward"], 0)

    def test_locked_path_is_rejected_before_read(self) -> None:
        with self.assertRaises(convergence.ConvergenceError):
            convergence.read_json(Path("/tmp/Validation-E/not-an-input.json"))

    def test_all_undefined_optional_cosine_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stage45f-convergence-cosine-") as directory:
            root = Path(directory)
            index_path = self._index(root)
            index = json.loads(index_path.read_text())
            for spec in index["arms"].values():
                rows_path = Path(spec["rows_json"])
                rows = json.loads(rows_path.read_text())
                for row in rows:
                    row["metrics"]["target"]["endpoint_target_vector_cosine"] = None
                rows_path.write_text(json.dumps(rows, sort_keys=True) + "\n")
                spec["rows_sha256"] = convergence.sha256_file(rows_path)
            index_path.write_text(json.dumps(index, sort_keys=True) + "\n")
            packet = convergence.build_packet(index_path, root / "packet.json")
            summary = packet["arm_summaries"]["LEGACY(seed2222)@1000"]["physical"]["endpoint_target_vector_cosine"]
            self.assertEqual(summary["status"], "ALL_UNDEFINED_ACCEPTED")


if __name__ == "__main__":
    unittest.main()
