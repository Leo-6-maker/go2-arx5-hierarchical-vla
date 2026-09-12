from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import stage45f_corelab_evidence_convergence as convergence
import stage45f_corelab_pi0136 as pi0136


ZERO = "0" * 64


class CorelabPI0136Test(unittest.TestCase):
    def _row(self, index: int, label: str, offset: float = 0.0) -> dict:
        return {
            "schema": "CORELAB_STAGE45C_TRAIN24_EVALUATION_ROW_V1",
            "window_id": convergence.WINDOW_IDS[index],
            "target_id": convergence.TARGETS[index // 6],
            "checkpoint": {"label": label, "sha256": ZERO},
            "model": {"label": label, "sha256": ZERO},
            "metrics": {"target": {
                "final_target_distance_m": 0.2 + offset,
                "absolute_progress_m": 0.1 - offset,
                "minimum_target_distance_m": 0.15 + offset,
                "endpoint_target_vector_cosine": 0.8 - offset,
                "path_length_m": 0.3 + offset,
                "roughness_position_second_difference_m": {"mean": 0.01 + offset},
                "progress_sign": "POSITIVE",
            }},
            "normalized_metrics": {
                "active_rmse": 0.2 + offset,
                "active_mae": 0.1 + offset,
                "active_max_abs": 0.4 + offset,
                "raw_active_outside_neg1_pos1_fraction": 0.0,
            },
            "sampler": {"steps_observed": 4, "official_public_sampler_called": False},
            "safety_gate_decision": None,
            "target_xyz_in_model_input": False,
            "finite": True,
        }

    def _h200_index(self, root: Path) -> Path:
        cfg = convergence.manifest()
        frozen = cfg["h200_campaign_surfaces"]["frozen_sources"]
        arms = {}
        for label in cfg["matrix"]["arms"]:
            route, seed, step = convergence._arm_axes(label)
            rows_path = root / f"h200_{route}_{seed}_{step}.json"
            rows_path.write_text(json.dumps([self._row(index, label) for index in range(24)], sort_keys=True) + "\n")
            route_cfg = cfg["h200_campaign_surfaces"]["routes"][route]
            arms[label] = {
                "rows_json": str(rows_path),
                "rows_sha256": convergence.sha256_file(rows_path),
                "provenance": {
                    "route": route, "seed": seed, "step": step,
                    "row_checkpoint_label": label,
                    "checkpoint_tree_sha256": ZERO,
                    "evidence_seal_sha256": ZERO,
                    "lineage_artifact_sha256": ZERO,
                    "training_result_sha256": ZERO,
                    "trainer_state_sha256": ZERO,
                    "hardware_runtime_sha256": ZERO,
                    "training_recipe_sha256": ZERO,
                    "route_branch": route_cfg["branch"],
                    "route_head": route_cfg["head"],
                    "normalization_identity": route_cfg["normalization"],
                    "stats_sha256": route_cfg["stats_sha256"],
                    "stats_payload_sha256": route_cfg["stats_payload_sha256"],
                    "hardware": "H200",
                    "decoder_identity": "accepted Stage45C decoder",
                    "metric_identity": "accepted Stage44R metrics",
                    "sampler_identity": "accepted E4",
                    "raw_before_derived": True,
                    "full_state_checkpoint": True,
                    "completed_step": step,
                    "target_xyz_in_model_input": False,
                    "base_tree_sha256": frozen["base_tree_sha256"],
                    "full24_tree_sha256": frozen["full24_tree_sha256"],
                    "validation_e_read": False,
                    "holdout_f_read": False,
                },
            }
        path = root / "h200_index.json"
        path.write_text(json.dumps({
            "schema": "STAGE45F_H200_TRAIN24_EVIDENCE_INDEX_V1",
            "split": "TRAIN24", "validation_e_read": False, "holdout_f_read": False,
            "arms": arms,
        }, sort_keys=True) + "\n")
        return path

    def _pass_lineage_ledger(self, root: Path) -> Path:
        arms = [{
            "arm": label,
            "audit_state": "PASS",
            "accepted_lineage_id": f"synthetic-{label}",
            "checkpoint_tree_sha256": ZERO,
            "training_result_sha256": ZERO,
            "trainer_state_sha256": ZERO,
            "lineage_artifact_sha256": ZERO,
            "duplicate_complete_lineages": 0,
        } for label in convergence.manifest()["matrix"]["arms"]]
        path = root / "lineages.json"
        path.write_text(json.dumps({
            "schema": "CORELAB_STAGE45F_LINEAGE_ACCEPTANCE_LEDGER_V1",
            "status": "PASS",
            "policy": {
                "performance_based_retry_selection": False,
                "duplicate_complete_lineage_requires_pi": True,
            },
            "arms": arms,
            "pass_arm_count": 18,
            "blocked_arm_count": 0,
        }, sort_keys=True) + "\n")
        return path

    def _hardware_index(self, root: Path) -> Path:
        cfg = convergence.manifest()
        hw_cfg = cfg["n2_cross_hardware"]
        frozen = cfg["h200_campaign_surfaces"]["frozen_sources"]
        labels = [f"N2(seed{seed})@{step}" for seed in convergence.SEEDS for step in convergence.STEPS]
        arms = {hardware: {} for hardware in pi0136.HARDWARES}
        for hardware_index, hardware in enumerate(pi0136.HARDWARES):
            for label in labels:
                _, seed, step = convergence._arm_axes(label)
                rows_path = root / f"{hardware}_{seed}_{step}.json"
                rows_path.write_text(json.dumps([self._row(index, label, hardware_index * 0.001) for index in range(24)], sort_keys=True) + "\n")
                arms[hardware][label] = {
                    "rows_json": str(rows_path),
                    "rows_sha256": convergence.sha256_file(rows_path),
                    "provenance": {
                        "route": "N2", "seed": seed, "step": step, "hardware": hardware,
                        "route_head": hw_cfg["historical_4080s_head"] if hardware == "RTX4080S" else hw_cfg["h200_head"],
                        "stats_sha256": hw_cfg["accepted_stats_file_sha256_by_hardware"][hardware],
                        "stats_payload_sha256": hw_cfg["stats_payload_sha256"],
                        "base_tree_sha256": frozen["base_tree_sha256"],
                        "full24_tree_sha256": frozen["full24_tree_sha256"],
                        "full_state_checkpoint": True,
                        "completed_step": step,
                        "target_xyz_in_model_input": False,
                        "row_checkpoint_label": label,
                        "checkpoint_tree_sha256": ZERO,
                        "evidence_seal_sha256": ZERO,
                        "lineage_artifact_sha256": ZERO,
                        "training_result_sha256": ZERO,
                        "trainer_state_sha256": ZERO,
                        "training_recipe_identity": {**cfg["h200_campaign_surfaces"]["training_recipe"], "seed": seed},
                        "science_source_identity": {
                            "groot_commit": frozen["groot_commit"],
                            "cosmos_revision": frozen["cosmos_revision"],
                            "cosmos_tree_sha256": frozen["cosmos_tree_sha256"],
                            "modality_sha256": frozen["modality_sha256"],
                            "config_sha256": frozen["config_sha256"],
                        },
                        "device_runtime_identity": {"device": hardware},
                    },
                }
        path = root / "hardware_index.json"
        path.write_text(json.dumps({
            "schema": "STAGE45F_N2_CROSS_HARDWARE_EVIDENCE_INDEX_V1",
            "split": "TRAIN24", "validation_e_read": False, "holdout_f_read": False,
            "arms": arms,
        }, sort_keys=True) + "\n")
        return path

    def test_static_ledgers_and_readiness_are_fail_closed(self) -> None:
        static = pi0136.static_audit()
        self.assertEqual(static["status"], "PASS")
        pending = pi0136.build_readiness(None)
        self.assertEqual(pending["counts"]["PENDING_VANDA"], 18)
        with tempfile.TemporaryDirectory(prefix="stage45f-pi0136-readiness-") as directory:
            root = Path(directory)
            ready = pi0136.build_readiness(self._h200_index(root), lineage_ledger=self._pass_lineage_ledger(root))
            self.assertEqual(ready["counts"]["READY_SEALED"], 18)

    def test_n2_cross_hardware_comparator_is_descriptive(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stage45f-pi0136-hardware-") as directory:
            root = Path(directory)
            result = pi0136.compare_n2_hardware(self._hardware_index(root))
            self.assertEqual(result["pair_count"], 6)
            self.assertFalse(result["winner_declared"])
            self.assertFalse(result["validation_e_read"])

    def test_mechanism_packet_requires_sealed_inputs_and_stays_neutral(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stage45f-pi0136-mechanism-") as directory:
            root = Path(directory)
            train_path = root / "train_packet.json"
            convergence.build_packet(self._h200_index(root), train_path)
            hardware_path = root / "hardware_packet.json"
            pi0136.compare_n2_hardware(self._hardware_index(root), hardware_path)
            arms = {}
            for label in convergence.manifest()["matrix"]["arms"]:
                route, seed, step = convergence._arm_axes(label)
                arms[label] = {
                    "route": route, "seed": seed, "step": step,
                    "checkpoint_tree_sha256": ZERO,
                    "D1": {
                        "seal_sha256": ZERO, "rows_sha256": ZERO, "row_count": 216,
                        "raw_before_derived": True, "target_xyz_in_model_input": False,
                        "metrics": {"teacher_forced_local_flow_rmse": 0.2, "teacher_forced_local_flow_cosine": 0.8},
                    },
                    "D2A": {
                        "seal_sha256": ZERO, "rows_sha256": ZERO, "row_count": 96,
                        "raw_before_derived": True, "target_xyz_in_model_input": False,
                        "metrics": {
                            "tf_to_fr_state_path_divergence_l2": 0.3,
                            "field_sensitivity_l2": 0.2,
                            "teacher_bridge_local_error_l2": 0.1,
                            "off_path_diagnostic_l2": 0.4,
                        },
                    },
                }
            index_path = root / "mechanism_index.json"
            index_path.write_text(json.dumps({
                "schema": "STAGE45F_H200_MECHANISM_EVIDENCE_INDEX_V1",
                "split": "TRAIN24", "validation_e_read": False, "holdout_f_read": False,
                "train24_decision_packet": str(train_path),
                "train24_decision_packet_sha256": convergence.sha256_file(train_path),
                "n2_hardware_packet": str(hardware_path),
                "n2_hardware_packet_sha256": convergence.sha256_file(hardware_path),
                "arms": arms,
            }, sort_keys=True) + "\n")
            packet = pi0136.build_mechanism_packet(index_path)
            self.assertEqual(packet["decision_state"], "PI_DECISION_REQUIRED")
            self.assertFalse(packet["winner_declared"])
            self.assertFalse(packet["p_value_computed"])

    def _freeze_request(self, root: Path) -> Path:
        required = set(convergence.manifest()["recipe_freeze"]["required_fields"]) - {"selection_generation_id"}
        sha_fields = {"base_tree", "dataset_tree", "split_lock", "train24_packet_sha256", "evidence_inventory_sha256"}
        bindings = {field: ({"identity": field} if field not in sha_fields | {"checkpoint_trees", "validation_generation"} else ZERO) for field in required}
        bindings["checkpoint_trees"] = {"N2(seed2222)@2000": ZERO, "N2(seed2223)@2000": ZERO, "N2(seed2224)@2000": ZERO}
        bindings["validation_generation"] = 1
        bindings["route_intervention_identity"] = {"route": "N2", "identity": "N2_BALANCED_XYZ_STATS"}
        path = root / "freeze_request.json"
        path.write_text(json.dumps({
            "schema": "CORELAB_STAGE45F_RECIPE_FREEZE_REQUEST_V1",
            "status": "SYNTHETIC_TEST_ONLY",
            "synthetic_fixture": True,
            "selection_generation_id": "SYNTHETIC_SELECTION_GENERATION_1",
            "bindings": bindings,
        }, sort_keys=True) + "\n")
        return path

    def test_recipe_freeze_and_validation_preflight_are_plan_only(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stage45f-pi0136-freeze-") as directory:
            root = Path(directory)
            bundle_path = root / "freeze_bundle.json"
            bundle = pi0136.freeze_recipe(self._freeze_request(root), bundle_path)
            self.assertEqual(bundle["status"], "FROZEN_SYNTHETIC_TEST_ONLY")
            self.assertFalse(bundle["execution_enabled"])
            request_path = root / "validation_preflight.json"
            request_path.write_text(json.dumps({
                "schema": "CORELAB_STAGE45F_VALIDATION_E_GENERATION1_PREFLIGHT_REQUEST_V1",
                "recipe_freeze_bundle": str(bundle_path),
                "recipe_freeze_bundle_sha256": convergence.sha256_file(bundle_path),
                "access_ledger": {
                    "prior_validation_read_count": 0,
                    "next_generation": 1,
                    "holdout_read_count": 0,
                    "post_validation_tuning_count": 0,
                    "selection_generation_id": bundle["selection_generation_id"],
                },
            }, sort_keys=True) + "\n")
            preflight = pi0136.validation_preflight(request_path)
            self.assertEqual(preflight["status"], "LOCKED_PLAN_ONLY")
            self.assertFalse(preflight["validation_read"])
            self.assertTrue(preflight["PI_GATE_REQUIRED"])

    def test_recipe_freeze_refuses_missing_selection_generation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stage45f-pi0136-refusal-") as directory:
            root = Path(directory)
            path = self._freeze_request(root)
            value = json.loads(path.read_text())
            value["selection_generation_id"] = ""
            path.write_text(json.dumps(value, sort_keys=True) + "\n")
            with self.assertRaises(convergence.ConvergenceError):
                pi0136.freeze_recipe(path)


if __name__ == "__main__":
    unittest.main()
