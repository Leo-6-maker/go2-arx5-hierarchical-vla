from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import stage45f_corelab_evidence_convergence as convergence
import stage45f_corelab_pi0143_compat as compat


ZERO = "0" * 64
GIT = "a" * 40


def row(index: int, label: str) -> dict:
    return {
        "schema": "CORELAB_STAGE45C_TRAIN24_EVALUATION_ROW_V1",
        "window_id": convergence.WINDOW_IDS[index], "target_id": convergence.TARGETS[index // 6],
        "metrics": {"target": {"final_target_distance_m": .2, "absolute_progress_m": .1, "minimum_target_distance_m": .15, "endpoint_target_vector_cosine": .8, "path_length_m": .3, "roughness_position_second_difference_m": {"mean": .01}, "progress_sign": "POSITIVE"}},
        "normalized_metrics": {"active_rmse": .2, "active_mae": .1, "active_max_abs": .3, "raw_active_outside_neg1_pos1_fraction": 0.0},
        "target_xyz_in_model_input": False, "finite": True,
    }


class CorelabPI0143CompatibilityTest(unittest.TestCase):
    def test_explicit_row_conversion_preserves_source_identity(self) -> None:
        bundle = compat.canonicalize_rows([row(i, "x") for i in range(24)], "CORELAB_STAGE45C_TRAIN24_EVALUATION_ROW_V1", ZERO)
        self.assertEqual(bundle["schema"], "STAGE45F_C10_CANONICAL_TRAIN24_ROW_BUNDLE_V1")
        self.assertEqual(bundle["source_schema"], "CORELAB_STAGE45C_TRAIN24_EVALUATION_ROW_V1")
        self.assertEqual(bundle["source_sha256"], ZERO)
        self.assertTrue(bundle["primary_physical_metrics_preserved"])
        with self.assertRaises(compat.CompatibilityError):
            compat.canonicalize_rows([row(i, "x") for i in range(24)], "UNKNOWN_ROW_V9", ZERO)

    def test_lineage_variants_and_identity_separation(self) -> None:
        common = {"training_head": GIT, "evaluation_head": GIT, "adapter_head": GIT, "orchestration_head": GIT, "checkpoint_tree_sha256": ZERO, "producer_fabrication": False}
        standard = {**common, "source_kind": "STANDARD_PASS_RESULT", "training_result": {"schema": "TRAIN_RESULT"}}
        self.assertFalse(compat.validate_lineage(standard)["recovered_primary"])
        recovered = {**common, "source_kind": "RECOVERED_PRIMARY", "training_result": None, "recovered_input_v2_sha256": ZERO, "primary_retry_substitution": False}
        self.assertTrue(compat.validate_lineage(recovered)["recovered_primary"])
        broken = {**recovered, "training_result": {"schema": "FABRICATED"}}
        with self.assertRaises(compat.CompatibilityError):
            compat.validate_lineage(broken)

    def test_external_handoff_and_campaign_history_accounting(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stage45f-pi0143-external-") as directory:
            parent = Path(directory)
            campaign_root = parent / "campaign"
            campaign_root.mkdir()
            external = parent / "diagnostic.json"
            external.write_text("{\"sealed\":true}\n", encoding="utf-8")
            handoff = parent / "handoff.json"
            handoff.write_text(json.dumps({"schema": "STAGE45E_EXTERNAL_DIAGNOSTIC_HANDOFF_V1", "validation_e_read": False, "holdout_f_read": False, "artifacts": [{"path": str(external), "sha256": convergence.sha256_file(external), "schema": "SEALED"}]}) + "\n", encoding="utf-8")
            self.assertEqual(compat.validate_external_diagnostic_handoff(handoff, campaign_root)["artifact_count"], 1)
        accounting = compat.validate_campaign_accounting({"campaign_history": {"training": True}, "cpu_closure_execution": {"training": False}, "accepted_unique_scientific_calls": 12, "all_attempted_model_calls": 13, "failed_attempt_model_calls": 1, "reused_readonly_model_calls": 0})
        self.assertTrue(accounting["campaign_history_training"])
        with self.assertRaises(compat.CompatibilityError):
            compat.validate_campaign_accounting({"campaign_history": {"training": False}, "cpu_closure_execution": {"training": False}, "accepted_unique_scientific_calls": 1, "all_attempted_model_calls": 1, "failed_attempt_model_calls": 0, "reused_readonly_model_calls": 0})

    def test_noise_decoder_and_l1_secondary_contracts(self) -> None:
        noise = {window: ZERO for window in convergence.WINDOW_IDS}
        descriptor = {"split": "TRAIN24", "window_count": 24, "window_ids": list(convergence.WINDOW_IDS), "noise_sha256_by_window": noise, "sampler": {"name": "canonical-E4", "steps_observed": 4, "official_public_sampler_called": False}, "target_xyz_in_model_input": False}
        self.assertTrue(compat.validate_window_noise_sampler(descriptor)["noise_identity_bound"])
        payload = {"schema": "STAGE45F_C10_DECODER_SEMANTIC_EQUIVALENCE_INPUT_V1", "stats_payload_sha256": ZERO, "stage45c_v2": [[0.1, 0.2], [0.3, 0.4]], "stage45e_with_stats": [[0.1, 0.2], [0.3, 0.4]]}
        self.assertEqual(compat.decoder_semantic_equivalence(payload)["status"], "PASS")
        raw = [{"schema": "VANDA_STAGE45E_L1_RAW_PREDICTION_ARTIFACT_V1", "window_id": convergence.WINDOW_IDS[i], "normalized_secondary_raw": {"active_rmse": .1, "active_mae": .2, "active_max_abs": .3, "raw_active_outside_neg1_pos1_fraction": 0.0}} for i in range(24)]
        reconstructed = compat.reconstruct_l1_secondary(raw, {"split": "TRAIN24", "source_kind": "SEALED_TRAIN_DESCRIPTOR"}, {"normalization": "FULL24_RUNTIME_V2", "payload_sha256": ZERO})
        self.assertTrue(reconstructed["primary_physical_metrics_replaced"] is False)

    def test_l1_n2_paired_delta_is_same_window_and_neutral(self) -> None:
        arms = {}
        for route, delta in (("L1", 0.0), ("N2", .01)):
            for seed in compat.SEEDS:
                for step in compat.STEPS:
                    arms[f"{route}(seed{seed})@{step}"] = [{"window_id": convergence.WINDOW_IDS[i], "target_id": convergence.TARGETS[i // 6], "physical": {metric: .2 + delta for metric in compat.PHYSICAL}} for i in range(24)]
        result = compat.l1_vs_n2_paired_delta(arms)
        self.assertEqual(result["status"], "PASS_DESCRIPTIVE_ONLY")
        self.assertFalse(result["winner_declared"])
        self.assertEqual(len(result["deltas"]), 6)

    def test_recipe_freeze_exact_one_route_one_tier_three_seeds(self) -> None:
        request = {"schema": "STAGE45F_C10_RECIPE_FREEZE_REQUEST_V1", "route": "N2", "checkpoint_step": 2000, "seeds": [2222, 2223, 2224], "checkpoint_trees": {f"N2(seed{seed})@2000": ZERO for seed in compat.SEEDS}, "best_seed_selection": False, "mixed_tier": False, "automatic_winner": False, "pi_selection_comment": 777}
        result = compat.strict_recipe_freeze(request)
        self.assertFalse(result["validation_execution_enabled"])
        for broken in ({**request, "seeds": [2222, 2223]}, {**request, "checkpoint_step": 1000, "checkpoint_trees": {"N2(seed2222)@1000": ZERO, "N2(seed2223)@2000": ZERO, "N2(seed2224)@2000": ZERO}}, {**request, "best_seed_selection": True}):
            with self.assertRaises(compat.CompatibilityError):
                compat.strict_recipe_freeze(broken)


if __name__ == "__main__":
    unittest.main()
