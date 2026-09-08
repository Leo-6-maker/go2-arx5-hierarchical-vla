"""CPU-only checks for the Stage45A FULL24 design package."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests" / "STAGE45A_FULL24_ADAPTATION_DESIGN_V1.json"
PROTOCOL = ROOT / "docs" / "stages" / "STAGE45A_FULL24_ADAPTATION_PROTOCOL.md"


class Stage45AProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with MANIFEST.open(encoding="utf-8") as handle:
            cls.payload = json.load(handle)

    def test_design_identity_and_return_gate(self) -> None:
        payload = self.payload
        self.assertEqual(payload["schema"], "CORELAB_STAGE45A_FULL24_ADAPTATION_DESIGN_V1")
        self.assertEqual(payload["status"], "DESIGN_ONLY_READY_FOR_PI")
        self.assertEqual(payload["token"], "CORELAB_STAGE45_FULL24_ADAPTATION_PROTOCOL_READY")
        self.assertEqual(payload["pi_gate"], "PI_GATE_STATE_0072")
        self.assertEqual(
            payload["repository"]["base_main_merge_tip"],
            "bfa14bf1b74eae908c0922575a4db671c903b9b8",
        )
        self.assertEqual(
            payload["repository"]["accepted_stage44r_head"],
            "bd0f02934e680789cf38adacb838e1ea555c707d",
        )

    def test_full24_identity_and_mapping(self) -> None:
        level1 = self.payload["data_ladder"]["level_1"]
        self.assertEqual(level1["id"], "FULL24_EXISTING_TRAIN")
        self.assertEqual(
            level1["canonical_tree_sha256"],
            "2f3badd2019a68c6549a66f85b5486faf626078a3e438342f67aacc4edc95789",
        )
        self.assertEqual(level1["regular_file_count"], 85)
        self.assertEqual(level1["total_bytes"], 919661)
        self.assertEqual(level1["window_count"], 24)
        self.assertFalse(level1["new_data_generation"])

        mapping = level1["window_mapping"]
        self.assertEqual(len(mapping), 24)
        targets = level1["target_order"]
        for index, row in enumerate(mapping):
            expected_target = targets[index // 6]
            expected_snapshot = index % 6
            self.assertEqual(row["window_index"], index)
            self.assertEqual(row["materialized_episode_index"], index)
            self.assertEqual(row["target_id"], expected_target)
            self.assertEqual(row["snapshot_index"], expected_snapshot)
            self.assertEqual(
                row["source_episode_id"],
                f"stage22_expert_{expected_target}",
            )
            self.assertEqual(
                row["source_example_id"],
                f"stage22_expert_{expected_target}:expert-{expected_snapshot:04d}",
            )

    def test_contract_split_and_stats_are_fail_closed(self) -> None:
        payload = self.payload
        split_lock = payload["split_lock"]
        self.assertEqual(
            split_lock["sha256"],
            "7e124f29d2e4d24c17fd1fefcf8753ececd3a8cd86e8c6a2f159e2e5d2217246",
        )
        self.assertEqual(
            split_lock["train_target_ids"],
            [
                "red_target_train_a",
                "red_target_train_b",
                "red_target_train_c",
                "red_target_train_d",
            ],
        )
        self.assertEqual(split_lock["validation_target_ids"], ["red_target_holdout_e"])
        self.assertEqual(split_lock["final_test_target_ids"], ["red_target_holdout_f"])
        self.assertFalse(split_lock["target_actor_in_model_input"])
        self.assertFalse(split_lock["holdout_e_read"])
        self.assertFalse(split_lock["holdout_f_read"])

        contract = payload["contract"]
        self.assertEqual(contract["structured_action_schema"], "StructuredActionV1")
        self.assertEqual(contract["state_v1_shape"], [10])
        self.assertEqual(contract["body_rgb_shape"], [96, 96, 3])
        self.assertEqual(contract["wrist_rgb_shape"], [96, 96, 3])
        self.assertEqual(contract["universal_action_shape"], [1, 40, 132])
        self.assertEqual(contract["decoded_physical_action_shape"], [40, 10])
        self.assertEqual(contract["active_loss_dimensions"], 10)
        self.assertEqual(contract["masked_padding_dimensions"], 122)
        self.assertFalse(contract["target_xyz_in_model_input"])
        self.assertFalse(contract["wbc_semantics_changed"])
        self.assertFalse(contract["d1_decoder_changed"])
        self.assertFalse(contract["safety_gate_changed"])
        self.assertFalse(contract["timing_or_handoff_changed"])

        normalization = payload["normalization"]
        self.assertEqual(normalization["structured_stats_source"], "FULL24 training labels only")
        self.assertEqual(normalization["selected_lower_bound"], "q01")
        self.assertEqual(normalization["selected_upper_bound"], "q99")
        self.assertTrue(normalization["inactive_mask_excluded"])
        self.assertIsNone(normalization["structured_stats_sha256"])
        self.assertEqual(
            normalization["structured_stats_hash_status"],
            "PENDING_CPU_PREFLIGHT_BEFORE_MODEL_CONSTRUCTION",
        )
        self.assertFalse(normalization["outcome_dependent_stats_swap"])

    def test_option_and_budget_policy(self) -> None:
        payload = self.payload
        self.assertEqual(payload["primary_option"], "B_FRESH_BASE_FULL24")
        option_a = payload["options"]["A_WARM_START_FULL24"]
        option_b = payload["options"]["B_FRESH_BASE_FULL24"]
        self.assertFalse(option_a["primary_coverage_claim_eligible"])
        self.assertTrue(option_b["primary_coverage_claim_eligible"])
        self.assertEqual(option_a["new_full24_steps"], 2000)
        self.assertEqual(option_b["new_full24_steps"], 2000)
        self.assertFalse(option_a["resume_from_checkpoint"])
        self.assertFalse(option_b["resume_from_checkpoint"])

        recipe = payload["recipe"]
        self.assertEqual(recipe["seed"], 2222)
        self.assertEqual(recipe["max_steps"], 2000)
        self.assertEqual(recipe["save_steps"], [1000, 2000])
        self.assertEqual(recipe["learning_rate"], 1e-4)
        self.assertEqual(recipe["weight_decay"], 1e-5)
        self.assertEqual(recipe["warmup_steps"], 10)
        self.assertTrue(recipe["bf16"])
        self.assertFalse(recipe["fp16"])
        self.assertFalse(recipe["resume_from_checkpoint"])
        self.assertEqual(recipe["num_gpus"], 1)

        policy = payload["checkpoint_policy"]
        self.assertEqual(policy["primary_endpoint"], "checkpoint-2000")
        self.assertEqual(policy["descriptive_checkpoint"], "checkpoint-1000")
        self.assertFalse(policy["loss_based_selection"])
        self.assertFalse(policy["adaptive_step_extension"])
        self.assertFalse(policy["validation_or_holdout_selection"])

    def test_future_ladder_and_resource_boundary(self) -> None:
        payload = self.payload
        level2 = payload["data_ladder"]["level_2"]
        self.assertEqual(level2["id"], "MULTIINIT120")
        self.assertEqual(level2["window_count"], 120)
        self.assertFalse(level2["generation_authorized"])
        self.assertFalse(level2["simulator_read"])

        boundary = payload["resource_boundary"]
        for key in (
            "gr00t_forwards",
            "wbc_steps",
            "reference_updates",
            "new_data_generation",
            "training_steps",
            "backward_steps",
            "optimizer_steps",
            "pbs_submissions",
        ):
            self.assertEqual(boundary[key], 0, key)
        self.assertFalse(boundary["gpu_or_simulator_execution"])
        self.assertEqual(boundary["h200"], "HOLD")
        self.assertEqual(boundary["vanda_a40"], "HOLD")
        self.assertEqual(boundary["validation_e"], "LOCKED")
        self.assertEqual(boundary["holdout_f"], "LOCKED")

        self.assertTrue(all(not value for value in payload["forbidden_operations"].values()))

    def test_protocol_document_present_and_design_only(self) -> None:
        text = PROTOCOL.read_text(encoding="utf-8")
        for marker in (
            "FULL24_EXISTING_TRAIN",
            "MULTIINIT120",
            "CORELAB_STAGE45_FULL24_ADAPTATION_PROTOCOL_READY",
            "target XYZ is not an input tensor",
            "does not authorize H200",
        ):
            self.assertIn(marker, text)


if __name__ == "__main__":
    unittest.main()
