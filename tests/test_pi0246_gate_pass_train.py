import importlib.util
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pi0246_gate_pass_train.py"
SPEC = importlib.util.spec_from_file_location("pi0246_gate_pass_train", SCRIPT)
CAMPAIGN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CAMPAIGN)


class ScheduleTests(unittest.TestCase):
    def test_projector_probe_fallback_covers_physical_loss_arms(self):
        integrity = {"projector_updated": False, "diffusion_updated": True,
                     "frozen_backbone_unchanged": True, "optimizer_membership_valid": True}
        self.assertTrue(CAMPAIGN.needs_projector_reconciliation("T", integrity))
        self.assertTrue(CAMPAIGN.needs_projector_reconciliation("P", integrity))
        self.assertTrue(CAMPAIGN.needs_projector_reconciliation("PT", integrity))
        self.assertFalse(CAMPAIGN.needs_projector_reconciliation("C", integrity))
        integrity["diffusion_updated"] = False
        self.assertFalse(CAMPAIGN.needs_projector_reconciliation("P", integrity))

    def test_accepts_exact_r0_prefix_with_125_exposures_per_window(self):
        cohort = [{"source_episode_id": f"source-{index}"} for index in range(16)]
        schedule = [
            {"optimizer_step": step, "fixed_index": (step - 1) % 16,
             "source_episode_id": cohort[(step - 1) % 16]["source_episode_id"]}
            for step in range(1, 2001)
        ]
        CAMPAIGN.verify_prefix(schedule, cohort)

    def test_rejects_unbalanced_or_reordered_prefix(self):
        cohort = [{"source_episode_id": f"source-{index}"} for index in range(16)]
        schedule = [
            {"optimizer_step": step, "fixed_index": (step - 1) % 16,
             "source_episode_id": cohort[(step - 1) % 16]["source_episode_id"]}
            for step in range(1, 2001)
        ]
        schedule[0]["fixed_index"] = 1
        with self.assertRaises(CAMPAIGN.CampaignError):
            CAMPAIGN.verify_prefix(schedule, cohort)

    def test_p_final_failure_joins_screen_inference_view_and_full_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            p_root = root / "P"
            (p_root / "durable_receipts").mkdir(parents=True)
            card_path = p_root / "run_card.json"
            card = {"arm": "P", "family": CAMPAIGN.P_ARM[0], "loss": CAMPAIGN.P_ARM[1],
                    "status": "SEALED_READY_NOT_STARTED", "paths": {"output_root": str(p_root)}}
            CAMPAIGN.write_json(card_path, card)
            CAMPAIGN.write_json(p_root / "run_card.seal.json", {"sha256": CAMPAIGN.sha256_file(card_path)})
            tree_sha, revision, restore_sha = "a" * 64, "b" * 40, "c" * 64
            receipt_path = p_root / "durable_receipts/checkpoint-2000.durable_receipt.json"
            CAMPAIGN.write_json(receipt_path, {
                "status": "DURABLE_VERIFIED", "family": CAMPAIGN.P_ARM[0], "checkpoint_step": 2000,
                "max_steps": 2000, "hf_manifest_commit_id": revision,
                "source_record": {"inventory": {"tree_sha256": tree_sha}},
            })
            view_path, verification_path = root / "view.json", root / "verification.json"
            CAMPAIGN.write_json(view_path, {
                "schema": "PI0246_INFERENCE_VIEW_VERIFICATION_V1", "status": "INFERENCE_VIEW_VERIFIED",
                "family": CAMPAIGN.P_ARM[0], "checkpoint_step": 2000, "optimizer_restored": False,
                "full_checkpoint_receipt_sha256": CAMPAIGN.sha256_file(receipt_path),
                "source_tree_sha256": tree_sha, "revision": revision,
                "restore_manifest_sha256": restore_sha,
            })
            comparisons = [{"source_episode_id": f"cell-{i}", "gate": "REJECT" if i == 0 else "WARN",
                           "xyz_rmse_m": 0.04, "so3_rmse_rad": 0.03} for i in range(16)]
            CAMPAIGN.write_json(verification_path, {
                "schema": "PI0246_PAIRED_PROFILE_CPU_VERIFICATION_V1", "status": "PASS_READONLY_PROFILE_REPLAY",
                "profile": "P2000", "records": 16, "model_weights_loaded": 0, "model_queries": 0,
                "gate_counts": {"REJECT": 1, "WARN": 15}, "comparisons": comparisons,
                "model_manifest_sha256": restore_sha, "model_revision": revision,
                "records_sha256": "d" * 64, "queries_sha256": "e" * 64,
            })
            result = CAMPAIGN.verify_p_final_failure(card_path, verification_path, view_path)
            self.assertEqual(result["status"], "VALID_JOINT_GATE_FAIL")
            self.assertEqual(result["metrics"]["gate_counts"], {"REJECT": 1, "WARN": 15})
            self.assertAlmostEqual(CAMPAIGN.percentile_linear([0, 1, 2, 3], .95), 2.85)

            passing = CAMPAIGN.read_json(verification_path)
            for row in passing["comparisons"]:
                row["gate"] = "WARN"
                row["xyz_rmse_m"] = 0.01
                row["so3_rmse_rad"] = 0.01
            passing["gate_counts"] = {"WARN": 16}
            CAMPAIGN.write_json(verification_path, passing)
            with self.assertRaises(CAMPAIGN.CampaignError):
                CAMPAIGN.verify_p_final_failure(card_path, verification_path, view_path)

    def test_p1_admission_requires_exact_query_order_and_r0_identity(self):
        sources = [f"source-{index}:obs-0001" for index in range(4)]
        noises = ["N0", "N1", "N2", "N0_roll11"]
        expected = [f"{model}__{source}__{noise}" for model in ("R0", "R2") for source in sources for noise in noises]
        manifest_sha = "a" * 64
        records = []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for key in expected:
                model, source, noise = key.split("__", 2)
                record = {"cell_key": key}
                if model == "R0":
                    record.update({"manifest_sha256": manifest_sha, "revision": "pinned-r0"})
                cell = root / "cells" / f"{model}__{source.split(':', 1)[0]}__{noise}"
                cell.mkdir(parents=True)
                CAMPAIGN.write_json(cell / "record.json", record)
                records.append(record)
            closure = {
                "schema": "PI0245_P1_CLOSURE_V1", "status": "PASS_COMPLETE_32_QUERY_MATRIX",
                "attempted_queries": 32, "completed_cells": 32, "query_order": expected,
                "cells": records, "noise_bank_manifest_sha256": CAMPAIGN.EXPECTED_NOISE_SHA256,
                "R0_records_sha256": "b" * 64,
            }
            path = root / "p1_closure.json"
            CAMPAIGN.write_json(path, closure)
            self.assertEqual(
                CAMPAIGN.verify_p1(path, expected_order=expected, expected_r0_manifest_sha256=manifest_sha,
                                    expected_r0_revision="pinned-r0", expected_r0_records_sha256="b" * 64)["status"],
                "PASS_COMPLETE_32_QUERY_MATRIX",
            )
            closure["query_order"][0], closure["query_order"][1] = closure["query_order"][1], closure["query_order"][0]
            CAMPAIGN.write_json(path, closure)
            with self.assertRaises(CAMPAIGN.CampaignError):
                CAMPAIGN.verify_p1(path, expected_order=expected, expected_r0_manifest_sha256=manifest_sha,
                                   expected_r0_revision="pinned-r0", expected_r0_records_sha256="b" * 64)


if __name__ == "__main__":
    unittest.main()
