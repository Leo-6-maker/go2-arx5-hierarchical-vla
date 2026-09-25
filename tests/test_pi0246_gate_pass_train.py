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
                "source_record": {"inventory": {"tree_sha256": tree_sha, "total_bytes": 123}},
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

    def test_pt_retirement_requires_a_matching_durable_hf_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "PT"
            training = output / "training"
            target = training / "checkpoint-1000"
            target.mkdir(parents=True)
            (target / "trainer_state.json").write_text("sealed full checkpoint\n", encoding="utf-8")
            digest = "a" * 64
            record = {
                "schema": "PI0243_DURABLE_CHECKPOINT_RECORD_V1",
                "family": "PT_FAMILY", "checkpoint_step": 1000, "max_steps": 2000,
                "global_step": 1000,
                "inventory": {
                    "file_count": 1, "total_bytes": 23, "tree_sha256": "b" * 64,
                    "entries": [{"path": "trainer_state.json", "bytes": 23, "sha256": digest}],
                },
            }
            receipt_path = output / "durable_receipts" / "checkpoint-1000.durable_receipt.json"
            readback_path = output / "durable_receipts" / "checkpoint-1000.hf_readback.json"
            receipt_path.parent.mkdir(parents=True)
            receipt = {
                "status": "DURABLE_VERIFIED", "family": "PT_FAMILY", "checkpoint_step": 1000,
                "max_steps": 2000, "durable_host": "huggingface",
                "durable_path": "leoyuer/pi0246-gate-pass-campaign/arms/PT/checkpoint-1000",
                "hf_manifest_commit_id": "f" * 40, "source_record": record,
                "durable_inventory": [{
                    "path": "arms/PT/checkpoint-1000/trainer_state.json", "bytes": 23, "sha256": "c" * 64,
                }],
            }
            CAMPAIGN.write_json(receipt_path, receipt)
            with self.assertRaises(CAMPAIGN.CampaignError):
                CAMPAIGN.retire_pt_checkpoint_after_verified_archive(
                    output, training, 1000, record, receipt_path, readback_path,
                    repo_id="leoyuer/pi0246-gate-pass-campaign", prefix="arms/PT", family="PT_FAMILY",
                )
            self.assertTrue(target.is_dir())
            receipt["durable_inventory"][0]["sha256"] = digest
            CAMPAIGN.write_json(receipt_path, receipt)
            CAMPAIGN.write_json(readback_path, {
                "status": "PASS_PT_FULL_HF_READBACK", "family": "PT_FAMILY",
                "checkpoint_step": 1000, "max_steps": 2000,
                "repo_id": "leoyuer/pi0246-gate-pass-campaign", "path_prefix": "arms/PT",
                "revision": "f" * 40, "source_tree_sha256": "b" * 64, "file_count": 1,
                "files": [{"path": "arms/PT/checkpoint-1000/trainer_state.json", "bytes": 23, "sha256": digest}],
            })
            intent, completed = CAMPAIGN.retire_pt_checkpoint_after_verified_archive(
                output, training, 1000, record, receipt_path, readback_path,
                repo_id="leoyuer/pi0246-gate-pass-campaign", prefix="arms/PT", family="PT_FAMILY",
            )
            self.assertFalse(target.exists())
            self.assertEqual(CAMPAIGN.read_json(intent)["status"], "DURABLE_HF_COPY_VERIFIED_DELETE_AUTHORIZED")
            self.assertEqual(CAMPAIGN.read_json(completed)["status"], "LOCAL_COPY_REMOVED_AFTER_DURABLE_VERIFICATION")

    def test_pt_readback_hashes_non_lfs_files_at_the_pinned_revision(self):
        record = {"inventory": {
            "tree_sha256": "a" * 64,
            "entries": [
                {"path": "weights.safetensors", "bytes": 8, "sha256": "b" * 64},
                {"path": "config.json", "bytes": 2, "sha256": "c" * 64},
            ],
        }}
        receipt = {
            "hf_manifest_commit_id": "d" * 40,
            "durable_inventory": [
                {"path": "arms/PT/checkpoint-2000/weights.safetensors", "bytes": 8, "sha256": "b" * 64},
                {"path": "arms/PT/checkpoint-2000/config.json", "bytes": 2, "sha256": None},
            ],
        }

        class StorageHelper:
            @staticmethod
            def stream_remote_sha(repo_id, path, revision, token):
                self.assertEqual(repo_id, "leoyuer/pi0246-gate-pass-campaign")
                self.assertEqual(revision, "d" * 40)
                self.assertEqual(token, "test-token-not-a-secret")
                return "c" * 64

        result = CAMPAIGN.verify_pt_hf_readback(
            receipt, record, storage_helper=StorageHelper(), token="test-token-not-a-secret",
            repo_id="leoyuer/pi0246-gate-pass-campaign", prefix="arms/PT",
            family="PT_FAMILY", step=2000,
        )
        self.assertEqual(result["status"], "PASS_PT_FULL_HF_READBACK")
        self.assertEqual(result["file_count"], 2)
        self.assertEqual(result["files"][1]["method"], "pinned_revision_stream_sha256")

    def test_pt_campaign_snapshot_is_isolated_and_byte_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, runtime, code, snapshot = (root / name for name in ("source", "runtime", "code", "pt-code"))
            for directory in (source, runtime, code / "scripts", snapshot / "scripts"):
                directory.mkdir(parents=True)
            files = {
                source / "pi0245_p1_stock.py": "p1 runner\n",
                runtime / "pi0246_gate_pass_train.py": "trainer\n",
                runtime / "pi0246_reconcile_c_update.py": "reconcile\n",
                code / "scripts/pi0243_storage_retirement.py": "readback helper\n",
            }
            destinations = {
                snapshot / "pi0245_p1_stock.py": files[source / "pi0245_p1_stock.py"],
                snapshot / "pi0246_gate_pass_train.py": files[runtime / "pi0246_gate_pass_train.py"],
                snapshot / "scripts/pi0246_reconcile_c_update.py": files[runtime / "pi0246_reconcile_c_update.py"],
                snapshot / "scripts/pi0243_storage_retirement.py": files[code / "scripts/pi0243_storage_retirement.py"],
            }
            for path, content in {**files, **destinations}.items():
                path.write_text(content, encoding="utf-8")
            CAMPAIGN.validate_pt_campaign_snapshot(snapshot, source, runtime, code)
            (snapshot / "pi0245_p1_stock.py").write_text("changed runner\n", encoding="utf-8")
            with self.assertRaises(CAMPAIGN.CampaignError):
                CAMPAIGN.validate_pt_campaign_snapshot(snapshot, source, runtime, code)
            with self.assertRaises(CAMPAIGN.CampaignError):
                CAMPAIGN.validate_pt_campaign_snapshot(source / "nested", source, runtime, code)


if __name__ == "__main__":
    unittest.main()
