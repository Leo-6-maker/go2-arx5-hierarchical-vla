import importlib.util
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pi0246_gate_pass_train.py"
SPEC = importlib.util.spec_from_file_location("pi0246_gate_pass_train", SCRIPT)
CAMPAIGN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CAMPAIGN)


class ScheduleTests(unittest.TestCase):
    def test_projector_probe_fallback_is_available_for_t_and_p_only(self):
        integrity = {"projector_updated": False, "diffusion_updated": True,
                     "frozen_backbone_unchanged": True, "optimizer_membership_valid": True}
        self.assertTrue(CAMPAIGN.needs_projector_reconciliation("T", integrity))
        self.assertTrue(CAMPAIGN.needs_projector_reconciliation("P", integrity))
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
