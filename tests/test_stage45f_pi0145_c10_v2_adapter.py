from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import stage45f_corelab_evidence_convergence as convergence
import stage45f_corelab_pi0145_v2_adapter as adapter


class PI0145VandaV2AdapterTest(unittest.TestCase):
    @staticmethod
    def _accounting(attempts: list[dict[str, object]], *, accepted: int = 8976, attempted: int | None = None, failed: int | None = None) -> dict[str, object]:
        if attempted is None:
            attempted = sum(int(item["counts"]["model_side_calls"]) for item in attempts)
        if failed is None:
            failed = sum(int(item["counts"]["model_side_calls"]) for item in attempts if item["status"] == "FAILED_ATTEMPT")
        return {
            "accepted_unique_scientific_calls": accepted,
            "all_attempted_model_calls": attempted,
            "failed_attempt_model_calls": failed,
            "reused_readonly_model_calls": 0,
            "diagnostic_attempt_inventory": attempts,
        }

    def test_attempt_identity_allows_failed_then_accepted_retry_for_same_unit(self) -> None:
        failed_root = hashlib.sha256(b"failed-attempt").hexdigest()
        accepted_root = hashlib.sha256(b"accepted-attempt").hexdigest()
        document = self._accounting([
            {"unit_key": "D1:legacy", "attempt_id": "job-1", "attempt_root_sha256": failed_root, "status": "FAILED_ATTEMPT", "counts": {"model_side_calls": 7, "counters_complete": True}},
            {"unit_key": "D1:legacy", "attempt_id": "job-2", "attempt_root_sha256": accepted_root, "status": "PASS_ACCEPTED", "counts": {"model_side_calls": 8976, "counters_complete": True}},
        ])
        result = adapter.validate_diagnostic_accounting(document)
        self.assertEqual(result["accepted_attempts"], 1)
        self.assertEqual(result["attempted_units"], 1)
        self.assertEqual(result["inventory_entries"], 2)

    def test_attempt_identity_rejects_duplicate_root_even_when_renamed(self) -> None:
        root = hashlib.sha256(b"same-immutable-root").hexdigest()
        document = self._accounting([
            {"unit_key": "E4:legacy", "attempt_id": "job-1", "attempt_root_sha256": root, "status": "FAILED_ATTEMPT", "counts": {"model_side_calls": 7, "counters_complete": True}},
            {"unit_key": "renamed-unit", "attempt_id": "renamed-job", "attempt_root_sha256": root, "status": "BLOCKED", "counts": {"model_side_calls": 0, "counters_complete": True}},
        ], accepted=8976, attempted=7, failed=7)
        with self.assertRaises(adapter.V2AdapterError) as caught:
            adapter.validate_diagnostic_accounting(document)
        self.assertIn("ATTEMPT_IDENTITY_DUPLICATE", str(caught.exception))

    def test_attempt_identity_rejects_two_accepted_passes_for_one_unit(self) -> None:
        document = self._accounting([
            {"unit_key": "D2A:n2", "attempt_id": "job-1", "attempt_root_sha256": "a" * 64, "status": "PASS_ACCEPTED", "counts": {"model_side_calls": 8976, "counters_complete": True}},
            {"unit_key": "D2A:n2", "attempt_id": "job-2", "attempt_root_sha256": "b" * 64, "status": "PASS_ACCEPTED", "counts": {"model_side_calls": 0, "counters_complete": True}},
        ], attempted=8976)
        with self.assertRaises(adapter.V2AdapterError) as caught:
            adapter.validate_diagnostic_accounting(document)
        self.assertIn("UNIT_MULTIPLE_ACCEPTED_PASS", str(caught.exception))

    def test_attempt_identity_fields_are_required(self) -> None:
        document = self._accounting([
            {"unit_key": "E4:l1", "status": "PASS_ACCEPTED", "counts": {"model_side_calls": 8976, "counters_complete": True}},
        ])
        with self.assertRaises(adapter.V2AdapterError) as caught:
            adapter.validate_diagnostic_accounting(document)
        self.assertIn("ATTEMPT_ID_MISSING", str(caught.exception))

    def test_attempt_accounting_rejects_unknown_or_missing_counts(self) -> None:
        root = hashlib.sha256(b"unknown-counts").hexdigest()
        unknown = self._accounting([
            {"unit_key": "E4:legacy", "attempt_id": "job-unknown", "attempt_root_sha256": root, "status": "UNKNOWN", "counts": {"model_side_calls": 0, "counters_complete": False}},
        ], attempted=0, failed=0)
        with self.assertRaises(adapter.V2AdapterError) as caught:
            adapter.validate_diagnostic_accounting(unknown)
        self.assertIn("counters unknown", str(caught.exception))

        missing = self._accounting([
            {"unit_key": "E4:legacy", "attempt_id": "job-missing", "attempt_root_sha256": root, "status": "BLOCKED", "counts": {"counters_complete": True}},
        ], attempted=0, failed=0)
        with self.assertRaises(adapter.V2AdapterError) as caught:
            adapter.validate_diagnostic_accounting(missing)
        self.assertIn("model_side_calls invalid", str(caught.exception))

    def test_attempt_accounting_rejects_each_counter_reconciliation_drift(self) -> None:
        root = hashlib.sha256(b"reconciliation").hexdigest()
        base = self._accounting([
            {"unit_key": "D1:n2", "attempt_id": "job-failed", "attempt_root_sha256": root, "status": "FAILED_ATTEMPT", "counts": {"model_side_calls": 7, "counters_complete": True}},
            {"unit_key": "D1:n2", "attempt_id": "job-pass", "attempt_root_sha256": hashlib.sha256(b"reconciliation-pass").hexdigest(), "status": "PASS_ACCEPTED", "counts": {"model_side_calls": 8976, "counters_complete": True}},
        ])
        for field, value in (
            ("accepted_unique_scientific_calls", 8975),
            ("all_attempted_model_calls", 8982),
            ("failed_attempt_model_calls", 6),
        ):
            broken = copy.deepcopy(base)
            broken[field] = value
            with self.subTest(field=field):
                with self.assertRaises(adapter.V2AdapterError):
                    adapter.validate_diagnostic_accounting(broken)

    def test_vanda_producer_fixture_reaches_synthetic_c10_consumer_and_real_entry_rejects_it(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stage45f-pi0145-v2-") as directory:
            final_path = adapter.build_synthetic_vanda_v2_fixture(Path(directory) / "fixture")
            result = adapter.verify_synthetic_v2_fixture(final_path)
            self.assertEqual(result["status"], "PASS_SYNTHETIC_ONLY")
            self.assertEqual(result["checked_arms"], 18)
            self.assertEqual(result["checked_phase_seals"], 9)
            self.assertFalse(result["real_h200_campaign_ingested"])
            with self.assertRaises(adapter.V2AdapterError):
                # This is the actual public entrypoint.  The synthetic marker
                # is deliberately rejected before it can be treated as real
                # H200 campaign evidence.
                convergence.verify_campaign(final_path)
            with self.assertRaises(adapter.V2AdapterError):
                # The neutral packet path performs the same admission before
                # it opens an evidence index, so synthetic evidence cannot be
                # smuggled into a decision packet either.
                convergence.build_packet(Path(directory) / "missing-index.json", Path(directory) / "packet.json", final_path)

    def test_v2_typed_history_and_integer_zero_counters_are_explicit(self) -> None:
        good = {
            "campaign_history": {"training_occurred": True},
            "cpu_closure_execution": {
            "gpu": 0, "pbs": 0, "training": 0, "model_construction": 0,
            "model_forward": 0, "Validation-E": 0, "Holdout-F": 0,
                "WBC": 0, "simulator": 0, "hardware": 0,
                "backward": 0, "optimizer_steps": 0, "public_sampler_calls": 0,
                "get_action_calls": 0, "merge": False,
                "model_imported": False,
            },
            "merge": False,
        }
        mapped = adapter.typed_boundary_mapping(good)
        self.assertFalse(mapped["training"])
        self.assertTrue(mapped["history_training"])
        for field in ("training_occurred", "gpu", "training"):
            broken = copy.deepcopy(good)
            if field == "training_occurred":
                broken["campaign_history"][field] = 1
            else:
                broken["cpu_closure_execution"][field] = False
            with self.subTest(field=field):
                with self.assertRaises(adapter.V2AdapterError):
                    adapter.typed_boundary_mapping(broken)
        missing = copy.deepcopy(good)
        del missing["cpu_closure_execution"]["pbs"]
        with self.assertRaises(adapter.V2AdapterError):
            adapter.typed_boundary_mapping(missing)

        # PI0146 failure surface: typed forbidden counters/flags must not be
        # silently accepted when a producer reports a nonzero value or the
        # wrong JSON type, including the nested merge flag.
        for field, value in (
            ("model_imported", True),
            ("backward", 7),
            ("optimizer_steps", 7),
            ("public_sampler_calls", 7),
        ):
            broken = copy.deepcopy(good)
            broken["cpu_closure_execution"][field] = value
            with self.subTest(field=field):
                with self.assertRaises(adapter.V2AdapterError):
                    adapter.typed_boundary_mapping(broken)
        nested_merge = copy.deepcopy(good)
        nested_merge["cpu_closure_execution"]["merge"] = True
        with self.assertRaises(adapter.V2AdapterError):
            adapter.typed_boundary_mapping(nested_merge)

    def test_handoff_requires_exact_historical_mapping_and_rejects_duplicate_arm(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stage45f-pi0145-mutation-") as directory:
            final_path = adapter.build_synthetic_vanda_v2_fixture(Path(directory) / "fixture")
            result_path = Path(final_path)
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            handoff_path = Path(payload["c10_handoff"])
            handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
            handoff["arms"][1]["label"] = handoff["arms"][0]["label"]
            handoff_path.write_text(json.dumps(handoff, sort_keys=True) + "\n", encoding="utf-8")
            payload["c10_handoff_sha256"] = convergence.sha256_file(handoff_path)
            result_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
            with self.assertRaises(adapter.V2AdapterError) as caught:
                adapter.verify_synthetic_v2_fixture(result_path)
            self.assertIn("arm identity", str(caught.exception))

    def test_descriptor_mutation_and_window_contract_are_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stage45f-pi0145-descriptor-") as directory:
            final_path = adapter.build_synthetic_vanda_v2_fixture(Path(directory) / "fixture")
            payload = json.loads(Path(final_path).read_text(encoding="utf-8"))
            handoff = json.loads(Path(payload["c10_handoff"]).read_text(encoding="utf-8"))
            current_ref = next(
                arm["diagnostic_evidence"]["E4"]
                for arm in handoff["arms"]
                if arm["diagnostic_evidence"]["E4"].get("source") == "CURRENT_PHASE_UNIT"
            )
            descriptor_path = Path(current_ref["descriptor_path"])
            descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
            descriptor["train24_contract"]["target_xyz_in_model_input"] = True
            descriptor_path.write_text(json.dumps(descriptor, sort_keys=True) + "\n", encoding="utf-8")
            inventory_path = Path(handoff["external_inventory"])
            inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
            unit = "E4:legacy"
            inventory["descriptors"][unit]["sha256"] = convergence.sha256_file(descriptor_path)
            inventory_path.write_text(json.dumps(inventory, sort_keys=True) + "\n", encoding="utf-8")
            # Rebind the arm edge as a producer would have to do when
            # resealing a descriptor.  The verifier must then reach the
            # mutated target-XYZ field rather than stopping at an earlier
            # stale edge mismatch.
            for arm in handoff["arms"]:
                ref = arm["diagnostic_evidence"]["E4"]
                if ref.get("source") == "CURRENT_PHASE_UNIT" and ref.get("descriptor_path") == str(descriptor_path):
                    ref["descriptor_sha256"] = convergence.sha256_file(descriptor_path)
            # The producer's CPU inventory includes the descriptor bytes.  A
            # properly resealed descriptor therefore requires a resealed
            # inventory and final-result pointer; otherwise the verifier must
            # (correctly) stop at immutable inventory drift first.
            cpu_inventory_path = Path(payload["cpu_inventory"])
            cpu_inventory = json.loads(cpu_inventory_path.read_text(encoding="utf-8"))
            for entry in cpu_inventory["entries"]:
                if entry["path"] == str(descriptor_path.relative_to(Path(payload["campaign_root"]))):
                    entry["sha256"] = convergence.sha256_file(descriptor_path)
                    entry["bytes"] = descriptor_path.stat().st_size
                if entry["path"] == str(inventory_path.relative_to(Path(payload["campaign_root"]))):
                    entry["sha256"] = convergence.sha256_file(inventory_path)
                    entry["bytes"] = inventory_path.stat().st_size
            cpu_inventory_path.write_text(json.dumps(cpu_inventory, sort_keys=True) + "\n", encoding="utf-8")
            payload["cpu_inventory_sha256"] = convergence.sha256_file(cpu_inventory_path)
            handoff["external_inventory_sha256"] = convergence.sha256_file(inventory_path)
            handoff_path = Path(payload["c10_handoff"])
            handoff_path.write_text(json.dumps(handoff, sort_keys=True) + "\n", encoding="utf-8")
            payload["c10_handoff_sha256"] = convergence.sha256_file(handoff_path)
            Path(final_path).write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
            with self.assertRaises(adapter.V2AdapterError) as caught:
                adapter.verify_synthetic_v2_fixture(final_path)
            self.assertIn("target XYZ", str(caught.exception))

    def test_source_contract_and_no_model_imports(self) -> None:
        contract = json.loads((ROOT / "manifests/STAGE45F_PI0145_VANDA_C10_PRODUCER_CONSUMER_V1.json").read_text(encoding="utf-8"))
        self.assertEqual(contract["producer"]["head"], adapter.VANDA_PRODUCER_HEAD)
        self.assertEqual(contract["consumer"]["entrypoint"], "scripts/stage45f_corelab_evidence_convergence.py::verify_campaign")
        self.assertFalse(contract["synthetic_policy"]["real_verify_campaign_accepts"])
        attempt_contract = contract["c10_contract"]["diagnostic_attempt_identity"]
        self.assertEqual(attempt_contract["deduplication_key"], "attempt_root_sha256")
        self.assertTrue(attempt_contract["failed_and_accepted_attempts_may_share_unit"])
        self.assertEqual(attempt_contract["max_accepted_passes_per_unit"], 1)
        source = (ROOT / "scripts/stage45f_corelab_pi0145_v2_adapter.py").read_text(encoding="utf-8")
        for forbidden in ("import torch", "import numpy", "import gr00t", "import subprocess", "from torch", "from numpy"):
            self.assertNotIn(forbidden, source)

    def test_nested_synthetic_markers_are_rejected(self) -> None:
        with self.assertRaises(adapter.V2AdapterError) as caught:
            adapter.require_no_synthetic_markers({"summary": {"phase_seals": [{"synthetic_test_only": "SYNTHETIC_TEST_ONLY"}]}})
        self.assertIn("synthetic evidence marker", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
