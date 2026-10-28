"""Deterministic CPU-only tests for the Stage45C late-binding candidate."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import stage45c_verify_late_binding_candidate as verifier  # noqa: E402


MANIFEST_PATH = ROOT / "manifests" / "STAGE45C_TRAIN24_LATE_BINDING_CANDIDATE_V1.json"
PREP_MANIFEST_PATH = ROOT / "manifests" / "STAGE45C_TRAIN24_EVALUATION_PROTOCOL_V1.json"


class TestStage45CLateBindingCandidate(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    def test_candidate_is_valid_and_not_finally_frozen(self) -> None:
        summary = verifier.validate_candidate(copy.deepcopy(self.manifest))
        self.assertEqual(summary["status"], "LATE_BINDING_CANDIDATE_NOT_FROZEN")
        self.assertEqual(summary["cohort_window_count"], 24)
        self.assertEqual(summary["checkpoint_count"], 4)
        self.assertEqual(summary["planned_forwards"], 96)
        self.assertEqual(summary["gr00t_forwards_observed"], 0)
        self.assertEqual(summary["historical_materialization"], "PENDING_VANDA_RECOVERY")
        self.assertFalse(summary["final_protocol_freeze_issued"])

    def test_authority_and_order_tampering_fails_closed(self) -> None:
        tampered = copy.deepcopy(self.manifest)
        tampered["accepted_authorities"]["full24_checkpoint_1000"]["tree_sha256"] = "0" * 64
        with self.assertRaises(verifier.CandidateError):
            verifier.validate_candidate(tampered)

        tampered = copy.deepcopy(self.manifest)
        tampered["full24_train24"]["window_ids"][1], tampered["full24_train24"]["window_ids"][2] = (
            tampered["full24_train24"]["window_ids"][2],
            tampered["full24_train24"]["window_ids"][1],
        )
        with self.assertRaises(verifier.CandidateError):
            verifier.validate_candidate(tampered)

    def test_historical_locator_and_execution_boundary_are_fail_closed(self) -> None:
        tampered = copy.deepcopy(self.manifest)
        tampered["historical_comparator"]["materialization_path"] = "/scratch/forbidden"
        with self.assertRaises(verifier.CandidateError):
            verifier.validate_candidate(tampered)

        tampered = copy.deepcopy(self.manifest)
        tampered["execution_boundary"]["gr00t_forwards"] = 1
        with self.assertRaises(verifier.CandidateError):
            verifier.validate_candidate(tampered)

        tampered = copy.deepcopy(self.manifest)
        tampered["integration"]["final_protocol_freeze_issued"] = True
        with self.assertRaises(verifier.CandidateError):
            verifier.validate_candidate(tampered)

    def test_candidate_has_no_node_local_paths_and_keeps_old_prep_untouched(self) -> None:
        text = MANIFEST_PATH.read_text(encoding="utf-8").lower()
        self.assertNotIn("/scratch/", text)
        self.assertNotIn("/var/cache/", text)
        self.assertNotIn("/home/", text)
        old = json.loads(PREP_MANIFEST_PATH.read_text(encoding="utf-8"))
        self.assertEqual(old["status"], "PREPARED_NOT_EXECUTED")
        self.assertEqual(old["pi_gate"], "PI_GATE_STATE_0088")
        self.assertNotEqual(old["schema"], self.manifest["schema"])

    def test_verifier_is_static_and_model_free(self) -> None:
        source = (ROOT / "scripts" / "stage45c_verify_late_binding_candidate.py").read_text(encoding="utf-8")
        self.assertNotIn("import torch", source)
        self.assertNotIn("import transformers", source)
        self.assertNotIn("model.forward", source)
        self.assertNotIn("subprocess", source)


if __name__ == "__main__":
    unittest.main()
