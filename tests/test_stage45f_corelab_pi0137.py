from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import stage45f_corelab_evidence_convergence as convergence
import stage45f_corelab_pi0137 as pi0137


class CorelabPI0137Test(unittest.TestCase):
    def _baseline(self, root: Path) -> tuple[Path, Path]:
        evidence = pi0137._build_evidence_index(root)
        return pi0137._build_contract_campaign(root, evidence)

    def _reseal(self, campaign: Path, identity: Path, payload: dict, *, keep_identity: bool = False) -> None:
        campaign.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")
        if not keep_identity:
            identity_payload = json.loads(identity.read_text(encoding="utf-8"))
            identity_payload["campaign_file_sha256"] = convergence.sha256_file(campaign)
            identity.write_text(json.dumps(identity_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _run_mutation(self, mutate, expected: str, *, keep_identity: bool = False) -> None:
        with tempfile.TemporaryDirectory(prefix="stage45f-pi0137-mutation-") as directory:
            root = Path(directory)
            campaign, identity = self._baseline(root)
            payload = json.loads(campaign.read_text(encoding="utf-8"))
            mutate(payload, root)
            self._reseal(campaign, identity, payload, keep_identity=keep_identity)
            with self.assertRaises(pi0137.PI0137Failure) as caught:
                pi0137.validate_synthetic_campaign(campaign, identity)
            self.assertEqual(caught.exception.reason_code, expected)

    def test_c11_exact_17_mutation_classes_have_machine_reason_codes(self) -> None:
        def inventory_drift(payload: dict, root: Path) -> None:
            path = Path(payload["inventory"])
            value = json.loads(path.read_text(encoding="utf-8"))
            value["entries"][0]["sha256"] = "0" * 64
            path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
            payload["inventory_sha256"] = convergence.sha256_file(path)

        def outside_summary(payload: dict, root: Path) -> None:
            path = root.parent / f"{root.name}-outside-summary.json"
            path.write_text("{}\n", encoding="utf-8")
            payload["summary"] = str(path)
            payload["summary_sha256"] = convergence.sha256_file(path)

        def symlink_rows(payload: dict, root: Path) -> None:
            target = Path(payload["arms"][0]["rows_file"])
            alias = root / "rows-alias.json"
            alias.symlink_to(target)
            payload["arms"][0]["rows_file"] = str(alias)

        def mutate_rows(payload: dict, root: Path, operation) -> None:
            arm = payload["arms"][0]
            path = Path(arm["rows_file"])
            rows = json.loads(path.read_text(encoding="utf-8"))
            operation(rows)
            path.write_text(json.dumps(rows, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")
            arm["rows_sha256"] = convergence.sha256_file(path)

        cases = [
            (lambda p, r: p["arms"].pop(), "MISSING_REQUIRED_ARM", False),
            (lambda p, r: p["arms"].append(copy.deepcopy(p["arms"][0])), "DUPLICATE_ARM_IDENTITY", False),
            (lambda p, r: p["arms"][0].__setitem__("seed", 9999), "ARM_AXIS_MISMATCH", False),
            (lambda p, r: p["arms"][0].__setitem__("checkpoint_tree_sha256", "0" * 64), "CHECKPOINT_TREE_SHA_MISMATCH", False),
            (lambda p, r: p["arms"][0].__setitem__("rows_sha256", "0" * 64), "ROW_BUNDLE_SHA_MISMATCH", False),
            (lambda p, r: p.__setitem__("summary_sha256", "0" * 64), "UPSTREAM_ARTIFACT_SHA_MISMATCH", False),
            (inventory_drift, "INVENTORY_IMMUTABILITY_VIOLATION", False),
            (outside_summary, "CAMPAIGN_ROOT_ESCAPE", False),
            (symlink_rows, "UNSAFE_PATH_ESCAPE", False),
            (lambda p, r: p.__setitem__("execution_head", "f" * 40), "EXECUTION_LINEAGE_IDENTITY_MISMATCH", False),
            (lambda p, r: p["arms"][0].__setitem__("stats_sha256", "f" * 64), "ROUTE_SCIENTIFIC_CONTRACT_DRIFT", False),
            (lambda p, r: p["phase_budgets"]["E4"]["LEGACY"].__setitem__("action_head_calls", 999), "CALL_COUNTER_BUDGET_MISMATCH", False),
            (lambda p, r: mutate_rows(p, r, lambda rows: rows.pop()), "TRAIN24_ROW_CONTRACT_DRIFT", False),
            (lambda p, r: mutate_rows(p, r, lambda rows: rows[0].__setitem__("synthetic_metric", math.nan)), "NONFINITE_EVIDENCE", False),
            (lambda p, r: p.__setitem__("validation_e_marker", "Validation-E"), "LOCKED_SPLIT_CONTAMINATION", False),
            (lambda p, r: p["arms"][0]["lineage"].__setitem__("performance_selected_retry", True), "PERFORMANCE_SELECTED_LINEAGE_REPLACEMENT", False),
            (lambda p, r: p.__setitem__("schema", "UNKNOWN_V99"), "UNSUPPORTED_SCHEMA_NO_HEURISTICS", False),
        ]
        for mutate, expected, keep_identity in cases:
            with self.subTest(reason_code=expected):
                self._run_mutation(mutate, expected, keep_identity=keep_identity)

    def test_same_campaign_identity_changed_bytes_is_rejected(self) -> None:
        self._run_mutation(lambda payload, root: payload.__setitem__("synthetic_note", "changed"), "CAMPAIGN_IDENTITY_BYTE_DRIFT", keep_identity=True)

    def test_static_shadow_and_full_synthetic_rehearsal(self) -> None:
        self.assertEqual(pi0137.static_audit()["mutation_case_count"], 17)
        self.assertFalse(pi0137.shadow_audit()["field_guessing"])
        with tempfile.TemporaryDirectory(prefix="stage45f-pi0137-parent-") as directory:
            output = Path(directory) / "rehearsal"
            result = pi0137.run_synthetic_rehearsal(output)
            self.assertEqual(result["token"], pi0137.SYNTHETIC_TOKEN)
            self.assertNotIn(pi0137.REAL_CORELAB_TOKEN, json.dumps(result))
            self.assertEqual(result["readiness_counts"]["READY_SEALED"], 18)
            self.assertTrue(result["idempotent_reingestion"])
            self.assertTrue(result["candidate_order_invariant"])
            self.assertFalse(result["real_h200_campaign_ingested"])

    def _selection_surface(self, root: Path) -> tuple[Path, Path, dict]:
        output = root / "rehearsal"
        pi0137.run_synthetic_rehearsal(output)
        packet_path = output / "train_packet_1.json"
        packet = json.loads(packet_path.read_text(encoding="utf-8"))
        selected = {"N2(seed2222)@2000": packet["arm_summaries"]["N2(seed2222)@2000"]["checkpoint_tree_sha256"]}
        selection = {"schema": "STAGE45F_PI_SELECTION_ARTIFACT_V1", "status": "PI_SELECTED", "selection_generation": 7, "source_decision_packet_sha256": convergence.sha256_file(packet_path), "selected_route": "N2", "selected_checkpoint_tier_or_exact_checkpoint_set": "EXACT_SET", "selected_checkpoint_set": selected, "selection_scope": "SYNTHETIC_UNIT_TEST_ONLY", "validation_generation": 1, "pi_authority_comment": 123456, "pi_authority_token": "SYNTHETIC_PI_SELECTION_TOKEN"}
        selection_path = root / "selection.json"
        selection_path.write_text(json.dumps(selection, sort_keys=True) + "\n", encoding="utf-8")
        return selection_path, packet_path, selection

    def test_c14_requires_explicit_pi_selection_and_exact_checkpoint_set(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stage45f-pi0137-selection-") as directory:
            root = Path(directory)
            selection_path, packet_path, selection = self._selection_surface(root)
            handoff = pi0137.generate_post_selection(selection_path, packet_path)
            self.assertFalse(handoff["winner_inferred"])
            self.assertFalse(handoff["validation_execution_enabled"])
            selection.pop("pi_authority_token")
            selection_path.write_text(json.dumps(selection, sort_keys=True) + "\n", encoding="utf-8")
            with self.assertRaises(pi0137.PI0137Failure) as caught:
                pi0137.generate_post_selection(selection_path, packet_path)
            self.assertEqual(caught.exception.reason_code, "PI_SELECTION_AUTHORITY_MISSING")

    def test_c15_one_shot_validation_guard_and_locked_alias(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stage45f-pi0137-validation-") as directory:
            root = Path(directory)
            selection_path, packet_path, _ = self._selection_surface(root)
            handoff_path = root / "handoff.json"
            handoff = pi0137.generate_post_selection(selection_path, packet_path, handoff_path)
            ledger = {"schema": "STAGE45F_VALIDATION_E_ACCESS_LEDGER_V1", **handoff["validation_access_ledger"], "completed_access_records": 0, "holdout_access_count": 0, "holdout_requested": False}
            ledger_path = root / "ledger.json"
            ledger_path.write_text(json.dumps(ledger, sort_keys=True) + "\n", encoding="utf-8")
            self.assertFalse(pi0137.validation_one_shot_guard(handoff_path, ledger_path)["accessed"])
            ledger["accessed"] = True
            ledger["completed_access_records"] = 1
            ledger_path.write_text(json.dumps(ledger, sort_keys=True) + "\n", encoding="utf-8")
            with self.assertRaises(pi0137.PI0137Failure) as caught:
                pi0137.validation_one_shot_guard(handoff_path, ledger_path)
            self.assertEqual(caught.exception.reason_code, "VALIDATION_GENERATION_ALREADY_ACCESSED")
            ledger["accessed"] = False
            ledger["completed_access_records"] = 0
            ledger["candidate_set_sha256"] = "f" * 64
            ledger_path.write_text(json.dumps(ledger, sort_keys=True) + "\n", encoding="utf-8")
            with self.assertRaises(pi0137.PI0137Failure) as caught:
                pi0137.validation_one_shot_guard(handoff_path, ledger_path)
            self.assertEqual(caught.exception.reason_code, "VALIDATION_SELECTION_GENERATION_DRIFT")
            ledger["candidate_set_sha256"] = handoff["validation_access_ledger"]["candidate_set_sha256"]
            ledger["holdout_requested"] = True
            ledger_path.write_text(json.dumps(ledger, sort_keys=True) + "\n", encoding="utf-8")
            with self.assertRaises(pi0137.PI0137Failure) as caught:
                pi0137.validation_one_shot_guard(handoff_path, ledger_path)
            self.assertEqual(caught.exception.reason_code, "HOLDOUT_FORBIDDEN")
            locked = root / "Validation-E"
            locked.mkdir()
            (locked / "never-open.json").write_text("{}\n", encoding="utf-8")
            alias = root / "alias.json"
            alias.symlink_to(locked / "never-open.json")
            with self.assertRaises(pi0137.PI0137Failure) as caught:
                pi0137.guarded_path(root, alias)
            self.assertIn(caught.exception.reason_code, {"UNSAFE_PATH_ESCAPE", "LOCKED_SPLIT_CONTAMINATION"})

    def test_c16_capsule_and_c17_branch_handoffs_remain_reference_only(self) -> None:
        template = json.loads(pi0137.CAPSULE_TEMPLATE.read_text(encoding="utf-8"))
        bindings = {name: {"sha256": "0" * 64, "reference_only": True} for name in template["required_reference_groups"]}
        capsule = pi0137.build_capsule(bindings)
        self.assertFalse(capsule["copy_large_checkpoint_bytes"])
        self.assertFalse(capsule["delete_or_rewrite_accepted_evidence"])
        handoff = {"schema": "STAGE45F_POST_SELECTION_RECIPE_FREEZE_HANDOFF_V1", "synthetic": True}
        branches = pi0137.build_branch_handoffs(handoff)
        self.assertEqual(len(branches["branches"]), 6)
        self.assertTrue(all(not item["execution_enabled"] for item in branches["branches"].values()))


if __name__ == "__main__":
    unittest.main()
