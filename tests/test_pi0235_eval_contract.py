"""CPU-only regression coverage for the PI0235 evaluator contract."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("pi0236_eval", ROOT / "scripts" / "pi0232_train_calib_eval.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def expected_rows(count: int = 4) -> list[dict[str, object]]:
    return [
        {
            "episode_index": index,
            "source_episode_id": f"rollout-{index // 2}:obs-{index:04d}",
            "source_rollout_id": f"rollout-{index // 2}",
            "target_id": f"target-{index // 2}",
        }
        for index in range(count)
    ]


def profile(step: int, decisions: list[str], *, rows: list[dict[str, object]] | None = None) -> dict[str, object]:
    rows = rows or expected_rows(len(decisions))
    records = []
    for index, (row, decision) in enumerate(zip(rows, decisions)):
        records.append({
            "episode_index": index,
            "source_episode_id": row["source_episode_id"],
            "source_rollout_id": row["source_rollout_id"],
            "target_id": row["target_id"],
            "target_xyz_in_model_input": False,
            "wbc_execution": False,
            "raw_sha256": "a" * 64,
            "decoded_sha256": "b" * 64,
            "reference_min_target_distance_m": 0.1 + index / 100,
            "safety_gate": {"decision": decision},
        })
    counts = {decision: decisions.count(decision) for decision in MODULE.DECISIONS}
    accepted = [record["reference_min_target_distance_m"] for record in records if record["safety_gate"]["decision"] != "REJECT"]
    cells: dict[str, list[float]] = {}
    for record in records:
        if record["safety_gate"]["decision"] != "REJECT":
            cells.setdefault(str(record["source_rollout_id"]), []).append(float(record["reference_min_target_distance_m"]))
    median = MODULE._median(accepted) if accepted else None
    return {
        "schema": MODULE.PROFILE_SCHEMA,
        "status": "PASS" if counts["REJECT"] == 0 else "SAFETY_GATE_REJECT",
        "checkpoint_step": step,
        "seed": 22301,
        "action_contract": "official_dense_relative_H40",
        "action_dt_s": 0.05,
        "model_input_keys": MODULE.MODEL_INPUT_KEYS,
        "target_xyz_in_model_input": False,
        "wbc_execution": False,
        "forward_count": len(records),
        "expected_forward_count": len(records),
        "safety_gate_counts": counts,
        "reference_min_target_distance_m": {"count": len(accepted), "median": median, "max": max(accepted) if accepted else None},
        "worst_cell_median_reference_min_target_distance_m": max((MODULE._median(values) for values in cells.values()), default=None),
        "cell_medians_reference_min_target_distance_m": {key: MODULE._median(values) for key, values in cells.items()},
        "records": records,
    }


class PI0235ContractTests(unittest.TestCase):
    def test_baseline_reject_is_complete_and_not_candidate(self) -> None:
        result = MODULE.validate_profile_payload(profile(1000, ["REJECT"]), step=1000, expected_rows=expected_rows(1))
        self.assertEqual(result["scientific_profile_status"], "SAFETY_GATE_REJECT")
        self.assertFalse(result["profile_is_admissible"])

    def test_reject_does_not_block_next_profile(self) -> None:
        MODULE.validate_profile_payload(profile(2500, ["REJECT"]), step=2500, expected_rows=expected_rows(1))
        result = MODULE.validate_profile_payload(profile(5000, ["WARN"]), step=5000, expected_rows=expected_rows(1))
        self.assertTrue(result["profile_is_admissible"])

    def test_warn_only_remains_eligible(self) -> None:
        result = MODULE.validate_profile_payload(profile(2500, ["WARN", "ACCEPT"]), step=2500, expected_rows=expected_rows(2))
        self.assertTrue(result["profile_is_admissible"])
        self.assertEqual(result["safety_gate_counts"]["WARN"], 1)

    def test_all_reject_preserves_null_ranking_statistics(self) -> None:
        payload = profile(2500, ["REJECT", "REJECT"])
        result = MODULE.validate_profile_payload(payload, step=2500, expected_rows=expected_rows(2))
        self.assertIsNone(payload["reference_min_target_distance_m"]["median"])
        self.assertIsNone(payload["worst_cell_median_reference_min_target_distance_m"])
        self.assertFalse(result["profile_is_admissible"])

    def test_missing_row_fails_closed(self) -> None:
        with self.assertRaises(MODULE.EvaluationContractError):
            MODULE.validate_profile_payload(profile(2500, ["ACCEPT"]), step=2500, expected_rows=expected_rows(2))

    def test_duplicate_or_out_of_order_rows_fail_closed(self) -> None:
        rows = expected_rows(2)
        payload = profile(2500, ["ACCEPT", "ACCEPT"], rows=rows)
        payload["records"][1]["episode_index"] = 0
        with self.assertRaises(MODULE.EvaluationContractError):
            MODULE.validate_profile_payload(payload, step=2500, expected_rows=rows)

    def test_unknown_gate_decision_fails_closed(self) -> None:
        payload = profile(2500, ["ACCEPT"])
        payload["records"][0]["safety_gate"]["decision"] = "UNKNOWN"
        with self.assertRaises(MODULE.EvaluationContractError):
            MODULE.validate_profile_payload(payload, step=2500, expected_rows=expected_rows(1))

    def test_hash_and_boundary_drift_fail_closed(self) -> None:
        payload = profile(2500, ["ACCEPT"])
        payload["records"][0]["decoded_sha256"] = "bad"
        with self.assertRaises(MODULE.EvaluationContractError):
            MODULE.validate_profile_payload(payload, step=2500, expected_rows=expected_rows(1))
        payload = profile(2500, ["ACCEPT"])
        payload["target_xyz_in_model_input"] = True
        with self.assertRaises(MODULE.EvaluationContractError):
            MODULE.validate_profile_payload(payload, step=2500, expected_rows=expected_rows(1))

    def test_selector_predicate_excludes_reject_and_accepts_warn(self) -> None:
        self.assertFalse(MODULE.profile_is_admissible(profile(2500, ["REJECT"])))
        self.assertTrue(MODULE.profile_is_admissible(profile(2500, ["WARN"])))

    def test_fixed_order_and_roles_are_unchanged(self) -> None:
        self.assertEqual(MODULE.STEPS, (1000, 2500, 5000, 7500, 10000))
        self.assertEqual(MODULE._profile_role(1000), "BASELINE_COMPARISON_ONLY")
        self.assertEqual(MODULE._profile_role(2500), "FRESH_SELECTION_CANDIDATE")

    def test_failure_manifest_is_distinct_from_scientific_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / "state.json"
            MODULE._failure_manifest(root, step=2500, error=RuntimeError("synthetic"), state_path=state)
            payload = json.loads((root / "pi0236_evaluation_failure.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "ENGINEERING_BLOCKED")
            self.assertEqual(payload["known_forward_count"], "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
