from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import stage45f_icl_context_v1 as context_v1


class Stage45FICLContextV1Test(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = context_v1.synthetic_context_fixture()

    def assert_rejected(self, value: dict, reason: str) -> None:
        with self.assertRaises(context_v1.ContextV1Error) as caught:
            context_v1.validate_context(value)
        self.assertEqual(caught.exception.reason_code, reason)
        self.assertIn(reason, str(caught.exception))

    def test_contract_and_source_audit_are_read_only(self) -> None:
        result = context_v1.static_audit()
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["upstream_commit"], context_v1.UPSTREAM_COMMIT)
        self.assertEqual(result["context_variant_count"], 9)
        self.assertEqual(result["integration_option_count"], 5)
        self.assertEqual(result["experiment_count"], 5)
        self.assertEqual(result["i0_existing_interface_variant_count"], 4)
        self.assertEqual(result["forbidden_top_level_imports"], [])
        self.assertTrue(all(value == 0 for value in result["execution_boundary"].values()))

    def test_all_four_sections_and_explicit_null_are_valid(self) -> None:
        result = context_v1.validate_context(self.fixture)
        self.assertEqual(result["status"], "PASS_CPU_ONLY")
        self.assertEqual(result["section_counts"], {section: 1 for section in context_v1.SECTIONS})
        self.assertTrue(result["pathless"])
        self.assertEqual(result["structured_action_abi"]["shape"], [40, 132])
        self.assertEqual(result["structured_action_abi"]["active_scalars"], 190)
        self.assertIsNone(self.fixture["interaction_history"][0]["executed_or_observed_response_digest"])

    def test_builder_and_index_are_order_invariant(self) -> None:
        reversed_record = copy.deepcopy(self.fixture)
        for section in context_v1.SECTIONS:
            reversed_record[section] = list(reversed(reversed_record[section]))
        left = context_v1.build_context_index([self.fixture])
        right = context_v1.build_context_index([reversed_record])
        self.assertEqual(left, right)
        self.assertTrue(left["pathless"])
        self.assertFalse(left["target_xyz_in_model_input"])
        self.assertEqual(left["record_count"], 1)

    def test_tensorization_plan_excludes_metadata_and_keeps_abi(self) -> None:
        result = context_v1.build_tensorization_plan(self.fixture)
        self.assertEqual(result["status"], "PLAN_ONLY_NO_TENSOR_MATERIALIZATION")
        self.assertFalse(result["target_id_forwarded"])
        self.assertFalse(result["context_metadata_forwarded"])
        self.assertNotIn("target_id", result["model_visible_fields"])
        self.assertIn("local_dynamics_context.state_after_payload_resolved", result["model_visible_fields"])
        plan = json.loads((ROOT / "manifests/ICL_CONTEXTTENSORV1_PLAN.json").read_text(encoding="utf-8"))
        self.assertEqual(context_v1.validate_tensorization_plan(plan)["status"], "PASS_CPU_ONLY_PLAN")
        plan["model_visible_fields"].append("target_id")
        with self.assertRaises(context_v1.ContextV1Error) as caught:
            context_v1.validate_tensorization_plan(plan)
        self.assertEqual(caught.exception.reason_code, "MODEL_VISIBILITY_DRIFT")

    def test_future_information_and_target_injection_fail_closed(self) -> None:
        future = copy.deepcopy(self.fixture)
        future["interaction_history"][0]["event_time"] = 1.1
        self.assert_rejected(future, "CAUSAL_TIMESTAMP_VIOLATION")
        for section in ("interaction_history", "local_dynamics_context", "failure_outcome_context"):
            available = copy.deepcopy(self.fixture)
            available[section][0]["event_time"] = 0.8
            available[section][0]["available_time"] = 1.1
            self.assert_rejected(available, "CAUSAL_TIMESTAMP_VIOLATION")
        missing_availability = copy.deepcopy(self.fixture)
        del missing_availability["local_dynamics_context"][0]["available_time"]
        self.assert_rejected(missing_availability, "AVAILABILITY_TIME_MISSING")
        target = copy.deepcopy(self.fixture)
        target["target_xyz_world"] = [0.30, 0.10, 0.58]
        self.assert_rejected(target, "FORBIDDEN_TARGET_XYZ")
        expert = copy.deepcopy(self.fixture)
        expert["failure_outcome_context"][0]["future_expert_action"] = context_v1.ZERO_SHA256
        self.assert_rejected(expert, "FUTURE_EXPERT_INFORMATION")

    def test_locked_split_and_path_leakage_fail_closed(self) -> None:
        locked = copy.deepcopy(self.fixture)
        locked["notes"] = "Holdout-F is not available"
        self.assert_rejected(locked, "LOCKED_SPLIT_CONTAMINATION")
        path = copy.deepcopy(self.fixture)
        path["source_identity"]["artifact_id"] = "../secret"
        self.assert_rejected(path, "PATH_LEAKAGE")
        absolute = copy.deepcopy(self.fixture)
        absolute["notes"] = "/tmp/context.json"
        self.assert_rejected(absolute, "PATH_LEAKAGE")

    def test_missing_response_is_not_silently_fabricated(self) -> None:
        missing = copy.deepcopy(self.fixture)
        del missing["interaction_history"][0]["executed_or_observed_response_digest"]
        self.assert_rejected(missing, "MISSING_EXPLICIT_NULL_RESPONSE")
        invalid = copy.deepcopy(self.fixture)
        invalid["local_dynamics_context"][0]["executed_or_observed_response_digest"] = "not-a-sha"
        self.assert_rejected(invalid, "INVALID_SHA256")
        missing_dynamics = copy.deepcopy(self.fixture)
        missing_dynamics["local_dynamics_context"][0]["state_after_digest"] = None
        self.assert_rejected(missing_dynamics, "INVALID_SHA256")

    def test_duplicate_query_ids_and_schema_drift_are_rejected(self) -> None:
        with self.assertRaises(context_v1.ContextV1Error) as caught:
            context_v1.build_context_index([self.fixture, copy.deepcopy(self.fixture)])
        self.assertEqual(caught.exception.reason_code, "DUPLICATE_QUERY_ID")
        unknown = copy.deepcopy(self.fixture)
        unknown["unexpected"] = True
        self.assert_rejected(unknown, "UNKNOWN_FIELD")
        abi = copy.deepcopy(self.fixture)
        abi["notes"] = "StructuredActionV1 shape [10, 10]"
        # ABI-like prose is harmless metadata; the ABI is fixed by the
        # contract and never read from model output in this preparation stage.
        self.assertEqual(context_v1.validate_context(abi)["structured_action_abi"]["shape"], [40, 132])

    def test_plan_manifests_have_no_execution_surface(self) -> None:
        ladder = json.loads((ROOT / "manifests/STAGE45F_ICL_EXPERIMENT_LADDER_V1.json").read_text(encoding="utf-8"))
        self.assertEqual(len(ladder["context_variants"]), 9)
        for experiment in ladder["experiments"].values():
            self.assertFalse(experiment["execution_enabled"])
            self.assertEqual(experiment["model_forward"], 0)
            self.assertEqual(experiment["training"], 0)
        self.assertIn("I0_EXISTING_INTERFACE_CONTEXT_PROBE", ladder["experiments"])
        self.assertIn("I0_ADAPTER_ZERO_GATE_PARITY", ladder["experiments"])
        self.assertTrue(ladder["experiments"]["I0_ADAPTER_ZERO_GATE_PARITY"]["actual_forward_locked_until_future_gate"])
        options = json.loads((ROOT / "manifests/STAGE45F_ICL_INTEGRATION_OPTIONS_V1.json").read_text(encoding="utf-8"))
        self.assertFalse(options["selection_policy"]["automatic_selection"])
        self.assertTrue(options["selection_policy"]["PI_DECISION_REQUIRED"])
        self.assertEqual(options["options"]["C1_ACTION_SIDE_CONTEXT_TOKENS"]["gr00t_dimension_binding"], "input_embedding_dim=1536")
        self.assertEqual(options["options"]["C2_VL_MEMORY_CONTEXT_TOKENS"]["gr00t_dimension_binding"], "backbone_embedding_dim=2048")


if __name__ == "__main__":
    unittest.main()
