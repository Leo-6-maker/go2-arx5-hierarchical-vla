"""CPU-only contract tests for the Stage41 LH10 harness and evidence seal."""

from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import stage41_corelab_evidence as EVIDENCE  # noqa: E402


def _continuity_event() -> dict:
    return {
        "physics_root_state_unchanged": True,
        "physics_dof_state_unchanged": True,
        "episode_length_unchanged": True,
        "arm_policy_history_unchanged": True,
        "dog_policy_history_unchanged": True,
        "reset_buf_unchanged": True,
        "physics_reset": False,
        "physics_teleport": False,
        "policy_history_reset": False,
    }


def _contract_fixture(native_dt: float = 0.025, action_dt: float = 0.075):
    native_steps = int(round(EVIDENCE.HANDOFF_S / native_dt))
    prefix_steps = int(round(EVIDENCE.HANDOFF_S / action_dt))
    cases = []
    for index in range(EVIDENCE.GR00T_FORWARD_COUNT):
        cases.append({
            "request_id": "stage41-q%d" % index,
            "observation_timestamp": float(index + 1),
            "response": {
                "worker_forward_count_before": index,
                "worker_forward_count_after": index + 1,
                "worker_forward_count_delta": 1,
                "action_contract": "STRUCTURED_ACTION_V1",
            },
            "safety_gate": {"decision": "ACCEPT"},
            "finite": True,
        })
    handoffs = []
    for index in range(EVIDENCE.HANDOFF_COUNT):
        event = _continuity_event()
        event.update({
            "handoff_index": index + 1,
            "native_handoff_steps": native_steps,
            "executor_prefix_steps": prefix_steps,
            "action_sample_dt_s": action_dt,
        })
        handoffs.append(event)
    completion = {"endpoint_reached": True}
    return cases, handoffs, [3] * EVIDENCE.GR00T_FORWARD_COUNT, ["ACCEPT"] * EVIDENCE.SAFETY_DECISION_COUNT, completion, native_steps, prefix_steps


class Stage41LongHorizonContractTest(unittest.TestCase):
    def test_dynamic_native_and_action_clocks_pass_without_magic_values(self):
        cases, handoffs, segments, decisions, completion, native_steps, prefix_steps = _contract_fixture()
        result = EVIDENCE.validate_long_horizon_contract(
            cases,
            handoffs,
            segments,
            decisions,
            completion,
            native_dt_s=0.025,
            expected_native_steps=[native_steps] * EVIDENCE.HANDOFF_COUNT,
            observation_timestamps=[case["observation_timestamp"] for case in cases],
            continuity_events=[_continuity_event()] + handoffs,
        )
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["native_handoff_steps"], [native_steps] * EVIDENCE.HANDOFF_COUNT)
        self.assertEqual(result["executor_prefix_steps"], [prefix_steps] * EVIDENCE.HANDOFF_COUNT)

    def test_native_dt_mismatch_fails_closed(self):
        cases, handoffs, segments, decisions, completion, _, _ = _contract_fixture()
        with self.assertRaisesRegex(EVIDENCE.Stage41EvidenceError, "native handoff schedule"):
            EVIDENCE.validate_long_horizon_contract(cases, handoffs, segments, decisions, completion, native_dt_s=0.02)

    def test_executor_prefix_is_derived_from_observed_action_period(self):
        cases, handoffs, segments, decisions, completion, _, _ = _contract_fixture()
        handoffs[0]["executor_prefix_steps"] = 39
        with self.assertRaisesRegex(EVIDENCE.Stage41EvidenceError, "executor action prefix"):
            EVIDENCE.validate_long_horizon_contract(cases, handoffs, segments, decisions, completion, native_dt_s=0.025)

    def test_stale_reordered_observation_fails_closed(self):
        cases, handoffs, segments, decisions, completion, _, _ = _contract_fixture()
        timestamps = [case["observation_timestamp"] for case in cases]
        timestamps[5] = timestamps[4]
        with self.assertRaisesRegex(EVIDENCE.Stage41EvidenceError, "stale or reordered"):
            EVIDENCE.validate_long_horizon_contract(cases, handoffs, segments, decisions, completion, native_dt_s=0.025, observation_timestamps=timestamps)

    def test_safety_reject_and_lifecycle_reset_fail_closed(self):
        cases, handoffs, segments, decisions, completion, _, _ = _contract_fixture()
        cases[3]["safety_gate"]["decision"] = "REJECT"
        with self.assertRaisesRegex(EVIDENCE.Stage41EvidenceError, "SafetyGate rejected"):
            EVIDENCE.validate_long_horizon_contract(cases, handoffs, segments, decisions, completion, native_dt_s=0.025)
        cases, handoffs, segments, decisions, completion, _, _ = _contract_fixture()
        handoffs[4]["physics_teleport"] = True
        with self.assertRaisesRegex(EVIDENCE.Stage41EvidenceError, "lifecycle"):
            EVIDENCE.validate_long_horizon_contract(cases, handoffs, segments, decisions, completion, native_dt_s=0.025, continuity_events=[_continuity_event()] + handoffs)

    def test_topology_and_counter_vectors_are_exact(self):
        cases, handoffs, segments, decisions, completion, _, _ = _contract_fixture()
        with self.assertRaisesRegex(EVIDENCE.Stage41EvidenceError, "exactly 10 reference"):
            EVIDENCE.validate_long_horizon_contract(cases, handoffs[:-1], segments, decisions, completion, native_dt_s=0.025)
        counters = EVIDENCE.build_counter_observation(
            {name: 100 for name in EVIDENCE.COUNTER_NAMES},
            {"gr00t_forward_count": 111, "wbc_step_count": 133, "reference_update_count": 110},
            [1] * 11,
            segments,
            [1] * 10,
        )
        self.assertEqual(counters["counters"]["gr00t_forward_count"]["delta"], 11)
        self.assertEqual(counters["counters"]["reference_update_count"]["delta"], 10)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bad = EVIDENCE.build_counter_observation(
                {name: 0 for name in EVIDENCE.COUNTER_NAMES},
                {"gr00t_forward_count": 11, "wbc_step_count": 33, "reference_update_count": 10},
                [1] * 10,
                segments,
                [1] * 10,
            )
            EVIDENCE.write_canonical_json(root / "counter.json", bad)
            manifest = {"records": [{"role": "counter_observations", "relative_path": "counter.json"}]}
            with self.assertRaisesRegex(EVIDENCE.Stage41EvidenceError, "per-request"):
                EVIDENCE.verify_counter_observations(
                    {"gr00t_forward_count": 11, "wbc_step_count": 33, "reference_update_count": 10}, manifest, root
                )


class Stage41SerializedEvidenceTest(unittest.TestCase):
    def _materialized_fixture(self, root: Path):
        cases, handoffs, segments, decisions, completion, native_steps, prefix_steps = _contract_fixture()
        observations = []
        for index, case in enumerate(cases):
            observation = copy.deepcopy(case)
            observation.update({
                "body_shape": [96, 96, 3], "body_dtype": "uint8",
                "wrist_shape": [96, 96, 3], "wrist_dtype": "uint8",
                "state_v1_shape": [10], "state_v1_sha256": "%064x" % (index + 1),
                "raw_shape": [1, 40, 132], "decoded_shape": [40, 10], "finite": True,
                "request": {"request_id": case["request_id"], "seed": EVIDENCE.DEV_SEED},
            })
            observation["response"]["status"] = "PASS"
            observations.append(observation)
        row = {"done": False, "reset_buf": False, "invalid_state": False, "termination_code": "NONE"}
        metadata = {
            "schema": "CORELAB_STAGE41_LH10_RUN_V1", "stage": EVIDENCE.STAGE_ID, "status": "PASS",
            "seed": EVIDENCE.DEV_SEED, "dev_seed": EVIDENCE.DEV_SEED, "handoff_count": 10,
            "bundle_sha256": "b" * 64, "forbidden_operations": EVIDENCE.expected_forbidden_operations(),
            "error": None, "worker_return_code": 0, "native_control_dt_s": 0.025,
            "native_handoff_steps": [native_steps] * 10, "executor_prefix_steps": [prefix_steps] * 10,
            "action_sample_dt_s": [0.075] * 10, "gr00t_forward_count": 11,
            "reference_update_count": 10, "wbc_step_count": 33,
        }
        payloads = {
            "state_v1": {"schema": "CORELAB_STAGE41_STATE_V1_EVIDENCE", "observations": observations},
            "safety_gate": {"decisions": decisions},
            "handoff": {"events": handoffs, "completion": completion},
            "initial_bind": {"events": [_continuity_event()]},
            "rows_before_handoff": {"segments": [[dict(row) for _ in range(3)] for _ in range(10)]},
            "rows_after_handoff": {"segments": [[dict(row) for _ in range(3)]]},
            "run_metadata": metadata,
        }
        roles = {}
        for role in sorted(EVIDENCE.REQUIRED_EVIDENCE_ROLES):
            filename = role + ".json"
            roles[role] = filename
            if role in payloads:
                EVIDENCE.write_canonical_json(root / filename, payloads[role])
            else:
                EVIDENCE.write_canonical_json(root / filename, {"role": role})
        manifest = EVIDENCE.build_evidence_manifest(root, roles)
        return manifest

    def test_serialized_pass_is_rechecked_before_seal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._materialized_fixture(root)
            result = {"execution_bundle_sha256": "b" * 64}
            checked = EVIDENCE._validate_serialized_run(manifest, root, result)
            self.assertEqual(checked["status"], "PASS")

    def test_serialized_row_termination_cannot_be_hidden_by_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._materialized_fixture(root)
            rows_path = root / "rows_before_handoff.json"
            rows = EVIDENCE.read_json(rows_path)
            rows["segments"][0][0]["done"] = True
            EVIDENCE.write_canonical_json(rows_path, rows)
            with self.assertRaisesRegex(EVIDENCE.Stage41EvidenceError, "lifecycle"):
                EVIDENCE._validate_serialized_run(manifest, root, {"execution_bundle_sha256": "b" * 64})


if __name__ == "__main__":
    unittest.main()
