"""CPU-only Vanda Stage40 fail-closed harness and negative fixtures."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from go2arx5_vla.adapters import GR00TToWBCAdapter, ReferenceAdapterConfig  # noqa: E402
from go2arx5_vla.contracts import TrajectoryChunkV1  # noqa: E402
from go2arx5_vla.safety import GateDecision, TrajectorySafetyGateV1  # noqa: E402
from go2arx5_vla.stage40_structured_decode import decode_raw_structured_action  # noqa: E402
from vanda_stage40_admission import (  # noqa: E402
    DEV_SEED,
    STAGE40_DECODER,
    STAGE40_HANDOFF_S,
    Stage40AdmissionError,
    handoff_steps_for_native_dt,
    require_development_seed,
    validate_artifact_identities,
    validate_fake_lifecycle,
    validate_safety_gate_decision,
    validate_stage40_result_schema,
    validate_structured_action_contract,
)
from vanda_stage40_telemetry import collect_phase, seal_result_file  # noqa: E402


SOURCE_FRAME = "NATIVE_WBC_RELATIVE_REFERENCE"
TCP = "x5_link6_gripper_center"


def _chunk(*, source_frame: str = SOURCE_FRAME, tcp: str = TCP) -> TrajectoryChunkV1:
    timestamps = np.arange(40, dtype=np.float64) * 0.05
    position = np.stack(
        [
            0.2 + np.linspace(0.0, 0.08, 40),
            np.zeros(40, dtype=np.float64),
            np.full(40, 0.4, dtype=np.float64),
        ],
        axis=-1,
    )
    orientation = np.tile(np.asarray([0.0, 0.0, 0.0, 1.0]), (40, 1))
    return TrajectoryChunkV1(
        chunk_id="harness-chunk",
        source="vanda-cpu-harness",
        observation_timestamp=0.0,
        point_timestamps=timestamps,
        eef_position=position,
        eef_orientation=orientation,
        source_frame=source_frame,
        tcp_semantic_id=tcp,
        validity_start=0.0,
        validity_end=1.95,
        gripper=np.full(40, 0.5),
    )


def _lifecycle_events() -> list[str]:
    return [
        "observation_q0",
        "q0_decode",
        "safety_q0",
        "initial_bind",
        "q0_prefix",
        "handoff_pause",
        "observation_q1",
        "q1_decode",
        "safety_q1",
        "reference_update",
        "q1_continuation",
        "reference_complete",
        "seal",
    ]


def _pass_result() -> dict[str, object]:
    return {
        "schema": "VANDA_STAGE40_V5_RESULT_V1",
        "stage": "Stage40",
        "pi_gate": "PI_GATE_STATE_0060",
        "status": "PASS",
        "seed": DEV_SEED,
        "handoff_target_s": STAGE40_HANDOFF_S,
        "handoff_steps": 20,
        "actual_switch_s": 1.0,
        "switch_tick_error_s": 0.025,
        "executed_prefix_steps": 20,
        "single_reference_handoff": True,
        "physics_reset": False,
        "physics_teleport": False,
        "policy_history_reset": False,
        "q0_forward_count": 1,
        "q1_forward_count": 1,
        "wbc_forward_count": 60,
        "gpu_uuid": "GPU-test-a40",
        "vram_baseline_mib": 1024.0,
        "vram_peak_mib": 24576.0,
        "artifact_identities": {"structured": "sha256:structured"},
        "runtime_identities": {"isaac_sif": "sha256:sif"},
        "git_head": "a74d550be212e70d7371d75fc968d6a893595030",
    }


class VandaStage40FailClosedTest(unittest.TestCase):
    def test_positive_structured_contract_and_handoff_schedule(self) -> None:
        contract = validate_structured_action_contract(
            raw_shape=(1, 40, 132),
            source_timestamps=np.arange(40, dtype=np.float64) * 0.05,
            timing_mode="T16",
            physical_shape=(40, 10),
        )
        self.assertEqual(contract["decoder"], STAGE40_DECODER)
        self.assertEqual(handoff_steps_for_native_dt(0.05), 20)

    def test_wrong_checkpoint_and_statistics_identity_block(self) -> None:
        expected = {"structured_checkpoint_1000": "checkpoint-ok", "structured_statistics": "stats-ok"}
        with self.assertRaises(Stage40AdmissionError):
            validate_artifact_identities(expected, {
                "structured_checkpoint_1000": "checkpoint-drift",
                "structured_statistics": "stats-ok",
            })
        with self.assertRaises(Stage40AdmissionError):
            validate_artifact_identities(expected, {
                "structured_checkpoint_1000": "checkpoint-ok",
                "structured_statistics": "stats-drift",
            })

    def test_wrong_horizon_dimension_timing_and_physical_dimension_block(self) -> None:
        common = {
            "source_timestamps": np.arange(40, dtype=np.float64) * 0.05,
            "timing_mode": "T16",
            "physical_shape": (40, 10),
        }
        with self.assertRaises(Stage40AdmissionError):
            validate_structured_action_contract(raw_shape=(1, 39, 132), **common)
        with self.assertRaises(Stage40AdmissionError):
            validate_structured_action_contract(
                raw_shape=(1, 40, 132), timing_mode="T20",
                source_timestamps=common["source_timestamps"],
                physical_shape=(40, 10),
            )
        with self.assertRaises(Stage40AdmissionError):
            validate_structured_action_contract(raw_shape=(1, 40, 132), physical_shape=(40, 9), **{
                key: value for key, value in common.items() if key != "physical_shape"
            })

    def test_generic_decode_path_is_forbidden(self) -> None:
        with self.assertRaises(Stage40AdmissionError):
            validate_structured_action_contract(
                raw_shape=(1, 40, 132),
                source_timestamps=np.arange(40, dtype=np.float64) * 0.05,
                timing_mode="T16",
                physical_shape=(40, 10),
                generic_decode_action_used=True,
            )

    def test_nonfinite_raw_action_blocks_at_actual_decoder(self) -> None:
        with self.assertRaises(ValueError):
            decode_raw_structured_action(
                np.full((1, 40, 132), np.nan, dtype=np.float32),
                np.zeros(10, dtype=np.float32),
                stats_path=ROOT / "does-not-matter-after-finite-check.json",
                source_timestamps=np.arange(40, dtype=np.float64) * 0.05,
                observation_timestamp=0.0,
                chunk_id="nan",
            )

    def test_wrong_frame_and_tcp_block_at_actual_adapter(self) -> None:
        adapter = GR00TToWBCAdapter(ReferenceAdapterConfig(
            expected_source_frame=SOURCE_FRAME,
            wbc_frame=SOURCE_FRAME,
            tcp_semantic_id=TCP,
        ))
        with self.assertRaises(ValueError):
            adapter.adapt(_chunk(source_frame="WRONG_FRAME"))
        with self.assertRaises(ValueError):
            adapter.adapt(_chunk(tcp="WRONG_TCP"))

    def test_safety_gate_reject_is_stop_not_repair(self) -> None:
        bad = SimpleNamespace(
            chunk_id="nan-gate",
            point_timestamps=np.arange(40, dtype=np.float64) * 0.05,
            eef_position=np.full((40, 3), np.nan, dtype=np.float64),
            eef_orientation=np.tile(np.asarray([0.0, 0.0, 0.0, 1.0]), (40, 1)),
            eef_rot6d=None,
        )
        result = TrajectorySafetyGateV1().evaluate(bad, now=0.0)
        self.assertIs(result.decision, GateDecision.REJECT)
        with self.assertRaises(Stage40AdmissionError):
            validate_safety_gate_decision("REJECT")

    def test_fake_full_lifecycle_passes(self) -> None:
        result = validate_fake_lifecycle(
            _lifecycle_events(),
            handoff_steps=20,
            executed_prefix_steps=20,
            safety_q0="ACCEPT",
            safety_q1="WARN",
        )
        self.assertEqual(result["reference_update_count"], 1)
        self.assertFalse(result["physics_reset"])

    def test_fake_negative_lifecycle_cases_block(self) -> None:
        cases = [
            (_lifecycle_events(), 19, 20, "prefix"),
            (_lifecycle_events() + ["second_reference_update"], 20, 20, "second handoff"),
            (["reset_before_handoff"] + _lifecycle_events(), 20, 20, "reset"),
            (_lifecycle_events(), 20, 20, "SafetyGate"),
        ]
        for events, handoff_steps, executed, label in cases[:3]:
            with self.subTest(label=label):
                with self.assertRaises(Stage40AdmissionError):
                    validate_fake_lifecycle(events, handoff_steps=handoff_steps, executed_prefix_steps=executed)
        with self.assertRaises(Stage40AdmissionError):
            validate_fake_lifecycle(
                _lifecycle_events(),
                handoff_steps=20,
                executed_prefix_steps=20,
                safety_q0="REJECT",
            )

    def test_formal_and_nondevelopment_seeds_block(self) -> None:
        self.assertEqual(require_development_seed(4242), 4242)
        for seed in (2223, 2228, 2233, 2238, 1):
            with self.subTest(seed=seed):
                with self.assertRaises(Stage40AdmissionError):
                    require_development_seed(seed)

    def test_result_schema_requires_full_a40_seal_fields(self) -> None:
        result = validate_stage40_result_schema(_pass_result())
        self.assertTrue(result["schema_valid"])
        bad = _pass_result()
        del bad["gpu_uuid"]
        with self.assertRaises(Stage40AdmissionError):
            validate_stage40_result_schema(bad)


class VandaStage40TelemetryTest(unittest.TestCase):
    def test_phase_telemetry_records_gpu_and_runtime_fields(self) -> None:
        def fake_runner(command, **kwargs):
            return subprocess.CompletedProcess(
                command,
                0,
                "0, GPU-test-a40, NVIDIA A40, 46068, 128, 575.57.08, 8.6\n",
                "",
            )

        with tempfile.TemporaryDirectory() as directory:
            row = collect_phase(
                "after_gr00t_construction",
                Path(directory),
                job_id="test-job",
                gpu_index="0",
                runtime_root="/runtime",
                cache_root="/cache",
                command_runner=fake_runner,
            )
            self.assertEqual(row["gpu"]["status"], "PASS")
            self.assertEqual(row["gpu"]["rows"][0]["name"], "NVIDIA A40")
            lines = (Path(directory) / "stage40_telemetry.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0])["phase"], "after_gr00t_construction")

    def test_result_seal_is_canonical_and_fail_closed(self) -> None:
        result = _pass_result()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "result.json"
            seal_path = root / "result.seal.json"
            input_path.write_text(json.dumps(result), encoding="utf-8")
            info = seal_result_file(input_path, seal_path)
            payload = seal_path.read_bytes()
            self.assertEqual(info["sha256"], hashlib.sha256(payload).hexdigest())
            self.assertTrue((root / "result.seal.json.sha256").is_file())

            invalid = dict(result)
            invalid["status"] = "BLOCKED"
            invalid_path = root / "invalid.json"
            invalid_path.write_text(json.dumps(invalid), encoding="utf-8")
            blocked_output = root / "blocked.seal.json"
            with self.assertRaises(ValueError):
                seal_result_file(invalid_path, blocked_output)
            self.assertFalse(blocked_output.exists())


if __name__ == "__main__":
    unittest.main()
