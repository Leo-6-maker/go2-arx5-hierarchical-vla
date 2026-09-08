#!/usr/bin/env python3
"""Run one bounded Stage41 LH10 GR00T-to-native-WBC episode.

The runner reuses the accepted Stage40 paused-inference mechanism but extends
the schedule to exactly ten causal replans.  A single model worker and a
single native simulator own the episode; every q(k) is requested only after
the preceding WBC segment has completed.  Evidence is written to a fresh run
directory and is sealed separately by :mod:`stage41_corelab_evidence`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import stage40_corelab_evidence as stage40  # noqa: E402
import stage41_corelab_evidence as evidence  # noqa: E402
from stage40_corelab_phase import (  # noqa: E402
    CHUNK_DURATION_S,
    HANDOFF_S,
    SOURCE_FRAME,
    TCP_SEMANTIC_ID,
    _case_request,
    _case_summary,
    _check_rows,
    _current_eef,
    _initial_bind,
    _load_pose_file,
    _native_args,
    _relative_path,
    _snapshot_gpu,
    _start_worker,
    _stop_worker,
    _step_rows,
    _tensor_hash,
    _update_reference,
    _write_json,
)
from go2arx5_vla.adapters import GR00TToWBCAdapter, ReferenceAdapterConfig  # noqa: E402
from go2arx5_vla.safety import GateDecision, TrajectorySafetyGateV1  # noqa: E402


HANDOFF_COUNT = evidence.HANDOFF_COUNT
PHASE = evidence.STAGE_ID
TARGET_POSITION_WORLD_M = np.asarray([0.30, 0.10, 0.58], dtype=np.float64)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _host_snapshot(path: Path, label: str) -> Dict[str, Any]:
    """Capture process/system RAM without depending on psutil."""

    status: Dict[str, int] = {}
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition(":")
            if key in {"VmRSS", "VmHWM", "VmSize"}:
                fields = value.strip().split()
                if fields:
                    status[key] = int(fields[0])
    except (OSError, ValueError):
        pass
    system: Dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition(":")
            if key in {"MemTotal", "MemAvailable"}:
                fields = value.strip().split()
                if fields:
                    system[key] = int(fields[0])
    except (OSError, ValueError):
        pass
    usage = resource.getrusage(resource.RUSAGE_SELF)
    value = {
        "schema": "CORELAB_STAGE41_HOST_RAM_SNAPSHOT_V1",
        "label": label,
        "process": {"status_kib": status, "maxrss": int(getattr(usage, "ru_maxrss", 0))},
        "system_kib": system,
    }
    _write_json(path, value)
    return value


def _telemetry_sample(root: Path, telemetry: List[Mapping[str, Any]], label: str) -> Dict[str, Any]:
    gpu = _snapshot_gpu(root / ("gpu_" + label + ".json"), label)
    host = _host_snapshot(root / ("host_" + label + ".json"), label)
    sample = {"label": label, "gpu": gpu, "host": host, "wall_time": time.time()}
    telemetry.append(sample)
    return sample


def _counter_evidence(native: Any, cases: Sequence[Mapping[str, Any]], rows_before: Sequence[Any], rows_after: Sequence[Any], handoffs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    before = {name: 0 for name in evidence.COUNTER_NAMES}
    after = {
        "gr00t_forward_count": len(cases),
        "wbc_step_count": int(getattr(native, "stage21_wbc_forward_count", 0)) if native is not None else 0,
        "reference_update_count": int(getattr(native, "stage21_reference_update_count", 0)) if native is not None else 0,
    }
    return evidence.build_counter_observation(
        before,
        after,
        gr00t_requests=[1] * len(cases),
        wbc_segments=[len(segment) for segment in list(rows_before) + list(rows_after)],
        reference_updates=[1] * len(handoffs),
    )


def _materialize_evidence(
    args: argparse.Namespace,
    root: Path,
    mailbox: Path,
    cases: Sequence[Mapping[str, Any]],
    safety_decisions: Sequence[str],
    native: Any,
    rows_before: Sequence[Any],
    rows_after: Sequence[Any],
    bind_events: Sequence[Any],
    handoffs: Sequence[Any],
    executor_events: Sequence[Any],
    state_events: Sequence[Any],
    telemetry: Sequence[Any],
    worker: Optional[Any],
    worker_return_code: Optional[int],
    status: str,
    completion: Optional[Mapping[str, Any]],
    error: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    summaries = [_case_summary(case, mailbox) for case in cases]
    body_values: Dict[str, np.ndarray] = {}
    wrist_values: Dict[str, np.ndarray] = {}
    raw_values: Dict[str, np.ndarray] = {}
    decoded_values: Dict[str, np.ndarray] = {}
    for case, summary in zip(cases, summaries):
        case_id = str(case["request_id"])
        with np.load(case["observation_npz"], allow_pickle=False) as data:
            body_values[case_id] = np.asarray(data["body_rgb"], dtype=np.uint8)
            wrist_values[case_id] = np.asarray(data["wrist_rgb"], dtype=np.uint8)
            state_value = np.asarray(data["state"], dtype=np.float32).reshape(-1)
        with np.load(case["response"]["artifact"], allow_pickle=False) as data:
            raw_values[case_id] = np.asarray(data["raw_universal_action"])
        decoded_values[case_id] = np.asarray(case["decoded"], dtype=np.float64)
        summary.update({
            "state_v1_shape": list(state_value.shape),
            "state_v1_dtype": str(state_value.dtype),
            "state_v1_sha256": stage40._array_sha(state_value),
            "raw_shape": list(raw_values[case_id].shape),
            "raw_sha256": stage40._array_sha(raw_values[case_id]),
            "decoded_shape": list(decoded_values[case_id].shape),
            "decoded_sha256": stage40._array_sha(decoded_values[case_id]),
            "finite": bool(np.isfinite(state_value).all() and np.isfinite(raw_values[case_id]).all() and np.isfinite(decoded_values[case_id]).all()),
        })
        summary["wrist_dtype"] = str(wrist_values[case_id].dtype)
    np.savez_compressed(root / "observation_body.npz", **body_values)
    np.savez_compressed(root / "observation_wrist.npz", **wrist_values)
    np.savez_compressed(root / "raw_action_d132.npz", **raw_values)
    np.savez_compressed(root / "decoded_action_d10_h40.npz", **decoded_values)
    _write_json(root / "state_v1.json", {"schema": "CORELAB_STAGE41_STATE_V1_EVIDENCE", "observations": summaries})
    _write_json(root / "worker_request.json", {"schema": "CORELAB_STAGE41_WORKER_REQUEST_EVIDENCE_V1", "requests": [item["request"] for item in summaries]})
    _write_json(root / "worker_response.json", {"schema": "CORELAB_STAGE41_WORKER_RESPONSE_EVIDENCE_V1", "responses": [item["response"] for item in summaries]})
    _write_json(root / "safety_gate.json", {"schema": "CORELAB_STAGE41_SAFETYGATE_EVIDENCE_V1", "decisions": list(safety_decisions), "cases": [item["safety_gate"] for item in summaries]})
    _write_json(root / "mailbox.json", {"schema": "CORELAB_STAGE41_MAILBOX_EVIDENCE_V1", "mailbox": str(mailbox), "requests": sorted(str(path) for path in (mailbox / "requests").glob("*")), "responses": sorted(str(path) for path in (mailbox / "responses").glob("*")), "worker_pid": None if worker is None else int(worker.pid), "worker_return_code": worker_return_code})
    _write_json(root / "logs.json", {"schema": "CORELAB_STAGE41_PHASE_LOG_EVIDENCE_V1", "phase": PHASE, "status": status, "worker_log": str(root / "worker.log"), "error": error})
    _write_json(root / "telemetry.json", {"schema": "CORELAB_STAGE41_TELEMETRY_EVIDENCE_V1", "samples": list(telemetry)})
    _write_json(root / "counter_observations.json", _counter_evidence(native, cases, rows_before, rows_after, handoffs))
    _write_json(root / "initial_bind.json", {"schema": "CORELAB_STAGE41_INITIAL_BIND_EVIDENCE_V1", "events": list(bind_events)})
    _write_json(root / "root_state.json", {"schema": "CORELAB_STAGE41_ROOT_STATE_EVIDENCE_V1", "events": list(state_events)})
    _write_json(root / "dof_state.json", {"schema": "CORELAB_STAGE41_DOF_STATE_EVIDENCE_V1", "events": list(state_events)})
    _write_json(root / "executor_bookkeeping.json", {"schema": "CORELAB_STAGE41_EXECUTOR_BOOKKEEPING_EVIDENCE_V1", "events": list(executor_events)})
    _write_json(root / "rows_before_handoff.json", {"schema": "CORELAB_STAGE41_ROWS_BEFORE_HANDOFF_EVIDENCE_V1", "segments": list(rows_before)})
    _write_json(root / "rows_after_handoff.json", {"schema": "CORELAB_STAGE41_ROWS_AFTER_HANDOFF_EVIDENCE_V1", "segments": list(rows_after)})
    _write_json(root / "handoff.json", {"schema": "CORELAB_STAGE41_HANDOFF_EVIDENCE_V1", "events": list(handoffs), "completion": completion})
    forbidden = evidence.expected_forbidden_operations()
    metadata = {
        "schema": "CORELAB_STAGE41_LH10_RUN_V1",
        "stage": PHASE,
        "status": status,
        "execution_class": evidence.EXECUTION_CLASS,
        "dev_seed": evidence.DEV_SEED,
        "seed": int(args.seed),
        "handoff_count": HANDOFF_COUNT,
        "bundle": str(args.bundle),
        "bundle_sha256": evidence.sha256_file(args.bundle),
        "predecessor_seal": str(args.predecessor_seal),
        "predecessor_seal_sha256": evidence.sha256_file(args.predecessor_seal),
        "gpu_uuid": os.environ.get("STAGE40_CT3_GPU_UUID", "unknown"),
        "native_control_dt_s": float(native.dt) if native is not None else None,
        "gr00t_forward_count": len(cases),
        "wbc_step_count": int(getattr(native, "stage21_wbc_forward_count", 0)) if native is not None else 0,
        "reference_update_count": int(getattr(native, "stage21_reference_update_count", 0)) if native is not None else 0,
        "native_handoff_steps": [int(event.get("native_handoff_steps", event.get("actual_native_control_steps", 0))) for event in handoffs],
        "executor_prefix_steps": [int(event.get("executor_prefix_steps", 0)) for event in handoffs],
        "action_sample_dt_s": [float(event.get("action_sample_dt_s", float("nan"))) for event in handoffs],
        "worker_return_code": worker_return_code,
        "worker_pid": None if worker is None else int(worker.pid),
        "training": False,
        "scientific_rows": int(sum(len(segment) for segment in rows_before) + sum(len(segment) for segment in rows_after)),
        "forbidden_operations": forbidden,
        "completion": completion,
        "error": error,
        "platform": {"python": platform.python_version(), "hostname": platform.node()},
    }
    _write_json(root / "run_metadata.json", metadata)
    roles = {
        "observation_body": "observation_body.npz",
        "observation_wrist": "observation_wrist.npz",
        "state_v1": "state_v1.json",
        "worker_request": "worker_request.json",
        "worker_response": "worker_response.json",
        "raw_action_d132": "raw_action_d132.npz",
        "decoded_action_d10_h40": "decoded_action_d10_h40.npz",
        "safety_gate": "safety_gate.json",
        "initial_bind": "initial_bind.json",
        "root_state": "root_state.json",
        "dof_state": "dof_state.json",
        "executor_bookkeeping": "executor_bookkeeping.json",
        "rows_before_handoff": "rows_before_handoff.json",
        "rows_after_handoff": "rows_after_handoff.json",
        "handoff": "handoff.json",
        "counter_observations": "counter_observations.json",
        "mailbox": "mailbox.json",
        "logs": "logs.json",
        "telemetry": "telemetry.json",
        "run_metadata": "run_metadata.json",
    }
    _write_json(root / "roles.json", roles)
    return {"summaries": summaries, "metadata": metadata, "roles": roles}


def _run(args: argparse.Namespace) -> int:
    phase_root = Path(args.output_dir).resolve()
    _require(not phase_root.exists(), "LH10 output directory already exists")
    phase_root.mkdir(parents=True)
    mailbox = phase_root / "mailbox"
    worker = None
    native = env = None
    cases: List[Dict[str, Any]] = []
    rows_before: List[Any] = []
    rows_after: List[Any] = []
    bind_events: List[Any] = []
    handoffs: List[Any] = []
    executor_events: List[Any] = []
    state_events: List[Any] = []
    telemetry: List[Any] = []
    completion: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    status = "BLOCKED"
    worker_return_code = None
    bundle = evidence.read_json(args.bundle)
    evidence.validate_bundle(bundle)
    predecessor_info = stage40.verify_stage_seal(
        Path(args.predecessor_seal), Path(args.predecessor_evidence_root), expected_stage="M2", _admit_without_predecessor=True
    )
    _require(bundle["predecessor"]["seal_sha256"] == predecessor_info["seal_sha256"], "bundle predecessor M2 seal mismatch")
    _telemetry_sample(phase_root, telemetry, "before_episode")
    trajectory = _load_pose_file(args.trajectory)
    try:
        worker = _start_worker(args, mailbox, phase_root / "worker.log")
        native, env, actors, loads, _ = stage40._build_native(_native_args(args), trajectory)
        gate = TrajectorySafetyGateV1()
        adapter = GR00TToWBCAdapter(ReferenceAdapterConfig(expected_source_frame=SOURCE_FRAME, wbc_frame=SOURCE_FRAME, tcp_semantic_id=TCP_SEMANTIC_ID, stale_timeout_s=0.75))
        sim_time = 0.0
        handoff_steps = int(round(HANDOFF_S / float(native.dt)))
        _require(handoff_steps > 0, "LH10 handoff schedule produced no native ticks")
        current_chunk, first_case = _case_request(native, mailbox, "stage41-q0", args.seed, sim_time, gate, adapter, args)
        cases.append(first_case)
        _require(first_case["safety_gate"]["decision"] != GateDecision.REJECT.value, "SAFETYGATE_REJECT_Q0")
        executor, initial_bind_event, state_event = _initial_bind(native, env, current_chunk, sim_time, {"stage": PHASE, "mode": "CORELAB_STAGE41_LH10"})
        bind_events.append(initial_bind_event)
        state_events.append(state_event)
        executor_events.append(initial_bind_event.get("executor_initial"))
        for handoff_index in range(HANDOFF_COUNT):
            segment, sim_time, termination = _step_rows(native, env, actors, chunk=current_chunk, sim_time=sim_time, steps=handoff_steps, update_count=handoff_index)
            _check_rows(segment, "handoff segment %d" % (handoff_index + 1))
            rows_before.append(segment)
            _telemetry_sample(phase_root, telemetry, "before_replan_q%d" % (handoff_index + 1))
            next_chunk, next_case = _case_request(native, mailbox, "stage41-q%d" % (handoff_index + 1), args.seed, sim_time, gate, adapter, args)
            cases.append(next_case)
            _require(next_case["safety_gate"]["decision"] != GateDecision.REJECT.value, "SAFETYGATE_REJECT_Q%d" % (handoff_index + 1))
            update_event, update_state_event = _update_reference(native, env, executor, current_chunk, next_chunk, sim_time, handoff_steps, handoff_index + 1)
            handoffs.append(update_event)
            state_events.append(update_state_event)
            executor_events.append(update_event.get("executor_update"))
            current_chunk = next_chunk
        final_segment, sim_time, termination = _step_rows(native, env, actors, chunk=current_chunk, sim_time=sim_time, steps=max(1, int(np.ceil(CHUNK_DURATION_S / float(native.dt)))), update_count=HANDOFF_COUNT)
        _check_rows(final_segment, "completion segment")
        rows_after.append(final_segment)
        native_time = float(native.traj_sim_time[0].detach().cpu().item())
        ref_s = float(native.traj_batch.s_ref(native.traj_sim_time)[0].detach().cpu().item())
        ref_L = float(native.traj_batch.L[0].detach().cpu().item())
        ref_duration = float(current_chunk.validity_end - current_chunk.validity_start)
        endpoint = bool(native_time >= ref_duration - max(float(native.dt) * 0.5, 1e-9) and ref_s >= ref_L - max(1e-4, 1e-3 * max(ref_L, 1.0)))
        completion = {"native_time_s": native_time, "reference_duration_s": ref_duration, "traj_s_ref": ref_s, "traj_L": ref_L, "endpoint_reached": endpoint, "termination": termination, "handoff_steps": handoff_steps, "sim_time_s": sim_time}
        observation_timestamps = [float(case["observation_timestamp"]) for case in cases]
        decisions = [str(case["safety_gate"]["decision"]) for case in cases]
        evidence.validate_long_horizon_contract(
            cases,
            handoffs,
            [len(segment) for segment in rows_before + rows_after],
            decisions,
            completion,
            native_dt_s=float(native.dt),
            observation_timestamps=observation_timestamps,
            continuity_events=[initial_bind_event] + handoffs,
        )
        _require(int(native.stage21_reference_update_count) == HANDOFF_COUNT, "LH10 reference update count mismatch")
        _require(int(native.stage21_wbc_forward_count) == sum(len(segment) for segment in rows_before + rows_after), "LH10 WBC count mismatch")
        status = "PASS"
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        status = "BLOCKED"
    finally:
        try:
            if native is not None:
                _telemetry_sample(phase_root, telemetry, "after_native")
        except Exception as exc:
            error = error or {"type": type(exc).__name__, "message": str(exc)}
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        worker_return_code = _stop_worker(worker, mailbox)
        try:
            _telemetry_sample(phase_root, telemetry, "after_episode")
        except Exception as exc:
            error = error or {"type": type(exc).__name__, "message": str(exc)}
        try:
            _materialize_evidence(args, phase_root, mailbox, cases, [str(case["safety_gate"]["decision"]) for case in cases], native, rows_before, rows_after, bind_events, handoffs, executor_events, state_events, telemetry, worker, worker_return_code, status, completion, error)
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
            status = "BLOCKED"
            _write_json(phase_root / "phase_error.json", error)
    _write_json(phase_root / "phase_result.json", {"schema": "CORELAB_STAGE41_PHASE_RESULT_V1", "stage": PHASE, "status": status, "cases": len(cases), "error": error, "completion": completion, "bundle": str(args.bundle), "bundle_sha256": evidence.sha256_file(args.bundle)})
    print(json.dumps({"stage": PHASE, "status": status, "output_dir": str(phase_root), "cases": len(cases), "wbc_steps": int(getattr(native, "stage21_wbc_forward_count", 0)) if native is not None else 0, "reference_updates": int(getattr(native, "stage21_reference_update_count", 0)) if native is not None else 0, "error": error}, sort_keys=True), flush=True)
    return 0 if status == "PASS" else 2


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--predecessor-seal", type=Path, required=True)
    parser.add_argument("--predecessor-evidence-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wbc-source-root", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--arm", type=Path, required=True)
    parser.add_argument("--dog", type=Path, required=True)
    parser.add_argument("--stage10-root", type=Path, required=True)
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--modality-config", type=Path, required=True)
    parser.add_argument("--cosmos-snapshot", type=Path, required=True)
    parser.add_argument("--structured-stats", type=Path, required=True)
    parser.add_argument("--model-python", type=Path, required=True)
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=evidence.DEV_SEED)
    parser.add_argument("--response-timeout-s", type=float, default=300.0)
    parser.add_argument("--worker-ready-timeout-s", type=float, default=1200.0)
    args = parser.parse_args(argv)
    if args.seed != evidence.DEV_SEED:
        parser.error("Stage41 development seed is fixed at 4242")
    for path in (args.bundle, args.predecessor_seal, args.predecessor_evidence_root, args.wbc_source_root, args.trajectory, args.arm, args.dog, args.stage10_root, args.base_model_path, args.model_path, args.dataset_dir, args.modality_config, args.cosmos_snapshot, args.structured_stats, args.model_python):
        if not Path(path).exists():
            parser.error("missing required path: %s" % path)
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
