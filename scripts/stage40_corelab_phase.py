#!/usr/bin/env python3
"""Run one direct Corelab Stage40 phase (C4--C7) on the frozen stack.

The native IsaacGym/WBC process remains the owner of the simulator.  A
separately loaded GR00T worker consumes only same-scene mailbox observations
and returns the raw StructuredActionV1 output.  This runner deliberately
keeps the phase boundaries separate so every forward, WBC step, and reference
update is observed and bound to its own Stage40 evidence bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from stage40_corelab_evidence import (  # noqa: E402
    COUNTER_NAMES,
    DEV_SEED,
    EXECUTION_CLASS,
    PREDECESSOR,
    build_counter_observation,
    canonical_json_bytes,
    read_json,
    sha256_bytes,
    sha256_file,
    validate_bundle,
    verify_stage_seal,
    write_canonical_json,
)
from stage40_live_one_handoff import (  # noqa: E402
    HANDOFF_S,
    CHUNK_DURATION_S,
    SOURCE_FRAME,
    TCP_SEMANTIC_ID,
    _build_native,
    _build_stage40_executor,
    _capture_scene,
    _current_eef,
    _current_state_v1,
    _jsonable,
    _load_pose_file,
    _request_model,
    _step_rows,
    _tensor_hash,
)
from go2arx5_vla.adapters import GR00TToWBCAdapter, ReferenceAdapterConfig  # noqa: E402
from go2arx5_vla.safety import GateDecision, TrajectorySafetyGateV1  # noqa: E402


PHASES = ("RTX-Q0", "RTX-bind", "M1", "M2")
MODEL_WORKER = SCRIPT_DIR / "stage21_same_scene_groot_worker.py"
TARGET_POSITION_WORLD_M = np.asarray([0.30, 0.10, 0.58], dtype=np.float64)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(_jsonable(value)))


def _array_sha(value: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(value)).tobytes()).hexdigest()


def _query_gpu() -> str:
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )


def _snapshot_gpu(path: Path, label: str) -> Dict[str, Any]:
    query = _query_gpu()
    value = {"schema": "CORELAB_STAGE40_GPU_SNAPSHOT_V1", "label": label, "query": query}
    _write_json(path, value)
    return value


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _start_worker(args: argparse.Namespace, mailbox: Path, log_path: Path) -> subprocess.Popen:
    _require(not (mailbox / "STOP").exists(), "stale mailbox STOP marker")
    mailbox.mkdir(parents=True, exist_ok=True)
    (mailbox / "requests").mkdir(parents=True, exist_ok=True)
    (mailbox / "responses").mkdir(parents=True, exist_ok=True)
    worker_env = os.environ.copy()
    # The native process uses the Python 3.8/IsaacGym library set.  Do not
    # leak that CUDA library path into the Python 3.12 model worker.
    worker_env.pop("LD_LIBRARY_PATH", None)
    worker_env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT / "src"), str(SCRIPT_DIR), str(args.stage10_root / "src" / "isaac-gr00t")]
    )
    cache_root = Path(args.cosmos_snapshot).resolve().parents[2]
    worker_env["HF_HUB_CACHE"] = str(cache_root)
    worker_env["TRANSFORMERS_CACHE"] = str(cache_root)
    worker_env["HF_HOME"] = str(cache_root.parent)
    worker_env["HF_HUB_OFFLINE"] = "1"
    worker_env["TRANSFORMERS_OFFLINE"] = "1"
    worker_env["HF_DATASETS_OFFLINE"] = "1"
    worker_env["PYTHONNOUSERSITE"] = "1"
    worker_env["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [
        str(args.model_python),
        "-B",
        str(MODEL_WORKER),
        "--stage10-root",
        str(args.stage10_root),
        "--base-model-path",
        str(args.base_model_path),
        "--model-path",
        str(args.model_path),
        "--dataset-dir",
        str(args.dataset_dir),
        "--modality-config",
        str(args.modality_config),
        "--cosmos-snapshot",
        str(args.cosmos_snapshot),
        "--mailbox",
        str(mailbox),
        "--device",
        "cuda:0",
        "--seed",
        str(args.seed),
        "--local-files-only",
        "--action-contract",
        "STRUCTURED_ACTION_V1",
    ]
    handle = log_path.open("w", encoding="utf-8")
    worker = subprocess.Popen(command, cwd=str(REPO_ROOT), env=worker_env, stdout=handle, stderr=subprocess.STDOUT)
    deadline = time.monotonic() + float(args.worker_ready_timeout_s)
    while time.monotonic() < deadline:
        if log_path.is_file():
            text = log_path.read_text(encoding="utf-8", errors="replace")
            if '"status": "READY"' in text:
                return worker
        if worker.poll() is not None:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-6000:]
            raise RuntimeError("GR00T worker exited before READY: rc=%s\n%s" % (worker.returncode, tail))
        time.sleep(0.25)
    worker.terminate()
    raise TimeoutError("timed out waiting for GR00T worker READY")


def _stop_worker(worker: Optional[subprocess.Popen], mailbox: Path) -> Optional[int]:
    if worker is None:
        return None
    try:
        _write_json(mailbox / "STOP", {"reason": "stage40_phase_finished"})
        try:
            return worker.wait(timeout=30.0)
        except subprocess.TimeoutExpired:
            worker.terminate()
            try:
                return worker.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                worker.kill()
                return worker.wait(timeout=10.0)
    finally:
        # Leave the marker and mailbox immutable for the evidence manifest.
        pass


def _native_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        mode="model_receding",
        source_root=args.wbc_source_root,
        trajectory=args.trajectory,
        arm=args.arm,
        dog=args.dog,
        sim_device=args.sim_device,
        seed=args.seed,
    )


def _case_summary(case: Mapping[str, Any], mailbox: Path) -> Dict[str, Any]:
    request_path = Path(str(case["request_path"]))
    response = dict(case["response"])
    artifact_path = Path(str(response["artifact"]))
    with np.load(case["observation_npz"], allow_pickle=False) as data:
        body = np.asarray(data["body_rgb"], dtype=np.uint8)
        wrist = np.asarray(data["wrist_rgb"], dtype=np.uint8)
        state = np.asarray(data["state"], dtype=np.float32).reshape(-1)
    with np.load(artifact_path, allow_pickle=False) as data:
        raw = np.asarray(data["raw_universal_action"])
    decoded = np.asarray(case["decoded"], dtype=np.float64)
    _require(raw.shape == (1, 40, 132) and np.isfinite(raw).all(), "raw action evidence is not [1,40,132]")
    _require(decoded.shape == (40, 10) and np.isfinite(decoded).all(), "decoded action evidence is not [40,10]")
    _require(state.shape == (10,) and np.isfinite(state).all(), "StateV1 evidence is not finite [10]")
    request = read_json(request_path)
    return {
        "request_id": str(case["request_id"]),
        "request_path": _relative_path(request_path, mailbox.parent),
        "response_path": _relative_path(mailbox / "responses" / (str(case["request_id"]) + ".json"), mailbox.parent),
        "artifact_path": _relative_path(artifact_path, mailbox.parent),
        "request": request,
        "response": response,
        "observation_timestamp": float(case["observation_timestamp"]),
        "body_shape": list(body.shape),
        "body_dtype": str(body.dtype),
        "body_sha256": _array_sha(body),
        "wrist_shape": list(wrist.shape),
        "wrist_dtype": str(wrist.dtype),
        "wrist_sha256": _array_sha(wrist),
        "state_v1": state.tolist(),
        "state_v1_sha256": _array_sha(state),
        "raw_shape": list(raw.shape),
        "raw_sha256": _array_sha(raw),
        "decoded_shape": list(decoded.shape),
        "decoded_sha256": _array_sha(decoded),
        "finite": True,
        "structured_decode": case["structured_decode"],
        "timestamp_contract": case["timestamp_contract"],
        "safety_gate": case["safety_gate"],
        "binding": case.get("binding"),
    }


def _materialize_evidence(
    args: argparse.Namespace,
    phase: str,
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
        with np.load(case["response"]["artifact"], allow_pickle=False) as data:
            raw_values[case_id] = np.asarray(data["raw_universal_action"])
        decoded_values[case_id] = np.asarray(case["decoded"], dtype=np.float64)
    np.savez_compressed(root / "observation_body.npz", **body_values)
    np.savez_compressed(root / "observation_wrist.npz", **wrist_values)
    np.savez_compressed(root / "raw_action_d132.npz", **raw_values)
    np.savez_compressed(root / "decoded_action_d10_h40.npz", **decoded_values)
    _write_json(root / "state_v1.json", {"schema": "CORELAB_STAGE40_STATE_V1_EVIDENCE", "observations": summaries})
    _write_json(root / "worker_request.json", {"schema": "CORELAB_STAGE40_WORKER_REQUEST_EVIDENCE_V1", "requests": [item["request"] for item in summaries]})
    _write_json(root / "worker_response.json", {"schema": "CORELAB_STAGE40_WORKER_RESPONSE_EVIDENCE_V1", "responses": [item["response"] for item in summaries]})
    _write_json(root / "safety_gate.json", {"schema": "CORELAB_STAGE40_SAFETYGATE_EVIDENCE_V1", "decisions": list(safety_decisions), "cases": [item["safety_gate"] for item in summaries]})
    _write_json(root / "mailbox.json", {"schema": "CORELAB_STAGE40_MAILBOX_EVIDENCE_V1", "mailbox": str(mailbox), "requests": sorted(str(p) for p in (mailbox / "requests").glob("*")), "responses": sorted(str(p) for p in (mailbox / "responses").glob("*")), "worker_return_code": worker_return_code})
    _write_json(root / "logs.json", {"schema": "CORELAB_STAGE40_PHASE_LOG_EVIDENCE_V1", "phase": phase, "status": status, "worker_log": str(root / "worker.log"), "error": error})
    _write_json(root / "telemetry.json", {"schema": "CORELAB_STAGE40_TELEMETRY_EVIDENCE_V1", "samples": list(telemetry)})
    _write_json(root / "counter_observations.json", _counter_evidence(phase, native, len(cases), len(handoffs), rows_before, rows_after))
    _write_json(root / "initial_bind.json", {"schema": "CORELAB_STAGE40_INITIAL_BIND_EVIDENCE_V1", "events": list(bind_events)})
    _write_json(root / "root_state.json", {"schema": "CORELAB_STAGE40_ROOT_STATE_EVIDENCE_V1", "events": list(state_events)})
    _write_json(root / "dof_state.json", {"schema": "CORELAB_STAGE40_DOF_STATE_EVIDENCE_V1", "events": list(state_events)})
    _write_json(root / "executor_bookkeeping.json", {"schema": "CORELAB_STAGE40_EXECUTOR_BOOKKEEPING_EVIDENCE_V1", "events": list(executor_events)})
    _write_json(root / "rows_before_handoff.json", {"schema": "CORELAB_STAGE40_ROWS_BEFORE_HANDOFF_EVIDENCE_V1", "segments": list(rows_before)})
    _write_json(root / "rows_after_handoff.json", {"schema": "CORELAB_STAGE40_ROWS_AFTER_HANDOFF_EVIDENCE_V1", "segments": list(rows_after)})
    _write_json(root / "handoff.json", {"schema": "CORELAB_STAGE40_HANDOFF_EVIDENCE_V1", "events": list(handoffs), "completion": completion})
    forbidden = {
        "training": False,
        "formal_seed_used": False,
        "validation_e_read": False,
        "holdout_f_read": False,
        "hardware_executed": False,
        "h200_used": False,
        "checkpoint_substitution": False,
        "model_substitution": False,
        "processor_substitution": False,
        "precision_change": False,
        "quantization": False,
        "cpu_offload": False,
        "clipping": False,
        "smoothing": False,
        "blending": False,
        "adaptive_retiming": False,
        "physics_reset": False,
        "physics_teleport": False,
        "policy_history_clear": False,
    }
    metadata = {
        "schema": "CORELAB_STAGE40_PHASE_RUN_V1",
        "stage": phase,
        "status": status,
        "execution_class": EXECUTION_CLASS[phase],
        "dev_seed": DEV_SEED,
        "seed": int(args.seed),
        "bundle": str(args.bundle),
        "bundle_sha256": sha256_file(args.bundle),
        "predecessor_seal": None if args.predecessor_seal is None else str(args.predecessor_seal),
        "predecessor_seal_sha256": None if args.predecessor_seal is None else sha256_file(args.predecessor_seal),
        "gpu_uuid": os.environ.get("STAGE40_CT3_GPU_UUID", "unknown"),
        "native_control_dt_s": float(native.dt) if native is not None else None,
        "gr00t_forward_count": len(cases),
        "wbc_step_count": int(getattr(native, "stage21_wbc_forward_count", 0)) if native is not None else 0,
        "reference_update_count": int(getattr(native, "stage21_reference_update_count", 0)) if native is not None else 0,
        "worker_return_code": worker_return_code,
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
        "counter_observations": "counter_observations.json",
        "mailbox": "mailbox.json",
        "logs": "logs.json",
        "telemetry": "telemetry.json",
        "run_metadata": "run_metadata.json",
    }
    if phase in {"RTX-bind", "M1", "M2"}:
        roles.update({
            "initial_bind": "initial_bind.json",
            "root_state": "root_state.json",
            "dof_state": "dof_state.json",
            "executor_bookkeeping": "executor_bookkeeping.json",
        })
    if phase in {"M1", "M2"}:
        roles.update({
            "rows_before_handoff": "rows_before_handoff.json",
            "rows_after_handoff": "rows_after_handoff.json",
            "handoff": "handoff.json",
        })
    _write_json(root / "roles.json", roles)
    return {"summaries": summaries, "metadata": metadata, "roles": roles}


def _counter_evidence(
    phase: str,
    native: Any,
    case_count: int,
    handoff_count: int,
    rows_before: Sequence[Any],
    rows_after: Sequence[Any],
) -> Dict[str, Any]:
    forwards = int(case_count)
    steps = int(getattr(native, "stage21_wbc_forward_count", 0)) if native is not None else 0
    updates = int(getattr(native, "stage21_reference_update_count", 0)) if native is not None else 0
    before = {name: 0 for name in COUNTER_NAMES}
    after = {"gr00t_forward_count": forwards, "wbc_step_count": steps, "reference_update_count": updates}
    return build_counter_observation(
        before,
        after,
        gr00t_requests=[1] * forwards,
        wbc_segments=[len(segment) for segment in list(rows_before) + list(rows_after)] if phase in {"M1", "M2"} else [],
        reference_updates=[1] * handoff_count,
    )


def _check_rows(rows: Sequence[Mapping[str, Any]], label: str) -> None:
    _require(len(rows) > 0, label + " produced no WBC rows")
    for row in rows:
        _require(not bool(row.get("done")), label + " reached done/reset")
        _require(not bool(row.get("reset_buf")), label + " reported reset_buf")
        _require(str(row.get("termination_code", "NONE")) == "NONE", label + " reported termination")


def _case_request(
    native: Any,
    mailbox: Path,
    request_id: str,
    seed: int,
    sim_time: float,
    gate: Any,
    adapter: Any,
    args: argparse.Namespace,
) -> Tuple[Any, Dict[str, Any]]:
    chunk, decoded, case = _request_model(
        native=native,
        mailbox=mailbox,
        request_id=request_id,
        seed=seed,
        sim_time=sim_time,
        gate=gate,
        adapter=adapter,
        response_timeout_s=args.response_timeout_s,
        structured_stats=args.structured_stats,
    )
    response = case["response"]
    _require(int(response.get("worker_forward_count_delta", -1)) == 1, request_id + " did not record exactly one worker forward")
    _require(response.get("action_contract") == "STRUCTURED_ACTION_V1", request_id + " action contract mismatch")
    _require(case["safety_gate"]["decision"] in {GateDecision.ACCEPT.value, GateDecision.WARN.value, GateDecision.REJECT.value}, "invalid SafetyGate decision")
    case = dict(case)
    case["decoded"] = np.asarray(decoded, dtype=np.float64)
    return chunk, case


def _initial_bind(
    native: Any,
    env: Any,
    chunk: Any,
    sim_time: float,
    executor_provenance: Mapping[str, Any],
) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    anchor = native.stage21_anchor_world.detach().cpu().numpy().reshape(3)
    current_position, current_quaternion = _current_eef(native)
    before = {
        "root": _tensor_hash(native.root_states),
        "dof": _tensor_hash(native.dof_state),
        "episode": int(native.episode_length_buf[0].detach().cpu().item()),
        "reset_buf": bool(native.reset_buf[0].detach().cpu().item()),
        "arm_history": _tensor_hash(env.arm_obs_history),
        "dog_history": _tensor_hash(env.dog_obs_history),
    }
    bind = native.initial_bind_reference(chunk)
    executor, handoff_steps = _build_stage40_executor(native_dt_s=float(native.dt), provenance=executor_provenance)
    initial_update = executor.submit_initial(
        chunk,
        receive_timestamp=sim_time,
        execute_timestamp=sim_time,
        current_achieved_position=current_position - anchor,
        current_achieved_orientation=current_quaternion,
    )
    after = {
        "root": _tensor_hash(native.root_states),
        "dof": _tensor_hash(native.dof_state),
        "episode": int(native.episode_length_buf[0].detach().cpu().item()),
        "reset_buf": bool(native.reset_buf[0].detach().cpu().item()),
        "arm_history": _tensor_hash(env.arm_obs_history),
        "dog_history": _tensor_hash(env.dog_obs_history),
    }
    event = dict(bind)
    event.update({
        "executor_initial": initial_update.as_dict(),
        "handoff_steps": int(handoff_steps),
        "physics_root_state_unchanged": before["root"] == after["root"],
        "physics_dof_state_unchanged": before["dof"] == after["dof"],
        "episode_length_unchanged": before["episode"] == after["episode"],
        "reset_buf_unchanged": before["reset_buf"] == after["reset_buf"],
        "arm_policy_history_unchanged": before["arm_history"] == after["arm_history"],
        "dog_policy_history_unchanged": before["dog_history"] == after["dog_history"],
        "physics_reset": False,
        "physics_teleport": False,
        "policy_history_reset": False,
    })
    state_event = {"phase": "INITIAL_BIND", "before": before, "after": after, "unchanged": before == after}
    _require(event["physics_root_state_unchanged"] and event["physics_dof_state_unchanged"], "initial bind changed physics state")
    _require(event["episode_length_unchanged"] and event["arm_policy_history_unchanged"] and event["dog_policy_history_unchanged"], "initial bind changed lifecycle state")
    return executor, event, state_event


def _update_reference(
    native: Any,
    env: Any,
    executor: Any,
    previous_chunk: Any,
    next_chunk: Any,
    sim_time: float,
    handoff_steps: int,
    handoff_index: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    current_position, current_quaternion = _current_eef(native)
    anchor = native.stage21_anchor_world.detach().cpu().numpy().reshape(3)
    local_position = current_position - anchor
    # Chunk timestamps are absolute on the paused simulation clock.  At a
    # handoff, compare the previous prediction at the actual switch time;
    # adding the chunk origin again would query a later/clipped point after
    # the first update.
    old_position, old_quaternion = _pose_at(previous_chunk, float(sim_time))
    before = {
        "root": _tensor_hash(native.root_states),
        "dof": _tensor_hash(native.dof_state),
        "episode": int(native.episode_length_buf[0].detach().cpu().item()),
        "reset_buf": bool(native.reset_buf[0].detach().cpu().item()),
        "arm_history": _tensor_hash(env.arm_obs_history),
        "dog_history": _tensor_hash(env.dog_obs_history),
    }
    update = executor.submit_update(
        next_chunk,
        receive_timestamp=sim_time,
        execute_timestamp=sim_time,
        current_achieved_position=local_position,
        current_achieved_orientation=current_quaternion,
        previous_predicted_position=old_position,
        previous_predicted_orientation=old_quaternion,
        executed_prefix_steps=handoff_steps,
    )
    native_update = native.load_reference_chunk(next_chunk)
    after = {
        "root": _tensor_hash(native.root_states),
        "dof": _tensor_hash(native.dof_state),
        "episode": int(native.episode_length_buf[0].detach().cpu().item()),
        "reset_buf": bool(native.reset_buf[0].detach().cpu().item()),
        "arm_history": _tensor_hash(env.arm_obs_history),
        "dog_history": _tensor_hash(env.dog_obs_history),
    }
    event = dict(native_update)
    event.update({
        "handoff_index": int(handoff_index),
        "executor_update": update.as_dict(),
        "actual_native_control_steps": int(handoff_steps),
        "physics_root_state_unchanged": before["root"] == after["root"],
        "physics_dof_state_unchanged": before["dof"] == after["dof"],
        "episode_length_unchanged": before["episode"] == after["episode"],
        "reset_buf_unchanged": before["reset_buf"] == after["reset_buf"],
        "arm_policy_history_unchanged": before["arm_history"] == after["arm_history"],
        "dog_policy_history_unchanged": before["dog_history"] == after["dog_history"],
        "physics_reset": False,
        "physics_teleport": False,
        "policy_history_reset": False,
        "reference_only": True,
    })
    state_event = {"phase": "RUNNING_UPDATE", "handoff_index": int(handoff_index), "before": before, "after": after, "unchanged": before == after}
    _require(event["physics_root_state_unchanged"] and event["physics_dof_state_unchanged"], "reference update changed physics state")
    _require(event["episode_length_unchanged"] and event["arm_policy_history_unchanged"] and event["dog_policy_history_unchanged"], "reference update changed lifecycle state")
    return event, state_event


def _pose_at(chunk: Any, timestamp: float) -> Tuple[np.ndarray, np.ndarray]:
    ts = np.asarray(chunk.point_timestamps, dtype=np.float64)
    index = int(np.clip(np.searchsorted(ts, float(timestamp), side="right") - 1, 0, len(ts) - 1))
    return np.asarray(chunk.eef_position[index], dtype=np.float64), np.asarray(chunk.eef_orientation[index], dtype=np.float64)


def _run_phase(args: argparse.Namespace) -> int:
    phase = str(args.phase)
    root = Path(args.output_dir).resolve()
    _require(not root.exists(), "phase output directory already exists")
    root.mkdir(parents=True)
    mailbox = root / "mailbox"
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
    # Validate the phase bundle and its predecessor entirely before importing
    #/constructing a native simulator.  The caller sets CUDA_VISIBLE_DEVICES
    # to the selected UUID before entering this Python process.
    bundle = read_json(args.bundle)
    validate_bundle(bundle, expected_stage=phase)
    expected_predecessor = PREDECESSOR[phase]
    if expected_predecessor is None:
        _require(args.predecessor_seal is None, "root phase cannot receive predecessor seal")
    else:
        _require(args.predecessor_seal is not None and args.predecessor_evidence_root is not None, "phase predecessor seal/evidence root required")
        # Verify the complete predecessor chain before touching CUDA.  The
        # predecessor seal for RTX-Q0 itself binds the root RTX-runtime seal;
        # later phases nest the same relationship one level at a time under
        # the canonical chain directory.
        predecessor_parent_stage = PREDECESSOR[expected_predecessor]
        predecessor_parent_seal = None
        if predecessor_parent_stage is not None:
            predecessor_parent_root = Path(args.predecessor_evidence_root).resolve().parent / predecessor_parent_stage
            predecessor_parent_seal = predecessor_parent_root / (predecessor_parent_stage + "_result.seal.json")
            _require(predecessor_parent_seal.is_file(), "nested predecessor seal is missing")
        predecessor_info = verify_stage_seal(
            args.predecessor_seal,
            args.predecessor_evidence_root,
            expected_stage=expected_predecessor,
            predecessor_seal=predecessor_parent_seal,
        )
        _require(bundle["predecessor"]["seal_sha256"] == predecessor_info["seal_sha256"], "bundle predecessor seal mismatch")
    gpu_before = _snapshot_gpu(root / "gpu_before.json", "before_phase")
    telemetry.append(gpu_before)
    trajectory = _load_pose_file(args.trajectory)
    try:
        worker = _start_worker(args, mailbox, root / "worker.log")
        native, env, actors, loads, _ = _build_native(_native_args(args), trajectory)
        gate = TrajectorySafetyGateV1()
        adapter = GR00TToWBCAdapter(ReferenceAdapterConfig(expected_source_frame=SOURCE_FRAME, wbc_frame=SOURCE_FRAME, tcp_semantic_id=TCP_SEMANTIC_ID, stale_timeout_s=0.75))
        sim_time = 0.0
        handoff_steps = int(round(HANDOFF_S / float(native.dt)))
        _require(handoff_steps > 0, "handoff schedule produced no native steps")
        current_chunk = None
        executor = None
        initial_bind_event = None
        if phase == "RTX-Q0":
            current_chunk, case = _case_request(native, mailbox, "stage40-q0", args.seed, sim_time, gate, adapter, args)
            cases.append(case)
            _require(case["safety_gate"]["decision"] != GateDecision.REJECT.value, "SAFETYGATE_REJECT_Q0")
            _require(int(native.stage21_wbc_forward_count) == 0 and int(native.stage21_reference_update_count) == 0, "RTX-Q0 performed native execution")
            status = "PASS"
        else:
            current_chunk, case = _case_request(native, mailbox, "stage40-q0", args.seed, sim_time, gate, adapter, args)
            cases.append(case)
            _require(case["safety_gate"]["decision"] != GateDecision.REJECT.value, "SAFETYGATE_REJECT_Q0")
            executor, initial_bind_event, state_event = _initial_bind(native, env, current_chunk, sim_time, {"stage": phase, "mode": "CORELAB_STAGE40"})
            bind_events.append(initial_bind_event)
            state_events.append(state_event)
            executor_events.append(initial_bind_event.get("executor_initial"))
            if phase == "RTX-bind":
                _require(int(native.stage21_wbc_forward_count) == 0 and int(native.stage21_reference_update_count) == 0, "RTX-bind performed rollout/update")
                status = "PASS"
            else:
                handoff_count = 1 if phase == "M1" else 3
                for handoff_index in range(handoff_count):
                    segment, sim_time, termination = _step_rows(native, env, actors, chunk=current_chunk, sim_time=sim_time, steps=handoff_steps, update_count=handoff_index)
                    _check_rows(segment, "pre-handoff segment %d" % handoff_index)
                    rows_before.append(segment)
                    if handoff_index < handoff_count:
                        next_chunk, next_case = _case_request(native, mailbox, "stage40-q%d" % (handoff_index + 1), args.seed, sim_time, gate, adapter, args)
                        cases.append(next_case)
                        _require(next_case["safety_gate"]["decision"] != GateDecision.REJECT.value, "SAFETYGATE_REJECT_Q%d" % (handoff_index + 1))
                        update_event, state_event = _update_reference(native, env, executor, current_chunk, next_chunk, sim_time, handoff_steps, handoff_index + 1)
                        handoffs.append(update_event)
                        state_events.append(state_event)
                        executor_events.append(update_event.get("executor_update"))
                        current_chunk = next_chunk
                final_segment, sim_time, termination = _step_rows(native, env, actors, chunk=current_chunk, sim_time=sim_time, steps=max(1, int(np.ceil(CHUNK_DURATION_S / float(native.dt)))), update_count=handoff_count)
                _check_rows(final_segment, "completion segment")
                rows_after.append(final_segment)
                native_time = float(native.traj_sim_time[0].detach().cpu().item())
                ref_s = float(native.traj_batch.s_ref(native.traj_sim_time)[0].detach().cpu().item())
                ref_L = float(native.traj_batch.L[0].detach().cpu().item())
                ref_duration = float(current_chunk.validity_end - current_chunk.validity_start)
                endpoint = bool(native_time >= ref_duration - max(float(native.dt) * 0.5, 1e-9) and ref_s >= ref_L - max(1e-4, 1e-3 * max(ref_L, 1.0)))
                completion = {"native_time_s": native_time, "reference_duration_s": ref_duration, "traj_s_ref": ref_s, "traj_L": ref_L, "endpoint_reached": endpoint, "termination": termination, "handoff_steps": handoff_steps, "sim_time_s": sim_time}
                _require(endpoint, "reference endpoint was not reached")
                _require(int(native.stage21_reference_update_count) == handoff_count, "reference update count mismatch")
                _require(int(native.stage21_wbc_forward_count) == sum(len(segment) for segment in rows_before + rows_after), "WBC count mismatch")
                _require(len(cases) == handoff_count + 1, "GR00T forward count mismatch")
                status = "PASS"
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        status = "BLOCKED"
    finally:
        try:
            if native is not None:
                telemetry.append(_snapshot_gpu(root / "gpu_after_native.json", "after_native"))
        except Exception as exc:
            error = error or {"type": type(exc).__name__, "message": str(exc)}
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        worker_return_code = _stop_worker(worker, mailbox)
        try:
            telemetry.append(_snapshot_gpu(root / "gpu_after.json", "after_phase"))
        except Exception as exc:
            error = error or {"type": type(exc).__name__, "message": str(exc)}
        try:
            _materialize_evidence(args, phase, root, mailbox, cases, [str(case["safety_gate"]["decision"]) for case in cases], native, rows_before, rows_after, bind_events, handoffs, executor_events, state_events, telemetry, worker_return_code, status, completion, error)
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
            status = "BLOCKED"
            _write_json(root / "phase_error.json", error)
    _write_json(root / "phase_result.json", {"schema": "CORELAB_STAGE40_PHASE_RESULT_V1", "stage": phase, "status": status, "cases": len(cases), "error": error, "completion": completion, "bundle": str(args.bundle), "bundle_sha256": sha256_file(args.bundle)})
    print(json.dumps({"stage": phase, "status": status, "output_dir": str(root), "cases": len(cases), "wbc_steps": int(getattr(native, "stage21_wbc_forward_count", 0)) if native is not None else 0, "reference_updates": int(getattr(native, "stage21_reference_update_count", 0)) if native is not None else 0, "error": error}, sort_keys=True), flush=True)
    return 0 if status == "PASS" else 2


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--predecessor-seal", type=Path)
    parser.add_argument("--predecessor-evidence-root", type=Path)
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
    parser.add_argument("--native-python", type=Path)
    parser.add_argument("--model-python", type=Path, required=True)
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=DEV_SEED)
    parser.add_argument("--response-timeout-s", type=float, default=300.0)
    parser.add_argument("--worker-ready-timeout-s", type=float, default=1200.0)
    args = parser.parse_args(argv)
    if args.seed != DEV_SEED:
        parser.error("Stage40 development seed is fixed at 4242")
    for path in (args.bundle, args.wbc_source_root, args.trajectory, args.arm, args.dog, args.stage10_root, args.base_model_path, args.model_path, args.dataset_dir, args.modality_config, args.cosmos_snapshot, args.structured_stats, args.model_python):
        if not Path(path).exists():
            parser.error("missing required path: %s" % path)
    return _run_phase(args)


if __name__ == "__main__":
    raise SystemExit(main())
