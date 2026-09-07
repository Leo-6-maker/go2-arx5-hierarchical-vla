#!/usr/bin/env python3
"""Stage40-local bounded three-handoff GR00T -> native WBC development run.

The native simulator remains the sole physics owner and the existing
``RecedingHorizonExecutorV1`` remains unchanged.  This file only repeats the
already-proven one-handoff orchestration three times under the frozen
0.9750000000000001-second schedule.  It is intentionally limited to the
development seed and simulation-only execution.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import platform
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from go2arx5_vla.adapters import GR00TToWBCAdapter, ReferenceAdapterConfig
from go2arx5_vla.safety import GateDecision, TrajectorySafetyGateV1

from stage21_native_receding_horizon import (
    SOURCE_FRAME,
    _build_native,
    _current_eef,
    _jsonable,
    _load_pose_file,
    _tensor_hash,
)
from stage40_live_one_handoff import (
    HANDOFF_S,
    PI_GATE,
    _build_stage40_executor,
    _pose_at,
    _request_model,
    _step_rows,
    _write_json,
)
from vanda_stage40_admission import (
    DEV_SEED,
    validate_receding_horizon_lifecycle,
    validate_stage40_receding_horizon_result_schema,
)


STAGE = "Stage40"
TCP_SEMANTIC_ID = "x5_link6_gripper_center"
HANDOFF_COUNT = 3


def _gpu_snapshot() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,memory.used,driver_version,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
    except (FileNotFoundError, OSError) as exc:
        return {"status": "NOT_AVAILABLE", "error": f"{type(exc).__name__}: {exc}", "rows": []}
    if completed.returncode != 0:
        return {
            "status": "QUERY_FAILED",
            "returncode": int(completed.returncode),
            "stderr": completed.stderr.strip(),
            "rows": [],
        }
    rows: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 7:
            continue
        try:
            memory_used = float(fields[4])
        except ValueError:
            memory_used = None
        rows.append({
            "index": fields[0],
            "uuid": fields[1],
            "name": fields[2],
            "memory_total_mib": fields[3],
            "memory_used_mib": memory_used,
            "driver_version": fields[5],
            "compute_capability": fields[6],
        })
    return {"status": "PASS", "query": command, "rows": rows}


def _gpu_identity(snapshot: Mapping[str, Any]) -> str:
    rows = snapshot.get("rows", [])
    return str(rows[0].get("uuid", "UNAVAILABLE")) if rows else "UNAVAILABLE"


def _gpu_used_mib(snapshot: Mapping[str, Any]) -> float | None:
    rows = snapshot.get("rows", [])
    if not rows:
        return None
    value = rows[0].get("memory_used_mib")
    return None if value is None else float(value)


def _clean_rows(rows: list[dict[str, Any]]) -> bool:
    return bool(rows) and all(
        bool(np.isfinite(np.asarray(row.get("achieved_pos", []), dtype=np.float64)).all())
        and not bool(row.get("done"))
        and not bool(row.get("reset_buf"))
        and str(row.get("termination_code", "NONE")) == "NONE"
        for row in rows
    )


def _bundle_identities(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("immutable_inputs", {}).get("rows", [])
    identities: dict[str, str] = {}
    for row in rows:
        name = str(row.get("logical_name", "")).strip()
        identity = row.get("identity")
        if name and identity:
            identities[name] = str(identity)
    if not identities:
        raise ValueError("execution bundle contains no immutable artifact identities")
    return identities


def _git_head(source_root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"unable to resolve execution git HEAD: {completed.stderr.strip()}")
    return completed.stdout.strip()


def run_v6(args: argparse.Namespace, trajectory: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Execute Q0..Q3 with exactly three reference-only updates."""

    native = env = None
    result: dict[str, Any] = {
        "schema": "VANDA_STAGE40_V6_RESULT_V1",
        "stage": STAGE,
        "pi_gate": PI_GATE,
        "mode": "STAGE40_3H_RECEDING_HORIZON_DEVELOPMENT",
        "status": "BLOCKED",
        "seed": int(args.seed),
        "handoff_target_s": HANDOFF_S,
        "handoff_count": HANDOFF_COUNT,
        "forward_count": 0,
        "reference_update_count": 0,
        "executed_prefix_steps": [],
        "cumulative_actual_switch_s": [],
        "cumulative_switch_error_s": [],
        "physics_reset": False,
        "physics_teleport": False,
        "policy_history_reset": False,
        "scientific_rows": 0,
        "simulation_only": True,
        "hardware_control": False,
        "formal_seeds_used": [],
    }
    gpu_baseline = _gpu_snapshot()
    gpu_peak_mib = _gpu_used_mib(gpu_baseline)
    try:
        if int(args.seed) != DEV_SEED:
            raise ValueError(f"V6 requires development seed {DEV_SEED}")
        native_args = argparse.Namespace(**vars(args))
        native_args.mode = "model_receding"
        native_args.seed = DEV_SEED
        native, env, actors, loads, _ = _build_native(native_args, trajectory)
        if actors is None or len(loads) != 2:
            raise RuntimeError("NATIVE_WBC_RUNTIME_NOT_READY")

        gate = TrajectorySafetyGateV1()
        adapter = GR00TToWBCAdapter(ReferenceAdapterConfig(
            expected_source_frame=SOURCE_FRAME,
            wbc_frame=SOURCE_FRAME,
            tcp_semantic_id=TCP_SEMANTIC_ID,
            stale_timeout_s=0.75,
        ))
        native_dt_s = float(native.dt)
        executor, handoff_steps = _build_stage40_executor(
            native_dt_s=native_dt_s,
            provenance={
                "mode": "STAGE40_3H_RECEDING_HORIZON_DEVELOPMENT",
                "handoff_count": HANDOFF_COUNT,
            },
        )
        sim_time = float(native.traj_sim_time[0].detach().cpu().item())
        events: list[str] = []
        model_cases: list[dict[str, Any]] = []
        observations: list[dict[str, Any]] = []
        handoffs: list[dict[str, Any]] = []
        segments: list[dict[str, Any]] = []
        all_rows: list[dict[str, Any]] = []

        events.append("observation_q0")
        q0, _, case0 = _request_model(
            native=native,
            mailbox=args.mailbox,
            request_id="stage40-3h-q0",
            seed=DEV_SEED,
            sim_time=sim_time,
            gate=gate,
            adapter=adapter,
            response_timeout_s=args.response_timeout_s,
            structured_stats=args.structured_stats,
        )
        model_cases.append({key: value for key, value in case0.items() if key not in {"chunk", "decoded"}})
        observations.append({
            "query": "Q0",
            "observation_timestamp_s": case0["observation_timestamp"],
            "body_sha256": case0["body_sha256"],
            "wrist_sha256": case0["wrist_sha256"],
            "state_v1_sha256": case0["state_v1_sha256"],
            "simulator_instance_id": case0["simulator_instance_id"],
        })
        events.extend(["q0_decode", "safety_q0"])
        if case0["safety_gate"]["decision"] == GateDecision.REJECT.value:
            raise RuntimeError("SAFETYGATE_REJECT_Q0")

        before_bind = {
            "root": _tensor_hash(native.root_states),
            "dof": _tensor_hash(native.dof_state),
            "episode": int(native.episode_length_buf[0].detach().cpu().item()),
        }
        events.append("initial_bind")
        initial_bind = native.initial_bind_reference(q0)
        after_bind = {
            "root": _tensor_hash(native.root_states),
            "dof": _tensor_hash(native.dof_state),
            "episode": int(native.episode_length_buf[0].detach().cpu().item()),
        }
        if before_bind != after_bind or not all(bool(initial_bind.get(key)) for key in (
            "physics_root_state_unchanged", "physics_dof_state_unchanged", "episode_length_unchanged",
        )):
            raise RuntimeError("INITIAL_BIND_CHANGED_NATIVE_STATE")
        anchor = native.stage21_anchor_world.detach().cpu().numpy().reshape(3)
        current_position, current_quaternion = _current_eef(native)
        executor.submit_initial(
            q0,
            receive_timestamp=sim_time,
            execute_timestamp=sim_time,
            current_achieved_position=current_position - anchor,
            current_achieved_orientation=current_quaternion,
        )
        current_chunk = q0
        events.append("q0_prefix")

        for handoff_index in range(1, HANDOFF_COUNT + 1):
            events.append(f"handoff_pause_{handoff_index}")
            prefix_started = time.perf_counter()
            root_before_prefix = _tensor_hash(native.root_states)
            dof_before_prefix = _tensor_hash(native.dof_state)
            rows, sim_time, termination = _step_rows(
                native,
                env,
                actors,
                chunk=current_chunk,
                sim_time=sim_time,
                steps=handoff_steps,
                update_count=handoff_index - 1,
            )
            prefix_wall_s = time.perf_counter() - prefix_started
            all_rows.extend(rows)
            root_after_prefix = _tensor_hash(native.root_states)
            dof_after_prefix = _tensor_hash(native.dof_state)
            segment: dict[str, Any] = {
                "handoff_index": handoff_index,
                "executed_prefix_steps": handoff_steps,
                "prefix_wall_s": prefix_wall_s,
                "row_count": len(rows),
                "native_step_count_before_observation": native.stage21_wbc_forward_count,
                "root_before_prefix": root_before_prefix,
                "dof_before_prefix": dof_before_prefix,
                "root_after_prefix": root_after_prefix,
                "dof_after_prefix": dof_after_prefix,
                "simulator_state_evolved": bool(
                    root_before_prefix != root_after_prefix or dof_before_prefix != dof_after_prefix
                ),
                "termination": termination,
            }
            if termination is not None or not _clean_rows(rows):
                raise RuntimeError(f"NATIVE_TERMINATION_BEFORE_HANDOFF_{handoff_index}")
            if not segment["simulator_state_evolved"]:
                raise RuntimeError(f"SIMULATOR_STATE_DID_NOT_EVOLVE_BEFORE_HANDOFF_{handoff_index}")
            segments.append(segment)

            events.append(f"observation_q{handoff_index}")
            request_id = f"stage40-3h-q{handoff_index}"
            next_chunk, _, next_case = _request_model(
                native=native,
                mailbox=args.mailbox,
                request_id=request_id,
                seed=DEV_SEED,
                sim_time=sim_time,
                gate=gate,
                adapter=adapter,
                response_timeout_s=args.response_timeout_s,
                structured_stats=args.structured_stats,
            )
            model_cases.append({key: value for key, value in next_case.items() if key not in {"chunk", "decoded"}})
            observations.append({
                "query": f"Q{handoff_index}",
                "observation_timestamp_s": next_case["observation_timestamp"],
                "body_sha256": next_case["body_sha256"],
                "wrist_sha256": next_case["wrist_sha256"],
                "state_v1_sha256": next_case["state_v1_sha256"],
                "simulator_instance_id": next_case["simulator_instance_id"],
            })
            events.extend([f"q{handoff_index}_decode", f"safety_q{handoff_index}"])
            if next_case["safety_gate"]["decision"] == GateDecision.REJECT.value:
                raise RuntimeError(f"SAFETYGATE_REJECT_Q{handoff_index}")

            current_position, current_quaternion = _current_eef(native)
            local_position = current_position - anchor
            elapsed_since_bind = float(sim_time - current_chunk.observation_timestamp)
            previous_position, previous_quaternion = _pose_at(
                current_chunk,
                float(current_chunk.point_timestamps[0] + elapsed_since_bind),
            )
            root_before_update = _tensor_hash(native.root_states)
            dof_before_update = _tensor_hash(native.dof_state)
            events.append(f"reference_update_{handoff_index}")
            update = executor.submit_update(
                next_chunk,
                receive_timestamp=sim_time,
                execute_timestamp=sim_time,
                current_achieved_position=local_position,
                current_achieved_orientation=current_quaternion,
                previous_predicted_position=previous_position,
                previous_predicted_orientation=previous_quaternion,
                executed_prefix_steps=handoff_steps,
            )
            native_update = native.load_reference_chunk(next_chunk)
            root_after_update = _tensor_hash(native.root_states)
            dof_after_update = _tensor_hash(native.dof_state)
            update_record = {
                **native_update,
                "executor": update.as_dict(),
                "physics_root_state_unchanged": root_before_update == root_after_update,
                "physics_dof_state_unchanged": dof_before_update == dof_after_update,
                "physics_reset": False,
                "physics_teleport": False,
                "policy_history_reset": False,
                "executed_prefix_steps": handoff_steps,
                "handoff_index": handoff_index,
            }
            if not update_record["physics_root_state_unchanged"] or not update_record["physics_dof_state_unchanged"]:
                raise RuntimeError(f"REFERENCE_UPDATE_CHANGED_NATIVE_STATE_{handoff_index}")
            handoffs.append(update_record)
            result["executed_prefix_steps"].append(handoff_steps)
            actual_switch_s = float(sim_time)
            result["cumulative_actual_switch_s"].append(actual_switch_s)
            result["cumulative_switch_error_s"].append(actual_switch_s - handoff_index * HANDOFF_S)
            current_chunk = next_chunk
            events.append(f"q{handoff_index}_continuation")
            phase_gpu = _gpu_used_mib(_gpu_snapshot())
            if phase_gpu is not None:
                gpu_peak_mib = phase_gpu if gpu_peak_mib is None else max(gpu_peak_mib, phase_gpu)

        final_steps = max(1, int(math.ceil(1.95 / native_dt_s)))
        final_rows, sim_time, final_termination = _step_rows(
            native,
            env,
            actors,
            chunk=current_chunk,
            sim_time=sim_time,
            steps=final_steps,
            update_count=HANDOFF_COUNT,
        )
        all_rows.extend(final_rows)
        events.append("q3_completion")
        if final_termination is not None or not _clean_rows(final_rows):
            raise RuntimeError("NATIVE_TERMINATION_DURING_Q3_COMPLETION")
        ref_s = float(native.traj_batch.s_ref(native.traj_sim_time)[0].detach().cpu().item())
        ref_L = float(native.traj_batch.L[0].detach().cpu().item())
        ref_duration = float(current_chunk.validity_end - current_chunk.validity_start)
        endpoint = (
            float(native.traj_sim_time[0].detach().cpu().item())
            >= ref_duration - max(native_dt_s * 0.5, 1e-9)
            and ref_s >= ref_L - max(1e-4, 1e-3 * max(ref_L, 1.0))
        )
        events.append("seal")
        safety_decisions = [case0["safety_gate"]["decision"]] + [
            case["safety_gate"]["decision"] for case in model_cases[1:]
        ]
        validate_receding_horizon_lifecycle(
            events,
            handoff_count=HANDOFF_COUNT,
            handoff_steps=handoff_steps,
            executed_prefix_steps=result["executed_prefix_steps"],
            safety_decisions=safety_decisions,
        )
        result.update({
            "status": "PASS" if endpoint and len(handoffs) == HANDOFF_COUNT else "BLOCKED",
            "handoff_steps": handoff_steps,
            "native_control_dt_s": native_dt_s,
            "forward_count": len(model_cases),
            "reference_update_count": len(handoffs),
            "q0_forward_count": 1,
            "q1_forward_count": 1,
            "q2_forward_count": 1,
            "q3_forward_count": 1,
            "wbc_step_count": native.stage21_wbc_forward_count,
            "scientific_rows": 0,
            "events": events,
            "observations": observations,
            "model_cases": model_cases,
            "handoffs": handoffs,
            "segments": segments,
            "rows": all_rows,
            "final_completion": {
                "native_time_s": float(native.traj_sim_time[0].detach().cpu().item()),
                "reference_duration_s": ref_duration,
                "traj_s_ref": ref_s,
                "traj_L": ref_L,
                "endpoint_reached": bool(endpoint),
            },
            "runtime": {
                "python": platform.python_version(),
                "torch": actors[2].__version__,
                "device": str(actors[3]),
                "simulator_instance_id": native.stage21_simulator_instance_id,
            },
            "artifact_identities": _bundle_identities(args.execution_bundle),
            "runtime_identities": {
                "execution_bundle": str(args.execution_bundle),
                "native_runtime": "IsaacGym/WBC runtime loaded",
                "gr00t_runtime": "Stage22 runtime worker loaded",
            },
            "git_head": _git_head(args.source_root),
            "claim_boundary": "TINY1 single-fixture synchronous four-forward three-handoff development evidence; not real-time, formal, generalization, or hardware",
        })
        gpu_final = _gpu_snapshot()
        gpu_peak_mib = max(value for value in (gpu_peak_mib, _gpu_used_mib(gpu_final)) if value is not None) if any(
            value is not None for value in (gpu_peak_mib, _gpu_used_mib(gpu_final))
        ) else None
        result["gpu"] = {"baseline": gpu_baseline, "final": gpu_final}
        result["gpu_uuid"] = _gpu_identity(gpu_final if gpu_final.get("rows") else gpu_baseline)
        result["vram_baseline_mib"] = _gpu_used_mib(gpu_baseline)
        result["vram_peak_mib"] = gpu_peak_mib
        if result["status"] == "PASS":
            validate_stage40_receding_horizon_result_schema(result)
    except Exception as exc:
        result.update({"status": "BLOCKED", "error": {"type": type(exc).__name__, "message": str(exc)}})
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        if "gpu_uuid" not in result:
            result["gpu_uuid"] = _gpu_identity(gpu_baseline)
        if "vram_baseline_mib" not in result:
            result["vram_baseline_mib"] = _gpu_used_mib(gpu_baseline)
        if "vram_peak_mib" not in result:
            result["vram_peak_mib"] = gpu_peak_mib
        _write_json(args.output_dir / "stage40_v6_result.json", result)
        _write_json(args.mailbox / "V6_RESULT", result)
        _write_json(args.mailbox / "STOP", {"reason": "stage40_v6_finished"})
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--arm", type=Path, required=True)
    parser.add_argument("--dog", type=Path, required=True)
    parser.add_argument("--mailbox", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--structured-stats", type=Path, required=True)
    parser.add_argument("--execution-bundle", type=Path, required=True)
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=DEV_SEED)
    parser.add_argument("--response-timeout-s", type=float, default=300.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.mailbox.mkdir(parents=True, exist_ok=True)
    trajectory = _load_pose_file(args.trajectory)
    result = run_v6(args, trajectory)
    print(json.dumps(_jsonable(result), indent=2), flush=True)
    return 0 if result.get("status") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
