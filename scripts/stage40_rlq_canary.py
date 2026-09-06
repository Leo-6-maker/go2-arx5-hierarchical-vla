#!/usr/bin/env python3
"""Stage40-RLQ bounded native reference lifecycle canary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from stage21_native_receding_horizon import (
    _build_native,
    _chunk_payload_equivalent,
    _jsonable,
    _load_pose_file,
    _make_chunk,
    _policy_step,
    _tensor_hash,
)


def _snap(native, env):
    return {
        "root": _tensor_hash(native.root_states),
        "dof": _tensor_hash(native.dof_state),
        "episode": int(native.episode_length_buf[0].detach().cpu().item()),
        "reset_buf": bool(native.reset_buf[0].detach().cpu().item()),
        "arm_history": _tensor_hash(env.arm_obs_history),
        "dog_history": _tensor_hash(env.dog_obs_history),
        "phase": native.stage21_reference_phase,
        "wbc_forward_count": native.stage21_wbc_forward_count,
    }


def run(args: argparse.Namespace) -> dict:
    result = {
        "schema": "STAGE40_RLQ_CANARY_V1",
        "stage": "Stage40-RLQ",
        "scientific_evidence": False,
        "groot": False,
        "training": False,
        "hardware_control": False,
        "status": "BLOCKED",
        "trajectory": str(args.trajectory),
    }
    native = env = None
    try:
        trajectory = _load_pose_file(args.trajectory)
        native_args = argparse.Namespace(**vars(args))
        native_args.mode = "gt_receding"
        native, env, actors, loads, _ = _build_native(native_args, trajectory)

        installed = native.stage21_current_chunk
        requested = _make_chunk(
            trajectory, chunk_id="rlq-initial", observation_timestamp=0.0, start_time_s=0.0
        )
        payload_equal = _chunk_payload_equivalent(installed, requested)
        if not payload_equal:
            raise RuntimeError("RLQ_INITIAL_PAYLOAD_MISMATCH")

        before_bind = _snap(native, env)
        initial_bind = native.initial_bind_reference(requested)
        after_bind = _snap(native, env)
        bind_ok = (
            initial_bind.get("lifecycle_phase") == "INITIAL_BIND"
            and initial_bind.get("physics_root_state_unchanged") is True
            and initial_bind.get("physics_dof_state_unchanged") is True
            and initial_bind.get("episode_length_unchanged") is True
            and before_bind["root"] == after_bind["root"]
            and before_bind["dof"] == after_bind["dof"]
            and before_bind["episode"] == after_bind["episode"]
            and before_bind["arm_history"] == after_bind["arm_history"]
            and before_bind["dog_history"] == after_bind["dog_history"]
        )
        if not bind_ok:
            raise RuntimeError("RLQ_INITIAL_BIND_INVARIANT_FAILED")

        pre_steps = []
        sim_time = 0.0
        for index in range(5):
            done, policy = _policy_step(native, env, actors)
            sim_time += float(native.dt)
            row = {
                "index": index,
                "done": bool(done),
                "reset_buf": bool(native.reset_buf[0].detach().cpu().item()),
                "sim_time": sim_time,
                "phase": native.stage21_reference_phase,
                "wbc_forward_count": native.stage21_wbc_forward_count,
                "arm_finite": bool(np.isfinite(np.asarray(policy["arm_action_6d"], dtype=np.float64)).all()),
                "dog_finite": bool(np.isfinite(np.asarray(policy["dog_action_12d"], dtype=np.float64)).all()),
            }
            pre_steps.append(row)
            if done:
                raise RuntimeError("RLQ_NATIVE_TASK_TERMINATED_BEFORE_RUNNING_UPDATE")
            if not row["reset_buf"]:
                break
        if not pre_steps or pre_steps[-1]["reset_buf"]:
            raise RuntimeError("RLQ_RESET_BUF_DID_NOT_NATURALLY_CLEAR")

        running_chunk = _make_chunk(
            trajectory,
            chunk_id="rlq-running",
            observation_timestamp=sim_time,
            start_time_s=sim_time,
        )
        before_update = _snap(native, env)
        running_update = native.load_reference_chunk(running_chunk)
        after_update = _snap(native, env)
        update_ok = (
            running_update.get("lifecycle_phase") == "RUNNING_UPDATE"
            and running_update.get("physics_root_state_unchanged") is True
            and running_update.get("physics_dof_state_unchanged") is True
            and running_update.get("episode_length_unchanged") is True
            and before_update["root"] == after_update["root"]
            and before_update["dof"] == after_update["dof"]
            and before_update["episode"] == after_update["episode"]
            and before_update["arm_history"] == after_update["arm_history"]
            and before_update["dog_history"] == after_update["dog_history"]
            and before_update["reset_buf"] is False
        )
        if not update_ok:
            raise RuntimeError("RLQ_RUNNING_UPDATE_INVARIANT_FAILED")

        done_after, policy_after = _policy_step(native, env, actors)
        after_step = _snap(native, env)
        if done_after or after_step["reset_buf"]:
            raise RuntimeError("RLQ_POST_UPDATE_NATIVE_STEP_TERMINATED")
        if not np.isfinite(np.asarray(policy_after["arm_action_6d"], dtype=np.float64)).all():
            raise RuntimeError("RLQ_POST_UPDATE_ARM_NONFINITE")
        if not np.isfinite(np.asarray(policy_after["dog_action_12d"], dtype=np.float64)).all():
            raise RuntimeError("RLQ_POST_UPDATE_DOG_NONFINITE")

        result.update(
            {
                "status": "REFERENCE_LIFECYCLE_REQUALIFIED",
                "payload_equivalent": payload_equal,
                "installed_chunk_id": installed.chunk_id,
                "initial_bind": initial_bind,
                "before_bind": before_bind,
                "after_bind": after_bind,
                "pre_running_steps": pre_steps,
                "running_update": running_update,
                "before_update": before_update,
                "after_update": after_update,
                "post_update_step": {
                    "done": bool(done_after),
                    "reset_buf": after_step["reset_buf"],
                    "phase": after_step["phase"],
                    "wbc_forward_count": after_step["wbc_forward_count"],
                },
                "actor_loads": loads,
                "canary_contract": {
                    "no_physics_reset": True,
                    "no_manual_reset_buf_clear": True,
                    "no_policy_history_clear": True,
                    "no_grroot": True,
                    "no_training": True,
                },
            }
        )
    except Exception as exc:
        result.update(
            {
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "failure_boundary": "RLQ_INITIAL_BIND_OR_RUNNING_UPDATE_CANARY",
            }
        )
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                result["teardown_error"] = {"type": type(exc).__name__, "message": str(exc)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "stage40_rlq_canary.json").write_text(
        json.dumps(_jsonable(result), indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(_jsonable(result), indent=2), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--arm", type=Path, required=True)
    parser.add_argument("--dog", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    return 0 if run(args).get("status") == "REFERENCE_LIFECYCLE_REQUALIFIED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
