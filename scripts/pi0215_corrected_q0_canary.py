#!/usr/bin/env python3
"""PI0215: one exact-lineage, model-free q0 replay canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_HEAD = "a0491ca73cabb82c5c5e0fad67a24423cd67a260"
CELL_ID = "MCTRL_7101/C0_NOMINAL"
SEED = 4242
Q0_TICKS = 49


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_json(value: Any) -> str:
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    return hashlib.sha256(payload).hexdigest()


def tensor_sha(value: Any) -> str:
    array = value.detach().cpu().numpy()
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--arm", type=Path, required=True)
    parser.add_argument("--dog", type=Path, required=True)
    parser.add_argument("--decoded-q0", type=Path, required=True)
    parser.add_argument("--expected-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sim-device", default="cuda:0")
    args = parser.parse_args()

    runtime = args.runtime_root.resolve(strict=True)
    output = args.output_dir.resolve()
    if output.exists():
        raise RuntimeError("refusing to overwrite canary output")
    output.mkdir(parents=True)
    if subprocess.check_output(["git", "-C", str(runtime), "rev-parse", "HEAD"], text=True).strip() != EXPECTED_HEAD:
        raise RuntimeError("runtime root is not exact PI0215 historical HEAD")

    sys.path.insert(0, str(runtime / "src"))
    sys.path.insert(0, str(runtime / "scripts"))
    from go2arx5_vla.providers import GR00TProviderConfig, GR00TTrajectoryProvider
    from stage21_native_model_receding_horizon import _bind_for_native
    from stage21_native_receding_horizon import (
        SOURCE_FRAME,
        TCP_SEMANTIC_ID,
        _build_native,
        _current_eef,
        _load_pose_file,
        _policy_step,
    )
    from stage40_live_one_handoff import _step_rows
    from stage43_corelab_initial_condition import _apply_initial_base_delta, _initial_bind

    required_modules = [
        "scripts/stage43_corelab_initial_condition.py",
        "scripts/stage43_corelab_evidence.py",
        "scripts/stage40_corelab_phase.py",
        "scripts/stage40_live_one_handoff.py",
        "scripts/stage21_native_receding_horizon.py",
        "scripts/native_wbc_benchmark.py",
        "src/go2arx5_vla/stage40_structured_decode.py",
        "src/go2arx5_vla/benchmark/lifecycle.py",
    ]
    module_hashes = {path: sha_file(runtime / path) for path in required_modules}
    expected = json.loads(args.expected_result.read_text())["q0_replay"]["expected_handoff_state"]
    result = {
        "schema": "LOCALPC_PI0215_CORRECTED_Q0_CANARY_V1",
        "status": "BLOCKED",
        "historical_execution_head": EXPECTED_HEAD,
        "cell_id": CELL_ID,
        "seed": SEED,
        "q0_ticks_expected": Q0_TICKS,
        "new_gr00t_model_load": 0,
        "new_gr00t_forward": 0,
        "q1_entered": False,
        "module_hashes": module_hashes,
        "trajectory_sha256": sha_file(args.trajectory),
        "wbc_arm_sha256": sha_file(args.arm),
        "wbc_dog_sha256": sha_file(args.dog),
        "decoded_q0_npz_sha256": sha_file(args.decoded_q0),
        "expected_handoff_state": expected,
    }
    native = env = None
    try:
        native_args = argparse.Namespace(
            mode="gt_receding",
            source_root=args.source_root,
            trajectory=args.trajectory,
            arm=args.arm,
            dog=args.dog,
            sim_device=args.sim_device,
            seed=SEED,
        )
        trajectory = _load_pose_file(args.trajectory)
        native, env, actors, _, _ = _build_native(native_args, trajectory)
        setup = _apply_initial_base_delta(native, env, "C0_NOMINAL", (0.0, 0.0, 0.0))
        with np.load(args.decoded_q0, allow_pickle=False) as data:
            decoded = np.asarray(data["C0_NOMINAL-q0"], dtype=np.float64)
        if decoded.shape != (40, 10) or not np.isfinite(decoded).all():
            raise RuntimeError("sealed q0 decoded action is not finite [40,10]")
        result["decoded_q0_array_sha256"] = hashlib.sha256(np.ascontiguousarray(decoded).tobytes()).hexdigest()
        provider = GR00TTrajectoryProvider(
            decoded[None, ...],
            config=GR00TProviderConfig(
                source_frame=SOURCE_FRAME,
                tcp_semantic_id=TCP_SEMANTIC_ID,
                action_sample_period_s=0.05,
                timestamp_provenance="TEST_ONLY_STAGE21_SIMULATION",
            ),
            chunk_id="C0_NOMINAL-q0",
            observation_timestamp=0.0,
            provenance={"stage": "PI0215", "sealed_reference": True, "model_forward": False},
        )
        chunk = provider.next_chunk(0.0)
        if chunk is None:
            raise RuntimeError("sealed q0 provider emitted no chunk")
        _, bound_chunk = _bind_for_native(chunk)
        _, bind_event, state_event = _initial_bind(
            native,
            env,
            bound_chunk,
            0.0,
            {"stage": "PI0215", "condition_id": "C0_NOMINAL", "mode": "SEALED_Q0_CANARY"},
        )
        rows, sim_time, termination = _step_rows(
            native,
            env,
            actors,
            chunk=bound_chunk,
            sim_time=0.0,
            steps=Q0_TICKS,
            update_count=0,
        )
        observed = {
            "root_tensor_sha256": tensor_sha(native.root_states),
            "dof_tensor_sha256": tensor_sha(native.dof_state),
            "arm_policy_history_sha256": tensor_sha(env.arm_obs_history),
            "dog_policy_history_sha256": tensor_sha(env.dog_obs_history),
            "episode_length_buf": int(native.episode_length_buf[0].detach().cpu().item()),
            "reset_buf": bool(native.reset_buf[0].detach().cpu().item()),
        }
        result.update({
            "status": "PASS" if observed == expected else "FAIL",
            "q0_ticks_observed": len(rows),
            "q0_termination": termination,
            "q0_wbc_forward_count": int(native.stage21_wbc_forward_count),
            "q0_reference_update_count": int(native.stage21_reference_update_count),
            "pre_q0_bind_state_exact": bool(state_event.get("unchanged")),
            "initial_bind_event": bind_event,
            "setup": setup,
            "observed_handoff_state": observed,
            "mismatch_fields": [key for key in expected if observed.get(key) != expected.get(key)],
            "sim_time_s": float(sim_time),
            "native_dt_s": float(native.dt),
            "rows_sha256": sha_json(rows),
            "first_row": rows[0] if rows else None,
            "last_row": rows[-1] if rows else None,
        })
    except Exception as exc:
        result["status"] = "BLOCKED"
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        if native is not None:
            result["wbc_forward_count_final"] = int(native.stage21_wbc_forward_count)
            result["reference_update_count_final"] = int(native.stage21_reference_update_count)
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                result.setdefault("teardown_error", {"type": type(exc).__name__, "message": str(exc)})
    (output / "q0_rows_summary.json").write_text(json.dumps(jsonable(result), sort_keys=True, indent=2) + "\n")
    result["result_sha256"] = sha_json(result)
    (output / "q0_canary_result.json").write_text(json.dumps(jsonable(result), sort_keys=True, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "output_dir": str(output), "result_sha256": result["result_sha256"], "mismatch_fields": result.get("mismatch_fields", []), "error": result.get("error")}, sort_keys=True), flush=True)
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

