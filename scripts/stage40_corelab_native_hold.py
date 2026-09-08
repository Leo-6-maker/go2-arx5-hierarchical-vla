#!/usr/bin/env python3
"""Hold a native Stage40 RTX constructor for the CT3 dual-runtime check.

The process constructs the already-qualified native IsaacGym/WBC environment,
captures one body/wrist observation and StateV1, then waits at a release
barrier.  It never calls the policy-step path, advances physics, replaces a
reference, or imports the GR00T model.  A cutover token is required even when
this file is invoked directly, so a stale shell command cannot start a GPU
lane accidentally.
"""

from __future__ import annotations

# The native runtime has a legacy ``typing.py`` in one optional site path.
# Import the interpreter's standard-library module before adding repository or
# compatibility paths; the launcher deliberately does not prepend that site
# path to PYTHONPATH.
import typing as _stdlib_typing  # noqa: F401

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np


CUTOVER_TOKEN = "VANDA_STAGE40_V2_QUEUE_RELEASED_FOR_CORELAB_TAKEOVER"
RELEASE_TOKEN = "CORELAB_STAGE40_CT3_RELEASE"
DEV_SEED = 4242


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)


def _write_array(path: Path, key: str, value: np.ndarray) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **{key: np.asarray(value)})
    temporary.replace(destination)


def _verify_cutover(token_file: Path, gpu_uuid: str) -> None:
    _require(token_file.is_file() and not token_file.is_symlink(), "cutover token file is missing or symlinked")
    observed = token_file.read_text(encoding="utf-8")
    _require(observed.splitlines() == [CUTOVER_TOKEN], "explicit Vanda cutover token is required")
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == gpu_uuid, "CUDA_VISIBLE_DEVICES must be the selected GPU UUID")


def _construction_trajectory() -> dict[str, np.ndarray]:
    """Return a neutral construction-only chunk source.

    CT3 does not execute or evaluate this trajectory.  Keeping it constant
    avoids borrowing a scientific trajectory or an unsealed model output just
    to satisfy the native constructor's initialization ABI.
    """

    return {
        "timestamp": np.asarray([0.0, 2.0], dtype=np.float64),
        "position": np.asarray([[0.30, 0.10, 0.58], [0.30, 0.10, 0.58]], dtype=np.float64),
        "quaternion": np.asarray([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    }


def _wait_for_release(release_marker: Path, abort_marker: Path, timeout_s: float) -> dict[str, Any]:
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        if abort_marker.exists():
            raise RuntimeError("CT3_ABORT_MARKER_PRESENT")
        if release_marker.is_file():
            payload = json.loads(release_marker.read_text(encoding="utf-8"))
            _require(payload.get("release_token") == RELEASE_TOKEN, "invalid CT3 release marker")
            return payload
        time.sleep(0.25)
    raise TimeoutError("timed out waiting for CT3 release marker")


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ready_marker = Path(args.ready_marker).resolve()
    release_marker = Path(args.release_marker).resolve()
    abort_marker = Path(args.abort_marker).resolve()
    gpu_uuid = str(args.gpu_uuid)
    _require(int(args.seed) == DEV_SEED, "CT3 development seed must be 4242")
    _verify_cutover(Path(args.cutover_token_file).resolve(), gpu_uuid)

    # Importing these helpers is intentionally delayed until all authorization
    # and CLI checks have passed.  No model/runtime helper is imported here.
    sys.path.insert(0, str(Path(args.repo_root).resolve() / "scripts"))
    sys.path.insert(0, str(Path(args.repo_root).resolve() / "src"))
    sys.path.insert(0, str(Path(args.source_root).resolve()))
    from stage21_native_receding_horizon import _build_native, _capture_scene, _current_state_v1, _sha256_array

    native = None
    env = None
    ready_payload: dict[str, Any] = {
        "schema": "CORELAB_STAGE40_CT3_NATIVE_READY_V1",
        "status": "STARTING",
        "stage": "RTX-runtime",
        "execution_class": "CORELAB_STAGE40_RTX_RUNTIME_QUALIFICATION_V1",
        "dev_seed": DEV_SEED,
        "gpu_uuid": gpu_uuid,
        "gr00t_forward_count": 0,
        "wbc_step_count": 0,
        "reference_update_count": 0,
    }
    try:
        native_args = argparse.Namespace(
            source_root=Path(args.source_root).resolve(),
            mode="camera_smoke",
            sim_device=str(args.sim_device),
            seed=DEV_SEED,
            arm=None,
            dog=None,
        )
        native, env, _actors, _loads, cfg = _build_native(native_args, _construction_trajectory())
        body_rgb, wrist_rgb = _capture_scene(native)
        state = _current_state_v1(native)
        state_values = np.asarray(state.as_array(), dtype=np.float32).reshape(-1)
        _require(body_rgb.shape == (96, 96, 3), f"body RGB must be [96,96,3], got {body_rgb.shape}")
        _require(wrist_rgb.shape == (96, 96, 3), f"wrist RGB must be [96,96,3], got {wrist_rgb.shape}")
        _require(state_values.shape == (10,) and np.isfinite(state_values).all(), "StateV1 must be finite [10]")
        _write_array(output_dir / "observation_body.npz", "body_rgb", np.asarray(body_rgb, dtype=np.uint8))
        _write_array(output_dir / "observation_wrist.npz", "wrist_rgb", np.asarray(wrist_rgb, dtype=np.uint8))
        _write_json(
            output_dir / "state_v1.json",
            {
                "schema": "CORELAB_STAGE40_STATE_V1_OBSERVATION_V1",
                "shape": [10],
                "dtype": "float32",
                "values": state_values.tolist(),
                "sha256": _sha256_array(state_values),
                "simulator_instance_id": native.stage21_simulator_instance_id,
                "simulation_time_s": 0.0,
            },
        )
        native_counters = {
            "gr00t_forward_count": {"before": 0, "after": 0, "delta": 0},
            "wbc_step_count": {
                "before": 0,
                "after": int(native.stage21_wbc_forward_count),
                "delta": int(native.stage21_wbc_forward_count),
            },
            "reference_update_count": {
                "before": 0,
                "after": int(native.stage21_reference_update_count),
                "delta": int(native.stage21_reference_update_count),
            },
        }
        _require(native_counters["wbc_step_count"]["delta"] == 0, "native constructor performed a WBC step")
        _require(native_counters["reference_update_count"]["delta"] == 0, "native constructor replaced a reference")
        _write_json(
            output_dir / "native_runtime.json",
            {
                "schema": "CORELAB_STAGE40_CT3_NATIVE_RUNTIME_V1",
                "status": "READY",
                "stage": "RTX-runtime",
                "execution_class": "CORELAB_STAGE40_RTX_RUNTIME_QUALIFICATION_V1",
                "gpu_uuid": gpu_uuid,
                "native_constructor": True,
                "native_python": os.environ.get("STAGE40_NATIVE_PYTHON", "unknown"),
                "native_control_dt_s": float(native.dt),
                "simulator_instance_id": native.stage21_simulator_instance_id,
                "camera_resolution": [int(cfg.env.recording_width_px), int(cfg.env.recording_height_px)],
                "body_shape": list(body_rgb.shape),
                "wrist_shape": list(wrist_rgb.shape),
                "state_v1_shape": list(state_values.shape),
                "body_sha256": _sha256_array(body_rgb),
                "wrist_sha256": _sha256_array(wrist_rgb),
                "state_v1_sha256": _sha256_array(state_values),
                "counters": native_counters,
                "forbidden_operations": {
                    "gr00t_forward": False,
                    "wbc_step": False,
                    "reference_update": False,
                    "physics_reset_after_constructor": False,
                    "physics_teleport": False,
                    "policy_history_clear": False,
                },
                "construction_only_trajectory": True,
            },
        )
        ready_payload.update({
            "status": "READY",
            "native_runtime": str(output_dir / "native_runtime.json"),
            "observation_body": str(output_dir / "observation_body.npz"),
            "observation_wrist": str(output_dir / "observation_wrist.npz"),
            "state_v1": str(output_dir / "state_v1.json"),
            "counters": native_counters,
            "simulator_instance_id": native.stage21_simulator_instance_id,
        })
        _write_json(ready_marker, ready_payload)
        release = _wait_for_release(release_marker, abort_marker, float(args.wait_timeout_s))
        runtime_path = output_dir / "native_runtime.json"
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        runtime.update({"status": "PASS", "released": True, "release": release})
        _write_json(runtime_path, runtime)
        ready_payload.update({"status": "PASS", "released": True})
        _write_json(ready_marker, ready_payload)
        return 0
    except Exception as exc:
        error = {
            "schema": "CORELAB_STAGE40_CT3_NATIVE_RUNTIME_ERROR_V1",
            "status": "ERROR",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "gpu_uuid": gpu_uuid,
            "gr00t_forward_count": 0,
            "wbc_step_count": 0,
            "reference_update_count": 0,
        }
        _write_json(output_dir / "native_runtime_error.json", error)
        ready_payload.update(error)
        try:
            _write_json(ready_marker, ready_payload)
        except Exception:
            pass
        return 2
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ready-marker", type=Path, required=True)
    parser.add_argument("--release-marker", type=Path, required=True)
    parser.add_argument("--abort-marker", type=Path, required=True)
    parser.add_argument("--cutover-token-file", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=DEV_SEED)
    parser.add_argument("--wait-timeout-s", type=float, default=1800.0)
    args = parser.parse_args(argv)
    if args.wait_timeout_s <= 0.0:
        parser.error("--wait-timeout-s must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
