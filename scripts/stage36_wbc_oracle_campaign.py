#!/usr/bin/env python3
"""Execute the pre-admitted Stage36 native WBC campaign.

The CPU authority harness creates a frozen plan.  This runner consumes that
plan inside one PBS allocation and invokes the existing native WBC launcher
once per admitted case.  It has no qsub path, no GR00T path, and no training
path.  A failed allocation preserves its evidence.  A later allocation may
use ``--resume-root`` to carry forward only immutable, finalized rows from a
previous allocation and dispatch missing rows into a fresh output root.
The PBS wrapper may pre-create a strictly limited runtime/log scaffold before
this runner starts; prior campaign output is never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


REPO = Path(__file__).resolve().parents[1]
PI_GATE_ID = "PI_GATE_STATE_0023"
NEXT_GATE = "GPT_PI_REVIEW_AFTER_STAGE36_LONG_HORIZON_LADDER"
EXPECTED_PROJECT = "CFP03-CF-004"
EXPECTED_REFERENCE_MODE = "BENCHMARK_CONTINUOUS"
EXPECTED_SEED = 13
EXPECTED_SECONDS = 2.0
MAX_ROWS = 96
EXPECTED_STATUS = "NATIVE_BENCHMARK_CONTINUOUS_PASS"
POST_RUN_COMPLETION_RETURN_CODE = -11
FINALIZED_NATIVE_STATUSES = {
    "NATIVE_BENCHMARK_CONTINUOUS_PASS",
    "NATIVE_BENCHMARK_CONTINUOUS_PARTIAL",
}
FINALIZED_NATIVE_TERMINATIONS = {
    "REFERENCE_COMPLETE",
    "FALL",
    "INVALID_STATE",
    "JOINT_LIMIT",
    "TRACKING_ABORT",
    "NATIVE_TASK_TERMINATION",
    "UNKNOWN",
}
AUTHORITY_RUN_REL = Path(
    "runs/2026-08-05/"
    "stage2_v3_trajtracking_rl_e2e_gait_wo_postproc_192018"
)
AUTHORITY_EXPECTED = {
    "arm_checkpoint": "99a3a9bda30f7372c7c31240cc3584757ba46cb3b6b9881f00b371906d077a0a",
    "dog_checkpoint": "51776ddfd7177213444c0bf0c8f017b66ffd28a8674458aa0a16c3b9f77cc670",
    "native_config": "56132d4ff9b866ce196bbc1b1353900b967c54f0f48d2d4e4b82c3555769a0bb",
    "native_urdf": "96c1a81e3dd42af5f1bf87ac8958a0aebd7e798ce06408462e0e49a8a2067e25",
    "native_reach_table": "e34ca5c595b30b737209f5dcdd7f3f1e0b95cdf6232fbc0f6f50e3ee7140f5fa",
    "trajectory_module": "8386841b29b58f324cd2e9afa086439add4183992be2466f688382be8ed9d76a",
    "wbc_env": "e39e1bb1dfe4c7a5c2b7309f4f158b6dcf1c9aac91ccbf8a60988ed8a64e992e",
    "legged_robot": "4dc9107af7c1f77dc3b1b686911addf1a1d1a5179dfb194d0b1f4d0bdb8d8a60",
    "wbc_wrapper": "9fa688ce7c3f4665ec48752f0f31639f05ab3309b53d00d95e3d0afd0cb300c7",
}
RUNTIME_CUDNN_LIB = Path(
    "/opt/miniconda3/envs/isaacgym/lib/python3.8/site-packages/"
    "nvidia/cudnn/lib/libcudnn.so.8"
)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_value(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(finite_value(item) for item in value)
    if isinstance(value, dict):
        return all(finite_value(item) for item in value.values())
    return True


def sanitize(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, dict):
        return {str(key): sanitize(item) for key, item in value.items()}
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(sanitize(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def log_event(output_root: Path, event: str, **fields: Any) -> None:
    payload = {"timestamp_utc": now_utc(), "event": event}
    payload.update(fields)
    with (output_root / "stage36_campaign_execution.log").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(sanitize(payload), sort_keys=True, allow_nan=False) + "\n")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_output_root_scaffold(output_root: Path) -> None:
    """Create a fresh output root or validate the PBS wrapper's empty scaffold.

    The wrapper redirects stdout through ``stage36_job.log`` and provisions
    scratch-backed runtime directories before invoking this process.  Keep
    the campaign's fail-closed protection by accepting only that exact
    pre-created shape; a result, row, arbitrary file, or symlink means that
    this is not a fresh allocation root.
    """

    if not output_root.exists():
        output_root.mkdir(parents=True)
        return
    if not output_root.is_dir() or output_root.is_symlink():
        raise SystemExit("FAIL_CLOSED output root is not a real directory: %s" % output_root)

    allowed_entries = {"runtime", "stage36_job.log"}
    entries = {path.name for path in output_root.iterdir()}
    if not entries <= allowed_entries or "runtime" not in entries:
        raise SystemExit("FAIL_CLOSED output scaffold is not fresh: %s" % output_root)

    runtime_root = output_root / "runtime"
    if not runtime_root.is_dir() or runtime_root.is_symlink():
        raise SystemExit("FAIL_CLOSED runtime scaffold is invalid: %s" % runtime_root)
    for path in runtime_root.rglob("*"):
        if path.is_symlink() or path.is_file():
            raise SystemExit("FAIL_CLOSED runtime scaffold contains data: %s" % path)

    log_path = output_root / "stage36_job.log"
    if log_path.exists() and (log_path.is_symlink() or not log_path.is_file()):
        raise SystemExit("FAIL_CLOSED job log scaffold is invalid: %s" % log_path)


def run_gpu_preflight(output_root: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "schema": "STAGE36_GPU_RUNTIME_PREFLIGHT_V1",
        "timestamp_utc": now_utc(),
        "hostname": platform.node(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "nvidia_smi": None,
        "torch_version": None,
        "torch_cuda_version": None,
        "cuda_available": None,
        "device_count": None,
        "device_names": [],
        "status": "BLOCKED",
        "model_constructed": False,
        "wbc_executed": False,
    }
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            capture_output=True,
            check=False,
        )
        names = [line.strip() for line in smi.stdout.splitlines() if line.strip()]
        result["nvidia_smi"] = {
            "returncode": smi.returncode,
            "stdout": smi.stdout.strip(),
            "stderr": smi.stderr.strip(),
        }
        result["device_names"] = names
        import isaacgym  # noqa: F401
        import torch

        result.update(
            {
                "torch_version": str(torch.__version__),
                "torch_cuda_version": str(torch.version.cuda),
                "cuda_available": bool(torch.cuda.is_available()),
                "device_count": int(torch.cuda.device_count()),
            }
        )
        result["status"] = (
            "PASS"
            if smi.returncode == 0
            and len(names) == 1
            and "H200" in names[0]
            and result["cuda_available"]
            and result["device_count"] == 1
            else "BLOCKED_GPU_RUNTIME"
        )
    except Exception as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
    write_json(output_root / "stage36_gpu_preflight.json", result)
    return result


def _resolved_file_record(path: Path, expected_sha256: str, logical_name: str) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "logical_name": logical_name,
        "path": str(path),
        "resolved_path": str(path.resolve()) if path.exists() else None,
        "exists": path.is_file(),
        "is_symlink": path.is_symlink(),
        "expected_sha256": expected_sha256,
        "sha256": None,
        "status": "BLOCKED",
    }
    if path.is_file() and not path.is_symlink():
        record["sha256"] = sha256_file(path)
        record["status"] = "PASS" if record["sha256"] == expected_sha256 else "MISMATCH"
    return record


def runtime_qualification(
    output_root: Path,
    source_root: Path,
    arm: Path,
    dog: Path,
    plan: Mapping[str, Any],
) -> Dict[str, Any]:
    """Qualify the recovered native runtime before the first scientific row.

    This is deliberately not a trajectory rollout: it verifies immutable
    inputs, loads the exact policy weights on CPU, performs repeated zero-input
    policy forwards, builds the exact controller configuration, and constructs
    then closes one native environment without reset() or step().
    """

    result: Dict[str, Any] = {
        "schema": "STAGE36_RUNTIME_QUALIFICATION_V1",
        "timestamp_utc": now_utc(),
        "runtime_identity_class": "RECONSTRUCTED_COMPATIBLE_RUNTIME",
        "source_root": str(source_root),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH", ""),
        "cudnn": {
            "path": str(RUNTIME_CUDNN_LIB),
            "resolved_path": str(RUNTIME_CUDNN_LIB.resolve())
            if RUNTIME_CUDNN_LIB.exists()
            else None,
            "exists": RUNTIME_CUDNN_LIB.is_file(),
            "sha256": sha256_file(RUNTIME_CUDNN_LIB)
            if RUNTIME_CUDNN_LIB.is_file()
            else None,
            "bound_by_library_path": str(RUNTIME_CUDNN_LIB.parent)
            in os.environ.get("LD_LIBRARY_PATH", "").split(":"),
        },
        "components": [],
        "imports": {},
        "policy_load": {},
        "synthetic_smoke": {},
        "config": {},
        "environment_construction": {
            "attempted": False,
            "reset_called": False,
            "step_called": False,
            "status": "NOT_ATTEMPTED",
        },
        "no_hidden_fallback": True,
        "status": "BLOCKED",
        "error": None,
        "scientific_rollout": False,
        "gr00t_model_constructed": False,
    }
    env = None
    try:
        authority_root = source_root.resolve()
        selected_run = authority_root / AUTHORITY_RUN_REL
        selected = {
            "arm_checkpoint": arm,
            "dog_checkpoint": dog,
            "native_config": selected_run / "scripts/config/wbc.py",
            "native_urdf": authority_root / "resources/robots/go2_x5_v3/urdf/go2_x5.urdf",
            "native_reach_table": authority_root / "resources/reach_tables/go2_x5_2d.pt",
            "trajectory_module": authority_root / "modules/trajectory.py",
            "wbc_env": authority_root / "go1_gym/envs/roboduet/wbc_env.py",
            "legged_robot": authority_root / "go1_gym/envs/roboduet/legged_robot.py",
            "wbc_wrapper": authority_root / "go1_gym/envs/roboduet/wbc_env_wrapper.py",
        }
        plan_hashes = plan.get("authority_component_hashes")
        if not isinstance(plan_hashes, dict):
            raise RuntimeError("plan is missing authority_component_hashes")
        if {
            key: plan_hashes.get(key)
            for key in AUTHORITY_EXPECTED
        } != AUTHORITY_EXPECTED:
            raise RuntimeError("plan authority hashes do not match frozen authority identities")
        result["components"] = [
            _resolved_file_record(path, AUTHORITY_EXPECTED[name], name)
            for name, path in selected.items()
        ]
        component_errors = [
            row["logical_name"]
            for row in result["components"]
            if row["status"] != "PASS"
        ]
        if component_errors:
            raise RuntimeError("authority component verification failed: %s" % ",".join(component_errors))
        if arm.resolve() != selected["arm_checkpoint"].resolve() or dog.resolve() != selected["dog_checkpoint"].resolve():
            raise RuntimeError("checkpoint path binding is not the selected authority run")

        if str(REPO / "scripts") not in sys.path:
            sys.path.insert(0, str(REPO / "scripts"))
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))
        # The recovered runtime contract requires IsaacGym before Torch.
        import isaacgym
        import torch
        import torchvision
        import torchaudio
        import gym
        trajectory = importlib.import_module("modules.trajectory")
        reachability = importlib.import_module("modules.reachability")
        config_module = importlib.import_module("go1_gym.envs.config")
        wbc_env_module = importlib.import_module("go1_gym.envs.roboduet.wbc_env")
        legged_robot_module = importlib.import_module("go1_gym.envs.roboduet.legged_robot")
        wrapper_module = importlib.import_module("go1_gym.envs.roboduet.wbc_env_wrapper")
        imported = {
            "isaacgym": isaacgym,
            "torch": torch,
            "torchvision": torchvision,
            "torchaudio": torchaudio,
            "gym": gym,
            "modules.trajectory": trajectory,
            "modules.reachability": reachability,
            "go1_gym.envs.config": config_module,
            "go1_gym.envs.roboduet.wbc_env": wbc_env_module,
            "go1_gym.envs.roboduet.legged_robot": legged_robot_module,
            "go1_gym.envs.roboduet.wbc_env_wrapper": wrapper_module,
        }
        result["imports"] = {
            name: {
                "origin": str(Path(getattr(module, "__file__")).resolve()),
                "under_source_root": name in ("isaacgym", "torch", "torchvision", "torchaudio", "gym")
                or str(Path(getattr(module, "__file__")).resolve()).startswith(str(source_root.resolve()) + "/"),
                "version": getattr(module, "__version__", None),
            }
            for name, module in imported.items()
        }
        source_module_names = (
            "modules.trajectory",
            "modules.reachability",
            "go1_gym.envs.config",
            "go1_gym.envs.roboduet.wbc_env",
            "go1_gym.envs.roboduet.legged_robot",
            "go1_gym.envs.roboduet.wbc_env_wrapper",
        )
        if not all(result["imports"][name]["under_source_root"] for name in source_module_names):
            raise RuntimeError("one or more controller modules resolved outside the exact source root")
        result["versions"] = {
            "torch": str(torch.__version__),
            "torch_cuda": str(torch.version.cuda),
            "torchvision": str(torchvision.__version__),
            "torchaudio": str(torchaudio.__version__),
            "gym": str(gym.__version__),
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
        }
        if not (
            platform.python_version() == "3.8.20"
            and str(torch.__version__).startswith("2.3.1+cu121")
            and str(torch.version.cuda) == "12.1"
            and str(torchvision.__version__).startswith("0.18.1+cu121")
            and str(torchaudio.__version__).startswith("2.3.1+cu121")
            and str(gym.__version__) == "0.23.1"
            and bool(torch.cuda.is_available())
            and int(torch.cuda.device_count()) == 1
            and shutil.which("ninja") is not None
            and result["cudnn"]["exists"]
            and result["cudnn"]["bound_by_library_path"]
        ):
            raise RuntimeError("runtime version/GPU/library contract failed")

        from native_wbc_benchmark import actor_model_classes, configure_eval, safe_load_actor

        ArmActor, DogActor = actor_model_classes()
        arm_model, dog_model = ArmActor(), DogActor()
        result["policy_load"] = {
            "arm": safe_load_actor("arm", arm, arm_model, torch),
            "dog": safe_load_actor("dog", dog, dog_model, torch),
        }
        zero_arm = torch.zeros((1, 12060), dtype=torch.float32)
        zero_dog = torch.zeros((1, 2700), dtype=torch.float32)
        with torch.inference_mode():
            arm_first = arm_model(zero_arm)
            dog_first = dog_model(zero_dog)
            arm_second = arm_model(zero_arm)
            dog_second = dog_model(zero_dog)
        synthetic = {
            "arm_shape": list(arm_first.shape),
            "dog_shape": list(dog_first.shape),
            "finite": bool(
                torch.isfinite(arm_first).all()
                and torch.isfinite(dog_first).all()
                and torch.isfinite(arm_second).all()
                and torch.isfinite(dog_second).all()
            ),
            "deterministic_exact": bool(
                torch.equal(arm_first, arm_second) and torch.equal(dog_first, dog_second)
            ),
            "tolerance": 0.0,
            "arm_first_sha256": hashlib.sha256(arm_first.detach().numpy().tobytes()).hexdigest(),
            "dog_first_sha256": hashlib.sha256(dog_first.detach().numpy().tobytes()).hexdigest(),
            "arm_second_sha256": hashlib.sha256(arm_second.detach().numpy().tobytes()).hexdigest(),
            "dog_second_sha256": hashlib.sha256(dog_second.detach().numpy().tobytes()).hexdigest(),
        }
        result["synthetic_smoke"] = synthetic
        if not synthetic["finite"] or not synthetic["deterministic_exact"]:
            raise RuntimeError("synthetic policy smoke failed")

        from go1_gym import MINI_GYM_ROOT_DIR
        from go1_gym.envs.config import RoboDuetRuntimeOptions, build_roboduet_config
        from go1_gym.utils.global_switch import global_switch

        options = RoboDuetRuntimeOptions(
            num_envs=1,
            robot="go2_x5",
            use_rot6d=True,
            dyna_gait=True,
            goal_reaching=True,
            traj_tracking=True,
            arm_action_mode="end_to_end",
            reach_table=True,
        )
        cfg = build_roboduet_config(options=options, debug=True)
        configure_eval(cfg)
        asset_path = Path(cfg.asset.file.format(MINI_GYM_ROOT_DIR=MINI_GYM_ROOT_DIR)).resolve()
        reach_path = Path(cfg.wbc.goal_reaching.reach_table_path.format(MINI_GYM_ROOT_DIR=MINI_GYM_ROOT_DIR)).resolve()
        result["config"] = {
            "status": "PASS",
            "asset_path": str(asset_path),
            "reach_table_path": str(reach_path),
            "asset_hash": sha256_file(asset_path) if asset_path.is_file() else None,
            "reach_table_hash": sha256_file(reach_path) if reach_path.is_file() else None,
            "asset_matches_authority": asset_path == selected["native_urdf"].resolve(),
            "reach_table_matches_authority": reach_path == selected["native_reach_table"].resolve(),
        }
        if not (
            result["config"]["asset_matches_authority"]
            and result["config"]["reach_table_matches_authority"]
            and result["config"]["asset_hash"] == AUTHORITY_EXPECTED["native_urdf"]
            and result["config"]["reach_table_hash"] == AUTHORITY_EXPECTED["native_reach_table"]
        ):
            raise RuntimeError("config asset/reach-table binding failed")

        global_switch.open_switch()
        result["environment_construction"]["attempted"] = True
        env = wbc_env_module.WBCEnv(
            sim_device="cuda:0",
            headless=True,
            num_envs=1,
            cfg=cfg,
            graphics_device_id=0,
        )
        result["environment_construction"].update(
            {
                "status": "PASS",
                "class": type(env).__name__,
                "device": str(env.device),
                "num_envs": int(env.num_envs),
                "reach_table_loaded": getattr(env, "reach_table", None) is not None,
            }
        )
        if not result["environment_construction"]["reach_table_loaded"]:
            raise RuntimeError("environment silently fell back without native reach table")
        result["status"] = "PASS"
    except Exception as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                result["environment_construction"]["close_error"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
                if result["status"] == "PASS":
                    result["status"] = "BLOCKED"
    write_json(output_root / "stage36_runtime_qualification.json", result)
    return result


def validate_plan(plan: Mapping[str, Any], source_root: Path) -> List[str]:
    errors: List[str] = []
    if plan.get("schema") != "STAGE36_WBC_ORACLE_CAMPAIGN_PLAN_V3":
        errors.append("plan_schema")
    if plan.get("pi_gate") != PI_GATE_ID:
        errors.append("pi_gate")
    if plan.get("plan_status") != "READY_FOR_E":
        errors.append("plan_not_ready")
    if plan.get("project") != EXPECTED_PROJECT or plan.get("queue") != "auto":
        errors.append("project_or_queue")
    if plan.get("one_pbs_allocation_only") is not True:
        errors.append("allocation_policy")
    rows = plan.get("rows")
    if not isinstance(rows, list) or not rows:
        errors.append("rows_missing")
        return errors
    if len(rows) > MAX_ROWS:
        errors.append("row_budget")
    representations = plan.get("representations")
    if not isinstance(representations, list) or not representations:
        errors.append("representations_missing")
        representations = []
    if len(set(str(value) for value in representations)) != len(representations):
        errors.append("duplicate_representations")
    if "DENSE" not in representations:
        errors.append("dense_representation_missing")
    allowed_representations = set(str(value) for value in representations)
    row_ids = []
    saw_robustness = False
    for row in rows:
        row_ids.append(row.get("row_id"))
        representation = str(row.get("representation"))
        if representation not in allowed_representations:
            errors.append("representation:%s" % row.get("row_id"))
        if representation == "DENSE":
            if row.get("geometry_k") is not None or row.get("timing_mode") is not None:
                errors.append("dense_candidate_fields:%s" % row.get("row_id"))
        elif not (
            representation.startswith("K")
            and "+T" in representation
            and row.get("geometry_k") in (4, 6, 9, 12)
            and row.get("timing_mode") in ("T0", "T1", "T3", "T5", "T9")
            and representation
            == "K%d+%s" % (int(row.get("geometry_k")), str(row.get("timing_mode")))
        ):
            errors.append("structured_representation:%s" % row.get("row_id"))
        if row.get("reference_mode") != EXPECTED_REFERENCE_MODE:
            errors.append("reference_mode:%s" % row.get("row_id"))
        phase = row.get("phase")
        if phase == "PRIMARY" and saw_robustness:
            errors.append("primary_after_robustness:%s" % row.get("row_id"))
        if phase == "ROBUSTNESS":
            saw_robustness = True
        if phase == "PRIMARY" and row.get("seed") != EXPECTED_SEED:
            errors.append("primary_seed:%s" % row.get("row_id"))
        if phase == "ROBUSTNESS" and row.get("seed") not in (14, 15):
            errors.append("robustness_seed:%s" % row.get("row_id"))
        if phase not in ("PRIMARY", "ROBUSTNESS"):
            errors.append("phase:%s" % row.get("row_id"))
        if float(row.get("seconds", -1.0)) != EXPECTED_SECONDS:
            errors.append("duration:%s" % row.get("row_id"))
        trajectory = Path(str(row.get("trajectory_path", "")))
        metadata = Path(str(row.get("metadata_path", "")))
        if not trajectory.is_file() or not metadata.is_file():
            errors.append("missing_input:%s" % row.get("row_id"))
        if not str(trajectory).startswith("/scratch/") or not str(metadata).startswith("/scratch/"):
            errors.append("non_scratch_input:%s" % row.get("row_id"))
        if trajectory.is_file() and row.get("trajectory_sha256") != sha256_file(trajectory):
            errors.append("trajectory_hash:%s" % row.get("row_id"))
        if metadata.is_file() and row.get("metadata_sha256") != sha256_file(metadata):
            errors.append("metadata_hash:%s" % row.get("row_id"))
        if not source_root.is_dir():
            errors.append("source_root_missing")
    if len(set(row_ids)) != len(row_ids):
        errors.append("duplicate_row_id")
    keys = [(row.get("case_id"), row.get("representation"), row.get("seed"), row.get("phase")) for row in rows]
    if len(set(keys)) != len(keys):
        errors.append("duplicate_case_representation_seed")
    if int(plan.get("sparse_rows", 0)) != sum(
        row.get("representation") != "DENSE" for row in rows
    ):
        errors.append("sparse_rows_count")
    if int(plan.get("row_count", -1)) != len(rows):
        errors.append("row_count")
    if int(plan.get("primary_row_count", -1)) != sum(
        row.get("phase") == "PRIMARY" for row in rows
    ):
        errors.append("primary_row_count")
    if int(plan.get("robustness_row_count", -1)) != sum(
        row.get("phase") == "ROBUSTNESS" for row in rows
    ):
        errors.append("robustness_row_count")
    native_transport = plan.get("native_transport", {})
    if (
        native_transport.get("sample_count") != 121
        or float(native_transport.get("duration_s", -1.0)) != 2.0
        or not native_transport.get("shared_across_representations", False)
    ):
        errors.append("native_transport_contract")
    return sorted(set(errors))


def native_command(
    runner: Path,
    source_root: Path,
    arm: Path,
    dog: Path,
    trajectory: Path,
    output_dir: Path,
    metadata: Path,
    seed: int,
    seconds: float,
) -> List[str]:
    # Keep this list literal and auditable.  There is exactly one permitted
    # reference mode, and no diagnostic termination or training switches are
    # ever appended by this runner.
    return [
        sys.executable,
        str(runner),
        "--source-root",
        str(source_root),
        "--arm",
        str(arm),
        "--dog",
        str(dog),
        "--reference-mode",
        EXPECTED_REFERENCE_MODE,
        "--trajectory",
        str(trajectory),
        "--output-dir",
        str(output_dir),
        "--seconds",
        str(seconds),
        "--seed",
        str(seed),
        "--metadata-json",
        str(metadata),
    ]


def read_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def finalized_native_output(
    summary: Mapping[str, Any], raw_rows: Sequence[Mapping[str, Any]], timed_out: bool
) -> bool:
    """Recognize a complete native result without treating a crash as a pass.

    IsaacGym can emit all native artifacts and the final JSON before the
    process dies during native-library teardown.  A non-zero exit is retained
    in provenance, but the row is resumable as completed only when the native
    summary, terminal state, raw rows, and finite-value checks are all intact.
    A timeout or an exception without a terminal native result remains a
    missing/incomplete row.
    """

    if timed_out or not isinstance(summary, Mapping) or not raw_rows:
        return False
    if summary.get("status") not in FINALIZED_NATIVE_STATUSES:
        return False
    if summary.get("error") is not None:
        return False
    if str(summary.get("termination")) not in FINALIZED_NATIVE_TERMINATIONS:
        return False
    if int(summary.get("steps", -1)) != len(raw_rows):
        return False
    if not finite_value(summary) or not finite_value(list(raw_rows)):
        return False
    return bool(
        summary.get("reference_completed", False)
        or str(summary.get("termination")) in FINALIZED_NATIVE_TERMINATIONS
    )


RESUME_PLAN_IGNORED_KEYS = {
    "timestamp_utc",
    "accepted_start_head",
    "execution_head",
    "native_runner_path",
    "native_runner_sha256",
    "pbs_wrapper_path",
    "pbs_wrapper_sha256",
    "authority_manifest_path",
    "authority_runtime_contract_path",
    "authority_tree_manifest_path",
}


def resume_plan_projection(value: Any, key: Optional[str] = None) -> Any:
    """Project away allocation/source identity while retaining science."""

    if isinstance(value, Mapping):
        return {
            str(name): resume_plan_projection(item, str(name))
            for name, item in value.items()
            if str(name) not in RESUME_PLAN_IGNORED_KEYS
        }
    if isinstance(value, list):
        return [resume_plan_projection(item) for item in value]
    return value


def resume_plan_fingerprint(plan: Mapping[str, Any]) -> str:
    payload = json.dumps(
        resume_plan_projection(plan), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resume_row_matches_plan(
    row: Mapping[str, Any], plan_row: Mapping[str, Any]
) -> bool:
    fields = (
        "row_id",
        "case_id",
        "family",
        "representation",
        "geometry_k",
        "timing_mode",
        "phase",
        "seed",
        "reference_mode",
        "seconds",
        "trajectory_path",
        "trajectory_sha256",
    )
    # Older campaign result rows did not repeat every immutable plan field
    # (notably ``seconds`` and the input hash).  The prior plan snapshot is
    # fingerprinted separately, so compare fields that the row actually
    # records and enrich the carried row from the current identical plan.
    return all(field not in row or row.get(field) == plan_row.get(field) for field in fields)


def load_resume_rows(
    resume_root: Path, output_root: Path, plan: Mapping[str, Any]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load only immutable finalized rows from a prior allocation.

    The prior root remains read-only.  Its raw/summary paths are retained in
    the new result and sealed as inherited evidence; no old row directory is
    copied, changed, or rerun.
    """

    resume_root = resume_root.absolute()
    output_root = output_root.absolute()
    if resume_root == output_root:
        raise RuntimeError("resume root must differ from current output root")
    if not resume_root.is_dir() or resume_root.is_symlink():
        raise RuntimeError("resume root is not a real directory")
    required = (
        resume_root / "stage36_result.json",
        resume_root / "stage36_result_manifest.json",
        resume_root / "stage36_raw_hash_manifest.json",
        resume_root / "stage36_artifact_seal.json",
        resume_root / "stage36_return.json",
        resume_root / "stage36_campaign_plan_snapshot.json",
    )
    missing = [str(path) for path in required if not path.is_file() or path.is_symlink()]
    if missing:
        raise RuntimeError("resume root missing immutable evidence: %s" % ",".join(missing))

    prior_result = load_json(resume_root / "stage36_result.json")
    prior_manifest = load_json(resume_root / "stage36_result_manifest.json")
    prior_raw_manifest = load_json(resume_root / "stage36_raw_hash_manifest.json")
    prior_return = load_json(resume_root / "stage36_return.json")
    prior_plan = load_json(resume_root / "stage36_campaign_plan_snapshot.json")
    if prior_result.get("schema") != "STAGE36_WBC_ORACLE_RESULT_V2":
        raise RuntimeError("resume result schema is not Stage36 V2")
    if prior_result.get("pi_gate") != PI_GATE_ID:
        raise RuntimeError("resume result PI gate mismatch")
    if prior_result.get("project") not in (None, EXPECTED_PROJECT):
        raise RuntimeError("resume result project mismatch")
    if resume_plan_fingerprint(prior_plan) != resume_plan_fingerprint(plan):
        raise RuntimeError("resume scientific plan mismatch")
    expected_result_sha = prior_manifest.get("result_sha256")
    if expected_result_sha and sha256_file(resume_root / "stage36_result.json") != expected_result_sha:
        raise RuntimeError("resume result manifest hash mismatch")
    expected_seal_sha = prior_return.get("ARTIFACT_SEAL_SHA256")
    if expected_seal_sha and sha256_file(resume_root / "stage36_artifact_seal.json") != expected_seal_sha:
        raise RuntimeError("resume artifact seal hash mismatch")

    plan_rows = {str(row.get("row_id")): row for row in plan.get("rows", [])}
    manifest_rows = {
        str(row.get("row_id")): row for row in prior_raw_manifest.get("rows", [])
    }
    prior_rows = prior_result.get("rows", [])
    if not isinstance(prior_rows, list):
        raise RuntimeError("resume result rows are not a list")
    if len({str(row.get("row_id")) for row in prior_rows}) != len(prior_rows):
        raise RuntimeError("resume result contains duplicate row IDs")

    inherited: List[Dict[str, Any]] = []
    for previous in prior_rows:
        row_id = str(previous.get("row_id"))
        plan_row = plan_rows.get(row_id)
        if plan_row is None or not _resume_row_matches_plan(previous, plan_row):
            raise RuntimeError("resume row does not match frozen plan: %s" % row_id)
        output_dir = Path(str(previous.get("output_dir", "")))
        raw_path = Path(str(previous.get("raw_rows_path", "")))
        summary_path = output_dir / "native_external_summary.json"
        if (
            not str(output_dir).startswith("/scratch/")
            or not output_dir.is_dir()
            or output_dir.is_symlink()
            or raw_path.is_symlink()
            or summary_path.is_symlink()
            or not raw_path.is_file()
            or not summary_path.is_file()
        ):
            raise RuntimeError("resume row evidence path is unsafe or missing: %s" % row_id)
        manifest_row = manifest_rows.get(row_id)
        if manifest_row is None:
            raise RuntimeError("resume raw manifest is missing row: %s" % row_id)
        raw_sha = sha256_file(raw_path)
        summary_sha = sha256_file(summary_path)
        if raw_sha != manifest_row.get("raw_rows_sha256") or summary_sha != manifest_row.get("summary_sha256"):
            raise RuntimeError("resume row hash mismatch: %s" % row_id)
        summary = load_json(summary_path)
        raw_rows = read_rows(raw_path)
        if not finalized_native_output(summary, raw_rows, bool(previous.get("timed_out", False))):
            raise RuntimeError("resume row is not finalized: %s" % row_id)
        # Recompute the row wrapper fields from the sealed raw/summary files.
        # This upgrades the old pre-resume bookkeeping for the observed
        # post-completion SIGSEGV without changing any scientific values.
        carried = row_result(
            plan_row,
            output_dir,
            int(previous.get("returncode", 2)),
            bool(previous.get("timed_out", False)),
        )
        for field in (
            "seconds",
            "trajectory_path",
            "trajectory_sha256",
            "metadata_path",
            "metadata_sha256",
        ):
            carried.setdefault(field, plan_row.get(field))
        carried.update(
            {
                "scientific_row_finalized": True,
                "resume_source_root": str(resume_root),
                "resume_source_output_dir": str(output_dir),
                "resume_source_raw_rows_sha256": raw_sha,
                "resume_source_summary_sha256": summary_sha,
            }
        )
        inherited.append(carried)
    return inherited, {
        "resume_root": str(resume_root),
        "resume_result_sha256": sha256_file(resume_root / "stage36_result.json"),
        "resume_artifact_seal_sha256": sha256_file(resume_root / "stage36_artifact_seal.json"),
        "inherited_row_count": len(inherited),
        "inherited_row_ids": [str(row.get("row_id")) for row in inherited],
    }


def safety_profile(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    transitions: List[Dict[str, Any]] = []
    previous: Optional[tuple] = None
    for row in rows:
        state = (
            row.get("termination_code"),
            bool(row.get("reference_complete", False)),
            bool(row.get("fall_flag", False)),
            bool(row.get("invalid_state", False)),
            bool(row.get("joint_limit_violation", False)),
        )
        if state != previous:
            transitions.append({"step": row.get("step"), "state": list(state)})
            previous = state
    violation = any(
        bool(row.get(name, False))
        for row in rows
        for name in ("fall_flag", "invalid_state", "joint_limit_violation", "joint_limit_termination")
    )
    finite = all(finite_value(dict(row)) for row in rows)
    return {
        "finite": finite,
        "safety_violation": violation,
        "transitions": transitions,
        "termination_codes": sorted({str(row.get("termination_code", "UNKNOWN")) for row in rows}),
        "row_count": len(rows),
        "reference_complete_seen": any(bool(row.get("reference_complete", False)) for row in rows),
    }


def row_result(plan_row: Mapping[str, Any], output_dir: Path, returncode: int, timed_out: bool) -> Dict[str, Any]:
    summary_path = output_dir / "native_external_summary.json"
    raw_rows_path = output_dir / "native_external_rows.jsonl"
    summary = load_json(summary_path) if summary_path.is_file() else {}
    raw_rows = read_rows(raw_rows_path)
    finite = finite_value(summary) and finite_value(raw_rows)
    safety = safety_profile(raw_rows)
    finalized = finalized_native_output(summary, raw_rows, timed_out)
    post_run_completion_signal = bool(
        finalized and returncode == POST_RUN_COMPLETION_RETURN_CODE
    )
    runner_pass = bool(
        finalized
        and finite
        and safety["finite"]
        and (returncode == 0 or post_run_completion_signal)
    )
    complete = bool(summary.get("reference_completed", False) or safety["reference_complete_seen"])
    result = {
        "row_id": plan_row["row_id"],
        "case_id": plan_row["case_id"],
        "family": plan_row["family"],
        "representation": plan_row["representation"],
        "geometry_k": plan_row.get("geometry_k"),
        "timing_mode": plan_row.get("timing_mode"),
        "phase": plan_row["phase"],
        "seed": plan_row["seed"],
        "reference_mode": plan_row["reference_mode"],
        "trajectory_path": plan_row["trajectory_path"],
        "trajectory_sha256_planned": plan_row.get("trajectory_sha256"),
        "output_dir": str(output_dir),
        "returncode": returncode,
        "timed_out": timed_out,
        "native_status": summary.get("status"),
        "native_termination": summary.get("termination"),
        "native_summary": sanitize(summary),
        "raw_rows_path": str(raw_rows_path),
        "raw_rows_count": len(raw_rows),
        "finite": finite and safety["finite"],
        "finite_flags": {
            "summary": finite_value(summary),
            "raw_rows": finite_value(raw_rows),
            "metrics": finite_value(summary.get("metrics", {})),
        },
        "safety_gate": safety,
        "reference_complete": complete,
        "scientific_row_finalized": finalized,
        "post_run_completion_signal": post_run_completion_signal,
        "runner_pass": runner_pass,
        "tracking_pass": bool(runner_pass and complete and not safety["safety_violation"]),
    }
    metrics = summary.get("metrics", {})
    result["profile_metrics"] = sanitize(metrics)
    if raw_rows:
        base = [row.get("base_pos", []) for row in raw_rows]
        try:
            import numpy as np

            base_array = np.asarray(base, dtype=np.float64)
            result["base_participation"] = {
                "finite": bool(np.isfinite(base_array).all()),
                "x_span_m": float(np.ptp(base_array[:, 0])) if len(base_array) else 0.0,
                "path_length_m": float(np.linalg.norm(np.diff(base_array, axis=0), axis=1).sum()) if len(base_array) > 1 else 0.0,
                "nontrivial_observed": bool(len(base_array) > 1 and np.ptp(base_array[:, 0]) > 1e-3),
            }
        except Exception as exc:
            result["base_participation"] = {"status": "UNAVAILABLE", "error": str(exc)}
    else:
        result["base_participation"] = {"status": "NO_ROWS"}
    return result


def aggregate_metric(rows: Sequence[Mapping[str, Any]], metric_path: Sequence[str]) -> Optional[float]:
    values: List[float] = []
    for row in rows:
        value: Any = row.get("profile_metrics", {})
        for key in metric_path:
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return None if not values else float(sum(values) / len(values))


def profile_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "row_count": len(rows),
        "runner_pass_count": sum(bool(row.get("runner_pass")) for row in rows),
        "tracking_pass_count": sum(bool(row.get("tracking_pass")) for row in rows),
        "all_finite": bool(rows) and all(bool(row.get("finite")) for row in rows),
        "position_rmse_mean_m": aggregate_metric(rows, ("position_m", "rmse")),
        "position_p95_mean_m": aggregate_metric(rows, ("position_m", "p95")),
        "rotation_rmse_mean_rad": aggregate_metric(rows, ("rotation_rad", "rmse")),
        "rotation_p95_mean_rad": aggregate_metric(rows, ("rotation_rad", "p95")),
        "time_in_tolerance_mean": aggregate_metric(rows, ("time_in_tolerance_fraction_0.05m_0.20rad",)),
        "base_participation_x_span_mean_m": aggregate_metric(rows, ("base_participation", "x_span_m")),
    }


def seal_entries(output_root: Path, exclude: Iterable[str]) -> List[Dict[str, Any]]:
    excluded = set(exclude)
    entries: List[Dict[str, Any]] = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name not in excluded:
            entries.append(
                {
                    "path": str(path),
                    "relative_path": path.relative_to(output_root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return entries


def inherited_seal_entries(
    rows: Sequence[Mapping[str, Any]], output_root: Path
) -> List[Dict[str, Any]]:
    """Seal the prior row files referenced by a resume result."""

    entries: List[Dict[str, Any]] = []
    seen = set()
    output_root = output_root.absolute()
    for row in rows:
        source_dir = Path(str(row.get("resume_source_output_dir", "")))
        if not source_dir or source_dir == output_root or not source_dir.is_dir():
            continue
        row_id = str(row.get("row_id"))
        for name in (
            "native_external_rollout.npz",
            "native_external_rows.jsonl",
            "native_external_summary.json",
            "run.log",
        ):
            path = source_dir / name
            key = str(path)
            if key in seen or not path.is_file() or path.is_symlink():
                continue
            seen.add(key)
            entries.append(
                {
                    "path": key,
                    "relative_path": "inherited/%s/%s" % (row_id, name),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "source": "resume_root",
                }
            )
    return entries


def make_return(
    output_root: Path,
    result: Mapping[str, Any],
    seal_path: Path,
    seal_sha256: Optional[str],
    status: str,
    error: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    runtime_result = result.get("runtime_qualification")
    environment_construction = (
        runtime_result.get("environment_construction")
        if isinstance(runtime_result, Mapping)
        else None
    )
    wbc_model_constructed = bool(
        isinstance(runtime_result, Mapping)
        and (
            bool(runtime_result.get("policy_load"))
            or (
                isinstance(environment_construction, Mapping)
                and bool(environment_construction.get("attempted"))
            )
        )
    )
    return {
        "ROLE": "CODEX_HOPPER_SERVER_LONG_HORIZON",
        "ACTIVE_PI_GATE": PI_GATE_ID,
        "ACTIVE_STAGE": "Stage36-RQ",
        "PR": 22,
        "PBS_JOB_ID": os.environ.get("PBS_JOBID"),
        "PBS_PROJECT": EXPECTED_PROJECT,
        "RESULT": status,
        "NEXT_GATE": NEXT_GATE,
        "ARTIFACT_ROOT": str(output_root),
        "ARTIFACT_SEAL_PATH": str(seal_path),
        "ARTIFACT_SEAL_SHA256": seal_sha256,
        "DETAIL_OUTPUTS": {
            "gpu_preflight": str(output_root / "stage36_gpu_preflight.json"),
            "runtime_qualification": str(output_root / "stage36_runtime_qualification.json"),
            "campaign_plan": str(output_root / "stage36_campaign_plan_snapshot.json"),
            "result": str(output_root / "stage36_result.json"),
            "raw_hash_manifest": str(output_root / "stage36_raw_hash_manifest.json"),
            "result_manifest": str(output_root / "stage36_result_manifest.json"),
            "artifact_seal": str(seal_path),
        },
        "PBS_SUBMITTED": True,
        "GPU_USED": True,
        "MODEL_CONSTRUCTED": wbc_model_constructed,
        "FORWARD": bool(
            result.get("wbc_call_started", False)
            or result.get("scientific_execution_begun", False)
        ),
        "WBC_CALL_STARTED": bool(result.get("wbc_call_started", False)),
        "TRAINING": False,
        "GR00T_MODEL_CONSTRUCTED": False,
        "SCIENTIFIC_CONTRACT_CHANGED": False,
        "HISTORICAL_ARTIFACTS_MUTATED": False,
        "ERROR": None if error is None else dict(error),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--arm", type=Path, required=True)
    parser.add_argument("--dog", type=Path, required=True)
    parser.add_argument("--native-runner", type=Path, default=REPO / "scripts/native_wbc_benchmark.py")
    parser.add_argument(
        "--resume-root",
        type=Path,
        help="fresh-allocation root whose finalized rows may be carried forward read-only",
    )
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    args = parser.parse_args(argv)
    # Keep symlink identity visible to the strict scaffold guard.  Resolving
    # here would erase a symlink before the guard could reject it.
    output_root = args.output_root.absolute()
    ensure_output_root_scaffold(output_root)
    scientific_execution_begun = False
    wbc_call_started = False
    rows: List[Dict[str, Any]] = []
    error: Optional[Dict[str, Any]] = None
    runtime_qualification_result: Optional[Dict[str, Any]] = None
    robustness_skip: Optional[Dict[str, Any]] = None
    resume_info: Optional[Dict[str, Any]] = None
    status = "BLOCKED"
    log_event(output_root, "START", plan=str(args.plan), pbs_job_id=os.environ.get("PBS_JOBID"))
    try:
        plan = load_json(args.plan)
        write_json(output_root / "stage36_campaign_plan_snapshot.json", plan)
        errors = validate_plan(plan, args.source_root.resolve())
        if errors:
            raise RuntimeError("plan validation failed: %s" % ",".join(errors))
        if args.resume_root is not None:
            inherited_rows, resume_info = load_resume_rows(
                args.resume_root,
                output_root,
                plan,
            )
            rows.extend(inherited_rows)
            scientific_execution_begun = bool(inherited_rows)
            wbc_call_started = bool(inherited_rows)
            log_event(
                output_root,
                "RESUME_SOURCE_ACCEPTED",
                **resume_info,
            )
        gpu = run_gpu_preflight(output_root)
        if gpu.get("status") != "PASS":
            raise RuntimeError("GPU preflight failed")
        if not args.native_runner.is_file() or not args.arm.is_file() or not args.dog.is_file():
            raise RuntimeError("native runner or exact checkpoint input missing")
        if sha256_file(args.native_runner) != plan.get("native_runner_sha256", sha256_file(args.native_runner)):
            raise RuntimeError("native runner SHA does not match plan")
        runtime_qualification_result = runtime_qualification(
            output_root,
            args.source_root.resolve(),
            args.arm.resolve(),
            args.dog.resolve(),
            plan,
        )
        log_event(
            output_root,
            "RUNTIME_QUALIFICATION",
            status=runtime_qualification_result.get("status"),
            runtime_identity_class=runtime_qualification_result.get("runtime_identity_class"),
        )
        if runtime_qualification_result.get("status") != "PASS":
            raise RuntimeError("native runtime qualification failed")
        rows_plan = plan["rows"]
        carried_row_ids = {str(row.get("row_id")) for row in rows}
        primary_plan_rows = [row for row in rows_plan if row.get("phase") == "PRIMARY"]
        for index, plan_row in enumerate(rows_plan, start=1):
            if str(plan_row.get("row_id")) in carried_row_ids:
                log_event(
                    output_root,
                    "ROW_REUSED_IMMUTABLE",
                    index=index,
                    row_id=plan_row["row_id"],
                )
                continue
            if plan_row.get("phase") == "ROBUSTNESS":
                primary_results = [row for row in rows if row.get("phase") == "PRIMARY"]
                primary_ids = {str(row.get("row_id")) for row in primary_results}
                expected_primary_ids = {
                    str(row.get("row_id")) for row in primary_plan_rows
                }
                primary_complete = bool(
                    len(primary_results) == len(primary_plan_rows)
                    and primary_ids == expected_primary_ids
                    and all(bool(row.get("tracking_pass")) for row in primary_results)
                )
                if not primary_complete:
                    robustness_skip = {
                        "event": "ROBUSTNESS_SKIPPED_PRIMARY_NOT_COMPLETE",
                        "first_skipped_row_id": plan_row.get("row_id"),
                        "primary_expected_count": len(primary_plan_rows),
                        "primary_completed_count": len(primary_results),
                        "primary_failed_row_ids": [
                            row.get("row_id")
                            for row in primary_results
                            if not bool(row.get("tracking_pass"))
                        ],
                    }
                    log_event(output_root, **robustness_skip)
                    break
            row_output = output_root / "rows" / str(plan_row["row_id"])
            row_output.mkdir(parents=True)
            command = native_command(
                args.native_runner,
                args.source_root.resolve(),
                args.arm.resolve(),
                args.dog.resolve(),
                Path(plan_row["trajectory_path"]).resolve(),
                row_output,
                Path(plan_row["metadata_path"]).resolve(),
                int(plan_row["seed"]),
                float(plan_row["seconds"]),
            )
            command_text = " ".join(command)
            if "qsub" in command_text or "--reference-mode BENCHMARK_CONTINUOUS" not in command_text:
                raise RuntimeError("selected native command violates execution contract")
            wbc_call_started = True
            log_event(output_root, "ROW_START", index=index, row_id=plan_row["row_id"], command=command)
            run_log = row_output / "run.log"
            timed_out = False
            with run_log.open("w", encoding="utf-8") as handle:
                try:
                    completed = subprocess.run(
                        command,
                        cwd=str(args.source_root.resolve()),
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        text=True,
                        check=False,
                        timeout=float(args.timeout_seconds),
                    )
                    returncode = int(completed.returncode)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    returncode = 124
                    handle.write("STAGE36_TIMEOUT\n")
            row = row_result(plan_row, row_output, returncode, timed_out)
            rows.append(row)
            if row["scientific_row_finalized"]:
                scientific_execution_begun = True
            log_event(
                output_root,
                "ROW_END",
                row_id=plan_row["row_id"],
                returncode=returncode,
                runner_pass=row["runner_pass"],
                tracking_pass=row["tracking_pass"],
            )
            if not row["tracking_pass"]:
                if row["scientific_row_finalized"]:
                    log_event(
                        output_root,
                        "ROW_SCIENTIFIC_RESULT_CONTINUE",
                        row_id=plan_row["row_id"],
                        phase=plan_row.get("phase"),
                        returncode=returncode,
                        safety_violation=bool(row["safety_gate"].get("safety_violation")),
                    )
                else:
                    log_event(
                        output_root,
                        "CAMPAIGN_STOPPED_ROW_FAILURE",
                        row_id=plan_row["row_id"],
                        phase=plan_row.get("phase"),
                        returncode=returncode,
                    )
                    break
        primary_rows = [row for row in rows if row.get("phase") == "PRIMARY"]
        core_rows = [row for row in primary_rows if row["family"] in ("N", "T", "P")]
        continuous_rows = [row for row in primary_rows if row["family"] == "C"]
        planned_representations = [str(value) for value in plan.get("representations", [])]
        primary_representations = [
            representation
            for representation in planned_representations
            if any(
                row.get("phase") == "PRIMARY" and row.get("representation") == representation
                for row in rows_plan
            )
        ]
        expected_core_rows = 24 * len(primary_representations)
        expected_continuous_rows = 8 * len(primary_representations)
        core_pass = (
            len(core_rows) == expected_core_rows > 0
            and all(row["tracking_pass"] for row in core_rows)
        )
        continuous_pass = (
            expected_continuous_rows > 0
            and len(continuous_rows) == expected_continuous_rows
            and all(row["tracking_pass"] for row in continuous_rows)
        )
        family_summary = {
            family: profile_summary([row for row in rows if row["family"] == family])
            for family in ("N", "T", "P", "C")
        }
        seed_summary = {
            str(seed): profile_summary([row for row in rows if row["seed"] == seed])
            for seed in sorted({int(row["seed"]) for row in rows})
        }
        t_summary = {"duration_2.0s": profile_summary(rows)}
        geometry_status = "NOT_QUALIFIED" if plan.get("k_star") is None else "QUALIFIED_CANDIDATE_NOT_RUN"
        timing_status = "NOT_QUALIFIED" if plan.get("t_star") is None else "QUALIFIED_CANDIDATE_NOT_RUN"
        all_planned_rows_completed = len(rows) == len(rows_plan) and all(
            bool(row.get("scientific_row_finalized")) for row in rows
        )
        final_classification = (
            "PARTIAL_OR_UNCLASSIFIED"
            if not all_planned_rows_completed
            else (
                "DENSE_ORACLE_WBC_CORE_QUALIFIED_STAGE37_AUTHORIZED"
                if core_pass
                else "WBC_BOTTLENECK_BEFORE_STRUCTURED_GR00T"
            )
        )
        result_payload = {
            "schema": "STAGE36_WBC_ORACLE_RESULT_V2",
            "timestamp_utc": now_utc(),
            "pi_gate": PI_GATE_ID,
            "pbs_job_id": os.environ.get("PBS_JOBID"),
            "project": EXPECTED_PROJECT,
            "reference_mode": EXPECTED_REFERENCE_MODE,
            "scientific_execution_begun": scientific_execution_begun,
            "wbc_call_started": wbc_call_started,
            "resume": resume_info,
            "row_count": len(rows),
            "maximum_rows": MAX_ROWS,
            "all_planned_rows_completed": all_planned_rows_completed,
            "all_finite": bool(rows) and all(row["finite"] for row in rows),
            "core24": {
                "row_count": len(core_rows),
                "tracking_pass": core_pass,
                "status": "PASS" if core_pass else "FAIL_OR_PARTIAL",
            },
            "continuous8": {
                "row_count": len(continuous_rows),
                "tracking_pass": continuous_pass,
                "status": "PASS" if continuous_pass else "FAIL_OR_PARTIAL",
            },
            "representation_summary": {
                representation: profile_summary(
                    [row for row in rows if row["representation"] == representation]
                )
                for representation in planned_representations
            },
            "robustness": {
                "planned_row_count": int(plan.get("robustness_row_count", 0)),
                "executed_row_count": sum(row.get("phase") == "ROBUSTNESS" for row in rows),
                "skipped": robustness_skip is not None,
                "skip": robustness_skip,
            },
            "geometry": {"k_star": plan.get("k_star"), "status": geometry_status},
            "timing": {"t_star": plan.get("t_star"), "status": timing_status},
            "runtime_qualification": runtime_qualification_result,
            "motion_api_classification": geometry_status,
            "continuous_base_classification": "NATIVE_WBC_PASS" if continuous_pass else "NATIVE_WBC_FAIL_OR_PARTIAL",
            "final_classification": final_classification,
            "per_family_summary": family_summary,
            "per_seed_summary": seed_summary,
            "per_t_summary": t_summary,
            "profile_metrics": [row["profile_metrics"] for row in rows],
            "safety_gate_transitions": [
                {"row_id": row["row_id"], "transitions": row["safety_gate"]["transitions"]}
                for row in rows
            ],
            "finite_flags": [row["finite_flags"] for row in rows],
            "rows": rows,
            "forbidden_execution": {
                "training": False,
                "backward": False,
                "optimizer_step": False,
                "scheduler_step": False,
                "resume": False,
                "new_checkpoint": False,
                "sampler": False,
                "get_action": False,
                "gr00t": False,
                "hardware": False,
                "second_allocation": False,
            },
            "next_gate": NEXT_GATE,
        }
        write_json(output_root / "stage36_result.json", result_payload)
        write_json(
            output_root / "stage36_raw_hash_manifest.json",
            {
                "schema": "STAGE36_RAW_HASH_MANIFEST_V1",
                "timestamp_utc": now_utc(),
                "rows": [
                    {
                        "row_id": row["row_id"],
                        "summary_path": str(Path(row["output_dir"]) / "native_external_summary.json"),
                        "summary_sha256": sha256_file(Path(row["output_dir"]) / "native_external_summary.json")
                        if (Path(row["output_dir"]) / "native_external_summary.json").is_file()
                        else None,
                        "raw_rows_path": row["raw_rows_path"],
                        "raw_rows_sha256": sha256_file(Path(row["raw_rows_path"]))
                        if Path(row["raw_rows_path"]).is_file()
                        else None,
                    }
                    for row in rows
                ],
            },
        )
        write_json(
            output_root / "stage36_result_manifest.json",
            {
                "schema": "STAGE36_RESULT_MANIFEST_V1",
                "result_path": str(output_root / "stage36_result.json"),
                "result_sha256": sha256_file(output_root / "stage36_result.json"),
                "raw_hash_manifest_path": str(output_root / "stage36_raw_hash_manifest.json"),
                "raw_hash_manifest_sha256": sha256_file(output_root / "stage36_raw_hash_manifest.json"),
                "row_count": len(rows),
                "all_finite": result_payload["all_finite"],
                "final_classification": final_classification,
            },
        )
        status = (
            "PASS_SCIENTIFIC_EXECUTION_NEEDS_PI_REVIEW"
            if core_pass and all_planned_rows_completed and robustness_skip is None
            else "PARTIAL_OR_FAILED_NEEDS_PI_REVIEW"
        )
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        log_event(output_root, "ERROR", **error)
        result_payload = {
            "schema": "STAGE36_WBC_ORACLE_RESULT_V2",
            "timestamp_utc": now_utc(),
            "pi_gate": PI_GATE_ID,
            "pbs_job_id": os.environ.get("PBS_JOBID"),
            "scientific_execution_begun": scientific_execution_begun,
            "wbc_call_started": wbc_call_started,
            "resume": resume_info,
            "row_count": len(rows),
            "rows": rows,
            "all_finite": False,
            "status": "PARTIAL_OR_BLOCKED",
            "runtime_qualification": runtime_qualification_result,
            "error": error,
            "next_gate": NEXT_GATE,
        }
        write_json(output_root / "stage36_result.json", result_payload)
        write_json(
            output_root / "stage36_raw_hash_manifest.json",
            {"schema": "STAGE36_RAW_HASH_MANIFEST_V1", "rows": [], "partial": True},
        )
        write_json(
            output_root / "stage36_result_manifest.json",
            {
                "schema": "STAGE36_RESULT_MANIFEST_V1",
                "result_path": str(output_root / "stage36_result.json"),
                "result_sha256": sha256_file(output_root / "stage36_result.json"),
                "partial": True,
            },
        )
    # The campaign log is complete before sealing.  No evidence is deleted or
    # rewritten if a row failed, and there is no replacement-run branch.
    seal_payload = {
        "schema": "STAGE36_ARTIFACT_SEAL_V2",
        "timestamp_utc": now_utc(),
        "pi_gate": PI_GATE_ID,
        "pbs_job_id": os.environ.get("PBS_JOBID"),
        "output_root": str(output_root),
        "entries": seal_entries(output_root, ("stage36_artifact_seal.json", "stage36_return.json")),
        "inherited_entries": inherited_seal_entries(rows, output_root),
        "resume": resume_info,
        "source_artifacts": [
            {
                "role": "evaluation_runner",
                "path": str(args.native_runner),
                "size_bytes": args.native_runner.stat().st_size,
                "sha256": sha256_file(args.native_runner),
            },
            {
                "role": "pbs_wrapper",
                "path": str(REPO / "hpc/pbs/stage36_wbc_oracle_qualification.pbs"),
                "size_bytes": (REPO / "hpc/pbs/stage36_wbc_oracle_qualification.pbs").stat().st_size,
                "sha256": sha256_file(REPO / "hpc/pbs/stage36_wbc_oracle_qualification.pbs"),
            },
        ],
        "historical_inputs_mutated": False,
        "second_allocation_submitted": False,
    }
    seal_path = output_root / "stage36_artifact_seal.json"
    write_json(seal_path, seal_payload)
    seal_sha = sha256_file(seal_path)
    final_return = make_return(output_root, result_payload, seal_path, seal_sha, status, error)
    write_json(output_root / "stage36_return.json", final_return)
    return 0 if status == "PASS_SCIENTIFIC_EXECUTION_NEEDS_PI_REVIEW" else 2


if __name__ == "__main__":
    raise SystemExit(main())
