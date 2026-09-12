#!/usr/bin/env python3
"""PI0148 historical runtime binding and bounded toy-CUDA smoke.

The control process is stdlib-only. It probes only historically recorded,
user-owned interpreters and exact project paths, then optionally asks one
selected interpreter to execute four 32x32 identity matmuls. The child
snippet imports Torch but never imports GR00T, Transformers, a simulator, or
any checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "STAGE45F_CORELAB_PI0148_RUNTIME_BINDING_AND_TOY_SMOKE_V1"
PI_GATE = "PI_GATE_STATE_0148"
PI_COMMENT = 5646319460
MAX_CALLS_PER_DEVICE = 4
MAX_CALLS_TOTAL = 8
TARGETS = {"4080S": "NVIDIA GeForce RTX 4080 SUPER", "5080": "NVIDIA GeForce RTX 5080"}

HISTORICAL_CANDIDATES = (
    {
        "path": "/var/cache/users/liuyu-nfs/go2-arx5-vla-stage10/runtime/venv/bin/python",
        "source": "RTX4080S_R0_RUNTIME_INVENTORY:runtimes.stage10_python; Stage21/45E launcher records",
        "role": "GR00T_STAGE10_MODEL_RUNTIME",
    },
    {
        "path": "/opt/miniconda3/envs/isaacgym/bin/python",
        "source": "hpc/pbs/stage40_b1.pbs and Stage16/40 native runtime records",
        "role": "NATIVE_STAGE40_RUNTIME",
    },
)

HISTORICAL_ARTIFACTS = (
    {
        "label": "BASE_REFERENCE_ONLY",
        "path": "/var/cache/users/liuyu-nfs/go2-arx5-vla-stage10/assets/GR00T-N1.7-3B",
        "source": "RTX4080S_R0_RUNTIME_INVENTORY:artifacts.base_reference_only",
        "expected_sha256": "407804ea5a62f4f8823f48811ae0edbb82fac101e9cf4d7273e6e2f692bb4d59",
        "hash_scope": "index file identity only; directory tree not rehashed in this smoke",
    },
    {
        "label": "STAGE10_PROVISIONAL_MODALITY",
        "path": "/var/cache/users/liuyu-nfs/go2-arx5-vla-stage10/artifacts/go2_arx5_modality.py",
        "source": "RTX4080S_R0_RUNTIME_INVENTORY:artifacts.frozen_modality.stage10_provisional_locator",
        "expected_sha256": "e61f9429ecc21ceab51d9d26cadc9018691eb8b5c74d60aec4abf7faee16f031",
        "hash_scope": "file bytes",
    },
    {
        "label": "GR00T_SOURCE",
        "path": "/home/liuyu-nfs/Projects/isaac-gr00t",
        "source": "RTX4080S_R0_RUNTIME_INVENTORY:artifacts.groot_source.path_on_rtx_host",
        "expected_sha256": None,
        "hash_scope": "git identity probed without modifying checkout",
    },
    {
        "label": "FULL24_AT_2000_ENGINEERING_REFERENCE",
        "path": "/scratch/e1192548/stage45b_pi0087_b2_h200_20260909T052311Z/checkpoint-2000",
        "source": "STAGE45C_TRAIN24_EVALUATION_PROTOCOL_FROZEN_V1:runtime_binding.checkpoint_manifest",
        "expected_sha256": "2113a6a3b2a91233a42214dbb017c0136acde282fda2a418066f9317e78981a5",
        "hash_scope": "checkpoint tree; no model load in this script",
    },
    {
        "label": "FULL24_RUNTIME_V2_STATS",
        "path": "/scratch/e1192548/stage45b_pi0083_runtime_parquet_v2/STAGE45B_FULL24_RUNTIME_PARQUET_STATS_V2.json",
        "source": "STAGE45C_TRAIN24_EVALUATION_PROTOCOL_FROZEN_V1:runtime_binding.structured_statistics.path",
        "expected_sha256": "3b3a2d51e685a0e580de4ebdfa05abcbcddfb7dc33b8d3a659994cdcd1bb0337",
        "hash_scope": "file bytes",
    },
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(argv: Sequence[str], *, env: Mapping[str, str] | None = None, timeout: int = 30) -> dict[str, Any]:
    try:
        result = subprocess.run(list(argv), capture_output=True, text=True, check=False, timeout=timeout, env=dict(env) if env is not None else None)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"argv": list(argv), "returncode": None, "stdout": "", "stderr": str(exc)}
    return {"argv": list(argv), "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def _parse_gpu_rows(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        values = [item.strip() for item in line.split(",")]
        if len(values) != 7:
            continue
        rows.append({
            "index": int(values[0]),
            "name": values[1],
            "uuid": values[2],
            "memory_total_mib": int(float(values[3])),
            "memory_used_mib": int(float(values[4])),
            "memory_free_mib": int(float(values[5])),
            "driver_version": values[6],
        })
    return rows


def _parse_process_rows(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        values = [item.strip() for item in line.split(",")]
        if len(values) != 4:
            continue
        rows.append({"uuid": values[0], "pid": int(values[1]), "name": values[2], "used_memory_mib": int(float(values[3]))})
    return rows


def query_devices() -> dict[str, Any]:
    inventory = _run(["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.used,memory.free,driver_version", "--format=csv,noheader,nounits"])
    processes = _run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"])
    return {
        "inventory_command": inventory,
        "process_command": processes,
        "gpus": _parse_gpu_rows(inventory["stdout"]) if inventory["returncode"] == 0 else [],
        "processes": _parse_process_rows(processes["stdout"]) if processes["returncode"] == 0 else [],
        "hostname": platform.node(),
    }


def probe_interpreter(candidate: Mapping[str, Any]) -> dict[str, Any]:
    raw = Path(str(candidate["path"]))
    base: dict[str, Any] = {"path": str(raw), "source": candidate["source"], "role": candidate["role"]}
    if not raw.is_file() or not os.access(raw, os.X_OK):
        base.update({"status": "ABSENT_AT_CHECKED_PATH", "probe": None})
        return base
    code = r'''
import importlib.metadata as md, json, sys
names = ["torch", "transformers", "accelerate", "safetensors", "numpy"]
dist = {d.metadata["Name"].lower(): d.version for d in md.distributions() if d.metadata.get("Name")}
out = {"sys_executable": sys.executable, "sys_prefix": sys.prefix, "base_prefix": sys.base_prefix, "python": sys.version.split()[0], "packages": {n: dist.get(n) for n in names}}
try:
    import torch
    out["torch"] = {"version": torch.__version__, "cuda": torch.version.cuda, "cuda_available": bool(torch.cuda.is_available()), "device_count": int(torch.cuda.device_count())}
except Exception as exc:
    out["torch_error"] = type(exc).__name__ + ":" + str(exc)
print(json.dumps(out, sort_keys=True))
'''
    result = _run([str(raw), "-c", code], timeout=45)
    base["probe_command"] = result
    try:
        parsed = json.loads(result["stdout"].strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        parsed = None
    base["probe"] = parsed
    if result["returncode"] != 0 or not isinstance(parsed, dict):
        base["status"] = "PRESENT_UNVERIFIED"
    elif isinstance(parsed.get("torch"), dict) and parsed["torch"].get("cuda_available") is True:
        base["status"] = "RUNTIME_METADATA_VERIFIED"
    else:
        base["status"] = "CANDIDATE_RUNTIME_FOUND"
    cfg = raw.parent.parent / "pyvenv.cfg"
    if cfg.is_file():
        base["environment_source"] = {"path": str(cfg), "sha256": _sha256(cfg)}
    return base


def artifact_observations() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in HISTORICAL_ARTIFACTS:
        path = Path(item["path"])
        record = {**item, "observed_path": str(path)}
        if not path.exists():
            record["status"] = "ABSENT_AT_CHECKED_PATH"
        elif path.is_file():
            observed = _sha256(path)
            record["observed_sha256"] = observed
            record["status"] = "PRESENT_VERIFIED" if item.get("expected_sha256") and observed == item["expected_sha256"] else "PRESENT_UNVERIFIED"
        else:
            record["status"] = "PRESENT_UNVERIFIED"
            record["observed_type"] = "directory"
        rows.append(record)
    return rows


_TOY_CUDA_CODE = r'''
import json, time, torch
started = time.time()
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")
if torch.cuda.device_count() != 1:
    raise RuntimeError(f"UUID binding did not expose exactly one device: {torch.cuda.device_count()}")
torch.cuda.set_device(0)
name = torch.cuda.get_device_name(0)
torch.cuda.reset_peak_memory_stats()
calls = []
with torch.inference_mode():
    for dtype_name, dtype in (("float32", torch.float32), ("bfloat16", torch.bfloat16)):
        for repeat in (1, 2):
            identity = torch.eye(32, device="cuda", dtype=dtype)
            result = identity @ identity
            torch.cuda.synchronize()
            finite = bool(torch.isfinite(result).all().item())
            max_error = float((result - identity).abs().max().item())
            if not finite or max_error != 0.0:
                raise RuntimeError(f"non-identity result dtype={dtype_name} repeat={repeat} finite={finite} max_error={max_error}")
            calls.append({"dtype": dtype_name, "repeat": repeat, "finite": finite, "max_abs_error": max_error})
print(json.dumps({"device_name": name, "calls": calls, "call_count": len(calls), "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()), "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()), "elapsed_s": time.time()-started}, sort_keys=True))
'''


def run_toy_smoke(interpreter: str, uuid: str, role: str, *, devices: Mapping[str, Any]) -> dict[str, Any]:
    target = next((row for row in devices.get("gpus", []) if row.get("uuid") == uuid), None)
    if not isinstance(target, Mapping):
        return {"status": "BLOCKED_UUID_NOT_OBSERVED", "attempted_calls": 0, "completed_calls": 0, "uuid": uuid, "role": role}
    processes = [row for row in devices.get("processes", []) if row.get("uuid") == uuid]
    if processes:
        return {"status": "BLOCKED_DEVICE_BUSY", "attempted_calls": 0, "completed_calls": 0, "uuid": uuid, "role": role, "processes": processes}
    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": uuid, "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    result = _run([interpreter, "-c", _TOY_CUDA_CODE], env=env, timeout=120)
    parsed = None
    if result["stdout"].strip():
        try:
            parsed = json.loads(result["stdout"].strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            parsed = None
    completed = int(parsed.get("call_count", 0)) if isinstance(parsed, Mapping) else 0
    attempted = completed if result["returncode"] == 0 else None
    return {
        "status": "PASS" if result["returncode"] == 0 and completed == MAX_CALLS_PER_DEVICE else "FAILED_STOPPED",
        "uuid": uuid,
        "role": role,
        "interpreter": interpreter,
        "attempted_calls": attempted,
        "completed_calls": completed,
        "command": result,
        "result": parsed,
    }


def build_report(*, role: str, uuid: str | None, interpreter: str | None, execute_smoke: bool) -> dict[str, Any]:
    devices = query_devices()
    runtimes = [probe_interpreter(item) for item in HISTORICAL_CANDIDATES]
    artifacts = artifact_observations()
    if execute_smoke and uuid and interpreter:
        smoke = run_toy_smoke(interpreter, uuid, role, devices=devices)
    else:
        smoke = {"status": "NOT_ATTEMPTED", "attempted_calls": 0, "completed_calls": 0, "reason": "runtime binding mode only"}
    return {
        "schema": SCHEMA,
        "status": "TOY_SMOKE_PASS" if smoke.get("status") == "PASS" else smoke.get("status", "RUNTIME_BINDING_COMPLETE"),
        "gate": {"name": PI_GATE, "issue_comment": PI_COMMENT},
        "host": devices["hostname"],
        "gpu_observation": devices,
        "runtime_candidates": runtimes,
        "artifact_observations": artifacts,
        "selected_binding": {"role": role, "uuid": uuid, "interpreter": interpreter, "source": "PI0148 historical launcher/runtime records"},
        "toy_cuda_smoke": smoke,
        "next_model_request": {
            "status": "REQUEST_ONLY_NOT_EXECUTED",
            "checkpoint": "FULL24@2000",
            "checkpoint_tree_sha256": "2113a6a3b2a91233a42214dbb017c0136acde282fda2a418066f9317e78981a5",
            "window_id": "stage22_expert_red_target_train_a:expert-0000",
            "sampler": {"steps": 4, "timestep_ids": [0, 250, 500, 750], "conditioning_passes": 1, "action_head_calls": 4, "target_xyz_in_model_input": False},
            "model_forward": 0,
            "reason": "PI0148 permits a concrete request after smoke; execution remains separately gated.",
        },
        "execution_boundary": {"model_construction": 0, "model_forward": 0, "training": 0, "backward": 0, "optimizer_steps": 0, "pbs": 0, "Validation-E": 0, "Holdout-F": 0, "ICL": 0, "WBC": 0, "simulator": 0, "hardware": 0, "real_h200_campaign_ingested": False},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=sorted(TARGETS), required=True)
    parser.add_argument("--uuid")
    parser.add_argument("--interpreter")
    parser.add_argument("--execute-smoke", action="store_true")
    args = parser.parse_args()
    report = build_report(role=args.role, uuid=args.uuid, interpreter=args.interpreter, execute_smoke=args.execute_smoke)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] in {"TOY_SMOKE_PASS", "NOT_ATTEMPTED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
