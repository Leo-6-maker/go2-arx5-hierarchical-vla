#!/usr/bin/env python3
"""PI0239 continuation runner.

This is a thin execution shell around the already-audited PI0238 mature
runner.  It owns identity, resume, runtime evidence, and the fixed G4
diagnostic shell; it does not alter the model, gate, scorer, or data contract.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

import pi0238_audit_to_sim as mature
import pi0238_r1_engineering_recovery as r1


PROTOCOL_COMMIT = "451bbd231b60f7534bb7c08c654f81d7bea322ee"
AUDITED_PARENT = "a30905e03d4009abc419cb7b60b39da3032cfef8"
FRESH_JOB = "623593.hopper-m-02"
SEED = 22301
HORIZON = 40
DT = 0.05
PHYSICAL_DIM = 10
UNIVERSAL_DIM = 132
QSTAT = "/cm/shared/apps/pbspro/current/bin/qstat"
MODALITY_SHA = "244f37f28e62b7b392a1598b78e78f00e67693f6c08a606b3c5358830ada8b25"
STAGE45B_MANIFEST_SHA = "6160d579b98dc5ee4b114ada4a4e00e88ee13df6ed1fc3d308b5879d67027d2c"
PHYSICAL_TOL = {"xyz_m": 1e-5, "so3_rad": 1e-4, "gripper": 1e-5}


class EngineeringBlocker(RuntimeError):
    pass


class ScienceFailure(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EngineeringBlocker(message)


sha256_file = mature.sha256_file
sha256_array = mature.sha256_array
json_load = mature.json_load
write_json = mature.write_json
load_window = mature.load_window
source_rollout = mature.source_rollout
full_metadata = mature.full_metadata
episode_rows = mature.episode_rows
tree_sha256 = mature.tree_sha256


def run_capture(argv: list[str], *, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    result = subprocess.run(argv, capture_output=True, text=True, check=False, env=None if env is None else dict(env))
    return {"argv": argv, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def cfg_paths(cfg: Mapping[str, Any]) -> dict[str, Path]:
    return {key: Path(value) for key, value in cfg["paths"].items() if isinstance(value, str)}


def load_config(path: Path) -> dict[str, Any]:
    payload = json_load(path)
    require(payload.get("schema") == "PI0239_RESOLVED_LAUNCH_V1", "PI0239 launch schema drift")
    require(payload.get("protocol_commit") == PROTOCOL_COMMIT, "PI0239 protocol commit drift")
    require(payload.get("audited_parent") == AUDITED_PARENT, "PI0239 audited parent drift")
    return payload


def under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def path_no_symlink(path: Path, *, allow_missing: bool = False) -> Path:
    require(path.is_absolute(), f"path is not absolute: {path}")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            raise EngineeringBlocker(f"symlink path component: {current}")
        if not current.exists() and not allow_missing:
            raise EngineeringBlocker(f"missing path component: {current}")
        if not current.exists():
            break
    return path.resolve(strict=not allow_missing)


def pbs_job_lines(text: str, own_job: str | None = None) -> list[str]:
    prefix = own_job.split(".", 1)[0] if own_job else None
    return [
        line for line in text.splitlines()
        if re.match(r"^\s*\d+\.[A-Za-z0-9_.-]+\b", line)
        and (not prefix or not re.match(rf"^\s*{re.escape(prefix)}(?:[.\s-])", line))
    ]


def runtime_env(cfg: Mapping[str, Any], *, cuda: bool) -> dict[str, str]:
    paths = cfg_paths(cfg)
    env = dict(os.environ)
    env.update({
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": "",
    })
    if not cuda:
        env["CUDA_VISIBLE_DEVICES"] = ""
    root = paths["stage45b_root"]
    torchcodec = paths["torchcodec_root"]
    torchlib = paths["torch_lib_root"]
    ambient = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join(str(item) for item in (root / "lib", root / "compat", torchcodec, torchlib) if item) + (":" + ambient if ambient else "")
    return env


def import_paths(args: SimpleNamespace) -> None:
    sys.path[:0] = [
        str(args.prep_root / "scripts"),
        str(args.vla_source / "src"),
        str(args.stage10_root / "src" / "isaac-gr00t"),
    ]


def module_identity(module: Any) -> dict[str, Any]:
    path = Path(module.__file__).resolve()
    return {"module": module.__name__, "file": str(path), "sha256": sha256_file(path)}


def loaded_ffmpeg_maps() -> list[str]:
    path = Path(f"/proc/{os.getpid()}/maps")
    if not path.exists():
        return ["NOT_OBSERVED_IN_SEALED_ARTIFACT"]
    return sorted({line.split()[-1] for line in path.read_text(errors="replace").splitlines() if ("libav" in line or "libsw" in line or "torchcodec" in line) and "/" in line})


def runtime_probe(args: SimpleNamespace, cfg: Mapping[str, Any]) -> dict[str, Any]:
    import_paths(args)
    os.environ.update(runtime_env(cfg, cuda=False))
    import torch  # type: ignore
    import torchcodec  # type: ignore
    from torchcodec.decoders import VideoDecoder  # type: ignore
    import stage21_same_scene_groot_inference as stage21  # type: ignore
    import stage40_b0_model_control as stage40  # type: ignore
    import go2_arx5_modality  # type: ignore
    from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset  # type: ignore
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader  # type: ignore
    core = Path(torchcodec.__file__).resolve().parent / "libtorchcodec_core6.so"
    ldd = run_capture(["ldd", str(core)]) if core.exists() else {"argv": ["ldd", str(core)], "returncode": 127, "stdout": "", "stderr": "missing core6"}
    modality = Path(go2_arx5_modality.__file__).resolve()
    return {
        "python": sys.executable,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cuda_version": str(torch.version.cuda),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "NOT_SET"),
        "cuda_available": bool(torch.cuda.is_available()),
        "torchcodec_version": importlib.metadata.version("torchcodec"),
        "torchcodec_file": str(Path(torchcodec.__file__).resolve()),
        "video_decoder_class": f"{VideoDecoder.__module__}.{VideoDecoder.__name__}",
        "loaded_ffmpeg_maps": loaded_ffmpeg_maps(),
        "ldd_torchcodec_core6": ldd,
        "stage21": module_identity(stage21),
        "stage40": module_identity(stage40),
        "modality": {"file": str(modality), "sha256": sha256_file(modality)},
        "dataset_class": f"{ShardedSingleStepDataset.__module__}.{ShardedSingleStepDataset.__name__}",
        "episode_loader_class": f"{LeRobotEpisodeLoader.__module__}.{LeRobotEpisodeLoader.__name__}",
        "stage45b_manifest": {"path": str(args.stage45b_manifest), "sha256": sha256_file(args.stage45b_manifest)},
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH", ""),
        "model_weight_loads": 0,
        "model_forward_calls": 0,
        "get_action_calls": 0,
    }


def evidence_copy_and_hash(old_root: Path, new_evidence: Path, names: list[str]) -> dict[str, Any]:
    new_evidence.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in names:
        source = old_root / "evidence" / name
        require(source.is_file(), f"missing reusable evidence: {source}")
        destination = new_evidence / name
        shutil.copy2(source, destination)
        rows.append({"name": name, "source": str(source), "destination": str(destination), "sha256": sha256_file(source), "copied_sha256": sha256_file(destination)})
        require(rows[-1]["sha256"] == rows[-1]["copied_sha256"], f"evidence copy hash drift: {name}")
    return {"status": "PASS", "rows": rows}


def reservation(path: Path, size: int) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        if hasattr(os, "posix_fallocate"):
            os.posix_fallocate(handle.fileno(), 0, size)
        else:
            handle.seek(size - 1)
            handle.write(b"\0")
        handle.flush()
        os.fsync(handle.fileno())
    stat = path.stat()
    allocated = int(getattr(stat, "st_blocks", 0) * 512)
    require(allocated >= size, f"reservation is sparse or short: allocated={allocated} size={size}")
    return {"path": str(path), "requested_bytes": size, "allocated_bytes": allocated, "fsync": True, "non_sparse": True, "sha256": sha256_file(path)}


def storage_observation(path: Path) -> dict[str, Any]:
    parent = path if path.exists() else path.parent
    usage = shutil.disk_usage(parent)
    stat = os.statvfs(parent)
    commands = {name: run_capture(argv) for name, argv in (("quota_s", ["quota", "-s"]), ("quota_v", ["quota", "-v"]), ("quota_user", ["quota", "-u", "e1192548"]))}
    return {
        "fileset_observation_path": str(parent),
        "df": {"total_bytes": usage.total, "used_bytes": usage.used, "free_bytes": usage.free},
        "free_inodes": int(stat.f_bavail),
        "quota_commands": commands,
        "quota_status": "UNOBSERVED" if not any(item["stdout"].strip() for item in commands.values()) else "OUTPUT_PRESENT_LIMIT_UNPROVEN",
    }


def verify_reuse(args: SimpleNamespace, cfg: Mapping[str, Any]) -> dict[str, Any]:
    paths = cfg_paths(cfg)
    old = paths["r2_root"]
    expected = cfg["reuse"]["evidence_sha256"]
    rows = []
    for name, digest in expected.items():
        path = old / "evidence" / name
        require(path.is_file(), f"reusable evidence missing: {path}")
        actual = sha256_file(path)
        require(actual == digest, f"reusable evidence hash drift: {name}")
        rows.append({"name": name, "path": str(path), "sha256": actual})
    # The old PI0238 runner's G0/G1/G2 evidence is copied only after this
    # check; no scientific census is recomputed here.
    copied = evidence_copy_and_hash(old, paths["output_root"] / "evidence", list(expected))
    return {"status": "PASS_REUSED", "source_root": str(old), "rows": rows, "copy": copied, "model_weight_loads": 0, "model_forward_calls": 0, "get_action_calls": 0}


def preflight(args: SimpleNamespace, cfg: Mapping[str, Any], *, initialize: bool) -> dict[str, Any]:
    paths = cfg_paths(cfg)
    scratch = Path("/scratch/e1192548")
    for name, path in paths.items():
        if name in {"output_root", "cache_root", "tmp_root", "launch_config"}:
            continue
        path_no_symlink(path, allow_missing=False)
        require(under(path, scratch), f"path outside approved scratch: {name}={path}")
    for name in ("output_root", "cache_root", "tmp_root"):
        path = paths[name]
        if initialize:
            require(not path.exists(), f"PI0239 path already exists: {path}")
        else:
            require(not path.exists(), f"verify-only path unexpectedly exists: {path}")
    require(sha256_file(paths["protocol"]).lower() == str(cfg["implementation"]["protocol_sha256"]).lower(), "protocol hash drift")
    require(sha256_file(paths["script"]).lower() == str(cfg["implementation"]["script_sha256"]).lower(), "runner hash drift")
    require(sha256_file(paths["mature_script"]).lower() == str(cfg["implementation"]["mature_script_sha256"]).lower(), "mature runner hash drift")
    require(sha256_file(paths["r1_helper"]).lower() == str(cfg["implementation"]["r1_helper_sha256"]).lower(), "R1 helper hash drift")
    require(sha256_file(paths["pbs"]).lower() == str(cfg["implementation"]["pbs_sha256"]).lower(), "PBS hash drift")
    require(sha256_file(paths["modality_config"]) == MODALITY_SHA, "modality SHA drift")
    require(sha256_file(paths["stage45b_manifest"]) == STAGE45B_MANIFEST_SHA, "Stage45B manifest drift")
    old = paths["r2_root"] / "evidence"
    require((old / "g0_identity_and_access.json").is_file(), "R2 G0 evidence missing")
    fresh_s2 = paths["p237_root"] / "evidence" / "pi0236_s2_evaluator_admission.json"
    require(fresh_s2.is_file(), "fresh family admission missing")
    fresh = json_load(fresh_s2).get("fresh_family", {})
    require(fresh.get("job_id") == FRESH_JOB, "fresh family is not 623593")
    fresh_root = Path(fresh["root"]).resolve()
    require(fresh_root == paths["fresh_root"].resolve(), "fresh root identity drift")
    for step in (2500, 5000, 7500, 10000):
        checkpoint = fresh_root / "training" / f"checkpoint-{step}"
        require(checkpoint.is_dir(), f"fresh checkpoint missing: {checkpoint}")
    qstat = run_capture([QSTAT, "-u", "e1192548"]) if Path(QSTAT).is_file() else {"returncode": 127, "stdout": "", "stderr": "qstat unavailable", "argv": [QSTAT, "-u", "e1192548"]}
    other_jobs = pbs_job_lines(qstat["stdout"] + qstat["stderr"], os.environ.get("PBS_JOBID"))
    storage = storage_observation(paths["output_root"])
    result = {
        "schema": "PI0239_PREFLIGHT_V1",
        "mode": "INITIALIZE" if initialize else "VERIFY_ONLY",
        "status": "PASS_WITH_QUOTA_UNOBSERVED" if not other_jobs else "BLOCKED_ACTIVE_JOB",
        "argv": list(sys.argv), "cwd": str(Path.cwd()), "python": sys.executable, "python_version": platform.python_version(),
        "paths": {key: str(value) for key, value in paths.items()},
        "qstat": qstat, "active_other_jobs": other_jobs,
        "storage": storage,
        "fresh_family": {"job_id": FRESH_JOB, "root": str(fresh_root), "checkpoint_steps": [2500, 5000, 7500, 10000]},
        "read_boundary": {"dev_e_source_read_count": 0, "final_f_source_read_count": 0, "locked_data_mutation": False, "full_tree_hashes": False},
        "model_weight_loads": 0, "model_forward_calls": 0, "get_action_calls": 0,
    }
    require(not other_jobs, f"another active PBS job exists: {other_jobs}")
    if initialize:
        paths["output_root"].mkdir(parents=True, exist_ok=False)
        paths["cache_root"].mkdir(parents=True, exist_ok=False)
        paths["tmp_root"].mkdir(parents=True, exist_ok=False)
        result["reservation"] = reservation(paths["output_root"] / "quota_reservation.bin", int(cfg["reservation_bytes"]))
        write_json(paths["output_root"] / "evidence" / "pi0239_preflight.json", result)
    return result


def namespace(cfg: Mapping[str, Any]) -> SimpleNamespace:
    paths = cfg_paths(cfg)
    values = {key: value for key, value in paths.items()}
    values["stage45b_manifest"] = paths["stage45b_manifest"]
    values["config"] = cfg
    return SimpleNamespace(**values)


def clock_control(args: SimpleNamespace) -> dict[str, Any]:
    paths = cfg_paths(args.config)
    import_paths(args)
    expert = json_load(args.p237_root / "evidence" / "pi0237_p2_expert_gate.json")
    old_casebook = json_load(paths["r2_root"] / "evidence" / "g2_rotation_casebook.json")
    case_ids = {(str(row["source_episode_id"]), int(row["episode_index"])) for row in old_casebook["rows"]}
    rows = [row for row in expert["rows"] if row.get("split") == "CALIB" and row.get("decision") == "REJECT" and (str(row["source_episode_id"]), int(row["episode_index"])) in case_ids]
    require(len(rows) == 12, f"fixed clock control membership drift: {len(rows)}")
    os.environ.update(runtime_env(args.config, cuda=False))
    modules = r1.runtime_modules(args)
    gate = modules["gate"]()
    selector = modules["selector"]
    controls = []
    for row in rows:
        state, action, timestamps = load_window(args.calib_dataset, row)
        state_matrix = modules["rot6d_rows_to_matrix"](state[3:9].reshape(1, 6))[0]
        chunk = selector._build_chunk(action[None, ...], chunk_id=f"pi0239-clock-{int(row['episode_index']):05d}")
        shifted = r1.shifted_chunk(chunk, 17.0, modules["TrajectoryChunkV1"])
        original_result = gate.evaluate(chunk, now=0.0, achieved_position=state[:3], achieved_orientation=modules["matrix_to_quat"](state_matrix))
        shifted_result = gate.evaluate(shifted, now=17.0, achieved_position=state[:3], achieved_orientation=modules["matrix_to_quat"](state_matrix))
        keys = ("linear_speed_mps", "angular_speed_radps", "initial_rotation_discontinuity_rad")
        deltas = {key: float(shifted_result.metrics.get(key)) - float(original_result.metrics.get(key)) for key in keys}
        invariant = all(np.isclose(deltas[key], 0.0, rtol=0.0, atol=1e-10) for key in keys)
        controls.append({
            "episode_index": int(row["episode_index"]), "source_episode_id": str(row["source_episode_id"]),
            "original": {"decision": original_result.decision.value, "issues": [issue.code for issue in original_result.issues], "metrics": original_result.metrics},
            "shifted": {"decision": shifted_result.decision.value, "issues": [issue.code for issue in shifted_result.issues], "metrics": shifted_result.metrics},
            "shift_s": 17.0, "metric_deltas": deltas, "invariant": invariant,
            "pose_delta_max": float(np.max(np.abs(np.asarray(chunk.eef_position) - np.asarray(shifted.eef_position)))),
            "orientation_delta_max": float(np.max(np.abs(np.asarray(chunk.eef_orientation) - np.asarray(shifted.eef_orientation)))),
            "dt_original": float(np.median(np.diff(chunk.point_timestamps))), "dt_shifted": float(np.median(np.diff(shifted.point_timestamps))),
            "timestamps_preserved_after_unshift": bool(np.allclose(np.asarray(shifted.point_timestamps) - 17.0, np.asarray(chunk.point_timestamps), rtol=0.0, atol=1e-12)),
            "state_sha256": sha256_array(state), "action_sha256": sha256_array(action), "timestamp_sha256": sha256_array(timestamps),
        })
    passed = all(row["invariant"] and row["pose_delta_max"] == 0.0 and row["orientation_delta_max"] == 0.0 and row["timestamps_preserved_after_unshift"] for row in controls)
    result = {
        "schema": "PI0239_G2_CLOCK_CONTROL_REV2_V1", "status": "PASS_READ_ONLY" if passed else "FAIL_READ_ONLY",
        "scientific_interpretation": "NOT_INTERPRETED_AS_CAUSAL_UNTIL_SAME_PROCESS_CONTROL",
        "same_process": True, "case_count": len(controls), "shift_s": 17.0, "rows": controls,
        "model_weight_loads": 0, "model_forward_calls": 0, "get_action_calls": 0,
    }
    write_json(paths["output_root"] / "evidence" / "g2_clock_control_rev2.json", result)
    if not passed:
        raise ScienceFailure("SCIENCE_FAIL_CLOCK_ORIGIN: same-process control remains non-invariant")
    return result


def mature_args(args: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(
        output_root=args.output_root, cache_root=args.cache_root, p237_root=args.p237_root,
        run_root=args.run_root, train_dataset=args.train_dataset, calib_dataset=args.calib_dataset,
        prep_root=args.prep_root, vla_source=args.vla_source, stage10_root=args.stage10_root,
        baseline=args.baseline, cosmos_snapshot=args.cosmos_snapshot, modality_config=args.modality_config,
    )


def g3(args: SimpleNamespace) -> dict[str, Any]:
    os.environ.update(runtime_env(args.config, cuda=False))
    runtime = runtime_probe(args, args.config)
    result = mature.g3(mature_args(args))
    trace = json_load(args.output_root / "evidence" / "g3_actual_batch_trace.json")
    enriched = {"schema": "PI0239_G3_RUNTIME_IDENTITY_V1", "runtime": runtime, "g3_result": result, "trace_status": trace.get("status"), "parity_status": trace.get("parity_status"), "numerical_status": trace.get("numerical_status"), "model_weight_loads": 0, "model_forward_calls": 0, "get_action_calls": 0}
    write_json(args.output_root / "evidence" / "g3_runtime_identity.json", enriched)
    if trace.get("parity_status") != "PASS":
        raise ScienceFailure("SCIENCE_FAIL_TRAIN_INFER_CONTRACT: G3 physical/action parity failed")
    return enriched


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch-config", type=Path, required=True)
    parser.add_argument("--phase", choices=("preflight", "initialize", "reuse", "clock", "g3"), required=True)
    args = parser.parse_args()
    cfg = load_config(args.launch_config)
    ns = namespace(cfg)
    if args.phase == "preflight":
        print(json.dumps(preflight(ns, cfg, initialize=False), indent=2, default=str))
        return 0
    if args.phase == "initialize":
        print(json.dumps(preflight(ns, cfg, initialize=True), indent=2, default=str))
        return 0
    try:
        if args.phase == "reuse":
            result = verify_reuse(ns, cfg)
        elif args.phase == "clock":
            result = clock_control(ns)
        else:
            result = g3(ns)
        print(json.dumps({"phase": args.phase, "result": result}, indent=2, default=str))
        return 0
    except ScienceFailure as exc:
        evidence = ns.output_root / "evidence"
        evidence.mkdir(parents=True, exist_ok=True)
        write_json(evidence / f"{args.phase}_science_failure.json", {"status": "SCIENCE_FAIL", "error": str(exc), "selected_checkpoint": "NONE", "model_weight_loads": 0, "model_forward_calls": 0, "get_action_calls": 0})
        raise
    except Exception as exc:
        evidence = ns.output_root / "evidence"
        evidence.mkdir(parents=True, exist_ok=True)
        write_json(evidence / f"{args.phase}_engineering_failure.json", {"status": "ENGINEERING_FAILURE", "error_type": type(exc).__name__, "error": str(exc), "argv": sys.argv, "selected_checkpoint": "NONE", "model_weight_loads": 0, "model_forward_calls": 0, "get_action_calls": 0})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
