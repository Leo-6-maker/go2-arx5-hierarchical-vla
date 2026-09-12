#!/usr/bin/env python3
"""PI0152 C2 bounded Stage43 short closed-loop runner.

This is a mechanical support wrapper around the accepted Stage43 native
runner.  It preserves the Stage43 timing, SafetyGate, binding, and WBC code,
while late-binding only the already qualified N2 seed2223@2000 checkpoint,
N2 statistics, and FULL24 processor metadata.  The accepted Stage43 bundle is
never modified; a canonical overlay records its exact SHA and the allowlisted
late-bound identities.

The parent process is intentionally the native Python 3.8/IsaacGym runtime.
The Stage43 runner starts the existing Python 3.12 GR00T worker as a child.
No retry is performed after a model or simulator attempt starts.  Conditions
are run in the frozen C0..C4 order and stop on the first non-PASS result.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

PI_GATE = "PI_GATE_STATE_0152"
PI_COMMENT = 5647237688
PR_POINTER_COMMENT = 5647244579
RETURN_TOKEN = "CORELAB_PI0152_POLICY_CHARACTERIZATION_FOR_PI_REVIEW"
BRANCH = "stage45f/corelab-local-initial-state-pilot"
EXPECTED_N2_TREE = "09bb76cb2b74a48644c120201042671fb7eaa42da3603ffb84c62f49a35d6152"
EXPECTED_N2_STATS_SHA = "0c9450e6aeb0765eebc676414b2a6a84f71dddb7eea1de5453551f062b25fdbb"
EXPECTED_N2_STATS_PAYLOAD_SHA = "508d4a8845e3bf5928416a506b95046ec0bb6c2be3eecbd09c39684a00865572"
EXPECTED_FULL24_TREE = "2f3badd2019a68c6549a66f85b5486faf626078a3e438342f67aacc4edc95789"
EXPECTED_BASE_TREE = "8717e322de3dd8c66e52bdb81b37badb0f235c518de6acd465532ed01569a6a4"
EXPECTED_GROOT_COMMIT = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
EXPECTED_COSMOS_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"
CONDITIONS = ("C0_NOMINAL", "C1_X_PLUS_5CM", "C2_X_MINUS_5CM", "C3_Y_PLUS_5CM", "C4_Y_MINUS_5CM")
SCHEDULE = (49, 49, 49, 98)
TOTAL_FORWARDS = 20
TOTAL_UPDATES = 15
TOTAL_WBC_STEPS = 1225
MIN_FREE_MIB = 14 * 1024
ALLOWED_GPU_NAMES = ("RTX 4080 SUPER",)


class ClosedLoopError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ClosedLoopError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "required JSON is missing: %s" % path)
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), "JSON object required: %s" % path)
    return value


def write_json(path: Path, value: Mapping[str, Any], *, overwrite: bool = False) -> str:
    path = Path(path)
    require(overwrite or not path.exists(), "refusing to overwrite evidence: %s" % path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(canonical_json(value))
    os.replace(temporary, path)
    return sha256_file(path)


def _resolve_regular(path: Path, label: str) -> Path:
    path = Path(path).resolve(strict=True)
    require(path.is_file() and not path.is_symlink(), "%s is not a regular file: %s" % (label, path))
    return path


def _resolve_directory(path: Path, label: str) -> Path:
    path = Path(path).resolve(strict=True)
    require(path.is_dir() and not path.is_symlink(), "%s is not a regular directory: %s" % (label, path))
    return path


def _nvidia_snapshot() -> Dict[str, Any]:
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False,
    )
    require(query.returncode == 0, "nvidia-smi failed: %s" % query.stderr[-1000:])
    rows: List[Dict[str, Any]] = []
    for line in query.stdout.splitlines():
        if not line.strip():
            continue
        fields = [item.strip() for item in line.split(",")]
        require(len(fields) == 6, "malformed nvidia-smi row: %s" % line)
        rows.append({"index": int(fields[0]), "name": fields[1], "uuid": fields[2], "memory_total_mib": int(fields[3]), "memory_used_mib": int(fields[4]), "memory_free_mib": int(fields[5])})
    return {"hostname": platform.node(), "rows": rows, "query_sha256": sha256_bytes(query.stdout.encode("utf-8")), "query": query.stdout}


def _select_gpu(snapshot: Mapping[str, Any], requested: Optional[str]) -> Dict[str, Any]:
    rows = list(snapshot["rows"])
    candidates = [row for row in rows if any(name in str(row["name"]) for name in ALLOWED_GPU_NAMES) and int(row["memory_total_mib"]) >= 15000 and int(row["memory_free_mib"]) >= MIN_FREE_MIB]
    if requested:
        candidates = [row for row in candidates if row["uuid"] == requested]
    require(candidates, "no eligible live Corelab GPU; observed=%s" % rows)
    candidates.sort(key=lambda row: (-int(row["memory_total_mib"]), -int(row["memory_free_mib"]), int(row["index"])))
    return dict(candidates[0])


def _verify_n2_identity(n2_root: Path, n2_stats: Path, full24: Path) -> Dict[str, Any]:
    """Verify sealed identities using the training result and exact stats bytes.

    The training result is an already-sealed authority produced by the N2
    training lane.  This wrapper does not recompute or alter the checkpoint.
    """
    n2_root = _resolve_directory(n2_root, "N2 root")
    n2_stats = _resolve_regular(n2_stats, "N2 stats")
    full24 = _resolve_directory(full24, "FULL24 dataset")
    result_path = n2_root / "training_result.json"
    result = read_json(result_path)
    require(result.get("status") == "PASS" and int(result.get("seed")) == 2223 and int(result.get("training_steps")) == 2000, "N2 training result is not sealed seed2223@2000")
    require(result.get("base_tree_sha256") == EXPECTED_BASE_TREE and result.get("full24_tree_sha256") == EXPECTED_FULL24_TREE, "N2 training lineage identity drift")
    cp = result.get("checkpoint_2000")
    require(isinstance(cp, dict) and cp.get("status") == "PASS" and cp.get("tree_sha256") == EXPECTED_N2_TREE, "N2 checkpoint-2000 tree identity drift")
    checkpoint = _resolve_directory(n2_root / "checkpoint-2000", "N2 checkpoint-2000")
    stats_sha = sha256_file(n2_stats)
    require(stats_sha == EXPECTED_N2_STATS_SHA, "N2 stats SHA drift: %s" % stats_sha)
    stats = read_json(n2_stats)
    require(stats.get("statistics_payload_sha256") == EXPECTED_N2_STATS_PAYLOAD_SHA, "N2 stats payload SHA drift")
    return {
        "training_result_path": str(result_path), "training_result_sha256": sha256_file(result_path),
        "seed": 2223, "checkpoint_2000_path": str(checkpoint), "checkpoint_2000_tree_sha256": EXPECTED_N2_TREE,
        "n2_stats_path": str(n2_stats), "n2_stats_sha256": stats_sha, "n2_stats_payload_sha256": EXPECTED_N2_STATS_PAYLOAD_SHA,
        "full24_dataset_path": str(full24), "full24_tree_expected": EXPECTED_FULL24_TREE,
    }


def _make_overlay(base_bundle: Path, output_root: Path, *, n2_root: Path, n2_stats: Path, full24: Path, binding: Mapping[str, Any]) -> Tuple[Path, Dict[str, Any]]:
    import stage43_corelab_evidence as evidence

    base_bundle = _resolve_regular(base_bundle, "Stage43 bundle")
    base = read_json(base_bundle)
    require(base_bundle.read_bytes() == evidence.canonical_json_bytes(base), "Stage43 base bundle is not canonical JSON")
    evidence.validate_stage43_bundle(base)
    evidence.verify_stage43_bundle_files(base, repo_root=REPO)
    overlay = copy.deepcopy(base)
    artifacts = overlay["artifacts"]
    checkpoint = _resolve_directory(n2_root / "checkpoint-2000", "N2 checkpoint-2000")
    artifacts["structured_checkpoint"] = {"label": "N2(seed2223)@2000", "path": str(checkpoint), "tree_sha256": EXPECTED_N2_TREE}
    artifacts["model_path"] = {"label": "N2(seed2223)@2000", "path": str(checkpoint), "tree_sha256": EXPECTED_N2_TREE}
    artifacts["structured_stats"] = {"path": str(n2_stats), "sha256": EXPECTED_N2_STATS_SHA, "size_bytes": int(n2_stats.stat().st_size)}
    artifacts["dataset_dir"] = {"label": "FULL24", "path": str(full24), "tree_sha256": EXPECTED_FULL24_TREE}
    overlay["stage45f_pi0152_n2_overlay"] = {
        "schema": "CORELAB_STAGE45F_PI0152_N2_STAGE43_OVERLAY_V1",
        "base_bundle_path": str(base_bundle), "base_bundle_sha256": sha256_file(base_bundle),
        "allowlisted_changes": ["artifacts.structured_checkpoint", "artifacts.model_path", "artifacts.structured_stats", "artifacts.dataset_dir"],
        "n2_binding": dict(binding), "target_xyz_in_model_input": False,
    }
    path = output_root / "stage43_n2_overlay_bundle.json"
    digest = write_json(path, overlay)
    require(path.read_bytes() == evidence.canonical_json_bytes(read_json(path)), "N2 overlay is not canonical JSON")
    return path, {"path": str(path), "sha256": digest, "base_bundle_sha256": binding["stage43_bundle_sha256"], "allowlisted_changes": overlay["stage45f_pi0152_n2_overlay"]["allowlisted_changes"]}


def _prepare_env(runtime_root: Path, *, gpu_uuid: str, native_env_root: Path, isaacgym_python: Path, native_compat: Path, wbc_source: Path) -> Dict[str, Any]:
    native_env_root = _resolve_directory(native_env_root, "native runtime root")
    isaacgym_python = _resolve_directory(isaacgym_python, "IsaacGym Python root")
    native_compat = _resolve_directory(native_compat, "native compatibility root")
    wbc_source = _resolve_directory(wbc_source, "WBC source root")
    runtime_root.mkdir(parents=True, exist_ok=False)
    home = runtime_root / "home"; tmp = runtime_root / "tmp"; xdg_cache = runtime_root / "cache"; xdg_config = runtime_root / "config"; xdg_data = runtime_root / "data"
    torch_extensions = runtime_root / "torch_extensions"
    for path in (home, tmp, xdg_cache, xdg_config, xdg_data, torch_extensions):
        path.mkdir(parents=True, exist_ok=False)
    bind = isaacgym_python / "isaacgym" / "_bindings" / "linux-x86_64"
    pyroot = native_env_root / "lib" / "python3.8" / "site-packages"
    # The historical native_py38_4080 venv supplies WBC-only packages (notably
    # yapf), while its interpreter deliberately inherits Torch/Isaac libraries
    # from the adjacent public_native environment.  Keep that exact layering;
    # adding the venv site directory to PYTHONPATH would shadow stdlib typing.
    public_env = native_env_root.parent / "public_native" / "python_env"
    torch_pyroot = pyroot if (pyroot / "torch" / "lib").is_dir() else public_env / "lib" / "python3.8" / "site-packages"
    torch_lib = torch_pyroot / "torch" / "lib"
    env_lib = native_env_root / "lib" if (native_env_root / "lib" / "libpython3.8.so.1.0").is_file() else public_env / "lib"
    ninja = native_env_root / "bin" / "ninja"
    require(bind.is_dir() and torch_lib.is_dir() and env_lib.is_dir() and ninja.is_file(), "native runtime libraries/ninja are incomplete")
    current_path = os.environ.get("PATH", "")
    # Updating PYTHONPATH in-process does not retroactively change sys.path;
    # insert the explicitly bound support roots as well so the native runner
    # sees the sealed pytorch3d compatibility shim without shadowing stdlib
    # typing from the historical venv.
    for import_root in (REPO / "scripts", REPO / "src", wbc_source, isaacgym_python, native_compat):
        if str(import_root) not in sys.path:
            sys.path.insert(0, str(import_root))
    os.environ.update({
        "CUDA_VISIBLE_DEVICES": gpu_uuid, "NVIDIA_VISIBLE_DEVICES": gpu_uuid,
        "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "HOME": str(home), "TMPDIR": str(tmp), "TMP": str(tmp), "TEMP": str(tmp),
        "XDG_CACHE_HOME": str(xdg_cache), "XDG_CONFIG_HOME": str(xdg_config), "XDG_DATA_HOME": str(xdg_data),
        "TORCH_EXTENSIONS_DIR": str(torch_extensions), "MAX_JOBS": "1", "CXX": "/usr/bin/g++-10",
        "PATH": str(native_env_root / "bin") + os.pathsep + current_path,
        "PYTHONPATH": os.pathsep.join((str(REPO / "scripts"), str(REPO / "src"), str(wbc_source), str(isaacgym_python), str(native_compat))),
        "LD_LIBRARY_PATH": os.pathsep.join((str(bind), str(torch_lib), str(env_lib))),
        "LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libstdc++.so.6",
        "CUDA_CACHE_PATH": str(xdg_cache / "cuda"), "TORCH_HOME": str(xdg_cache / "torch"),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
        "GROOT_HF_LOCAL_FIRST": "1", "GROOT_PATCH_MISTRAL": "1", "NO_ALBUMENTATIONS_UPDATE": "1",
    })
    return {
        "native_env_root": str(native_env_root), "native_python": sys.executable, "isaacgym_python": str(isaacgym_python),
        "native_compat_root": str(native_compat), "wbc_source_root": str(wbc_source), "bind_dir": str(bind), "torch_lib_dir": str(torch_lib),
        "env_lib_dir": str(env_lib), "ninja": str(ninja), "torch_extensions_dir": str(torch_extensions), "gpu_uuid": gpu_uuid,
        "python": platform.python_version(), "environment_isolated": True,
    }


def _trajectory_artifacts(output_root: Path, condition_ids: Sequence[str], condition_roots: Mapping[str, Path]) -> Dict[str, Any]:
    """Persist desired/reference and achieved EEF rows without changing source runs."""
    import numpy as np

    arrays: Dict[str, Any] = {}
    condition_meta: Dict[str, Any] = {}
    for condition in condition_ids:
        root = condition_roots[condition]
        segments: List[Mapping[str, Any]] = []
        for name in ("rows_before_handoff.json", "rows_after_handoff.json"):
            payload = read_json(root / name)
            for segment in payload.get("segments", []):
                require(isinstance(segment, list), "row segment is not a list: %s" % condition)
                segments.extend(segment)
        desired = np.asarray([row["desired_pos"] for row in segments], dtype=np.float64)
        achieved = np.asarray([row["achieved_pos"] for row in segments], dtype=np.float64)
        sim_time = np.asarray([row["sim_time"] for row in segments], dtype=np.float64)
        require(desired.shape == achieved.shape == (245, 3) and sim_time.shape == (245,) and np.isfinite(desired).all() and np.isfinite(achieved).all(), "trajectory row ABI drift: %s" % condition)
        arrays[condition + "_desired_reference"] = desired
        arrays[condition + "_achieved_eef"] = achieved
        arrays[condition + "_sim_time"] = sim_time
        condition_meta[condition] = {"rows": 245, "desired_reference_shape": list(desired.shape), "achieved_eef_shape": list(achieved.shape), "desired_reference_array_sha256": sha256_bytes(desired.tobytes()), "achieved_eef_array_sha256": sha256_bytes(achieved.tobytes())}
    path = output_root / "short_closed_loop_trajectories.npz"
    np.savez_compressed(path, **arrays)
    return {"path": str(path), "sha256": sha256_file(path), "conditions": condition_meta}


def _run(args: argparse.Namespace) -> Dict[str, Any]:
    import stage43_corelab_evidence as evidence
    import stage43_corelab_initial_condition as runner

    output_root = Path(args.output_root).resolve()
    require(not output_root.exists(), "C2 output root must be fresh: %s" % output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    base_bundle = _resolve_regular(Path(args.stage43_bundle), "Stage43 bundle")
    n2_root = _resolve_directory(Path(args.n2_root), "N2 root")
    n2_stats = _resolve_regular(Path(args.n2_stats), "N2 stats")
    full24 = _resolve_directory(Path(args.full24_dataset), "FULL24 dataset")
    binding = _verify_n2_identity(n2_root, n2_stats, full24)
    binding["stage43_bundle_sha256"] = sha256_file(base_bundle)
    base_doc = read_json(base_bundle)
    evidence.validate_stage43_bundle(base_doc)
    binding["stage43_repository_head"] = base_doc["repository"]["head"]
    overlay_path, overlay_doc = _make_overlay(base_bundle, output_root, n2_root=n2_root, n2_stats=n2_stats, full24=full24, binding=binding)
    # Patch only the late-bound model/stats identity in the bundle verifier and
    # the decoder's hash guard.  The decoder math and Stage43
    # protocol/SafetyGate/WBC code remain byte-identical; the original bundle
    # remains immutable.  The base bundle was validated above with its
    # historical STRUCTURED@1000 values.  N2 has its own sealed statistics
    # route, so the hash guard must be rebound together with the artifact
    # rather than failing closed on the historical Stage40-only SHA.
    evidence.STRUCTURED_CHECKPOINT_TREE_SHA256 = EXPECTED_N2_TREE
    evidence.STRUCTURED_STATS_SHA256 = EXPECTED_N2_STATS_SHA
    import go2arx5_vla.stage40_structured_decode as structured_decode
    structured_decode.EXPECTED_STRUCTURED_STATS_SHA256 = EXPECTED_N2_STATS_SHA
    gpu_snapshot = _nvidia_snapshot()
    gpu = _select_gpu(gpu_snapshot, args.gpu_uuid)
    env_info = _prepare_env(output_root / "runtime", gpu_uuid=str(gpu["uuid"]), native_env_root=Path(args.native_env_root), isaacgym_python=Path(args.isaacgym_python), native_compat=Path(args.native_compat_root), wbc_source=Path(args.wbc_source_root))
    c2_binding = {
        "schema": "CORELAB_STAGE45F_PI0152_C2_BINDING_V1",
        "authority": {"pi_gate": PI_GATE, "issue_comment": PI_COMMENT, "pr_pointer_comment": PR_POINTER_COMMENT},
        "repository": {"branch": BRANCH, "current_head": str(args.current_head)},
        "overlay": overlay_doc,
        "n2": binding,
        "gpu_admission": {"selected": gpu, "snapshot": gpu_snapshot},
        "runtime": env_info,
        "protocol": {
            "condition_order": list(CONDITIONS), "schedule_native_ticks": list(SCHEDULE),
            "forwards_per_condition": 4, "reference_updates_per_condition": 3,
            "nominal_total_forwards": TOTAL_FORWARDS, "nominal_total_updates": TOTAL_UPDATES,
            "nominal_total_wbc_steps": TOTAL_WBC_STEPS,
        },
        "execution_boundary": {"training": False, "validation_e_read": False, "holdout_f_read": False, "hardware": False, "icl": False, "merge": False},
    }
    write_json(output_root / "c2_binding.json", c2_binding)
    # Late-bind runner module constants after the overlay has been sealed.
    runner.evidence.STRUCTURED_CHECKPOINT_TREE_SHA256 = EXPECTED_N2_TREE
    runner.evidence.STRUCTURED_STATS_SHA256 = EXPECTED_N2_STATS_SHA
    condition_roots: Dict[str, Path] = {}
    condition_returns: List[Dict[str, Any]] = []
    status = "COMPLETED"
    error: Optional[Dict[str, Any]] = None
    for index, condition in enumerate(CONDITIONS):
        # Re-admit before each fresh worker/native start.  A busy shared GPU
        # is a hard stop, never a reason to kill or reprioritize another job.
        live = _select_gpu(_nvidia_snapshot(), str(gpu["uuid"]))
        condition_root = output_root / "conditions" / ("%02d_%s" % (index, condition))
        run_args = argparse.Namespace(
            bundle=overlay_path, output_dir=condition_root, condition_id=condition,
            repository_head=base_doc["repository"]["head"], gpu_uuid=str(gpu["uuid"]),
            source_root=Path(args.wbc_source_root), trajectory=Path(base_doc["artifacts"]["trajectory"]["path"]),
            arm=Path(base_doc["artifacts"]["wbc_arm_checkpoint"]["path"]), dog=Path(base_doc["artifacts"]["wbc_dog_checkpoint"]["path"]),
            stage10_root=Path(base_doc["artifacts"]["stage10_root"]["path"]), base_model_path=Path(base_doc["artifacts"]["base_model_path"]["path"]),
            model_path=n2_root / "checkpoint-2000", dataset_dir=full24,
            modality_config=Path(base_doc["artifacts"]["modality_config"]["path"]), cosmos_snapshot=Path(base_doc["artifacts"]["cosmos_snapshot_path"]["path"]),
            structured_stats=n2_stats, model_python=Path(args.model_python), wbc_source_root=Path(args.wbc_source_root),
            sim_device="cuda:0", seed=4242, response_timeout_s=float(args.response_timeout_s), worker_ready_timeout_s=float(args.worker_ready_timeout_s),
        )
        write_json(output_root / "conditions" / ("%02d_%s_admission.json" % (index, condition)), {"condition_id": condition, "gpu": live, "model_attempt_started": False})
        try:
            rc = int(runner._run(run_args))
        except Exception as exc:
            rc = 2
            error = {"type": type(exc).__name__, "message": str(exc)}
        result_pointer = read_json(condition_root / "run_result.json") if (condition_root / "run_result.json").is_file() else {"status": "BLOCKED", "error": error}
        condition_status = str(result_pointer.get("status"))
        condition_returns.append({"condition_id": condition, "status": condition_status, "runner_return_code": rc, "output_dir": str(condition_root), "run_result_sha256": sha256_file(condition_root / "run_result.json") if (condition_root / "run_result.json").is_file() else None, "live_gpu_at_admission": live, "error": result_pointer.get("error")})
        if condition_status == "PASS":
            condition_roots[condition] = condition_root
        else:
            status = "PARTIAL" if condition_roots else "BLOCKED"
            error = result_pointer.get("error") or error
            break
    trajectory = _trajectory_artifacts(output_root, tuple(condition_roots), condition_roots) if condition_roots else None
    counters = {"conditions_started": len(condition_returns), "conditions_passed": len(condition_roots), "conditions_expected": 5, "gr00t_forwards": len(condition_roots) * 4, "reference_updates": len(condition_roots) * 3, "wbc_steps": len(condition_roots) * sum(SCHEDULE), "model_attempts": len(condition_returns), "training": False, "gpu": bool(condition_returns), "simulator": bool(condition_returns), "wbc": bool(condition_returns), "hardware": False, "validation_e_read": False, "holdout_f_read": False}
    result = {"schema": "CORELAB_STAGE45F_PI0152_POLICY_CHARACTERIZATION_V2", "status": status, "token": RETURN_TOKEN, "authority": {"pi_gate": PI_GATE, "issue_comment": PI_COMMENT, "pr_pointer_comment": PR_POINTER_COMMENT}, "repository": {"branch": BRANCH, "current_head": str(args.current_head), "stage43_runner_head": base_doc["repository"]["head"]}, "binding": {"overlay": overlay_doc, "n2": binding, "fixed_target_world_m": [0.3, 0.1, 0.58], "target_xyz_in_model_input": False}, "gpu_runtime": {"initial_snapshot": gpu_snapshot, "selected_gpu": gpu, "isolated_runtime": env_info}, "conditions": condition_returns, "trajectory_artifacts": trajectory, "protocol": {"condition_order": list(CONDITIONS), "schedule_native_ticks": list(SCHEDULE), "native_dt_expected_s": 0.02, "handoff_s": 0.9750000000000001, "seed": 4242, "stop_on_first_failure": True}, "counters": counters, "error": error, "execution_boundary": {"new_training": False, "validation_e_read": False, "holdout_f_read": False, "icl": False, "hardware": False, "merge": False}, "claim_boundary": "N2 seed2223@2000 bounded Stage43 simulator/WBC short closed-loop characterization; no policy/training/route-selection claim", "recommendation": "Use the already sealed offline decoded failure plus this bounded closed-loop outcome to choose exactly one PI-approved intervention; do not infer a winner or auto-start training.", "timestamp_utc": time.time()}
    result_sha = write_json(output_root / "pi0152_policy_characterization.json", result)
    write_json(output_root / "pi0152_policy_characterization.json.sha256", {"file": "pi0152_policy_characterization.json", "sha256": result_sha})
    result["result_sha256"] = result_sha
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage43-bundle", type=Path, required=True)
    parser.add_argument("--n2-root", type=Path, required=True)
    parser.add_argument("--n2-stats", type=Path, required=True)
    parser.add_argument("--full24-dataset", type=Path, required=True)
    parser.add_argument("--native-env-root", type=Path, required=True)
    parser.add_argument("--isaacgym-python", type=Path, required=True)
    parser.add_argument("--native-compat-root", type=Path, required=True)
    parser.add_argument("--wbc-source-root", type=Path, required=True)
    parser.add_argument("--model-python", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu-uuid", default=None)
    parser.add_argument("--current-head", required=True)
    parser.add_argument("--response-timeout-s", type=float, default=300.0)
    parser.add_argument("--worker-ready-timeout-s", type=float, default=1200.0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    result = _run(args)
    print(json.dumps({"status": result["status"], "token": result["token"], "conditions_passed": result["counters"]["conditions_passed"], "gr00t_forwards": result["counters"]["gr00t_forwards"], "wbc_steps": result["counters"]["wbc_steps"], "output_root": str(Path(args.output_root).resolve()), "result_sha256": result.get("result_sha256")}, sort_keys=True), flush=True)
    return 0 if result["status"] == "COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
