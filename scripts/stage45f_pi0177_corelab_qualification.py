#!/usr/bin/env python3
"""PI0177 Corelab CPU/static qualification and one TRAIN24 canary.

The preflight is model-free and binds the exact Legacy(seed2222)@2000
checkpoint, FULL24 inputs, GR00T/Cosmos/base/stats/modality identities, and
the frozen branch.  Only an admitted ``--canary`` may import Torch/GR00T; it
performs one TRAIN24 first-window load/conditioning/four-step action-head
check and never reads Validation-E or submits H200/PBS work.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


PI_GATE = 5661695527
SOURCE_HEAD = "ce3c0179f3974648cdf6527e145d0328b0ac9262"
BRANCH = "stage45f/localpc-validation-e-hopper-resume"
TARGET_ARM = "Legacy(seed2222)@2000"
TARGET_TREE = "2113a6a3b2a91233a42214dbb017c0136acde282fda2a418066f9317e78981a5"
TARGET_ID = "red_target_holdout_e"
TRAIN_TARGETS = ("red_target_train_a", "red_target_train_b", "red_target_train_c", "red_target_train_d")
TRAIN_WINDOW_IDS = tuple(
    f"stage22_expert_{target}:expert-{index:04d}"
    for target in TRAIN_TARGETS
    for index in range(6)
)
RAW_SHAPE = (1, 40, 132)
TIMESTEP_IDS = (0, 250, 500, 750)
NOISE_SEED = 4242
GROOT_COMMIT = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
GROOT_PAYLOAD_TREE = "c0d57a6a74ab24c48abdd1c13420e9d1c893dc0d7b5ca370fa9b6f7af32f5f1b"
GROOT_MANIFEST_SHA = "96b883e56b54f763bec60d3a336632ad531f4fc52d099f803e5f9efba60eab11"
COSMOS_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"
COSMOS_TREE = "1cfec77c944cba341ee5d7c31b4be6bb615582364bae48f18c5686e6ec23e8a9"
MODALITY_SHA = "e61f9429ecc21ceab51d9d26cadc9018691eb8b5c74d60aec4abf7faee16f031"
STATS_SHA = "3b3a2d51e685a0e580de4ebdfa05abcbcddfb7dc33b8d3a659994cdcd1bb0337"
STATS_PAYLOAD_SHA = "752898989bb3308e663c4b6cf88b3cc1c139897a6860e6aed1d9d708d0f6fd75"
BASE_TREE = "8717e322de3dd8c66e52bdb81b37badb0f235c518de6acd465532ed01569a6a4"
BASE_CONFIG_SHA = "85c1b4690ae090559e79a45193e598b65d6146eedf14750884da65e6d31032be"
BASE_STATS_SHA = "c97b1b07a82a8a8858771d56d278732d97dd8eae506032c146c77eb53828afc9"
DATASET_TREE = "2f3badd2019a68c6549a66f85b5486faf626078a3e438342f67aacc4edc95789"
SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parent
FORBIDDEN_TOP_LEVEL = {"torch", "transformers", "gr00t", "numpy", "cv2", "cuda"}
FORBIDDEN_CALLS = {"backward", "get_action", "qsub", "sbatch", "srun", "train", "save_checkpoint", "save_model", "save_pretrained"}
ALLOWED_CHANGES = {
    "scripts/stage45f_pi0177_validation_e_hopper_resume.py",
    "scripts/stage45f_pi0177_corelab_qualification.py",
    "manifests/STAGE45F_PI0177_VALIDATION_E_GEN1_HOPPER_RESUME_V1.json",
    "tests/test_stage45f_pi0177_hopper_resume.py",
    "hpc/pbs/stage45f_pi0177_validation_e_gen1_h200.pbs",
}


class QualificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise QualificationError(message)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    path = Path(path)
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not stat.S_ISLNK(mode), f"regular JSON file required: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any, *, overwrite: bool = False) -> None:
    path = Path(path)
    if not overwrite:
        require(not path.exists() and not path.is_symlink(), f"fresh record required: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    try:
        directory_fd = os.open(path.parent, os.O_RDONLY)
    except OSError:
        directory_fd = None
    if directory_fd is not None:
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def regular_file(path: Path, label: str) -> Path:
    path = Path(path)
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not stat.S_ISLNK(mode), f"{label} is not a regular file: {path}")
    return path.resolve()


def regular_dir(path: Path, label: str) -> Path:
    path = Path(path)
    mode = path.lstat().st_mode
    require(stat.S_ISDIR(mode) and not stat.S_ISLNK(mode), f"{label} is not a regular directory: {path}")
    return path.resolve()


def run_text(argv: Sequence[str]) -> str:
    completed = subprocess.run(list(argv), capture_output=True, text=True, check=False)
    require(completed.returncode == 0, f"command failed: {' '.join(argv)}: {completed.stderr[-1000:]}")
    return completed.stdout.strip()


def _module_static(path: Path) -> dict[str, Any]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    top_imports: set[str] = set()
    calls: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_imports.add(node.module.split(".", 1)[0])
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ""
            if name in FORBIDDEN_CALLS:
                calls.append(name)
    return {
        "path": str(path),
        "top_level_forbidden_imports": sorted(top_imports.intersection(FORBIDDEN_TOP_LEVEL)),
        "forbidden_call_hits": sorted(set(calls)),
    }


def static_guard() -> dict[str, Any]:
    self_audit = _module_static(Path(__file__))
    runner = _module_static(REPO / "scripts/stage45f_pi0177_validation_e_hopper_resume.py")
    status = "PASS" if not self_audit["top_level_forbidden_imports"] and not self_audit["forbidden_call_hits"] and not runner["top_level_forbidden_imports"] and not runner["forbidden_call_hits"] else "FAIL"
    return {
        "schema": "STAGE45F_PI0177_CORELAB_QUALIFICATION_STATIC_GUARD_V1",
        "status": status,
        "qualification_module": self_audit,
        "hopper_runner": runner,
        "model_imported": False,
        "validation_e_model_calls": 0,
        "training": False,
        "pbs_submissions": 0,
        "h200_submissions": 0,
    }


def checkout_audit(repo: Path, expected_head: str) -> dict[str, Any]:
    repo = regular_dir(repo, "qualification repository")
    root = run_text(["git", "-C", str(repo), "rev-parse", "--show-toplevel"])
    branch = run_text(["git", "-C", str(repo), "branch", "--show-current"])
    head = run_text(["git", "-C", str(repo), "rev-parse", "HEAD"])
    status = run_text(["git", "-C", str(repo), "status", "--porcelain=v1", "--untracked-files=all"])
    require(Path(root).resolve() == repo, f"repository root drift: {root}")
    require(branch == BRANCH and head == expected_head and not status, "qualification checkout identity/cleanliness drift")
    run_text(["git", "-C", str(repo), "merge-base", "--is-ancestor", SOURCE_HEAD, head])
    changed = run_text(["git", "-C", str(repo), "diff", "--name-only", f"{SOURCE_HEAD}..{head}"]).splitlines()
    require(set(changed) <= ALLOWED_CHANGES, f"unexpected PI0177 changes: {changed}")
    return {"root": root, "branch": branch, "head": head, "clean": True, "changed_paths": changed}


def _imports(repo: Path) -> tuple[Any, Any, Any, Any]:
    for path in (repo / "src", repo / "scripts"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import stage45c_h200_eval_runtime_adapter as runtime
    import stage45f_pi0159_validation_e_continuation as continuation
    import stage45f_vanda_validation_e_admission as admission
    import stage45f_vanda_validation_e_gen1 as evaluator
    return runtime, continuation, admission, evaluator


def artifact_audit(config: Mapping[str, Any], repo: Path) -> dict[str, Any]:
    runtime, continuation, admission, evaluator = _imports(repo)
    freeze = admission.static_plan(repo / "manifests/STAGE45F_PI0154_VALIDATION_E_GEN1_RECIPE_FREEZE_V1.json")
    base = regular_dir(Path(str(config["base_model"])), "BASE model")
    base_tree, base_entries = evaluator.tree_sha256(base, "payload/GR00T-N1.7-3B")
    require(base_tree == BASE_TREE, f"BASE tree drift: {base_tree}")
    base_config = regular_file(base / "processor_config.json", "BASE processor config")
    base_stats = regular_file(base / "statistics.json", "BASE processor statistics")
    require(sha256_file(base_config) == BASE_CONFIG_SHA and sha256_file(base_stats) == BASE_STATS_SHA, "BASE processor identity drift")
    dataset = regular_dir(Path(str(config["dataset"])), "FULL24 TRAIN dataset")
    dataset_audit = runtime._verify_dataset(dataset)
    require(dataset_audit["tree_sha256"] == DATASET_TREE, "FULL24 dataset identity drift")
    stats = regular_file(Path(str(config["stats"])), "FULL24 runtime stats")
    stats_audit = runtime._verify_runtime_stats(stats)
    require(sha256_file(stats) == STATS_SHA, "FULL24 stats SHA drift")
    groot = regular_dir(Path(str(config["groot_source"])), "GR00T source")
    groot_head = run_text(["git", "-C", str(groot), "rev-parse", "HEAD"])
    require(groot_head == GROOT_COMMIT, f"GR00T source HEAD drift: {groot_head}")
    manifest = regular_file(Path(str(config["groot_manifest"])), "GR00T source manifest")
    require(sha256_file(manifest) == GROOT_MANIFEST_SHA, "GR00T manifest SHA drift")
    manifest_text = manifest.read_text(encoding="utf-8")
    require(GROOT_COMMIT in manifest_text and GROOT_PAYLOAD_TREE in manifest_text, "GR00T manifest provenance drift")
    modality = regular_file(Path(str(config["modality"])), "modality")
    require(sha256_file(modality) == MODALITY_SHA, "modality SHA drift")
    cosmos = regular_dir(Path(str(config["cosmos_root"])), "Cosmos cache")
    cosmos_tree, cosmos_entries = continuation.cosmos_tree_digest(cosmos)
    require(cosmos_tree == COSMOS_TREE, f"Cosmos tree drift: {cosmos_tree}")
    refs = regular_file(cosmos / "refs" / "main", "Cosmos refs/main")
    require(refs.read_text(encoding="utf-8").strip() == COSMOS_REVISION, "Cosmos revision drift")
    checkpoint = regular_dir(Path(str(config["checkpoint"])), TARGET_ARM)
    checkpoint_tree, checkpoint_entries = evaluator.tree_sha256(checkpoint, checkpoint.name)
    require(checkpoint_tree == TARGET_TREE, f"Legacy checkpoint tree drift: {checkpoint_tree}")
    trainer_state = load_json(checkpoint / "trainer_state.json")
    require(int(trainer_state.get("global_step", -1)) == 2000, "Legacy checkpoint global_step drift")
    snapshot_root = regular_dir(Path(str(config["train_snapshot_root"])), "TRAIN24 snapshot root")
    train_snapshots = runtime._verify_snapshots(snapshot_root)
    require([item["window_id"] for item in train_snapshots] == list(TRAIN_WINDOW_IDS), "TRAIN24 snapshot order drift")
    snapshot_summary = [{"window_id": item["window_id"], "path": item["path"], "sha256": item["sha256"], "target_id": item["target_id"]} for item in train_snapshots]
    return {
        "schema": "STAGE45F_PI0177_CORELAB_CPU_STATIC_ARTIFACT_AUDIT_V1",
        "status": "PASS",
        "freeze": freeze,
        "base": {"path": str(base), "tree_sha256": base_tree, "file_count": len(base_entries), "processor_config_sha256": sha256_file(base_config), "statistics_sha256": sha256_file(base_stats)},
        "dataset": dataset_audit,
        "stats": {"path": str(stats), "sha256": sha256_file(stats), "payload_sha256": STATS_PAYLOAD_SHA, "runtime_audit": stats_audit},
        "groot": {"path": str(groot), "git_head": groot_head, "manifest": str(manifest), "manifest_sha256": GROOT_MANIFEST_SHA},
        "modality": {"path": str(modality), "sha256": MODALITY_SHA},
        "cosmos": {"path": str(cosmos), "tree_sha256": cosmos_tree, "entry_count": cosmos_entries, "revision": COSMOS_REVISION},
        "checkpoint": {"label": TARGET_ARM, "path": str(checkpoint), "tree_sha256": checkpoint_tree, "file_count": len(checkpoint_entries), "total_bytes": sum(int(item["bytes"]) for item in checkpoint_entries), "global_step": 2000, "trainer_state_sha256": sha256_file(checkpoint / "trainer_state.json")},
        "train24_snapshots": snapshot_summary,
        "train24_window_count": len(snapshot_summary),
        "target_xyz_in_model_input": False,
        "validation_e_model_calls": 0,
        "holdout_f_read": False,
        "training": False,
    }


def preflight(config_path: Path, repo: Path) -> dict[str, Any]:
    config = load_json(config_path)
    require(isinstance(config, Mapping) and config.get("schema") == "STAGE45F_PI0177_CORELAB_QUALIFICATION_V1", "qualification config drift")
    require(config.get("pi_gate") == PI_GATE and config.get("source_head") == SOURCE_HEAD and config.get("branch") == BRANCH, "qualification authority drift")
    checkout = checkout_audit(repo, str(config["expected_head"]))
    guard = static_guard()
    require(guard["status"] == "PASS", "qualification static guard failed")
    output_root = Path(str(config["output_root"])).resolve()
    require(output_root.parent.is_dir(), f"qualification output parent unavailable: {output_root.parent}")
    require(not output_root.exists() and not output_root.is_symlink(), f"qualification output is not fresh: {output_root}")
    output_root.mkdir(parents=False, exist_ok=False)
    audit = artifact_audit(config, repo)
    record = {
        "schema": "STAGE45F_PI0177_CORELAB_CPU_STATIC_QUALIFICATION_V1",
        "status": "PASS_CPU_STATIC",
        "timestamp_utc": now_utc(),
        "pi_gate": PI_GATE,
        "source_head": SOURCE_HEAD,
        "execution_head": config["expected_head"],
        "execution_branch": BRANCH,
        "checkout": checkout,
        "static_guard": guard,
        "artifact_audit": audit,
        "canary": {"status": "NOT_RUN_UNTIL_CPU_STATIC_PASS", "validation_e_model_calls": 0},
        "validation_e_model_calls": 0,
        "training": False,
        "backward": False,
        "optimizer_steps": 0,
        "checkpoint_writes": 0,
        "holdout_f_read": False,
        "h200_submissions": 0,
        "pbs_submissions": 0,
        "wbc_steps": 0,
        "simulator_steps": 0,
        "hardware_steps": 0,
    }
    write_json(output_root / "cpu_static_qualification.json", record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return record


def _gpu_snapshot() -> dict[str, Any]:
    query = "index,name,uuid,memory.total,memory.used,memory.free"
    completed = subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False)
    require(completed.returncode == 0, f"nvidia-smi failed: {completed.stderr[-500:]}")
    rows = []
    for line in completed.stdout.splitlines():
        values = [item.strip() for item in line.split(",")]
        if len(values) == 6:
            rows.append({"index": int(values[0]), "name": values[1], "uuid": values[2], "memory_total_mib": int(values[3]), "memory_used_mib": int(values[4]), "memory_free_mib": int(values[5])})
    require(rows, "nvidia-smi returned no GPUs")
    return {"query": query, "gpus": rows, "captured_at_utc": now_utc()}


def canary(config_path: Path, repo: Path) -> dict[str, Any]:
    config = load_json(config_path)
    output_root = Path(str(config["output_root"])).resolve()
    preflight_record = load_json(regular_file(output_root / "cpu_static_qualification.json", "CPU/static qualification"))
    require(preflight_record.get("status") == "PASS_CPU_STATIC", "CPU/static qualification is not PASS")
    require(not (output_root / "canary_result.json").exists(), "canary result already exists; no repeat allowed")
    runtime, _continuation, _admission, _evaluator = _imports(repo)
    gpu_before = _gpu_snapshot()
    selected_index = int(config.get("gpu_index", 0))
    selected_before = next((item for item in gpu_before["gpus"] if item["index"] == selected_index), None)
    require(selected_before is not None, f"configured GPU index is unavailable: {selected_index}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_index)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["PYTHONNOUSERSITE"] = "1"
    import numpy as np
    import torch
    import stage26_h1_c1_fixed_noise_local_flow as stage26

    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    dataset = regular_dir(Path(str(config["dataset"])), "FULL24 TRAIN dataset")
    groot = regular_dir(Path(str(config["groot_source"])), "GR00T source")
    base = regular_dir(Path(str(config["base_model"])), "BASE model")
    modality = regular_file(Path(str(config["modality"])), "modality")
    cosmos = regular_dir(Path(str(config["cosmos_root"])), "Cosmos cache")
    checkpoint = regular_dir(Path(str(config["checkpoint"])), TARGET_ARM)
    start = time.perf_counter()
    model = None
    conditioning_calls = 0
    action_head_calls = 0
    raw_sha = None
    raw_array_sha = None
    try:
        processor_started = time.perf_counter()
        processor, embodiment, modality_configs = stage26._build_runtime_processor(
            groot_source=groot,
            base_checkpoint=base,
            dataset_dir=dataset,
            modality_path=modality,
            cosmos_snapshot=cosmos / "snapshots" / COSMOS_REVISION,
        )
        processor_elapsed = time.perf_counter() - processor_started
        records = preflight_record["artifact_audit"]["train24_snapshots"]
        windows = runtime._prepare_train24_windows(processor, embodiment, modality_configs, dataset, records[:1])
        require(len(windows) == 1 and windows[0]["window_id"] == TRAIN_WINDOW_IDS[0], "TRAIN24 first-window preparation drift")
        torch_module, load_model = runtime._load_torch_runtime()
        load_started = time.perf_counter()
        model = load_model(checkpoint, "cuda:0")
        model.eval()
        load_elapsed = time.perf_counter() - load_started
        generator = torch.Generator(device="cpu")
        generator.manual_seed(NOISE_SEED)
        noise = torch.randn(RAW_SHAPE, generator=generator, device="cpu", dtype=torch.float32).contiguous()
        condition_started = time.perf_counter()
        _, action_input, processed, state_features = runtime._prepare_model_conditioning(model, windows[0]["model_inputs"])
        conditioning_elapsed = time.perf_counter() - condition_started
        conditioning_calls = 1
        current = noise.to(device=processed.backbone_features.device, dtype=next(model.parameters()).dtype)
        with torch.inference_mode():
            for timestep in TIMESTEP_IDS:
                velocity = runtime._offline_step(model.action_head, processed, action_input, state_features, current, int(timestep))
                require(tuple(int(value) for value in velocity.shape) == RAW_SHAPE and bool(torch.isfinite(velocity).all().item()), "TRAIN24 canary action-head output drift")
                current = current + 0.25 * velocity
                action_head_calls += 1
        raw = current.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy().astype(np.float32, copy=False)
        require(raw.shape == RAW_SHAPE and bool(np.isfinite(raw).all()), "TRAIN24 canary raw output drift")
        raw_sha = sha256_file(_write_canary_array(output_root / "train24_first_window_raw.npy", raw))
        raw_array_sha = hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest()
        result = {
            "schema": "STAGE45F_PI0177_CORELAB_TRAIN24_FIRST_WINDOW_CANARY_V1",
            "status": "PASS",
            "timestamp_utc": now_utc(),
            "pi_gate": PI_GATE,
            "host": run_text(["hostname"]),
            "gpu_index_requested": selected_index,
            "gpu_before": gpu_before,
            "gpu_selected": selected_before,
            "runtime": {"python": str(Path(sys.executable).resolve()), "python_version": sys.version, "torch_version": str(torch.__version__), "cuda_version": str(torch.version.cuda)},
            "checkpoint": {"label": TARGET_ARM, "path": str(checkpoint), "tree_sha256": TARGET_TREE, "global_step": 2000},
            "train_window_id": TRAIN_WINDOW_IDS[0],
            "model_loads": 1,
            "conditioning": conditioning_calls,
            "action_head_calls": action_head_calls,
            "model_side_total": conditioning_calls + action_head_calls,
            "model_calls": 5,
            "processor_build_elapsed_s": processor_elapsed,
            "model_load_elapsed_s": load_elapsed,
            "conditioning_elapsed_s": conditioning_elapsed,
            "raw_shape": list(RAW_SHAPE),
            "raw_dtype": "float32",
            "raw_file": "train24_first_window_raw.npy",
            "raw_file_sha256": raw_sha,
            "raw_array_sha256": raw_array_sha,
            "peak_cuda_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_cuda_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "validation_e_model_calls": 0,
            "training": False,
            "backward": False,
            "optimizer_steps": 0,
            "checkpoint_writes": 0,
            "holdout_f_read": False,
            "pbs_submissions": 0,
            "h200_submissions": 0,
            "wbc_steps": 0,
            "simulator_steps": 0,
            "hardware_steps": 0,
        }
    finally:
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()
    result["gpu_after"] = _gpu_snapshot()
    result["elapsed_s"] = time.perf_counter() - start
    write_json(output_root / "canary_result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _write_canary_array(path: Path, array: Any) -> Path:
    import numpy as np

    require(not path.exists() and not path.is_symlink(), f"canary array is not fresh: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return path


def plan() -> dict[str, Any]:
    return {
        "schema": "STAGE45F_PI0177_CORELAB_QUALIFICATION_PLAN_V1",
        "status": "READY_NOT_EXECUTED",
        "pi_gate": PI_GATE,
        "source_head": SOURCE_HEAD,
        "branch": BRANCH,
        "cpu_static": {"validation_e_model_calls": 0, "h200_submissions": 0, "pbs_submissions": 0},
        "canary": {"train_cohort": "FULL24", "windows": 1, "model_loads": 1, "conditioning": 1, "action_head": 4, "model_side_total": 5},
        "validation_e_model_calls": 0,
        "training": 0,
        "holdout_f_read": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--plan-only", action="store_true")
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--canary", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--repo", type=Path, default=REPO)
    args = parser.parse_args(argv)
    if args.plan_only:
        print(json.dumps(plan(), indent=2, sort_keys=True))
        return 0
    require(args.config is not None, "selected mode requires --config")
    config_path = args.config.resolve(strict=True)
    if args.preflight:
        preflight(config_path, args.repo.resolve(strict=True))
        return 0
    canary(config_path, args.repo.resolve(strict=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
