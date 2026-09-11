#!/usr/bin/env python3
"""Stage45E Corelab N2 admission, canary and fresh-BASE training runner.

The default/static path is dependency-light and does not import torch or
GR00T.  ``--mode canary`` and ``--mode train`` are explicit, path-bound GPU
operations.  They require the caller to provide every external identity; no
historical scratch path is guessed and no resource workaround is attempted.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import stage45e_n2_statistics as n2  # noqa: E402

STAGE = "Stage45E"
BRANCH = "stage45e/n2-balanced-xyz-normalization"
PI_GATE = "PI_GATE_STATE_0122"
EXPECTED_BASE_TREE = "8717e322de3dd8c66e52bdb81b37badb0f235c518de6acd465532ed01569a6a4"
EXPECTED_FULL24_TREE = "2f3badd2019a68c6549a66f85b5486faf626078a3e438342f67aacc4edc95789"
EXPECTED_GROOT_COMMIT = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
EXPECTED_COSMOS_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"
EXPECTED_COSMOS_TREE = "1cfec77c944cba341ee5d7c31b4be6bb615582364bae48f18c5686e6ec23e8a9"
EXPECTED_MODALITY_SHA = "e61f9429ecc21ceab51d9d26cadc9018691eb8b5c74d60aec4abf7faee16f031"
EXPECTED_CONFIG_SHA = "9c3cd1d7bea603b95b1c777d8bcfebd61bd7eb5d86ca8ebbc3175f8141498a7c"
TRAINING_STEPS = 2000

# Keep the failure seal truthful if an exception occurs after GPU/model
# admission.  This process-local ledger is updated immediately before each
# model-side operation and is never used to change the scientific budget.
_LAST_EXECUTION_BOUNDARY: dict[str, Any] = {
    "gpu": False,
    "model_constructed": False,
    "gr00t_forwards": 0,
    "training_steps": 0,
    "backward_steps": 0,
    "optimizer_steps": 0,
    "scheduler_steps": 0,
    "checkpoint_writes": 0,
    "pbs_submissions": 0,
    "wbc_steps": 0,
    "validation_e_read": False,
    "holdout_f_read": False,
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def read_json(path: Path) -> Any:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"required JSON is not a regular file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any, *, overwrite: bool = False) -> None:
    path = Path(path)
    if path.exists() and not overwrite:
        raise RuntimeError(f"refusing to overwrite evidence file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _tree_entries(root: Path, prefix: str) -> list[dict[str, Any]]:
    root = Path(root).resolve()
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"tree root is not a non-symlink directory: {root}")
    entries: list[dict[str, Any]] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for dirname in dirnames:
            child = directory_path / dirname
            if child.is_symlink() or not stat.S_ISDIR(child.lstat().st_mode):
                raise RuntimeError(f"unsafe directory entry: {child}")
        for filename in filenames:
            child = directory_path / filename
            if child.is_symlink() or not stat.S_ISREG(child.lstat().st_mode):
                raise RuntimeError(f"unsafe file entry: {child}")
            entries.append(
                {
                    "path": f"{prefix}/{child.relative_to(root).as_posix()}",
                    "bytes": int(child.stat().st_size),
                    "sha256": sha256_file(child),
                }
            )
    return sorted(entries, key=lambda item: str(item["path"]))


def _tree_digest(entries: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(
            f"{entry['path']}\0{entry['bytes']}\0{entry['sha256']}\n".encode("utf-8")
        )
    return digest.hexdigest()


def _cosmos_tree_digest(root: Path) -> tuple[str, int]:
    root = Path(root).resolve()
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"Cosmos cache root is not a non-symlink directory: {root}")
    entries: list[dict[str, Any]] = []

    def visit(directory: Path, relative: Path) -> None:
        for item in sorted(directory.iterdir(), key=lambda value: value.name):
            rel = (relative / item.name).as_posix()
            mode = item.lstat().st_mode
            if stat.S_ISDIR(mode):
                entries.append({"path": rel, "kind": "directory", "size_bytes": 0, "sha256": None})
                visit(item, relative / item.name)
            elif stat.S_ISLNK(mode):
                resolved = item.resolve(strict=True)
                if not resolved.is_file() or not resolved.is_relative_to(root):
                    raise RuntimeError(f"Cosmos symlink escapes cache root: {item}")
                entries.append(
                    {
                        "path": rel,
                        "kind": "symlink",
                        "target": os.readlink(item),
                        "resolved_path": resolved.relative_to(root).as_posix(),
                        "size_bytes": int(resolved.stat().st_size),
                        "sha256": sha256_file(resolved),
                    }
                )
            elif stat.S_ISREG(mode):
                entries.append(
                    {
                        "path": rel,
                        "kind": "regular_file",
                        "target": None,
                        "resolved_path": None,
                        "size_bytes": int(item.stat().st_size),
                        "sha256": sha256_file(item),
                    }
                )
            else:
                raise RuntimeError(f"unsupported Cosmos cache entry: {item}")

    visit(root, Path("."))
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda value: str(value["path"])):
        digest.update(
            "\t".join(
                (
                    str(entry["kind"]),
                    str(entry["path"]),
                    str(entry.get("size_bytes", 0)),
                    str(entry.get("sha256") or "-"),
                    str(entry.get("target") or "-"),
                    str(entry.get("resolved_path") or "-"),
                )
            ).encode("utf-8")
            + b"\n"
        )
    return digest.hexdigest(), len(entries)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def _gpu_identity(expected_uuid: str) -> dict[str, Any]:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if query.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {query.stderr[-1000:]}")
    rows = [line.strip() for line in query.stdout.splitlines() if line.strip()]
    parsed: list[dict[str, Any]] = []
    for row in rows:
        fields = [part.strip() for part in row.split(",")]
        if len(fields) != 5:
            raise RuntimeError(f"unexpected nvidia-smi row: {row!r}")
        parsed.append(
            {
                "index": fields[0],
                "name": fields[1],
                "uuid": fields[2],
                "memory_total_mib": int(fields[3]),
                "memory_used_mib_at_start": int(fields[4]),
            }
        )
    selected = [item for item in parsed if item["uuid"] == expected_uuid]
    if len(selected) != 1:
        raise RuntimeError(f"requested GPU UUID is not uniquely visible: {expected_uuid}; observed={parsed!r}")
    gpu = selected[0]
    if "RTX 4080 SUPER" not in gpu["name"]:
        raise RuntimeError(f"requested GPU is not the frozen RTX 4080S resource: {gpu!r}")
    if gpu["memory_total_mib"] < 30000:
        raise RuntimeError(f"requested GPU does not have the preferred 32GB capacity: {gpu!r}")
    gpu["host"] = subprocess.check_output(["hostname"], text=True).strip()
    gpu["query_sha256"] = hashlib.sha256(query.stdout.encode("utf-8")).hexdigest()
    return gpu


def _regular_external_paths(args: argparse.Namespace) -> dict[str, Path]:
    names = (
        "base_model",
        "dataset",
        "groot",
        "cosmos",
        "modality",
        "config",
        "runtime_stats",
        "n2_stats",
    )
    paths: dict[str, Path] = {}
    for name in names:
        value = getattr(args, name)
        if value is None:
            raise RuntimeError(f"--{name.replace('_', '-')} is required for GPU mode")
        path = Path(value).resolve()
        if not path.exists() or path.is_symlink() and name not in {"cosmos"}:
            raise RuntimeError(f"required {name} path is unavailable: {path}")
        paths[name] = path
    return paths


def _verify_external_identities(paths: Mapping[str, Path], *, n2_document: Mapping[str, Any]) -> dict[str, Any]:
    base_entries = _tree_entries(paths["base_model"], "payload/GR00T-N1.7-3B")
    base_tree = _tree_digest(base_entries)
    if base_tree != EXPECTED_BASE_TREE:
        raise RuntimeError(f"BASE tree identity mismatch: {base_tree}")
    dataset_identity = n2.validate_dataset_identity(paths["dataset"])
    if dataset_identity["tree_sha256"] != EXPECTED_FULL24_TREE:
        raise RuntimeError("FULL24 dataset tree identity mismatch")
    runtime_stats = read_json(paths["runtime_stats"])
    runtime_check = n2.validate_source_document(runtime_stats, source_path=paths["runtime_stats"])
    if runtime_check["outer_sha256"] != n2.SOURCE_STATS_SHA256:
        raise RuntimeError("runtime-V2 statistics outer identity mismatch")
    verified_n2 = n2.verify_n2_document(n2_document, source=runtime_stats)
    if n2_document.get("source_runtime_v2", {}).get("outer_sha256") != n2.SOURCE_STATS_SHA256:
        raise RuntimeError("N2 source runtime-V2 binding mismatch")
    modality_sha = sha256_file(paths["modality"])
    if modality_sha != EXPECTED_MODALITY_SHA:
        raise RuntimeError(f"modality identity mismatch: {modality_sha}")
    config_sha = sha256_file(paths["config"])
    if config_sha != EXPECTED_CONFIG_SHA:
        raise RuntimeError(f"training config identity mismatch: {config_sha}")
    groot_head = _git(paths["groot"], "rev-parse", "HEAD")
    if groot_head != EXPECTED_GROOT_COMMIT:
        raise RuntimeError(f"GR00T source commit mismatch: {groot_head}")
    cosmos_refs = paths["cosmos"] / "refs" / "main"
    if not cosmos_refs.is_file() or cosmos_refs.is_symlink():
        raise RuntimeError(f"Cosmos refs/main is unavailable: {cosmos_refs}")
    cosmos_revision = cosmos_refs.read_text(encoding="utf-8").strip()
    if cosmos_revision != EXPECTED_COSMOS_REVISION:
        raise RuntimeError(f"Cosmos revision mismatch: {cosmos_revision}")
    cosmos_tree, cosmos_count = _cosmos_tree_digest(paths["cosmos"])
    if cosmos_tree != EXPECTED_COSMOS_TREE:
        raise RuntimeError(f"Cosmos cache tree mismatch: {cosmos_tree}")
    return {
        "base_tree_sha256": base_tree,
        "full24": dataset_identity,
        "runtime_stats": runtime_check,
        "n2_artifact_sha256": sha256_file(paths["n2_stats"]),
        "n2_verification": verified_n2,
        "modality_sha256": modality_sha,
        "training_config_sha256": config_sha,
        "groot_commit": groot_head,
        "cosmos_revision": cosmos_revision,
        "cosmos_tree_sha256": cosmos_tree,
        "cosmos_entry_count": cosmos_count,
    }


def _static_audit() -> dict[str, Any]:
    manifest_path = REPO / "manifests/STAGE45E_N2_BALANCED_XYZ_NORMALIZATION_V1.json"
    manifest = read_json(manifest_path)
    if manifest.get("schema") != "CORELAB_STAGE45E_N2_BALANCED_XYZ_NORMALIZATION_V1":
        raise RuntimeError("N2 manifest schema mismatch")
    if manifest.get("repository", {}).get("branch") != BRANCH:
        raise RuntimeError("N2 manifest branch mismatch")
    if manifest.get("intervention", {}).get("common_half_range_m") != n2.HALF_RANGE_M:
        raise RuntimeError("N2 manifest half-range mismatch")
    if manifest.get("preserved_contracts", {}).get("mask_active_scalars") != 190:
        raise RuntimeError("N2 manifest mask identity mismatch")
    forbidden = {"torch", "gr00t"}
    checked: dict[str, list[str]] = {}
    for path in (REPO / "scripts/stage45e_n2_statistics.py", REPO / "scripts/stage45e_n2_corelab.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports: list[str] = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
        checked[str(path.relative_to(REPO))] = imports
        if any(name == item or name.startswith(item + ".") for name in imports for item in forbidden):
            raise RuntimeError(f"top-level model runtime import in static script: {path}")
    return {
        "schema": "CORELAB_STAGE45E_N2_STATIC_AUDIT_V1",
        "status": "PASS",
        "manifest_sha256": sha256_file(manifest_path),
        "checked_top_level_imports": checked,
        "model_constructed": False,
        "gr00t_forwards": 0,
        "gpu": False,
        "training_steps": 0,
        "optimizer_steps": 0,
        "wbc_steps": 0,
        "validation_e_read": False,
        "holdout_f_read": False,
    }


def _prepare_runtime_environment(paths: Mapping[str, Path], output_root: Path, gpu: Mapping[str, Any]) -> None:
    cache_root = output_root / "cache"
    tmp_root = output_root / "tmp"
    cache_root.mkdir(parents=True, exist_ok=False)
    tmp_root.mkdir(parents=True, exist_ok=False)
    hub_root = cache_root / "huggingface" / "hub"
    hub_root.mkdir(parents=True, exist_ok=False)
    xdg_root = cache_root / "xdg"
    xdg_root.mkdir(parents=True, exist_ok=False)
    cosmos_link = hub_root / "models--nvidia--Cosmos-Reason2-2B"
    cosmos_link.symlink_to(paths["cosmos"], target_is_directory=True)
    os.environ.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu["index"]),
            "TMPDIR": str(tmp_root),
            "TMP": str(tmp_root),
            "TEMP": str(tmp_root),
            "XDG_CACHE_HOME": str(xdg_root),
            "HF_HOME": str(cache_root / "huggingface"),
            "TRANSFORMERS_CACHE": str(hub_root),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "STAGE45B_LABEL_ANCHOR_RECONCILIATION": "1",
            "STAGE45E_N2_STATS_ARTIFACT": str(paths["n2_stats"]),
        }
    )


def _load_pipeline(paths: Mapping[str, Path], output_root: Path) -> tuple[Any, Any, Any, Any, Any]:
    """Construct the exact official pipeline after all admission checks."""

    # These imports are intentionally inside the post-admission function.
    from stage45b_full24_fresh_base_train import _bind_cosmos_processor, _load_config
    from go2arx5_vla.stage38_structured_training import install_structured_dataset_factory

    # GR00T setup writes final_model_config.json into this directory.  A
    # fresh evidence root must provide it explicitly; relying on a prior
    # scratch tree would make the runtime non-reproducible.
    (output_root / "experiment_cfg").mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(paths["groot"]))
    _bind_cosmos_processor(paths["cosmos"] / "snapshots" / EXPECTED_COSMOS_REVISION)
    install_structured_dataset_factory()
    config = _load_config(
        paths["config"],
        output_root=output_root,
        base_model=paths["base_model"],
        dataset=paths["dataset"],
    )
    from gr00t.model import MODEL_REGISTRY

    pipeline_type = MODEL_REGISTRY.get(type(config.model))
    if pipeline_type is None:
        raise RuntimeError(f"no GR00T pipeline registered for {type(config.model)!r}")
    pipeline = pipeline_type(config, output_root / "experiment_cfg")
    pipeline.setup()
    return pipeline, config, pipeline.model, pipeline.train_dataset, pipeline.data_collator


def _sample_batch(dataset: Any, collator: Any) -> Any:
    sample = next(iter(dataset))
    collated = collator([sample])
    if isinstance(collated, Mapping) and "inputs" in collated:
        return collated["inputs"]
    return collated


def _parameter_sample(model: Any) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        flat = parameter.detach().reshape(-1)
        count = min(int(flat.numel()), 256)
        records.append(
            {
                "name": name,
                "numel": int(parameter.numel()),
                "sample_count": count,
                "sample": flat[:count].float().cpu().tolist(),
            }
        )
        if len(records) >= 16:
            break
    if not records:
        raise RuntimeError("N2 training graph exposes no trainable parameters")
    return {"parameter_count_sampled": len(records), "records": records}


def _parameter_delta(before: Mapping[str, Any], model: Any) -> dict[str, Any]:
    after = _parameter_sample(model)
    if [record["name"] for record in before["records"]] != [record["name"] for record in after["records"]]:
        raise RuntimeError("trainable parameter sample identity changed during canary")
    max_delta = 0.0
    for left, right in zip(before["records"], after["records"]):
        if left["numel"] != right["numel"] or left["sample_count"] != right["sample_count"]:
            raise RuntimeError(f"trainable parameter shape changed: {left['name']}")
        delta = max(
            (abs(float(a) - float(b)) for a, b in zip(left["sample"], right["sample"])),
            default=0.0,
        )
        max_delta = max(max_delta, delta)
    return {"parameter_sample_max_abs_delta": max_delta, "weight_update": 0 if max_delta == 0.0 else 1}


def _finite_tensor_tree(value: Any, torch: Any) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, Mapping):
        return all(_finite_tensor_tree(item, torch) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tensor_tree(item, torch) for item in value)
    return True


def _run_canary(paths: Mapping[str, Path], output_root: Path, gpu: Mapping[str, Any]) -> dict[str, Any]:
    global _LAST_EXECUTION_BOUNDARY
    import torch

    _prepare_runtime_environment(paths, output_root, gpu)
    pipeline, config, model, dataset, collator = _load_pipeline(paths, output_root)
    _LAST_EXECUTION_BOUNDARY.update({"gpu": True, "model_constructed": True})
    device = torch.device("cuda:0")
    model.to(device)
    model.eval()
    torch.cuda.reset_peak_memory_stats(device)
    inference_inputs = _sample_batch(dataset, collator)
    # ``get_action`` is the inference-only path.  The collator also emits
    # supervised ``action``/``action_mask`` fields for the backward canary;
    # passing those through would select the training/RTC assertion path in
    # GR00T rather than the frozen inference contract.
    if isinstance(inference_inputs, Mapping):
        inference_inputs = dict(inference_inputs)
        inference_inputs.pop("action", None)
        inference_inputs.pop("action_mask", None)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _LAST_EXECUTION_BOUNDARY["gr00t_forwards"] = int(_LAST_EXECUTION_BOUNDARY["gr00t_forwards"]) + 1
        inference_output = model.get_action(inference_inputs)
    inference_action = inference_output["action_pred"]
    if tuple(inference_action.shape) != (1, 40, 132):
        raise RuntimeError(f"N2 inference canary action shape mismatch: {tuple(inference_action.shape)}")
    if not torch.isfinite(inference_action).all().item():
        raise RuntimeError("N2 inference canary produced non-finite action")

    model.train()
    backward_inputs = _sample_batch(dataset, collator)
    before = _parameter_sample(model)
    model.zero_grad(set_to_none=True)
    with torch.enable_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _LAST_EXECUTION_BOUNDARY["gr00t_forwards"] = int(_LAST_EXECUTION_BOUNDARY["gr00t_forwards"]) + 1
        outputs = model(backward_inputs)
        loss = outputs["loss"] if isinstance(outputs, Mapping) else outputs.loss
        if not torch.is_tensor(loss) or loss.ndim != 0 or not torch.isfinite(loss).item():
            raise RuntimeError("N2 one-batch loss is not finite scalar")
        _LAST_EXECUTION_BOUNDARY["backward_steps"] = int(_LAST_EXECUTION_BOUNDARY["backward_steps"]) + 1
        loss.backward()
    gradient_count = 0
    gradient_max_abs = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        gradient_count += 1
        if not torch.isfinite(parameter.grad).all().item():
            raise RuntimeError("N2 one-batch gradient contains non-finite values")
        gradient_max_abs = max(gradient_max_abs, float(parameter.grad.detach().abs().max().item()))
    delta = _parameter_delta(before, model)
    model.zero_grad(set_to_none=True)
    result = {
        "schema": "CORELAB_STAGE45E_N2_RESOURCE_CANARY_V1",
        "status": "PASS",
        "stage": STAGE,
        "pi_gate": PI_GATE,
        "gpu": dict(gpu),
        "model_constructed": True,
        "inference_windows": 1,
        "inference_forward_calls": 1,
        "inference_action_shape": list(inference_action.shape),
        "inference_action_dtype": str(inference_action.dtype),
        "training_batch_forward_calls": 1,
        "backward_steps": 1,
        "optimizer_steps": 0,
        "scheduler_steps": 0,
        "checkpoint_writes": 0,
        "weight_update": delta["weight_update"],
        "parameter_sample_max_abs_delta": delta["parameter_sample_max_abs_delta"],
        "loss": float(loss.detach().float().cpu().item()),
        "gradient_parameter_count": gradient_count,
        "gradient_max_abs": gradient_max_abs,
        "cuda_max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "cuda_max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "cuda_memory_allocated_bytes_after_clear": int(torch.cuda.memory_allocated(device)),
        "execution_boundary": {
            "gpu": True,
            "gr00t_forwards": 2,
            "training_steps": 0,
            "backward_steps": 1,
            "optimizer_steps": 0,
            "scheduler_steps": 0,
            "checkpoint_writes": 0,
            "wbc_steps": 0,
            "pbs_submissions": 0,
            "validation_e_read": False,
            "holdout_f_read": False,
        },
    }
    write_json(output_root / "stage45e_n2_resource_canary.json", result)
    del pipeline, config, model, dataset, collator
    torch.cuda.empty_cache()
    return result


def _checkpoint_record(path: Path, expected_step: int) -> dict[str, Any]:
    if not path.is_dir() or path.is_symlink():
        raise RuntimeError(f"missing checkpoint directory: {path}")
    state_path = path / "trainer_state.json"
    state = read_json(state_path)
    required = ["optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json"]
    missing = [name for name in required if not (path / name).is_file()]
    entries = _tree_entries(path, f"stage45e/{path.name}")
    return {
        "path": str(path),
        "global_step": state.get("global_step"),
        "expected_step": expected_step,
        "required_state_files_present": not missing,
        "missing_state_files": missing,
        "tree_sha256": _tree_digest(entries),
        "file_count": len(entries),
        "total_bytes": sum(int(item["bytes"]) for item in entries),
        "status": "PASS" if state.get("global_step") == expected_step and not missing else "BLOCKED",
    }


def _run_training(paths: Mapping[str, Path], output_root: Path, gpu: Mapping[str, Any], canary: Mapping[str, Any]) -> dict[str, Any]:
    if canary.get("status") != "PASS":
        raise RuntimeError("fresh N2 training requires a passing resource canary")
    if canary.get("weight_update") != 0 or canary.get("optimizer_steps") != 0:
        raise RuntimeError("resource canary mutated model state")
    import torch  # noqa: F401  (runtime is already admitted by the canary)

    from stage45b_full24_fresh_base_train import _load_config

    config = _load_config(
        paths["config"],
        output_root=output_root,
        base_model=paths["base_model"],
        dataset=paths["dataset"],
    )
    from go2arx5_vla.stage38_structured_training import install_structured_dataset_factory

    install_structured_dataset_factory()
    from gr00t.experiment.experiment import run as gr00t_run

    gr00t_run(config)
    checkpoint_1000 = _checkpoint_record(output_root / "checkpoint-1000", 1000)
    checkpoint_2000 = _checkpoint_record(output_root / "checkpoint-2000", 2000)
    sidecar_path = output_root / "experiment_cfg" / "structured_action_statistics.json"
    sidecar = read_json(sidecar_path)
    n2_doc = read_json(paths["n2_stats"])
    if sidecar.get("statistics") != n2_doc.get("statistics"):
        raise RuntimeError("training sidecar statistics do not match sealed N2 statistics")
    result = {
        "schema": "CORELAB_STAGE45E_N2_FRESH_BASE_TRAINING_RESULT_V1",
        "status": "STAGE45E_N2_CORELAB_2000_TRAINING_COMPLETE_FOR_PI_REVIEW"
        if checkpoint_1000["status"] == "PASS" and checkpoint_2000["status"] == "PASS"
        else "INCOMPLETE",
        "stage": STAGE,
        "pi_gate": PI_GATE,
        "starting_checkpoint": "BASE",
        "base_tree_sha256": EXPECTED_BASE_TREE,
        "full24_tree_sha256": EXPECTED_FULL24_TREE,
        "n2_stats_sha256": sha256_file(paths["n2_stats"]),
        "n2_stats_payload_sha256": n2_doc.get("statistics_payload_sha256"),
        "seed": 2222,
        "max_steps": TRAINING_STEPS,
        "save_steps": [1000, 2000],
        "gpu": dict(gpu),
        "checkpoint_1000": checkpoint_1000,
        "checkpoint_2000": checkpoint_2000,
        "training_steps": TRAINING_STEPS,
        "backward_steps": TRAINING_STEPS,
        "optimizer_steps": TRAINING_STEPS,
        "gr00t_forwards": TRAINING_STEPS,
        "wbc_steps": 0,
        "pbs_submissions": 0,
        "validation_e_read": False,
        "holdout_f_read": False,
        "next_gate": "PI_REVIEW_BEFORE_EVALUATION",
        "timestamp_utc": now_utc(),
    }
    write_json(output_root / "stage45e_n2_training_result.json", result)
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    global _LAST_EXECUTION_BOUNDARY
    _LAST_EXECUTION_BOUNDARY = {
        "gpu": False,
        "model_constructed": False,
        "gr00t_forwards": 0,
        "training_steps": 0,
        "backward_steps": 0,
        "optimizer_steps": 0,
        "scheduler_steps": 0,
        "checkpoint_writes": 0,
        "pbs_submissions": 0,
        "wbc_steps": 0,
        "validation_e_read": False,
        "holdout_f_read": False,
    }
    if args.mode == "static":
        return _static_audit()
    paths = _regular_external_paths(args)
    n2_document = read_json(paths["n2_stats"])
    identities = _verify_external_identities(paths, n2_document=n2_document)
    gpu = _gpu_identity(args.gpu_uuid)
    output_root = Path(args.output_root).resolve()
    if output_root.exists() or output_root.is_symlink():
        raise RuntimeError(f"output root must be new: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(output_root).free < 64 * 1024**3:
        raise RuntimeError("at least 64 GiB free disk is required for full-state checkpoints")
    preflight = {
        "schema": "CORELAB_STAGE45E_N2_ADMISSION_V1",
        "status": "PASS",
        "stage": STAGE,
        "pi_gate": PI_GATE,
        "branch": BRANCH,
        "repository_head": _git(REPO, "rev-parse", "HEAD"),
        "identities": identities,
        "gpu": gpu,
        "execution_boundary": {
            "gpu": False,
            "model_constructed": False,
            "gr00t_forwards": 0,
            "training_steps": 0,
            "backward_steps": 0,
            "optimizer_steps": 0,
            "scheduler_steps": 0,
            "checkpoint_writes": 0,
            "pbs_submissions": 0,
            "wbc_steps": 0,
            "validation_e_read": False,
            "holdout_f_read": False,
        },
    }
    write_json(output_root / "stage45e_n2_admission.json", preflight)
    canary = _run_canary(paths, output_root, gpu)
    if args.mode == "canary":
        canary["return_token"] = "STAGE45E_N2_CORELAB_RUNTIME_CANARY_PASS"
        return canary
    return _run_training(paths, output_root, gpu, canary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("static", "canary", "train"), default="static")
    parser.add_argument("--base-model", dest="base_model", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--groot", type=Path)
    parser.add_argument("--cosmos", type=Path)
    parser.add_argument("--modality", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--runtime-stats", dest="runtime_stats", type=Path)
    parser.add_argument("--n2-stats", dest="n2_stats", type=Path)
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    output_root = Path(args.output_root).resolve() if args.output_root is not None else None
    try:
        result = run(args)
    except Exception as exc:  # pragma: no cover - fail-closed CLI surface
        if output_root is not None:
            output_root.mkdir(parents=True, exist_ok=True)
            failure = {
                "schema": "CORELAB_STAGE45E_N2_FAILURE_V1",
                "status": "BLOCKED",
                "stage": STAGE,
                "pi_gate": PI_GATE,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "execution_boundary": dict(_LAST_EXECUTION_BOUNDARY),
                "retry_or_substitution": False,
                "timestamp_utc": now_utc(),
            }
            write_json(output_root / "stage45e_n2_failure.json", failure, overwrite=True)
        print(f"STAGE45E_N2_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(_jsonable(result), indent=2, sort_keys=True))
    return 0 if result.get("status") in {"PASS", "STAGE45E_N2_CORELAB_2000_TRAINING_COMPLETE_FOR_PI_REVIEW"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
