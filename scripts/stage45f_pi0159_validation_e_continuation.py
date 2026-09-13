#!/usr/bin/env python3
"""PI0159 continuation wrapper for the frozen Validation-E experiment.

PI0154 already admitted the six Validation-E snapshots and sealed the exact
checkpoint/statistics identities.  PI0159 is a continuation of that same
experiment after a purely mechanical offline model-loading failure.  This
wrapper therefore keeps the PI0154 evaluator and input admission intact,
adds a PI0159 continuation admission, and binds a fresh runtime cache
namespace to the already sealed Cosmos cache before the evaluator imports
Torch/GR00T.

The wrapper is standard-library-only until it calls the existing PI0154
runner.  It never downloads from Hugging Face, changes the model/data/stats/
noise contract, retries a model call, trains, or reads Holdout-F.  It records
the actual prior and continuation call counters and leaves the PI0154 failure
evidence untouched.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import stat
import sys
from typing import Any, Iterable, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
SCRIPTS = REPO / "scripts"
for _path in (SRC, SCRIPTS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import stage45f_vanda_validation_e_admission as prior_admission  # noqa: E402


PI_GATE_COMMENT = 5651409238
AUTHORITY_TOKEN = "PI_GATE_STATE_0159"
PR_NUMBER = 71
EXPECTED_BRANCH = prior_admission.EXPECTED_BRANCH
PRIOR_GATE_COMMENT = prior_admission.EXPECTED_GATE
PRIOR_FREEZE_PATH = prior_admission.FREEZE_PATH
PRIOR_FREEZE_SCHEMA = prior_admission.EXPECTED_SCHEMA
EXPECTED_MODEL_ID = "models--nvidia--Cosmos-Reason2-2B"
EXPECTED_COSMOS_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"
EXPECTED_COSMOS_TREE_SHA256 = "1cfec77c944cba341ee5d7c31b4be6bb615582364bae48f18c5686e6ec23e8a9"
EXPECTED_GROOT_COMMIT = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
EXPECTED_STATS_SHA256 = prior_admission.EXPECTED_STATS_SHA
EXPECTED_NOISE_SEED = prior_admission.EXPECTED_NOISE_SEED
EXPECTED_TIMESTEP_IDS = tuple(prior_admission.EXPECTED_BUCKETS)
EXPECTED_T_VALUES = tuple(prior_admission.EXPECTED_T)
EXPECTED_NOMINAL_CALLS = dict(prior_admission.EXPECTED_CALLS)
PRIOR_MODEL_LOAD_ATTEMPTS = 1
MAX_TOTAL_MODEL_LOAD_ATTEMPTS = 8
MAX_CONTINUATION_MODEL_LOAD_ATTEMPTS = MAX_TOTAL_MODEL_LOAD_ATTEMPTS - PRIOR_MODEL_LOAD_ATTEMPTS
MAX_ALLOCATIONS = 2
EXPECTED_ARM_KEYS = tuple(
    (route, seed)
    for seed in (2222, 2223, 2224)
    for route in ("Legacy", "L1")
)
FORBIDDEN_TOP_LEVEL = {"torch", "transformers", "gr00t", "cuda", "cv2", "numpy"}
FORBIDDEN_CALLS = {"get_action", "backward", "save_pretrained", "save_model", "save_checkpoint", "train", "qsub"}


class ContinuationError(RuntimeError):
    """Raised when PI0159 continuation cannot be admitted fail-closed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContinuationError(message)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha(value: Any) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def load_json(path: Path) -> Any:
    require(path.is_file() and not path.is_symlink(), f"regular JSON file required: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_durable(path: Path, value: Any, *, overwrite: bool = False) -> None:
    path = Path(path)
    if not overwrite:
        require(not path.exists() and not path.is_symlink(), f"record must be fresh: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def regular_file(path: Path, label: str) -> Path:
    require(path.is_file() and not path.is_symlink() and stat.S_ISREG(path.lstat().st_mode), f"{label} is not a regular file: {path}")
    return path.resolve()


def regular_directory(path: Path, label: str) -> Path:
    require(path.is_dir() and not path.is_symlink() and stat.S_ISDIR(path.lstat().st_mode), f"{label} is not a regular directory: {path}")
    return path.resolve()


def tree_digest(entries: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda item: str(item["path"])):
        line = "\t".join((
            str(entry["kind"]),
            str(entry["path"]),
            str(entry.get("size_bytes", 0)),
            str(entry.get("sha256") or "-"),
            str(entry.get("target") or "-"),
            str(entry.get("resolved_path") or "-"),
        ))
        digest.update((line + "\n").encode("utf-8"))
    return digest.hexdigest()


def cosmos_tree_digest(root: Path) -> tuple[str, int]:
    """Hash the accepted Cosmos tree without importing model libraries."""

    root = regular_directory(root, "Cosmos cache root")
    entries: list[dict[str, Any]] = []

    def visit(directory: Path, relative: Path) -> None:
        for item in sorted(directory.iterdir(), key=lambda candidate: candidate.name):
            rel = (relative / item.name).as_posix()
            mode = item.lstat().st_mode
            if stat.S_ISDIR(mode):
                entries.append({"path": rel, "kind": "directory", "size_bytes": 0, "sha256": None})
                visit(item, relative / item.name)
            elif stat.S_ISLNK(mode):
                target = os.readlink(item)
                resolved = item.resolve(strict=True)
                require(resolved.is_file(), f"Cosmos symlink is not a file: {item}")
                try:
                    resolved_rel = resolved.relative_to(root).as_posix()
                except ValueError as exc:
                    raise ContinuationError(f"Cosmos symlink escapes root: {item}") from exc
                entries.append({
                    "path": rel,
                    "kind": "symlink",
                    "target": target,
                    "resolved_path": resolved_rel,
                    "size_bytes": int(resolved.stat().st_size),
                    "sha256": sha256_file(resolved),
                })
            elif stat.S_ISREG(mode):
                entries.append({
                    "path": rel,
                    "kind": "regular_file",
                    "size_bytes": int(item.stat().st_size),
                    "sha256": sha256_file(item),
                })
            else:
                raise ContinuationError(f"unsupported Cosmos entry: {item}")

    visit(root, Path("."))
    return tree_digest(entries), len(entries)


def static_guard() -> dict[str, Any]:
    import ast

    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(Path(__file__)))
    imports: set[str] = set()
    forbidden_calls: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".", 1)[0])
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ""
            if name in FORBIDDEN_CALLS:
                forbidden_calls.append(name)
    markers = {
        "pi0159_gate": str(PI_GATE_COMMENT) in source and AUTHORITY_TOKEN in source,
        "prior_freeze_preserved": "prior_admission.EXPECTED_SCHEMA" in source and "prior_admission_path" in source,
        "offline_cache_binding": "HF_HOME" in source and "HF_HUB_CACHE" in source and "HUGGINGFACE_HUB_CACHE" in source,
        "local_first_bridge": "GROOT_HF_LOCAL_FIRST" in source and "GROOT_PATCH_MISTRAL" in source,
        "fresh_namespace": "runtime_namespace" in source and "os.symlink" in source,
        "raw_first_inherited": "stage45f_validation_e_result.json" in source and "stage45f_validation_e_failure.json" in source,
        "holdout_locked": '"holdout_f_read": False' in source,
        "training_locked": '"training": False' in source,
    }
    return {
        "schema": "STAGE45F_PI0159_CONTINUATION_STATIC_GUARD_V1",
        "status": "PASS" if not imports.intersection(FORBIDDEN_TOP_LEVEL) and not forbidden_calls and all(markers.values()) else "FAIL",
        "top_level_forbidden_imports": sorted(imports.intersection(FORBIDDEN_TOP_LEVEL)),
        "forbidden_call_hits": sorted(set(forbidden_calls)),
        "markers": markers,
        "pbs": 0,
        "h200": 0,
        "model_import": 0,
        "model_forward": 0,
        "training": 0,
        "holdout_f_read": 0,
    }


def _config(config_path: Path) -> dict[str, Any]:
    value = load_json(config_path)
    require(isinstance(value, Mapping), "PI0159 config must be an object")
    config = dict(value)
    require(config.get("schema") == "STAGE45F_PI0159_VALIDATION_E_CONTINUATION_CONFIG_V1", "PI0159 config schema drift")
    require(config.get("pi_gate") == PI_GATE_COMMENT and config.get("authority_token") == AUTHORITY_TOKEN, "PI0159 authority drift")
    require(config.get("pr_number") == PR_NUMBER, "PI0159 PR identity drift")
    require(config.get("branch") == EXPECTED_BRANCH, "PI0159 branch drift")
    require(config.get("max_allocations") == MAX_ALLOCATIONS, "PI0159 allocation ceiling drift")
    require(config.get("prior_model_load_attempts") == PRIOR_MODEL_LOAD_ATTEMPTS, "prior model-load accounting drift")
    require(config.get("max_model_loads_total") == MAX_TOTAL_MODEL_LOAD_ATTEMPTS, "PI0159 model-load ceiling drift")
    require(config.get("max_new_model_loads") == MAX_CONTINUATION_MODEL_LOAD_ATTEMPTS, "PI0159 continuation model-load ceiling drift")
    require(config.get("max_new_model_side_calls") == 210, "PI0159 continuation call ceiling drift")
    require(config.get("expected_head"), "PI0159 expected HEAD is missing")
    require(config.get("evaluation_code_commit") == "522ea718e99ff8b1c4b2aa08bec680ed5adfcae5", "PI0154 evaluator identity drift")
    require(config.get("cosmos_revision") == EXPECTED_COSMOS_REVISION, "Cosmos revision drift")
    require(config.get("cosmos_tree_sha256") == EXPECTED_COSMOS_TREE_SHA256, "Cosmos tree identity drift")
    require(config.get("stats_sha256") == EXPECTED_STATS_SHA256, "stats identity drift")
    require(config.get("noise_seed") == EXPECTED_NOISE_SEED, "noise seed drift")
    require(config.get("timestep_ids") == list(EXPECTED_TIMESTEP_IDS) and config.get("t_values") == list(EXPECTED_T_VALUES), "E4 schedule drift")
    require(config.get("nominal_calls") == EXPECTED_NOMINAL_CALLS, "nominal call budget drift")
    require(config.get("holdout_f_read") is False and config.get("training") is False, "locked execution boundary opened")
    return config


def _checkpoint_arms(binding: Mapping[str, Any]) -> list[dict[str, Any]]:
    arms = binding.get("checkpoint_arms")
    require(isinstance(arms, list) and len(arms) == 6, "prior admission does not contain six checkpoint arms")
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for arm in arms:
        require(isinstance(arm, Mapping), "checkpoint arm is not an object")
        key = (str(arm.get("route")), int(arm.get("seed")))
        require(key not in by_key, f"duplicate checkpoint arm: {key}")
        by_key[key] = dict(arm)
    require(set(by_key) == set(EXPECTED_ARM_KEYS), f"checkpoint matrix identity drift: {sorted(by_key)}")
    return [by_key[key] for key in EXPECTED_ARM_KEYS]


def validate_prior_admission(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    prior_path = regular_file(Path(str(config["prior_admission_path"])), "PI0154 admission")
    require(sha256_file(prior_path) == str(config["prior_admission_sha256"]), "PI0154 admission SHA drift")
    binding = load_json(prior_path)
    require(isinstance(binding, Mapping), "PI0154 admission is not an object")
    require(binding.get("schema") == "STAGE45F_PI0154_VALIDATION_E_ADMISSION_V1", "PI0154 admission schema drift")
    require(binding.get("status") == "ADMITTED_BEFORE_MODEL_IMPORT", "PI0154 admission status drift")
    require(binding.get("pi_gate") == PRIOR_GATE_COMMENT, "PI0154 authority identity drift")
    require(binding.get("execution_branch") == EXPECTED_BRANCH, "PI0154 branch identity drift")
    require(binding.get("model_imported") is False and binding.get("model_construction") is False and binding.get("model_forward") is False and binding.get("training") is False, "prior admission crossed model boundary")
    arms = _checkpoint_arms(binding)
    require(binding.get("stats", {}).get("sha256") == EXPECTED_STATS_SHA256, "PI0154 admitted stats drift")
    validation_input = binding.get("validation_input")
    require(isinstance(validation_input, Mapping), "PI0154 validation input binding missing")
    require(validation_input.get("target_id") == "red_target_holdout_e", "Validation-E target drift")
    require(validation_input.get("snapshot_ids") == list(prior_admission.EXPECTED_SNAPSHOT_IDS), "Validation-E window identity drift")
    require(validation_input.get("holdout_f_read") is False, "Holdout-F was not locked in prior admission")
    ledger = regular_file(Path(str(validation_input["access_ledger"])), "PI0154 access ledger")
    require(sha256_file(ledger) == validation_input.get("access_ledger_sha256"), "PI0154 access ledger SHA drift")
    freeze = regular_file(PRIOR_FREEZE_PATH, "PI0154 recipe freeze")
    require(sha256_file(freeze) == binding.get("recipe_freeze", {}).get("sha256"), "PI0154 recipe freeze SHA drift")
    failure_path = regular_file(Path(str(config["prior_failure_path"])), "PI0154 failure seal")
    require(sha256_file(failure_path) == str(config["prior_failure_sha256"]), "PI0154 failure seal SHA drift")
    failure = load_json(failure_path)
    require(isinstance(failure, Mapping), "PI0154 failure seal is not an object")
    prior_counters = failure.get("actual_counters")
    require(isinstance(prior_counters, Mapping), "PI0154 failure counters are missing")
    require(int(prior_counters.get("model_loads_attempted", -1)) == PRIOR_MODEL_LOAD_ATTEMPTS, "prior model-load attempt count drift")
    require(int(prior_counters.get("model_loads_completed", -1)) == 0, "prior model-load unexpectedly completed")
    require(int(prior_counters.get("model_forward_completed", -1)) == 0, "prior model forward unexpectedly completed")
    return {
        "path": str(prior_path),
        "sha256": sha256_file(prior_path),
        "binding": dict(binding),
        "arms": arms,
        "access_ledger": str(ledger),
        "access_ledger_sha256": sha256_file(ledger),
        "freeze": str(freeze),
        "freeze_sha256": sha256_file(freeze),
        "failure": str(failure_path),
        "failure_sha256": sha256_file(failure_path),
        "failure_phase": failure.get("phase"),
        "failure_error": failure.get("error"),
        "prior_counters": dict(prior_counters),
    }, dict(binding)


def bind_offline_runtime(config: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    """Create a fresh cache overlay linked only to the accepted Cosmos tree."""

    cosmos = regular_directory(Path(str(config["cosmos_root"])), "accepted Cosmos cache")
    hf_home = regular_directory(Path(str(config["hf_home"])), "accepted HF_HOME")
    hub = regular_directory(hf_home / "hub", "accepted HF hub cache")
    model_entry = hub / EXPECTED_MODEL_ID
    require(model_entry.exists(), f"accepted Cosmos Hub entry is missing: {model_entry}")
    require(model_entry.resolve(strict=True) == cosmos, "HF Hub entry does not bind to accepted Cosmos cache")
    refs_main = regular_file(cosmos / "refs" / "main", "Cosmos refs/main")
    require(refs_main.read_text(encoding="utf-8").strip() == EXPECTED_COSMOS_REVISION, "Cosmos refs/main drift")
    snapshot = regular_directory(cosmos / "snapshots" / EXPECTED_COSMOS_REVISION, "Cosmos pinned snapshot")
    observed_tree_sha, entry_count = cosmos_tree_digest(cosmos)
    require(observed_tree_sha == EXPECTED_COSMOS_TREE_SHA256, f"Cosmos tree SHA drift: {observed_tree_sha}")

    namespace = output_root / "runtime_namespace"
    require(not namespace.exists() and not namespace.is_symlink(), "fresh runtime namespace is unavailable")
    runtime_hf = namespace / "huggingface"
    runtime_hub = runtime_hf / "hub"
    runtime_hub.mkdir(parents=True, exist_ok=False)
    for directory in (
        runtime_hf / "datasets",
        runtime_hf / "modules",
        namespace / "torch",
        namespace / "xdg_cache",
        namespace / "xdg_config",
        namespace / "xdg_data",
        namespace / "tmp",
        namespace / "triton",
        namespace / "torch_extensions",
    ):
        directory.mkdir(parents=True, exist_ok=False)
    runtime_model_entry = runtime_hub / EXPECTED_MODEL_ID
    os.symlink(str(cosmos), runtime_model_entry, target_is_directory=True)
    values = {
        "HF_HOME": str(runtime_hf),
        "HF_HUB_CACHE": str(runtime_hub),
        "HUGGINGFACE_HUB_CACHE": str(runtime_hub),
        "TRANSFORMERS_CACHE": str(runtime_hub),
        "HF_DATASETS_CACHE": str(runtime_hf / "datasets"),
        "HF_MODULES_CACHE": str(runtime_hf / "modules"),
        "TORCH_HOME": str(namespace / "torch"),
        "XDG_CACHE_HOME": str(namespace / "xdg_cache"),
        "XDG_CONFIG_HOME": str(namespace / "xdg_config"),
        "XDG_DATA_HOME": str(namespace / "xdg_data"),
        "TMPDIR": str(namespace / "tmp"),
        "TMP": str(namespace / "tmp"),
        "TEMP": str(namespace / "tmp"),
        "TRITON_CACHE_DIR": str(namespace / "triton"),
        "TORCH_EXTENSIONS_DIR": str(namespace / "torch_extensions"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "GROOT_HF_LOCAL_FIRST": "1",
        "GROOT_PATCH_MISTRAL": "1",
        "GROOT_COMMIT_HASH": EXPECTED_GROOT_COMMIT,
        "NO_ALBUMENTATIONS_UPDATE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": "",
    }
    os.environ.update(values)
    return {
        "schema": "STAGE45F_PI0159_OFFLINE_RUNTIME_BINDING_V1",
        "status": "PASS_BEFORE_MODEL_IMPORT",
        "canonical_hf_home": str(hf_home),
        "canonical_hf_hub_cache": str(hub),
        "canonical_model_entry": str(model_entry),
        "cosmos_root": str(cosmos),
        "cosmos_revision": EXPECTED_COSMOS_REVISION,
        "cosmos_snapshot": str(snapshot),
        "cosmos_tree_sha256": observed_tree_sha,
        "cosmos_tree_entry_count": entry_count,
        "runtime_namespace": str(namespace),
        "runtime_hf_home": values["HF_HOME"],
        "runtime_hf_hub_cache": values["HF_HUB_CACHE"],
        "runtime_model_entry": str(runtime_model_entry),
        "runtime_model_entry_is_symlink": runtime_model_entry.is_symlink(),
        "runtime_model_entry_resolved": str(runtime_model_entry.resolve(strict=True)),
        "environment": {key: values[key] for key in sorted(values)},
        "network_download": False,
        "replacement_model": False,
    }


def effective_binding(config: Mapping[str, Any], prior: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    binding = dict(prior["binding"])
    binding["schema"] = "STAGE45F_PI0159_VALIDATION_E_CONTINUATION_BINDING_V1"
    binding["pi0159_gate"] = PI_GATE_COMMENT
    binding["pi0159_authority_token"] = AUTHORITY_TOKEN
    binding["prior_pi0154_admission_path"] = prior["path"]
    binding["prior_pi0154_admission_sha256"] = prior["sha256"]
    binding["prior_failure_seal_path"] = prior["failure"]
    binding["prior_failure_seal_sha256"] = prior["failure_sha256"]
    binding["execution_head"] = str(config["expected_head"])
    binding["execution_branch"] = EXPECTED_BRANCH
    binding["output_root"] = str(output_root.resolve())
    binding["checkpoint_arms"] = prior["arms"]
    binding["continuation"] = {
        "schema": "STAGE45F_PI0159_CONTINUATION_IDENTITY_V1",
        "gate": PI_GATE_COMMENT,
        "order": [f"{route}(seed{seed})@2000" for route, seed in EXPECTED_ARM_KEYS],
        "prior_model_load_attempts": PRIOR_MODEL_LOAD_ATTEMPTS,
        "max_total_model_load_attempts": MAX_TOTAL_MODEL_LOAD_ATTEMPTS,
        "max_new_model_load_attempts": MAX_CONTINUATION_MODEL_LOAD_ATTEMPTS,
        "max_allocations": MAX_ALLOCATIONS,
        "scientific_contract_changed": False,
        "holdout_f_read": False,
        "training": False,
    }
    return binding


def evidence_inventory(root: Path, excluded: set[str]) -> tuple[Path, str, int]:
    entries: list[dict[str, Any]] = []
    for directory, _, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for filename in sorted(filenames):
            path = directory_path / filename
            relative = path.relative_to(root).as_posix()
            if relative in excluded or path.is_symlink():
                continue
            require(path.is_file(), f"evidence inventory found non-file: {path}")
            entries.append({"path": relative, "bytes": int(path.stat().st_size), "sha256": sha256_file(path)})
    entries.sort(key=lambda item: item["path"])
    inventory = root / "stage45f_pi0159_evidence_inventory.json"
    write_json_durable(inventory, {
        "schema": "STAGE45F_PI0159_EVIDENCE_INVENTORY_V1",
        "entries": entries,
        "entry_count": len(entries),
        "excluded": sorted(excluded),
    })
    return inventory, sha256_file(inventory), len(entries)


def _runner_result(output_root: Path) -> dict[str, Any]:
    result_path = regular_file(output_root / "stage45f_validation_e_result.json", "PI0154 evaluator result")
    result = load_json(result_path)
    require(isinstance(result, Mapping), "PI0154 evaluator result is not an object")
    counters = result.get("actual_counters")
    require(isinstance(counters, Mapping), "PI0154 evaluator counters are missing")
    expected = {
        "model_loads_attempted": 6,
        "model_loads_completed": 6,
        "conditioning_attempts": 36,
        "conditioning_completed": 36,
        "action_head_attempts": 144,
        "action_head_completed": 144,
        "model_forward_completed": 144,
        "raw_prediction_count": 36,
        "raw_manifest_records": 36,
        "derived_row_count": 36,
        "training_steps": 0,
        "backward_steps": 0,
        "optimizer_steps": 0,
        "checkpoint_writes": 0,
        "public_sampler_calls": 0,
        "get_action_calls": 0,
    }
    for key, value in expected.items():
        require(int(counters.get(key, -1)) == value, f"PI0159 exact counter drift: {key}={counters.get(key)!r}")
    require(int(counters["model_loads_attempted"]) + PRIOR_MODEL_LOAD_ATTEMPTS <= MAX_TOTAL_MODEL_LOAD_ATTEMPTS, "PI0159 cumulative model-load ceiling exceeded")
    return {
        "path": str(result_path),
        "sha256": sha256_file(result_path),
        "result": dict(result),
        "counters": dict(counters),
    }


def execute(config_path: Path) -> int:
    phase = "START"
    output_root: Path | None = None
    config: dict[str, Any] | None = None
    prior: dict[str, Any] | None = None
    continuation_binding_path: Path | None = None
    try:
        config = _config(config_path.resolve(strict=True))
        output_root = Path(str(config["output_root"])).resolve()
        require(not output_root.exists() and not output_root.is_symlink(), "PI0159 output root must be fresh")
        require(output_root.parent.is_dir(), "PI0159 output parent is unavailable")
        output_root.mkdir(parents=False, exist_ok=False)
        phase = "PRIOR_ADMISSION_RECOVERY_AUDIT"
        prior, _ = validate_prior_admission(config)
        phase = "OFFLINE_CACHE_RESOLUTION"
        runtime_binding = bind_offline_runtime(config, output_root)
        phase = "CONTINUATION_ADMISSION"
        binding = effective_binding(config, prior, output_root)
        continuation_binding_path = output_root / "stage45f_pi0159_continuation_admission.json"
        write_json_durable(continuation_binding_path, binding)
        effective_config = dict(config)
        effective_config.update({
            "admission_path": str(continuation_binding_path),
            "output_root": str(output_root),
            "prior_admission_path": prior["path"],
        })
        effective_config_path = output_root / "stage45f_pi0159_effective_runner_config.json"
        write_json_durable(effective_config_path, effective_config)
        preflight = {
            "schema": "STAGE45F_PI0159_CONTINUATION_PRE_MODEL_ADMISSION_V1",
            "status": "PASS_PRE_MODEL",
            "pi_gate": PI_GATE_COMMENT,
            "authority_token": AUTHORITY_TOKEN,
            "pr_number": PR_NUMBER,
            "execution_head": config["expected_head"],
            "execution_branch": EXPECTED_BRANCH,
            "prior_admission": {key: value for key, value in prior.items() if key not in {"binding"}},
            "runtime_binding": runtime_binding,
            "effective_binding_path": str(continuation_binding_path),
            "effective_binding_sha256": sha256_file(continuation_binding_path),
            "effective_runner_config_path": str(effective_config_path),
            "effective_runner_config_sha256": sha256_file(effective_config_path),
            "nominal_calls": EXPECTED_NOMINAL_CALLS,
            "max_total_model_load_attempts": MAX_TOTAL_MODEL_LOAD_ATTEMPTS,
            "prior_model_load_attempts": PRIOR_MODEL_LOAD_ATTEMPTS,
            "max_new_model_load_attempts": MAX_CONTINUATION_MODEL_LOAD_ATTEMPTS,
            "holdout_f_read": False,
            "training": False,
            "network_download": False,
            "model_imported": False,
        }
        write_json_durable(output_root / "stage45f_pi0159_pre_model_admission.json", preflight)
        guard = static_guard()
        require(guard["status"] == "PASS", "PI0159 continuation static guard failed")
        write_json_durable(output_root / "stage45f_pi0159_static_guard.json", guard)
        phase = "PI0154_EVALUATOR_EXECUTION"
        # Importing this module is still model-side-free; its execute function
        # performs the original complete admission before importing Torch.
        import stage45f_vanda_validation_e_gen1 as evaluator

        return_code = int(evaluator.execute(effective_config_path))
        if return_code != 0:
            failure_path = output_root / "stage45f_validation_e_failure.json"
            prior_failure = load_json(failure_path) if failure_path.is_file() and not failure_path.is_symlink() else {}
            continuation_failure = {
                "schema": "STAGE45F_PI0159_VALIDATION_E_CONTINUATION_FAILURE_SEAL_V1",
                "status": "PARTIAL" if prior_failure.get("model_import_started") else "BLOCKED",
                "pi_gate": PI_GATE_COMMENT,
                "authority_token": AUTHORITY_TOKEN,
                "execution_head": config["expected_head"],
                "execution_branch": EXPECTED_BRANCH,
                "phase": prior_failure.get("phase", phase),
                "current": prior_failure.get("current"),
                "actual_counters": prior_failure.get("actual_counters", {}),
                "model_import_started": bool(prior_failure.get("model_import_started")),
                "model_execution_started": bool(prior_failure.get("model_execution_started")),
                "training": False,
                "backward": False,
                "optimizer_steps": 0,
                "checkpoint_writes": 0,
                "validation_e_read": True,
                "holdout_f_read": False,
                "automatic_retry": False,
                "prior_pi0154_failure": prior["failure"] if prior else None,
                "prior_pi0154_failure_sha256": prior["failure_sha256"] if prior else None,
                "runtime_binding_path": str(output_root / "stage45f_pi0159_pre_model_admission.json"),
                "runtime_binding_sha256": sha256_file(output_root / "stage45f_pi0159_pre_model_admission.json"),
                "error": prior_failure.get("error", f"PI0154 evaluator returned {return_code}"),
                "timestamp_utc": now_utc(),
            }
            write_json_durable(output_root / "stage45f_pi0159_validation_e_failure.json", continuation_failure)
            print(f"VANDA_PI0159_VALIDATION_E_BLOCKED: {continuation_failure['error']}", file=sys.stderr)
            return 2
        phase = "PI0159_READONLY_EVIDENCE_SEAL"
        run = _runner_result(output_root)
        excluded = {
            "stage45f_pi0159_evidence_inventory.json",
            "stage45f_pi0159_validation_e_seal.json",
            "stage45f_pi0159_validation_e_result.json",
            "stage45f_pi0159_validation_e_failure.json",
        }
        inventory_path, inventory_sha, inventory_count = evidence_inventory(output_root, excluded)
        raw_manifest = regular_file(output_root / "raw_predictions.jsonl", "raw prediction manifest")
        derived_manifest = regular_file(output_root / "derived_rows.jsonl", "derived evidence manifest")
        cpu_summary = regular_file(output_root / "cpu_summary.json", "CPU summary")
        paired_summary = regular_file(output_root / "paired_summary.json", "paired summary")
        seal = {
            "schema": "STAGE45F_PI0159_VALIDATION_E_CONTINUATION_SEAL_V1",
            "status": "PASS",
            "token": "VANDA_PI0159_VALIDATION_E_CONTINUATION_FOR_PI_REVIEW",
            "pi_gate": PI_GATE_COMMENT,
            "authority_token": AUTHORITY_TOKEN,
            "pr_number": PR_NUMBER,
            "execution_head": config["expected_head"],
            "execution_branch": EXPECTED_BRANCH,
            "pbs_jobid": os.environ.get("PBS_JOBID"),
            "host": socket.gethostname(),
            "prior_pi0154_admission_path": prior["path"],
            "prior_pi0154_admission_sha256": prior["sha256"],
            "prior_pi0154_failure_path": prior["failure"],
            "prior_pi0154_failure_sha256": prior["failure_sha256"],
            "continuation_admission_path": str(continuation_binding_path),
            "continuation_admission_sha256": sha256_file(continuation_binding_path),
            "pre_model_admission_path": str(output_root / "stage45f_pi0159_pre_model_admission.json"),
            "pre_model_admission_sha256": sha256_file(output_root / "stage45f_pi0159_pre_model_admission.json"),
            "offline_runtime_binding": runtime_binding,
            "checkpoint_arms": prior["arms"],
            "execution_order": [f"{route}(seed{seed})@2000" for route, seed in EXPECTED_ARM_KEYS],
            "target_id": "red_target_holdout_e",
            "snapshot_ids": list(prior_admission.EXPECTED_SNAPSHOT_IDS),
            "snapshot_count": 6,
            "checkpoint_count": 6,
            "stats_sha256": EXPECTED_STATS_SHA256,
            "noise_seed": EXPECTED_NOISE_SEED,
            "timestep_ids": list(EXPECTED_TIMESTEP_IDS),
            "t_values": list(EXPECTED_T_VALUES),
            "nominal_calls": EXPECTED_NOMINAL_CALLS,
            "prior_model_load_attempts": PRIOR_MODEL_LOAD_ATTEMPTS,
            "continuation_model_load_attempts": run["counters"]["model_loads_attempted"],
            "cumulative_model_load_attempts": PRIOR_MODEL_LOAD_ATTEMPTS + run["counters"]["model_loads_attempted"],
            "actual_counters": run["counters"],
            "runner_result_path": run["path"],
            "runner_result_sha256": run["sha256"],
            "raw_manifest": str(raw_manifest),
            "raw_manifest_sha256": sha256_file(raw_manifest),
            "derived_manifest": str(derived_manifest),
            "derived_manifest_sha256": sha256_file(derived_manifest),
            "cpu_summary": str(cpu_summary),
            "cpu_summary_sha256": sha256_file(cpu_summary),
            "paired_summary": str(paired_summary),
            "paired_summary_sha256": sha256_file(paired_summary),
            "inventory_path": str(inventory_path),
            "inventory_sha256": inventory_sha,
            "inventory_entry_count": inventory_count,
            "raw_before_derived": True,
            "training": False,
            "backward": False,
            "optimizer_steps": 0,
            "checkpoint_writes": 0,
            "validation_e_read": True,
            "holdout_f_read": False,
            "automatic_retry": False,
            "network_download": False,
            "replacement_model": False,
            "scientific_contract_changed": False,
        }
        seal_path = output_root / "stage45f_pi0159_validation_e_seal.json"
        write_json_durable(seal_path, seal)
        result = {
            "schema": "STAGE45F_PI0159_VALIDATION_E_CONTINUATION_RESULT_V1",
            "status": "COMPLETED",
            "token": "VANDA_PI0159_VALIDATION_E_CONTINUATION_FOR_PI_REVIEW",
            "execution_head": config["expected_head"],
            "execution_branch": EXPECTED_BRANCH,
            "seal_path": str(seal_path),
            "seal_sha256": sha256_file(seal_path),
            "actual_counters": run["counters"],
            "nominal_calls": EXPECTED_NOMINAL_CALLS,
            "continuation_model_load_attempts": run["counters"]["model_loads_attempted"],
            "cumulative_model_load_attempts": PRIOR_MODEL_LOAD_ATTEMPTS + run["counters"]["model_loads_attempted"],
            "raw_output_count": run["counters"]["raw_prediction_count"],
            "cpu_summary": str(cpu_summary),
            "cpu_summary_sha256": sha256_file(cpu_summary),
            "paired_summary": str(paired_summary),
            "paired_summary_sha256": sha256_file(paired_summary),
            "inventory_path": str(inventory_path),
            "inventory_sha256": inventory_sha,
            "inventory_entry_count": inventory_count,
            "training": False,
            "holdout_f_read": False,
            "automatic_retry": False,
        }
        write_json_durable(output_root / "stage45f_pi0159_validation_e_result.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        if output_root is not None and output_root.is_dir():
            try:
                failure_path = output_root / "stage45f_pi0159_validation_e_failure.json"
                if not failure_path.exists() and not failure_path.is_symlink():
                    write_json_durable(failure_path, {
                        "schema": "STAGE45F_PI0159_VALIDATION_E_CONTINUATION_FAILURE_SEAL_V1",
                        "status": "BLOCKED",
                        "pi_gate": PI_GATE_COMMENT,
                        "authority_token": AUTHORITY_TOKEN,
                        "execution_head": config.get("expected_head") if config else None,
                        "execution_branch": EXPECTED_BRANCH,
                        "phase": phase,
                        "actual_counters": {},
                        "model_import_started": False,
                        "model_execution_started": False,
                        "training": False,
                        "backward": False,
                        "optimizer_steps": 0,
                        "checkpoint_writes": 0,
                        "validation_e_read": False,
                        "holdout_f_read": False,
                        "automatic_retry": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "timestamp_utc": now_utc(),
                    })
            except Exception:
                pass
        print(f"VANDA_PI0159_VALIDATION_E_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


def plan() -> dict[str, Any]:
    guard = static_guard()
    return {
        "schema": "STAGE45F_PI0159_VALIDATION_E_CONTINUATION_PLAN_V1",
        "status": "PASS_STATIC_NOT_EXECUTED" if guard["status"] == "PASS" else "FAIL",
        "pi_gate": PI_GATE_COMMENT,
        "authority_token": AUTHORITY_TOKEN,
        "prior_gate": PRIOR_GATE_COMMENT,
        "execution_order": [f"{route}(seed{seed})@2000" for route, seed in EXPECTED_ARM_KEYS],
        "target_id": "red_target_holdout_e",
        "snapshot_count": 6,
        "nominal_calls": EXPECTED_NOMINAL_CALLS,
        "prior_model_load_attempts": PRIOR_MODEL_LOAD_ATTEMPTS,
        "max_total_model_load_attempts": MAX_TOTAL_MODEL_LOAD_ATTEMPTS,
        "max_new_model_load_attempts": MAX_CONTINUATION_MODEL_LOAD_ATTEMPTS,
        "max_allocations": MAX_ALLOCATIONS,
        "offline_model_download": False,
        "training": False,
        "holdout_f_read": False,
        "static_guard": guard,
        "return_token": "VANDA_PI0159_VALIDATION_E_CONTINUATION_FOR_PI_REVIEW",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--plan-only", action="store_true")
    modes.add_argument("--execute", action="store_true")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.plan_only:
            print(json.dumps(plan(), indent=2, sort_keys=True))
            return 0
        require(args.config is not None, "--execute requires --config")
        return execute(args.config)
    except Exception as exc:
        print(f"VANDA_PI0159_VALIDATION_E_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
