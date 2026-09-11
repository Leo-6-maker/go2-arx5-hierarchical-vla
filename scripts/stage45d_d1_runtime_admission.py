#!/usr/bin/env python3
"""CPU-only Stage45D-D1 runtime/admission closure.

This module describes the late-bound boundary between the frozen Corelab D1
scientific harness and a future runtime substrate.  It deliberately contains
no Torch, GR00T, CUDA, PBS, or model imports.  The static template is
validated now; the future admission function is ordered so checkout and
identity failures occur before a model can be imported.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
ADMISSION_MANIFEST_PATH = REPO / "manifests" / "STAGE45D_D1_RUNTIME_ADMISSION_MANIFEST_V1.json"
COMPATIBILITY_MAP_PATH = REPO / "manifests" / "STAGE45D_D1_RUNTIME_COMPATIBILITY_MAP_V1.json"

ADMISSION_SCHEMA = "CORELAB_STAGE45D_D1_RUNTIME_ADMISSION_MANIFEST_V1"
COMPATIBILITY_SCHEMA = "CORELAB_STAGE45D_D1_RUNTIME_COMPATIBILITY_MAP_V1"
PROTOCOL_PROVENANCE_HEAD = "1dd5837267cb1af34cd2ae7a978f70d6c8f78bdb"
PROTOCOL_SHA256 = "24cebea0d5bd884c8eb4ca1db2b9d55dfae57326936939ccd09d3868c96309b3"
SUPPORT_BRANCH = "stage45d/corelab-d1-protocol-correction"
PI_GATE_COMMENT = 5629752688
PR51_POINTER_COMMENT = 5629755293
ISSUE50_POINTER_COMMENT = 5629756230
RETURN_TOKEN = "STAGE45D_D1_RUNTIME_ADMISSION_READY_FOR_H200_GATE"

CHECKPOINT_ORDER = ("STRUCTURED@1000", "FULL24@1000", "FULL24@2000")
STATS_ROUTES = {
    "STRUCTURED@1000": "HISTORICAL_STAGE40_FIXED_STATS",
    "FULL24@1000": "STAGE45C_RUNTIME_V2_STATS_BRIDGE",
    "FULL24@2000": "STAGE45C_RUNTIME_V2_STATS_BRIDGE",
}
STATS_SHA256 = {
    "STRUCTURED@1000": "0553436da13b4500e896cf4ec06ef36aba8d230536595cdd219818d912567206",
    "FULL24@1000": "3b3a2d51e685a0e580de4ebdfa05abcbcddfb7dc33b8d3a659994cdcd1bb0337",
    "FULL24@2000": "3b3a2d51e685a0e580de4ebdfa05abcbcddfb7dc33b8d3a659994cdcd1bb0337",
}
STATS_PAYLOAD_SHA256 = {
    "STRUCTURED@1000": None,
    "FULL24@1000": "752898989bb3308e663c4b6cf88b3cc1c139897a6860e6aed1d9d708d0f6fd75",
    "FULL24@2000": "752898989bb3308e663c4b6cf88b3cc1c139897a6860e6aed1d9d708d0f6fd75",
}
CHECKPOINT_TREE_SHA256 = {
    "STRUCTURED@1000": "30b14d36e47e08dfbb9d3b353d199971863d6e7f6ceedea2d478ef367d0a1137",
    "FULL24@1000": "38a00f578eededcebd26fa07ecfdba095a4edc64afd949599f1de5284efea12c",
    "FULL24@2000": "2113a6a3b2a91233a42214dbb017c0136acde282fda2a418066f9317e78981a5",
}
CHECKPOINT_IDENTITIES = {
    label: {
        "tree_sha256": CHECKPOINT_TREE_SHA256[label],
        "stats_route": STATS_ROUTES[label],
        "stats_sha256": STATS_SHA256[label],
        "stats_payload_sha256": STATS_PAYLOAD_SHA256[label],
    }
    for label in CHECKPOINT_ORDER
}
DATASET_IDENTITY = {
    "id": "FULL24_EXISTING_TRAIN",
    "split": "TRAIN_ONLY",
    "window_count": 24,
    "frame_count": 960,
    "window_order": "target-major then snapshot-major",
    "tree_sha256": "2f3badd2019a68c6549a66f85b5486faf626078a3e438342f67aacc4edc95789",
    "split_lock_sha256": "7e124f29d2e4d24c17fd1fefcf8753ececd3a8cd86e8c6a2f159e2e5d2217246",
    "manifest_sha256": "c1105002485ebee88c13f56a2198b7b5edbdb9d5a0f4d641f4e644f7b211e566",
    "target_xyz_in_model_input": False,
}
SOURCE_IDENTITIES = {
    "groot_source_commit": "51d4c89f72fda44cbf77285c6a8114b52676b8a1",
    "cosmos_revision": "9ce19a195e423419c349abfc86fd07178b230561",
    "cosmos_tree_sha256": "1cfec77c944cba341ee5d7c31b4be6bb615582364bae48f18c5686e6ec23e8a9",
    "groot_runtime_manifest_sha256": "96b883e56b54f763bec60d3a336632ad531f4fc52d099f803e5f9efba60eab11",
    "groot_payload_tree_sha256": "c0d57a6a74ab24c48abdd1c13420e9d1c893dc0d7b5ca370fa9b6f7af32f5f1b",
    "modality_sha256": "e61f9429ecc21ceab51d9d26cadc9018691eb8b5c74d60aec4abf7faee16f031",
    "processor_config_sha256": "85c1b4690ae090559e79a45193e598b65d6146eedf14750884da65e6d31032be",
    "decoder_source_sha256": "2175381660a94c59e54388c01b8fd397bd11439fb1d016d5a2116c01d5158fae",
    "metrics_source_sha256": "913be32a7bc418b3213f7d2277d59831d40fec9ab81a171efc5918bb8e8f5ed4",
}
CALL_BUDGET = {
    "checkpoint_order": list(CHECKPOINT_ORDER),
    "windows_per_checkpoint": 24,
    "noise_seeds": [2223, 2224, 2225],
    "t_values": [0.06776241796562847, 0.36966943557751086, 0.7837719744658148],
    "t_buckets": [67, 369, 783],
    "action_head_calls": 648,
    "conditioning_passes": 72,
    "total_model_side_calls": 720,
    "adaptive_extension": False,
    "checkpoint_order_is_frozen": True,
}
SAMPLER_CONTRACT = {
    "checkpoint_order": list(CHECKPOINT_ORDER),
    "noise_seeds": [2223, 2224, 2225],
    "t_values": [0.06776241796562847, 0.36966943557751086, 0.7837719744658148],
    "t_buckets": [67, 369, 783],
    "same_noise_across_checkpoints": True,
    "free_running_sampler_calls": 0,
}
FORBIDDEN_IMPORT_PREFIXES = ("torch", "transformers", "gr00t", "cuda")
LOCKED_BOUNDARIES = {
    "Validation-E": "LOCKED",
    "Holdout-F": "LOCKED",
    "WBC": "LOCKED",
    "simulator": "LOCKED",
    "hardware": "LOCKED",
    "new_data": "LOCKED",
}
ADMISSION_ORDER = (
    "pre_model_import_guard",
    "protocol_provenance_binding",
    "observed_checkout_binding",
    "observed_host_gpu_runtime_binding",
    "authority_token_binding",
    "checkpoint_stats_dataset_source_identity_binding",
    "sampler_seed_t_grid_call_budget_binding",
    "target_xyz_exclusion_binding",
    "locked_boundary_binding",
    "fresh_output_root_binding",
    "zero_counter_and_execution_boundary_binding",
)


class RuntimeAdmissionError(RuntimeError):
    """Raised when a future D1 runtime is not exactly admissible."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeAdmissionError(message)


def _read_json(path: Path) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"JSON must be a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeAdmissionError(f"invalid JSON: {path}") from exc
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{field} must be an object")
    return value


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_compatibility_map(path: Path = COMPATIBILITY_MAP_PATH) -> dict[str, Any]:
    """Validate the static Corelab↔Vanda responsibility map."""

    payload = _read_json(path)
    _require(payload.get("schema") == COMPATIBILITY_SCHEMA, "compatibility-map schema drift")
    _require(payload.get("status") == "STATIC_ONLY_NOT_EXECUTED", "compatibility map must remain static-only")
    authority = _mapping(payload.get("authority"), field="compatibility.authority")
    _require(authority.get("issue_11_comment") == PI_GATE_COMMENT, "PI0117 compatibility authority drift")
    _require(authority.get("pr_51_comment") == PR51_POINTER_COMMENT, "PR #51 compatibility pointer drift")
    _require(authority.get("issue_50_comment") == ISSUE50_POINTER_COMMENT, "Issue #50 pointer drift")
    provenance = _mapping(payload.get("protocol_provenance"), field="compatibility.protocol_provenance")
    _require(provenance.get("head") == PROTOCOL_PROVENANCE_HEAD, "protocol provenance HEAD drift")
    _require(provenance.get("sha256") == PROTOCOL_SHA256, "protocol provenance SHA drift")
    runtime = _mapping(payload.get("vanda_runtime_reference"), field="compatibility.vanda_runtime_reference")
    _require(runtime.get("pr_number") == 49, "Vanda PR reference drift")
    _require(runtime.get("head") == "7d055069ba1d391c17457f709f7fc9aff544ef02", "Vanda runtime HEAD drift")
    responsibilities = payload.get("responsibility_map")
    _require(isinstance(responsibilities, list) and len(responsibilities) >= 10, "compatibility responsibility map is incomplete")
    for row in responsibilities:
        item = _mapping(row, field="compatibility.responsibility_map[]")
        for key in ("boundary", "d1_contract", "vanda_mechanism", "late_bound_identity"):
            _require(isinstance(item.get(key), str) and item[key], f"compatibility row missing {key}")
    _require(payload.get("scientific_semantics_transfer") == "NONE", "compatibility map permits scientific semantic transfer")
    return payload


def validate_admission_manifest(path: Path = ADMISSION_MANIFEST_PATH) -> dict[str, Any]:
    """Validate the unmaterialized, late-bound admission package."""

    payload = _read_json(path)
    _require(payload.get("schema") == ADMISSION_SCHEMA, "runtime-admission schema drift")
    _require(payload.get("status") == "STATIC_TEMPLATE_NOT_EXECUTED", "runtime-admission package must remain static-only")
    _require(payload.get("return_token") == RETURN_TOKEN, "runtime-admission return token drift")
    authority = _mapping(payload.get("authority"), field="admission.authority")
    _require(authority.get("issue_11_comment") == PI_GATE_COMMENT, "PI0117 authority drift")
    _require(authority.get("pr_51_comment") == PR51_POINTER_COMMENT, "PR #51 pointer drift")
    _require(authority.get("issue_50_comment") == ISSUE50_POINTER_COMMENT, "Issue #50 pointer drift")
    provenance = _mapping(payload.get("protocol_provenance"), field="admission.protocol_provenance")
    _require(provenance.get("head") == PROTOCOL_PROVENANCE_HEAD, "protocol provenance HEAD drift")
    _require(provenance.get("sha256") == PROTOCOL_SHA256, "protocol provenance SHA drift")
    checkout = _mapping(payload.get("execution_checkout"), field="admission.execution_checkout")
    _require(checkout.get("binding_mode") == "OBSERVED_RUNTIME_ONLY", "execution checkout is not late-bound")
    for key in ("authorized_branch", "authorized_head", "authorized_clean_worktree", "observed_repo_root", "observed_branch", "observed_head", "observed_clean_worktree"):
        _require(checkout.get(key) is None, f"static template contains execution checkout value: {key}")
    observed = _mapping(payload.get("runtime_observation"), field="admission.runtime_observation")
    for key in ("host", "python", "torch", "cuda", "gpu", "pbs_job_id", "offline_mode"):
        _require(observed.get(key) is None, f"static template contains observed runtime value: {key}")
    _require(payload.get("admission_order") == list(ADMISSION_ORDER), "admission ordering drift")
    identities = _mapping(payload.get("identity_bindings"), field="admission.identity_bindings")
    _require(identities.get("checkpoint_order") == list(CHECKPOINT_ORDER), "checkpoint order drift")
    _require(identities.get("stats_routes") == STATS_ROUTES, "statistics route map drift")
    _require(identities.get("stats_sha256") == STATS_SHA256, "statistics identity map drift")
    _require(identities.get("stats_payload_sha256") == STATS_PAYLOAD_SHA256, "statistics payload identity drift")
    _require(identities.get("checkpoint_tree_sha256") == CHECKPOINT_TREE_SHA256, "checkpoint identity map drift")
    _require(identities.get("dataset") == DATASET_IDENTITY, "dataset identity drift")
    _require(identities.get("source_identities") == SOURCE_IDENTITIES, "source identity map drift")
    _require(payload.get("call_budget") == CALL_BUDGET, "call budget/order drift")
    _require(payload.get("locked_boundaries") == LOCKED_BOUNDARIES, "locked boundary map drift")
    guard = _mapping(payload.get("pre_model_guard"), field="admission.pre_model_guard")
    _require(guard.get("required") is True, "pre-model guard is not required")
    _require(tuple(guard.get("forbidden_import_prefixes", ())) == FORBIDDEN_IMPORT_PREFIXES, "pre-model import guard drift")
    _require(payload.get("execution_authorized") is False, "static admission must not authorize execution")
    _require(payload.get("pi_authority_token") is None, "static admission token must be absent")
    counters = _mapping(payload.get("zero_counters"), field="admission.zero_counters")
    _require(all(value == 0 for value in counters.values()), "static admission counters are nonzero")
    return payload


def model_imports_present(imported_modules: Mapping[str, Any] | None = None) -> list[str]:
    """Return forbidden model/runtime modules already present in the process."""

    modules = sys.modules if imported_modules is None else imported_modules
    found = []
    for name in modules:
        if name == "" or not isinstance(name, str):
            continue
        if any(name == prefix or name.startswith(prefix + ".") for prefix in FORBIDDEN_IMPORT_PREFIXES):
            found.append(name)
    return sorted(found)


def _require_pre_model_guard(imported_modules: Mapping[str, Any] | None = None) -> None:
    found = model_imports_present(imported_modules)
    _require(not found, "model-side import already present before admission: " + ",".join(found))


def _fresh_output_root(path_value: Any) -> Path:
    _require(isinstance(path_value, str) and path_value, "output_root must be a nonempty path")
    path = Path(path_value).expanduser()
    _require(not path.is_symlink(), "output_root may not be a symlink")
    if path.exists():
        _require(path.is_dir(), "output_root must be a directory")
        _require(not any(path.iterdir()), "output_root must be fresh and empty")
    return path.resolve()


def _exact(value: Any, expected: Any, *, field: str) -> None:
    _require(value == expected, f"{field} differs from the frozen D1 contract")


def validate_future_admission(
    binding: Mapping[str, Any],
    observed: Mapping[str, Any],
    *,
    authority_token: str | None,
    imported_modules: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a future materialized binding before any model import.

    The pre-model import guard is deliberately first.  Every subsequent
    comparison is exact, including the execution checkout HEAD; the immutable
    protocol provenance HEAD is never accepted as a substitute.
    """

    _require_pre_model_guard(imported_modules)
    _require(binding.get("schema") == "CORELAB_STAGE45D_D1_RUNTIME_BINDING_V2", "runtime binding schema mismatch")
    _require(binding.get("execution_authorized") is True, "runtime binding is not execution-authorized")
    token = binding.get("pi_authority_token")
    _require(isinstance(token, str) and token, "runtime binding authority token is absent")
    _require(authority_token == token, "runtime binding authority token mismatch")
    _exact(binding.get("protocol_provenance_head"), PROTOCOL_PROVENANCE_HEAD, field="protocol_provenance_head")
    checkout = _mapping(binding.get("execution_checkout"), field="binding.execution_checkout")
    observed_checkout = _mapping(observed.get("execution_checkout"), field="observed.execution_checkout")
    _exact(checkout.get("binding_mode"), "OBSERVED_RUNTIME_ONLY", field="execution_checkout.binding_mode")
    for key in ("authorized_repo_root", "authorized_branch", "authorized_head", "authorized_clean_worktree"):
        _require(checkout.get(key) not in (None, ""), f"{key} is not late-bound")
    _require(checkout.get("authorized_clean_worktree") is True, "authorized execution checkout must be clean")
    for key in ("observed_repo_root", "observed_branch", "observed_head", "observed_clean_worktree"):
        _require(observed_checkout.get(key) not in (None, ""), f"observed {key} is missing")
    _exact(observed_checkout.get("observed_repo_root"), checkout.get("authorized_repo_root"), field="execution repo root")
    _exact(observed_checkout.get("observed_branch"), checkout.get("authorized_branch"), field="execution branch")
    _exact(observed_checkout.get("observed_head"), checkout.get("authorized_head"), field="execution HEAD")
    _exact(observed_checkout.get("observed_clean_worktree"), checkout.get("authorized_clean_worktree"), field="execution worktree")
    _require(str(observed_checkout["observed_repo_root"]).startswith("/"), "observed repo root must be absolute")
    _require(checkout.get("authorized_head") != PROTOCOL_PROVENANCE_HEAD, "execution HEAD may not alias protocol provenance HEAD")

    _exact(observed.get("runtime_identity"), binding.get("runtime_identity"), field="runtime identity")
    _exact(binding.get("source_identities"), SOURCE_IDENTITIES, field="source identities")
    _exact(observed.get("source_identities"), SOURCE_IDENTITIES, field="source identities")
    _exact(binding.get("checkpoint_identities"), CHECKPOINT_IDENTITIES, field="checkpoint identities")
    _exact(observed.get("checkpoint_identities"), CHECKPOINT_IDENTITIES, field="checkpoint identities")
    _exact(binding.get("statistics_routes"), STATS_ROUTES, field="statistics routes")
    _exact(observed.get("statistics_routes"), STATS_ROUTES, field="statistics routes")
    _exact(binding.get("dataset_identity"), DATASET_IDENTITY, field="dataset identity")
    _exact(observed.get("dataset_identity"), DATASET_IDENTITY, field="dataset identity")
    _exact(binding.get("sampler"), SAMPLER_CONTRACT, field="sampler/seed/t grid")
    _exact(observed.get("sampler"), SAMPLER_CONTRACT, field="sampler/seed/t grid")
    _exact(observed.get("host_gpu_runtime"), binding.get("host_gpu_runtime"), field="host/GPU/runtime observation")
    _exact(observed.get("call_budget"), CALL_BUDGET, field="call budget")
    _exact(observed.get("target_xyz_in_model_input"), False, field="target XYZ exclusion")
    _exact(observed.get("locked_boundaries"), LOCKED_BOUNDARIES, field="locked boundaries")
    _exact(binding.get("zero_counters"), {"model_construction": 0, "model_forward": 0, "gpu": 0, "pbs": 1, "training_steps": 0}, field="pre-model runtime counters")
    _exact(observed.get("zero_counters"), binding.get("zero_counters"), field="observed zero counters")
    output = _mapping(binding.get("output"), field="binding.output")
    observed_output = _mapping(observed.get("output"), field="observed.output")
    _exact(observed_output.get("root"), output.get("root"), field="output root")
    _require(observed_output.get("fresh") is True and output.get("fresh_required") is True, "output root freshness is not asserted")
    output_root = _fresh_output_root(observed_output.get("root"))
    _require(observed.get("model_imported") is False, "model import occurred before admission")
    _require(observed.get("model_construction") is False and observed.get("model_forward") is False, "model execution occurred before admission")
    _require(observed.get("offline_mode") is True, "offline mode is required")
    return {
        "schema": "CORELAB_STAGE45D_D1_RUNTIME_ADMISSION_V2",
        "status": "PASS_PRE_MODEL",
        "execution_branch": checkout["authorized_branch"],
        "execution_head": checkout["authorized_head"],
        "execution_worktree_clean": checkout["authorized_clean_worktree"],
        "output_root": str(output_root),
        "admission_order": list(ADMISSION_ORDER),
        "model_imported": False,
        "model_construction": False,
        "model_forward": False,
        "counters": dict(binding["zero_counters"]),
    }


__all__ = [
    "ADMISSION_MANIFEST_PATH",
    "COMPATIBILITY_MAP_PATH",
    "ADMISSION_ORDER",
    "CALL_BUDGET",
    "CHECKPOINT_ORDER",
    "COMPATIBILITY_SCHEMA",
    "FORBIDDEN_IMPORT_PREFIXES",
    "LOCKED_BOUNDARIES",
    "PROTOCOL_PROVENANCE_HEAD",
    "RuntimeAdmissionError",
    "validate_admission_manifest",
    "validate_compatibility_map",
    "validate_future_admission",
    "model_imports_present",
]
