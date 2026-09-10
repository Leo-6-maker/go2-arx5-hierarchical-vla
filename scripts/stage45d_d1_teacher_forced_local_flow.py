#!/usr/bin/env python3
"""Stage45D D1 teacher-forced/local-flow execution harness.

The default ``--plan-only``/``--audit-only`` paths are CPU/static only.  They
validate the reviewed protocol, execution manifest, source-equivalence map,
checkpoint-specific statistics routing, flow algebra, evidence lifecycle, and
the late-bound runtime schema.  They never import Torch/GR00T/CUDA and never
construct a model.

``--execute`` is deliberately fail-closed in this PI0107 static closure pass.
It can proceed only when a future PI-authorized runtime binding explicitly
enables execution, supplies a matching authority token, and opts into the
later runtime adapter.  The guarded import is below all static/admission
checks, so an accidental execute flag cannot expose a model on Corelab.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
SCRIPTS = REPO / "scripts"
for _path in (SRC, SCRIPTS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from go2arx5_vla.structured_action import structured_action_mask  # noqa: E402
from stage45d_d1_runtime_admission import (  # noqa: E402
    ADMISSION_MANIFEST_PATH,
    COMPATIBILITY_MAP_PATH as ADMISSION_COMPATIBILITY_MAP_PATH,
    RuntimeAdmissionError,
    validate_admission_manifest,
    validate_compatibility_map,
)


PROTOCOL_PATH = REPO / "manifests" / "STAGE45D_D1_PROTOCOL_CORRECTED_V1.json"
EXECUTION_MANIFEST_PATH = REPO / "manifests" / "STAGE45D_D1_EXECUTION_MANIFEST_V1.json"
RUNTIME_BINDING_PATH = REPO / "manifests" / "STAGE45D_D1_RUNTIME_BINDING_TEMPLATE_V1.json"
RUNTIME_ADMISSION_MANIFEST_PATH = ADMISSION_MANIFEST_PATH
COMPATIBILITY_MAP_PATH = ADMISSION_COMPATIBILITY_MAP_PATH

PROTOCOL_SHA256 = "24cebea0d5bd884c8eb4ca1db2b9d55dfae57326936939ccd09d3868c96309b3"
PROTOCOL_PROVENANCE_HEAD = "1dd5837267cb1af34cd2ae7a978f70d6c8f78bdb"
GROOT_SOURCE_COMMIT = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
FULL24_STATS_SHA256 = "3b3a2d51e685a0e580de4ebdfa05abcbcddfb7dc33b8d3a659994cdcd1bb0337"
FULL24_STATS_PAYLOAD_SHA256 = "752898989bb3308e663c4b6cf88b3cc1c139897a6860e6aed1d9d708d0f6fd75"
HISTORICAL_STATS_SHA256 = "0553436da13b4500e896cf4ec06ef36aba8d230536595cdd219818d912567206"
STRUCTURED_TREE_SHA256 = "30b14d36e47e08dfbb9d3b353d199971863d6e7f6ceedea2d478ef367d0a1137"
FULL24_1000_TREE_SHA256 = "38a00f578eededcebd26fa07ecfdba095a4edc64afd949599f1de5284efea12c"
FULL24_2000_TREE_SHA256 = "2113a6a3b2a91233a42214dbb017c0136acde282fda2a418066f9317e78981a5"
FULL24_DATASET_TREE_SHA256 = "2f3badd2019a68c6549a66f85b5486faf626078a3e438342f67aacc4edc95789"
FULL24_SPLIT_SHA256 = "7e124f29d2e4d24c17fd1fefcf8753ececd3a8cd86e8c6a2f159e2e5d2217246"
FULL24_MANIFEST_SHA256 = "c1105002485ebee88c13f56a2198b7b5edbdb9d5a0f4d641f4e644f7b211e566"
GOLDEN_PROMOTION_SHA256 = "a6f803e257ceeb0434027d8a35e32a7b8c66baf8226234bd6299cebf417446f9"
GOLDEN_RELEASE_V2_SHA256 = "8306b76f327b260310a22a5d22551ebf299366f8fea4124bcbfbdab87e807666"
INTEGRITY_AUDIT_SHA256 = "a672e39b45e025f9a7681c1a1a45d807dd9be2a4641e54db60b036252bb59539"
COSMOS_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"
COSMOS_TREE_SHA256 = "1cfec77c944cba341ee5d7c31b4be6bb615582364bae48f18c5686e6ec23e8a9"
GROOT_RUNTIME_MANIFEST_SHA256 = "96b883e56b54f763bec60d3a336632ad531f4fc52d099f803e5f9efba60eab11"
GROOT_PAYLOAD_TREE_SHA256 = "c0d57a6a74ab24c48abdd1c13420e9d1c893dc0d7b5ca370fa9b6f7af32f5f1b"
MODALITY_SHA256 = "e61f9429ecc21ceab51d9d26cadc9018691eb8b5c74d60aec4abf7faee16f031"
PROCESSOR_CONFIG_SHA256 = "85c1b4690ae090559e79a45193e598b65d6146eedf14750884da65e6d31032be"
DECODER_SOURCE_SHA256 = "2175381660a94c59e54388c01b8fd397bd11439fb1d016d5a2116c01d5158fae"
METRICS_SOURCE_SHA256 = "913be32a7bc418b3213f7d2277d59831d40fec9ab81a171efc5918bb8e8f5ed4"

RAW_SHAPE = (1, 40, 132)
PHYSICAL_SHAPE = (40, 10)
T_VALUES = (0.06776241796562847, 0.36966943557751086, 0.7837719744658148)
T_BUCKETS = (67, 369, 783)
NOISE_SEEDS = (2223, 2224, 2225)
CHECKPOINTS = ("STRUCTURED@1000", "FULL24@1000", "FULL24@2000")
TRAIN24_WINDOWS = 24
ACTION_HEAD_CALLS = len(CHECKPOINTS) * TRAIN24_WINDOWS * len(NOISE_SEEDS) * len(T_VALUES)
CONDITIONING_PASSES = len(CHECKPOINTS) * TRAIN24_WINDOWS
TOTAL_MODEL_SIDE_CALLS = ACTION_HEAD_CALLS + CONDITIONING_PASSES

SOURCE_EQUIVALENCE = {
    "pinned_groot_source_commit": GROOT_SOURCE_COMMIT,
    "backbone_conditioning": {
        "path": "scripts/stage38_structured_offline_evaluation.py",
        "symbol": "_prepare_model_conditioning",
        "operations": ["model.prepare_input", "model.backbone", "action_head.process_backbone_output"],
    },
    "action_state_encoder": {
        "path": "scripts/stage38_structured_offline_evaluation.py",
        "symbol": "_prepare_model_conditioning",
        "operations": ["action_head.state_encoder", "action_head.action_encoder"],
    },
    "t_discretization": {
        "path": "scripts/stage38_structured_offline_evaluation.py",
        "symbol": "_load_fixed_probes",
        "contract": "t values and integer buckets are the reviewed D1 grid",
    },
    "dit_action_head_forward": {
        "path": "scripts/stage38_structured_offline_evaluation.py",
        "symbol": "_offline_step",
        "operations": ["action_encoder", "action_head.model", "action_decoder"],
    },
    "velocity_decoder": {
        "path": "scripts/stage38_structured_offline_evaluation.py",
        "symbol": "_offline_step",
        "operation": "velocity = pred[:, -actions.shape[1]:]",
    },
    "physical_decode_bridge": {
        "path": "scripts/stage45c_h200_eval_runtime_adapter.py",
        "symbol": "_decode_v2",
        "contract": "Stage40 inverse-normalization + StructuredActionV1 materialization with runtime-V2 statistics",
    },
}

STATS_ROUTES = {
    "STRUCTURED@1000": {
        "route": "HISTORICAL_STAGE40_FIXED_STATS",
        "stats_sha256": HISTORICAL_STATS_SHA256,
        "tree_sha256": STRUCTURED_TREE_SHA256,
        "historical_fixed_identity_check": True,
    },
    "FULL24@1000": {
        "route": "STAGE45C_RUNTIME_V2_STATS_BRIDGE",
        "stats_sha256": FULL24_STATS_SHA256,
        "payload_sha256": FULL24_STATS_PAYLOAD_SHA256,
        "tree_sha256": FULL24_1000_TREE_SHA256,
        "historical_fixed_identity_check": False,
    },
    "FULL24@2000": {
        "route": "STAGE45C_RUNTIME_V2_STATS_BRIDGE",
        "stats_sha256": FULL24_STATS_SHA256,
        "payload_sha256": FULL24_STATS_PAYLOAD_SHA256,
        "tree_sha256": FULL24_2000_TREE_SHA256,
        "historical_fixed_identity_check": False,
    },
}

RUNTIME_PATH_KEYS = (
    "base_checkpoint",
    "structured_1000_checkpoint",
    "full24_1000_checkpoint",
    "full24_2000_checkpoint",
    "full24_dataset",
    "snapshot_root",
    "groot_source",
    "cosmos_root",
    "modality_config",
    "base_processor_config",
    "base_processor_statistics",
    "structured_1000_statistics",
    "full24_runtime_statistics",
    "golden_release_manifest",
    "groot_runtime_manifest",
    "full24_manifest",
    "checkpoint_integrity_audit",
    "runtime_admission",
    "output_root",
)


class HarnessError(RuntimeError):
    """Raised when the D1 harness cannot prove its frozen contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HarnessError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def read_json(path: Path) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"JSON must be a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HarnessError(f"invalid JSON: {path}") from exc
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def validate_reviewed_protocol(path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    _require(sha256_file(path) == PROTOCOL_SHA256, "reviewed corrected protocol SHA256 drift")
    payload = read_json(path)
    _require(payload.get("schema") == "CORELAB_STAGE45D_D1_TEACHER_FORCED_DIAGNOSTIC_PROTOCOL_CORRECTED_V1", "reviewed protocol schema drift")
    _require(payload.get("status") == "CORRECTED_FOR_PI_REVIEW_NOT_EXECUTED", "reviewed protocol is not static-only")
    _require(payload.get("authority", {}).get("issue_11_comment") == 5617025466, "PI0104 authority drift")
    _require(payload.get("authority", {}).get("issue_50_comment") == 5617028048, "PI0104 execution pointer drift")
    _require(payload.get("supervision_mask", {}).get("shape") == [40, 132], "reviewed mask shape drift")
    _require(payload.get("supervision_mask", {}).get("active_scalar_count") == 190, "reviewed mask active count drift")
    _require(payload.get("scientific_scope", {}).get("split") == "TRAIN_ONLY", "reviewed split drift")
    _require(payload.get("scientific_scope", {}).get("target_xyz_in_model_input") is False, "target XYZ is not excluded")
    return payload


def validate_source_equivalence() -> dict[str, Any]:
    checks: dict[str, Any] = {"pinned_groot_source_commit": GROOT_SOURCE_COMMIT}
    required = {
        "scripts/stage38_structured_offline_evaluation.py": ("_prepare_model_conditioning", "_load_fixed_probes", "_offline_step"),
        "scripts/stage45c_h200_eval_runtime_adapter.py": ("_decode_v2",),
    }
    for relative, symbols in required.items():
        path = REPO / relative
        _require(path.is_file(), f"source-equivalence file missing: {relative}")
        text = path.read_text(encoding="utf-8")
        for symbol in symbols:
            _require(symbol in text, f"source-equivalence anchor missing: {relative}:{symbol}")
        checks[relative] = {"sha256": sha256_file(path), "symbols": list(symbols)}
    checks["map"] = SOURCE_EQUIVALENCE
    return checks


def validate_mask_contract() -> dict[str, Any]:
    mask = np.asarray(structured_action_mask("T16"), dtype=np.float64)
    _require(mask.shape == RAW_SHAPE[1:], f"T16 mask shape drift: {mask.shape}")
    _require(int(np.count_nonzero(mask)) == 190, "T16 mask active scalar count drift")
    expected = np.zeros((40, 132), dtype=np.float64)
    expected[:15, :9] = 1.0
    expected[:15, 9] = 1.0
    expected[:, 10] = 1.0
    _require(np.array_equal(mask, expected), "T16 mask topology differs from Stage45 D1 law")
    legacy = np.zeros_like(mask)
    legacy[:, :10] = 1.0
    _require(not np.array_equal(legacy, mask), "legacy dense first-ten mask was accepted")
    return {
        "source": "go2arx5_vla.structured_action.structured_action_mask(\"T16\")",
        "shape": list(mask.shape),
        "active_scalar_count": int(np.count_nonzero(mask)),
        "inactive_scalar_count": int(mask.size - np.count_nonzero(mask)),
        "block_counts": {
            "geometry": int(np.count_nonzero(mask[:15, :9])),
            "timing": int(np.count_nonzero(mask[:15, 9])),
            "gripper": int(np.count_nonzero(mask[:, 10])),
            "remaining": int(np.count_nonzero(mask[:, 11:])),
        },
        "legacy_first_ten_rejected": True,
    }


def construct_flow(noise: Any, x1: Any, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct the frozen float32 interpolation and local-flow target."""

    x0 = np.asarray(noise, dtype=np.float32)
    endpoint = np.asarray(x1, dtype=np.float32)
    _require(x0.shape == RAW_SHAPE and endpoint.shape == RAW_SHAPE, "flow tensors must be [1,40,132]")
    _require(np.isfinite(x0).all() and np.isfinite(endpoint).all(), "flow tensors must be finite")
    _require(np.isfinite(t) and 0.0 <= float(t) <= 1.0, "flow t must be in [0,1]")
    x_t = ((1.0 - float(t)) * x0 + float(t) * endpoint).astype(np.float32, copy=False)
    v_target = (endpoint - x0).astype(np.float32, copy=False)
    _require(np.isfinite(x_t).all() and np.isfinite(v_target).all(), "constructed flow tensors are nonfinite")
    return x0, x_t, v_target


def implied_endpoint(x_t: Any, v_hat: Any, t: float) -> np.ndarray:
    """Map a predicted local velocity to its implied normalized endpoint."""

    state = np.asarray(x_t, dtype=np.float32)
    velocity = np.asarray(v_hat, dtype=np.float32)
    _require(state.shape == RAW_SHAPE and velocity.shape == RAW_SHAPE, "implied-endpoint tensors must be [1,40,132]")
    _require(np.isfinite(state).all() and np.isfinite(velocity).all(), "implied-endpoint inputs are nonfinite")
    _require(np.isfinite(t) and 0.0 <= float(t) <= 1.0, "implied-endpoint t must be in [0,1]")
    endpoint = (state + (1.0 - float(t)) * velocity).astype(np.float32, copy=False)
    _require(np.isfinite(endpoint).all(), "implied endpoint is nonfinite")
    return endpoint


def inverse_normalize_implied_endpoint(
    checkpoint: str,
    x1_hat: Any,
    lower: Any,
    upper: Any,
    mask: Any,
    *,
    stats_sha256: str,
    stats_payload_sha256: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the Stage45C V2 inverse-normalization law to ``x1_hat``.

    This is intentionally a CPU bridge only.  It returns the physical-model
    input tensor for the existing StructuredAction decoder; it never imports
    or calls that decoder.  Routing and statistics identities are checked
    before any conversion, and the original normalized ``x1_hat`` remains
    untouched by the caller.
    """

    route = select_stats_route(checkpoint)
    _require(stats_sha256 == route["stats_sha256"], f"{checkpoint} statistics SHA does not match its route")
    if checkpoint.startswith("FULL24"):
        _require(route["route"] == "STAGE45C_RUNTIME_V2_STATS_BRIDGE", "FULL24 route is not the V2 bridge")
        _require(stats_payload_sha256 == FULL24_STATS_PAYLOAD_SHA256, "FULL24 runtime-V2 payload SHA drift")
    else:
        _require(route["route"] == "HISTORICAL_STAGE40_FIXED_STATS", "historical route drift")

    normalized = np.asarray(x1_hat, dtype=np.float32)
    lower_array = np.asarray(lower, dtype=np.float64)
    upper_array = np.asarray(upper, dtype=np.float64)
    active_mask = np.asarray(mask, dtype=np.float64)
    _require(normalized.shape == RAW_SHAPE and np.isfinite(normalized).all(), "implied endpoint shape/finite contract")
    _require(lower_array.shape == (132,) and upper_array.shape == (132,), "statistics bounds must be [132]")
    _require(np.isfinite(lower_array).all() and np.isfinite(upper_array).all(), "statistics bounds are nonfinite")
    _require(np.all(upper_array >= lower_array), "statistics bounds are not ordered")
    canonical_mask = np.asarray(structured_action_mask("T16"), dtype=np.float64)
    _require(active_mask.shape == canonical_mask.shape and np.array_equal(active_mask, canonical_mask), "decode mask is not canonical T16")

    clipped = np.clip(normalized[0].astype(np.float64), -1.0, 1.0)
    values = (clipped + 1.0) * 0.5 * (upper_array[None, :] - lower_array[None, :]) + lower_array[None, :]
    values[active_mask == 0.0] = 0.0
    _require(values.shape == (40, 132) and np.isfinite(values).all(), "inverse-normalized implied endpoint is invalid")
    return values, {
        "checkpoint": checkpoint,
        "stats_route": route["route"],
        "stats_sha256": stats_sha256,
        "stats_payload_sha256": stats_payload_sha256,
        "normalized_source": "x1_hat",
        "decoder_input_shape": list(values.shape),
        "mask_active_scalar_count": int(np.count_nonzero(active_mask)),
    }


def validate_flow_algebra() -> dict[str, Any]:
    noise = np.linspace(-0.75, 0.75, num=np.prod(RAW_SHAPE), dtype=np.float32).reshape(RAW_SHAPE)
    endpoint = np.linspace(0.6, -0.4, num=np.prod(RAW_SHAPE), dtype=np.float32).reshape(RAW_SHAPE)
    rows = []
    for t in T_VALUES:
        x0, x_t, v_target = construct_flow(noise, endpoint, t)
        recovered = implied_endpoint(x_t, v_target, t)
        error = float(np.max(np.abs(recovered.astype(np.float64) - endpoint.astype(np.float64))))
        _require(error <= 2e-6, f"oracle implied endpoint reconstruction drift at t={t}: {error}")
        rows.append({"t": t, "max_abs_oracle_endpoint_error": error, "x0_sha256": sha256_bytes(x0.tobytes()), "x_t_sha256": sha256_bytes(x_t.tobytes()), "v_target_sha256": sha256_bytes(v_target.tobytes()), "x1_hat_sha256": sha256_bytes(recovered.tobytes())})
    return {
        "formula": "x_t=(1-t)*x0+t*x1; v_target=x1-x0; x1_hat=x_t+(1-t)*v_hat",
        "oracle_v_hat_equals_v_target": True,
        "t_values": list(T_VALUES),
        "rows": rows,
    }


def select_stats_route(checkpoint: str) -> dict[str, Any]:
    try:
        return dict(STATS_ROUTES[checkpoint])
    except KeyError as exc:
        raise HarnessError(f"unknown D1 checkpoint: {checkpoint}") from exc


def validate_physical_decode_route(checkpoint: str, route: str) -> None:
    expected = select_stats_route(checkpoint)["route"]
    _require(route == expected, f"{checkpoint} must route through {expected}, got {route}")
    if checkpoint.startswith("FULL24"):
        _require(route != "HISTORICAL_STAGE40_FIXED_STATS", "FULL24 may not use historical Structured-only fixed stats")


def validate_stats_routing() -> dict[str, Any]:
    for checkpoint, details in STATS_ROUTES.items():
        validate_physical_decode_route(checkpoint, details["route"])
    try:
        validate_physical_decode_route("FULL24@1000", "HISTORICAL_STAGE40_FIXED_STATS")
    except HarnessError:
        legacy_rejected = True
    else:
        legacy_rejected = False
    _require(legacy_rejected, "FULL24 historical fixed-stat route was not rejected")
    lower = np.full((132,), -1.0, dtype=np.float64)
    upper = np.full((132,), 1.0, dtype=np.float64)
    endpoint = np.zeros(RAW_SHAPE, dtype=np.float32)
    converted, details = inverse_normalize_implied_endpoint(
        "FULL24@1000",
        endpoint,
        lower,
        upper,
        structured_action_mask("T16"),
        stats_sha256=FULL24_STATS_SHA256,
        stats_payload_sha256=FULL24_STATS_PAYLOAD_SHA256,
    )
    _require(converted.shape == (40, 132) and details["normalized_source"] == "x1_hat", "V2 bridge output contract drift")
    return {"routes": STATS_ROUTES, "full24_historical_route_rejected": legacy_rejected}


def validate_call_budget() -> dict[str, int]:
    _require(ACTION_HEAD_CALLS == 648, "D1 action-head budget drift")
    _require(CONDITIONING_PASSES == 72, "D1 conditioning budget drift")
    _require(TOTAL_MODEL_SIDE_CALLS == 720, "D1 total model-side budget drift")
    return {"action_head_calls": ACTION_HEAD_CALLS, "conditioning_passes": CONDITIONING_PASSES, "total_model_side_calls": TOTAL_MODEL_SIDE_CALLS}


def validate_execution_manifest(path: Path = EXECUTION_MANIFEST_PATH) -> dict[str, Any]:
    payload = read_json(path)
    _require(payload.get("schema") == "CORELAB_STAGE45D_D1_EXECUTION_MANIFEST_V1", "execution manifest schema drift")
    _require(payload.get("status") == "READY_FOR_PI_REVIEW_NOT_EXECUTED", "execution manifest must remain unexecuted")
    _require(payload.get("protocol", {}).get("sha256") == PROTOCOL_SHA256, "execution manifest protocol SHA drift")
    protocol = payload.get("protocol", {})
    _require(protocol.get("provenance_head") == PROTOCOL_PROVENANCE_HEAD, "execution manifest protocol provenance head drift")
    _require(protocol.get("provenance_role") == "IMMUTABLE_PROTOCOL_PROVENANCE_ONLY", "execution manifest protocol provenance role drift")
    _require(protocol.get("execution_checkout_identity") == "LATE_BOUND_OBSERVED_RUNTIME_HEAD_BRANCH_WORKTREE", "execution checkout identity is not late-bound")
    runner = payload.get("runner", {})
    _require(runner.get("path") == "scripts/stage45d_d1_teacher_forced_local_flow.py", "runner path drift")
    _require(runner.get("sha256") == sha256_file(Path(__file__)), "execution manifest runner SHA drift")
    _require(payload.get("source_equivalence", {}).get("pinned_groot_source_commit") == GROOT_SOURCE_COMMIT, "GR00T source pin drift")
    source_equivalence = payload.get("source_equivalence", {})
    for section, details in source_equivalence.items():
        if not isinstance(details, Mapping) or "path" not in details:
            continue
        source_path = REPO / str(details["path"])
        _require(source_path.is_file(), f"source-equivalence path missing: {details['path']}")
        _require(details.get("sha256") == sha256_file(source_path), f"source-equivalence SHA drift: {section}")
    _require(payload.get("flow", {}).get("noise_shape") == list(RAW_SHAPE), "flow noise shape drift")
    _require(payload.get("flow", {}).get("t_values") == list(T_VALUES), "flow t grid drift")
    _require(payload.get("flow", {}).get("t_buckets") == list(T_BUCKETS), "flow bucket grid drift")
    _require(payload.get("call_budget", {}).get("action_head_calls") == ACTION_HEAD_CALLS, "manifest action-head budget drift")
    _require(payload.get("call_budget", {}).get("conditioning_passes") == CONDITIONING_PASSES, "manifest conditioning budget drift")
    _require(payload.get("call_budget", {}).get("total_model_side_calls") == TOTAL_MODEL_SIDE_CALLS, "manifest total budget drift")
    admission = payload.get("runtime_admission", {})
    _require(admission.get("schema") == "CORELAB_STAGE45D_D1_RUNTIME_ADMISSION_MANIFEST_V1", "runtime-admission schema reference drift")
    _require(admission.get("manifest_path") == "manifests/STAGE45D_D1_RUNTIME_ADMISSION_MANIFEST_V1.json", "runtime-admission manifest path drift")
    _require(admission.get("compatibility_map_path") == "manifests/STAGE45D_D1_RUNTIME_COMPATIBILITY_MAP_V1.json", "compatibility-map path drift")
    _require(admission.get("admission_module") == "scripts/stage45d_d1_runtime_admission.py", "runtime-admission module path drift")
    _require(admission.get("execution_checkout_binding") == "OBSERVED_RUNTIME_ONLY", "runtime-admission checkout binding drift")
    _require(admission.get("pre_model_import_guard") is True and admission.get("model_import_before_admission") is False, "runtime-admission import ordering drift")
    return payload


def validate_runtime_binding_template(path: Path = RUNTIME_BINDING_PATH) -> dict[str, Any]:
    payload = read_json(path)
    _require(payload.get("schema") == "CORELAB_STAGE45D_D1_RUNTIME_BINDING_V1", "runtime binding schema drift")
    _require(payload.get("status") == "LATE_BOUND_NOT_EXECUTED", "runtime binding is not late-bound")
    _require(payload.get("protocol_sha256") == PROTOCOL_SHA256, "runtime binding protocol SHA drift")
    _require(payload.get("protocol_provenance_head") == PROTOCOL_PROVENANCE_HEAD, "runtime binding protocol provenance head drift")
    _require(payload.get("protocol_provenance_role") == "IMMUTABLE_PROTOCOL_PROVENANCE_ONLY", "runtime binding provenance role drift")
    _require(payload.get("runner_path") == "scripts/stage45d_d1_teacher_forced_local_flow.py", "runtime binding runner path drift")
    _require(payload.get("runner_sha256") == sha256_file(Path(__file__)), "runtime binding runner SHA drift")
    _require(payload.get("execution_manifest_sha256") == sha256_file(EXECUTION_MANIFEST_PATH), "runtime binding execution manifest SHA drift")
    _require(payload.get("execution_authorized") is False, "PI0107 binding must not authorize execution")
    _require(payload.get("pi_authority_token") is None, "PI authority token must remain absent")
    admission = payload.get("runtime_admission", {})
    _require(admission.get("schema") == "CORELAB_STAGE45D_D1_RUNTIME_ADMISSION_MANIFEST_V1", "runtime binding admission schema drift")
    _require(admission.get("manifest_path") == "manifests/STAGE45D_D1_RUNTIME_ADMISSION_MANIFEST_V1.json", "runtime binding admission path drift")
    _require(admission.get("compatibility_map_path") == "manifests/STAGE45D_D1_RUNTIME_COMPATIBILITY_MAP_V1.json", "runtime binding compatibility path drift")
    _require(admission.get("binding_mode") == "OBSERVED_RUNTIME_ONLY", "runtime binding mode drift")
    _require(admission.get("pre_model_import_guard") is True, "runtime binding pre-model guard drift")
    checkout = payload.get("execution_checkout", {})
    _require(checkout.get("binding_mode") == "OBSERVED_RUNTIME_ONLY", "runtime binding checkout mode drift")
    _require(checkout.get("protocol_provenance_head_may_substitute") is False, "protocol provenance head may not substitute for execution head")
    for key in ("authorized_branch", "authorized_head", "authorized_clean_worktree", "observed_repo_root", "observed_branch", "observed_head", "observed_clean_worktree"):
        _require(checkout.get(key) is None, f"runtime template contains execution checkout value: {key}")
    paths = payload.get("paths")
    _require(isinstance(paths, Mapping) and tuple(sorted(paths)) == tuple(sorted(RUNTIME_PATH_KEYS)), "runtime binding path key set drift")
    _require(all(value is None for value in paths.values()), "runtime template must not contain node-local paths")
    _require(payload.get("historical_comparator", {}).get("runtime_materialization_status") == "PENDING_VANDA_RECOVERY", "historical comparator state drift")
    _require(payload.get("historical_comparator", {}).get("materialization_path") is None, "historical comparator scratch path was invented")
    return payload


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _write_json_durable(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    _fsync_file(temporary)
    os.replace(temporary, path)


class ProbeEvidenceWriter:
    """Small raw-before-derived writer used by the future execution path."""

    def __init__(self, output_root: Path) -> None:
        self.output_root = Path(output_root)
        self.raw_written = False

    def write_raw(self, *, identity: Mapping[str, Any], arrays: Mapping[str, Any]) -> dict[str, Any]:
        raw_dir = self.output_root / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        required_identity = {
            "checkpoint",
            "checkpoint_tree_sha256",
            "stats_sha256",
            "window_id",
            "noise_seed",
            "t",
            "t_bucket",
        }
        _require(required_identity.issubset(identity), "raw evidence identity is incomplete")
        required_arrays = {"v_hat", "x0", "x_t", "v_target", "x1", "x1_hat", "mask"}
        _require(required_arrays.issubset(arrays), "raw evidence is missing a required flow tensor")
        records: dict[str, Any] = {}
        for name, value in arrays.items():
            array = np.asarray(value, dtype=np.float32)
            _require(np.isfinite(array).all(), f"raw evidence array is nonfinite: {name}")
            if name == "mask":
                _require(array.shape == (40, 132), "raw mask evidence shape drift")
            elif name in {"v_hat", "x0", "x_t", "v_target", "x1", "x1_hat"}:
                _require(array.shape == RAW_SHAPE, f"raw flow evidence shape drift: {name}")
            path = raw_dir / f"{name}.npy"
            np.save(path, array, allow_pickle=False)
            _fsync_file(path)
            records[name] = {"path": str(path.relative_to(self.output_root)), "sha256": sha256_file(path), "array_sha256": sha256_bytes(np.ascontiguousarray(array).tobytes()), "shape": list(array.shape), "dtype": str(array.dtype)}
        manifest = {"schema": "CORELAB_STAGE45D_D1_RAW_PROBE_MANIFEST_V1", "identity": dict(identity), "raw": records, "derived": False}
        _write_json_durable(self.output_root / "probe_manifest.json", manifest)
        self.raw_written = True
        return manifest

    def write_derived(self, *, normalized_metrics: Mapping[str, Any], physical_metrics: Mapping[str, Any] | None = None) -> dict[str, Any]:
        _require(self.raw_written, "derived metrics cannot be written before raw evidence")
        value = {"schema": "CORELAB_STAGE45D_D1_DERIVED_METRICS_V1", "normalized": dict(normalized_metrics), "physical": None if physical_metrics is None else dict(physical_metrics), "raw_before_derived": True}
        _write_json_durable(self.output_root / "derived_metrics.json", value)
        return value

    def write_failure_seal(self, *, reason: str, counters: Mapping[str, int] | None = None) -> dict[str, Any]:
        value = {"schema": "CORELAB_STAGE45D_D1_FAILURE_SEAL_V1", "status": "FAIL", "reason": str(reason), "raw_before_derived": self.raw_written, "counters": dict(counters or {})}
        _write_json_durable(self.output_root / "FAILURE_SEAL.json", value)
        return value


def validate_evidence_lifecycle() -> dict[str, Any]:
    import tempfile

    with tempfile.TemporaryDirectory(prefix="stage45d_d1_harness_") as temporary:
        writer = ProbeEvidenceWriter(Path(temporary))
        try:
            writer.write_derived(normalized_metrics={"should": "fail"})
        except HarnessError:
            derived_before_raw_rejected = True
        else:
            derived_before_raw_rejected = False
        _require(derived_before_raw_rejected, "derived-before-raw ordering was accepted")
        noise = np.zeros(RAW_SHAPE, dtype=np.float32)
        raw = writer.write_raw(
            identity={
                "checkpoint": "FULL24@1000",
                "checkpoint_tree_sha256": FULL24_1000_TREE_SHA256,
                "stats_sha256": FULL24_STATS_SHA256,
                "window_id": "stage22_expert_red_target_train_a:expert-0000",
                "noise_seed": 2223,
                "t": T_VALUES[0],
                "t_bucket": 67,
            },
            arrays={
                "x0": noise,
                "x_t": noise,
                "v_target": noise,
                "v_hat": noise,
                "x1": noise,
                "x1_hat": noise,
                "mask": structured_action_mask("T16"),
            },
        )
        derived = writer.write_derived(normalized_metrics={"masked_rmse": 0.0})
        failure = writer.write_failure_seal(reason="synthetic static lifecycle check", counters={"model_forward": 0})
        _require(raw["derived"] is False and derived["raw_before_derived"] is True and failure["raw_before_derived"] is True, "evidence lifecycle flags drift")
    return {"raw_before_derived": True, "derived_before_raw_rejected": derived_before_raw_rejected, "failure_seal_written": True}


def validate_static_contract() -> dict[str, Any]:
    protocol = validate_reviewed_protocol()
    execution = validate_execution_manifest()
    binding = validate_runtime_binding_template()
    admission = validate_admission_manifest()
    compatibility = validate_compatibility_map()
    result = {
        "verdict": "PASS",
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "execution_manifest_sha256": sha256_file(EXECUTION_MANIFEST_PATH),
        "runner_sha256": sha256_file(Path(__file__)),
        "protocol_schema": protocol["schema"],
        "execution_manifest_schema": execution["schema"],
        "runtime_binding_schema": binding["schema"],
        "runtime_admission_schema": admission["schema"],
        "compatibility_map_schema": compatibility["schema"],
        "source_equivalence": validate_source_equivalence(),
        "mask_contract": validate_mask_contract(),
        "flow_algebra": validate_flow_algebra(),
        "checkpoint_stats_routing": validate_stats_routing(),
        "call_budget": validate_call_budget(),
        "evidence_lifecycle": validate_evidence_lifecycle(),
        "execution_boundary": {
            "model_construction": 0,
            "model_forward": 0,
            "gpu_h200": 0,
            "pbs_qsub": 0,
            "training": 0,
            "validation_e_read": 0,
            "holdout_f_read": 0,
            "wbc_steps": 0,
            "simulator_steps": 0,
            "hardware_steps": 0,
        },
    }
    return result


def _execute_requires_future_authority(binding: Mapping[str, Any]) -> None:
    _require(binding.get("execution_authorized") is True, "PI0107 Track S keeps D1 execution unauthorized")
    authority = os.environ.get("STAGE45D_D1_PI_AUTHORITY_TOKEN")
    _require(authority and authority == binding.get("pi_authority_token"), "future PI execution token is missing or mismatched")
    _require(os.environ.get("STAGE45D_D1_RUNTIME_IMPLEMENTATION") == "authorized", "future D1 runtime implementation gate is closed")


def execute(*, binding: Mapping[str, Any]) -> int:
    """Guarded future runtime entry point; static preparation never reaches imports."""

    _execute_requires_future_authority(binding)
    # Intentionally delayed until a later PI gate opens all three guards.
    # The actual runtime adapter must remain source-mapped to the accepted
    # Stage38/Stage45C implementation and is not loaded under PI0107 Track S.
    raise HarnessError("D1 runtime implementation is deferred to the next PI-authorized execution pass")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--plan-only", action="store_true", help="validate and print the frozen plan")
    modes.add_argument("--audit-only", action="store_true", help="validate and print the static admission audit")
    modes.add_argument("--execute", action="store_true", help="guarded future execution; never authorized by PI0107 Track S")
    parser.add_argument("--runtime-binding", type=Path, default=RUNTIME_BINDING_PATH)
    args = parser.parse_args()
    try:
        if args.execute:
            validate_reviewed_protocol()
            validate_execution_manifest()
            binding = validate_runtime_binding_template(args.runtime_binding)
            return execute(binding=binding)
        result = validate_static_contract()
    except (OSError, HarnessError, RuntimeAdmissionError, KeyError, TypeError, ValueError) as exc:
        print(json.dumps({"verdict": "FAIL", "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
