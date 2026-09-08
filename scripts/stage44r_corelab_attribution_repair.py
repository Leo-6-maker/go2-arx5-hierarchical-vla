#!/usr/bin/env python3
"""Stage44R zero-forward attribution repair.

This repair deliberately consumes only sealed Stage44/Stage43/Stage22
artifacts.  It does not construct a model, import a worker, touch IsaacGym,
run WBC, read validation/holdout data, or perform any new GR00T forward.

R1 audits expert-label target semantics in the common
``NATIVE_WBC_RELATIVE_REFERENCE`` frame.  R2 attributes each of the twenty
Stage43 references through three layers: decoded GR00T output, the installed
native reference, and achieved EEF.  R3 reuses the twenty-four existing
Stage44 decoded outputs and reports target-direction metrics, with
``red_target_train_a:expert-0000`` as the primary fit diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

import stage44_corelab_evidence as stage44


PI_GATE = "PI_GATE_STATE_0070"
BRANCH = "stage44r/corelab-attribution-repair"
STAGE44_PARENT = "f6c4759cdbd3be513d20d6b3934ebdbbdeedd5f4"
STAGE44_RESULT_SHA256 = "ed9790e62ab93c7292c1cd5257f5918f16241735847b6dbd71fa16ab4010f00d"
STAGE44_SEAL_SHA256 = "d62c38fe28cba75611558bf2c5a53cd327783691a355e2f2a3832b9b8cb57f95"
STAGE44_BUNDLE_SHA256 = "d836090b1ceb282e1bd6cea99955056509986ee351a6bf97fa1712e02a4148d6"
STAGE44_A3_SHA256 = "45828059332543389d97a6631ad3136600f03482393b10b19b722fc5ff939291"
STAGE44_RESULT_TOKEN = "CORELAB_STAGE44_TRAINING_LINEAGE_AND_BEHAVIOR_DIAGNOSIS_CHARACTERIZED"
PRIMARY_EXAMPLE_ID = "stage22_expert_red_target_train_a:expert-0000"
TARGET_TOLERANCE_M = 1.0e-9
ANCHOR_TOLERANCE_M = 1.0e-5
BOUND_TOLERANCE_M = 1.0e-5


class Stage44RRepairError(ValueError):
    """Raised when a zero-forward attribution repair is not admissible."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage44RRepairError(message)


def _read(path: Path, *, canonical: bool = True) -> Dict[str, Any]:
    return stage44.read_json(Path(path), canonical=canonical)


def _sha256_file(path: Path) -> str:
    return stage44.sha256_file(Path(path))


def _sha256_array(value: Any) -> str:
    return stage44.sha256_array(value)


def _finite(value: Any, shape: Tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    _require(tuple(array.shape) == shape and bool(np.isfinite(array).all()), "%s must be finite %s" % (label, shape))
    return array


def _stats(values: Iterable[float]) -> Dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    _require(array.size > 0 and bool(np.isfinite(array).all()), "metric sample is empty/nonfinite")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95.0)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _git_head(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN"


def _git_parent(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD^"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN"


def _trajectory_metrics(positions_world: Any, target_world: Any, *, prefix: str) -> Dict[str, Any]:
    positions = _finite(positions_world, (np.asarray(positions_world).shape[0], 3), prefix + ".positions_world")
    target = _finite(target_world, (3,), prefix + ".target_world")
    distances = np.linalg.norm(positions - target[None, :], axis=1)
    _require(bool(np.isfinite(distances).all()), prefix + " distances are nonfinite")
    displacement = positions[-1] - positions[0]
    target_vector = target - positions[0]
    displacement_norm = float(np.linalg.norm(displacement))
    target_norm = float(np.linalg.norm(target_vector))
    cosine = None if displacement_norm <= 1.0e-12 or target_norm <= 1.0e-12 else float(np.dot(displacement, target_vector) / (displacement_norm * target_norm))
    progress = float(distances[0] - distances[-1])
    best_progress = float(distances[0] - np.min(distances))
    sign = "POSITIVE" if progress > TARGET_TOLERANCE_M else ("ZERO" if abs(progress) <= TARGET_TOLERANCE_M else "NEGATIVE")
    return {
        "initial_target_distance_m": float(distances[0]),
        "minimum_target_distance_m": float(np.min(distances)),
        "minimum_target_index": int(np.argmin(distances)),
        "final_target_distance_m": float(distances[-1]),
        "absolute_progress_m": progress,
        "best_progress_m": best_progress,
        "progress_sign": sign,
        "endpoint_displacement_m": displacement.tolist(),
        "target_displacement_m": target_vector.tolist(),
        "endpoint_target_vector_cosine": cosine,
        "path_length_m": float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))),
        "sample_count": int(positions.shape[0]),
    }


def _snapshot_payload(row: Mapping[str, Any]) -> Dict[str, Any]:
    path = Path(str(row["snapshot_path"])).resolve(strict=True)
    payload = _read(path, canonical=False)
    _require(payload.get("example_id") == row["example_id"], "snapshot/example mismatch: %s" % row["example_id"])
    _require(_sha256_file(path) == row["snapshot_sha256"], "snapshot hash mismatch: %s" % row["example_id"])
    return payload


def _anchor_from_snapshot(payload: Mapping[str, Any], example_id: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    state = _finite(payload.get("state_v1", {}).get("values"), (10,), example_id + ".state_v1")
    achieved_world = _finite(payload.get("raw_telemetry", {}).get("achieved_position_world_m"), (3,), example_id + ".achieved_world")
    anchor = achieved_world - state[:3]
    target = _finite(payload.get("target_position_world_m"), (3,), example_id + ".target_world")
    _require(payload.get("source_frame") == stage44.SOURCE_FRAME, "expert source frame drift: %s" % example_id)
    return state, anchor, target


def audit_r1(a1: Mapping[str, Any]) -> Dict[str, Any]:
    """Recompute expert target semantics from world telemetry and local labels."""

    rows = list(a1.get("rows", []))
    _require(len(rows) == stage44.SNAPSHOT_COUNT, "A1 row count is not 24")
    output_rows: List[Dict[str, Any]] = []
    anchors: List[np.ndarray] = []
    target_progress: List[float] = []
    cosines: List[float] = []
    endpoint_errors: List[float] = []
    for row in rows:
        payload = _snapshot_payload(row)
        example_id = str(row["example_id"])
        state, anchor, target = _anchor_from_snapshot(payload, example_id)
        desired = _finite(payload.get("future_desired_eef_9d"), (stage44.SNAPSHOT_HORIZON, 9), example_id + ".expert_label")
        labels_world = desired[:, :3] + anchor[None, :]
        metrics = _trajectory_metrics(labels_world, target, prefix=example_id + ".expert")
        target_local = target - anchor
        endpoint_error = float(np.linalg.norm(desired[-1, :3] - target_local))
        initial_world_error = float(np.max(np.abs(labels_world[0] - np.asarray(payload["raw_telemetry"]["achieved_position_world_m"], dtype=np.float64))))
        _require(initial_world_error <= BOUND_TOLERANCE_M, "expert label initial world pose is not telemetry-bound: %s" % example_id)
        _require(np.max(np.abs(desired[0, :3] - state[:3])) <= BOUND_TOLERANCE_M, "expert label does not start at StateV1: %s" % example_id)
        _require(metrics["endpoint_target_vector_cosine"] is not None, "expert target cosine is undefined: %s" % example_id)
        anchors.append(anchor)
        target_progress.append(metrics["absolute_progress_m"])
        cosines.append(float(metrics["endpoint_target_vector_cosine"]))
        endpoint_errors.append(endpoint_error)
        output_rows.append({
            "example_id": example_id,
            "target_id": row["target_id"],
            "snapshot_path": str(Path(str(row["snapshot_path"])).resolve()),
            "snapshot_sha256": row["snapshot_sha256"],
            "anchor_world_m": anchor.tolist(),
            "target_world_m": target.tolist(),
            "target_local_m": target_local.tolist(),
            "state_v1_position_local_m": state[:3].tolist(),
            "target_world_metadata_only": True,
            "expert": metrics,
            "expert_endpoint_error_m": endpoint_error,
            "expert_label_sha256": row.get("label_sha256"),
            "source_frame": stage44.SOURCE_FRAME,
            "tcp_semantic_id": stage44.TCP_SEMANTIC_ID,
        })
    anchor_array = np.asarray(anchors, dtype=np.float64)
    _require(anchor_array.shape == (stage44.SNAPSHOT_COUNT, 3), "expert anchor count drift")
    anchor_spread = np.ptp(anchor_array, axis=0)
    _require(float(np.max(anchor_spread)) <= ANCHOR_TOLERANCE_M, "expert snapshots do not share a stable translation anchor")
    _require(all(value > TARGET_TOLERANCE_M for value in target_progress), "expert labels are not all target-progress positive")
    _require(all(value > 0.0 for value in cosines), "expert labels are not all target-direction positive")
    return {
        "schema": "CORELAB_STAGE44R_R1_EXPERT_LABEL_TARGET_SEMANTICS_V1",
        "stage": "Stage44R",
        "pi_gate": PI_GATE,
        "status": "PASS",
        "token": "CORELAB_STAGE44R_R1_EXPERT_LABEL_TARGET_SEMANTICS_PASS",
        "snapshot_count": len(output_rows),
        "target_count": len(stage44.TRAIN_TARGET_ORDER),
        "examples_per_target": a1["examples_per_target"],
        "anchor_world_m": np.mean(anchor_array, axis=0).tolist(),
        "anchor_spread_max_abs_m": float(np.max(anchor_spread)),
        "expert_progress_positive_count": int(sum(value > TARGET_TOLERANCE_M for value in target_progress)),
        "expert_cosine_positive_count": int(sum(value > 0.0 for value in cosines)),
        "expert_progress_m": _stats(target_progress),
        "expert_endpoint_error_m": _stats(endpoint_errors),
        "expert_endpoint_cosine": _stats(cosines),
        "all_expert_labels_target_progress_positive": True,
        "all_expert_labels_target_cosine_positive": True,
        "rows": output_rows,
        "forbidden_operations": stage44.forbidden_operations(),
        "claim_boundary": "expert label target semantics only; no learned-policy or task-success claim",
    }


def _load_a3(a3_root: Path, seal: Mapping[str, Any], bundle: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    manifest_path = Path(a3_root).resolve(strict=True) / "a3_train_forwards_manifest.json"
    _require(_sha256_file(manifest_path) == STAGE44_A3_SHA256, "Stage44 A3 manifest hash mismatch")
    manifest = _read(manifest_path)
    _require(manifest.get("status") == "PASS" and manifest.get("gr00t_forward_count_observed") == stage44.SNAPSHOT_COUNT, "Stage44 A3 is not the sealed 24-forward PASS")
    _require(manifest.get("bundle_sha256") == STAGE44_BUNDLE_SHA256 and manifest.get("bundle_sha256") == _sha256_file(Path(bundle["_path"])), "Stage44 A3 bundle binding mismatch")
    _require(manifest.get("model_input_keys") == list(stage44.MODEL_INPUT_KEYS) and manifest.get("target_xyz_in_model_input") is False, "Stage44 A3 model-input boundary drift")
    rows = manifest.get("rows", [])
    _require(len(rows) == stage44.SNAPSHOT_COUNT, "Stage44 A3 row count drift")
    return manifest, {str(row["example_id"]): dict(row) for row in rows}


def _load_a3_decoded(a3_row: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = Path(str(a3_row["artifact_path"])).resolve(strict=True)
    with np.load(path, allow_pickle=False) as data:
        _require(set(data.files) == {"raw_universal_action", "decoded_physical_action", "input_state", "body_rgb", "wrist_rgb"}, "A3 artifact key drift")
        raw = np.asarray(data["raw_universal_action"], dtype=np.float32)
        decoded = np.asarray(data["decoded_physical_action"], dtype=np.float64)
        state = np.asarray(data["input_state"], dtype=np.float32).reshape(-1)
        body = np.asarray(data["body_rgb"], dtype=np.uint8)
        wrist = np.asarray(data["wrist_rgb"], dtype=np.uint8)
    _require(raw.shape == (1, 40, 132) and np.isfinite(raw).all(), "A3 raw ABI/finiteness drift")
    _require(decoded.shape == (40, 10) and np.isfinite(decoded).all(), "A3 decoded ABI/finiteness drift")
    _require(state.shape == (10,) and np.isfinite(state).all(), "A3 state ABI/finiteness drift")
    _require(body.shape == (96, 96, 3) and wrist.shape == (96, 96, 3), "A3 image ABI drift")
    _require(_sha256_file(path) == a3_row["artifact_sha256"], "A3 artifact hash drift: %s" % a3_row["example_id"])
    _require(_sha256_array(raw) == a3_row["raw_action_sha256"] and _sha256_array(decoded) == a3_row["decoded_action_sha256"], "A3 action hash drift: %s" % a3_row["example_id"])
    _require(_sha256_array(state) == a3_row["state_v1_sha256"], "A3 state hash drift: %s" % a3_row["example_id"])
    _require(_sha256_array(body) == a3_row["body_sha256"] and _sha256_array(wrist) == a3_row["wrist_sha256"], "A3 image hash drift: %s" % a3_row["example_id"])
    return raw, decoded, state.astype(np.float64), body, wrist


def _comparison_metrics(decoded: np.ndarray, label_full: np.ndarray, anchor: np.ndarray, target: np.ndarray, *, prefix: str) -> Dict[str, Any]:
    _require(decoded.shape == (40, 10) and label_full.shape == (40, 10), prefix + " action shape drift")
    delta = decoded - label_full
    prediction_world = decoded[:, :3] + anchor[None, :]
    target_metrics = _trajectory_metrics(prediction_world, target, prefix=prefix + ".target")
    return {
        "target": target_metrics,
        "position_rmse_m": float(np.sqrt(np.mean(delta[:, :3] * delta[:, :3]))),
        "position_max_abs_m": float(np.max(np.abs(delta[:, :3]))),
        "endpoint_error_m": float(np.linalg.norm(decoded[-1, :3] - label_full[-1, :3])),
        "prediction_path_length_m": float(np.sum(np.linalg.norm(np.diff(decoded[:, :3], axis=0), axis=1))),
        "expert_path_length_m": float(np.sum(np.linalg.norm(np.diff(label_full[:, :3], axis=0), axis=1))),
        "eef_rot6d_rmse": float(np.sqrt(np.mean(delta[:, 3:9] * delta[:, 3:9]))),
        "eef_rot6d_max_abs": float(np.max(np.abs(delta[:, 3:9]))),
        "gripper_rmse": float(np.sqrt(np.mean(delta[:, 9:10] * delta[:, 9:10]))),
        "gripper_max_abs": float(np.max(np.abs(delta[:, 9:10]))),
        "full_action_rmse_mixed_units": float(np.sqrt(np.mean(delta * delta))),
    }


def audit_r3(a1: Mapping[str, Any], r1: Mapping[str, Any], a3_manifest: Mapping[str, Any], a3_rows: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """Reanalyze existing Stage44 outputs; this function performs no forward."""

    r1_rows = {str(row["example_id"]): row for row in r1["rows"]}
    output_rows: List[Dict[str, Any]] = []
    for example_id in stage44._expected_snapshot_ids():
        _require(example_id in r1_rows and example_id in a3_rows, "R3 missing sealed example: %s" % example_id)
        a3_row = a3_rows[example_id]
        _require(a3_row.get("status") == "PASS" and a3_row.get("forward_count_delta_observed") == 1, "R3 A3 row is not PASS: %s" % example_id)
        _, decoded, state, _, _ = _load_a3_decoded(a3_row)
        label_row = r1_rows[example_id]
        payload = _snapshot_payload(label_row)
        label = _finite(payload.get("future_desired_eef_9d"), (40, 9), example_id + ".label")
        gripper = _finite(payload.get("future_gripper"), (40, 1), example_id + ".gripper")
        label_full = np.concatenate([label, gripper], axis=1)
        anchor = _finite(label_row["anchor_world_m"], (3,), example_id + ".anchor")
        target = _finite(label_row["target_world_m"], (3,), example_id + ".target")
        _require(np.max(np.abs(decoded[0, :3] - state[:3])) <= BOUND_TOLERANCE_M, "R3 decoded action does not start at input StateV1: %s" % example_id)
        metrics = _comparison_metrics(decoded, label_full, anchor, target, prefix=example_id + ".prediction")
        output_rows.append({
            "example_id": example_id,
            "target_id": label_row["target_id"],
            "primary_fit_memorization_diagnostic": example_id == PRIMARY_EXAMPLE_ID,
            "a3_artifact_path": str(Path(str(a3_row["artifact_path"])).resolve()),
            "raw_action_sha256": a3_row["raw_action_sha256"],
            "decoded_action_sha256": a3_row["decoded_action_sha256"],
            "target_world_m": target.tolist(),
            "anchor_world_m": anchor.tolist(),
            "model_input_keys": list(stage44.MODEL_INPUT_KEYS),
            "target_xyz_in_model_input": False,
            "prediction": metrics,
            "expert": label_row["expert"],
            "finite": True,
        })
    _require(output_rows[0]["example_id"] == PRIMARY_EXAMPLE_ID, "R3 primary row is not train_a/expert-0000")
    prediction_progress = [row["prediction"]["target"]["absolute_progress_m"] for row in output_rows]
    prediction_cosines = [row["prediction"]["target"]["endpoint_target_vector_cosine"] for row in output_rows]
    return {
        "schema": "CORELAB_STAGE44R_R3_EXISTING_A3_TARGET_DIRECTION_V1",
        "stage": "Stage44R",
        "pi_gate": PI_GATE,
        "status": "PASS",
        "token": "CORELAB_STAGE44R_R3_EXISTING_A3_TARGET_DIRECTION_PASS",
        "new_gr00t_forwards": 0,
        "rows": output_rows,
        "primary_example_id": PRIMARY_EXAMPLE_ID,
        "primary_prediction": output_rows[0]["prediction"],
        "prediction_progress_m": _stats(prediction_progress),
        "prediction_endpoint_cosine": _stats([float(value) for value in prediction_cosines if value is not None]),
        "prediction_progress_positive_count": int(sum(value > TARGET_TOLERANCE_M for value in prediction_progress)),
        "prediction_cosine_positive_count": int(sum(value is not None and value > 0.0 for value in prediction_cosines)),
        "output_diversity": {
            "unique_raw_actions": len({row["raw_action_sha256"] for row in output_rows}),
            "unique_decoded_actions": len({row["decoded_action_sha256"] for row in output_rows}),
        },
        "forbidden_operations": stage44.forbidden_operations(),
        "claim_boundary": "existing 24 TRAIN output target-direction characterization; no new forward and no task-success claim",
    }


def _stage43_layer_segment(condition_root: Path, index: int) -> Tuple[List[Mapping[str, Any]], Mapping[str, Any]]:
    before = _read(condition_root / "rows_before_handoff.json")["segments"]
    after = _read(condition_root / "rows_after_handoff.json")["segments"]
    _require(len(before) == 3 and len(after) == 1 and all(len(segment) == 49 for segment in before) and len(after[0]) == 98, "Stage43 native segment schedule drift")
    if index < 3:
        return list(before[index]), {"source": "rows_before_handoff", "segment_index": index}
    return list(after[0]), {"source": "rows_after_handoff", "segment_index": 0}


def audit_r2(stage43_root: Path, stage43: Mapping[str, Any], a2: Mapping[str, Any]) -> Dict[str, Any]:
    """Attribute each Stage43 q reference through model/binding/EEF layers."""

    output_conditions: List[Dict[str, Any]] = []
    all_layers: Dict[str, List[float]] = {"gr00t_decoded": [], "installed_reference": [], "achieved_eef": []}
    negative_first_layers: List[str] = []
    for condition in stage43["result"].get("conditions", []):
        condition_id = str(condition["condition_id"])
        croot = Path(stage43_root).resolve(strict=True) / "conditions" / condition_id
        state_doc = _read(croot / "state_v1.json")
        worker_doc = _read(croot / "worker_response.json")
        bind_doc = _read(croot / "initial_bind.json")
        handoff_doc = _read(croot / "handoff.json")
        setup_doc = _read(croot / "condition_setup.json")
        _require(len(state_doc["observations"]) == 4 and len(worker_doc["responses"]) == 4, "Stage43 q count drift: %s" % condition_id)
        _require(len(bind_doc["events"]) == 1 and len(handoff_doc["events"]) == 3, "Stage43 bind/update count drift: %s" % condition_id)
        bind_event = bind_doc["events"][0]
        anchor = _finite(bind_event.get("anchor_world_m"), (3,), condition_id + ".anchor")
        target = _finite(condition.get("target_world_m"), (3,), condition_id + ".target")
        _require(np.allclose(target, np.asarray(stage44.TARGET_POSITION_WORLD_M, dtype=np.float64), rtol=0.0, atol=1e-9), "Stage43 target metadata drift: %s" % condition_id)
        with np.load(croot / "raw_action_d132.npz", allow_pickle=False) as raw_doc, np.load(croot / "decoded_action_d10_h40.npz", allow_pickle=False) as decoded_doc:
            raw_keys = set(raw_doc.files); decoded_keys = set(decoded_doc.files)
            expected_keys = {"%s-q%d" % (condition_id, i) for i in range(4)}
            _require(raw_keys == expected_keys and decoded_keys == expected_keys, "Stage43 action archive key drift: %s" % condition_id)
            qrows: List[Dict[str, Any]] = []
            previous_achieved: np.ndarray | None = None
            for index in range(4):
                qid = "%s-q%d" % (condition_id, index)
                observation = state_doc["observations"][index]
                response = worker_doc["responses"][index]
                _require(observation.get("request_id") == qid and response.get("request_id") == qid, "Stage43 q ordering drift: %s" % qid)
                _require(observation.get("request", {}).get("source_frame") == stage44.SOURCE_FRAME and observation.get("request", {}).get("target_position_world_m") == target.tolist(), "Stage43 q metadata drift: %s" % qid)
                raw = np.asarray(raw_doc[qid], dtype=np.float32)
                decoded = np.asarray(decoded_doc[qid], dtype=np.float64)
                _require(raw.shape == (1, 40, 132) and decoded.shape == (40, 10) and np.isfinite(raw).all() and np.isfinite(decoded).all(), "Stage43 q ABI/finiteness drift: %s" % qid)
                _require(_sha256_array(raw) == observation.get("raw_sha256") == response.get("raw_sha256"), "Stage43 q raw hash attribution drift: %s" % qid)
                _require(_sha256_array(decoded) == observation.get("decoded_sha256"), "Stage43 q decoded hash attribution drift: %s" % qid)
                state = _finite(observation.get("state_v1"), (10,), qid + ".state")
                _require(np.max(np.abs(decoded[0, :3] - state[:3])) <= BOUND_TOLERANCE_M, "Stage43 q decoded action does not start at StateV1: %s" % qid)
                decoded_world = decoded[:, :3] + anchor[None, :]
                model_metrics = _trajectory_metrics(decoded_world, target, prefix=qid + ".gr00t_decoded")
                bind_source = bind_event["executor_initial"]["handoff"] if index == 0 else handoff_doc["events"][index - 1]["executor_update"]["handoff"]
                bound_first_local = _finite(bind_source.get("new_first_position"), (3,), qid + ".bound_first_local")
                _require(np.max(np.abs(bound_first_local - decoded[0, :3])) <= BOUND_TOLERANCE_M, "decoded-to-bound first point mismatch: %s" % qid)
                segment, segment_meta = _stage43_layer_segment(croot, index)
                installed_native_world = _finite([row.get("desired_pos") for row in segment], (len(segment), 3), qid + ".installed_native_world")
                installed_world = np.vstack([bound_first_local + anchor, installed_native_world])
                installed_metrics = _trajectory_metrics(installed_world, target, prefix=qid + ".installed_reference")
                if index == 0:
                    previous_achieved = _finite(setup_doc["event"]["s0"]["eef_world_m"], (3,), qid + ".s0")
                _require(previous_achieved is not None, "missing achieved segment start: %s" % qid)
                achieved_segment_world = _finite([row.get("achieved_pos") for row in segment], (len(segment), 3), qid + ".achieved_world")
                achieved_world = np.vstack([previous_achieved, achieved_segment_world])
                achieved_metrics = _trajectory_metrics(achieved_world, target, prefix=qid + ".achieved_eef")
                tracking = np.linalg.norm(installed_native_world - achieved_segment_world, axis=1)
                _require(bool(np.isfinite(tracking).all()), "nonfinite Stage43 tracking layer: %s" % qid)
                safety = observation.get("safety_gate", {})
                qrows.append({
                    "request_id": qid,
                    "q_index": index,
                    "safety_gate_decision": safety.get("decision"),
                    "reference_source": "GR00T_FORWARD_OUTPUT",
                    "raw_action_sha256": _sha256_array(raw),
                    "decoded_action_sha256": _sha256_array(decoded),
                    "decoded_reference_frame": stage44.SOURCE_FRAME,
                    "anchor_world_m": anchor.tolist(),
                    "target_world_m": target.tolist(),
                    "bound_first_position_local_m": bound_first_local.tolist(),
                    "native_segment": segment_meta,
                    "gr00t_decoded": model_metrics,
                    "installed_reference": installed_metrics,
                    "achieved_eef": achieved_metrics,
                    "installed_vs_achieved_tracking_rms_m": float(np.sqrt(np.mean(tracking * tracking))),
                    "installed_vs_achieved_tracking_max_m": float(np.max(tracking)),
                    "first_negative_layer": next((name for name, item in (("gr00t_decoded", model_metrics), ("installed_reference", installed_metrics), ("achieved_eef", achieved_metrics)) if item["progress_sign"] == "NEGATIVE"), None),
                    "target_xyz_in_model_forward": False,
                })
                for name, item in (("gr00t_decoded", model_metrics), ("installed_reference", installed_metrics), ("achieved_eef", achieved_metrics)):
                    all_layers[name].append(float(item["absolute_progress_m"]))
                negative_first_layers.append(str(qrows[-1]["first_negative_layer"]))
                previous_achieved = achieved_segment_world[-1]
        output_conditions.append({
            "condition_id": condition_id,
            "target_world_m": target.tolist(),
            "anchor_world_m": anchor.tolist(),
            "q_count": 4,
            "rows": qrows,
            "negative_q_count_by_layer": {name: int(sum(item["progress_sign"] == "NEGATIVE" for row in qrows for item in [row[name]])) for name in all_layers},
            "first_negative_layer_by_q": [row["first_negative_layer"] for row in qrows],
            "all_q_references_are_model_outputs": True,
            "expert_label_reference_count": 0,
        })
    _require(len(output_conditions) == 5 and len(negative_first_layers) == 20, "Stage43 condition/q total drift")
    return {
        "schema": "CORELAB_STAGE44R_R2_STAGE43_REFERENCE_ATTRIBUTION_V1",
        "stage": "Stage44R",
        "pi_gate": PI_GATE,
        "status": "PASS",
        "token": "CORELAB_STAGE44R_R2_STAGE43_REFERENCE_ATTRIBUTION_PASS",
        "condition_count": len(output_conditions),
        "reference_count": len(negative_first_layers),
        "new_gr00t_forwards": 0,
        "layer_order": ["gr00t_decoded", "installed_reference", "achieved_eef"],
        "target_progress_by_layer_m": {name: _stats(values) for name, values in all_layers.items()},
        "negative_progress_count_by_layer": {name: int(sum(value < -TARGET_TOLERANCE_M for value in values)) for name, values in all_layers.items()},
        "first_negative_layer_counts": {name: int(negative_first_layers.count(name)) for name in ("gr00t_decoded", "installed_reference", "achieved_eef", "None")},
        "conditions": output_conditions,
        "all_stage43_references_are_actual_gr00t_outputs": True,
        "forbidden_operations": stage44.forbidden_operations(),
        "claim_boundary": "twenty existing Stage43 reference/binding/achieved layers; no new forward and no task-success claim",
    }


def _decision(r1: Mapping[str, Any], r2: Mapping[str, Any], r3: Mapping[str, Any], stage44_result: Mapping[str, Any]) -> Dict[str, Any]:
    primary = r3["primary_prediction"]
    primary_target = primary["target"]
    expert_ok = bool(r1["all_expert_labels_target_progress_positive"] and r1["all_expert_labels_target_cosine_positive"])
    primary_model_positive = bool(primary_target["progress_sign"] == "POSITIVE" and (primary_target["endpoint_target_vector_cosine"] or 0.0) > 0.0)
    # R3's primary row is the TRAIN fit/memorization sample.  It has no WBC
    # rollout counterpart.  Binding/tracking faults must therefore be tested
    # only within the same Stage43 q across its three R2 layers; never compare
    # R3 train_a/expert-0000 directly with Stage43 C0/q0.
    def _directed(item: Mapping[str, Any]) -> bool:
        return bool(item["progress_sign"] == "POSITIVE" and (item["endpoint_target_vector_cosine"] or 0.0) > 0.0)

    r2_rows = [row for condition in r2["conditions"] for row in condition["rows"]]
    binding_mismatch_rows = [row for row in r2_rows if _directed(row["gr00t_decoded"]) and not _directed(row["installed_reference"])]
    tracking_mismatch_rows = [row for row in r2_rows if _directed(row["gr00t_decoded"]) and _directed(row["installed_reference"]) and not _directed(row["achieved_eef"])]
    stage43_q0 = r2["conditions"][0]["rows"][0]
    if not expert_ok:
        classification = "STAGE44R_EXPERT_LABEL_SEMANTICS_LIMITATION_SUPPORTED"
        recommendation = "FIX_EXPERT_DATASET_BEFORE_MODEL_DECISION"
        reason = "The expert labels themselves are not uniformly target-progress positive; do not expand data or fine-tune until label generation is repaired."
    elif binding_mismatch_rows:
        classification = "STAGE44_REFERENCE_BINDING_MISMATCH_SUPPORTED"
        recommendation = "REPAIR_REFERENCE_BINDING_BEFORE_TRAINING"
        reason = "At least one Stage43 q is target-directed at the decoded GR00T layer but not at the installed-reference layer; the conversion/binding layer is the first failure boundary."
    elif tracking_mismatch_rows:
        classification = "STAGE44_WBC_TRACKING_LIMITATION_SUPPORTED"
        recommendation = "REPAIR_WBC_TRACKING_BEFORE_TRAINING"
        reason = "At least one Stage43 q is target-directed through the installed-reference layer but not at achieved EEF; this repair does not authorize a training change."
    elif primary_model_positive:
        classification = "STAGE44_TINY1_SINGLE_SAMPLE_COVERAGE_LIMITATION_SUPPORTED"
        recommendation = "STAGE45_DATA_EXPANSION_DESIGN_AFTER_PI"
        reason = "The exact train_a/expert-0000 diagnostic is target-directed, expert labels are target-directed, and the broader train/Stage43 evidence contains directional failures; the limitation is consistent with coverage rather than a universal failure on the memorized sample."
    else:
        classification = "STAGE44_STRUCTURED_TRAINING_FIT_LIMITATION_SUPPORTED"
        recommendation = "TRAINING_FIT_DIAGNOSTIC_BEFORE_DATA_EXPANSION"
        reason = "The exact train_a/expert-0000 diagnostic is not target-directed despite being the repeatedly trained sample; investigate fit, denoising, normalization, and checkpoint exposure before collecting more data."
    return {
        "schema": "CORELAB_STAGE44R_CAUSAL_DECISION_CONTEXT_V1",
        "status": "PASS",
        "classification": classification,
        "recommendation": recommendation,
        "reason": reason,
        "expert_labels_target_directed": expert_ok,
        "primary_example_id": PRIMARY_EXAMPLE_ID,
        "primary_model_target_directed": primary_model_positive,
        "stage43_q0_gr00t_target_directed": _directed(stage43_q0["gr00t_decoded"]),
        "stage43_q0_installed_reference_target_directed": _directed(stage43_q0["installed_reference"]),
        "stage43_q0_achieved_eef_target_directed": _directed(stage43_q0["achieved_eef"]),
        "stage43_binding_mismatch_count": len(binding_mismatch_rows),
        "stage43_tracking_mismatch_count": len(tracking_mismatch_rows),
        "stage43_gr00t_negative_count": int(sum(not _directed(row["gr00t_decoded"]) for row in r2_rows)),
        "broader_prediction_positive_count": r3["prediction_progress_positive_count"],
        "broader_prediction_count": len(r3["rows"]),
        "stage43_all_actual_progress_negative": bool(stage44_result.get("decision_context", {}).get("stage43_all_actual_progress_negative", True)),
        "data_expansion_supported": classification == "STAGE44_TINY1_SINGLE_SAMPLE_COVERAGE_LIMITATION_SUPPORTED",
        "h200_fine_tuning_executed": False,
        "h200_fine_tuning_authorized": False,
        "success_threshold_selected": False,
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    repo = Path(args.repo).resolve(strict=True)
    stage44_root = Path(args.stage44_root).resolve(strict=True)
    stage43_root = Path(args.stage43_root).resolve(strict=True)
    output_root = Path(args.output_root).resolve()
    _require(not output_root.exists(), "Stage44R output root already exists: %s" % output_root)
    output_root.mkdir(parents=True)
    stage44_result_path = stage44_root / "stage44_result.json"
    stage44_seal_path = stage44_root / "stage44_result.seal.json"
    stage44_bundle_path = stage44_root / "stage44_execution_bundle.json"
    stage44_result = _read(stage44_result_path)
    stage44_seal = _read(stage44_seal_path)
    stage44_bundle = _read(stage44_bundle_path)
    stage44_bundle["_path"] = str(stage44_bundle_path)
    _require(_sha256_file(stage44_result_path) == STAGE44_RESULT_SHA256, "Stage44 result hash mismatch")
    _require(_sha256_file(stage44_seal_path) == STAGE44_SEAL_SHA256, "Stage44 seal hash mismatch")
    _require(_sha256_file(stage44_bundle_path) == STAGE44_BUNDLE_SHA256, "Stage44 bundle hash mismatch")
    _require(stage44_result.get("status") == "PASS" and stage44_result.get("token") == STAGE44_RESULT_TOKEN, "Stage44 result is not the accepted raw-evidence package")
    _require(stage44_seal.get("result_sha256") == STAGE44_RESULT_SHA256 and stage44_seal.get("bundle_sha256") == STAGE44_BUNDLE_SHA256 and stage44_seal.get("a3_sha256") == STAGE44_A3_SHA256, "Stage44 seal predecessor mismatch")
    _require(stage44_result.get("observed_counters") == {"backward_steps": 0, "gr00t_forward_count": 24, "optimizer_steps": 0, "reference_update_count": 0, "train_snapshots": 24, "training_steps": 0, "wbc_forward_count": 0}, "Stage44 counter contract drift")
    _require(_git_parent(repo) == STAGE44_PARENT, "Stage44R branch parent is not the accepted Stage44 commit")
    _require(_git_head(repo) == STAGE44_PARENT or _git_head(repo) != "UNKNOWN", "Stage44R Git head is unavailable")
    a0 = _read(stage44_root / "a0_training_lineage.json")
    a1 = _read(stage44_root / "a1_expert_label_audit.json")
    a2 = _read(stage44_root / "a2_stage43_reference_attribution.json")
    _require(_sha256_file(stage44_root / "a0_training_lineage.json") == stage44_seal["a0_sha256"], "A0 hash mismatch")
    _require(_sha256_file(stage44_root / "a1_expert_label_audit.json") == stage44_seal["a1_sha256"], "A1 hash mismatch")
    _require(_sha256_file(stage44_root / "a2_stage43_reference_attribution.json") == stage44_seal["a2_sha256"], "A2 hash mismatch")
    _require(a2.get("expert_label_reference_count") == 0 and a2.get("model_reference_count") == 20, "Stage44 A2 attribution contract drift")
    a3_manifest, a3_rows = _load_a3(Path(args.a3_root), stage44_seal, stage44_bundle)
    r1 = audit_r1(a1)
    r2 = audit_r2(stage43_root, {"root": str(stage43_root), "result": _read(stage43_root / "stage43_result.json"), "bundle": _read(stage43_root / "stage43_execution_bundle.json"), "result_sha256": stage44.STAGE43_RESULT_SHA256, "bundle_sha256": stage44.STAGE43_BUNDLE_SHA256, "seal_sha256": stage44.STAGE43_SEAL_SHA256}, a2)
    r3 = audit_r3(a1, r1, a3_manifest, a3_rows)
    stage43_actual_negative = all(item.get("progress_sign") == "NEGATIVE" for item in _read(stage43_root / "stage43_result.json").get("conditions", []))
    stage44_result["decision_context"] = {"stage43_all_actual_progress_negative": stage43_actual_negative}
    decision = _decision(r1, r2, r3, stage44_result)
    result = {
        "schema": "CORELAB_STAGE44R_ATTRIBUTION_ANALYSIS_REPAIR_RESULT_V1",
        "stage": "Stage44R",
        "pi_gate": PI_GATE,
        "status": "PASS",
        "token": "CORELAB_STAGE44R_ATTRIBUTION_ANALYSIS_REPAIRED",
        "return_token": "CORELAB_STAGE44R_ATTRIBUTION_ANALYSIS_REPAIRED",
        "repository": {"branch": BRANCH, "head": _git_head(repo), "parent_stage44_commit": STAGE44_PARENT},
        "source_stage44": {"result_sha256": STAGE44_RESULT_SHA256, "seal_sha256": STAGE44_SEAL_SHA256, "bundle_sha256": STAGE44_BUNDLE_SHA256, "a3_sha256": STAGE44_A3_SHA256},
        "observed_counters": {"new_gr00t_forward_count": 0, "new_wbc_step_count": 0, "new_reference_update_count": 0, "new_training_steps": 0, "new_backward_steps": 0, "new_optimizer_steps": 0},
        "stages": {"R1": {"status": r1["status"], "token": r1["token"]}, "R2": {"status": r2["status"], "token": r2["token"]}, "R3": {"status": r3["status"], "token": r3["token"]}},
        "r1_expert_label_semantics": r1,
        "r2_stage43_reference_attribution": r2,
        "r3_existing_a3_target_direction": r3,
        "decision_context": decision,
        "stop_gate": {"status": "STOP_PI", "next_action": "PI_REVIEW_STAGE44R_DECISION", "do_not_start": ["new GR00T forward", "WBC", "training", "fine-tuning", "A40/Vanda", "H200", "validation/final-test reads", "success-threshold selection"]},
        "forbidden_operations": stage44.forbidden_operations(),
        "claim_boundary": "Stage44 raw evidence retained; zero-forward attribution repair only; no generalization, task-success, or fine-tuning result",
    }
    result_path = output_root / "stage44r_result.json"
    stage44.write_json(result_path, result)
    seal = {
        "schema": "CORELAB_STAGE44R_RESULT_SEAL_V1",
        "stage": "Stage44R",
        "status": "PASS",
        "result_sha256": _sha256_file(result_path),
        "source_stage44_result_sha256": STAGE44_RESULT_SHA256,
        "source_stage44_seal_sha256": STAGE44_SEAL_SHA256,
        "source_stage44_bundle_sha256": STAGE44_BUNDLE_SHA256,
        "source_stage44_a3_sha256": STAGE44_A3_SHA256,
        "r1_sha256": stage44.write_json(output_root / "r1_expert_label_target_semantics.json", r1),
        "r2_sha256": stage44.write_json(output_root / "r2_stage43_reference_attribution.json", r2),
        "r3_sha256": stage44.write_json(output_root / "r3_existing_a3_target_direction.json", r3),
        "repository_head": _git_head(repo),
        "new_gr00t_forward_count": 0,
        "new_wbc_step_count": 0,
        "new_training_steps": 0,
        "forbidden_operations": stage44.forbidden_operations(),
    }
    stage44.write_json(output_root / "stage44r_result.seal.json", seal)
    report = _render_report(result, r1, r2, r3, decision, seal)
    (output_root / "STAGE44R_CORELAB_ATTRIBUTION_REPAIR.md").write_text(report, encoding="utf-8")
    return result


def _render_report(result: Mapping[str, Any], r1: Mapping[str, Any], r2: Mapping[str, Any], r3: Mapping[str, Any], decision: Mapping[str, Any], seal: Mapping[str, Any]) -> str:
    primary = r3["primary_prediction"]["target"]
    return "\n".join([
        "# Stage44R Corelab zero-forward attribution repair",
        "",
        "Status: `%s`" % result["token"],
        "",
        "No new GR00T forward, WBC step, training step, validation/holdout read, A40 wait, or H200 operation was performed.",
        "",
        "## Source package",
        "",
        "- Stage44 result `%s` and seal `%s` were revalidated; A3 raw/decode evidence is retained." % (result["source_stage44"]["result_sha256"], result["source_stage44"]["seal_sha256"]),
        "- Stage44R branch `%s` is based on Stage44 commit `%s`." % (result["repository"]["branch"], result["repository"]["parent_stage44_commit"]),
        "",
        "## R1 expert semantics",
        "",
        "- %d/24 expert labels have positive target progress and %d/24 have positive endpoint target cosine." % (r1["expert_progress_positive_count"], r1["expert_cosine_positive_count"]),
        "- The inferred translation anchor spread is %.3g m; labels are evaluated in world coordinates after adding the sealed local-frame anchor." % r1["anchor_spread_max_abs_m"],
        "",
        "## R2 Stage43 layer attribution",
        "",
        "- Twenty existing references are compared as decoded GR00T → installed native reference → achieved EEF; no expert label is substituted." ,
        "- Negative-progress counts: GR00T decoded `%d`, installed reference `%d`, achieved EEF `%d`." % tuple(r2["negative_progress_count_by_layer"][name] for name in ("gr00t_decoded", "installed_reference", "achieved_eef")),
        "",
        "## R3 existing Stage44 outputs",
        "",
        "- Primary `%s`: predicted progress `%.6f m`, endpoint target cosine `%.6f`, endpoint error `%.6f m`." % (PRIMARY_EXAMPLE_ID, primary["absolute_progress_m"], primary["endpoint_target_vector_cosine"], r3["rows"][0]["prediction"]["endpoint_error_m"]),
        "- Across 24 outputs: `%d/24` positive target progress; raw/decode output diversity remains `%s/%s` unique." % (r3["prediction_progress_positive_count"], r3["output_diversity"]["unique_raw_actions"], r3["output_diversity"]["unique_decoded_actions"]),
        "",
        "## Decision context",
        "",
        "- Classification: `%s`." % decision["classification"],
        "- Recommendation: `%s`." % decision["recommendation"],
        "- This is a bounded attribution result, not a task-success or generalization claim; H200 fine-tuning remains unauthorized.",
        "",
        "Seal: `%s`." % seal["result_sha256"],
        "",
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--stage44-root", type=Path, required=True)
    parser.add_argument("--stage43-root", type=Path, required=True)
    parser.add_argument("--a3-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({"stage": "Stage44R", "status": result["status"], "token": result["token"], "classification": result["decision_context"]["classification"], "result_root": str(Path(args.output_root).resolve()), "new_gr00t_forward_count": 0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
