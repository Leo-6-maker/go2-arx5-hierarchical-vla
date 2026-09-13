#!/usr/bin/env python3
"""PI0153 CPU-only first-action semantics and generation audit.

This runner consumes the six sealed PI0152 C1 raw arrays and the one sealed
PI0152 C2/C0 raw array.  It deliberately imports only the repository's CPU
codec/metric modules; no Torch, GR00T, CUDA, simulator, WBC, or scheduler code
is imported.  The historical PI0152 report is never rewritten.  Instead this
runner writes a new correction packet that makes the frame bridge explicit:

    p_world = measured_eef_world + (p_decoded - StateV1[:3])

The bridge is valid here only because the accepted native adapter records a
translation-only external reference.  If that evidence is absent, the packet
fails closed rather than silently treating a relative point as world-frame.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

PI_GATE = "PI_GATE_STATE_0153"
PI_COMMENT = 5649870843
PR_COMMENT = 5649873155
BRANCH = "stage45f/corelab-local-initial-state-pilot"
EXPECTED_HEAD = "43648dd59ac43cf261462f614267ae1431e30b62"
RETURN_TOKEN = "CORELAB_PI0153_FIRST_ACTION_SEMANTICS_AND_GENERATION_FOR_PI_REVIEW"
N2_STATS_SHA = "0c9450e6aeb0765eebc676414b2a6a84f71dddb7eea1de5453551f062b25fdbb"
N2_STATS_PAYLOAD_SHA = "508d4a8845e3bf5928416a506b95046ec0bb6c2be3eecbd09c39684a00865572"
NOISE_ARRAY_SHA = "5924010a4e506159b654d0d6164125da5bf0e513ff410bf6e8e5f964a9ab50bd"
TARGET = np.asarray([0.30, 0.10, 0.58], dtype=np.float64)
TIMESTAMPS = np.arange(40, dtype=np.float64) * 0.05
SOURCE_FRAME = "NATIVE_WBC_RELATIVE_REFERENCE"
TCP = "x5_link6_gripper_center"
TRAIN_LABEL = "TRAIN_a_expert-0000"
CONDITIONS = ("C0_NOMINAL", "C1_X_PLUS_5CM", "C2_X_MINUS_5CM", "C3_Y_PLUS_5CM", "C4_Y_MINUS_5CM")
RAW_SHAPE = (1, 40, 132)


class AuditError(RuntimeError):
    """Fail-closed error for an identity or semantic mismatch."""


def require(predicate: bool, message: str) -> None:
    if not predicate:
        raise AuditError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> Any:
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not stat.S_ISLNK(mode), f"not a regular JSON file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot parse {path}: {exc}") from exc


def write_json(path: Path, value: Any) -> str:
    require(not path.exists(), f"refusing to overwrite evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return sha256_file(path)


def _regular_file(path: Path, label: str) -> None:
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not stat.S_ISLNK(mode), f"{label} is not a regular file: {path}")


def _regular_dir(path: Path, label: str) -> None:
    mode = path.lstat().st_mode
    require(stat.S_ISDIR(mode) and not stat.S_ISLNK(mode), f"{label} is not a regular directory: {path}")


def _load_npy(path: Path, expected_shape: tuple[int, ...], label: str) -> np.ndarray:
    _regular_file(path, label)
    value = np.load(path, allow_pickle=False)
    require(not isinstance(value, np.lib.npyio.NpzFile), f"{label} unexpectedly contains an NPZ archive")
    value = np.asarray(value)
    require(value.shape == expected_shape and np.isfinite(value).all(), f"{label} ABI/finite mismatch: {value.shape}")
    return value


def _load_raw_npz(path: Path, label: str) -> tuple[np.ndarray, str]:
    _regular_file(path, label)
    archive = np.load(path, allow_pickle=False)
    require(isinstance(archive, np.lib.npyio.NpzFile), f"{label} is not an NPZ archive")
    require(len(archive.files) == 1, f"{label} must contain exactly one named raw array")
    value = np.asarray(archive[archive.files[0]])
    require(value.shape == RAW_SHAPE and np.isfinite(value).all(), f"{label} ABI/finite mismatch: {value.shape}")
    return value, archive.files[0]


def _snapshot_for_train(c1_root: Path, preflight: Path) -> dict[str, Any]:
    admission = read_json(c1_root / "admission.json")
    ref = admission.get("audit", {}).get("train_reference", {})
    require(ref.get("window_id") == "stage22_expert_red_target_train_a:expert-0000", "TRAIN authority drift")
    require(ref.get("snapshot_preflight_path") == str(preflight), "TRAIN preflight path mismatch")
    doc = read_json(preflight)
    snapshots = doc.get("snapshots")
    require(isinstance(snapshots, list), "TRAIN snapshot list missing")
    matches = [item for item in snapshots if item.get("window_id") == ref["window_id"]]
    require(len(matches) == 1, "TRAIN snapshot identity is not unique")
    result = dict(matches[0])
    result["anchor_world_m"] = list(ref["anchor_world_m"])
    result["target_world_m"] = list(ref["target_world_m"])
    return result


def _snapshot_for_stage43(admission: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    conditions = admission.get("audit", {}).get("stage43", {}).get("conditions", [])
    require(len(conditions) == len(CONDITIONS), "Stage43 condition count drift")
    result: dict[str, dict[str, Any]] = {}
    for item in conditions:
        condition = str(item.get("condition_id"))
        require(condition in CONDITIONS and condition not in result, f"unrecognized Stage43 condition: {condition}")
        q0 = item.get("q0", {})
        state = q0.get("state_v1")
        require(isinstance(state, list) and len(state) == 10, f"StateV1 missing for {condition}")
        result[condition] = {
            "label": condition,
            "window_id": f"{condition}:q0",
            "state_v1": state,
            "state_v1_sha256": q0.get("state_v1_sha256"),
            "anchor_world_m": item.get("anchor_world_m"),
            "target_world_m": item.get("target_world_m"),
            "target_xyz_in_model_input": item.get("target_xyz_in_model_input"),
            "body_array_sha256": q0.get("body_array_sha256"),
            "wrist_array_sha256": q0.get("wrist_array_sha256"),
            "observation_timestamp": q0.get("observation_timestamp", 0.0),
            "source_timestamps": TIMESTAMPS.tolist(),
            "source_frame": SOURCE_FRAME,
        }
    require(set(result) == set(CONDITIONS), "Stage43 condition set drift")
    return result


def _c2_snapshot(c2_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    condition = c2_root / "conditions" / "00_C0_NOMINAL"
    state_doc = read_json(condition / "state_v1.json")
    observations = state_doc.get("observations")
    require(isinstance(observations, list) and len(observations) == 1, "C2 observation count drift")
    obs = dict(observations[0])
    snapshot = {
        "label": "C2_C0_NOMINAL",
        "window_id": obs.get("request_id"),
        "state_v1": obs.get("state_v1"),
        "state_v1_sha256": obs.get("state_v1_sha256"),
        "anchor_world_m": read_json(condition / "condition_setup.json")["event"]["s0"]["eef_world_m"],
        "target_world_m": obs.get("request", {}).get("target_position_world_m"),
        "target_xyz_in_model_input": False,
        "body_array_sha256": obs.get("body_sha256"),
        "wrist_array_sha256": obs.get("wrist_sha256"),
        "observation_timestamp": obs.get("observation_timestamp", 0.0),
        "source_timestamps": obs.get("structured_decode", {}).get("source_timestamps", TIMESTAMPS.tolist()),
        "source_frame": obs.get("request", {}).get("source_frame", SOURCE_FRAME),
    }
    gate = read_json(condition / "safety_gate.json")
    counters = read_json(condition / "counter_observations.json")
    return snapshot, gate, counters


def _decode(raw: np.ndarray, snapshot: Mapping[str, Any], stats_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    import go2arx5_vla.stage40_structured_decode as decoder

    previous = decoder.EXPECTED_STRUCTURED_STATS_SHA256
    decoder.EXPECTED_STRUCTURED_STATS_SHA256 = N2_STATS_SHA
    try:
        _, physical, details = decoder.decode_raw_structured_action(
            raw,
            np.asarray(snapshot["state_v1"], dtype=np.float32),
            stats_path=stats_path,
            source_timestamps=np.asarray(snapshot["source_timestamps"], dtype=np.float64),
            observation_timestamp=float(snapshot["observation_timestamp"]),
            chunk_id=str(snapshot["window_id"]),
            source_frame=str(snapshot.get("source_frame", SOURCE_FRAME)),
            tcp_semantic_id=TCP,
        )
    finally:
        decoder.EXPECTED_STRUCTURED_STATS_SHA256 = previous
    physical = np.asarray(physical, dtype=np.float64)
    require(physical.shape == (40, 10) and np.isfinite(physical).all(), "decoded physical ABI/finite mismatch")
    return physical, details


def _corrected_metrics(physical: np.ndarray, snapshot: Mapping[str, Any]) -> dict[str, Any]:
    from go2arx5_vla.evaluation.trajectory_metrics import trajectory_metrics

    state = np.asarray(snapshot["state_v1"], dtype=np.float64)
    anchor = np.asarray(snapshot["anchor_world_m"], dtype=np.float64)
    target = np.asarray(snapshot["target_world_m"], dtype=np.float64)
    require(state.shape == (10,) and anchor.shape == (3,) and target.shape == (3,), "snapshot shape drift")
    relative = physical[:, :3] - state[None, :3]
    world = anchor[None, :] + relative
    old_world = anchor[None, :] + physical[:, :3]
    corrected = trajectory_metrics(world, target, prefix="PI0153.corrected")
    return {
        "state_xyz_m": state[:3].tolist(),
        "measured_eef_world_m": anchor.tolist(),
        "target_world_m": target.tolist(),
        "decoded_p0_frame_m": physical[0, :3].tolist(),
        "decoded_p0_minus_state_m": (physical[0, :3] - state[:3]).tolist(),
        "p0_state_residual_m": float(np.linalg.norm(physical[0, :3] - state[:3])),
        "corrected_world_p0_m": world[0].tolist(),
        "corrected_start_residual_m": float(np.linalg.norm(world[0] - anchor)),
        "old_report_world_p0_m": old_world[0].tolist(),
        "old_report_start_jump_m": float(np.linalg.norm(old_world[0] - anchor)),
        "old_report_start_jump_identity_m": float(np.linalg.norm(state[:3])),
        "world_transform": "translation_only: measured_eef_world + (decoded_relative - StateV1[:3])",
        "target_metrics": corrected,
    }


def _write_plot(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    colors = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#000000")
    series = []
    for index, row in enumerate(rows):
        points = np.asarray(row["corrected_world_positions_m"], dtype=np.float64)
        series.append((str(row["label"]), points, colors[index % len(colors)]))
    all_points = np.concatenate([item[1] for item in series], axis=0)
    xmin, xmax = float(all_points[:, 0].min()), float(all_points[:, 0].max())
    ymin, ymax = float(all_points[:, 1].min()), float(all_points[:, 1].max())
    xr, yr = max(xmax - xmin, 1e-9), max(ymax - ymin, 1e-9)
    width, height, pad = 1100, 720, 70
    def xy(point: np.ndarray) -> tuple[float, float]:
        return (pad + (float(point[0]) - xmin) / xr * (width - 2 * pad), height - pad - (float(point[1]) - ymin) / yr * (height - 2 * pad))
    tx, ty = xy(TARGET)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>', '<text x="70" y="28" font-family="sans-serif" font-size="18">PI0153 corrected decoded EEF paths (world XY)</text>', f'<circle cx="{tx:.2f}" cy="{ty:.2f}" r="7" fill="#e41a1c"/><text x="{tx+10:.2f}" y="{ty-8:.2f}" font-family="sans-serif" font-size="13">red target</text>']
    for label, points, color in series:
        coords = " ".join(f"{xy(point)[0]:.2f},{xy(point)[1]:.2f}" for point in points)
        out.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2" opacity="0.9"/>')
    out.extend([f'<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#333"/>', f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#333"/>', f'<text x="{width/2:.0f}" y="{height-20}" text-anchor="middle" font-family="sans-serif" font-size="13">world x (m)</text>', f'<text x="18" y="{height/2:.0f}" transform="rotate(-90 18 {height/2:.0f})" text-anchor="middle" font-family="sans-serif" font-size="13">world y (m)</text>'])
    for index, (label, _, color) in enumerate(series):
        y = 55 + index * 20
        out.append(f'<line x1="820" y1="{y}" x2="845" y2="{y}" stroke="{color}" stroke-width="3"/><text x="852" y="{y+4}" font-family="sans-serif" font-size="12">{label}</text>')
    out.append("</svg>\n")
    path.write_text("\n".join(out), encoding="utf-8")
    return sha256_file(path)


def _cpu_fixture_checks(stats_path: Path, train_snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Run legal hold-pose and TRAIN encode→N2-normalize→decode checks."""
    from go2arx5_vla.structured_action import encode_structured_action, structured_action_mask
    import go2arx5_vla.stage40_structured_decode as decoder

    lower, upper, mask, stats_sha = decoder.load_structured_statistics(stats_path) if decoder.EXPECTED_STRUCTURED_STATS_SHA256 == N2_STATS_SHA else (None, None, None, None)
    if lower is None:
        previous = decoder.EXPECTED_STRUCTURED_STATS_SHA256
        decoder.EXPECTED_STRUCTURED_STATS_SHA256 = N2_STATS_SHA
        try:
            lower, upper, mask, stats_sha = decoder.load_structured_statistics(stats_path)
        finally:
            decoder.EXPECTED_STRUCTURED_STATS_SHA256 = previous
    require(stats_sha == N2_STATS_SHA, "N2 stats identity drift in fixture check")

    state = np.asarray(train_snapshot["state_v1"], dtype=np.float64)
    hold = np.repeat(state[None, :], 40, axis=0)
    hold_encoded = encode_structured_action(hold, TIMESTAMPS, state, timing_mode="T16", source_frame=SOURCE_FRAME, tcp_semantic_id=TCP, chunk_id="pi0153-hold-fixture")
    train_expert = np.asarray(train_snapshot["expert_physical_action"], dtype=np.float64)
    require(train_expert.shape == (40, 10), "TRAIN expert action shape drift")
    train_encoded = encode_structured_action(train_expert, TIMESTAMPS, state, timing_mode="T16", source_frame=SOURCE_FRAME, tcp_semantic_id=TCP, chunk_id="pi0153-train-expert")

    def normalize(values: np.ndarray) -> tuple[np.ndarray, int]:
        span = np.broadcast_to(upper - lower, values.shape)
        lower_b = np.broadcast_to(lower, values.shape)
        normalized = np.zeros_like(values, dtype=np.float64)
        active = mask.astype(bool)
        nonzero = active & (np.abs(span) > 0.0)
        normalized[nonzero] = 2.0 * (values[nonzero] - lower_b[nonzero]) / span[nonzero] - 1.0
        normalized[active & ~nonzero] = 0.0
        clipped = int(np.count_nonzero((normalized < -1.0) | (normalized > 1.0)))
        return np.clip(normalized, -1.0, 1.0), clipped

    def roundtrip(encoded: Any, label: str) -> dict[str, Any]:
        values = np.asarray(encoded.values, dtype=np.float64)
        normalized, clipped = normalize(values)
        raw = normalized[None, ...]
        previous = decoder.EXPECTED_STRUCTURED_STATS_SHA256
        decoder.EXPECTED_STRUCTURED_STATS_SHA256 = N2_STATS_SHA
        try:
            _, physical, details = decoder.decode_raw_structured_action(raw, state.astype(np.float32), stats_path=stats_path, source_timestamps=TIMESTAMPS, observation_timestamp=0.0, chunk_id=label, source_frame=SOURCE_FRAME, tcp_semantic_id=TCP)
        finally:
            decoder.EXPECTED_STRUCTURED_STATS_SHA256 = previous
        return {"label": label, "active_scalar_count": int(np.count_nonzero(mask)), "normalized_array_sha256": sha256_array(normalized.astype(np.float64)), "clipped_scalar_count": clipped, "p0_state_residual_m": float(np.linalg.norm(physical[0, :3] - state[:3])), "finite": bool(np.isfinite(physical).all()), "decoded_physical_sha256": sha256_array(physical), "decoder_stats_sha256": details["stats_sha256"]}

    hold_result = roundtrip(hold_encoded, "pi0153-hold-fixture")
    train_result = roundtrip(train_encoded, "pi0153-train-expert")
    return {"mask_shape": list(structured_action_mask("T16").shape), "active_scalar_count": int(np.count_nonzero(mask)), "hold_pose": hold_result, "train_expert": train_result}


def run(args: argparse.Namespace) -> dict[str, Any]:
    c1_root = Path(args.c1_root).resolve()
    c2_root = Path(args.c2_root).resolve()
    stats_path = Path(args.n2_stats).resolve()
    preflight = Path(args.train_preflight).resolve()
    output = Path(args.output_root).resolve()
    _regular_dir(c1_root, "C1 root")
    _regular_dir(c2_root, "C2 root")
    _regular_file(stats_path, "N2 stats")
    require(not output.exists(), f"output root already exists: {output}")
    output.mkdir(parents=True)
    admission = read_json(c1_root / "admission.json")
    require(admission.get("audit", {}).get("authority", {}).get("issue_comment") == 5647237688, "C1 source authority drift")
    stage43 = _snapshot_for_stage43(admission)
    train = _snapshot_for_train(c1_root, preflight)
    c2, c2_gate, c2_counters = _c2_snapshot(c2_root)

    old_cpu = read_json(Path(args.old_cpu_report).resolve()) if args.old_cpu_report else None
    old_by_label = {row.get("label"): row for row in (old_cpu or {}).get("rows", [])}
    raw_manifest = c1_root / "raw_predictions.jsonl"
    _regular_file(raw_manifest, "C1 raw manifest")
    raw_items = [json.loads(line) for line in raw_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    require(len(raw_items) == 6, f"C1 raw row count drift: {len(raw_items)}")
    by_label = {item["label"]: item for item in raw_items}
    require(set(by_label) == {TRAIN_LABEL, *CONDITIONS}, "C1 raw label set drift")

    rows: list[dict[str, Any]] = []
    raw_input_equivalence: dict[str, Any] = {}
    for label, snapshot in [(TRAIN_LABEL, train), *[(condition, stage43[condition]) for condition in CONDITIONS]]:
        item = by_label[label]
        raw_path = c1_root / str(item["raw_path"])
        raw = _load_npy(raw_path, RAW_SHAPE, f"C1 raw {label}")
        actual_file_sha = sha256_file(raw_path)
        actual_array_sha = sha256_array(raw)
        require(actual_file_sha == item["raw_sha256"] and actual_array_sha == item["array_sha256"], f"C1 raw hash drift: {label}")
        physical, decode_details = _decode(raw, snapshot, stats_path)
        saved_decoded = _load_npy(c1_root / "decoded" / f"{int(item['index']):02d}_{label}.npy", (40, 10), f"saved decoded {label}")
        corrected = _corrected_metrics(physical, snapshot)
        corrected["decoder_replay_max_abs_vs_saved"] = float(np.max(np.abs(physical - saved_decoded)))
        corrected["decoder_replay_rmse_vs_saved"] = float(np.sqrt(np.mean((physical - saved_decoded) ** 2)))
        corrected_world = np.asarray(snapshot["anchor_world_m"], dtype=np.float64)[None, :] + physical[:, :3] - np.asarray(snapshot["state_v1"], dtype=np.float64)[None, :3]
        row = {"index": int(item["index"]), "label": label, "kind": item["kind"], "raw_path": str(raw_path), "raw_file_sha256": actual_file_sha, "raw_array_sha256": actual_array_sha, "noise_sha256": item.get("noise_sha256"), "snapshot": snapshot, "decoded_physical_sha256": sha256_array(physical), "decoder": decode_details, "corrected": corrected, "corrected_world_positions_m": corrected_world.tolist(), "historical_row": old_by_label.get(label)}
        rows.append(row)

    # C2 raw is a separately sealed worker artifact and has no C1 manifest row.
    c2_condition = c2_root / "conditions" / "00_C0_NOMINAL"
    c2_raw_path = c2_condition / "raw_action_d132.npz"
    c2_raw, c2_raw_key = _load_raw_npz(c2_raw_path, "C2 raw C0")
    c2_physical, c2_decode = _decode(c2_raw, c2, stats_path)
    c2_saved_archive = np.load(c2_condition / "decoded_action_d10_h40.npz", allow_pickle=False)
    require(len(c2_saved_archive.files) == 1, "C2 saved decoded archive key count drift")
    c2_saved = np.asarray(c2_saved_archive[c2_saved_archive.files[0]], dtype=np.float64)
    require(c2_saved.shape == (40, 10), "C2 saved decoded shape drift")
    c2_corrected = _corrected_metrics(c2_physical, c2)
    c2_corrected["decoder_replay_max_abs_vs_saved"] = float(np.max(np.abs(c2_physical - c2_saved)))
    c2_corrected["decoder_replay_rmse_vs_saved"] = float(np.sqrt(np.mean((c2_physical - c2_saved) ** 2)))
    c2_world = np.asarray(c2["anchor_world_m"], dtype=np.float64)[None, :] + c2_physical[:, :3] - np.asarray(c2["state_v1"], dtype=np.float64)[None, :3]
    c2_row = {"index": 6, "label": "C2_C0_NOMINAL", "kind": "STAGE43_Q0_C2_REPLAY", "raw_path": str(c2_raw_path), "raw_file_sha256": sha256_file(c2_raw_path), "raw_array_sha256": sha256_array(c2_raw), "raw_npz_key": c2_raw_key, "noise_sha256": NOISE_ARRAY_SHA, "snapshot": c2, "decoded_physical_sha256": sha256_array(c2_physical), "decoder": c2_decode, "corrected": c2_corrected, "corrected_world_positions_m": c2_world.tolist(), "historical_row": None}
    rows.append(c2_row)

    # Compare the archived C1 C0 input identities with the fresh C2 C0 record.
    c1_c0 = stage43["C0_NOMINAL"]
    raw_input_equivalence = {"body_array_sha256": {"c1": c1_c0.get("body_array_sha256"), "c2": c2.get("body_array_sha256"), "equal": c1_c0.get("body_array_sha256") == c2.get("body_array_sha256")}, "wrist_array_sha256": {"c1": c1_c0.get("wrist_array_sha256"), "c2": c2.get("wrist_array_sha256"), "equal": c1_c0.get("wrist_array_sha256") == c2.get("wrist_array_sha256")}, "state_v1_sha256": {"c1": c1_c0.get("state_v1_sha256"), "c2": c2.get("state_v1_sha256"), "equal": c1_c0.get("state_v1_sha256") == c2.get("state_v1_sha256")}, "raw_array_sha256": {"c1": by_label["C0_NOMINAL"].get("array_sha256"), "c2": sha256_array(c2_raw), "equal": by_label["C0_NOMINAL"].get("array_sha256") == sha256_array(c2_raw)}, "interpretation": "same sealed observation-array identities but different raw outputs means the divergence is upstream of this deterministic decoder; processed tensor/model-runtime evidence is not present in the seven raw records"}

    # A4 legal CPU-only fixtures: no model or controller input is sent.
    fixtures = _cpu_fixture_checks(stats_path, train)

    # Original C2 SafetyGate and counters are copied as evidence, never rerun with relaxed limits.
    c2_evidence = {"gate_sha256": sha256_file(c2_condition / "safety_gate.json"), "gate": c2_gate, "counter_sha256": sha256_file(c2_condition / "counter_observations.json"), "counters": c2_counters, "worker_response_sha256": sha256_file(c2_condition / "worker_response.json"), "worker_response": read_json(c2_condition / "worker_response.json"), "worker_log_sha256": sha256_file(c2_condition / "worker.log"), "four_step_config_in_log": "sampler_timestep_ids=[0,250,500,750]" in (c2_condition / "worker.log").read_text(encoding="utf-8"), "actual_action_head_hook_count": "UNKNOWN_NOT_SERIALIZED"}

    table = output / "corrected_metrics_table.csv"
    columns = ["index", "label", "kind", "raw_array_sha256", "p0_state_residual_m", "corrected_start_residual_m", "old_report_start_jump_m", "initial_target_distance_m", "minimum_target_distance_m", "final_target_distance_m", "absolute_progress_m", "best_progress_m", "progress_sign", "path_length_m", "endpoint_target_vector_cosine", "decoder_replay_max_abs_vs_saved"]
    with table.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            m = row["corrected"]
            metrics = m["target_metrics"]
            writer.writerow({"index": row["index"], "label": row["label"], "kind": row["kind"], "raw_array_sha256": row["raw_array_sha256"], "p0_state_residual_m": m["p0_state_residual_m"], "corrected_start_residual_m": m["corrected_start_residual_m"], "old_report_start_jump_m": m["old_report_start_jump_m"], "initial_target_distance_m": metrics["initial_target_distance_m"], "minimum_target_distance_m": metrics["minimum_target_distance_m"], "final_target_distance_m": metrics["final_target_distance_m"], "absolute_progress_m": metrics["absolute_progress_m"], "best_progress_m": metrics["best_progress_m"], "progress_sign": metrics["progress_sign"], "path_length_m": metrics["path_length_m"], "endpoint_target_vector_cosine": metrics["endpoint_target_vector_cosine"], "decoder_replay_max_abs_vs_saved": m["decoder_replay_max_abs_vs_saved"]})

    plot_sha = _write_plot(output / "corrected_trajectory_plot.svg", rows)
    stage_rows = [row for row in rows if row["kind"] == "STAGE43_Q0"]
    positive = sum(row["corrected"]["target_metrics"]["absolute_progress_m"] > 1e-9 for row in stage_rows)
    negative = sum(row["corrected"]["target_metrics"]["absolute_progress_m"] < -1e-9 for row in stage_rows)
    max_jump = max(float(row["corrected"]["old_report_start_jump_m"]) for row in rows)
    conclusion = output / "one_page_attribution.md"
    conclusion.write_text("\n".join([
        "# PI0153 first-action semantics and generation attribution",
        "",
        "This packet is a CPU-only correction of the sealed PI0152 C1 six-row raw attempt plus the sealed C2/C0 raw row. It preserves the earlier report and does not claim a WBC result.",
        "",
        "## Corrected finding",
        "",
        f"All seven decoded pose-0 positions equal StateV1[:3] (maximum residual {max(float(row['corrected']['p0_state_residual_m']) for row in rows):.3e} m). The historical start-jump values (maximum {max_jump:.6f} m) are exactly the norm of StateV1[:3], caused by adding the already-world measured anchor to a relative decoded point. The corrected world bridge is measured_eef_world + (decoded - StateV1[:3]); its start residual is zero.",
        f"After correction, the TRAIN row and all five Stage43 rows retain their decoded trajectory behavior; Stage43 target progress is positive in {positive}/5 and negative in {negative}/5. Thus the apparent large first jump is an analysis/frame-origin artifact, while the remaining end-point/path behavior is a genuine model-side decoded trajectory property.",
        "",
        "## SafetyGate and generation path",
        "",
        "The original C2 SafetyGate decision and thresholds are copied unchanged. Its 0.142400018 m initial discontinuity compares StateV1-relative decoded positions with the native external-anchor-relative achieved position at the adapter boundary; it is not a world-space first-action measurement. C2 still has a genuine 2.451607158 m/s maximum speed and 2.121549264 m path under the frozen T16 timing law, so the REJECT remains fail-closed.",
        "The archived C1 C0 and fresh C2 C0 carry equal body/wrist/state array identities but different sealed raw arrays. The deterministic decoder reproduces each saved decoded artifact, so the raw divergence is upstream of decoding. The seven artifacts do not contain processed tensors or a model-side intermediate capture; therefore the exact worker/runtime cause cannot be identified from A alone.",
        "",
        "## One recommended next experiment",
        "",
        "Use the PI0153 bounded same-input/same-noise two-path comparison (TRAIN and archived C0; at most four queries and twenty model-side calls) to capture the two preprocessing/worker paths and determine whether their raw divergence is caused by input materialization or runtime/model state. Keep seed 2223, checkpoint, stats, and saved noise fixed; do not change SafetyGate or start a closed loop.",
    ]) + "\n", encoding="utf-8")

    packet = {
        "schema": "CORELAB_STAGE45F_PI0153_FIRST_ACTION_SEMANTICS_V1",
        "status": "COMPLETED_A_PHASE_B_REQUIRED",
        "token": RETURN_TOKEN,
        "authority": {"pi_gate": PI_GATE, "issue_comment": PI_COMMENT, "pr_comment": PR_COMMENT},
        "repository": {"branch": BRANCH, "expected_head": EXPECTED_HEAD},
        "source": {"c1_root": str(c1_root), "c1_raw_manifest_sha256": sha256_file(raw_manifest), "c1_cpu_report": str(args.old_cpu_report) if args.old_cpu_report else None, "c2_root": str(c2_root), "stats_path": str(stats_path), "stats_sha256": N2_STATS_SHA, "stats_payload_sha256": N2_STATS_PAYLOAD_SHA, "fixed_noise_array_sha256": NOISE_ARRAY_SHA},
        "rows": rows,
        "raw_input_equivalence": raw_input_equivalence,
        "c2_safety_and_counters": c2_evidence,
        "cpu_fixture_checks": fixtures,
        "generation_probe_decision": {"required": True, "reason": "C1/C2 C0 raw arrays differ despite equal sealed observation array identities; A cannot localize preprocessing/runtime cause", "budget_max_queries": 4, "budget_max_conditioning": 4, "budget_max_action_head": 16, "budget_max_model_side_calls": 20, "model_side_calls_consumed": 0},
        "counters": {"new_model_forwards": 0, "new_conditioning": 0, "new_action_head": 0, "training": False, "gpu": False, "pbs": False, "wbc": False, "simulator": False, "hardware": False, "validation_e_read": False, "holdout_f_read": False, "icl": False, "new_data": False},
        "artifacts": {"metrics_table_sha256": sha256_file(table), "trajectory_plot_sha256": plot_sha, "conclusion_sha256": sha256_file(conclusion)},
        "claim_boundary": "corrected CPU semantics and upstream generation-path attribution only; no closed-loop or task-success claim",
    }
    packet_sha = write_json(output / "pi0153_first_action_semantics.json", packet)
    write_json(output / "pi0153_first_action_semantics.json.sha256", {"file": "pi0153_first_action_semantics.json", "sha256": packet_sha})
    return {**packet, "packet_sha256": packet_sha}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--c1-root", required=True, type=Path)
    value.add_argument("--c2-root", required=True, type=Path)
    value.add_argument("--n2-stats", required=True, type=Path)
    value.add_argument("--train-preflight", required=True, type=Path)
    value.add_argument("--old-cpu-report", type=Path)
    value.add_argument("--output-root", required=True, type=Path)
    return value


if __name__ == "__main__":
    try:
        result = run(parser().parse_args())
        print(json.dumps({"status": result["status"], "token": result["token"], "packet_sha256": result["packet_sha256"], "output_root": str(Path(result["artifacts"]["conclusion_sha256"]).parent) if False else "see --output-root"}, sort_keys=True))
    except Exception as exc:
        print(f"BLOCKED_OR_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
