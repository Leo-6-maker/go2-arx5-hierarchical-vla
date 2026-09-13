#!/usr/bin/env python3
"""PI0155 CPU-only supplement to the accepted PI0153 semantics packet.

This file deliberately does not run a model.  It reads only the sealed
PI0153 packet, its already-materialised decoded arrays, and the accepted
TRAIN source snapshot.  The supplement keeps the original packet immutable
while making three previously implicit boundaries explicit:

* the TRAIN anchor is derived from a canonical parquet label, not proven to
  be an independently measured EEF world pose;
* the historical C2 worker did not serialize the actual injected ``x0``;
* the legal TRAIN expert and hold fixtures have full trajectory metrics.

The output is a small, reviewable correction layer.  It never reads
Validation-E/Holdout-F, starts a GPU, calls GR00T, or sends anything to WBC.
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
from typing import Any, Mapping

import numpy as np


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

PI_GATE = "PI_GATE_STATE_0155"
PI_COMMENT = 5650148861
PR_COMMENT = 5650151180
RETURN_TOKEN = "CORELAB_PI0155_GENERATION_PATH_DIAGNOSIS_FOR_PI_REVIEW"

EXPECTED_A_PACKET_SHA = "a5f45b8242d4439d39915abd0d4ac9979060708343924f737d8f531b2326508d"
EXPECTED_N2_STATS_SHA = "0c9450e6aeb0765eebc676414b2a6a84f71dddb7eea1de5453551f062b25fdbb"
EXPECTED_NOISE_SHA = "5924010a4e506159b654d0d6164125da5bf0e513ff410bf6e8e5f964a9ab50bd"
TRAIN_LABEL = "TRAIN_a_expert-0000"
TRAIN_WINDOW = "stage22_expert_red_target_train_a:expert-0000"
TARGET = np.asarray([0.30, 0.10, 0.58], dtype=np.float64)
HORIZON = 40
PHYSICAL_DIM = 10
DT = 0.05


class SupplementError(RuntimeError):
    """Fail-closed error for a stale or ambiguous sealed input."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SupplementError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def read_json(path: Path) -> Any:
    path = Path(path)
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not stat.S_ISLNK(mode), f"not a regular JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> str:
    require(not path.exists(), f"refusing to overwrite evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return sha256_file(path)


def load_array(path: Path, shape: tuple[int, ...], label: str) -> np.ndarray:
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not stat.S_ISLNK(mode), f"{label} is not a regular file: {path}")
    value = np.asarray(np.load(path, allow_pickle=False))
    require(value.shape == shape and np.isfinite(value).all(), f"{label} shape/finite mismatch: {value.shape}")
    return value


def _numeric_summary(value: np.ndarray) -> dict[str, Any]:
    value = np.asarray(value, dtype=np.float64)
    require(value.size > 0 and np.isfinite(value).all(), "metric input is empty/non-finite")
    return {
        "count": int(value.size),
        "min": float(value.min()),
        "mean": float(value.mean()),
        "median": float(np.median(value)),
        "p95": float(np.percentile(value, 95.0)),
        "max": float(value.max()),
    }


def _trajectory_metrics(position: np.ndarray, *, target: np.ndarray | None, label: str) -> dict[str, Any]:
    position = np.asarray(position, dtype=np.float64)
    require(position.shape == (HORIZON, 3) and np.isfinite(position).all(), f"{label} position ABI/finite mismatch")
    delta = np.diff(position, axis=0)
    speed = np.linalg.norm(delta, axis=1) / DT
    second = np.linalg.norm(np.diff(position, n=2, axis=0), axis=1)
    result: dict[str, Any] = {
        "label": label,
        "sample_count": HORIZON,
        "dt_s": DT,
        "path_length_m": float(np.linalg.norm(delta, axis=1).sum()),
        "displacement_m": (position[-1] - position[0]).tolist(),
        "speed_mps": _numeric_summary(speed) if speed.size else None,
        "roughness_second_difference_m": _numeric_summary(second) if second.size else None,
        "position_sha256_float64": sha256_array(position),
    }
    if target is not None:
        target = np.asarray(target, dtype=np.float64).reshape(3)
        distances = np.linalg.norm(position - target[None, :], axis=1)
        initial = float(distances[0])
        final = float(distances[-1])
        result["target"] = {
            "target_world_m": target.tolist(),
            "initial_target_distance_m": initial,
            "minimum_target_distance_m": float(distances.min()),
            "minimum_target_index": int(distances.argmin()),
            "final_target_distance_m": final,
            "absolute_progress_m": initial - final,
            "best_progress_m": initial - float(distances.min()),
            "progress_sign": "POSITIVE" if initial - final > 1e-9 else ("ZERO" if abs(initial - final) <= 1e-9 else "NEGATIVE"),
        }
    else:
        result["target"] = None
    return result


def _reference_pair_metrics(expert: np.ndarray, hold: np.ndarray, state: np.ndarray, anchor: np.ndarray) -> dict[str, Any]:
    """Return full state-relative and source-anchor metrics for legal fixtures."""
    state = np.asarray(state, dtype=np.float64).reshape(10)
    anchor = np.asarray(anchor, dtype=np.float64).reshape(3)
    # The source action is in the same relative pose convention as StateV1.
    # Keep both representations so the report cannot silently promote a
    # parquet-derived origin to a measured EEF pose.
    expert_relative = expert[:, :3] - state[None, :3]
    hold_relative = hold[:, :3] - state[None, :3]
    relative_target = TARGET - anchor + state[:3]
    expert_world_if_anchor = anchor[None, :] + expert_relative
    hold_world_if_anchor = anchor[None, :] + hold_relative
    expert_endpoint_error = float(np.linalg.norm(expert[-1, :3] - expert[-1, :3]))
    hold_endpoint_error = float(np.linalg.norm(hold[-1, :3] - hold[0, :3]))
    return {
        "expert": {
            "physical_action_sha256": sha256_array(expert),
            "frame": "StateV1_relative_action_frame",
            "relative": _trajectory_metrics(expert_relative, target=relative_target, label="TRAIN.expert.relative"),
            "world_if_anchor_applied": _trajectory_metrics(expert_world_if_anchor, target=TARGET, label="TRAIN.expert.world_if_anchor"),
            "endpoint_reconstruction_error_m": expert_endpoint_error,
            "endpoint_xyz_m": expert[-1, :3].tolist(),
        },
        "hold": {
            "physical_action_sha256": sha256_array(hold),
            "frame": "StateV1_relative_action_frame",
            "relative": _trajectory_metrics(hold_relative, target=relative_target, label="TRAIN.hold.relative"),
            "world_if_anchor_applied": _trajectory_metrics(hold_world_if_anchor, target=TARGET, label="TRAIN.hold.world_if_anchor"),
            "endpoint_reconstruction_error_m": hold_endpoint_error,
            "endpoint_xyz_m": hold[-1, :3].tolist(),
        },
        "comparison": {
            "expert_vs_hold_position_rmse_m": float(np.sqrt(np.mean((expert[:, :3] - hold[:, :3]) ** 2))),
            "expert_vs_hold_position_max_abs_m": float(np.max(np.abs(expert[:, :3] - hold[:, :3]))),
            "expert_path_minus_hold_path_m": float(np.linalg.norm(np.diff(expert[:, :3], axis=0), axis=1).sum() - np.linalg.norm(np.diff(hold[:, :3], axis=0), axis=1).sum()),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    packet_path = Path(args.a_packet).resolve()
    preflight_path = Path(args.train_preflight).resolve()
    c1_root = Path(args.c1_root).resolve()
    output = Path(args.output_root).resolve()
    require(not output.exists(), f"output already exists: {output}")
    require(sha256_file(packet_path) == EXPECTED_A_PACKET_SHA, "accepted PI0153 packet hash drift")
    packet = read_json(packet_path)
    require(packet.get("schema") == "CORELAB_STAGE45F_PI0153_FIRST_ACTION_SEMANTICS_V1", "PI0153 packet schema drift")
    require(len(packet.get("rows", [])) == 7, "PI0153 packet is not the accepted seven-row set")
    require(packet.get("authority", {}).get("issue_comment") == 5649870843, "PI0153 authority drift")

    preflight = read_json(preflight_path)
    source = preflight.get("canonical_label_source", {})
    require(source.get("kind") == "SEALED_FULL24_TRAIN_PARQUET", "TRAIN source kind drift")
    require(source.get("anchor_derivation") == "target_world_m_minus_canonical_parquet_action_pose39", "TRAIN anchor derivation drift")
    require(source.get("label_materialization") == "canonical_parquet_action_and_observation_state", "TRAIN label materialization drift")
    require(source.get("historical_source_json_execution_authority") is False, "historical source execution authority unexpectedly true")
    snapshots = [item for item in preflight.get("snapshots", []) if item.get("window_id") == TRAIN_WINDOW]
    require(len(snapshots) == 1, "TRAIN snapshot identity is not unique")
    train_snapshot = snapshots[0]
    expert = np.asarray(train_snapshot.get("expert_physical_action"), dtype=np.float64)
    state = np.asarray(train_snapshot.get("state_v1"), dtype=np.float64).reshape(-1)
    anchor = np.asarray(train_snapshot.get("anchor_world_m"), dtype=np.float64).reshape(-1)
    require(expert.shape == (HORIZON, PHYSICAL_DIM) and np.isfinite(expert).all(), "TRAIN expert action ABI/finite drift")
    require(state.shape == (PHYSICAL_DIM,) and np.isfinite(state).all(), "TRAIN StateV1 ABI/finite drift")
    require(anchor.shape == (3,) and np.isfinite(anchor).all(), "TRAIN anchor ABI/finite drift")

    hold = np.repeat(state[None, :], HORIZON, axis=0)
    reference_metrics = _reference_pair_metrics(expert, hold, state, anchor)

    c2_rows = [row for row in packet["rows"] if row.get("label") == "C2_C0_NOMINAL"]
    require(len(c2_rows) == 1, "C2 row missing from accepted packet")
    c2_row = c2_rows[0]
    c2_counter = packet.get("c2_safety_and_counters", {}).get("counters", {})
    c2_noise = {
        "status": "UNKNOWN_NOT_RECORDED",
        "actual_x0_sha256": None,
        "actual_x0_dtype": None,
        "actual_x0_path": None,
        "seed": 4242,
        "expected_source_noise_sha256": EXPECTED_NOISE_SHA,
        "historical_packet_field_was": c2_row.get("noise_sha256"),
        "reason": "historical C2 worker serialized raw output but did not persist the torch.randn x0 tensor or an effective model-entry copy",
        "counter_hook_count": "UNKNOWN_NOT_SERIALIZED",
    }

    train_frame = {
        "window_id": TRAIN_WINDOW,
        "source_kind": source.get("kind"),
        "anchor_derivation": source.get("anchor_derivation"),
        "label_materialization": source.get("label_materialization"),
        "historical_source_json_execution_authority": source.get("historical_source_json_execution_authority"),
        "anchor_world_m": anchor.tolist(),
        "frame_source_class": "REFERENCE_ORIGIN_DERIVED_NOT_PROVEN_MEASURED_EEF",
        "measured_eef_world_proven": False,
        "absolute_world_target_metrics": "UNVERIFIED",
        "relative_reference_metrics": "VALID_FOR_SOURCE_ACTION_FRAME",
        "explanation": "anchor_world_m is target_world_m minus canonical parquet action pose39; no independent measured EEF/world-frame pose is present in this source record",
    }

    output.mkdir(parents=True)
    csv_path = output / "pi0155_reference_metrics.csv"
    columns = ["fixture", "frame", "path_length_m", "speed_mean_mps", "speed_max_mps", "final_target_distance_m", "absolute_progress_m", "position_sha256_float64"]
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for fixture in ("expert", "hold"):
            for frame in ("relative", "world_if_anchor_applied"):
                metrics = reference_metrics[fixture][frame]
                target = metrics["target"] or {}
                writer.writerow({
                    "fixture": fixture,
                    "frame": frame,
                    "path_length_m": metrics["path_length_m"],
                    "speed_mean_mps": metrics["speed_mps"]["mean"],
                    "speed_max_mps": metrics["speed_mps"]["max"],
                    "final_target_distance_m": target.get("final_target_distance_m"),
                    "absolute_progress_m": target.get("absolute_progress_m"),
                    "position_sha256_float64": metrics["position_sha256_float64"],
                })

    evidence = {
        "schema": "CORELAB_STAGE45F_PI0155_CPU_SUPPLEMENT_V1",
        "status": "PASS",
        "token": RETURN_TOKEN,
        "authority": {"pi_gate": PI_GATE, "issue_comment": PI_COMMENT, "pr_comment": PR_COMMENT},
        "accepted_inputs": {
            "pi0153_packet": {"path": str(packet_path), "sha256": sha256_file(packet_path), "expected_sha256": EXPECTED_A_PACKET_SHA},
            "train_preflight": {"path": str(preflight_path), "sha256": sha256_file(preflight_path)},
            "c1_root": str(c1_root),
            "c2_root": str(args.c2_root),
            "n2_stats_expected_sha256": EXPECTED_N2_STATS_SHA,
        },
        "train_frame_source": train_frame,
        "historical_c2_noise": c2_noise,
        "reference_metrics": reference_metrics,
        "historical_c2_counters": c2_counter,
        "execution_boundary": {
            "new_model_forwards": 0,
            "new_conditioning": 0,
            "new_action_head": 0,
            "gpu": False,
            "training": False,
            "pbs": False,
            "wbc": False,
            "simulator": False,
            "hardware": False,
            "validation_e_read": False,
            "holdout_f_read": False,
        },
        "artifacts": {"metrics_csv": str(csv_path), "metrics_csv_sha256": sha256_file(csv_path)},
        "claim_boundary": "PI0153 accepted A preserved; TRAIN absolute world target metrics remain UNVERIFIED until a measured EEF/world-frame source is proven; C2 actual x0 remains UNKNOWN_NOT_RECORDED",
    }
    report_sha = write_json(output / "pi0155_cpu_supplement.json", evidence)
    write_json(output / "pi0155_cpu_supplement.json.sha256", {"file": "pi0155_cpu_supplement.json", "sha256": report_sha})
    conclusion = output / "one_page_attribution.md"
    conclusion.write_text(
        "\n".join([
            "# PI0155 CPU supplement",
            "",
            "The accepted PI0153 seven-row packet is unchanged. This correction layer adds provenance and fixture metrics only; it performs no model or controller operation.",
            "",
            "## TRAIN frame",
            "",
            "The accepted preflight labels TRAIN as `SEALED_FULL24_TRAIN_PARQUET` and derives `anchor_world_m` as `target_world_m_minus_canonical_parquet_action_pose39`. That is a reference-origin derivation, not an independently measured EEF world pose. Relative StateV1-frame metrics are valid; absolute world target metrics using that anchor are explicitly `UNVERIFIED`.",
            "",
            "## Historical C2 noise",
            "",
            f"The historical packet's expected source noise SHA remains `{EXPECTED_NOISE_SHA}`, but the C2 worker did not serialize actual `x0`; actual noise is therefore `UNKNOWN_NOT_RECORDED`, with seed 4242 retained independently.",
            "",
            "## Fixture reference",
            "",
            "Full path, speed, target-distance (where frame-qualified), endpoint, and expert-versus-hold metrics are in `pi0155_reference_metrics.csv` and the JSON report. The hold fixture is repeated StateV1; no model, WBC, or SafetyGate call was made.",
        ]) + "\n",
        encoding="utf-8",
    )
    conclusion_sha = sha256_file(conclusion)
    evidence["artifacts"]["conclusion"] = str(conclusion)
    evidence["artifacts"]["conclusion_sha256"] = conclusion_sha
    # The report is written once above to preserve a deterministic no-overwrite
    # contract; the conclusion hash is intentionally also carried in a small
    # sidecar rather than mutating the already sealed JSON.
    write_json(output / "pi0155_cpu_supplement_artifacts.json", evidence["artifacts"])
    return {"status": evidence["status"], "token": RETURN_TOKEN, "report_sha256": report_sha, "metrics_csv_sha256": sha256_file(csv_path), "conclusion_sha256": conclusion_sha, "output_root": str(output), "train_frame_source": train_frame, "historical_c2_noise": c2_noise, "reference_metrics": reference_metrics}


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a-packet", required=True, type=Path)
    parser.add_argument("--train-preflight", required=True, type=Path)
    parser.add_argument("--c1-root", required=True, type=Path)
    parser.add_argument("--c2-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


if __name__ == "__main__":
    try:
        print(json.dumps(run(parser().parse_args()), sort_keys=True))
    except Exception as exc:
        print(f"BLOCKED_OR_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
