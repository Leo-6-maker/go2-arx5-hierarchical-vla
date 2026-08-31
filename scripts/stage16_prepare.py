#!/usr/bin/env python3
"""Prepare the auditable Stage16 native benchmark manifests.

This is an offline step.  It reads the immutable Stage15 shape manifest and
reset-bound inputs, analyzes the 17 native static-relocation candidates, and
creates derived progressive references without changing any source shape.
The resulting NPZ files stay node-local; the JSON manifests contain their
provenance and are intended to be copied into the Stage16 report bundle.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

from go2arx5_vla.benchmark.native_suite import build_native_suite
from go2arx5_vla.benchmark.progressive_relocation import (
    _reach_ratio,
    analyze_candidate,
    initial_reference_continuity,
    load_npz,
    load_reach_table,
    make_progressive_relocation,
    select_progressive_ids,
    sha256_file,
    trajectory_descriptor,
    trajectory_hash,
)
from go2arx5_vla.experiments.native_wbc import make_case, write_manifest


EXPECTED_HASHES = {
    "archive_sha256": "b52a53bef87340e6244ead352fa434390faee65246cd32ab8e6648336a018167",
    "native_urdf_sha256": "96c1a81e3dd42af5f1bf87ac8958a0aebd7e798ce06408462e0e49a8a2067e25",
    "arm_checkpoint_sha256": "99a3a9bda30f7372c7c31240cc3584757ba46cb3b6b9881f00b371906d077a0a",
    "dog_checkpoint_sha256": "51776ddfd7177213444c0bf0c8f017b66ffd28a8674458aa0a16c3b9f77cc670",
    "config_sha256": "56132d4ff9b866ce196bbc1b1353900b967c54f0f48d2d4e4b82c3555769a0bb",
}


DEFAULT_PREFERRED_PROGRESSIVE_IDS = (
    "E1_boundary_01",
    "E2_static_0_0",
    "E2_static_0_1",
    "E2_static_0_2",
    "E2_static_0_3",
    "E2_static_1_0",
    "E2_static_2_1",
    "E2_static_3_3",
)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _array(payload: Mapping[str, Any], key: str) -> np.ndarray:
    return np.asarray(payload[key], dtype=np.float64)


def _verify_hashes(paths: Mapping[str, Path]) -> Dict[str, str]:
    actual: Dict[str, str] = {}
    for key, path in paths.items():
        value = sha256_file(path)
        actual[key] = value
        expected = EXPECTED_HASHES[key]
        if value != expected:
            raise RuntimeError(
                "%s mismatch for %s: expected %s, got %s"
                % (key, path, expected, value)
            )
    return actual


def _fixed_base_summary(
    position: np.ndarray,
    shoulder_position: np.ndarray,
    shoulder_quaternion: np.ndarray,
    reach_table: np.ndarray,
    timestamp: np.ndarray,
) -> Dict[str, Any]:
    rho = _reach_ratio(position, shoulder_position, shoulder_quaternion, reach_table)
    unreachable = rho > 1.0
    indices = np.flatnonzero(unreachable)
    first = int(indices[0]) if len(indices) else None
    return {
        "fixed_base_max_rho_pointwise": float(np.max(rho)),
        "fixed_base_feasible_fraction": float(np.mean(~unreachable)),
        "first_fixed_base_unreachable_index": first,
        "first_fixed_base_unreachable_time_s": None if first is None else float(timestamp[first]),
        "fixed_base_rho_p95": float(np.percentile(rho, 95)),
        "fixed_base_rho": rho.tolist(),
    }


def _parent_hash(
    trajectory_id: str,
    timestamp: np.ndarray,
    position: np.ndarray,
    quaternion: np.ndarray,
    shape_sha256: str,
) -> str:
    return trajectory_hash(
        timestamp,
        position,
        quaternion,
        {
            "trajectory_id": trajectory_id,
            "shape_sha256": shape_sha256,
            "reference_kind": "STAGE15_NATIVE_BOUND_REFERENCE",
        },
    )


def _case_metadata(
    *,
    row: Mapping[str, Any],
    case_hash: str,
    path: Path,
    duration_s: float,
    reset_seed: int,
    episode_binding_id: str,
    immutable_hashes: Mapping[str, str],
    runtime_version: str,
    simulator: str,
    asset: str,
    source_kind: str,
    derived: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    placement = row.get("placement_profile", {})
    native_classification = row.get("native_classification", {})
    metadata: Dict[str, Any] = {
        "trajectory_id": str(row["trajectory_id"]),
        "trajectory_hash": case_hash,
        "shape_id": str(row.get("shape_id") or row["trajectory_id"]),
        "placement_id": str(
            row.get("placement_id")
            or placement.get("profile_id")
            or "UNSPECIFIED"
        ),
        "episode_binding_id": episode_binding_id,
        "reset_seed": int(reset_seed),
        "reference_interval": [0.0, float(duration_s)],
        "archive_sha256": immutable_hashes["archive_sha256"],
        "arm_checkpoint_sha256": immutable_hashes["arm_checkpoint_sha256"],
        "dog_checkpoint_sha256": immutable_hashes["dog_checkpoint_sha256"],
        "config_sha256": immutable_hashes["config_sha256"],
        "runtime_version": runtime_version,
        "simulator": simulator,
        "asset": asset,
        "reference_mode": "ALIGNED_NATIVE_CONTINUOUS_REFERENCE",
        "trajectory_path": str(path),
        "source_kind": source_kind,
        "shape_file": row.get("shape_file"),
        "shape_sha256": row.get("shape_sha256"),
        "original_file": row.get("original_file"),
        "original_file_sha256": row.get("original_file_sha256"),
        "placement_profile": placement,
        "native_classification": native_classification,
    }
    if derived is not None:
        metadata["derived_provenance"] = dict(derived)
    return metadata


def _materialize_progressive(
    *,
    row: Mapping[str, Any],
    bound_entry: Mapping[str, Any],
    trace: Mapping[str, Any],
    reach_table: np.ndarray,
    output_dir: Path,
    analysis: Mapping[str, Any],
) -> tuple[Dict[str, np.ndarray], Dict[str, Any], str, Dict[str, Any]]:
    trajectory_id = str(row["trajectory_id"])
    bound_data = load_npz(Path(str(bound_entry["path"])))
    timestamp = _array(bound_data, "timestamp")
    position = _array(bound_data, "position") + _array(trace, "anchor_position")[None, :]
    quaternion = _array(bound_data, "quaternion")
    parent_hash = _parent_hash(
        trajectory_id,
        timestamp,
        position,
        quaternion,
        str(row.get("shape_sha256", "")),
    )
    derived, provenance = make_progressive_relocation(
        timestamp,
        position,
        quaternion,
        _array(bound_entry, "reset_tcp_position_world"),
        _array(bound_entry, "reset_tcp_quaternion_xyzw"),
        parent_id=trajectory_id,
        parent_hash=parent_hash,
    )
    out_timestamp = derived["timestamp"]
    out_position = derived["position"]
    out_quaternion = derived["quaternion"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (trajectory_id + ".npz")
    anchor = _array(trace, "anchor_position")
    np.savez_compressed(
        output_path,
        timestamp=out_timestamp,
        position=out_position - anchor[None, :],
        quaternion=out_quaternion,
    )
    fixed = _fixed_base_summary(
        out_position,
        _array(trace, "shoulder_position"),
        _array(trace, "shoulder_quaternion_xyzw"),
        reach_table,
        out_timestamp,
    )
    derived_hash = str(provenance["derived_trajectory_hash"])
    derived_analysis: Dict[str, Any] = {
        "trajectory_id": trajectory_id,
        "parent_id": trajectory_id,
        "native_class": row["native_classification"]["native_class"],
        "cohort": "PROGRESSIVE_BASE_RELOCATION",
        "source_kind": "DERIVED_PROGRESSIVE_RELOCATION",
        "path": str(output_path),
        "derived_trajectory_hash": derived_hash,
        "parent_trajectory_hash": parent_hash,
        "descriptor": trajectory_descriptor(out_timestamp, out_position, out_quaternion),
        "continuity": initial_reference_continuity(
            out_position,
            out_quaternion,
            _array(bound_entry, "reset_tcp_position_world"),
            _array(bound_entry, "reset_tcp_quaternion_xyzw"),
        ),
        "native_parent_analysis": dict(analysis),
        "fixed_base": fixed,
        "lead_in": provenance,
        "challenge_start_time_s": float(provenance["join_time_s"]),
        "challenge_source_values_unchanged": True,
    }
    return derived, provenance, parent_hash, derived_analysis


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage15-manifest", type=Path, required=True)
    parser.add_argument("--bound-json", type=Path, required=True)
    parser.add_argument("--reset-trace", type=Path, required=True)
    parser.add_argument("--reach-table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--arm-checkpoint", type=Path, required=True)
    parser.add_argument("--dog-checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reset-seed", type=int, default=13)
    parser.add_argument(
        "--progressive-id",
        action="append",
        dest="progressive_ids",
        help="preferred native static candidate; may be repeated",
    )
    parser.add_argument("--target-count", type=int, default=28)
    parser.add_argument(
        "--runtime-version",
        default="python3.8.20;torch2.3.1;cuda12.1;isaacgym_native",
    )
    args = parser.parse_args()

    immutable_paths = {
        "archive_sha256": args.archive,
        "native_urdf_sha256": args.urdf,
        "arm_checkpoint_sha256": args.arm_checkpoint,
        "dog_checkpoint_sha256": args.dog_checkpoint,
        "config_sha256": args.config,
    }
    immutable_hashes = _verify_hashes(immutable_paths)
    args.output.mkdir(parents=True, exist_ok=True)

    manifest = _load_json(args.stage15_manifest)
    entries = list(manifest["entries"])
    by_id = {str(row["trajectory_id"]): row for row in entries}
    bound_payload = _load_json(args.bound_json)
    bound_by_id = {str(row["trajectory_id"]): row for row in bound_payload["entries"]}
    trace = load_npz(args.reset_trace)
    reach_table = load_reach_table(args.reach_table)

    static_analyses = []
    for row in entries:
        native_class = row.get("native_classification", {}).get("native_class")
        if native_class != "NATIVE_STATIC_BASE_RELOCATION_SUFFICIENT":
            continue
        trajectory_id = str(row["trajectory_id"])
        if trajectory_id not in bound_by_id:
            raise KeyError("missing bound input for static candidate: %s" % trajectory_id)
        shape_data = load_npz(Path(str(row["shape_file"])))
        bound_data = load_npz(Path(str(bound_by_id[trajectory_id]["path"])))
        analysis = analyze_candidate(
            row,
            bound_by_id[trajectory_id],
            shape_data,
            bound_data,
            trace,
            reach_table,
        )
        analysis["bound_trajectory_hash"] = _parent_hash(
            trajectory_id,
            _array(bound_data, "timestamp"),
            _array(bound_data, "position") + _array(trace, "anchor_position")[None, :],
            _array(bound_data, "quaternion"),
            str(row.get("shape_sha256", "")),
        )
        static_analyses.append(analysis)

    preferred = args.progressive_ids or list(DEFAULT_PREFERRED_PROGRESSIVE_IDS)
    progressive_ids = select_progressive_ids(
        static_analyses,
        count=8,
        preferred_ids=preferred,
    )
    if not 6 <= len(progressive_ids) <= 10:
        raise RuntimeError("progressive cohort size is outside the Stage16 target range")

    progressive_dir = args.output / "progressive_relocation"
    progressive_results: Dict[str, Dict[str, Any]] = {}
    progressive_arrays: Dict[str, Dict[str, np.ndarray]] = {}
    for trajectory_id in progressive_ids:
        row = by_id[trajectory_id]
        derived, provenance, parent_hash, derived_analysis = _materialize_progressive(
            row=row,
            bound_entry=bound_by_id[trajectory_id],
            trace=trace,
            reach_table=reach_table,
            output_dir=progressive_dir,
            analysis=next(item for item in static_analyses if item["trajectory_id"] == trajectory_id),
        )
        progressive_results[trajectory_id] = derived_analysis
        progressive_arrays[trajectory_id] = derived

    continuity_payload = {
        "schema": "STAGE16_INITIAL_REFERENCE_CONTINUITY_V1",
        "reset_trace": str(args.reset_trace),
        "reset_seed": int(args.reset_seed),
        "candidate_count": len(entries),
        "candidates": [
            {
                "trajectory_id": str(row["trajectory_id"]),
                "native_class": row["native_classification"]["native_class"],
                "family": row.get("family"),
                "continuity": analysis["continuity"],
            }
            for row in entries
            for analysis in [
                next(
                    (
                        item
                        for item in static_analyses
                        if item["trajectory_id"] == str(row["trajectory_id"])
                    ),
                    None,
                )
            ]
            if analysis is not None
        ],
    }
    # Compute continuity for the non-static strata as well, because the
    # continuity audit covers every candidate, not only the relocation cohort.
    continuity_by_id = {
        item["trajectory_id"]: item["continuity"] for item in static_analyses
    }
    for row in entries:
        trajectory_id = str(row["trajectory_id"])
        if trajectory_id in continuity_by_id:
            continue
        if trajectory_id not in bound_by_id:
            continue
        bound_data = load_npz(Path(str(bound_by_id[trajectory_id]["path"])))
        position = _array(bound_data, "position") + _array(trace, "anchor_position")[None, :]
        continuity_by_id[trajectory_id] = initial_reference_continuity(
            position,
            _array(bound_data, "quaternion"),
            _array(bound_by_id[trajectory_id], "reset_tcp_position_world"),
            _array(bound_by_id[trajectory_id], "reset_tcp_quaternion_xyzw"),
        )
    continuity_payload["candidates"] = [
        {
            "trajectory_id": str(row["trajectory_id"]),
            "native_class": row["native_classification"]["native_class"],
            "family": row.get("family"),
            "continuity": continuity_by_id.get(str(row["trajectory_id"])),
        }
        for row in entries
        if str(row["trajectory_id"]) in continuity_by_id
    ]
    _write_json(args.output / "continuity" / "initial_continuity_analysis.json", continuity_payload)

    progressive_payload = {
        "schema": "STAGE16_PROGRESSIVE_RELOCATION_COHORT_V1",
        "source_manifest": str(args.stage15_manifest),
        "bound_inputs": str(args.bound_json),
        "reset_trace": str(args.reset_trace),
        "candidate_count": len(static_analyses),
        "selected_count": len(progressive_ids),
        "selected_ids": progressive_ids,
        "candidates": static_analyses,
        "selected": [progressive_results[trajectory_id] for trajectory_id in progressive_ids],
    }
    _write_json(args.output / "progressive_relocation" / "progressive_cohort.json", progressive_payload)

    first_eight_ids = [
        "E1_W0_00",
        "E1_W0_01",
        "E1_ang_0_00",
        "E1_curvature_1_00",
        "E1_coupling_1_00",
        progressive_ids[0],
        progressive_ids[1],
        "E1_boundary_00",
    ]
    suite_rows = build_native_suite(
        entries,
        progressive_ids,
        target_count=args.target_count,
        required_ids=first_eight_ids,
    )
    suite_cases = []
    suite_case_by_id = {}
    for row in suite_rows:
        trajectory_id = str(row["trajectory_id"])
        if trajectory_id in progressive_arrays:
            arrays = progressive_arrays[trajectory_id]
            path = progressive_dir / (trajectory_id + ".npz")
            source_kind = "DERIVED_PROGRESSIVE_RELOCATION"
            derived_provenance = progressive_results[trajectory_id]["lead_in"]
        else:
            bound_entry = bound_by_id[trajectory_id]
            raw = load_npz(Path(str(bound_entry["path"])))
            arrays = {
                "timestamp": _array(raw, "timestamp"),
                "position": _array(raw, "position") + _array(trace, "anchor_position")[None, :],
                "quaternion": _array(raw, "quaternion"),
            }
            path = Path(str(bound_entry["path"]))
            source_kind = "STAGE15_NATIVE_BOUND_REFERENCE"
            derived_provenance = None
        case_hash = trajectory_hash(
            arrays["timestamp"],
            arrays["position"],
            arrays["quaternion"],
            {
                "trajectory_id": trajectory_id,
                "source_kind": source_kind,
                "shape_sha256": row.get("shape_sha256"),
            },
        )
        duration_s = float(arrays["timestamp"][-1] - arrays["timestamp"][0])
        episode_binding_id = "stage16-reset-%d" % int(args.reset_seed)
        metadata = _case_metadata(
            row=row,
            case_hash=case_hash,
            path=path,
            duration_s=duration_s,
            reset_seed=args.reset_seed,
            episode_binding_id=episode_binding_id,
            immutable_hashes=immutable_hashes,
            runtime_version=args.runtime_version,
            simulator="IsaacGym/PhysX native WBC",
            asset="go2_x5.urdf",
            source_kind=source_kind,
            derived=derived_provenance,
        )
        case = make_case(
            row,
            trajectory_path=str(path),
            trajectory_hash=case_hash,
            reset_seed=args.reset_seed,
            reference_interval=(0.0, duration_s),
            episode_binding_id=episode_binding_id,
            metadata=metadata,
        )
        suite_cases.append(case)
        suite_case_by_id[trajectory_id] = case

    manifest_metadata = {
        "stage": "STAGE16",
        "reference_mode": "ALIGNED_NATIVE_CONTINUOUS_REFERENCE",
        "reset_seed": int(args.reset_seed),
        "source_manifest": str(args.stage15_manifest),
        "bound_inputs": str(args.bound_json),
        "progressive_cohort": str(args.output / "progressive_relocation" / "progressive_cohort.json"),
        "immutable_hashes": immutable_hashes,
        "runtime_version": args.runtime_version,
        "simulator": "IsaacGym/PhysX native WBC",
        "asset": "go2_x5.urdf",
        "private_artifact_policy": "node-local references only; no checkpoints, archive, or rollout NPZ are Git-tracked",
    }
    write_manifest(args.output / "benchmark" / "native_suite_manifest.json", suite_cases, metadata=manifest_metadata)

    missing = [trajectory_id for trajectory_id in first_eight_ids if trajectory_id not in suite_case_by_id]
    if missing:
        raise RuntimeError("first-eight case missing from suite: %s" % ", ".join(missing))
    write_manifest(
        args.output / "benchmark" / "first_eight_manifest.json",
        [suite_case_by_id[trajectory_id] for trajectory_id in first_eight_ids],
        metadata=dict(manifest_metadata, gate="FIRST_8_NATIVE_CONTINUOUS"),
    )

    immutable_payload = {
        "schema": "STAGE16_IMMUTABLE_INPUTS_V1",
        "paths": {key: str(path) for key, path in immutable_paths.items()},
        "sha256": immutable_hashes,
        "p1m_and_gr00t": "not used by Stage16 native benchmark preparation",
    }
    _write_json(args.output / "manifests" / "immutable_inputs.json", immutable_payload)
    _write_json(
        args.output / "manifests" / "selection_summary.json",
        {
            "native_class_counts": manifest.get("native_class_counts", {}),
            "static_candidate_count": len(static_analyses),
            "progressive_ids": progressive_ids,
            "native_suite_count": len(suite_cases),
            "first_eight_ids": first_eight_ids,
        },
    )
    print(
        json.dumps(
            {
                "static_candidate_count": len(static_analyses),
                "progressive_ids": progressive_ids,
                "native_suite_count": len(suite_cases),
                "first_eight_count": len(first_eight_ids),
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
