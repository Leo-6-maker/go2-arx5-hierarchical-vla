#!/usr/bin/env python3
"""Build four-case, seed-specific Stage16 perturbation manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from go2arx5_vla.benchmark.progressive_relocation import (
    load_npz,
    make_progressive_relocation,
    trajectory_hash,
)


DEFAULT_IDS = (
    "E1_W0_00",             # easy
    "E1_curvature_1_00",    # curvature
    "E2_static_0_0",        # progressive relocation
    "E1_boundary_00",        # boundary stress control
)


def _array(payload: Mapping[str, Any], key: str) -> np.ndarray:
    return np.asarray(payload[key], dtype=np.float64)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-manifest", type=Path, required=True)
    parser.add_argument("--bound-json", type=Path, required=True)
    parser.add_argument("--reset-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ids", nargs="*", default=list(DEFAULT_IDS))
    args = parser.parse_args()

    full = json.loads(args.full_manifest.read_text())
    full_cases = {str(case["trajectory_id"]): case for case in full["cases"]}
    bound = json.loads(args.bound_json.read_text())
    bound_cases = {str(case["trajectory_id"]): case for case in bound["entries"]}
    trace = load_npz(args.reset_trace)
    anchor = _array(trace, "anchor_position")
    cases = []
    provenance = []
    for trajectory_id in args.ids:
        if trajectory_id not in full_cases:
            raise KeyError("case is absent from full Stage16 manifest: %s" % trajectory_id)
        if trajectory_id not in bound_cases:
            raise KeyError("case is absent from seed bound inputs: %s" % trajectory_id)
        base_case = json.loads(json.dumps(full_cases[trajectory_id]))
        bound_entry = bound_cases[trajectory_id]
        raw = load_npz(Path(str(bound_entry["path"])))
        timestamp = _array(raw, "timestamp")
        position = _array(raw, "position") + anchor[None, :]
        quaternion = _array(raw, "quaternion")
        is_progressive = trajectory_id == "E2_static_0_0"
        derived_meta = None
        if is_progressive:
            derived, derived_meta = make_progressive_relocation(
                timestamp,
                position,
                quaternion,
                _array(bound_entry, "reset_tcp_position_world"),
                _array(bound_entry, "reset_tcp_quaternion_xyzw"),
                parent_id=trajectory_id,
                parent_hash=trajectory_hash(
                    timestamp,
                    position,
                    quaternion,
                    {"trajectory_id": trajectory_id, "source_kind": "STAGE16_SEED_BOUND_REFERENCE"},
                ),
            )
            timestamp = derived["timestamp"]
            position = derived["position"]
            quaternion = derived["quaternion"]
            output_path = args.output / "progressive_relocation" / (trajectory_id + ".npz")
            source_kind = "DERIVED_PROGRESSIVE_RELOCATION"
        else:
            output_path = Path(str(bound_entry["path"]))
            source_kind = "STAGE16_SEED_BOUND_REFERENCE"
        if is_progressive:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                output_path,
                timestamp=timestamp,
                position=position - anchor[None, :],
                quaternion=quaternion,
            )
        duration_s = float(timestamp[-1] - timestamp[0])
        case_hash = trajectory_hash(
            timestamp,
            position,
            quaternion,
            {"trajectory_id": trajectory_id, "source_kind": source_kind},
        )
        binding_id = "stage16-reset-%d" % int(args.seed)
        base_case["reset_seed"] = int(args.seed)
        base_case["episode_binding_id"] = binding_id
        base_case["reference_interval"] = [0.0, duration_s]
        base_case["trajectory_path"] = str(output_path)
        base_case["trajectory_hash"] = case_hash
        metadata = dict(base_case.get("metadata", {}))
        metadata.update(
            {
                "trajectory_id": trajectory_id,
                "trajectory_hash": case_hash,
                "reset_seed": int(args.seed),
                "episode_binding_id": binding_id,
                "reference_interval": [0.0, duration_s],
                "reference_mode": "ALIGNED_NATIVE_CONTINUOUS_REFERENCE",
                "trajectory_path": str(output_path),
                "source_kind": source_kind,
            }
        )
        if derived_meta is not None:
            metadata["derived_provenance"] = derived_meta
        base_case["metadata"] = metadata
        for key in (
            "trajectory_id",
            "trajectory_hash",
            "reset_seed",
            "episode_binding_id",
            "reference_interval",
            "reference_mode",
            "trajectory_path",
        ):
            if key in metadata:
                base_case[key] = metadata[key]
        cases.append(base_case)
        provenance.append(
            {
                "trajectory_id": trajectory_id,
                "seed": int(args.seed),
                "source_bound_input": str(bound_entry["path"]),
                "output_path": str(output_path),
                "source_kind": source_kind,
                "trajectory_hash": case_hash,
                "derived_provenance": derived_meta,
            }
        )

    payload = {
        "schema": "STAGE16_REPRESENTATIVE_PERTURBATION_MANIFEST_V1",
        "reference_mode": "ALIGNED_NATIVE_CONTINUOUS_REFERENCE",
        "reset_seed": int(args.seed),
        "source_full_manifest": str(args.full_manifest),
        "source_bound_inputs": str(args.bound_json),
        "source_reset_trace": str(args.reset_trace),
        "case_count": len(cases),
        "cases": cases,
    }
    _write_json(args.output / "representative_manifest.json", payload)
    _write_json(
        args.output / "representative_provenance.json",
        {"schema": "STAGE16_REPRESENTATIVE_PERTURBATION_PROVENANCE_V1", "cases": provenance},
    )
    print(json.dumps({"seed": args.seed, "case_count": len(cases), "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
