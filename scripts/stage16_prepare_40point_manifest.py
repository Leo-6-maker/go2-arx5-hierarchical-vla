#!/usr/bin/env python3
"""Create the matched Stage16 GR00T40 interface-simulation manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--continuous-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.continuous_manifest.read_text())
    payload = json.loads(json.dumps(source))
    payload["schema"] = "STAGE16_40POINT_MATCHED_MANIFEST_V1"
    payload["reference_mode"] = "GR00T40_RECONSTRUCTED_REFERENCE"
    payload["interface"] = {
        "input": "continuous trajectory",
        "sampling": "40 timestamped EEF points",
        "chunk_contract": "TrajectoryChunkV1",
        "adapter": "canonical GR00T-to-WBC adapter",
        "preview": "native 9-point preview",
        "controller": "native WBC",
        "actual_gr00t_called": False,
        "camera_used": False,
    }
    for case in payload.get("cases", []):
        case["reference_mode"] = "GR00T40_RECONSTRUCTED_REFERENCE"
        metadata = dict(case.get("metadata", {}))
        metadata["reference_mode"] = "GR00T40_RECONSTRUCTED_REFERENCE"
        metadata["continuous_reference_mode"] = "ALIGNED_NATIVE_CONTINUOUS_REFERENCE"
        metadata["interface"] = payload["interface"]
        case["metadata"] = metadata
    payload["source_continuous_manifest"] = str(args.continuous_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"case_count": len(payload.get("cases", [])), "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
