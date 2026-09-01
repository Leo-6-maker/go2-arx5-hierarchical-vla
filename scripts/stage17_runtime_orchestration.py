#!/usr/bin/env python3
"""Validate the provider/gate/binding/adapter orchestration on one good case.

The optional native-run flag invokes the existing native Stage16 launcher with
the same known-good benchmark file.  It never accepts an actual GR00T output
path, so this script cannot accidentally send the unadapted model output to
WBC.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from go2arx5_vla.adapters import GR00TToWBCAdapter, ReferenceAdapterConfig
from go2arx5_vla.contracts import EpisodeReferenceBindingV1, TrajectoryChunkV1
from go2arx5_vla.providers import BenchmarkTrajectoryProvider
from go2arx5_vla.runtime import ChunkLifecycleTracker
from go2arx5_vla.safety import TrajectorySafetyGateV1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-native-wbc", action="store_true")
    parser.add_argument("--native-source-root", type=Path)
    parser.add_argument("--native-arm", type=Path)
    parser.add_argument("--native-dog", type=Path)
    parser.add_argument("--native-output-dir", type=Path)
    parser.add_argument("--native-sim-device", default="cuda:0")
    args = parser.parse_args()

    with np.load(args.trajectory, allow_pickle=False) as data:
        timestamps = np.asarray(data["timestamp"], dtype=np.float64)
        positions = np.asarray(data["position"], dtype=np.float64)
        quaternions = np.asarray(data["quaternion"], dtype=np.float64)
    chunk = TrajectoryChunkV1(
        chunk_id="stage17-known-good-E1_W0_00",
        source="Stage16KnownGoodBenchmark",
        observation_timestamp=float(timestamps[0]),
        point_timestamps=timestamps,
        eef_position=positions,
        eef_orientation=quaternions,
        source_frame="NATIVE_WORLD_ALIGNED",
        tcp_semantic_id="x5_link6_gripper_center",
        validity_start=float(timestamps[0]),
        validity_end=float(timestamps[-1]),
        provenance={
            "trajectory_file": str(args.trajectory),
            "reference_mode": "ALIGNED_NATIVE_CONTINUOUS_REFERENCE",
            "actual_gr00t_called": False,
        },
    )
    provider = BenchmarkTrajectoryProvider([chunk])
    emitted = provider.next_chunk(float(timestamps[0]))
    if emitted is None:
        raise RuntimeError("benchmark provider did not emit the known-good chunk")
    gate = TrajectorySafetyGateV1().evaluate(
        emitted,
        now=float(timestamps[0]),
        achieved_position=positions[0],
        achieved_orientation=quaternions[0],
    )
    if gate.decision.value == "REJECT":
        raise RuntimeError(json.dumps(gate.as_dict()))
    binding = EpisodeReferenceBindingV1.from_external_chunk(
        emitted,
        episode_id="stage17-known-good-E1_W0_00",
    )
    bound = binding.bind_external_chunk(emitted)
    adapter = GR00TToWBCAdapter(
        ReferenceAdapterConfig(
            expected_source_frame="NATIVE_WORLD_ALIGNED",
            wbc_frame="NATIVE_WORLD_ALIGNED",
            tcp_semantic_id="x5_link6_gripper_center",
            stale_timeout_s=0.75,
        )
    )
    preview, gripper = adapter.adapt(bound, controller_timestamp=float(timestamps[0]))
    if preview.xyz_rot6d.shape != (9, 9) or not np.isfinite(preview.xyz_rot6d).all():
        raise RuntimeError("known-good WBC preview failed shape/finite validation")

    lifecycle = ChunkLifecycleTracker()
    receive_event = lifecycle.submit(emitted, receive_timestamp=float(timestamps[0] + 0.03))
    fresh_event = lifecycle.sample(execute_timestamp=float(timestamps[0] + 0.10))
    aging_event = lifecycle.sample(execute_timestamp=float(timestamps[0] + 0.40))
    stale_event = lifecycle.sample(execute_timestamp=float(timestamps[0] + 0.90))
    older = TrajectoryChunkV1(
        chunk_id="stage17-older-chunk",
        source=emitted.source,
        observation_timestamp=float(timestamps[0] - 1.0),
        point_timestamps=timestamps - 1.0,
        eef_position=positions,
        eef_orientation=quaternions,
        source_frame=emitted.source_frame,
        tcp_semantic_id=emitted.tcp_semantic_id,
        validity_start=float(timestamps[0] - 1.0),
        validity_end=float(timestamps[-1] - 1.0),
    )
    out_of_order_event = lifecycle.submit(older, receive_timestamp=float(timestamps[0] + 0.04))

    result = {
        "schema": "STAGE17_RUNTIME_ORCHESTRATION_V1",
        "trajectory_id": "E1_W0_00",
        "trajectory_path": str(args.trajectory),
        "actual_gr00t_called": False,
        "chain": [
            "BenchmarkTrajectoryProvider",
            "TrajectoryChunkV1",
            "TrajectorySafetyGateV1",
            "EpisodeReferenceBindingV1",
            "GR00TToWBCAdapter",
            "WBCPreviewV1",
            "native WBC optional known-good execution",
        ],
        "gate": gate.as_dict(),
        "binding": {
            "episode_id": binding.episode_id,
            "semantics": binding.binding_semantics,
            "position_preservation_m": float(np.max(np.linalg.norm(bound.eef_position - emitted.eef_position, axis=-1))),
        },
        "preview": {
            "shape": list(preview.xyz_rot6d.shape),
            "finite": bool(np.isfinite(preview.xyz_rot6d).all()),
            "gripper_parallel_channel": gripper is not None,
        },
        "chunk_lifecycle": {
            "receive": receive_event.as_dict(),
            "fresh": fresh_event.as_dict(),
            "aging": aging_event.as_dict(),
            "stale": stale_event.as_dict(),
            "out_of_order": out_of_order_event.as_dict(),
        },
        "native_wbc": {"requested": bool(args.run_native_wbc), "status": "NOT_REQUESTED"},
    }
    native_failure = False
    if args.run_native_wbc:
        required = (args.native_source_root, args.native_arm, args.native_dog, args.native_output_dir)
        if any(value is None for value in required):
            parser.error("--run-native-wbc requires source root, arm, dog, and output dir")
        args.native_output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(Path(__file__).with_name("native_wbc_benchmark.py")),
            "--source-root", str(args.native_source_root),
            "--arm", str(args.native_arm),
            "--dog", str(args.native_dog),
            "--reference-mode", "BENCHMARK_CONTINUOUS",
            "--trajectory", str(args.trajectory),
            "--output-dir", str(args.native_output_dir),
            "--seconds", "2.0",
            "--seed", "13",
            "--sim-device", str(args.native_sim_device),
        ]
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        (args.native_output_dir / "stage17_orchestration_stdout.log").write_text(completed.stdout + completed.stderr)
        result["native_wbc"] = {
            "requested": True,
            "status": "PASS" if completed.returncode == 0 else "FAIL",
            "returncode": completed.returncode,
            "output_dir": str(args.native_output_dir),
            "command_reference": "native_wbc_benchmark.py known-good benchmark only",
        }
        native_failure = completed.returncode != 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if native_failure:
        # Preserve the pure interface/lifecycle evidence even when an optional
        # launcher fails during its own environment or simulator startup.
        raise SystemExit(1)


if __name__ == "__main__":
    main()
