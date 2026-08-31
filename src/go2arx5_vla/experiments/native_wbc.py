"""Native WBC experiment manifest and invocation helpers.

The actual IsaacGym process remains a node-local launcher because the source,
checkpoints, and assets are private.  This module defines the stable metadata
boundary used by that launcher and by the analysis/reporting scripts.
"""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


REQUIRED_ROLLOUT_METADATA = (
    "trajectory_id",
    "trajectory_hash",
    "shape_id",
    "placement_id",
    "episode_binding_id",
    "reset_seed",
    "reference_interval",
    "archive_sha256",
    "arm_checkpoint_sha256",
    "dog_checkpoint_sha256",
    "config_sha256",
    "runtime_version",
    "simulator",
    "asset",
    "reference_mode",
)


@dataclass(frozen=True)
class NativeBenchmarkCase:
    """One immutable reference plus one native reset binding."""

    trajectory_id: str
    trajectory_hash: str
    shape_id: str
    placement_id: str
    episode_binding_id: str
    reset_seed: int
    reference_interval: tuple[float, float]
    reference_mode: str
    trajectory_path: str
    cohort: str
    native_class: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        result = {
            "trajectory_id": self.trajectory_id,
            "trajectory_hash": self.trajectory_hash,
            "shape_id": self.shape_id,
            "placement_id": self.placement_id,
            "episode_binding_id": self.episode_binding_id,
            "reset_seed": int(self.reset_seed),
            "reference_interval": list(self.reference_interval),
            "reference_mode": self.reference_mode,
            "trajectory_path": self.trajectory_path,
            "cohort": self.cohort,
            "native_class": self.native_class,
            "metadata": dict(self.metadata),
        }
        for key in REQUIRED_ROLLOUT_METADATA:
            if key not in result and key in self.metadata:
                result[key] = self.metadata[key]
        missing = [key for key in REQUIRED_ROLLOUT_METADATA if key not in result]
        if missing:
            raise ValueError("benchmark case metadata missing: %s" % ", ".join(missing))
        return result


def make_case(
    row: Mapping[str, Any],
    *,
    trajectory_path: str,
    trajectory_hash: str,
    reset_seed: int,
    reference_interval: Sequence[float],
    episode_binding_id: Optional[str] = None,
    reference_mode: str = "ALIGNED_NATIVE_CONTINUOUS_REFERENCE",
    metadata: Optional[Mapping[str, Any]] = None,
) -> NativeBenchmarkCase:
    """Convert a suite row into the explicit rollout metadata contract."""

    placement = row.get("placement_profile", {})
    placement_id = str(row.get("placement_id") or placement.get("profile_id") or "UNSPECIFIED")
    native_class = str(row.get("native_class") or row.get("native_classification", {}).get("native_class") or "UNKNOWN")
    return NativeBenchmarkCase(
        trajectory_id=str(row["trajectory_id"]),
        trajectory_hash=str(trajectory_hash),
        shape_id=str(row.get("shape_id") or row["trajectory_id"]),
        placement_id=placement_id,
        episode_binding_id=str(episode_binding_id or ("stage16-reset-%s" % reset_seed)),
        reset_seed=int(reset_seed),
        reference_interval=(float(reference_interval[0]), float(reference_interval[1])),
        reference_mode=reference_mode,
        trajectory_path=str(trajectory_path),
        cohort=str(row.get("cohort", "UNSPECIFIED")),
        native_class=native_class,
        metadata={} if metadata is None else dict(metadata),
    )


def write_manifest(path: Path, cases: Sequence[NativeBenchmarkCase], *, metadata: Optional[Mapping[str, Any]] = None) -> None:
    payload = {
        "schema": "STAGE16_NATIVE_BENCHMARK_MANIFEST_V1",
        "reference_mode": "ALIGNED_NATIVE_CONTINUOUS_REFERENCE",
        "case_count": len(cases),
        "metadata": {} if metadata is None else dict(metadata),
        "cases": [case.as_dict() for case in cases],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def launcher_command(
    *,
    launcher: Path,
    source_root: Path,
    arm_checkpoint: Path,
    dog_checkpoint: Path,
    trajectory: Path,
    output_dir: Path,
    seconds: float,
    seed: int,
    device: str = "cuda:0",
    reference_mode: str = "BENCHMARK_CONTINUOUS",
) -> str:
    """Return a shell-safe, inspection-friendly native launcher command."""

    args = [
        str(launcher),
        "--source-root", str(source_root),
        "--arm", str(arm_checkpoint),
        "--dog", str(dog_checkpoint),
        "--reference-mode", reference_mode,
        "--trajectory", str(trajectory),
        "--output-dir", str(output_dir),
        "--seconds", str(float(seconds)),
        "--seed", str(int(seed)),
        "--sim-device", device,
    ]
    return " ".join(shlex.quote(value) for value in args)
