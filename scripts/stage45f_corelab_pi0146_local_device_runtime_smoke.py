#!/usr/bin/env python3
"""PI0146 B0/B2 read-only local device and artifact census.

This module deliberately does not import the Torch, GR00T, Transformers or a
simulator.  It records the host/device state that is observable at the
Corelab login node and turns already-approved manifest identities into a
plan for a later, explicitly released fixed-checkpoint smoke.  It never
constructs a model, performs CUDA work, opens a checkpoint, or resolves a
scratch path by guesswork.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import platform
import socket
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


SCHEMA = "STAGE45F_CORELAB_PI0146_LOCAL_DEVICE_RUNTIME_SMOKE_V1"
BASE_HEAD = "3cd8d11c211a344c09f08280af1105f4dbbac219"
CHECKPOINT_TREE = "2113a6a3b2a91233a42214dbb017c0136acde282fda2a418066f9317e78981a5"
DATASET_TREE = "2f3badd2019a68c6549a66f85b5486faf626078a3e438342f67aacc4edc95789"
STATS_SHA = "3b3a2d51e685a0e580de4ebdfa05abcbcddfb7dc33b8d3a659994cdcd1bb0337"
STATS_PAYLOAD_SHA = "752898989bb3308e663c4b6cf88b3cc1c139897a6860e6aed1d9d708d0f6fd75"
PROCESSOR_SHA = "85c1b4690ae090559e79a45193e598b65d6146eedf14750884da65e6d31032be"
DECODER_SHA = "2175381660a94c59e54388c01b8fd397bd11439fb1d016d5a2116c01d5158fae"
GROOT_COMMIT = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
COSMOS_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"
WINDOW_ID = "stage22_expert_red_target_train_a:expert-0000"

_GPU_QUERY = (
    "index,name,uuid,memory.total,memory.used,memory.free,driver_version"
)
_PROCESS_QUERY = "gpu_uuid,pid,process_name,used_memory"
_PACKAGES = ("torch", "transformers", "accelerate", "numpy", "safetensors")
_APPROVED_MANIFESTS = (
    "manifests/STAGE45D_D1_CORELAB_4080S_PREFLIGHT_PROFILE_V1.json",
    "manifests/STAGE45C_CHECKPOINT_RUNTIME_MAPPING_V1.json",
    "manifests/STAGE45C_TRAIN24_LATE_BINDING_CANDIDATE_V1.json",
)


class CensusError(RuntimeError):
    """A read-only census failure that must remain visible in the report."""


def _command(
    argv: Sequence[str],
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Run one bounded read-only command and return output without raising."""

    try:
        result = runner(
            list(argv),
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
        return {"argv": list(argv), "returncode": None, "stdout": "", "stderr": str(exc)}
    return {
        "argv": list(argv),
        "returncode": result.returncode,
        "stdout": result.stdout or "",
        "stderr": result.stderr or "",
    }


def _csv_rows(text: str, fields: Sequence[str]) -> list[dict[str, str]]:
    rows = []
    for values in csv.reader(text.splitlines(), skipinitialspace=True):
        if not values:
            continue
        rows.append({field: value.strip() for field, value in zip(fields, values)})
    return rows


def device_census(
    runner: Callable[..., Any] = subprocess.run,
    *,
    hostname: str | None = None,
) -> dict[str, Any]:
    """Collect only nvidia-smi inventory/process metadata; never CUDA compute."""

    inventory = _command(
        ["nvidia-smi", f"--query-gpu={_GPU_QUERY}", "--format=csv,noheader"], runner
    )
    processes = _command(
        ["nvidia-smi", f"--query-compute-apps={_PROCESS_QUERY}", "--format=csv,noheader"], runner
    )
    available = inventory["returncode"] == 0
    return {
        "hostname": hostname if hostname is not None else socket.gethostname(),
        "platform": platform.platform(),
        "nvidia_smi_available": available,
        "status": "READY_READ_ONLY" if available else "BLOCKED_DEVICE_UNAVAILABLE",
        "gpu_rows": _csv_rows(inventory["stdout"], _GPU_QUERY.split(",")) if available else [],
        "process_rows": _csv_rows(processes["stdout"], _PROCESS_QUERY.split(",")) if processes["returncode"] == 0 else [],
        "inventory_command": inventory,
        "process_command": processes,
        "cuda_compute_calls": 0,
        "model_forward_calls": 0,
    }


def package_metadata() -> dict[str, Any]:
    """Read installed distributions without importing any model/runtime module."""

    versions: dict[str, str | None] = {}
    for package in _PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"python": platform.python_version(), "distributions": versions}


def _walk_paths(value: Any, *, key: str = "") -> list[tuple[str, str]]:
    """Extract only explicitly labelled path fields from approved manifests."""

    found: list[tuple[str, str]] = []
    if isinstance(value, Mapping):
        for child_key, child in value.items():
            if child_key in {"path", "materialization_path", "runtime_path", "checkpoint_path"} and isinstance(child, str):
                found.append((f"{key}.{child_key}".strip("."), child))
            found.extend(_walk_paths(child, key=f"{key}.{child_key}".strip(".")))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_walk_paths(child, key=f"{key}[{index}]"))
    return found


def artifact_inventory(repo_root: Path) -> dict[str, Any]:
    """Check approved manifest files and their explicitly recorded paths only."""

    repo_root = Path(repo_root).resolve()
    manifest_records: list[dict[str, Any]] = []
    path_records: list[dict[str, Any]] = []
    for relative in _APPROVED_MANIFESTS:
        path = repo_root / relative
        record: dict[str, Any] = {"manifest": relative, "exists": path.is_file()}
        if path.is_file():
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
                record["schema"] = document.get("schema")
                manifest_records.append(record)
                for field, raw in _walk_paths(document):
                    candidate = Path(raw)
                    resolved = (repo_root / candidate).resolve() if not candidate.is_absolute() else candidate
                    path_records.append(
                        {
                            "manifest": relative,
                            "field": field,
                            "recorded_path": raw,
                            "exists": resolved.exists(),
                            "is_file": resolved.is_file(),
                            "is_dir": resolved.is_dir(),
                            "path_checked": str(resolved),
                            "source": "approved_manifest",
                        }
                    )
            except (OSError, json.JSONDecodeError) as exc:
                record["error"] = str(exc)
                manifest_records.append(record)
        else:
            manifest_records.append(record)
    return {
        "approved_manifests": manifest_records,
        "explicit_path_checks": path_records,
        "checkpoint": {"label": "FULL24@2000", "tree_sha256": CHECKPOINT_TREE, "bytes_present": False},
        "dataset": {"label": "FULL24_EXISTING_TRAIN", "tree_sha256": DATASET_TREE, "bytes_present": False},
        "stats": {"route": "FULL24_RUNTIME_V2", "sha256": STATS_SHA, "payload_sha256": STATS_PAYLOAD_SHA, "bytes_present": False},
        "historical_comparator": {
            "label": "STRUCTURED@1000",
            "tree_sha256": "30b14d36e47e08dfbb9d3b353d199971863d6e7f6ceedea2d478ef367d0a1137",
            "stats_sha256": "0553436da13b4500e896cf4ec06ef36aba8d230536595cdd219818d912567206",
            "runtime_materialization_status": "PENDING_VANDA_RECOVERY",
            "materialization_path": None,
        },
    }


def fixed_checkpoint_smoke_plan(artifacts: Mapping[str, Any]) -> dict[str, Any]:
    """Describe the smallest later inference qualification without running it."""

    return {
        "status": "PLAN_ONLY_REQUIRES_NEXT_PI_GATE",
        "authority_class": "engineering_reference_only",
        "checkpoint": {"label": "FULL24@2000", "tree_sha256": CHECKPOINT_TREE, "substitution_allowed": False},
        "cohort": {"split": "TRAIN_ONLY", "window_id": WINDOW_ID, "window_count": 1, "target_xyz_in_model_input": False},
        "runtime": {
            "groot_source_commit": GROOT_COMMIT,
            "cosmos_revision": COSMOS_REVISION,
            "processor_config_sha256": PROCESSOR_SHA,
            "stats_route": "FULL24_RUNTIME_V2",
            "stats_sha256": STATS_SHA,
            "stats_payload_sha256": STATS_PAYLOAD_SHA,
            "decoder_source_sha256": DECODER_SHA,
            "input_keys": ["body_rgb", "wrist_rgb", "state"],
            "body_rgb_shape": [96, 96, 3],
            "wrist_rgb_shape": [96, 96, 3],
            "state_v1_shape": [10],
            "language": "move the end effector to the red target",
        },
        "sampler": {
            "development_seed": 4242,
            "steps": 4,
            "timestep_ids": [0, 250, 500, 750],
            "noise_shape": [1, 40, 132],
            "same_noise_across_checkpoints": True,
            "official_public_sampler_called": False,
        },
        "planned_call_ceiling": {"conditioning_passes": 1, "action_head_calls": 4, "total_model_side_calls": 5},
        "predeclared_evidence": {
            "raw_before_derived": True,
            "finite_required": True,
            "checkpoint_sha_bound": True,
            "physical_decoder": "canonical StructuredActionV1 only",
            "tolerance_rule": "PI must freeze raw/physical tolerance before execution; no post-hoc threshold",
        },
        "availability": {
            "checkpoint_bytes": artifacts.get("checkpoint", {}).get("bytes_present", False),
            "dataset_bytes": artifacts.get("dataset", {}).get("bytes_present", False),
            "stats_bytes": artifacts.get("stats", {}).get("bytes_present", False),
            "missing_bytes_must_block": True,
        },
        "execution_boundary": {
            "model_construction": 0,
            "model_forward": 0,
            "cuda_compute": 0,
            "training": 0,
            "backward": 0,
            "optimizer_steps": 0,
            "pbs": 0,
            "Validation-E": 0,
            "Holdout-F": 0,
            "WBC": 0,
            "simulator": 0,
            "hardware": 0,
        },
    }


def build_report(repo_root: Path, *, runner: Callable[..., Any] = subprocess.run) -> dict[str, Any]:
    artifacts = artifact_inventory(repo_root)
    devices = device_census(runner)
    return {
        "schema": SCHEMA,
        "status": "Q0_READ_ONLY_CENSUS_COMPLETE",
        "gate": {"name": "PI_GATE_STATE_0146", "issue_comment": 5645700727, "branch_base": BASE_HEAD},
        "device_census": devices,
        "package_metadata": package_metadata(),
        "approved_artifact_inventory": artifacts,
        "fixed_checkpoint_smoke_plan": fixed_checkpoint_smoke_plan(artifacts),
        "execution_boundary": {
            "model_construction": 0,
            "model_forward": 0,
            "cuda_compute": 0,
            "training": 0,
            "backward": 0,
            "optimizer_steps": 0,
            "pbs": 0,
            "Validation-E": 0,
            "Holdout-F": 0,
            "ICL": 0,
            "new_data": 0,
            "WBC": 0,
            "simulator": 0,
            "hardware": 0,
            "real_h200_campaign_ingested": False,
        },
        "skips": [
            "B1 non-model CUDA smoke not executed: current Q0 instruction explicitly forbids CUDA computation.",
            "No remote 4080S/5080 host was queried; this census is local-host only and nvidia-smi is unavailable here.",
            "No checkpoint/statistics bytes were materialized or downloaded; recorded identities remain plan-bound.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.repo_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "device_status": report["device_census"]["status"], "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
