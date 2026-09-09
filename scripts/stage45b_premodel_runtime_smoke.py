#!/usr/bin/env python3
"""Non-scientific Stage45B runtime qualification.

This module deliberately stops at imports, provenance checks, and (for the
H200 mode) a one-element CUDA allocation.  It never loads BASE weights,
constructs a model, invokes a trainer, creates a checkpoint, or creates the
scientific output root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping


BRANCH = "stage45b/full24-fresh-base-execution"
BASE_TREE_SHA256 = "8717e322de3dd8c66e52bdb81b37badb0f235c518de6acd465532ed01569a6a4"
FULL24_TREE_SHA256 = "2f3badd2019a68c6549a66f85b5486faf626078a3e438342f67aacc4edc95789"
FULL24_FILE_COUNT = 85
FULL24_BYTES = 919661
RUNTIME_STATS_SCHEMA = "CORELAB_STAGE45B_FULL24_RUNTIME_PARQUET_STATS_V2"
RUNTIME_PROTOCOL_REVISION = "RUNTIME_PARQUET_PROVENANCE_V2"
RUNTIME_STATS_SHA256 = "RUNTIME_STATS_SHA_FILLED_BY_REFREEZE_MANIFEST"
RUNTIME_PAYLOAD_SHA256 = "RUNTIME_STATS_PAYLOAD_SHA_FILLED_BY_REFREEZE_MANIFEST"
B0_STATS_SHA256 = "49c7139dff09d5795e31205c7e72cb11e5801ce5f176572ebc1ec8d91f310c3d"
B0_PAYLOAD_SHA256 = "c31c818121ac01a402fc70e577db225ffb6f5d110649aef9dada94338c8a5032"
GROOT_SHA = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
COSMOS_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"
COSMOS_TREE_SHA256 = "1cfec77c944cba341ee5d7c31b4be6bb615582364bae48f18c5686e6ec23e8a9"
MODALITY_SHA256 = "e61f9429ecc21ceab51d9d26cadc9018691eb8b5c74d60aec4abf7faee16f031"
CONFIG_SHA256 = "9c3cd1d7bea603b95b1c777d8bcfebd61bd7eb5d86ca8ebbc3175f8141498a7c"
EXPECTED_REAL_PYTHON = "/app1/Hopper/shared/apps/python/3.12.3/bin/python3.12"
OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "PYTHONNOUSERSITE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_tree_entries(root: Path, prefix: str) -> list[dict[str, Any]]:
    root = Path(root)
    root_stat = root.lstat()
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise RuntimeError(f"tree root is not a non-symlink directory: {root}")
    entries: list[dict[str, Any]] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for dirname in sorted(dirnames):
            child = directory_path / dirname
            if child.is_symlink() or not stat.S_ISDIR(child.lstat().st_mode):
                raise RuntimeError(f"unsafe directory entry: {child}")
        for filename in sorted(filenames):
            child = directory_path / filename
            if child.is_symlink() or not stat.S_ISREG(child.lstat().st_mode):
                raise RuntimeError(f"unsafe file entry: {child}")
            entries.append(
                {
                    "path": f"{prefix}/{child.relative_to(root).as_posix()}",
                    "bytes": int(child.stat().st_size),
                    "sha256": sha256_file(child),
                }
            )
    return sorted(entries, key=lambda item: str(item["path"]))


def tree_digest(entries: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(
            f"{entry['path']}\0{entry['bytes']}\0{entry['sha256']}\n".encode("utf-8")
        )
    return digest.hexdigest()


def cosmos_tree_digest(root: Path) -> tuple[str, int]:
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"Cosmos cache root is not a non-symlink directory: {root}")
    entries: list[dict[str, Any]] = []

    def visit(directory: Path, relative: Path) -> None:
        for item in sorted(directory.iterdir(), key=lambda value: value.name):
            rel = (relative / item.name).as_posix()
            mode = item.lstat().st_mode
            if stat.S_ISDIR(mode):
                entries.append({"path": rel, "kind": "directory", "size_bytes": 0, "sha256": None})
                visit(item, relative / item.name)
            elif stat.S_ISLNK(mode):
                target = os.readlink(item)
                resolved = item.resolve(strict=True)
                try:
                    resolved_rel = resolved.relative_to(root).as_posix()
                except ValueError as exc:
                    raise RuntimeError(f"Cosmos symlink escapes root: {item}") from exc
                if not resolved.is_file():
                    raise RuntimeError(f"Cosmos symlink does not resolve to a file: {item}")
                entries.append(
                    {
                        "path": rel,
                        "kind": "symlink",
                        "target": target,
                        "resolved_path": resolved_rel,
                        "size_bytes": int(resolved.stat().st_size),
                        "sha256": sha256_file(resolved),
                    }
                )
            elif stat.S_ISREG(mode):
                entries.append(
                    {
                        "path": rel,
                        "kind": "regular_file",
                        "size_bytes": int(item.stat().st_size),
                        "sha256": sha256_file(item),
                    }
                )
            else:
                raise RuntimeError(f"unsupported Cosmos entry: {item}")

    visit(root, Path("."))
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda value: str(value["path"])):
        digest.update(
            "\t".join(
                (
                    str(entry["kind"]),
                    str(entry["path"]),
                    str(entry.get("size_bytes", 0)),
                    str(entry.get("sha256") or "-"),
                    str(entry.get("target") or "-"),
                    str(entry.get("resolved_path") or "-"),
                )
            ).encode("utf-8")
            + b"\n"
        )
    return digest.hexdigest(), len(entries)


def run_git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def regular_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} is not a regular file: {path}")


def import_chain(repo: Path, groot: Path) -> dict[str, str]:
    sys.path.insert(0, str(repo / "scripts"))
    sys.path.insert(0, str(repo / "src"))
    sys.path.insert(0, str(groot))
    from gr00t.configs.base_config import get_default_config
    from stage38_structured_train import normalize_loaded_modality_configs
    from go2arx5_vla.stage38_structured_training import install_structured_dataset_factory
    from gr00t.experiment.experiment import run

    return {
        "gr00t.configs.base_config.get_default_config": getattr(get_default_config, "__name__", "present"),
        "stage38_structured_train.normalize_loaded_modality_configs": getattr(
            normalize_loaded_modality_configs, "__name__", "present"
        ),
        "go2arx5_vla.stage38_structured_training.install_structured_dataset_factory": getattr(
            install_structured_dataset_factory, "__name__", "present"
        ),
        "gr00t.experiment.experiment.run": getattr(run, "__name__", "present"),
    }


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def qualify(args: argparse.Namespace) -> dict[str, Any]:
    repo = Path(args.repo).resolve()
    bundle_path = Path(args.bundle).resolve()
    base = Path(args.base_model_path).resolve()
    dataset = Path(args.dataset_dir).resolve()
    groot = Path(args.groot_source).resolve()
    cosmos = Path(args.cosmos_root).resolve()
    modality = Path(args.modality_config).resolve()
    config = Path(args.config_path).resolve()
    stats = Path(args.stats_artifact).resolve()
    evidence_path = Path(args.evidence_path).resolve()
    requested_python = Path(args.requested_python).resolve()
    expected_head = args.expected_head

    if run_git(repo, "branch", "--show-current") != BRANCH:
        raise RuntimeError("execution branch mismatch")
    if run_git(repo, "rev-parse", "HEAD") != expected_head:
        raise RuntimeError("execution HEAD mismatch")
    if run_git(repo, "status", "--porcelain=v1"):
        raise RuntimeError("execution checkout is dirty")
    if not requested_python.is_file() or not os.access(requested_python, os.X_OK):
        raise RuntimeError(f"requested Stage45B Python is not executable: {requested_python}")
    actual_python = Path(sys.executable)
    actual_realpath = str(actual_python.resolve())
    if actual_realpath != str(requested_python):
        raise RuntimeError(
            f"exact Python mismatch: requested={requested_python} actual={actual_python} realpath={actual_realpath}"
        )
    if actual_realpath != EXPECTED_REAL_PYTHON:
        raise RuntimeError(f"unexpected Python realpath: {actual_realpath}")

    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if bundle.get("schema") != "CORELAB_STAGE45B_FULL24_RUNTIME_PARQUET_V2_EXECUTION_MANIFEST_V1":
        raise RuntimeError("bundle schema mismatch")
    if bundle.get("repository", {}).get("branch") != BRANCH:
        raise RuntimeError("bundle branch mismatch")
    source_head = str(bundle.get("repository", {}).get("source_head", ""))
    if len(source_head) != 40:
        raise RuntimeError("bundle source_head is not a commit SHA")
    if source_head != expected_head:
        raise RuntimeError("bundle source_head is not the exact refrozen execution HEAD")
    for key in ("runner", "pbs", "runtime_qualification"):
        item = bundle["execution_files"][key]
        path = repo / str(item["path"])
        regular_file(path, f"execution file {key}")
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"execution file hash mismatch: {key}")

    base_entries = regular_tree_entries(base, "payload/GR00T-N1.7-3B")
    base_digest = tree_digest(base_entries)
    if base_digest != BASE_TREE_SHA256:
        raise RuntimeError(f"BASE tree mismatch: {base_digest}")
    full_entries = regular_tree_entries(dataset, "payload/lerobot_train")
    full_digest = tree_digest(full_entries)
    full_bytes = sum(int(item["bytes"]) for item in full_entries)
    if full_digest != FULL24_TREE_SHA256 or len(full_entries) != FULL24_FILE_COUNT or full_bytes != FULL24_BYTES:
        raise RuntimeError(f"FULL24 identity mismatch: {full_digest}, {len(full_entries)}, {full_bytes}")
    cosmos_digest, cosmos_count = cosmos_tree_digest(cosmos)
    revision_path = cosmos / "refs/main"
    regular_file(revision_path, "Cosmos refs/main")
    revision = revision_path.read_text(encoding="utf-8").strip()
    if revision != COSMOS_REVISION or cosmos_digest != COSMOS_TREE_SHA256:
        raise RuntimeError(f"Cosmos identity mismatch: revision={revision} tree={cosmos_digest}")
    groot_head = run_git(groot, "rev-parse", "HEAD")
    if groot_head != GROOT_SHA or run_git(groot, "status", "--porcelain=v1"):
        raise RuntimeError("GR00T identity mismatch")
    if sha256_file(modality) != MODALITY_SHA256:
        raise RuntimeError("modality identity mismatch")
    if sha256_file(config) != CONFIG_SHA256:
        raise RuntimeError("training config identity mismatch")
    stats_document = json.loads(stats.read_text(encoding="utf-8"))
    observed_stats_sha = sha256_file(stats)
    if observed_stats_sha != RUNTIME_STATS_SHA256:
        raise RuntimeError("runtime-parquet V2 statistics identity mismatch")
    if (
        stats_document.get("schema") != RUNTIME_STATS_SCHEMA
        or stats_document.get("pi_gate") != "PI_GATE_STATE_0083"
        or stats_document.get("protocol_revision") != RUNTIME_PROTOCOL_REVISION
        or stats_document.get("statistics_payload_sha256") != RUNTIME_PAYLOAD_SHA256
        or stats_document.get("canonical_label_source", {}).get("tree_sha256")
        != FULL24_TREE_SHA256
    ):
        raise RuntimeError("runtime-parquet V2 statistics contract mismatch")
    historical = bundle.get("historical_b0_stats")
    if (
        not isinstance(historical, Mapping)
        or historical.get("sha256") != B0_STATS_SHA256
        or historical.get("payload_sha256") != B0_PAYLOAD_SHA256
        or historical.get("execution_stats_authority") is not False
    ):
        raise RuntimeError("historical B0 binding mismatch")

    for name, expected in OFFLINE_ENV.items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"offline/runtime environment mismatch: {name}")
    if args.mode in {"cpu", "h200"} and "STAGE45B_OUTPUT_ROOT" in os.environ:
        raise RuntimeError("scientific output root leaked into non-scientific smoke")
    if args.mode == "scientific" and os.environ.get("STAGE45B_LABEL_ANCHOR_RECONCILIATION") != "1":
        raise RuntimeError("Stage45B label-anchor reconciliation is not enabled")
    scientific_root = Path(args.scientific_output_root).resolve() if args.scientific_output_root else None
    if args.mode == "scientific":
        if scientific_root is None or scientific_root.exists() or scientific_root.is_symlink():
            raise RuntimeError("scientific output root must be absent before model runner")
    if not evidence_path.parent.is_dir() or evidence_path.is_symlink():
        raise RuntimeError("smoke evidence path is not in a regular writable evidence root")
    if scientific_root is not None and evidence_path.is_relative_to(scientific_root):
        raise RuntimeError("smoke evidence path is inside scientific output root")

    torch_imported = False
    transformers_imported = False
    import_results: dict[str, str] = {}
    import torch

    torch_imported = True
    torch_version = str(torch.__version__)
    torch_cuda_build = str(torch.version.cuda)
    torch_file = str(Path(torch.__file__).resolve())
    import transformers

    transformers_imported = True
    transformers_version = str(getattr(transformers, "__version__", "unknown"))
    import_results = import_chain(repo, groot)
    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count())
    gpu_identity: dict[str, Any] | None = None
    cuda_probe: dict[str, Any] = {"allocation": False, "synchronization": False}
    if args.mode == "cpu":
        if cuda_available or device_count != 0:
            raise RuntimeError(f"CPU smoke observed CUDA: available={cuda_available} count={device_count}")
    else:
        if not cuda_available or device_count != 1:
            raise RuntimeError(f"H200 smoke requires one CUDA device: available={cuda_available} count={device_count}")
        name = str(torch.cuda.get_device_name(0))
        if "H200" not in name:
            raise RuntimeError(f"unexpected GPU name: {name}")
        props = torch.cuda.get_device_properties(0)
        probe = torch.zeros((1,), device="cuda", dtype=torch.float32)
        torch.cuda.synchronize()
        cuda_probe = {"allocation": True, "synchronization": True, "value": float(probe.item())}
        gpu_identity = {
            "name": name,
            "index": 0,
            "uuid": os.environ.get("CUDA_DEVICE_UUID"),
            "memory_total_bytes": int(getattr(props, "total_memory", 0)),
            "compute_capability": [int(props.major), int(props.minor)],
        }

    token = {
        "cpu": "VANDA_HOPPER_STAGE45B_PREMODEL_CPU_PBS_QUALIFIED",
        "h200": "VANDA_HOPPER_STAGE45B_PREMODEL_H200_RUNTIME_QUALIFIED",
        "scientific": "VANDA_STAGE45B_SCIENTIFIC_PREMODEL_RUNTIME_PASS",
    }[args.mode]
    return {
        "schema": "VANDA_STAGE45B_PREMODEL_RUNTIME_SMOKE_V1",
        "status": "PASS",
        "token": token,
        "mode": args.mode,
        "timestamp_utc": now_utc(),
        "host": subprocess.check_output(["hostname"], text=True).strip(),
        "pbs_jobid": os.environ.get("PBS_JOBID"),
        "execution_head": expected_head,
        "branch": BRANCH,
        "runtime": {
            "requested_interpreter": str(Path(args.requested_python)),
            "readlink_f": str(requested_python),
            "sys_executable": str(actual_python),
            "sys_executable_realpath": actual_realpath,
            "python_version": sys.version,
            "torch_import": torch_imported,
            "torch_version": torch_version,
            "torch_cuda_build": torch_cuda_build,
            "torch_file": torch_file,
            "transformers_import": transformers_imported,
            "transformers_version": transformers_version,
            "required_import_chain": import_results,
            "cuda_is_available": cuda_available,
            "cuda_device_count": device_count,
            "gpu": gpu_identity,
            "cuda_probe": cuda_probe,
        },
        "identities": {
            "base_tree_sha256": base_digest,
            "full24_tree_sha256": full_digest,
            "full24_regular_files": len(full_entries),
            "full24_bytes": full_bytes,
            "runtime_stats_sha256": observed_stats_sha,
            "runtime_stats_payload_sha256": stats_document["statistics_payload_sha256"],
            "historical_b0_stats_sha256": B0_STATS_SHA256,
            "historical_b0_payload_sha256": B0_PAYLOAD_SHA256,
            "protocol_revision": RUNTIME_PROTOCOL_REVISION,
            "groot_sha": groot_head,
            "cosmos_revision": revision,
            "cosmos_tree_sha256": cosmos_digest,
            "cosmos_entry_count": cosmos_count,
            "modality_sha256": sha256_file(modality),
            "config_sha256": sha256_file(config),
        },
        "scientific_exposure": {
            "model_construction": False,
            "forward": False,
            "backward": False,
            "optimizer": False,
            "trainer": False,
            "checkpoint_creation": False,
            "scientific_output_root_created": False,
            "validation_e_read": False,
            "holdout_f_read": False,
            "wbc": False,
            "simulator": False,
            "hardware": False,
            "new_data": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("cpu", "h200", "scientific"), required=True)
    parser.add_argument("--evidence-path", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--groot-source", type=Path, required=True)
    parser.add_argument("--cosmos-root", type=Path, required=True)
    parser.add_argument("--modality-config", type=Path, required=True)
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--stats-artifact", type=Path, required=True)
    parser.add_argument("--requested-python", default=os.environ.get("STAGE45B_PYTHON", ""))
    parser.add_argument("--scientific-output-root", type=Path)
    args = parser.parse_args()
    try:
        result = qualify(args)
    except Exception as exc:
        failure = {
            "schema": "VANDA_STAGE45B_PREMODEL_RUNTIME_SMOKE_V1",
            "status": "FAIL",
            "mode": args.mode,
            "timestamp_utc": now_utc(),
            "pbs_jobid": os.environ.get("PBS_JOBID"),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "scientific_exposure": {
                "model_construction": False,
                "forward": False,
                "backward": False,
                "optimizer": False,
                "trainer": False,
                "checkpoint_creation": False,
                "scientific_output_root_created": False,
            },
        }
        try:
            write_json(Path(args.evidence_path).resolve(), failure)
        except Exception:
            pass
        print(f"STAGE45B_PREMODEL_BLOCKED: {failure['error']}", file=sys.stderr)
        return 2
    write_json(Path(args.evidence_path).resolve(), result)
    print(result["token"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
