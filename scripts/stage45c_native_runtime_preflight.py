#!/usr/bin/env python3
"""Stage45C compute-native preflight using the accepted Stage45B contract.

This is execution plumbing only.  It imports the immutable Stage45B native
runtime verifier, verifies the exact FFmpeg/Torch/TorchCodec link contract, and
then exercises the Stage45C evaluation input path without constructing a
model.  It never trains, performs a forward pass, writes a checkpoint, or
submits PBS work.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import traceback
from typing import Any, Mapping


EXPECTED_STAGE45B_HELPER_SHA256 = (
    "b1d610c02e5c7dde9d90f366916a2c2c5204f89febeb548e2324154c6288b956"
)
EXPECTED_STAGE45B_SNAPSHOT_VERIFIER_SHA256 = (
    "88cca0085143e14826bc8bf4390950fa9577e14830cea0ca770f4de847611712"
)
EXPECTED_GROOT_MANIFEST_FILE_SHA256 = (
    "96b883e56b54f763bec60d3a336632ad531f4fc52d099f803e5f9efba60eab11"
)
EXPECTED_GROOT_MANIFEST_PAYLOAD_SHA256 = (
    "a9f10b837b6086137fa519cc148275a25bf6a5f98d58df8e3c50e0a43e5b8250"
)
EXPECTED_GROOT_PAYLOAD_TREE_SHA256 = (
    "c0d57a6a74ab24c48abdd1c13420e9d1c893dc0d7b5ca370fa9b6f7af32f5f1b"
)
EXPECTED_GROOT_ARCHIVE_SHA256 = (
    "705681fece08cb8a7c1963d4531464c90b4b031f2ab5aa2c8e25e74676310939"
)
EXPECTED_PYTHON_REALPATH = "/app1/Hopper/shared/apps/python/3.12.3/bin/python3.12"
EXPECTED_TORCH = "2.9.0+cu128"
EXPECTED_CUDA_BUILD = "12.8"
EXPECTED_TORCHCODEC = "0.8.0"
EXPECTED_TORCH_TREE_SHA256 = (
    "3d13fc0b2c9734fdc765a3f6dc93222fbfee1e1de4781e6c4335832e6b9c3b89"
)
EXPECTED_TORCHCODEC_TREE_SHA256 = (
    "9368f12938dc12d0c630d2de3e8282655e1ad6b93a7710b38d3707c02186e6b2"
)
EXPECTED_STAGE45C_TOKEN = "STAGE45C_GOLDEN_NATIVE_RUNTIME_TRANSPLANT_QUALIFIED"
EXPECTED_SCHEMA = "STAGE45C_GOLDEN_NATIVE_RUNTIME_TRANSPLANT_QUALIFIED_V1"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def regular_file(path: Path, label: str) -> None:
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not stat.S_ISLNK(mode), f"{label} is not a regular file: {path}")


def regular_directory(path: Path, label: str) -> None:
    mode = path.lstat().st_mode
    require(stat.S_ISDIR(mode) and not stat.S_ISLNK(mode), f"{label} is not a regular directory: {path}")


def load_exact_module(name: str, path: Path, expected_sha256: str) -> Any:
    regular_file(path, name)
    observed = sha256_file(path)
    require(observed == expected_sha256, f"{name} SHA256 mismatch: {observed}")
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"cannot load {name}: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return value.tolist()
        except Exception:
            pass
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def module_source_records(root: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    for name, module in sorted(sys.modules.items()):
        if name != "gr00t" and not name.startswith("gr00t."):
            continue
        source = getattr(module, "__file__", None)
        if source is None:
            continue
        resolved = Path(source).resolve()
        require(resolved.is_relative_to(root), f"GR00T import escaped sealed source root: {name} -> {resolved}")
        records[name] = str(resolved)
    require(records, "no GR00T module source files were observed")
    return records


def tensor_summary(value: Any, torch: Any) -> Any:
    if hasattr(value, "shape"):
        finite = None
        try:
            finite = bool(torch.isfinite(value).all().item())
        except Exception:
            pass
        return {
            "type": type(value).__name__,
            "shape": [int(item) for item in value.shape],
            "dtype": str(getattr(value, "dtype", "")),
            "finite": finite,
        }
    if isinstance(value, Mapping):
        return {str(key): tensor_summary(item, torch) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [tensor_summary(item, torch) for item in value[:4]]
    return type(value).__name__


def qualify(args: argparse.Namespace) -> dict[str, Any]:
    repo = Path(args.repo).resolve()
    protocol = Path(args.protocol).resolve()
    mapping = Path(args.mapping).resolve()
    binding_path = Path(args.runtime_binding).resolve()
    runtime_root = Path(args.ffmpeg_runtime_root).resolve()
    torchcodec_root_arg = Path(args.torchcodec_package_root).resolve()
    torch_library_root_arg = Path(args.torch_library_root).resolve()
    config_path = Path(args.config).resolve()
    output_root_arg = Path(args.output_root).resolve()
    evidence_path = Path(args.evidence).resolve()
    qualification_root = Path(args.qualification_root).resolve()
    helper_path = Path(args.stage45b_helper).resolve()
    snapshot_verifier_path = Path(args.stage45b_snapshot_verifier).resolve()
    groot_archive_path = Path(args.groot_snapshot_archive).resolve()
    requested_python = Path(args.requested_python).resolve()

    regular_directory(repo, "Stage45C repository")
    regular_file(protocol, "frozen protocol")
    regular_file(mapping, "checkpoint mapping")
    regular_file(binding_path, "runtime binding")
    regular_directory(runtime_root, "sealed FFmpeg runtime")
    regular_directory(torchcodec_root_arg, "TorchCodec package root")
    regular_directory(torch_library_root_arg, "Torch library root")
    regular_file(config_path, "frozen training config")
    require(sha256_file(config_path) == "9c3cd1d7bea603b95b1c777d8bcfebd61bd7eb5d86ca8ebbc3175f8141498a7c", "frozen training config bytes changed")
    require(evidence_path.parent.is_dir() and not evidence_path.exists() and not evidence_path.is_symlink(), "preflight evidence path must be new")
    require(not qualification_root.exists() and not qualification_root.is_symlink(), "preflight qualification root must be absent")
    require(Path(sys.executable).resolve() == requested_python, "preflight requested Python is not the executing Python")
    require(requested_python == Path(EXPECTED_PYTHON_REALPATH), f"exact Python realpath mismatch: {requested_python}")

    def git(*git_args: str) -> str:
        result = subprocess.run(["git", "-C", str(repo), *git_args], capture_output=True, text=True, check=False)
        require(result.returncode == 0, f"git failed: {git_args}: {result.stderr.strip()}")
        return result.stdout.strip()

    execution_head = git("rev-parse", "HEAD")
    branch = git("branch", "--show-current")
    require(branch == "stage45c/b2-checkpoint-train24-evaluation", f"Stage45C branch mismatch: {branch}")
    require(execution_head == args.expected_head, f"Stage45C HEAD mismatch: {execution_head}")
    require(git("status", "--porcelain=v1") == "", "Stage45C worktree is dirty")

    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    require(isinstance(binding, Mapping), "runtime binding must be an object")
    bound_output_root = Path(binding.get("paths", {}).get("output_root", "")).resolve()
    require(bound_output_root == output_root_arg, "submission output root does not match runtime binding")
    native_identity = binding.get("native_runtime", {})
    if native_identity:
        require(native_identity.get("ffmpeg_runtime_root") == str(runtime_root), "binding FFmpeg runtime root mismatch")
        require(native_identity.get("ffmpeg_runtime_id") == args.ffmpeg_runtime_id, "binding FFmpeg runtime ID mismatch")
        require(native_identity.get("ffmpeg_manifest_sha256") == args.ffmpeg_manifest_sha256, "binding FFmpeg manifest SHA mismatch")
        require(native_identity.get("torchcodec_package_root") == str(torchcodec_root_arg), "binding TorchCodec root mismatch")
        require(native_identity.get("torch_library_root") == str(torch_library_root_arg), "binding Torch library root mismatch")
        require(native_identity.get("stage45b_helper_path") == str(helper_path), "binding Stage45B helper path mismatch")
        require(native_identity.get("stage45b_helper_sha256") == EXPECTED_STAGE45B_HELPER_SHA256, "binding Stage45B helper SHA mismatch")
        require(native_identity.get("stage45b_snapshot_verifier_path") == str(snapshot_verifier_path), "binding snapshot verifier path mismatch")
        require(native_identity.get("stage45b_snapshot_verifier_sha256") == EXPECTED_STAGE45B_SNAPSHOT_VERIFIER_SHA256, "binding snapshot verifier SHA mismatch")
        require(native_identity.get("groot_snapshot_archive_path") == str(groot_archive_path), "binding GR00T snapshot archive path mismatch")
        require(native_identity.get("groot_snapshot_archive_sha256") == EXPECTED_GROOT_ARCHIVE_SHA256, "binding GR00T snapshot archive SHA mismatch")
        require(native_identity.get("groot_manifest_path") == str(binding.get("paths", {}).get("groot_runtime_manifest", "")), "binding GR00T manifest path mismatch")
        require(native_identity.get("groot_manifest_file_sha256") == EXPECTED_GROOT_MANIFEST_FILE_SHA256, "binding GR00T manifest file SHA mismatch")
        require(native_identity.get("groot_manifest_payload_sha256") == EXPECTED_GROOT_MANIFEST_PAYLOAD_SHA256, "binding GR00T manifest payload SHA mismatch")
        require(native_identity.get("groot_payload_tree_sha256") == EXPECTED_GROOT_PAYLOAD_TREE_SHA256, "binding GR00T payload tree SHA mismatch")
        require(native_identity.get("stage45c_native_preflight_path") == str(Path(__file__).resolve()), "binding native preflight path mismatch")
        require(native_identity.get("stage45c_native_preflight_sha256") == sha256_file(Path(__file__).resolve()), "binding native preflight SHA mismatch")
        require(native_identity.get("config_path") == str(config_path), "binding config path mismatch")
        require(native_identity.get("config_sha256") == sha256_file(config_path), "binding config SHA mismatch")
        qualification_path = Path(str(native_identity.get("video_runtime_qualification_path", ""))).resolve()
        regular_file(qualification_path, "login-plane video qualification")
        require(native_identity.get("video_runtime_qualification_sha256") == sha256_file(qualification_path), "binding video qualification SHA mismatch")

    adapter_path = repo / "scripts" / "stage45c_h200_eval_runtime_adapter.py"
    adapter_sha = sha256_file(adapter_path)
    bound_adapter = binding.get("execution_code_identity", {}).get("adapter", {})
    require(bound_adapter.get("sha256") == adapter_sha, "runtime binding adapter identity does not match checkout")
    adapter = load_exact_module("stage45c_pi0094_adapter", adapter_path, adapter_sha)
    stage26_path = repo / "scripts" / "stage26_h1_c1_fixed_noise_local_flow.py"
    stage26_sha = sha256_file(stage26_path)
    bound_stage26 = binding.get("execution_code_identity", {}).get("stage26", {})
    if bound_stage26:
        require(bound_stage26.get("sha256") == stage26_sha, "runtime binding Stage26 identity does not match checkout")
    stage26 = load_exact_module("stage45c_pi0094_stage26", stage26_path, stage26_sha)
    mapping_summary = adapter.audit_protocol_and_mapping(protocol, mapping)
    adapter.validate_runtime_binding(binding, mapping_summary, require_paths=True)
    adapter_preflight = adapter.runtime_preflight(binding, mapping_summary)

    helper = load_exact_module("stage45b_pi0094_native_helper", helper_path, EXPECTED_STAGE45B_HELPER_SHA256)
    artifact = helper.verify_runtime_artifact(runtime_root, expected_id=args.ffmpeg_runtime_id)
    environment = helper.verify_environment(runtime_root)
    ld_parts = [part for part in os.environ.get("LD_LIBRARY_PATH", "").split(":") if part]
    expected_ld_prefix = [
        str(runtime_root / "lib"),
        str(runtime_root / "compat"),
        str(torchcodec_root_arg),
        str(torch_library_root_arg),
    ]
    require(ld_parts[:4] == expected_ld_prefix, f"native LD_LIBRARY_PATH prefix mismatch: {ld_parts[:4]}")
    require(not os.environ.get("PYTHONPATH"), "ambient PYTHONPATH is present")
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "PYTHONNOUSERSITE", "PYTHONDONTWRITEBYTECODE"):
        require(os.environ.get(name) == "1", f"offline/native environment mismatch: {name}")

    import torch
    import torchcodec
    from torchcodec._core.ops import _get_json_ffmpeg_library_versions

    require(str(torch.__version__) == EXPECTED_TORCH, f"Torch version mismatch: {torch.__version__}")
    require(str(torch.version.cuda) == EXPECTED_CUDA_BUILD, f"Torch CUDA build mismatch: {torch.version.cuda}")
    torch_root = Path(torch.__file__).resolve().parent
    torch_library_root = (torch_root / "lib").resolve()
    require(torch_library_root == torch_library_root_arg, f"Torch library root mismatch: {torch_library_root}")
    torch_identity = helper.package_identity(torch_root)
    require(torch_identity["tree_sha256"] == EXPECTED_TORCH_TREE_SHA256, "Torch package bytes changed")

    torchcodec_version = __import__("importlib.metadata", fromlist=["version"]).version("torchcodec")
    require(str(torchcodec_version) == EXPECTED_TORCHCODEC, f"TorchCodec version mismatch: {torchcodec_version}")
    torchcodec_root = Path(torchcodec.__file__).resolve().parent
    require(torchcodec_root == torchcodec_root_arg, f"TorchCodec package root mismatch: {torchcodec_root}")
    torchcodec_identity = helper.package_identity(torchcodec_root)
    require(torchcodec_identity["tree_sha256"] == EXPECTED_TORCHCODEC_TREE_SHA256, "TorchCodec package bytes changed")

    native_bindings = helper.native_binding_record(artifact, torchcodec_root)
    ffmpeg_versions = json.loads(_get_json_ffmpeg_library_versions())
    require(ffmpeg_versions.get("ffmpeg_version") == "6.1.1", f"native FFmpeg version mismatch: {ffmpeg_versions}")
    require(ffmpeg_versions.get("libavcodec") == [60, 31, 102], "native libavcodec version mismatch")
    require(ffmpeg_versions.get("libavformat") == [60, 16, 100], "native libavformat version mismatch")
    require(ffmpeg_versions.get("libavutil") == [58, 29, 100], "native libavutil version mismatch")
    require(ffmpeg_versions.get("libavfilter") == [9, 12, 100], "native libavfilter version mismatch")
    maps_after_native_load = helper.mapped_libav_paths(runtime_root)
    native_ldd_libav = {
        str(binding_item["resolved_path"])
        for record in native_bindings["ldd"]
        for binding_item in record["bindings"]
        if str(binding_item["name"]).startswith(("libav", "libsw"))
    }
    require(native_ldd_libav.issubset(set(maps_after_native_load["sealed_runtime"])), "native libav/libsw mappings escaped sealed runtime")

    dataset = Path(adapter_preflight["binding"]["paths"]["full24_dataset"]).resolve()
    groot = Path(adapter_preflight["binding"]["paths"]["groot_source"]).resolve()
    require(not (groot / ".git").exists(), "GR00T source is not gitless")
    snapshot_verifier = load_exact_module(
        "stage45b_pi0094_snapshot_verifier",
        snapshot_verifier_path,
        EXPECTED_STAGE45B_SNAPSHOT_VERIFIER_SHA256,
    )
    groot_manifest = Path(adapter_preflight["binding"]["paths"]["groot_runtime_manifest"]).resolve()
    groot_archive_verification = snapshot_verifier.verify_snapshot(
        groot_archive_path,
        groot_manifest,
        expected_archive_sha256=EXPECTED_GROOT_ARCHIVE_SHA256,
        expected_manifest_file_sha256=EXPECTED_GROOT_MANIFEST_FILE_SHA256,
        expected_manifest_payload_sha256=EXPECTED_GROOT_MANIFEST_PAYLOAD_SHA256,
        expected_payload_tree_sha256=EXPECTED_GROOT_PAYLOAD_TREE_SHA256,
    )
    groot_tree_verification = snapshot_verifier.verify_extracted(
        groot,
        groot_manifest,
        expected_manifest_file_sha256=EXPECTED_GROOT_MANIFEST_FILE_SHA256,
        expected_manifest_payload_sha256=EXPECTED_GROOT_MANIFEST_PAYLOAD_SHA256,
        expected_payload_tree_sha256=EXPECTED_GROOT_PAYLOAD_TREE_SHA256,
    )
    video_sweep = helper.sweep_all_videos(dataset, runtime_root)

    scripts_root = repo / "scripts"
    src_root = repo / "src"
    for item in (scripts_root, src_root, groot):
        if str(item) not in sys.path:
            sys.path.insert(0, str(item))
    processor, embodiment, modality_configs = stage26._build_runtime_processor(
        groot_source=groot,
        base_checkpoint=Path(adapter_preflight["binding"]["paths"]["base_checkpoint"]),
        dataset_dir=dataset,
        modality_path=Path(adapter_preflight["binding"]["paths"]["modality_config"]),
        cosmos_snapshot=Path(adapter_preflight["binding"]["paths"]["cosmos_root"]) / "snapshots" / adapter.COSMOS_REVISION,
    )

    import gr00t.data.dataset.lerobot_episode_loader as loader_module

    decode_calls: list[dict[str, Any]] = []
    original_decode = loader_module.get_frames_by_indices

    def traced_decode(video_path: str, indices: Any, decoder_kwargs: dict | None = None) -> Any:
        index_list = [int(item) for item in indices]
        value = original_decode(video_path, index_list, decoder_kwargs=decoder_kwargs)
        decode_calls.append({
            "path": str(Path(video_path).resolve()),
            "indices": index_list,
            "count": len(index_list),
            "decoder_kwargs": dict(decoder_kwargs or {}),
            "decoded_shape": list(value.shape),
            "decoded_dtype": str(value.dtype),
        })
        return value

    loader_module.get_frames_by_indices = traced_decode
    try:
        windows = adapter._prepare_train24_windows(
            processor,
            embodiment,
            modality_configs,
            dataset,
            adapter_preflight["snapshots"],
        )
    finally:
        loader_module.get_frames_by_indices = original_decode

    require(len(windows) == 24, f"Stage45C evaluator loader window count drift: {len(windows)}")
    # The frozen processor's tensor ABI packs the two RGB streams into
    # ``pixel_values``; the source loader still exposes/decodes the two named
    # modalities independently.  Do not invent body_rgb/wrist_rgb tensor keys.
    logical_keys = {"pixel_values", "state"}
    window_summaries = []
    for window in windows:
        model_inputs = window["model_inputs"]
        require(logical_keys.issubset(set(model_inputs)), f"Stage45C loader missing RGB/state inputs: {sorted(model_inputs)}")
        require(window["target_xyz_in_model_input"] is False, "numeric target entered Stage45C model inputs")
        window_summaries.append({
            "window_id": window["window_id"],
            "loader_index": window["loader_index"],
            "logical_model_input_keys": window["logical_model_input_keys"],
            "model_inputs": tensor_summary(model_inputs, torch),
        })
    require({item["loader_index"] for item in window_summaries} == set(range(24)), "Stage45C loader did not consume all 24 samples")
    observed_loader_paths = {item["path"] for item in decode_calls}
    observed_loader_modalities = {
        "external_rgb" if "observation.images.external_rgb" in path else "wrist_rgb" if "observation.images.wrist_rgb" in path else "other"
        for path in observed_loader_paths
    }
    require(
        {"external_rgb", "wrist_rgb"}.issubset(observed_loader_modalities),
        f"Stage45C evaluator loader did not consume both RGB modalities: observed={sorted(observed_loader_modalities)} paths={sorted(observed_loader_paths)[:4]}",
    )
    require(
        decode_calls and all(item["indices"] == list(range(40)) for item in decode_calls),
        "Stage45C evaluator loader timestamp selection changed",
    )

    stats = json.loads(Path(adapter_preflight["binding"]["paths"]["full24_runtime_statistics"]).read_text(encoding="utf-8"))
    production_loader = helper.production_loader_qualification(
        repo=repo,
        base=Path(adapter_preflight["binding"]["paths"]["base_checkpoint"]),
        dataset=dataset,
        groot=groot,
        cosmos=Path(adapter_preflight["binding"]["paths"]["cosmos_root"]),
        config_path=config_path,
        qualification_root=qualification_root / "production_loader",
        expected_stats=stats.get("statistics", {}),
        runtime_root=runtime_root,
    )

    final_maps = helper.mapped_libav_paths(runtime_root)
    return {
        "schema": EXPECTED_SCHEMA,
        "status": "PASS",
        "token": EXPECTED_STAGE45C_TOKEN,
        "pi_gate": "PI_GATE_STATE_0094",
        "timestamp_utc": now_utc(),
        "host": subprocess.check_output(["hostname"], text=True).strip(),
        "execution": {
            "branch": branch,
            "head": execution_head,
            "worktree_clean": True,
            "requested_python": str(requested_python),
            "python_realpath": str(Path(sys.executable).resolve()),
        },
        "stage45b_contract_source": {
            "helper_path": str(helper_path),
            "helper_sha256": EXPECTED_STAGE45B_HELPER_SHA256,
            "activation_order": expected_ld_prefix,
            "environment": environment,
        },
        "ffmpeg_runtime": artifact,
        "torch_runtime": {
            "version": str(torch.__version__),
            "cuda_build": str(torch.version.cuda),
            "file": str(Path(torch.__file__).resolve()),
            "package_identity": torch_identity,
            "library_root": str(torch_library_root),
        },
        "torchcodec_runtime": {
            "version": str(torchcodec_version),
            "package_root": str(torchcodec_root),
            "package_identity": torchcodec_identity,
            "ffmpeg_library_versions": ffmpeg_versions,
            "native_bindings": native_bindings,
            "mapped_libav_paths_after_native_load": maps_after_native_load,
            "mapped_libav_paths_after_full_qualification": final_maps,
        },
        "stage45c_adapter_preflight": adapter_preflight,
        "stage26_runtime_processor": {
            "path": str(stage26_path),
            "sha256": stage26_sha,
        },
        "frozen_training_config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "full24_video_sweep": video_sweep,
        "stage45c_evaluator_loader": {
            "window_count": len(windows),
            "decode_call_count": len(decode_calls),
            "decoded_loader_paths": sorted(observed_loader_paths),
            "decoded_loader_modalities": sorted(observed_loader_modalities),
            "decode_calls": decode_calls,
            "windows": window_summaries,
            "model_constructed": False,
            "model_forward": False,
        },
        "groot_imports": {
            "source_root": str(groot),
            "git_directory_absent": not (groot / ".git").exists(),
            "all_imports_from_sealed_source": True,
            "module_files": module_source_records(groot),
        },
        "groot_snapshot": {
            "verifier_path": str(snapshot_verifier_path),
            "verifier_sha256": EXPECTED_STAGE45B_SNAPSHOT_VERIFIER_SHA256,
            "archive_path": str(groot_archive_path),
            "archive_sha256": EXPECTED_GROOT_ARCHIVE_SHA256,
            "manifest_path": str(groot_manifest),
            "manifest_file_sha256": EXPECTED_GROOT_MANIFEST_FILE_SHA256,
            "manifest_payload_sha256": EXPECTED_GROOT_MANIFEST_PAYLOAD_SHA256,
            "payload_tree_sha256": EXPECTED_GROOT_PAYLOAD_TREE_SHA256,
            "archive_verification": groot_archive_verification,
            "extracted_tree_verification": groot_tree_verification,
        },
        "production_loader": production_loader,
        "scientific_exposure": {
            "model_constructed": False,
            "forward": False,
            "backward": False,
            "optimizer": False,
            "training": False,
            "checkpoint_creation": False,
            "pbs_submissions": 0,
            "validation_e_read": False,
            "holdout_f_read": False,
            "wbc": False,
            "simulator": False,
            "hardware": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--runtime-binding", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--requested-python", type=Path, required=True)
    parser.add_argument("--ffmpeg-runtime-root", type=Path, required=True)
    parser.add_argument("--ffmpeg-runtime-id", required=True)
    parser.add_argument("--ffmpeg-manifest-sha256", required=True)
    parser.add_argument("--torchcodec-package-root", type=Path, required=True)
    parser.add_argument("--torch-library-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stage45b-helper", type=Path, required=True)
    parser.add_argument("--stage45b-snapshot-verifier", type=Path, required=True)
    parser.add_argument("--groot-snapshot-archive", type=Path, required=True)
    parser.add_argument("--qualification-root", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = qualify(args)
    except Exception as exc:
        failure = {
            "schema": EXPECTED_SCHEMA,
            "status": "FAIL",
            "token": None,
            "pi_gate": "PI_GATE_STATE_0094",
            "timestamp_utc": now_utc(),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "scientific_exposure": {
                "model_constructed": False,
                "forward": False,
                "backward": False,
                "optimizer": False,
                "training": False,
                "checkpoint_creation": False,
                "pbs_submissions": 0,
            },
        }
        try:
            write_json(Path(args.evidence).resolve(), failure)
        except Exception:
            pass
        print(f"STAGE45C_NATIVE_RUNTIME_PREFLIGHT_BLOCKED: {failure['error']}", file=sys.stderr)
        return 2
    write_json(Path(args.evidence).resolve(), result)
    print(EXPECTED_STAGE45C_TOKEN)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
