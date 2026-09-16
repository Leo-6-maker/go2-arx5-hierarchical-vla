#!/usr/bin/env python3
"""Minimal PI0205 C80/S25/S50 schedule consumer and matched trainer.

--trace is dependency-free and validates the frozen schedule plus the real
FULL24/support paths. --trace-runtime uses the actual GR00T processor, dataset
factory, and collator without constructing a model. --train is the one-arm
training entry point.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import traceback
from datetime import datetime, timezone
from typing import Any, Mapping
import zipfile


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
SCRIPTS = REPO / "scripts"
for _path in (SRC, SCRIPTS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

SCHEDULE_SHA256 = "03634d5adec0366f1f8550f8cb10f48cc0096dc6e2d69f4d9de4a90c10376bb7"
SCHEDULE_SCHEMA = "CORELAB_PI0200_E256_SUPPORT_DOSE_SCHEDULE_V1"
ARM_ORDER = ("C80", "S25", "S50")
UPDATE_COUNT = 80
ROWS_SHA256 = "ab08414ac5cdda3cfadf1a08e5a85ccc3ec96fec74b32b6bf34bb5625f84f876"
VISITED_TREE_SHA256 = "f5ccefcb17594d563dce24aae3e476b7c56d2e6e75304a7358b4ec887aafb416"
EXPERT_TREE_SHA256 = "9b77dbddaf7b7fca6aacf55b0f68b202b398c8f1aa1dd0f0437d5dc18398be13"
FULL24_TREE_SHA256 = "2f3badd2019a68c6549a66f85b5486faf626078a3e438342f67aacc4edc95789"
FULL24_FILE_COUNT = 85
FULL24_BYTES = 919661
STRUCTURED_STATS_SHA256 = "0553436da13b4500e896cf4ec06ef36aba8d230536595cdd219818d912567206"
MODALITY_SHA256 = "e61f9429ecc21ceab51d9d26cadc9018691eb8b5c74d60aec4abf7faee16f031"
CONFIG_SHA256 = "9c3cd1d7bea603b95b1c777d8bcfebd61bd7eb5d86ca8ebbc3175f8141498a7c"
PROCESSOR_CONFIG_SHA256 = "85c1b4690ae090559e79a45193e598b65d6146eedf14750884da65e6d31032be"
PROCESSOR_STATS_SHA256 = "c97b1b07a82a8a8858771d56d278732d97dd8eae506032c146c77eb53828afc9"
BASE_TREE_SHA256 = "231e2cff6690242601147b3812298c806030f6562116df248e8b45c5ce53cd16"
BASE_TREE_PREFIX = "617089/E/checkpoint-256"
GROOT_MANIFEST_SHA256 = "96b883e56b54f763bec60d3a336632ad531f4fc52d099f803e5f9efba60eab11"
GROOT_MANIFEST_PAYLOAD_SHA256 = "a9f10b837b6086137fa519cc148275a25bf6a5f98d58df8e3c50e0a43e5b8250"
GROOT_PAYLOAD_TREE_SHA256 = "c0d57a6a74ab24c48abdd1c13420e9d1c893dc0d7b5ca370fa9b6f7af32f5f1b"
COSMOS_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"

DEFAULT_SCHEDULE = REPO / "reports/stage45g_pi0200_e256_support_dose_response_20260916/dose_response_schedule.json"
DEFAULT_ROWS = REPO / "reports/stage45f_pi0182_state_support_training_admission_20260914/origin_classified_rows.jsonl"
DEFAULT_SUPPORT_ROOT = REPO / "reports/stage45f_pi0181_visited_state_support_20260914"
DEFAULT_DATASET = Path("/var/cache/users/liuyu-nfs/stage45c_pi0100_train24_eval/evidence_completion_v3_pi0101_track_s/payload/full24_dataset/lerobot_train")
DEFAULT_BASE = Path("/hpctmp/e1192548/stage45e_h200_unified_replication_20260912/training/l1_seed2223_20260912T055239Z_613020.hopper-m-02/checkpoint-2000")
DEFAULT_GROOT = Path("/scratch/e1192548/stage45c_pi0094_native_runtime_20260909/groot-source-v4")
DEFAULT_GROOT_MANIFEST = Path("/scratch/e1192548/stage45b_pi0086_groot_snapshot__zhu4zz2/STAGE45B_GROOT_SOURCE_SNAPSHOT_V2.manifest.json")
DEFAULT_COSMOS = Path("/scratch/e1192548/stage22_h200_cosmos_v2_20260902/hf_home/hub/models--nvidia--Cosmos-Reason2-2B")
DEFAULT_MODALITY = Path("/scratch/e1192548/stage22_handoff_deb920a_20260902/payload/modality/go2_arx5_modality.py")
DEFAULT_CONFIG = Path("/scratch/e1192548/stage28_h2_20260904/stage28_h2_600695.hopper-m-02/h2_training/experiment_cfg/config.yaml")
DEFAULT_STATS = Path("/scratch/e1192548/leon-storage/runs/stage38_structured_groot_offline_20260906/stage38d_train_a7f6880/experiment_cfg/structured_action_statistics.json")
DEFAULT_FFMPEG = Path("/scratch/e1192548/STAGE45B_FFMPEG6_NATIVE_RUNTIME_V1__9243a7880b06ac3984ddf4a1a667b685441ec0f34d6d529fa0f96c3c894e10cf")


class HarnessError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise HarnessError(message)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    _regular_file(path, "JSON")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HarnessError(f"invalid JSON: {path}") from exc


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists() and not path.is_symlink(), f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _regular_file(path: Path, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise HarnessError(f"{label} unavailable: {path}") from exc
    require(stat.S_ISREG(mode), f"{label} is not a regular file: {path}")


def _regular_dir(path: Path, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise HarnessError(f"{label} unavailable: {path}") from exc
    require(stat.S_ISDIR(mode), f"{label} is not a regular directory: {path}")


def _child(root: Path, relative: str, label: str) -> Path:
    require("\x00" not in relative and "\\" not in relative, f"unsafe {label} path: {relative!r}")
    root = root.resolve()
    path = (root / relative).resolve()
    require(path.is_relative_to(root), f"{label} escapes root: {relative}")
    _regular_file(path, label)
    return path


def tree_digest(root: Path, prefix: str) -> tuple[str, list[dict[str, Any]]]:
    _regular_dir(root, "tree root")
    entries: list[dict[str, Any]] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for dirname in sorted(dirnames):
            child = directory_path / dirname
            require(not child.is_symlink(), f"symlink directory is forbidden: {child}")
        for filename in sorted(filenames):
            child = directory_path / filename
            _regular_file(child, "tree entry")
            entries.append({
                "path": f"{prefix}/{child.relative_to(root).as_posix()}",
                "bytes": int(child.stat().st_size),
                "sha256": sha256_file(child),
            })
    entries.sort(key=lambda item: str(item["path"]))
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(f"{entry['path']}\0{entry['bytes']}\0{entry['sha256']}\n".encode("utf-8"))
    return digest.hexdigest(), entries


def load_schedule(path: Path) -> dict[str, Any]:
    _regular_file(path, "schedule")
    observed = sha256_file(path)
    require(observed == SCHEDULE_SHA256, f"schedule SHA mismatch: {observed}")
    schedule = read_json(path)
    require(schedule.get("schema") == SCHEDULE_SCHEMA, "schedule schema drift")
    require(schedule.get("status") == "SEALED", "schedule status drift")
    require(schedule.get("seed") == 6164 and schedule.get("update_count") == UPDATE_COUNT, "schedule seed/count drift")
    require(schedule.get("full24_tree_sha256") == FULL24_TREE_SHA256, "schedule FULL24 tree drift")
    require(schedule.get("source_rows_sha256") == ROWS_SHA256, "schedule row binding drift")
    require(schedule.get("source_schedule_sha256") == "defd516cc2922d34ea5fcafef35b0c44288d8e78c4791c66cc51c7c7cd402e5e", "source schedule binding drift")
    full24_sequence = schedule.get("full24_sequence", [])
    require(len(full24_sequence) == UPDATE_COUNT, "full24 sequence length drift")
    require(all(isinstance(slot, int) and 0 <= slot < 24 for slot in full24_sequence), "full24 sequence value drift")
    require(len(schedule.get("visited_cycle", [])) == 20, "visited cycle length drift")
    require(canonical_sha(full24_sequence) == schedule.get("full24_sequence_sha256"), "full24 sequence digest drift")
    require(canonical_sha(schedule["visited_cycle"]) == schedule.get("visited_cycle_sha256"), "visited cycle digest drift")
    arms = schedule.get("arms")
    require(isinstance(arms, dict), "schedule arms missing")
    expected_counts = {
        "C80": {"FULL24": 80, "STATE_SUPPORT20": 0},
        "S25": {"FULL24": 60, "STATE_SUPPORT20": 20},
        "S50": {"FULL24": 40, "STATE_SUPPORT20": 40},
    }
    for arm in ARM_ORDER:
        arm_spec = arms.get(arm)
        require(isinstance(arm_spec, dict), f"{arm} schedule missing")
        slots = arm_spec.get("slots")
        require(isinstance(slots, list) and len(slots) == UPDATE_COUNT, f"{arm} schedule length drift")
        require(arm_spec.get("sequence_count") == UPDATE_COUNT, f"{arm} sequence count drift")
        require(arm_spec.get("counts") == expected_counts[arm], f"{arm} schedule counts drift")
        require(canonical_sha(slots) == arm_spec.get("sequence_sha256"), f"{arm} sequence digest drift")
        require([int(slot["position"]) for slot in slots] == list(range(UPDATE_COUNT)), f"{arm} positions are not contiguous")
        for slot in slots:
            require(slot["source"] in {"FULL24", "VISITED20"}, f"unknown {arm} source")
            if slot["source"] == "FULL24":
                require(slot["visited_row_id"] is None and 0 <= int(slot["full24_slot"]) < 24, f"invalid {arm} FULL24 slot")
                require(int(slot["full24_slot"]) == int(full24_sequence[int(slot["position"])]), f"{arm} FULL24 master binding drift")
            else:
                require(slot["full24_slot"] is None and isinstance(slot["visited_row_id"], str), f"invalid {arm} support slot")
        schedule[arm] = slots
    return schedule


def load_rows(rows_path: Path, support_root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    _regular_file(rows_path, "row overlay")
    observed = sha256_file(rows_path)
    require(observed == ROWS_SHA256, f"row overlay SHA mismatch: {observed}")
    rows: list[dict[str, Any]] = []
    for line in rows_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            require(isinstance(item, dict), "row overlay line is not an object")
            rows.append(item)
    require(len(rows) == 20, f"row count drift: {len(rows)}")
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = str(row.get("row_id", ""))
        require(row_id and row_id not in by_id, f"duplicate row id: {row_id}")
        require(row.get("model_input_fields") == ["body_rgb", "wrist_rgb", "state_v1", "language"], f"model input drift: {row_id}")
        require(row.get("target_excluded_from_model_input", True) is True, f"target leakage binding drift: {row_id}")
        require(row.get("origin_class") in {"INITIAL_SUPPORT", "AUTONOMOUS_VISITED"}, f"origin drift: {row_id}")
        payload = row.get("payload_files")
        require(isinstance(payload, dict), f"payload map missing: {row_id}")
        _child(support_root, str(payload["visited_row_npz"]), f"visited row {row_id}")
        _child(support_root, str(payload["expert_label_npz"]), f"expert label {row_id}")
        by_id[row_id] = row
    require(sum(row["origin_class"] == "INITIAL_SUPPORT" for row in rows) == 5, "initial-support count drift")
    require(sum(row["origin_class"] == "AUTONOMOUS_VISITED" for row in rows) == 15, "visited-support count drift")
    return by_id, {"sha256": observed, "row_count": len(rows)}


def validate_pack(root: Path, manifest_name: str, expected_schema: str, expected_tree: str) -> dict[str, Any]:
    manifest_path = _child(root, manifest_name, "pack manifest")
    manifest = read_json(manifest_path)
    require(manifest.get("schema") == expected_schema, f"pack schema drift: {manifest_name}")
    require(manifest.get("tree_sha256") == expected_tree, f"pack tree binding drift: {manifest_name}")
    entries = manifest.get("entries")
    require(isinstance(entries, list) and len(entries) == 20, f"pack entry count drift: {manifest_name}")
    observed_files = 0
    for entry in entries:
        relative = str(entry.get("path", ""))
        path = _child(root, relative, f"pack entry {relative}")
        require(int(path.stat().st_size) == int(entry["bytes"]), f"pack size drift: {relative}")
        require(sha256_file(path) == entry["sha256"], f"pack file SHA drift: {relative}")
        with zipfile.ZipFile(path) as archive:
            require(archive.testzip() is None, f"corrupt NPZ archive: {relative}")
        observed_files += 1
    return {"schema": manifest["schema"], "tree_sha256": expected_tree, "file_count": observed_files}


def validate_dataset(dataset: Path) -> dict[str, Any]:
    observed_tree, entries = tree_digest(dataset, "payload/lerobot_train")
    require(observed_tree == FULL24_TREE_SHA256, f"FULL24 tree mismatch: {observed_tree}")
    require(len(entries) == FULL24_FILE_COUNT, f"FULL24 file count drift: {len(entries)}")
    require(sum(int(entry["bytes"]) for entry in entries) == FULL24_BYTES, "FULL24 byte count drift")
    info = read_json(_child(dataset, "meta/info.json", "FULL24 info"))
    require(info.get("total_episodes") == 24 and info.get("total_frames") == 960, "FULL24 meta count drift")
    splits_path = dataset / "meta" / "splits.json"
    if splits_path.is_file() and not splits_path.is_symlink():
        splits = read_json(splits_path)
        require(splits.get("train") == "0:24", "FULL24 split drift")
    return {"tree_sha256": observed_tree, "file_count": len(entries), "bytes": sum(int(entry["bytes"]) for entry in entries)}


def trace_sequence(schedule: Mapping[str, Any], arm: str) -> list[dict[str, Any]]:
    require(arm in ARM_ORDER, f"unsupported arm: {arm}")
    sequence = []
    for slot in schedule[arm]:
        if slot["source"] == "FULL24":
            sample_id = f"FULL24:{int(slot['full24_slot'])}"
        else:
            sample_id = f"STATE_SUPPORT20:{slot['visited_row_id']}"
        sequence.append({
            "position": int(slot["position"]),
            "source": str(slot["source"]),
            "sample_id": sample_id,
            "full24_slot": slot["full24_slot"],
            "visited_row_id": slot["visited_row_id"],
        })
    require(len(sequence) == UPDATE_COUNT, f"{arm} trace length drift")
    return sequence


def static_validate(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    schedule = load_schedule(args.schedule)
    rows, rows_identity = load_rows(args.rows, args.support_root)
    dataset_identity = validate_dataset(args.dataset)
    visited_identity = validate_pack(args.support_root, "visited_pack_manifest.json", "CORELAB_PI0181_VISITED_STATE_PACK_V1", VISITED_TREE_SHA256)
    expert_identity = validate_pack(args.support_root, "expert_label_manifest.json", "CORELAB_PI0181_EXPERT_LABEL_PACK_V1", EXPERT_TREE_SHA256)
    optional = {}
    for path, expected, label in (
        (args.config, CONFIG_SHA256, "training config"),
        (args.modality, MODALITY_SHA256, "modality"),
        (args.stats, STRUCTURED_STATS_SHA256, "structured statistics"),
    ):
        if path is not None:
            _regular_file(path, label)
            digest = sha256_file(path)
            require(digest == expected, f"{label} SHA mismatch: {digest}")
            optional[label.replace(" ", "_")] = digest
    return schedule, rows, {
        "schedule_sha256": SCHEDULE_SHA256,
        "schedule_schema": SCHEDULE_SCHEMA,
        "structured_stats_sha256": STRUCTURED_STATS_SHA256,
        "rows": rows_identity,
        "dataset": dataset_identity,
        "visited_pack": visited_identity,
        "expert_pack": expert_identity,
        "optional": optional,
    }, {"schedule_path": str(args.schedule), "rows_path": str(args.rows), "support_root": str(args.support_root), "dataset": str(args.dataset)}


def static_trace(args: argparse.Namespace) -> int:
    schedule, rows, identities, paths = static_validate(args)
    arms = ARM_ORDER if args.arm == "all" else (args.arm,)
    traces = {}
    for arm in arms:
        sequence = trace_sequence(schedule, arm)
        for item in sequence:
            if item["source"] == "STATE_SUPPORT20":
                require(item["visited_row_id"] in rows, f"schedule references unknown row: {item['visited_row_id']}")
        traces[arm] = {
            "sequence_sha256": canonical_sha(sequence),
            "sequence_count": len(sequence),
            "counts": {
                "FULL24": sum(item["source"] == "FULL24" for item in sequence),
                "STATE_SUPPORT20": sum(item["source"] == "VISITED20" for item in sequence),
            },
            "sequence": sequence,
        }
    result = {
        "schema": "CORELAB_PI0205_DOSE_SCHEDULE_CPU_TRACE_V1",
        "status": "PASS",
        "timestamp_utc": now_utc(),
        "arms": traces,
        "identities": identities,
        "paths": paths,
        "model_construction": False,
        "model_forward": 0,
        "training": False,
        "gpu": 0,
        "Validation-E": 0,
        "Holdout-F": 0,
    }
    write_json(args.output, result)
    print(json.dumps({"status": result["status"], "output": str(args.output), "arms": {key: value["sequence_sha256"] for key, value in traces.items()}}, sort_keys=True))
    return 0


def validate_runtime_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    for name in ("base_model", "groot_source", "groot_manifest", "cosmos_root", "modality", "config", "stats", "ffmpeg_root"):
        require(getattr(args, name) is not None, f"missing runtime input: {name}")
    schedule, rows, identities, paths = static_validate(args)
    _regular_dir(args.base_model, "base checkpoint")
    base_tree, base_entries = tree_digest(args.base_model, BASE_TREE_PREFIX)
    require(base_tree == BASE_TREE_SHA256, f"base checkpoint tree mismatch: {base_tree}")
    trainer_state = read_json(_child(args.base_model, "trainer_state.json", "base trainer state"))
    require(int(trainer_state.get("global_step", -1)) == 2000, "base checkpoint global_step drift")
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
        _child(args.base_model, name, f"base {name}")
    _regular_dir(args.groot_source, "GR00T source")
    _regular_file(args.groot_manifest, "GR00T manifest")
    require(sha256_file(args.groot_manifest) == GROOT_MANIFEST_SHA256, "GR00T manifest SHA drift")
    _regular_dir(args.cosmos_root, "Cosmos cache entry")
    refs_main = _child(args.cosmos_root, "refs/main", "Cosmos refs/main")
    require(refs_main.read_text(encoding="utf-8").strip() == COSMOS_REVISION, "Cosmos revision drift")
    _regular_dir(args.ffmpeg_root / "lib", "FFmpeg lib")
    _regular_dir(args.ffmpeg_root / "compat", "FFmpeg compat")
    from stage45b_groot_source_snapshot import verify_extracted
    snapshot = verify_extracted(
        args.groot_source,
        args.groot_manifest,
        expected_manifest_file_sha256=GROOT_MANIFEST_SHA256,
        expected_manifest_payload_sha256=GROOT_MANIFEST_PAYLOAD_SHA256,
        expected_payload_tree_sha256=GROOT_PAYLOAD_TREE_SHA256,
    )
    identities["base_checkpoint"] = {"tree_sha256": base_tree, "file_count": len(base_entries)}
    identities["groot_snapshot"] = snapshot
    identities["groot_manifest_sha256"] = GROOT_MANIFEST_SHA256
    identities["cosmos_revision"] = COSMOS_REVISION
    identities["ffmpeg_root"] = str(args.ffmpeg_root)
    identities["base_processor_config_sha256"] = PROCESSOR_CONFIG_SHA256
    identities["base_processor_statistics_sha256"] = PROCESSOR_STATS_SHA256
    return schedule, rows, identities, paths


def prepare_runtime(output_root: Path, cosmos_root: Path, ffmpeg_root: Path, groot_source: Path, *, trace_only: bool) -> dict[str, str]:
    require(not output_root.exists() and not output_root.is_symlink(), f"runtime output must be fresh: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    cache = output_root / "runtime_cache"
    tmp = output_root / "runtime_tmp"
    hub = cache / "huggingface" / "hub"
    hub.mkdir(parents=True, exist_ok=False)
    tmp.mkdir(parents=True, exist_ok=False)
    link = hub / "models--nvidia--Cosmos-Reason2-2B"
    link.symlink_to(cosmos_root.resolve(), target_is_directory=True)
    site = Path(sys.prefix) / "lib" / "python3.12" / "site-packages"
    ld_parts = [str(ffmpeg_root / "lib"), str(ffmpeg_root / "compat"), str(site / "torchcodec"), str(site / "torch" / "lib")]
    ambient = os.environ.get("LD_LIBRARY_PATH", "")
    values = {
        "TMPDIR": str(tmp),
        "TMP": str(tmp),
        "TEMP": str(tmp),
        "XDG_CACHE_HOME": str(cache / "xdg"),
        "HF_HOME": str(cache / "huggingface"),
        "HF_HUB_CACHE": str(hub),
        "HUGGINGFACE_HUB_CACHE": str(hub),
        "TRANSFORMERS_CACHE": str(hub),
        "TORCH_HOME": str(cache / "torch"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "GROOT_COMMIT_HASH": "51d4c89f72fda44cbf77285c6a8114b52676b8a1",
        "GROOT_HF_LOCAL_FIRST": "1",
        "GROOT_PATCH_MISTRAL": "1",
        "STAGE45B_LABEL_ANCHOR_RECONCILIATION": "1",
        "LD_LIBRARY_PATH": ":".join(ld_parts + ([ambient] if ambient else [])),
    }
    os.environ.update(values)
    if trace_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, str(groot_source))
    sys.path.insert(0, str(SRC))
    sys.path.insert(0, str(SCRIPTS))
    return values


def bind_cosmos_processor(cosmos_root: Path) -> None:
    from gr00t.model.gr00t_n1d7 import processing_gr00t_n1d7 as processing
    snapshot = (cosmos_root / "snapshots" / COSMOS_REVISION).resolve(strict=True)
    original = getattr(processing, "_pi0183_original_build_processor", None)
    if original is None:
        original = processing.build_processor
        processing._pi0183_original_build_processor = original

        def build_local_processor(model_name: str, loading_kwargs: dict[str, Any]) -> Any:
            if model_name == "nvidia/Cosmos-Reason2-2B":
                model_name = str(snapshot)
            return original(model_name, loading_kwargs)

        processing.build_processor = build_local_processor


def load_config(args: argparse.Namespace, output_root: Path, update_count: int) -> Any:
    from gr00t.configs.base_config import get_default_config
    from stage38_structured_train import normalize_loaded_modality_configs
    config = get_default_config().load(args.config)
    config.load_config_path = None
    config.training.output_dir = str(output_root)
    config.training.experiment_name = None
    config.training.start_from_checkpoint = str(args.base_model)
    config.training.resume_from_checkpoint = False
    config.training.save_only_model = False
    config.training.max_steps = update_count
    config.training.save_steps = update_count
    config.training.save_total_limit = 1
    config.training.optim = "adamw_torch"
    config.training.learning_rate = 1e-4
    config.training.weight_decay = 1e-5
    config.training.lr_scheduler_type = "cosine"
    config.training.warmup_steps = 10
    config.training.warmup_ratio = 0.0
    config.training.bf16 = True
    config.training.fp16 = False
    config.training.global_batch_size = 1
    config.training.gradient_accumulation_steps = 1
    config.training.dataloader_num_workers = 0
    config.training.eval_strategy = "no"
    config.training.use_wandb = False
    config.training.transformers_local_files_only = True
    config.training.transformers_cache_dir = os.environ["HF_HUB_CACHE"]
    config.data.seed = 6164
    config.data.num_shards_per_epoch = 1
    require(len(config.data.datasets) == 1, "exactly one FULL24 dataset is required")
    config.data.datasets[0].dataset_paths = [str(args.dataset)]
    config.data.datasets[0].mix_ratio = 1.0
    config.model.action_horizon = 40
    config.model.max_action_dim = 132
    config.model.max_state_dim = 132
    config.model.tune_projector = True
    config.model.tune_diffusion_model = True
    config.model.tune_visual = False
    config.model.tune_llm = False
    config.model.tune_vlln = False
    config.model.use_relative_action = True
    config.model.use_percentiles = True
    normalize_loaded_modality_configs(config)
    require(config.training.start_from_checkpoint == str(args.base_model), "base binding drift")
    require(config.training.max_steps == update_count and config.training.save_steps == update_count, "step binding drift")
    require(config.data.seed == 6164 and config.data.num_shards_per_epoch == 1, "seed/sampler binding drift")
    require(config.training.eval_strategy == "no" and config.training.resume_from_checkpoint is False, "eval/resume binding drift")
    return config


def install_scheduled_factory(config: Any, args: argparse.Namespace, schedule: Mapping[str, Any], rows: Mapping[str, Mapping[str, Any]], processor_config_sha256: str):
    import numpy as np
    import torch
    from gr00t.data.dataset.sharded_mixture_dataset import ShardedMixtureDataset
    from gr00t.data.types import MessageType, VLAStepData
    from gr00t.model.gr00t_n1d7 import setup as gr00t_setup
    from go2arx5_vla.stage38_structured_training import (
        StructuredStateSupportAdapterV1,
        build_structured_gr00t_dataset_factory,
    )

    base_factory = build_structured_gr00t_dataset_factory()

    class ScheduledDataset(torch.utils.data.IterableDataset):
        # ponytail: one thin schedule wrapper; no custom ledger or framework.
        def __init__(self, full_dataset: Any, statistics: Mapping[str, Any], processor: Any):
            super().__init__()
            self.full_dataset = full_dataset
            self.statistics = statistics
            self.processor = processor
            self.dataset_path = full_dataset.dataset_path
            self.seed = int(config.data.seed)
            self.slots = list(schedule[args.arm])
            self.rows = rows
            self.last_sample_id: str | None = None
            self.adapter = StructuredStateSupportAdapterV1(
                normalizer=full_dataset.structured_action_normalizer,
                processor_config_sha256=processor_config_sha256,
                processor_statistics_sha256=PROCESSOR_STATS_SHA256,
                structured_statistics_sha256=STRUCTURED_STATS_SHA256,
                modality_sha256=MODALITY_SHA256,
                stage45b_label_reconciliation=True,
            )

        def __len__(self) -> int:
            return len(self.slots)

        def get_dataset_statistics(self) -> dict[str, Any]:
            return dict(self.statistics)

        def get_initial_actions(self) -> Any:
            return self.full_dataset.get_initial_actions()

        def get_modality_configs(self) -> Any:
            return self.full_dataset.get_modality_configs()

        def set_processor(self, processor: Any) -> None:
            self.processor = processor
            self.full_dataset.set_processor(processor)

        def reset_seed(self, seed: int) -> None:
            raise HarnessError(f"fresh arm forbids sampler reset: {seed}")

        def _support_sample(self, row_id: str) -> dict[str, Any]:
            row = self.rows[row_id]
            payload = row["payload_files"]
            visited_path = _child(args.support_root, str(payload["visited_row_npz"]), f"visited row {row_id}")
            expert_path = _child(args.support_root, str(payload["expert_label_npz"]), f"expert label {row_id}")
            with np.load(visited_path, allow_pickle=False) as visited:
                body = np.asarray(visited["body_rgb"])
                wrist = np.asarray(visited["wrist_rgb"])
                state = np.asarray(visited["state_v1"])
            with np.load(expert_path, allow_pickle=False) as expert:
                expert_action = np.asarray(expert["expert_action_structured_v1"])
                timestamps = np.asarray(expert["source_timestamps"], dtype=np.float64)
            require(body.dtype == np.uint8 and body.shape == (96, 96, 3), f"body binding drift: {row_id}")
            require(wrist.dtype == np.uint8 and wrist.shape == (96, 96, 3), f"wrist binding drift: {row_id}")
            require(state.shape == (10,) and np.issubdtype(state.dtype, np.floating), f"state binding drift: {row_id}")
            require(expert_action.shape == (40, 10) and np.issubdtype(expert_action.dtype, np.floating), f"expert action binding drift: {row_id}")
            require(timestamps.shape == (40,) and np.isfinite(timestamps).all() and np.all(np.diff(timestamps) > 0), f"timestamp binding drift: {row_id}")
            require(hashlib.sha256(np.ascontiguousarray(body).tobytes()).hexdigest() == row["body_sha256"], f"body array SHA drift: {row_id}")
            require(hashlib.sha256(np.ascontiguousarray(wrist).tobytes()).hexdigest() == row["wrist_sha256"], f"wrist array SHA drift: {row_id}")
            require(hashlib.sha256(np.ascontiguousarray(state).tobytes()).hexdigest() == row["state_v1_sha256"], f"state array SHA drift: {row_id}")
            require(hashlib.sha256(np.ascontiguousarray(expert_action).tobytes()).hexdigest() == row["expert_action_sha256"], f"expert array SHA drift: {row_id}")
            content = VLAStepData(
                images={"external_rgb": [body], "wrist_rgb": [wrist]},
                states={"eef_9d": state[:9][None, :], "gripper_position": state[9:10][None, :]},
                actions={"eef_9d": expert_action[:, :9], "gripper_position": expert_action[:, 9:10]},
                text=str(row["language"]),
                embodiment=self.full_dataset.embodiment_tag,
            )
            processed = self.processor([{"type": MessageType.EPISODE_STEP.value, "content": content}])
            datapoint = self.adapter.build_datapoint(
                {
                    "body_rgb": body,
                    "wrist_rgb": wrist,
                    "state_v1": state,
                    "language": str(row["language"]),
                },
                expert_action=expert_action,
                source_timestamps=timestamps,
                chunk_id=row_id,
                observation_timestamp=float(row["observation_timestamp_s"]),
                provenance={"origin_class": row["origin_class"], "source_row_id": row_id},
            )
            processed["action"] = datapoint["action"]
            processed["action_mask"] = datapoint["action_mask"]
            return processed

        def __iter__(self):
            for slot in self.slots:
                if slot["source"] == "FULL24":
                    full24_slot = int(slot["full24_slot"])
                    self.last_sample_id = f"FULL24:{full24_slot}"
                    episode_data = self.full_dataset.episode_loader[full24_slot]
                    yield self.full_dataset.get_datapoint(episode_data, 0, episode_index=full24_slot)
                else:
                    row_id = str(slot["visited_row_id"])
                    require(row_id in self.rows, f"unknown support row: {row_id}")
                    self.last_sample_id = f"STATE_SUPPORT20:{row_id}"
                    yield self._support_sample(row_id)

    class ScheduledFactory:
        def __init__(self, config: Any):
            self.config = config

        def build(self, processor: Any):
            original_shards = self.config.data.num_shards_per_epoch
            self.config.data.num_shards_per_epoch = 1
            try:
                mixture, _ = base_factory(self.config).build(processor)
            finally:
                self.config.data.num_shards_per_epoch = original_shards
            require(isinstance(mixture, ShardedMixtureDataset), "unexpected Stage38 dataset factory result")
            require(len(mixture.datasets) == 1, "scheduled dataset requires one FULL24 dataset")
            full_dataset = mixture.datasets[0]
            require(full_dataset.structured_action_statistics.get("statistics_recomputed") is False, "S20 statistics were recomputed")
            scheduled = ScheduledDataset(full_dataset, mixture.get_dataset_statistics(), processor)
            return scheduled, None

    gr00t_setup.DatasetFactory = ScheduledFactory
    return ScheduledDataset


def build_runtime_dataset(args: argparse.Namespace, output_root: Path, *, trace_only: bool):
    schedule, rows, identities, paths = validate_runtime_inputs(args)
    env = prepare_runtime(output_root, args.cosmos_root, args.ffmpeg_root, args.groot_source, trace_only=trace_only)
    os.environ.update({
        "STAGE45F_FROZEN_STRUCTURED_STATS_PATH": str(args.stats),
        "STAGE45F_FROZEN_STRUCTURED_STATS_SHA256": STRUCTURED_STATS_SHA256,
        "STAGE45F_PI0183_SCHEDULE_PATH": str(args.schedule),
        "STAGE45F_PI0183_ARM": args.arm,
        "STAGE45F_PI0183_SUPPORT_ROOT": str(args.support_root),
    })
    bind_cosmos_processor(args.cosmos_root)
    config = load_config(args, output_root, int(schedule["update_count"]))
    install_scheduled_factory(config, args, schedule, rows, PROCESSOR_CONFIG_SHA256)
    from gr00t.model.gr00t_n1d7.setup import Gr00tN1d7Pipeline
    save_cfg_dir = output_root / "experiment_cfg"
    save_cfg_dir.mkdir(parents=False, exist_ok=False)
    pipeline = Gr00tN1d7Pipeline(config, save_cfg_dir)
    train_dataset, _ = pipeline._create_dataset(save_cfg_dir)
    return train_dataset, pipeline.return_processor().collator, schedule, identities, paths, env


def runtime_trace(args: argparse.Namespace) -> int:
    dataset, collator, schedule, identities, paths, env = build_runtime_dataset(args, args.output, trace_only=True)
    expected = trace_sequence(schedule, args.arm)
    seen = []
    iterator = iter(dataset)
    for item in expected:
        sample = next(iterator)
        require(getattr(dataset, "last_sample_id", None) == item["sample_id"], f"trainer sequence drift at {item['position']}")
        require(set(sample) <= {"vlm_content", "state", "action", "action_mask", "embodiment_id"}, "unexpected model-bound field")
        action = sample["action"]
        mask = sample["action_mask"]
        require(tuple(action.shape) == (40, 132) and tuple(mask.shape) == (40, 132), f"sample shape drift at {item['position']}")
        batch = collator([sample])
        inputs = batch["inputs"]
        require(tuple(inputs["action"].shape) == (1, 40, 132), f"collated action shape drift at {item['position']}")
        require(tuple(inputs["action_mask"].shape) == (1, 40, 132), f"collated mask shape drift at {item['position']}")
        seen.append(item["sample_id"])
    require(seen == [item["sample_id"] for item in expected], "runtime trace sequence mismatch")
    result = {
        "schema": "CORELAB_PI0205_DOSE_RUNTIME_CPU_TRACE_V1",
        "status": "PASS",
        "timestamp_utc": now_utc(),
        "arm": args.arm,
        "sequence_sha256": canonical_sha(expected),
        "sequence_count": len(seen),
        "identities": identities,
        "paths": paths,
        "runtime_environment": env,
        "trainer_path": "Gr00tN1d7Pipeline._create_dataset -> scheduled DatasetFactory -> processor.collator",
        "model_construction": False,
        "model_forward": 0,
        "training": False,
        "gpu": 0,
        "Validation-E": 0,
        "Holdout-F": 0,
    }
    write_json(args.output / "runtime_trace.json", result)
    print(json.dumps({"status": result["status"], "output": str(args.output / "runtime_trace.json"), "sequence_sha256": result["sequence_sha256"]}, sort_keys=True))
    return 0


def train(args: argparse.Namespace) -> int:
    output_root = args.output
    schedule, rows, identities, paths = validate_runtime_inputs(args)
    schedule_updates = int(schedule["update_count"])
    updates = schedule_updates if args.max_steps is None else int(args.max_steps)
    require(1 <= updates <= schedule_updates, f"max_steps must be in [1, {schedule_updates}]")
    env = prepare_runtime(output_root, args.cosmos_root, args.ffmpeg_root, args.groot_source, trace_only=False)
    os.environ.update({
        "STAGE45F_FROZEN_STRUCTURED_STATS_PATH": str(args.stats),
        "STAGE45F_FROZEN_STRUCTURED_STATS_SHA256": STRUCTURED_STATS_SHA256,
        "STAGE45F_PI0183_SCHEDULE_PATH": str(args.schedule),
        "STAGE45F_PI0183_ARM": args.arm,
        "STAGE45F_PI0183_SUPPORT_ROOT": str(args.support_root),
    })
    bind_cosmos_processor(args.cosmos_root)
    config = load_config(args, output_root, updates)
    install_scheduled_factory(config, args, schedule, rows, PROCESSOR_CONFIG_SHA256)
    metadata = {
        "schema": "CORELAB_PI0205_DOSE_ARM_TRAINING_INVOCATION_V1",
        "status": "STARTED",
        "timestamp_utc": now_utc(),
        "arm": args.arm,
        "base_family": "L1",
        "base_seed": 2223,
        "seed": 6164,
        "updates": updates,
        "schedule_sha256": SCHEDULE_SHA256,
        "sequence_sha256": canonical_sha(trace_sequence(schedule, args.arm)),
        "identities": identities,
        "paths": paths,
        "runtime_environment": env,
        "model_construction": 0,
        "model_forward": 0,
        "backward": 0,
        "optimizer_steps": 0,
        "Validation-E": 0,
        "Holdout-F": 0,
    }
    write_json(output_root / "training_invocation.json", metadata)
    from gr00t.experiment.experiment import run as gr00t_run
    gr00t_run(config)
    final_checkpoint = output_root / f"checkpoint-{updates}"
    trainer_state = read_json(_child(final_checkpoint, "trainer_state.json", "final trainer state"))
    require(int(trainer_state.get("global_step", -1)) == updates, f"final global_step is not {updates}")
    checkpoint_prefix = f"pi0205_{args.arm}/checkpoint-{updates}"
    checkpoint_tree, checkpoint_entries = tree_digest(final_checkpoint, checkpoint_prefix)
    result = {
        "schema": "CORELAB_PI0205_DOSE_ARM_TRAINING_RESULT_V1",
        "status": "PASS",
        "timestamp_utc": now_utc(),
        "arm": args.arm,
        "output_root": str(output_root),
        "final_checkpoint": str(final_checkpoint),
        "global_step": updates,
        "updates": updates,
        "schedule_sha256": SCHEDULE_SHA256,
        "sequence_sha256": canonical_sha(trace_sequence(schedule, args.arm)),
        "identities": identities,
        "model_construction": True,
        "model_forward": updates,
        "backward": updates,
        "optimizer_steps": updates,
        "checkpoint_tree_sha256": checkpoint_tree,
        "checkpoint_tree_prefix": checkpoint_prefix,
        "checkpoint_file_count": len(checkpoint_entries),
        "checkpoint_bytes": sum(int(entry["bytes"]) for entry in checkpoint_entries),
        "Validation-E": 0,
        "Holdout-F": 0,
        "WBC": 0,
        "simulator": 0,
        "hardware": 0,
    }
    write_json(output_root / "training_result.json", result)
    print(json.dumps({"status": result["status"], "arm": args.arm, "checkpoint": result["final_checkpoint"]}, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--trace", action="store_true")
    modes.add_argument("--trace-runtime", action="store_true")
    modes.add_argument("--train", action="store_true")
    parser.add_argument("--arm", choices=(*ARM_ORDER, "all"), required=True)
    parser.add_argument("--max-steps", type=int, help="short discarded canary only; defaults to the sealed 80 updates")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE)
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--support-root", type=Path, default=DEFAULT_SUPPORT_ROOT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--groot-source", type=Path, default=DEFAULT_GROOT)
    parser.add_argument("--groot-manifest", type=Path, default=DEFAULT_GROOT_MANIFEST)
    parser.add_argument("--cosmos-root", type=Path, default=DEFAULT_COSMOS)
    parser.add_argument("--modality", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--stats", type=Path)
    parser.add_argument("--ffmpeg-root", type=Path, default=DEFAULT_FFMPEG)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.trace:
            return static_trace(args)
        require(args.arm in ARM_ORDER, "--trace-runtime/--train require one of C80, S25, S50")
        if args.trace_runtime:
            return runtime_trace(args)
        return train(args)
    except Exception as exc:
        failure = {
            "schema": "CORELAB_PI0183_FAILURE_V1",
            "status": "BLOCKED",
            "timestamp_utc": now_utc(),
            "mode": "trace" if args.trace else "trace-runtime" if args.trace_runtime else "train",
            "arm": args.arm,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "model_construction": False if not args.train else "UNKNOWN",
            "model_forward": 0 if not args.train else "UNKNOWN",
            "training": False if not args.train else "UNKNOWN",
            "Validation-E": 0,
            "Holdout-F": 0,
        }
        try:
            target = args.output / "failure.json" if args.output.suffix == "" else args.output
            if not target.exists() and not target.is_symlink():
                write_json(target, failure)
        except Exception:
            pass
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
