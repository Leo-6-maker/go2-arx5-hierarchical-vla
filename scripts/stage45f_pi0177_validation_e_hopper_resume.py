#!/usr/bin/env python3
"""PI0180 Validation-E generation-1 Hopper completion runner.

The parent process owns the durable ledger and starts one fresh child for each
unfinished checkpoint arm.  Children load exactly one checkpoint, seal raw
arrays, and exit; decoding and all scientific readout happen later in the
CPU-only parent posthoc phase.  The module is standard-library-only until a
child has crossed the immutable preflight boundary.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import stat
import subprocess
import sys
import time
from typing import Any


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
SCRIPTS = REPO / "scripts"
for _path in (SRC, SCRIPTS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import stage45f_pi0159_validation_e_continuation as continuation  # noqa: E402


PI_GATE = 5664544367
PR71_POINTER = 5661697942
PR73_DONOR = 5661699939
SOURCE_HEAD = "ce3c0179f3974648cdf6527e145d0328b0ac9262"
BRANCH = "stage45f/localpc-validation-e-hopper-resume"
DONOR_HEAD = "bdc0a3f68210927629d3b64a4cda97b0af030efc"
FREEZE_PATH = REPO / "manifests/STAGE45F_PI0154_VALIDATION_E_GEN1_RECIPE_FREEZE_V1.json"
RETURN_TOKEN = "LOCALPC_PI0180_VALIDATION_E_GEN1_FOR_PI_REVIEW — COMPLETED"
SCIENCE_HOLD = "LOCALPC_PI0180_SCIENCE_REVIEW_HOLD"
SCHEMA = "STAGE45F_PI0180_VALIDATION_E_GEN1_HOPPER_RESUME_V1"
LEDGER_SCHEMA = "STAGE45F_PI0180_EXECUTION_LEDGER_V1"
RAW_SCHEMA = "STAGE45F_PI0154_RAW_PREDICTION_V1"
DERIVED_SCHEMA = "STAGE45F_PI0180_DERIVED_EVIDENCE_REFERENCE_V1"
TARGET_ID = "red_target_holdout_e"
NOISE_SEED = 4242
TIMESTEP_IDS = [0, 250, 500, 750]
T_VALUES = [0.0, 0.25, 0.5, 0.75]
RAW_SHAPE = (1, 40, 132)
EXPECTED_SOURCE_RAW_MANIFEST_SHA = "0ebe595377d32164ff395d9d0f6d779b74a3654e473f12863eca2c6e83d0f8df"
EXPECTED_STATS_SHA = "3b3a2d51e685a0e580de4ebdfa05abcbcddfb7dc33b8d3a659994cdcd1bb0337"
EXPECTED_STATS_PAYLOAD_SHA = "752898989bb3308e663c4b6cf88b3cc1c139897a6860e6aed1d9d708d0f6fd75"
EXPECTED_GROOT_COMMIT = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
EXPECTED_GROOT_PAYLOAD_TREE = "c0d57a6a74ab24c48abdd1c13420e9d1c893dc0d7b5ca370fa9b6f7af32f5f1b"
EXPECTED_GROOT_MANIFEST_SHA = "96b883e56b54f763bec60d3a336632ad531f4fc52d099f803e5f9efba60eab11"
EXPECTED_COSMOS_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"
EXPECTED_COSMOS_TREE = "1cfec77c944cba341ee5d7c31b4be6bb615582364bae48f18c5686e6ec23e8a9"
EXPECTED_FFMPEG_CAPSULE = "/hpctmp/e1192548/stage45f_pi0163_ffmpeg_capsule_v1/STAGE45B_FFMPEG6_NATIVE_RUNTIME_V1__9243a7880b06ac3984ddf4a1a667b685441ec0f34d6d529fa0f96c3c894e10cf.tar"
EXPECTED_FFMPEG_CAPSULE_SHA = "3ddd7e94d2f97cedf77998bc9516143a427ce87945b31203689b8456283a633a"
EXPECTED_FFMPEG_RUNTIME_ID = "9243a7880b06ac3984ddf4a1a667b685441ec0f34d6d529fa0f96c3c894e10cf"
EXPECTED_FFMPEG_MANIFEST_SHA = "6160d579b98dc5ee4b114ada4a4e00e88ee13df6ed1fc3d308b5879d67027d2c"
EXPECTED_FFMPEG_RUNTIME_ROOT = "/scratch/e1192548/STAGE45B_FFMPEG6_NATIVE_RUNTIME_V1__9243a7880b06ac3984ddf4a1a667b685441ec0f34d6d529fa0f96c3c894e10cf"
EXPECTED_VIDEO = "/scratch/e1192548/leon-storage/runs/stage39_vis_offline_mp4_20260906/v0_c0_expert.mp4"
EXPECTED_MODALITY_SHA = "e61f9429ecc21ceab51d9d26cadc9018691eb8b5c74d60aec4abf7faee16f031"
EXPECTED_BASE_TREE = "8717e322de3dd8c66e52bdb81b37badb0f235c518de6acd465532ed01569a6a4"
EXPECTED_BASE_CONFIG_SHA = "85c1b4690ae090559e79a45193e598b65d6146eedf14750884da65e6d31032be"
EXPECTED_BASE_STATS_SHA = "c97b1b07a82a8a8858771d56d278732d97dd8eae506032c146c77eb53828afc9"
EXPECTED_SNAPSHOT_LOCK_SHA = "23c35e2204b9555c8431a3f86fd7e1767740bf7196341972e847bc43c141aec8"
EXPECTED_SPLIT_LOCK_SHA = "7e124f29d2e4d24c17fd1fefcf8753ececd3a8cd86e8c6a2f159e2e5d2217246"
EXPECTED_ARM_KEYS = tuple(
    (route, seed)
    for route in ("Legacy", "L1")
    for seed in (2222, 2223, 2224)
)
EXPECTED_ARM_LABELS = tuple(f"{route}(seed{seed})@2000" for route, seed in EXPECTED_ARM_KEYS)
EXPECTED_WINDOWS = tuple(f"stage22_expert_{TARGET_ID}:expert-{index:04d}" for index in range(6))
EXPECTED_CHECKPOINT_TREES = {
    "Legacy(seed2222)@2000": "2113a6a3b2a91233a42214dbb017c0136acde282fda2a418066f9317e78981a5",
    "L1(seed2222)@2000": "53293d0ae3555c00acb7595b6c6dd19ed92450668344123cdc46efd5b69fcd49",
    "Legacy(seed2223)@2000": "4895df0a0d5adf7e9010bce70fe2aaf447bbb7e4d4a28a179316a502d666592b",
    "L1(seed2223)@2000": "231e2cff6690242601147b3812298c806030f6562116df248e8b45c5ce53cd16",
    "Legacy(seed2224)@2000": "fc4dc36959a44398aed73b0b8f949dcbbfe42d1d15157e87667c17850384d2e9",
    "L1(seed2224)@2000": "3c25b67036fcf3af037e3ad98d7a76b97c1933cd54445dd984436dbac5951b92",
}
EXPECTED_CALLS = {"conditioning": 36, "action_head": 144, "model_side_total": 180, "raw_outputs": 36}
NEW_NOMINAL = {"model_loads": 5, "conditioning": 30, "action_head": 120, "model_side_total": 150, "raw": 30}
NEW_CEILING = {"model_loads": 7, "conditioning": 34, "action_head": 136, "model_side_total": 170}
FORBIDDEN_TOP_LEVEL = {"torch", "transformers", "gr00t", "cuda", "cv2", "numpy"}
FORBIDDEN_CALLS = {"backward", "get_action", "qsub", "save_checkpoint", "save_model", "save_pretrained", "train"}


class PI0180Error(RuntimeError):
    pass


class MechanicalInterruption(PI0180Error):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PI0180Error(message)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def load_json(path: Path) -> Any:
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"regular JSON file required: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any, *, overwrite: bool = False) -> None:
    continuation.write_json_durable(Path(path), value, overwrite=overwrite)


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_text(path: Path, value: str, *, overwrite: bool = False) -> None:
    path = Path(path)
    if not overwrite:
        require(not path.exists() and not path.is_symlink(), f"text record must be fresh: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    try:
        directory_fd = os.open(path.parent, os.O_RDONLY)
    except OSError:
        directory_fd = None
    if directory_fd is not None:
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def regular_file(path: Path, label: str) -> Path:
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and stat.S_ISREG(path.lstat().st_mode), f"{label} is not a regular file: {path}")
    return path.resolve()


def regular_dir(path: Path, label: str) -> Path:
    path = Path(path)
    require(path.is_dir() and not path.is_symlink() and stat.S_ISDIR(path.lstat().st_mode), f"{label} is not a regular directory: {path}")
    return path.resolve()


def run_text(argv: Sequence[str]) -> str:
    completed = subprocess.run(list(argv), capture_output=True, text=True, check=False)
    require(completed.returncode == 0, f"command failed: {' '.join(argv)}: {completed.stderr[-1200:]}")
    return completed.stdout.strip()


def safe_label(label: str) -> str:
    return label.replace("(", "_").replace(")", "").replace("@", "_at_")


def raw_path(label: str, index: int) -> Path:
    return Path("raw") / safe_label(label) / f"window-{index:02d}.npy"


def derived_path(label: str, index: int) -> Path:
    return Path("derived") / safe_label(label) / f"window-{index:02d}.json"


def static_guard() -> dict[str, Any]:
    import ast

    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(Path(__file__)))
    top_imports: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_imports.add(node.module.split(".", 1)[0])
    forbidden_imports = sorted(top_imports.intersection(FORBIDDEN_TOP_LEVEL))
    forbidden_calls: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ""
            if name in FORBIDDEN_CALLS:
                forbidden_calls.append(name)
    qsub_call_seen = any(
        isinstance(node, ast.Call)
        and (node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else "") == "qsub"
        for node in ast.walk(tree)
    )
    raw_before_decode = source.index("RAW_SEALED") < source.index("_decode_v2")
    markers = {
        "append_only_ledger": "class ExecutionLedger" in source and "append_jsonl" in source,
        "fresh_child_per_arm": "--arm-child" in source and "subprocess.run(command" in source,
        "raw_before_decode": raw_before_decode,
        "cpu_posthoc": "def cpu_posthoc" in source and "CPU_POSTHOC" in source,
        "resume_safe": "missing_cells" in source and "NEW_CEILING" in source,
        "phase_telemetry": "rss_bytes" in source and "cuda_memory" in source,
        "no_training": '"training": False' in source,
        "holdout_f_locked": '"holdout_f_read": False' in source,
        "no_h200_submit": not qsub_call_seen,
    }
    return {
        "schema": "STAGE45F_PI0180_EXECUTION_STATIC_GUARD_V1",
        "status": "PASS" if not forbidden_imports and not forbidden_calls and all(markers.values()) else "FAIL",
        "top_level_forbidden_imports": forbidden_imports,
        "forbidden_call_hits": sorted(set(forbidden_calls)),
        "markers": markers,
        "model_imported": False,
        "validation_e_model_calls": 0,
        "training": False,
        "pbs_submissions": 0,
    }


def _rss_bytes() -> int | None:
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        return None
    return None


def _cuda_memory() -> dict[str, Any]:
    torch = sys.modules.get("torch")
    if torch is None or not getattr(torch, "cuda", None) or not torch.cuda.is_available():
        return {"available": False}
    device = torch.device("cuda:0")
    return {
        "available": True,
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


class ExecutionLedger:
    """Append-only phase ledger; one child is active at a time."""

    def __init__(self, path: Path, *, generation: str, gpu_uuid: str | None) -> None:
        self.path = Path(path)
        self.generation = generation
        self.gpu_uuid = gpu_uuid
        self.sequence = 0
        self._validate_existing()
        self.emit("LEDGER_OPEN", phase="ADMISSION", resumed=self.sequence > 0)

    def _validate_existing(self) -> None:
        if not self.path.exists():
            return
        require(self.path.is_file() and not self.path.is_symlink(), f"ledger is not a regular file: {self.path}")
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            require(line.strip(), f"ledger contains a blank line {line_number}")
            record = json.loads(line)
            require(isinstance(record, Mapping), f"ledger record {line_number} is not an object")
            sequence = int(record.get("sequence", -1))
            require(sequence == self.sequence + 1, f"ledger sequence gap at line {line_number}")
            self.sequence = sequence

    def emit(self, event: str, *, phase: str, **details: Any) -> int:
        self.sequence += 1
        record = {
            "schema": LEDGER_SCHEMA,
            "sequence": self.sequence,
            "event": event,
            "phase": phase,
            "timestamp_utc": now_utc(),
            "monotonic_s": time.monotonic(),
            "rss_bytes": _rss_bytes(),
            "cuda_memory": _cuda_memory(),
            "generation": self.generation,
            "gpu_uuid": self.gpu_uuid,
            "model_runtime_imported": any(name in sys.modules for name in ("torch", "transformers", "gr00t")),
            "holdout_f_read": False,
            **details,
        }
        append_jsonl(self.path, record)
        return self.sequence

    def begin(self, phase: str, **details: Any) -> dict[str, Any]:
        sequence = self.emit("BEGIN", phase=phase, operation=phase, **details)
        return {"sequence": sequence, "monotonic_s": time.monotonic(), "phase": phase}

    def end(self, token: Mapping[str, Any], *, status: str = "COMPLETED", **details: Any) -> int:
        duration = max(0.0, time.monotonic() - float(token["monotonic_s"]))
        return self.emit("END", phase=str(token["phase"]), operation=str(token["phase"]), begin_sequence=int(token["sequence"]), duration_s=duration, status=status, **details)

    def count_begin(self, phase: str) -> int:
        if not self.path.exists():
            return 0
        return sum(1 for item in self._records() if item.get("event") == "BEGIN" and item.get("phase") == phase)

    def _records(self) -> list[dict[str, Any]]:
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _copy_file_durable(source: Path, destination: Path) -> None:
    source = regular_file(source, "copy source")
    destination = Path(destination)
    require(not destination.exists() and not destination.is_symlink(), f"copy destination is not fresh: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as source_handle, destination.open("xb") as destination_handle:
        for block in iter(lambda: source_handle.read(1 << 20), b""):
            destination_handle.write(block)
        destination_handle.flush()
        os.fsync(destination_handle.fileno())
    directory_fd = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _load_config(config_path: Path) -> dict[str, Any]:
    value = load_json(config_path.resolve(strict=True))
    require(isinstance(value, Mapping), "PI0180 config is not an object")
    config = dict(value)
    require(config.get("schema") == SCHEMA, "PI0180 config schema drift")
    require(config.get("pi_gate") == PI_GATE, "PI0180 gate drift")
    require(config.get("branch") == BRANCH, "PI0180 branch drift")
    require(config.get("source_head") == SOURCE_HEAD, "PI0180 source HEAD drift")
    require(config.get("donor_head") == DONOR_HEAD, "PI0180 donor HEAD drift")
    require(config.get("expected_calls") == EXPECTED_CALLS, "PI0180 nominal call budget drift")
    require(config.get("new_call_ceiling") == NEW_CEILING, "PI0180 recovery ceiling drift")
    require(config.get("holdout_f_read") is False and config.get("training") is False, "PI0180 execution boundary opened")
    require(config.get("checkpoint_arms") == list(EXPECTED_ARM_LABELS), "PI0180 checkpoint order drift")
    require(config.get("window_ids") == list(EXPECTED_WINDOWS), "PI0180 window order drift")
    require(config.get("ffmpeg_capsule") == EXPECTED_FFMPEG_CAPSULE, "FFmpeg capsule path drift")
    require(config.get("ffmpeg_capsule_sha256") == EXPECTED_FFMPEG_CAPSULE_SHA, "FFmpeg capsule SHA drift")
    require(config.get("ffmpeg_runtime_id") == EXPECTED_FFMPEG_RUNTIME_ID, "FFmpeg runtime identity drift")
    require(config.get("ffmpeg_manifest_sha256") == EXPECTED_FFMPEG_MANIFEST_SHA, "FFmpeg manifest SHA drift")
    require(config.get("ffmpeg_runtime_root") == EXPECTED_FFMPEG_RUNTIME_ROOT, "FFmpeg runtime root drift")
    require(config.get("video") == EXPECTED_VIDEO, "runtime probe video drift")
    require(config.get("cosmos_revision") == EXPECTED_COSMOS_REVISION, "Cosmos revision drift")
    return config


def _checkout_audit(repo: Path, expected_head: str) -> dict[str, Any]:
    repo = regular_dir(repo, "execution repository")
    root = run_text(["git", "-C", str(repo), "rev-parse", "--show-toplevel"])
    branch = run_text(["git", "-C", str(repo), "branch", "--show-current"])
    head = run_text(["git", "-C", str(repo), "rev-parse", "HEAD"])
    status = run_text(["git", "-C", str(repo), "status", "--porcelain=v1", "--untracked-files=all"])
    status = "\n".join(
        line
        for line in status.splitlines()
        if not line.startswith(("?? st45f_pi0177.o", "?? st45f_pi0177.e"))
    )
    require(Path(root).resolve() == repo, f"repository root drift: {root}")
    require(branch == BRANCH, f"branch drift: {branch}")
    require(head == expected_head, f"execution HEAD drift: {head} != {expected_head}")
    require(not status, "execution checkout is dirty")
    run_text(["git", "-C", str(repo), "merge-base", "--is-ancestor", SOURCE_HEAD, head])
    changed = run_text(["git", "-C", str(repo), "diff", "--name-only", f"{SOURCE_HEAD}..{head}"]).splitlines()
    allowed = {
        "scripts/stage45f_pi0177_validation_e_hopper_resume.py",
        "scripts/stage45f_pi0177_corelab_qualification.py",
        "manifests/STAGE45F_PI0177_VALIDATION_E_GEN1_HOPPER_RESUME_V1.json",
        "tests/test_stage45f_pi0177_hopper_resume.py",
        "hpc/pbs/stage45f_pi0177_validation_e_gen1_h200.pbs",
    }
    require(set(changed) <= allowed, f"unexpected changes outside PI0180 addition: {changed}")
    return {"root": root, "branch": branch, "head": head, "clean": True, "changed_paths": changed}


def _freeze_audit(expected_head: str) -> dict[str, Any]:
    import stage45f_vanda_validation_e_admission as admission

    freeze = load_json(FREEZE_PATH)
    require(isinstance(freeze, Mapping), "recipe freeze is not an object")
    admission._validate_freeze(freeze, expected_code_commit="522ea718e99ff8b1c4b2aa08bec680ed5adfcae5")
    return {
        "path": str(FREEZE_PATH),
        "sha256": sha256_file(FREEZE_PATH),
        "checkpoint_arms": [dict(item) for item in freeze["checkpoint_arms"]],
        "expected_head": expected_head,
        "scientific_contract_changed": False,
    }


def _checkpoint_prefix(label: str) -> str:
    route, seed = ("Legacy", 2222) if label.startswith("Legacy(seed2222)") else ("", 0)
    if label == "Legacy(seed2222)@2000":
        return "checkpoint-2000"
    if label.startswith("Legacy("):
        return "stage45e_legacy/checkpoint-2000"
    if label.startswith("L1("):
        return "stage45e_l1/checkpoint-2000"
    raise PI0180Error(f"unsupported checkpoint label: {label}")


def _artifact_audit(config: Mapping[str, Any], binding: Mapping[str, Any], expected_head: str) -> dict[str, Any]:
    import stage45c_h200_eval_runtime_adapter as stage45c
    import stage45f_vanda_validation_e_gen1 as evaluator

    require("torch" not in sys.modules and "transformers" not in sys.modules and "gr00t" not in sys.modules, "model runtime imported before artifact audit")
    stats = regular_file(Path(str(config["stats"])), "FULL24 runtime stats")
    require(sha256_file(stats) == EXPECTED_STATS_SHA, "FULL24 runtime stats SHA drift")
    base = regular_dir(Path(str(config["base_model"])), "BASE model")
    base_tree, base_entries = evaluator.tree_sha256(base, "payload/GR00T-N1.7-3B")
    require(base_tree == EXPECTED_BASE_TREE, f"BASE tree drift: {base_tree}")
    base_config = regular_file(base / "processor_config.json", "BASE processor config")
    base_stats = regular_file(base / "statistics.json", "BASE processor statistics")
    require(sha256_file(base_config) == EXPECTED_BASE_CONFIG_SHA, "BASE processor config SHA drift")
    require(sha256_file(base_stats) == EXPECTED_BASE_STATS_SHA, "BASE processor statistics SHA drift")
    dataset = regular_dir(Path(str(config["dataset"])), "FULL24 dataset")
    dataset_audit = stage45c._verify_dataset(dataset)
    groot = regular_dir(Path(str(config["groot_source"])), "GR00T source")
    groot_head = run_text(["git", "-C", str(groot), "rev-parse", "HEAD"])
    require(groot_head == EXPECTED_GROOT_COMMIT, f"GR00T source HEAD drift: {groot_head}")
    groot_manifest = regular_file(Path(str(config["groot_manifest"])), "GR00T source manifest")
    require(sha256_file(groot_manifest) == EXPECTED_GROOT_MANIFEST_SHA, "GR00T source manifest SHA drift")
    manifest_text = groot_manifest.read_text(encoding="utf-8")
    require(EXPECTED_GROOT_COMMIT in manifest_text and EXPECTED_GROOT_PAYLOAD_TREE in manifest_text, "GR00T provenance manifest identity drift")
    modality = regular_file(Path(str(config["modality"])), "modality")
    require(sha256_file(modality) == EXPECTED_MODALITY_SHA, "modality SHA drift")
    cosmos = regular_dir(Path(str(config["cosmos_root"])), "Cosmos cache")
    cosmos_tree, cosmos_entries = continuation.cosmos_tree_digest(cosmos)
    require(cosmos_tree == EXPECTED_COSMOS_TREE, f"Cosmos tree drift: {cosmos_tree}")
    refs = regular_file(cosmos / "refs" / "main", "Cosmos refs/main")
    require(refs.read_text(encoding="utf-8").strip() == EXPECTED_COSMOS_REVISION, "Cosmos revision drift")
    checkpoint_paths = config.get("checkpoints")
    require(isinstance(checkpoint_paths, Mapping), "checkpoint path map missing")
    checkpoint_audit: list[dict[str, Any]] = []
    for label in EXPECTED_ARM_LABELS:
        path = regular_dir(Path(str(checkpoint_paths[label])), f"checkpoint {label}")
        observed, entries = evaluator.tree_sha256(path, _checkpoint_prefix(label))
        require(observed == EXPECTED_CHECKPOINT_TREES[label], f"checkpoint tree drift {label}: {observed}")
        state = load_json(path / "trainer_state.json")
        require(int(state.get("global_step", -1)) == 2000, f"checkpoint global_step drift: {label}")
        checkpoint_audit.append({
            "label": label,
            "path": str(path),
            "tree_sha256": observed,
            "hash_prefix": _checkpoint_prefix(label),
            "file_count": len(entries),
            "total_bytes": sum(int(item["bytes"]) for item in entries),
            "trainer_state_sha256": sha256_file(path / "trainer_state.json"),
        })
    require(isinstance(binding.get("validation_input"), Mapping), "Validation-E binding is missing")
    validation_input = binding["validation_input"]
    require(validation_input.get("target_id") == TARGET_ID, "Validation-E target drift")
    require(validation_input.get("snapshot_ids") == list(EXPECTED_WINDOWS), "Validation-E windows drift")
    require(validation_input.get("holdout_f_read") is False, "Holdout-F was not locked")
    source_root = regular_dir(Path(str(config["source_root"])), "Validation-E source root")
    require(Path(str(validation_input["source_root"])).resolve() == source_root, "Validation-E source root binding drift")
    windows = evaluator.validate_snapshot_records(binding)
    require([item["window_id"] for item in windows] == list(EXPECTED_WINDOWS), "Validation-E snapshot order drift")
    return {
        "schema": "STAGE45F_PI0180_PRE_MODEL_ARTIFACT_AUDIT_V1",
        "status": "PASS",
        "execution_head": expected_head,
        "base": {"path": str(base), "tree_sha256": base_tree, "file_count": len(base_entries)},
        "dataset": dataset_audit,
        "groot": {"path": str(groot), "git_head": groot_head, "manifest": str(groot_manifest), "manifest_sha256": sha256_file(groot_manifest)},
        "cosmos": {"path": str(cosmos), "tree_sha256": cosmos_tree, "entry_count": cosmos_entries, "revision": EXPECTED_COSMOS_REVISION},
        "stats": {"path": str(stats), "sha256": sha256_file(stats), "payload_sha256": EXPECTED_STATS_PAYLOAD_SHA},
        "modality": {"path": str(modality), "sha256": sha256_file(modality)},
        "checkpoint_audit": checkpoint_audit,
        "snapshot_ids": list(EXPECTED_WINDOWS),
        "snapshot_count": len(windows),
        "target_xyz_in_model_input": False,
        "validation_e_model_calls": 0,
        "holdout_f_read": False,
        "training": False,
    }


def _source_inventory(config: Mapping[str, Any]) -> dict[str, Any]:
    import stage45f_pi0163_validation_e_recovery as recovery

    source_root = regular_dir(Path(str(config["preserved_raw_root"])), "allocation-8 raw root")
    inventory = recovery.validate_allocation8_inventory(recovery.raw_inventory_isolated(source_root))
    require(inventory.get("raw_manifest_sha256") == EXPECTED_SOURCE_RAW_MANIFEST_SHA, "allocation-8 raw manifest SHA drift")
    require(inventory.get("preserved_raw_count") == 6 and inventory.get("missing_raw_count") == 30, "allocation-8 raw count drift")
    require(
        [(item["arm"], item["window_id"]) for item in inventory["preserved_cells"]]
        == [(EXPECTED_ARM_LABELS[0], window_id) for window_id in EXPECTED_WINDOWS],
        "allocation-8 preserved arm drift",
    )
    return inventory


def _runtime_binding(config: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    binding_path = output_root / "runtime_binding.json"
    if binding_path.exists():
        binding = load_json(binding_path)
        require(isinstance(binding, Mapping), "runtime binding is not an object")
        environment = binding.get("environment")
        require(isinstance(environment, Mapping), "runtime binding environment is missing")
        os.environ.update({str(key): str(value) for key, value in environment.items()})
        return dict(binding)
    binding = continuation.bind_offline_runtime(config, output_root)
    write_json(binding_path, binding)
    return binding


def _preserved_reuse(config: Mapping[str, Any], inventory: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    source_root = regular_dir(Path(str(config["preserved_raw_root"])), "allocation-8 raw root")
    source_manifest = regular_file(source_root / "raw_predictions.jsonl", "allocation-8 raw manifest")
    require(sha256_file(source_manifest) == EXPECTED_SOURCE_RAW_MANIFEST_SHA, "source raw manifest SHA drift")
    raw_manifest = output_root / "raw_predictions.jsonl"
    if not raw_manifest.exists():
        _copy_file_durable(source_manifest, raw_manifest)
        _copy_file_durable(source_manifest, output_root / "source_raw_manifest.jsonl")
        for item in inventory["preserved_cells"]:
            _copy_file_durable(source_root / str(item["path"]), output_root / str(item["path"]))
    else:
        require((output_root / "source_raw_manifest.jsonl").is_file(), "source raw manifest witness is missing")
        require(sha256_file(output_root / "source_raw_manifest.jsonl") == EXPECTED_SOURCE_RAW_MANIFEST_SHA, "source raw witness SHA drift")
    output_inventory = _output_inventory(output_root)
    for source_item in inventory["preserved_cells"]:
        key = (str(source_item["arm"]), str(source_item["window_id"]))
        matching = next((item for item in output_inventory["preserved_cells"] if (str(item["arm"]), str(item["window_id"])) == key), None)
        require(matching is not None, f"preserved raw cell missing after reuse: {key}")
        require(matching["file_sha256"] == source_item["file_sha256"] and matching["array_sha256"] == source_item["array_sha256"], f"preserved raw bytes drift: {key}")
    reuse = {
        "schema": "STAGE45F_PI0180_VALID_RAW_REUSE_V1",
        "status": "PASS_BYTE_PRESERVED",
        "source_root": str(source_root),
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": EXPECTED_SOURCE_RAW_MANIFEST_SHA,
        "preserved_raw_count": 6,
        "model_loads_for_preserved_arm": 0,
        "model_forwards_for_preserved_arm": 0,
        "cells": [
            {
                "arm": item["arm"],
                "window_id": item["window_id"],
                "path": item["path"],
                "file_sha256": item["file_sha256"],
                "array_sha256": item["array_sha256"],
                "source_allocation": 8,
            }
            for item in inventory["preserved_cells"]
        ],
        "scientific_contract_changed": False,
    }
    write_json(output_root / "raw_reuse.json", reuse, overwrite=(output_root / "raw_reuse.json").exists())
    return reuse


def _runtime_capsule_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    capsule = regular_file(Path(str(config["ffmpeg_capsule"])), "FFmpeg runtime capsule")
    capsule_sha256 = sha256_file(capsule)
    require(capsule_sha256 == EXPECTED_FFMPEG_CAPSULE_SHA, "FFmpeg runtime capsule SHA drift")
    runtime_root = regular_dir(Path(str(config["ffmpeg_runtime_root"])), "sealed FFmpeg runtime root")
    manifest = regular_file(runtime_root / "STAGE45B_FFMPEG6_NATIVE_RUNTIME_V1.json", "sealed FFmpeg runtime manifest")
    require(sha256_file(manifest) == EXPECTED_FFMPEG_MANIFEST_SHA, "sealed FFmpeg runtime manifest SHA drift")
    import tarfile

    with tarfile.open(capsule, mode="r:") as archive:
        member = archive.getmember("STAGE45B_FFMPEG6_NATIVE_RUNTIME_V1.json")
        manifest_file = archive.extractfile(member)
        require(manifest_file is not None, "FFmpeg capsule manifest cannot be read")
        require(sha256_bytes(manifest_file.read()) == EXPECTED_FFMPEG_MANIFEST_SHA, "FFmpeg capsule manifest SHA drift")
    return {
        "capsule_path": str(capsule),
        "capsule_bytes": int(capsule.stat().st_size),
        "capsule_sha256": capsule_sha256,
        "runtime_root": str(runtime_root),
        "runtime_id": EXPECTED_FFMPEG_RUNTIME_ID,
        "manifest_sha256": EXPECTED_FFMPEG_MANIFEST_SHA,
        "manifest_path": str(manifest),
        "video_path": EXPECTED_VIDEO,
        "compute_node_bootstrap_required": True,
    }


def _output_inventory(output_root: Path) -> dict[str, Any]:
    import stage45f_pi0163_validation_e_recovery as recovery

    return recovery.raw_inventory_isolated(output_root)


def _validate_output_raw(config: Mapping[str, Any], output_root: Path, source_inventory: Mapping[str, Any]) -> dict[str, Any]:
    import stage45f_pi0163_validation_e_recovery as recovery

    inventory = _output_inventory(output_root)
    require(int(inventory["preserved_raw_count"]) <= 36 and int(inventory["missing_raw_count"]) >= 0, "output raw cardinality invalid")
    source_by_cell = {(str(item["arm"]), str(item["window_id"])): item for item in source_inventory["preserved_cells"]}
    for item in inventory["preserved_cells"]:
        key = (str(item["arm"]), str(item["window_id"]))
        record = item["record"]
        if key in source_by_cell:
            expected = source_by_cell[key]
            require(record.get("source_allocation", 8) == 8, f"preserved source allocation drift: {key}")
            require(item["file_sha256"] == expected["file_sha256"] and item["array_sha256"] == expected["array_sha256"], f"preserved raw identity drift: {key}")
        else:
            require(record.get("source_allocation") == 10, f"new raw source allocation drift: {key}")
            require(record.get("raw_reused_without_model_call") is False, f"new raw reuse flag drift: {key}")
            require(record.get("generation") == "Validation-E-gen1", f"new raw generation drift: {key}")
    return inventory


def _prepare_output(config: Mapping[str, Any], source_inventory: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    output_root = Path(str(config["output_root"])).resolve()
    require(output_root.parent.is_dir(), f"output parent is unavailable: {output_root.parent}")
    require(not output_root.exists() and not output_root.is_symlink(), f"output root is not fresh: {output_root}")
    output_root.mkdir(parents=False, exist_ok=False)
    runtime_binding = _runtime_binding(config, output_root)
    reuse = _preserved_reuse(config, source_inventory, output_root)
    inventory = _validate_output_raw(config, output_root, source_inventory)
    write_json(output_root / "stage45f_pi0180_initial_raw_inventory.json", inventory)
    return output_root, runtime_binding


def preflight(config_path: Path, repo: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    expected_head = str(config["expected_head"])
    checkout = _checkout_audit(repo, expected_head)
    guard = static_guard()
    require(guard["status"] == "PASS", "PI0180 static guard failed")
    freeze = _freeze_audit(expected_head)
    binding = load_json(regular_file(Path(str(config["admission_path"])), "Validation-E admission"))
    require(isinstance(binding, Mapping), "Validation-E admission is not an object")
    source_inventory = _source_inventory(config)
    artifact_audit = _artifact_audit(config, binding, expected_head)
    runtime_envelope = _runtime_capsule_audit(config)
    output_root, runtime_binding = _prepare_output(config, source_inventory)
    effective_path = output_root / "effective_config.json"
    write_json(effective_path, config)
    record = {
        "schema": "STAGE45F_PI0180_PRE_MODEL_ADMISSION_V1",
        "status": "PASS_PRE_MODEL",
        "pi_gate": PI_GATE,
        "pr71_pointer": PR71_POINTER,
        "pr73_donor": PR73_DONOR,
        "execution_branch": BRANCH,
        "execution_head": expected_head,
        "source_head": SOURCE_HEAD,
        "donor_head": DONOR_HEAD,
        "checkout": checkout,
        "static_guard": guard,
        "freeze": freeze,
        "artifact_audit": artifact_audit,
        "runtime_envelope": runtime_envelope,
        "source_raw_inventory": source_inventory,
        "source_raw_manifest_sha256": EXPECTED_SOURCE_RAW_MANIFEST_SHA,
        "runtime_binding_path": str(output_root / "runtime_binding.json"),
        "runtime_binding_sha256": sha256_file(output_root / "runtime_binding.json"),
        "effective_config_path": str(effective_path),
        "effective_config_sha256": sha256_file(effective_path),
        "preserved_arm": EXPECTED_ARM_LABELS[0],
        "preserved_arm_model_loads": 0,
        "new_cells": 30,
        "new_nominal_calls": NEW_NOMINAL,
        "new_call_ceiling": NEW_CEILING,
        "holdout_f_read": False,
        "training": False,
        "validation_e_model_calls": 0,
        "qsub_called": False,
        "scientific_contract_changed": False,
    }
    write_json(output_root / "stage45f_pi0180_preflight.json", record)
    ledger = ExecutionLedger(output_root / "execution_ledger.jsonl", generation="Validation-E-gen1", gpu_uuid=None)
    ledger.emit("PRE_MODEL_ADMISSION", phase="ADMISSION", execution_head=expected_head, validation_e_model_calls=0, qsub_called=False)
    for item in source_inventory["preserved_cells"]:
        ledger.emit(
            "RAW_REUSED",
            phase="RAW_SEAL",
            arm=item["arm"],
            window_id=item["window_id"],
            path=item["path"],
            file_sha256=item["file_sha256"],
            array_sha256=item["array_sha256"],
            raw_reused_without_model_call=True,
            source_allocation=8,
            validation_e_model_calls=0,
        )
    print(json.dumps(record, indent=2, sort_keys=True))
    return record


def _load_preflight(config: Mapping[str, Any]) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    output_root = Path(str(config["output_root"])).resolve()
    preflight_path = regular_file(output_root / "stage45f_pi0180_preflight.json", "PI0180 preflight")
    record = load_json(preflight_path)
    require(isinstance(record, Mapping) and record.get("status") == "PASS_PRE_MODEL", "PI0180 preflight is not PASS")
    require(record.get("execution_head") == config.get("expected_head") and record.get("execution_branch") == BRANCH, "PI0180 preflight identity drift")
    runtime_binding = load_json(regular_file(output_root / "runtime_binding.json", "runtime binding"))
    require(isinstance(runtime_binding, Mapping), "runtime binding is not an object")
    environment = runtime_binding.get("environment")
    require(isinstance(environment, Mapping), "runtime environment is missing")
    os.environ.update({str(key): str(value) for key, value in environment.items()})
    return output_root, dict(record), dict(runtime_binding)


def _observe_h200_gpu(output_root: Path) -> dict[str, Any]:
    raw = run_text(["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.used", "--format=csv,noheader,nounits"])
    rows = []
    for line in raw.splitlines():
        fields = [item.strip() for item in line.split(",")]
        require(len(fields) == 5, f"malformed nvidia-smi row: {line}")
        rows.append({"index": int(fields[0]), "name": fields[1], "uuid": fields[2], "total_vram_mib": int(fields[3]), "used_vram_mib": int(fields[4])})
    require(len(rows) == 1 and rows[0]["index"] == 0, f"expected exactly one visible GPU at index 0: {rows}")
    gpu_name = str(rows[0]["name"])
    require("H200" in gpu_name.upper(), f"admitted GPU is not H200: {gpu_name}")
    require(rows[0]["total_vram_mib"] * 1024**2 >= 80 * 1024**3, "admitted H200 VRAM is below 80 GiB")
    gpu = {
        "schema": "STAGE45F_PI0180_OBSERVED_H200_GPU_V1",
        "host": socket.gethostname(),
        "pbs_jobid": os.environ["PBS_JOBID"],
        "selected_device": "cuda:0",
        "gpu_name": gpu_name,
        "gpu_uuid": rows[0]["uuid"],
        "total_vram_bytes": rows[0]["total_vram_mib"] * 1024**2,
        "memory_used_mib_at_admission": rows[0]["used_vram_mib"],
        "visible_gpu_count": 1,
        "all_visible_gpus": rows,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi_stdout_sha256": sha256_bytes((raw + "\n").encode("utf-8")),
        "allowed_gpu_models": ["H200"],
    }
    job_key = "".join(character if character.isalnum() or character in "_-" else "_" for character in os.environ["PBS_JOBID"])
    path = output_root / f"gpu_runtime_admission_{job_key}.json"
    if path.exists():
        old = load_json(path)
        require(old.get("gpu_uuid") == gpu.get("gpu_uuid"), "GPU UUID changed inside one allocation")
        return dict(old)
    record = {
        "schema": "STAGE45F_PI0180_GPU_RUNTIME_ADMISSION_V1",
        "status": "PASS",
        "gpu": gpu,
        "gpu_uuid": gpu["gpu_uuid"],
        "selected_device": "cuda:0",
        "model_loads_before_admission": 0,
        "validation_e_model_calls_before_admission": 0,
    }
    write_json(path, record)
    summary_path = output_root / "gpu_runtime_admission.json"
    if not summary_path.exists():
        write_json(summary_path, record)
    return record


def _runtime_record_path(output_root: Path, environment_key: str, label: str) -> Path:
    value = os.environ.get(environment_key)
    require(value, f"{environment_key} is required after runtime probe")
    path = Path(value).resolve()
    require(path.parent.is_relative_to(output_root.resolve()), f"{label} escaped output root: {path}")
    return regular_file(path, label)


def _record_runtime_admission(config: Mapping[str, Any], output_root: Path, ledger: ExecutionLedger) -> dict[str, Any]:
    bootstrap_token = ledger.begin("RUNTIME_BOOTSTRAP", runtime_id=EXPECTED_FFMPEG_RUNTIME_ID)
    try:
        bootstrap_path = _runtime_record_path(output_root, "PI0180_BOOTSTRAP_RECORD", "runtime bootstrap record")
        bootstrap = load_json(bootstrap_path)
        require(isinstance(bootstrap, Mapping) and bootstrap.get("status") == "PASS", "runtime bootstrap did not PASS")
        require(bootstrap.get("source_runtime_id") == EXPECTED_FFMPEG_RUNTIME_ID, "runtime bootstrap identity drift")
        require(bootstrap.get("source_manifest_sha256") == EXPECTED_FFMPEG_MANIFEST_SHA, "runtime bootstrap manifest drift")
        require(bootstrap.get("capsule_sha256") == EXPECTED_FFMPEG_CAPSULE_SHA, "runtime bootstrap capsule drift")
        ledger.end(bootstrap_token, status="PASS", record_path=str(bootstrap_path), record_sha256=sha256_file(bootstrap_path), destination=bootstrap.get("destination"))
    except Exception as exc:
        ledger.end(bootstrap_token, status="ERROR", error=f"{type(exc).__name__}: {exc}")
        raise

    probe_token = ledger.begin("RUNTIME_IMPORT_PROBE", runtime_id=EXPECTED_FFMPEG_RUNTIME_ID)
    try:
        probe_path = _runtime_record_path(output_root, "PI0180_COMPUTE_IMPORT_RECORD", "runtime import probe record")
        probe = load_json(probe_path)
        require(isinstance(probe, Mapping) and probe.get("status") == "PASS", "runtime import probe did not PASS")
        require(probe.get("mode") == "compute", "runtime import probe mode drift")
        require(probe.get("model_construction") is False and int(probe.get("model_forward", -1)) == 0, "runtime import probe crossed model execution")
        require(probe.get("validation_e_read") is False and probe.get("holdout_f_read") is False, "runtime import probe read scientific payload")
        require(probe.get("cosmos", {}).get("revision") == EXPECTED_COSMOS_REVISION, "runtime import probe Cosmos revision drift")
        ledger.end(probe_token, status="PASS", record_path=str(probe_path), record_sha256=sha256_file(probe_path), ffmpeg_root=probe.get("ffmpeg_root"), torch=probe.get("torch"), torchcodec=probe.get("torchcodec"))
    except Exception as exc:
        ledger.end(probe_token, status="ERROR", error=f"{type(exc).__name__}: {exc}")
        raise
    return {"bootstrap": dict(bootstrap), "probe": dict(probe)}


def _noise_manifest(output_root: Path, hashes: list[str]) -> None:
    path = output_root / "sampler_noise_manifest.json"
    record = {
        "schema": "STAGE45F_PI0180_CPU_FLOAT32_NOISE_BANK_V1",
        "seed": NOISE_SEED,
        "device": "cpu",
        "dtype": "float32",
        "shape": list(RAW_SHAPE),
        "window_ids": list(EXPECTED_WINDOWS),
        "noise_sha256_by_window": hashes,
        "same_noise_across_all_six_checkpoint_arms": True,
    }
    if path.exists():
        old = load_json(path)
        require(old == record, "sampler noise manifest drift")
    else:
        write_json(path, record)


def _raw_record(output_root: Path, arm: Mapping[str, Any], window_index: int, raw_np: Any, noise_sha: str, gpu_uuid: str | None) -> dict[str, Any]:
    import numpy as np
    import stage45f_vanda_validation_e_gen1 as evaluator

    label = str(arm["label"])
    path = output_root / raw_path(label, window_index)
    require(not path.exists() and not path.is_symlink(), f"raw path is not fresh: {path}")
    evaluator.write_npy_durable(path, raw_np)
    file_sha = sha256_file(path)
    array_sha = sha256_bytes(np.ascontiguousarray(raw_np).astype(np.float32, copy=False).tobytes())
    record = {
        "schema": RAW_SCHEMA,
        "checkpoint": dict(arm),
        "window_id": EXPECTED_WINDOWS[window_index],
        "path": path.relative_to(output_root).as_posix(),
        "file_sha256": file_sha,
        "array_sha256": array_sha,
        "shape": list(RAW_SHAPE),
        "dtype": "float32",
        "noise_sha256": noise_sha,
        "timestep_ids": list(TIMESTEP_IDS),
        "action_head_calls_observed": 4,
        "decoded": False,
        "raw_reused_without_model_call": False,
        "source_allocation": 10,
        "source_job_id": os.environ.get("PBS_JOBID"),
        "generation": "Validation-E-gen1",
        "gpu_uuid": gpu_uuid,
        "unique_query": True,
    }
    evaluator.append_jsonl_durable(output_root / "raw_predictions.jsonl", record)
    return record


def _signal_handler(signum: int, _frame: Any) -> None:
    raise MechanicalInterruption(f"received signal {signum}")


def arm_child(config_path: Path, arm_label: str) -> int:
    config = _load_config(config_path)
    require(arm_label in EXPECTED_ARM_LABELS, f"unknown arm: {arm_label}")
    output_root, preflight_record, runtime_binding = _load_preflight(config)
    source_inventory = _source_inventory(config)
    inventory = _validate_output_raw(config, output_root, source_inventory)
    missing = [(str(item["arm"]), str(item["window_id"])) for item in inventory["missing_cells"] if str(item["arm"]) == arm_label]
    ledger = ExecutionLedger(output_root / "execution_ledger.jsonl", generation="Validation-E-gen1", gpu_uuid=None)
    current: dict[str, Any] = {"arm": arm_label, "window_id": None, "step": None, "phase": "CHILD_START"}
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    if not missing:
        ledger.emit("CHILD_ARM_SKIPPED", phase="ARM_CHILD", arm=arm_label, reason="all_cells_already_sealed")
        print(f"PI0180_CHILD_ARM_SKIPPED {arm_label}")
        return 0
    require(ledger.count_begin("MODEL_LOAD") + 1 <= NEW_CEILING["model_loads"], "PI0180 model-load ceiling exceeded")
    require(ledger.count_begin("CONDITIONING") + len(missing) <= NEW_CEILING["conditioning"], "PI0180 conditioning ceiling exceeded")
    require(ledger.count_begin("ACTION_HEAD") + len(missing) * 4 <= NEW_CEILING["action_head"], "PI0180 action-head ceiling exceeded")
    require(ledger.count_begin("CONDITIONING") + ledger.count_begin("ACTION_HEAD") + len(missing) * 5 <= NEW_CEILING["model_side_total"], "PI0180 model-side ceiling exceeded")
    gpu_uuid = os.environ.get("PI0180_GPU_UUID")
    if not gpu_uuid:
        gpu_admission = load_json(regular_file(output_root / "gpu_runtime_admission.json", "GPU runtime admission"))
        gpu_uuid = str(gpu_admission["gpu_uuid"])
    ledger.gpu_uuid = gpu_uuid
    try:
        import numpy as np
        import torch
        import stage26_h1_c1_fixed_noise_local_flow as stage26
        import stage45c_h200_eval_runtime_adapter as runtime
        import stage45f_vanda_validation_e_gen1 as evaluator

        current["phase"] = "PROCESSOR_BUILD"
        processor_token = ledger.begin("PROCESSOR_BUILD", arm=arm_label)
        runtime_processor, embodiment, _modality_configs = stage26._build_runtime_processor(
            groot_source=regular_dir(Path(str(config["groot_source"])), "GR00T source"),
            base_checkpoint=regular_dir(Path(str(config["base_model"])), "BASE model"),
            dataset_dir=regular_dir(Path(str(config["dataset"])), "FULL24 dataset"),
            modality_path=regular_file(Path(str(config["modality"])), "modality"),
            cosmos_snapshot=regular_dir(Path(str(config["cosmos_root"])), "Cosmos cache") / "snapshots" / EXPECTED_COSMOS_REVISION,
        )
        ledger.end(processor_token, arm=arm_label)
        current["phase"] = "WINDOW_PREPARATION"
        binding = load_json(regular_file(Path(str(config["admission_path"])), "Validation-E admission"))
        windows_meta = evaluator.validate_snapshot_records(binding)
        window_token = ledger.begin("WINDOW_PREPARATION", arm=arm_label, window_count=6)
        windows = evaluator._prepare_windows(runtime_processor, embodiment, windows_meta)
        require(len(windows) == 6 and [item["window_id"] for item in windows] == list(EXPECTED_WINDOWS), "prepared E window order drift")
        ledger.end(window_token, arm=arm_label, window_count=len(windows))
        noise_token = ledger.begin("NOISE_BANK_READY", arm=arm_label)
        noise_bank, noise_hashes = evaluator._make_noise_bank(torch)
        _noise_manifest(output_root, noise_hashes)
        ledger.end(noise_token, arm=arm_label, noise_sha256_by_window=noise_hashes)
        runtime_identity = {
            "python": str(Path(sys.executable).resolve()),
            "python_version": sys.version,
            "torch_version": str(torch.__version__),
            "cuda_version": str(torch.version.cuda),
            "gpu_uuid": gpu_uuid,
            "generation": "Validation-E-gen1",
        }
        runtime_sha = canonical_sha(runtime_identity)
        arm = next(dict(item) for item in _freeze_arms(preflight_record) if item["label"] == arm_label)
        checkpoint_path = regular_dir(Path(str(config["checkpoints"][arm_label])), f"checkpoint {arm_label}")
        current["phase"] = "MODEL_LOAD"
        load_token = ledger.begin("MODEL_LOAD", arm=arm_label, checkpoint=arm_label, checkpoint_path=str(checkpoint_path))
        model = None
        load_completed = False
        try:
            model = runtime._load_torch_runtime()[1](checkpoint_path, "cuda:0")
            model.eval()
            ledger.end(load_token, arm=arm_label, checkpoint=arm_label, load_status="COMPLETED")
            load_completed = True
            for _label, window_id in missing:
                index = EXPECTED_WINDOWS.index(str(window_id))
                window = windows[index]
                current.update({"phase": "CONDITIONING", "window_id": window_id, "step": None})
                condition_token = ledger.begin("CONDITIONING", arm=arm_label, window_id=window_id)
                _, action_input, processed, state_features = runtime._prepare_model_conditioning(model, window["model_inputs"])
                ledger.end(condition_token, arm=arm_label, window_id=window_id)
                current_action = noise_bank[index].to(device=processed.backbone_features.device, dtype=next(model.parameters()).dtype)
                with torch.inference_mode():
                    for step, timestep in enumerate(TIMESTEP_IDS):
                        current.update({"phase": "ACTION_HEAD", "step": step})
                        action_token = ledger.begin("ACTION_HEAD", arm=arm_label, window_id=window_id, step=step, timestep_id=timestep)
                        velocity = runtime._offline_step(model.action_head, processed, action_input, state_features, current_action, int(timestep))
                        require(tuple(int(value) for value in velocity.shape) == RAW_SHAPE and bool(torch.isfinite(velocity).all().item()), f"action-head output drift: {arm_label}:{window_id}:{step}")
                        current_action = current_action + 0.25 * velocity
                        ledger.end(action_token, arm=arm_label, window_id=window_id, step=step, timestep_id=timestep)
                raw_np = current_action.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy().astype(np.float32, copy=False)
                require(raw_np.shape == RAW_SHAPE and bool(np.isfinite(raw_np).all()), f"raw output drift: {arm_label}:{window_id}")
                raw_token = ledger.begin("RAW_SEAL", arm=arm_label, window_id=window_id)
                record = _raw_record(output_root, arm, index, raw_np, noise_hashes[index], gpu_uuid)
                ledger.end(raw_token, arm=arm_label, window_id=window_id, path=record["path"], file_sha256=record["file_sha256"], array_sha256=record["array_sha256"])
                ledger.emit("RAW_SEALED", phase="RAW_SEALED", arm=arm_label, window_id=window_id, path=record["path"], file_sha256=record["file_sha256"], array_sha256=record["array_sha256"], raw_before_decode=True)
        except Exception as exc:
            if not load_completed:
                ledger.end(load_token, status="ERROR", arm=arm_label, checkpoint=arm_label, error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            release_token = ledger.begin("MODEL_RELEASE", arm=arm_label)
            del model
            gc.collect()
            torch.cuda.empty_cache()
            ledger.end(release_token, arm=arm_label)
        ledger.emit("ARM_CHILD_COMPLETE", phase="ARM_CHILD", arm=arm_label, raw_cells_sealed=len(missing), validation_e_model_calls=0)
        print(f"PI0180_CHILD_ARM_COMPLETE {arm_label} {len(missing)}")
        return 0
    except BaseException as exc:
        phase = current["phase"]
        if isinstance(exc, MechanicalInterruption):
            ledger.emit("INTERRUPTION", phase=phase, arm=arm_label, current=current, error=str(exc))
        else:
            ledger.emit("ERROR", phase=phase, arm=arm_label, current=current, error=f"{type(exc).__name__}: {exc}")
        append_jsonl(output_root / "child_failures.jsonl", {"schema": "STAGE45F_PI0180_CHILD_FAILURE_V1", "status": "INTERRUPTED" if isinstance(exc, MechanicalInterruption) else "ERROR", "arm": arm_label, "phase": phase, "current": current, "error": f"{type(exc).__name__}: {exc}", "validation_e_model_calls": 0})
        print(f"PI0180_CHILD_ARM_FAILED {arm_label}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 143 if isinstance(exc, MechanicalInterruption) else 2


def _freeze_arms(preflight_record: Mapping[str, Any]) -> list[dict[str, Any]]:
    audit = preflight_record.get("artifact_audit")
    require(isinstance(audit, Mapping), "preflight checkpoint audit missing")
    entries = audit.get("checkpoint_audit")
    require(isinstance(entries, list) and len(entries) == 6, "preflight checkpoint audit count drift")
    return [dict(item) for item in entries]


def _ledger_counts(ledger_path: Path) -> dict[str, int]:
    ledger = ExecutionLedger.__new__(ExecutionLedger)
    ledger.path = ledger_path
    records = ledger._records()
    return {
        "model_loads": sum(1 for item in records if item.get("event") == "BEGIN" and item.get("phase") == "MODEL_LOAD"),
        "conditioning": sum(1 for item in records if item.get("event") == "BEGIN" and item.get("phase") == "CONDITIONING"),
        "action_head": sum(1 for item in records if item.get("event") == "BEGIN" and item.get("phase") == "ACTION_HEAD"),
        "model_side_total": sum(1 for item in records if item.get("event") == "BEGIN" and item.get("phase") in {"CONDITIONING", "ACTION_HEAD"}),
        "raw_sealed": sum(1 for item in records if item.get("event") == "RAW_SEALED"),
    }


def _load_raw_records(output_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    records: dict[tuple[str, str], dict[str, Any]] = {}
    path = regular_file(output_root / "raw_predictions.jsonl", "raw manifest")
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        record = json.loads(line)
        require(isinstance(record, Mapping), f"raw record {line_number} is not an object")
        checkpoint = record.get("checkpoint")
        require(isinstance(checkpoint, Mapping), f"raw record {line_number} checkpoint missing")
        key = (str(checkpoint["label"]), str(record["window_id"]))
        require(key not in records, f"duplicate raw cell: {key}")
        records[key] = dict(record)
    return records


def _posthoc_windows(binding: Mapping[str, Any]) -> list[dict[str, Any]]:
    import numpy as np
    import stage45f_vanda_validation_e_gen1 as evaluator

    windows = evaluator.validate_snapshot_records(binding)
    result = []
    for item in windows:
        snapshot = {
            "expert_physical_action": item["expert_physical_action"],
            "anchor_world_m": item["anchor_world_m"],
            "target_world_m": item["target_world_m"],
        }
        result.append({**dict(item), "snapshot": snapshot, "state_v1": np.asarray(item["state_v1"], dtype=np.float32)})
    return result


def _validate_derived_manifest(output_root: Path, raw_records: Mapping[tuple[str, str], Mapping[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    path = output_root / "derived_rows.jsonl"
    if not path.exists():
        return {}
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        record = json.loads(line)
        require(isinstance(record, Mapping), f"derived record {line_number} is not an object")
        checkpoint = record.get("checkpoint")
        key = (str(checkpoint["label"]), str(record["window_id"]))
        require(key in raw_records and key not in result, f"derived cell invalid or duplicated: {key}")
        require(record.get("raw_path") == raw_records[key].get("path"), f"derived raw path drift: {key}")
        require(record.get("raw_file_sha256") == raw_records[key].get("file_sha256"), f"derived raw SHA drift: {key}")
        derived_file = regular_file(output_root / str(record["derived_path"]), f"derived file {key}")
        require(sha256_file(derived_file) == record.get("derived_file_sha256"), f"derived file SHA drift: {key}")
        result[key] = dict(record)
    return result


def cpu_posthoc(config: Mapping[str, Any], output_root: Path, preflight_record: Mapping[str, Any], ledger: ExecutionLedger) -> dict[str, Any]:
    import numpy as np
    import stage45c_h200_eval_runtime_adapter as runtime
    import stage45f_vanda_validation_e_gen1 as evaluator
    from go2arx5_vla.evaluation.checkpoint_eval import aggregate_evaluation_rows

    source_inventory = _source_inventory(config)
    inventory = _validate_output_raw(config, output_root, source_inventory)
    require(int(inventory["missing_raw_count"]) == 0, f"raw matrix incomplete: {inventory['missing_raw_count']} missing")
    ledger.emit("CPU_POSTHOC_OPEN", phase="CPU_POSTHOC", raw_count=36, model_calls=0)
    binding = load_json(regular_file(Path(str(config["admission_path"])), "Validation-E admission"))
    windows = _posthoc_windows(binding)
    raw_records = _load_raw_records(output_root)
    derived_records = _validate_derived_manifest(output_root, raw_records)
    rows_by_arm: dict[str, list[dict[str, Any]]] = {label: [] for label in EXPECTED_ARM_LABELS}
    runtime_sha = canonical_sha({"execution_head": config["expected_head"], "freeze_sha256": preflight_record["freeze"]["sha256"], "runtime": "CPU_POSTHOC_ONLY"})
    for label in EXPECTED_ARM_LABELS:
        arm = next(item for item in preflight_record["artifact_audit"]["checkpoint_audit"] if item["label"] == label)
        arm_identity = {"label": label, "route": "L1" if label.startswith("L1") else "Legacy", "seed": int(label.split("seed", 1)[1].split(")", 1)[0]), "tier": 2000, "tree_sha256": EXPECTED_CHECKPOINT_TREES[label]}
        for index, window in enumerate(windows):
            key = (label, window["window_id"])
            raw_record = raw_records[key]
            if key in derived_records:
                derived_file = output_root / str(derived_records[key]["derived_path"])
                row = load_json(derived_file)
                require(isinstance(row, Mapping), f"derived row is not an object: {key}")
                rows_by_arm[label].append(dict(row))
                continue
            ledger_token = ledger.begin("DECODE_DERIVED", arm=label, window_id=window["window_id"], device="cpu")
            raw_path = regular_file(output_root / str(raw_record["path"]), f"raw array {key}")
            raw_np = np.load(raw_path, allow_pickle=False)
            require(tuple(int(value) for value in raw_np.shape) == RAW_SHAPE and str(raw_np.dtype) == "float32" and bool(np.isfinite(raw_np).all()), f"raw array drift: {key}")
            physical, details = runtime._decode_v2(raw_np, window["state_v1"], Path(str(config["stats"])).resolve(), window["source_timestamps"], float(window["observation_timestamp"]), f"{window['window_id']}:{label}")
            row = evaluator._metric_row(runtime, raw_np, physical, details, window, arm_identity, runtime_sha, str(raw_record["path"]), str(raw_record["file_sha256"]))
            row["route"] = arm_identity["route"]
            row["seed"] = arm_identity["seed"]
            row["tier"] = 2000
            row["noise_sha256"] = raw_record["noise_sha256"]
            row["sampler"] = {"name": "direct-source-mapped-action-head-E4", "t_values": list(T_VALUES), "timestep_ids": list(TIMESTEP_IDS), "dt": 0.25, "carried_state": True, "reset_between_stages": False, "official_public_sampler_called": False, "get_action_calls": 0}
            row["stats"] = {"sha256": EXPECTED_STATS_SHA, "payload_sha256": EXPECTED_STATS_PAYLOAD_SHA, "route": "FULL24_RUNTIME_V2"}
            derived_file = output_root / derived_path(label, index)
            write_json(derived_file, row, overwrite=derived_file.exists())
            derived_record = {
                "schema": DERIVED_SCHEMA,
                "checkpoint": arm_identity,
                "window_id": window["window_id"],
                "raw_path": raw_record["path"],
                "raw_file_sha256": raw_record["file_sha256"],
                "derived_path": derived_file.relative_to(output_root).as_posix(),
                "derived_file_sha256": sha256_file(derived_file),
                "decoded": True,
                "raw_reused_without_model_call": key[0] == EXPECTED_ARM_LABELS[0],
                "source_allocation": 8 if key[0] == EXPECTED_ARM_LABELS[0] else 10,
            }
            append_jsonl(output_root / "derived_rows.jsonl", derived_record)
            derived_records[key] = derived_record
            rows_by_arm[label].append(row)
            ledger.end(ledger_token, arm=label, window_id=window["window_id"], derived_path=derived_record["derived_path"], derived_file_sha256=derived_record["derived_file_sha256"])
        require(len(rows_by_arm[label]) == 6, f"derived row count drift: {label}")
        aggregate = aggregate_evaluation_rows(rows_by_arm[label], expected_window_ids=list(EXPECTED_WINDOWS))
        aggregate.update({"route": arm_identity["route"], "seed": arm_identity["seed"], "tier": 2000, "checkpoint_tree_sha256": EXPECTED_CHECKPOINT_TREES[label], "model_loads": 0 if label == EXPECTED_ARM_LABELS[0] else 1, "model_forwards": 0 if label == EXPECTED_ARM_LABELS[0] else 24, "raw_reused_without_model_call": label == EXPECTED_ARM_LABELS[0]})
        aggregate_path = output_root / f"aggregate_{safe_label(label)}.json"
        if aggregate_path.exists():
            existing = load_json(aggregate_path)
            require(existing.get("window_count") == 6 and existing.get("window_ids") == list(EXPECTED_WINDOWS), f"aggregate drift: {label}")
        else:
            write_json(aggregate_path, aggregate)
        ledger.emit("ARM_AGGREGATE", phase="CPU_POSTHOC", arm=label, row_count=6, aggregate_path=aggregate_path.relative_to(output_root).as_posix(), aggregate_file_sha256=sha256_file(aggregate_path))
    paired = evaluator.paired_summary(rows_by_arm, [{"label": label, "route": "L1" if label.startswith("L1") else "Legacy", "seed": int(label.split("seed", 1)[1].split(")", 1)[0]), "tier": 2000} for label in EXPECTED_ARM_LABELS])
    seed_summary = []
    for seed in (2222, 2223, 2224):
        rows = paired["paired_by_seed"][str(seed)]
        fields = ("final_target_distance_delta_m", "progress_delta_m", "minimum_target_distance_delta_m", "path_length_delta_m", "absolute_roughness_delta_m")
        seed_summary.append({"seed": seed, "window_count": len(rows), "mean_l1_minus_legacy": {field: sum(float(row[field]) for row in rows) / len(rows) for field in fields}, "descriptive_only": True})
    write_json(output_root / "paired_summary.json", paired, overwrite=(output_root / "paired_summary.json").exists())
    target_summary = {
        "schema": "STAGE45F_PI0180_TARGET_DISTANCE_PROGRESS_PATH_ROUGHNESS_SUMMARY_V1",
        "descriptive_only": True,
        "arms": {label: [rows_by_arm[label][index]["metrics"]["target"] | {"window_id": rows_by_arm[label][index]["window_id"], "roughness": rows_by_arm[label][index]["metrics"]["prediction_roughness_position_second_difference_m"]} for index in range(6)] for label in EXPECTED_ARM_LABELS},
        "paired_by_seed": seed_summary,
        "target_excluded_from_model_input": True,
    }
    write_json(output_root / "target_progress_path_roughness_summary.json", target_summary, overwrite=(output_root / "target_progress_path_roughness_summary.json").exists())
    counts = _ledger_counts(output_root / "execution_ledger.jsonl")
    require(counts["model_loads"] <= NEW_CEILING["model_loads"] and counts["conditioning"] <= NEW_CEILING["conditioning"] and counts["action_head"] <= NEW_CEILING["action_head"] and counts["model_side_total"] <= NEW_CEILING["model_side_total"], "PI0180 ledger ceiling drift")
    final_inventory = _validate_output_raw(config, output_root, source_inventory)
    write_json(output_root / "final_raw_inventory.json", final_inventory, overwrite=(output_root / "final_raw_inventory.json").exists())
    evidence_entries = []
    excluded = {"evidence_inventory.json", "stage45f_pi0180_final_seal.json"}
    for directory, _, filenames in os.walk(output_root, followlinks=False):
        for filename in sorted(filenames):
            path = Path(directory) / filename
            relative = path.relative_to(output_root).as_posix()
            if relative in excluded or path.is_symlink():
                continue
            require(path.is_file(), f"evidence entry is not regular: {path}")
            evidence_entries.append({"path": relative, "bytes": int(path.stat().st_size), "sha256": sha256_file(path)})
    evidence_entries.sort(key=lambda item: item["path"])
    write_json(output_root / "evidence_inventory.json", {"schema": "STAGE45F_PI0180_EVIDENCE_INVENTORY_V1", "entries": evidence_entries, "entry_count": len(evidence_entries), "excluded": sorted(excluded)}, overwrite=(output_root / "evidence_inventory.json").exists())
    result = {
        "schema": SCHEMA,
        "status": "COMPLETED",
        "token": RETURN_TOKEN,
        "science_hold": SCIENCE_HOLD,
        "pi_gate": PI_GATE,
        "execution_branch": BRANCH,
        "execution_head": config["expected_head"],
        "output_root": str(output_root),
        "raw_cells": 36,
        "derived_cells": 36,
        "arm_aggregates": 6,
        "paired_cells": 18,
        "seed_summaries": 3,
        "actual_counters": counts,
        "checkpoint_arms": list(EXPECTED_ARM_LABELS),
        "raw_manifest": str(output_root / "raw_predictions.jsonl"),
        "raw_manifest_sha256": sha256_file(output_root / "raw_predictions.jsonl"),
        "derived_manifest": str(output_root / "derived_rows.jsonl"),
        "derived_manifest_sha256": sha256_file(output_root / "derived_rows.jsonl"),
        "execution_ledger": str(output_root / "execution_ledger.jsonl"),
        "gpu_runtime_admission": load_json(output_root / "gpu_runtime_admission.json"),
        "target_progress_path_roughness_summary": str(output_root / "target_progress_path_roughness_summary.json"),
        "paired_summary": str(output_root / "paired_summary.json"),
        "seed_paired_summary": seed_summary,
        "training": False,
        "backward": False,
        "optimizer_steps": 0,
        "checkpoint_writes": 0,
        "holdout_f_read": False,
        "validation_e_model_calls": 36,
        "e_s_training": 0,
        "icl": 0,
        "dagger": 0,
        "wbc": 0,
        "simulator": 0,
        "hardware": 0,
        "scientific_winner_selected": False,
    }
    write_json(output_root / "stage45f_pi0180_result.json", result, overwrite=(output_root / "stage45f_pi0180_result.json").exists())
    seal = {**result, "schema": "STAGE45F_PI0180_VALIDATION_E_GEN1_FINAL_SEAL_V1", "final_result_sha256": sha256_file(output_root / "stage45f_pi0180_result.json"), "evidence_inventory_sha256": sha256_file(output_root / "evidence_inventory.json"), "final_raw_inventory_sha256": sha256_file(output_root / "final_raw_inventory.json")}
    write_json(output_root / "stage45f_pi0180_final_seal.json", seal, overwrite=(output_root / "stage45f_pi0180_final_seal.json").exists())
    ledger.emit("FINAL_SEAL", phase="FINAL_SEAL", status="COMPLETED", result_path="stage45f_pi0180_result.json", result_file_sha256=sha256_file(output_root / "stage45f_pi0180_result.json"), raw_count=36, derived_count=36, science_hold=SCIENCE_HOLD)
    print(RETURN_TOKEN)
    print(SCIENCE_HOLD)
    return result


def execute(config_path: Path, *, continuation_mode: bool = False) -> int:
    config = _load_config(config_path)
    output_root, preflight_record, _runtime = _load_preflight(config)
    repo = Path(str(config.get("repo", REPO))).resolve()
    _checkout_audit(repo, str(config["expected_head"]))
    guard = static_guard()
    require(guard["status"] == "PASS", "PI0180 static guard failed at execution")
    source_inventory = _source_inventory(config)
    _validate_output_raw(config, output_root, source_inventory)
    if (output_root / "stage45f_pi0180_final_seal.json").exists():
        result = load_json(output_root / "stage45f_pi0180_result.json")
        require(result.get("status") == "COMPLETED", "existing final result is not completed")
        print(RETURN_TOKEN)
        print(SCIENCE_HOLD)
        return 0
    ledger_path = output_root / "execution_ledger.jsonl"
    prior_records = []
    if ledger_path.exists():
        prior_records = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    primary_runs = sum(1 for item in prior_records if item.get("event") == "EXECUTION_OPEN" and item.get("continuation") is False)
    continuation_runs = sum(1 for item in prior_records if item.get("event") == "EXECUTION_OPEN" and item.get("continuation") is True)
    if continuation_mode:
        require(primary_runs == 1 and continuation_runs == 0, "PI0180 continuation allocation is not the single authorized final continuation")
    else:
        require(primary_runs == 0 and continuation_runs == 0, "PI0180 primary allocation was already consumed")
    ledger = ExecutionLedger(ledger_path, generation="Validation-E-gen1", gpu_uuid=None)
    _record_runtime_admission(config, output_root, ledger)
    gpu_admission = _observe_h200_gpu(output_root)
    ledger.gpu_uuid = str(gpu_admission["gpu_uuid"])
    os.environ["PI0180_GPU_UUID"] = str(gpu_admission["gpu_uuid"])
    ledger.emit("EXECUTION_OPEN", phase="ADMISSION", continuation=continuation_mode, gpu_uuid=gpu_admission["gpu_uuid"], runtime_admission_status="PASS", qsub_called=False)
    for label in EXPECTED_ARM_LABELS:
        inventory = _validate_output_raw(config, output_root, source_inventory)
        missing = [(str(item["arm"]), str(item["window_id"])) for item in inventory["missing_cells"] if str(item["arm"]) == label]
        if not missing:
            ledger.emit("ARM_ALREADY_COMPLETE", phase="ARM_CHILD", arm=label, raw_cells=6)
            continue
        counts = _ledger_counts(output_root / "execution_ledger.jsonl")
        require(counts["model_loads"] < NEW_CEILING["model_loads"], "PI0180 model-load ceiling exhausted before child")
        command = [sys.executable, str(Path(__file__).resolve()), "--arm-child", "--config", str(config_path.resolve()), "--arm", label]
        child_env = dict(os.environ)
        child_env["PYTHONUNBUFFERED"] = "1"
        completed = subprocess.run(command, capture_output=True, text=True, check=False, env=child_env)
        log_base = output_root / "child_logs" / f"{safe_label(label)}_{counts['model_loads'] + 1}"
        write_text(log_base.with_suffix(".stdout.log"), completed.stdout)
        write_text(log_base.with_suffix(".stderr.log"), completed.stderr)
        ledger.emit("ARM_CHILD_EXIT", phase="ARM_CHILD", arm=label, returncode=completed.returncode, missing_cells_requested=len(missing), stdout_log=str(log_base.with_suffix(".stdout.log").relative_to(output_root)), stderr_log=str(log_base.with_suffix(".stderr.log").relative_to(output_root)))
        require(completed.returncode == 0, f"PI0180 child failed for {label}: {completed.stderr[-1200:]}")
        after = _validate_output_raw(config, output_root, source_inventory)
        require(not any(str(item["arm"]) == label for item in after["missing_cells"]), f"child did not seal all requested cells: {label}")
    return_code = 0
    try:
        cpu_posthoc(config, output_root, preflight_record, ledger)
    except Exception as exc:
        ledger.emit("ERROR", phase="CPU_POSTHOC", error=f"{type(exc).__name__}: {exc}")
        write_json(output_root / "stage45f_pi0180_failure.json", {"schema": "STAGE45F_PI0180_FAILURE_SEAL_V1", "status": "PARTIAL", "token": "LOCALPC_PI0180_VALIDATION_E_GEN1_FOR_PI_REVIEW — PARTIAL", "phase": "CPU_POSTHOC", "error": f"{type(exc).__name__}: {exc}", "actual_counters": _ledger_counts(output_root / "execution_ledger.jsonl"), "raw_inventory": _output_inventory(output_root), "training": False, "holdout_f_read": False})
        return_code = 2
    return return_code


def posthoc_only(config_path: Path) -> int:
    config = _load_config(config_path)
    output_root, preflight_record, _runtime = _load_preflight(config)
    source_inventory = _source_inventory(config)
    _validate_output_raw(config, output_root, source_inventory)
    ledger = ExecutionLedger(output_root / "execution_ledger.jsonl", generation="Validation-E-gen1", gpu_uuid=None)
    cpu_posthoc(config, output_root, preflight_record, ledger)
    return 0


def plan() -> dict[str, Any]:
    guard = static_guard()
    return {
        "schema": "STAGE45F_PI0180_VALIDATION_E_GEN1_PLAN_V1",
        "status": "PASS_STATIC_NOT_EXECUTED" if guard["status"] == "PASS" else "FAIL",
        "pi_gate": PI_GATE,
        "source_head": SOURCE_HEAD,
        "branch": BRANCH,
        "donor_head": DONOR_HEAD,
        "preserved": {"arm": EXPECTED_ARM_LABELS[0], "raw": 6, "model_loads": 0, "conditioning": 0, "action_head": 0},
        "new": NEW_NOMINAL,
        "new_ceiling": NEW_CEILING,
        "final": {"raw": 36, "derived": 36, "arm_aggregates": 6, "paired_cells": 18, "seed_summaries": 3},
        "fresh_child_per_arm": True,
        "raw_before_decode": True,
        "cpu_posthoc": True,
        "training": 0,
        "holdout_f_read": 0,
        "h200_submissions_by_runner": 0,
        "return_token": RETURN_TOKEN,
        "science_hold": SCIENCE_HOLD,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--plan-only", action="store_true")
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--arm-child", action="store_true")
    modes.add_argument("--posthoc-only", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--arm")
    parser.add_argument("--continuation", action="store_true")
    args = parser.parse_args(argv)
    if args.plan_only:
        print(json.dumps(plan(), indent=2, sort_keys=True))
        return 0
    require(args.config is not None, "selected mode requires --config")
    config_path = args.config.resolve(strict=True)
    if args.preflight:
        preflight(config_path, args.repo.resolve(strict=True))
        return 0
    if args.arm_child:
        require(args.arm is not None, "--arm-child requires --arm")
        return arm_child(config_path, args.arm)
    if args.posthoc_only:
        return posthoc_only(config_path)
    return execute(config_path, continuation_mode=args.continuation)


if __name__ == "__main__":
    raise SystemExit(main())
