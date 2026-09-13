#!/usr/bin/env python3
"""PI0164 controlled-supervision pilot.

This launcher is intentionally a thin, path-bound overlay on the accepted
Stage45E GR00T training path.  It keeps the model, StructuredActionV1 codec,
N2 statistics, optimizer, noise/timestep law, and trainable-module selection
unchanged.  The only experiment variable is the deterministic training-slot
source:

* E: 256 FULL24 slots;
* S: the same 256-slot sequence, with the odd zero-based slots replaced by one
  fixed, independently qualified q0 expert label.

The script is executable only with ``--mode train`` after all identity and
resource checks have been written to a fresh output root.  It never submits
PBS work and never enters WBC, simulator, ICL, Validation-E, or Holdout-F.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import stat
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
SRC = REPO / "src"
for _path in (SCRIPTS, SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

PI_GATE = "PI_GATE_STATE_0164"
PI_COMMENT = 5653126447
PR_COMMENT = 5653131645
BRANCH = "stage45f/corelab-local-initial-state-pilot"
PROTOCOL_BASE_HEAD = "cf4857666a47c9bd723e92842680e31c5e46c380"
EXPECTED_HEAD = PROTOCOL_BASE_HEAD
RETURN_TOKEN = "CORELAB_PI0164_CONTROLLED_SUPERVISION_PILOT_FOR_PI_REVIEW"

CHECKPOINT_TREE = "09bb76cb2b74a48644c120201042671fb7eaa42da3603ffb84c62f49a35d6152"
N2_STATS_SHA = "0c9450e6aeb0765eebc676414b2a6a84f71dddb7eea1de5453551f062b25fdbb"
N2_STATS_PAYLOAD_SHA = "508d4a8845e3bf5928416a506b95046ec0bb6c2be3eecbd09c39684a00865572"
BASE_TREE = "8717e322de3dd8c66e52bdb81b37badb0f235c518de6acd465532ed01569a6a4"
FULL24_TREE = "2f3badd2019a68c6549a66f85b5486faf626078a3e438342f67aacc4edc95789"
GROOT_COMMIT = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
COSMOS_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"
COSMOS_TREE = "1cfec77c944cba341ee5d7c31b4be6bb615582364bae48f18c5686e6ec23e8a9"
COSMOS_MODEL_DIR = "models--nvidia--Cosmos-Reason2-2B"
MODALITY_SHA = "e61f9429ecc21ceab51d9d26cadc9018691eb8b5c74d60aec4abf7faee16f031"
CONFIG_SHA = "9c3cd1d7bea603b95b1c777d8bcfebd61bd7eb5d86ca8ebbc3175f8141498a7c"
NOISE_ARRAY_SHA = "5924010a4e506159b654d0d6164125da5bf0e513ff410bf6e8e5f964a9ab50bd"
Q0_NPZ_SHA = "e0376e138bb1610a557164d3afffa3da1d7d0446a3b28406374360bbe7c441ad"
Q0_JSON_SHA = "a4d53a962af98b074b8a9f4a88a2b238e15254db362b2be06b2134d454c67e06"

TRAIN_SEED = 6164
MAX_STEPS = 256
SAVE_STEPS = 64
HORIZON = 40
ACTION_DIM = 132
STATE_DIM = 10
ACTIVE_SCALARS = 190
Q0_CONDITION = "C1_X_PLUS_5CM"
LANGUAGE = "move the end effector to the red target"
FFMPEG6_LIB = "/var/cache/users/liuyu-nfs/go2-arx5-wbc-stage13/stage13c/runtime_env/public_native/python_env/lib"


class PilotError(RuntimeError):
    """Fail-closed PI0164 error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PilotError(message)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def read_json(path: Path) -> Any:
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"required JSON is unavailable: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PilotError(f"cannot parse JSON {path}: {exc}") from exc


def write_json(path: Path, value: Any, *, overwrite: bool = False) -> str:
    path = Path(path)
    require(overwrite or not path.exists(), f"refusing to overwrite evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    return sha256_file(path)


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def tree_entries(root: Path, prefix: str) -> list[dict[str, Any]]:
    root = Path(root).resolve()
    require(root.is_dir() and not root.is_symlink(), f"tree root unavailable: {root}")
    entries: list[dict[str, Any]] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for dirname in dirnames:
            child = directory_path / dirname
            require(child.is_dir() and not child.is_symlink(), f"unsafe tree directory: {child}")
        for filename in filenames:
            child = directory_path / filename
            require(child.is_file() and not child.is_symlink(), f"unsafe tree file: {child}")
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
    for entry in sorted(entries, key=lambda item: str(item["path"])):
        digest.update(
            f"{entry['path']}\0{entry['bytes']}\0{entry['sha256']}\n".encode("utf-8")
        )
    return digest.hexdigest()


def _cosmos_tree_digest(root: Path) -> tuple[str, int]:
    root = Path(root).resolve()
    require(root.is_dir() and not root.is_symlink(), f"Cosmos root unavailable: {root}")
    entries: list[dict[str, Any]] = []

    def visit(directory: Path, relative: Path) -> None:
        for item in sorted(directory.iterdir(), key=lambda value: value.name):
            rel = (relative / item.name).as_posix()
            mode = item.lstat().st_mode
            if stat.S_ISDIR(mode):
                entries.append({"path": rel, "kind": "directory", "size_bytes": 0, "sha256": None})
                visit(item, relative / item.name)
            elif stat.S_ISLNK(mode):
                resolved = item.resolve(strict=True)
                require(resolved.is_file() and resolved.is_relative_to(root), f"Cosmos symlink escapes root: {item}")
                entries.append({"path": rel, "kind": "symlink", "target": os.readlink(item), "resolved_path": resolved.relative_to(root).as_posix(), "size_bytes": int(resolved.stat().st_size), "sha256": sha256_file(resolved)})
            elif stat.S_ISREG(mode):
                entries.append({"path": rel, "kind": "regular_file", "target": None, "resolved_path": None, "size_bytes": int(item.stat().st_size), "sha256": sha256_file(item)})
            else:
                raise PilotError(f"unsupported Cosmos entry: {item}")

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


def _cosmos_model_root(root: Path) -> Path:
    """Resolve either a HF hub cache root or its model-specific directory."""
    root = Path(root).resolve()
    nested = root / COSMOS_MODEL_DIR
    if nested.is_dir() and (nested / "refs" / "main").is_file():
        return nested
    return root


def gpu_snapshot() -> dict[str, Any]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    require(result.returncode == 0, f"nvidia-smi failed: {result.stderr[-1000:]}")
    rows = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields = [item.strip() for item in line.split(",")]
        require(len(fields) == 7, f"malformed nvidia-smi row: {line!r}")
        rows.append(
            {
                "index": int(fields[0]),
                "name": fields[1],
                "uuid": fields[2],
                "memory_total_mib": int(fields[3]),
                "memory_used_mib": int(fields[4]),
                "memory_free_mib": int(fields[5]),
                "utilization_gpu_percent": int(fields[6]),
            }
        )
    return {
        "hostname": subprocess.check_output(["hostname"], text=True).strip(),
        "query_sha256": sha256_bytes(result.stdout.encode("utf-8")),
        "query": result.stdout,
        "rows": rows,
    }


def select_gpu(snapshot: Mapping[str, Any], requested_uuid: str | None) -> dict[str, Any]:
    candidates = [
        row
        for row in snapshot["rows"]
        if "4080 SUPER" in str(row["name"]) and int(row["memory_total_mib"]) >= 30000
    ]
    if requested_uuid:
        candidates = [row for row in candidates if str(row["uuid"]) == str(requested_uuid)]
    require(candidates, f"no visible RTX 4080 SUPER resource: {snapshot['rows']}")
    candidates.sort(key=lambda row: (-int(row["memory_free_mib"]), int(row["index"])))
    selected = dict(candidates[0])
    # A model-side run must not start with an obviously unusable card.  This is
    # a resource gate only; it does not alter precision, batch, graph, or data.
    require(int(selected["memory_free_mib"]) >= 12000, f"insufficient free VRAM for exact graph: {selected}")
    return selected


def _external_identities(args: argparse.Namespace) -> dict[str, Any]:
    base = Path(args.base_model).resolve()
    dataset = Path(args.dataset).resolve()
    groot = Path(args.groot).resolve()
    cosmos = Path(args.cosmos).resolve()
    modality = Path(args.modality).resolve()
    config = Path(args.config).resolve()
    n2_stats = Path(args.n2_stats).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    for name, path in (("base_model", base), ("dataset", dataset), ("groot", groot), ("cosmos", cosmos), ("modality", modality), ("config", config), ("n2_stats", n2_stats), ("checkpoint", checkpoint)):
        require(path.exists() and (not path.is_symlink() or name == "cosmos"), f"required {name} path unavailable: {path}")
    base_tree = tree_digest(tree_entries(base, "payload/GR00T-N1.7-3B"))
    require(base_tree == BASE_TREE, f"BASE tree identity drift: {base_tree}")
    dataset_entries = tree_entries(dataset, "payload/lerobot_train")
    dataset_tree = tree_digest(dataset_entries)
    require(dataset_tree == FULL24_TREE, f"FULL24 tree identity drift: {dataset_tree}")
    require(len(dataset_entries) == 85 and sum(int(item["bytes"]) for item in dataset_entries) == 919661, "FULL24 file count/bytes drift")
    n2_sha = sha256_file(n2_stats)
    require(n2_sha == N2_STATS_SHA, f"N2 statistics SHA drift: {n2_sha}")
    cp_entries = tree_entries(checkpoint, "stage45e/checkpoint-2000")
    cp_tree = tree_digest(cp_entries)
    require(cp_tree == CHECKPOINT_TREE, f"N2 checkpoint tree drift: {cp_tree}")
    require(sha256_file(modality) == MODALITY_SHA, "modality identity drift")
    require(sha256_file(config) == CONFIG_SHA, "training config identity drift")
    groot_head = git_remote(groot, "rev-parse", "HEAD")
    require(groot_head == GROOT_COMMIT, f"GR00T source commit drift: {groot_head}")
    cosmos_model = _cosmos_model_root(cosmos)
    refs_main = cosmos_model / "refs" / "main"
    require(refs_main.is_file() and not refs_main.is_symlink(), f"Cosmos refs/main unavailable: {refs_main}")
    cosmos_revision = refs_main.read_text(encoding="utf-8").strip()
    require(cosmos_revision == COSMOS_REVISION, f"Cosmos revision drift: {cosmos_revision}")
    cosmos_tree, cosmos_count = _cosmos_tree_digest(cosmos_model)
    require(cosmos_tree == COSMOS_TREE, f"Cosmos tree drift: {cosmos_tree}")
    n2_doc = read_json(n2_stats)
    require(n2_doc.get("statistics_payload_sha256") == N2_STATS_PAYLOAD_SHA, "N2 statistics payload SHA drift")
    require(n2_doc.get("statistics", {}).get("active_scalar_count") == ACTIVE_SCALARS, "T16 active scalar count drift")
    return {
        "base_model": {"path": str(base), "tree_sha256": base_tree, "file_count": len(tree_entries(base, "payload/GR00T-N1.7-3B"))},
        "dataset": {"path": str(dataset), "tree_sha256": dataset_tree, "file_count": len(dataset_entries), "total_bytes": sum(int(item["bytes"]) for item in dataset_entries), "episodes": 24, "frames": 960},
        "n2_checkpoint": {"path": str(checkpoint), "tree_sha256": cp_tree, "file_count": len(cp_entries), "total_bytes": sum(int(item["bytes"]) for item in cp_entries)},
        "n2_stats": {"path": str(n2_stats), "sha256": n2_sha, "payload_sha256": N2_STATS_PAYLOAD_SHA},
        "modality_sha256": MODALITY_SHA,
        "training_config_sha256": CONFIG_SHA,
        "groot_commit": groot_head,
        "cosmos_revision": cosmos_revision,
        "cosmos_cache_root": str(cosmos),
        "cosmos_model_root": str(cosmos_model),
        "cosmos_tree_sha256": cosmos_tree,
        "cosmos_entry_count": cosmos_count,
    }


def git_remote(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True, stderr=subprocess.STDOUT).strip()


def _load_q0(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np

    npz = Path(args.q0_npz).resolve()
    meta = Path(args.q0_json).resolve()
    require(npz.is_file() and not npz.is_symlink(), f"q0 NPZ unavailable: {npz}")
    require(meta.is_file() and not meta.is_symlink(), f"q0 JSON unavailable: {meta}")
    npz_sha, meta_sha = sha256_file(npz), sha256_file(meta)
    require(npz_sha == Q0_NPZ_SHA, f"q0 NPZ SHA drift: {npz_sha}")
    require(meta_sha == Q0_JSON_SHA, f"q0 JSON SHA drift: {meta_sha}")
    document = read_json(meta)
    require(document.get("condition_id") == Q0_CONDITION, "q0 condition identity drift")
    require(document.get("expert_chunk_source") == "STAGE22_SAME_SCENE_EXPERT", "q0 expert source is not independent same-scene expert")
    require(document.get("target_excluded_from_model_input") is True, "q0 target exclusion witness missing")
    with np.load(npz, allow_pickle=False) as data:
        required = {"body_rgb", "wrist_rgb", "state_v1", "expert_physical_local", "expert_timestamps", "eef_local_m", "anchor_world_m", "target_world_m"}
        require(required.issubset(set(data.files)), f"q0 keys drift: {data.files}")
        body = np.asarray(data["body_rgb"])
        wrist = np.asarray(data["wrist_rgb"])
        state = np.asarray(data["state_v1"], dtype=np.float32).reshape(-1)
        expert = np.asarray(data["expert_physical_local"], dtype=np.float64)
        timestamps = np.asarray(data["expert_timestamps"], dtype=np.float64).reshape(-1)
        eef_local = np.asarray(data["eef_local_m"], dtype=np.float64).reshape(-1)
        anchor = np.asarray(data["anchor_world_m"], dtype=np.float64).reshape(-1)
        target = np.asarray(data["target_world_m"], dtype=np.float64).reshape(-1)
    require(body.shape == (96, 96, 3) and body.dtype == np.uint8, "q0 body ABI drift")
    require(wrist.shape == (96, 96, 3) and wrist.dtype == np.uint8, "q0 wrist ABI drift")
    require(state.shape == (STATE_DIM,) and expert.shape == (HORIZON, 10) and timestamps.shape == (HORIZON,), "q0 state/expert ABI drift")
    require(eef_local.shape == anchor.shape == target.shape == (3,), "q0 pose ABI drift")
    require(np.isfinite(state).all() and np.isfinite(expert).all() and np.isfinite(timestamps).all(), "q0 nonfinite numeric input")
    require(np.all(np.diff(timestamps) > 0.0), "q0 timestamps are not strictly increasing")
    require(np.allclose(eef_local, state[:3], atol=1e-5, rtol=0.0), "q0 local EEF does not match StateV1 prefix")
    return {
        "npz_path": str(npz),
        "json_path": str(meta),
        "npz_sha256": npz_sha,
        "json_sha256": meta_sha,
        "body": body,
        "wrist": wrist,
        "state": state,
        "expert": expert,
        "timestamps": timestamps,
        "anchor": anchor,
        "target": target,
        "metadata": document,
        "qualification": {
            "status": "PASS",
            "source": document.get("expert_chunk_source"),
            "chunk_sha256": document.get("expert_chunk_sha256"),
            "strictly_increasing_timestamps": True,
            "eef_local_matches_state_prefix": True,
            "target_excluded_from_model_input": True,
        },
    }


def prepare_environment(output: Path, gpu: Mapping[str, Any], cosmos: Path, n2_stats: Path) -> None:
    cache = output / "cache"
    tmp = output / "tmp"
    cache.mkdir(parents=True, exist_ok=False)
    tmp.mkdir(parents=True, exist_ok=False)
    hub = cache / "huggingface" / "hub"
    hub.mkdir(parents=True, exist_ok=False)
    (cache / "xdg").mkdir(parents=True, exist_ok=False)
    (hub / COSMOS_MODEL_DIR).symlink_to(_cosmos_model_root(cosmos), target_is_directory=True)
    old_ld = os.environ.get("LD_LIBRARY_PATH", "")
    ld = FFMPEG6_LIB if not old_ld else FFMPEG6_LIB + os.pathsep + old_ld
    os.environ.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu["index"]),
            "TMPDIR": str(tmp),
            "TMP": str(tmp),
            "TEMP": str(tmp),
            "XDG_CACHE_HOME": str(cache / "xdg"),
            "HF_HOME": str(cache / "huggingface"),
            "HF_HUB_CACHE": str(hub),
            "TRANSFORMERS_CACHE": str(hub),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "LD_LIBRARY_PATH": ld,
            "STAGE45B_LABEL_ANCHOR_RECONCILIATION": "1",
            "STAGE45E_N2_STATS_ARTIFACT": str(n2_stats),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )


def _load_config(config_path: Path, output: Path, checkpoint: Path, dataset: Path) -> Any:
    from stage45b_full24_fresh_base_train import _load_config as stock_load_config

    config = stock_load_config(config_path, output_root=output, base_model=checkpoint, dataset=dataset)
    config.data.seed = TRAIN_SEED
    config.training.max_steps = MAX_STEPS
    config.training.save_steps = SAVE_STEPS
    config.training.save_total_limit = 5
    config.training.global_batch_size = 1
    config.training.gradient_accumulation_steps = 1
    config.training.dataloader_num_workers = 0
    config.training.eval_strategy = "no"
    config.training.resume_from_checkpoint = False
    config.training.save_only_model = False
    config.training.optim = "adamw_torch"
    config.training.learning_rate = 1e-4
    config.training.weight_decay = 1e-5
    config.training.lr_scheduler_type = "cosine"
    config.training.warmup_steps = 10
    config.training.warmup_ratio = 0.0
    config.training.bf16 = True
    config.training.fp16 = False
    config.training.tf32 = True
    require(int(config.data.seed) == TRAIN_SEED, "training seed override failed")
    require(int(config.training.max_steps) == MAX_STEPS and int(config.training.save_steps) == SAVE_STEPS, "training budget override failed")
    return config


def _build_q_point(processor: Any, embodiment: Any, q0: Mapping[str, Any], stats_doc: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np
    from gr00t.data.types import MessageType, VLAStepData
    from go2arx5_vla.stage38_structured_training import StructuredActionNormalizer
    from go2arx5_vla.structured_action import encode_structured_action, reconcile_stage45b_label_anchor

    state = np.asarray(q0["state"], dtype=np.float32)
    step = VLAStepData(
        images={"external_rgb": [q0["body"]], "wrist_rgb": [q0["wrist"]]},
        states={"eef_9d": state[:9][None, :], "gripper_position": state[9:10][None, :]},
        actions={},
        text=LANGUAGE,
        embodiment=embodiment,
    )
    processed = processor([{"type": MessageType.EPISODE_STEP.value, "content": step}])
    label_state, reconciliation = reconcile_stage45b_label_anchor(q0["expert"], state)
    label = encode_structured_action(
        q0["expert"],
        q0["timestamps"],
        label_state,
        timing_mode="T16",
        chunk_id="PI0164:C1_X_PLUS_5CM:q0:expert",
        source_frame="NATIVE_WBC_RELATIVE_REFERENCE",
        tcp_semantic_id="x5_link6_gripper_center",
        observation_timestamp=float(q0["timestamps"][0]),
        provenance={
            "stage": "PI0164",
            "source": "PI0163_C1_q0_independent_same_scene_expert",
            "q0_source_sha256": q0["npz_sha256"],
            "label_side_anchor_reconciliation": reconciliation,
        },
    )
    normalizer = StructuredActionNormalizer(stats_doc.get("statistics", stats_doc), clip_outliers=True)
    processed["action"] = normalizer.normalize(label)
    processed["action_mask"] = label.mask.astype(np.float32, copy=True)
    require(processed["action"].shape == (HORIZON, ACTION_DIM), "Q action shape drift")
    require(processed["action_mask"].shape == (HORIZON, ACTION_DIM), "Q action mask shape drift")
    require(int(np.count_nonzero(processed["action_mask"])) == ACTIVE_SCALARS, "Q T16 mask drift")
    require(np.isfinite(processed["action"]).all(), "Q normalized action is nonfinite")
    return processed


def _make_slot_sequence() -> list[int]:
    # Keep the protocol generator dependency-free so admission/static checks
    # can run from the system Python.  The RNG and resulting sequence are
    # recorded in the manifest; this is the sole deterministic sampler law
    # used by both arms.
    rng = random.Random(TRAIN_SEED)
    order = list(range(24))
    slots: list[int] = []
    while len(slots) < MAX_STEPS:
        cycle = list(order)
        rng.shuffle(cycle)
        slots.extend(int(item) for item in cycle)
    return slots[:MAX_STEPS]


def _install_dataset_factory(config: Any, q0: Mapping[str, Any], stats_doc: Mapping[str, Any], arm: str) -> dict[str, Any]:
    """Patch only the GR00T setup DatasetFactory symbol for this run."""

    import numpy as np
    import torch
    from go2arx5_vla.stage38_structured_training import build_structured_gr00t_dataset_factory
    from gr00t.model.gr00t_n1d7 import setup as gr00t_setup

    base_factory_type = build_structured_gr00t_dataset_factory()
    slots = _make_slot_sequence()
    slot_sha = sha256_bytes((json.dumps(slots, separators=(",", ":")) + "\n").encode("utf-8"))

    class FixedSequenceDataset(torch.utils.data.IterableDataset):
        def __init__(self, base_mixture: Any, base_dataset: Any, processor: Any, embodiment: Any):
            super().__init__()
            self.base_mixture = base_mixture
            self.base_dataset = base_dataset
            self.processor = processor
            self.embodiment = embodiment
            self.seed = TRAIN_SEED
            self.sequence = tuple(slots)
            self.arm = arm
            self._q0 = q0
            self._stats = stats_doc
            self._base_refs = tuple((idx, 0) for idx in range(24))
            self._q_cache: dict[str, Any] | None = None
            self.structured_action_statistics = getattr(base_mixture, "structured_action_statistics", None)
            self.structured_action_contract = getattr(base_mixture, "structured_action_contract", None)

        def __len__(self) -> int:
            return MAX_STEPS

        def reset_seed(self, seed: int) -> None:
            require(int(seed) == TRAIN_SEED, f"unexpected trainer seed reset: {seed}")
            self.seed = int(seed)

        def get_dataset_statistics(self) -> Any:
            return self.base_mixture.get_dataset_statistics()

        def get_initial_actions(self) -> list[Any]:
            if hasattr(self.base_mixture, "get_initial_actions"):
                return self.base_mixture.get_initial_actions()
            return []

        def _base_point(self, index: int) -> Mapping[str, Any]:
            episode_index, step_index = self._base_refs[int(index)]
            episode_data = self.base_dataset.episode_loader[int(episode_index)]
            return self.base_dataset.get_datapoint(episode_data, int(step_index), episode_index=int(episode_index))

        def _q_point(self) -> Mapping[str, Any]:
            # Build a fresh q point at each occurrence, matching the processor
            # boundary used for native TRAIN points while keeping the q label
            # fixed and source-bound.
            return _build_q_point(self.processor, self.embodiment, self._q0, self._stats)

        def __iter__(self):
            for position, index in enumerate(self.sequence):
                use_q = self.arm == "S" and (position % 2 == 1)
                point = self._q_point() if use_q else self._base_point(index)
                yield copy.deepcopy(point)

    class PI0164Factory:
        def __init__(self, config: Any):
            self.config = config

        def build(self, processor: Any):
            base_mixture, _ = base_factory_type(self.config).build(processor)
            require(len(base_mixture.datasets) == 1, "PI0164 requires exactly one FULL24 dataset")
            base_dataset = base_mixture.datasets[0]
            sequence_dataset = FixedSequenceDataset(base_mixture, base_dataset, processor, getattr(base_dataset, "embodiment_tag", None))
            return sequence_dataset, None

    gr00t_setup.DatasetFactory = PI0164Factory
    return {
        "arm": arm,
        "sequence_length": len(slots),
        "sequence_sha256": slot_sha,
        "sequence_prefix": slots[:24],
        "full24_slots": int(MAX_STEPS if arm == "E" else MAX_STEPS // 2),
        "q_slots": int(0 if arm == "E" else MAX_STEPS // 2),
        "q_source": "PI0163_C1_X_PLUS_5CM_q0" if arm == "S" else None,
        "q_label_stats_route": "N2_BALANCED_XYZ",
        "q_target_xyz_in_model_input": False,
        "dataset_factory": "PI0164Factory(FULL24 base + deterministic overlay)",
    }


def _checkpoint_record(path: Path, expected_step: int) -> dict[str, Any]:
    require(path.is_dir() and not path.is_symlink(), f"checkpoint unavailable: {path}")
    state = read_json(path / "trainer_state.json")
    entries = tree_entries(path, f"stage45f_pi0164/{path.name}")
    required = ["optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json"]
    missing = [name for name in required if not (path / name).is_file() or (path / name).is_symlink()]
    return {
        "path": str(path),
        "global_step": state.get("global_step"),
        "expected_step": expected_step,
        "tree_sha256": tree_digest(entries),
        "file_count": len(entries),
        "total_bytes": sum(int(item["bytes"]) for item in entries),
        "required_state_files_present": not missing,
        "missing_state_files": missing,
        "status": "PASS" if state.get("global_step") == expected_step and not missing else "BLOCKED",
    }


def _parameter_sample(model: Any) -> dict[str, Any]:
    records = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        flat = parameter.detach().reshape(-1).float().cpu().tolist()
        records.append({"name": name, "numel": int(parameter.numel()), "sample": flat[:128]})
        if len(records) >= 8:
            break
    require(records, "no trainable parameters exposed")
    return {"records": records}


def train_arm(
    args: argparse.Namespace,
    arm: str,
    root: Path,
    identities: Mapping[str, Any],
    gpu: Mapping[str, Any],
    q0: Mapping[str, Any],
    repository: Mapping[str, Any],
) -> dict[str, Any]:
    """Run one fresh-optimizer arm; exceptions are sealed and never retried here."""
    output = root / arm
    require(not output.exists(), f"arm output already exists: {output}")
    output.mkdir(parents=True, exist_ok=False)
    stats_doc = read_json(Path(args.n2_stats))
    binding = {
        "schema": "CORELAB_PI0164_TRAINING_BINDING_V1",
        "authority": {"pi_gate": PI_GATE, "issue_comment": PI_COMMENT, "pr_comment": PR_COMMENT},
        "arm": arm,
        "repository": dict(repository),
        "checkpoint": identities["n2_checkpoint"],
        "stats": identities["n2_stats"],
        "recipe": {
            "seed": TRAIN_SEED,
            "max_steps": MAX_STEPS,
            "save_steps": [64, 128, 192, 256],
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "optimizer": "adamw_torch",
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "scheduler": "cosine",
            "warmup_steps": 10,
            "bf16": True,
            "fp16": False,
            "tf32": True,
            "workers": 0,
            "trainable_modules": ["projector", "diffusion"],
            "frozen_modules": ["visual", "llm", "vlln"],
            "loss": "masked StructuredActionV1 flow matching / T16 / 190 active scalars",
        },
        "source_overlay": {"arm": arm, "full24_dataset_tree_sha256": FULL24_TREE, "q0_npz_sha256": q0["npz_sha256"], "q0_json_sha256": q0["json_sha256"]},
        "execution_boundary_before_model": {"gpu": False, "model_construction": 0, "model_forward": 0, "training": False, "backward": False, "optimizer_steps": 0, "Validation-E": 0, "Holdout-F": 0, "WBC": 0, "simulator": 0, "hardware": False, "PBS": 0},
    }
    write_json(output / "training_binding.json", binding)
    prepare_environment(output, gpu, Path(args.cosmos).resolve(), Path(args.n2_stats).resolve())
    # Runtime-only imports begin after admission and environment binding.
    sys.path.insert(0, str(Path(args.groot).resolve()))
    from stage45b_full24_fresh_base_train import _bind_cosmos_processor

    cosmos_snapshot = _cosmos_model_root(Path(args.cosmos).resolve()) / "snapshots" / COSMOS_REVISION
    require(cosmos_snapshot.is_dir(), f"Cosmos snapshot unavailable: {cosmos_snapshot}")
    _bind_cosmos_processor(cosmos_snapshot)
    config = _load_config(Path(args.config).resolve(), output, Path(args.checkpoint).resolve(), Path(args.dataset).resolve())
    overlay = _install_dataset_factory(config, q0, stats_doc, arm)
    write_json(output / "sampler_overlay.json", {"schema": "CORELAB_PI0164_DETERMINISTIC_SLOT_OVERLAY_V1", **overlay, "seed": TRAIN_SEED, "no_eval_x0_used_for_training": True, "paired_e_slots_identical": True})

    # Each arm is a fresh optimizer/scheduler/RNG run under the fixed PI seed.
    # The official trainer also seeds its dataloader, but explicit process-local
    # seeding here makes the fresh-arm boundary observable and deterministic.
    import torch

    random.seed(TRAIN_SEED)
    torch.manual_seed(TRAIN_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(TRAIN_SEED)

    # Count complete model calls even if a later save/serialization operation
    # fails.  The count is observational only and cannot change the budget.
    counter = {"model_forward": 0}
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7
    original_forward = Gr00tN1d7.forward

    def counted_forward(model: Any, inputs: dict):
        counter["model_forward"] += 1
        return original_forward(model, inputs)

    Gr00tN1d7.forward = counted_forward
    try:
        from gr00t.experiment.experiment import run as gr00t_run

        gr00t_run(config)
    except Exception as exc:
        checkpoint_steps = []
        for candidate in sorted(output.glob("checkpoint-*")):
            state_path = candidate / "trainer_state.json"
            if state_path.is_file():
                try:
                    checkpoint_steps.append(int(read_json(state_path).get("global_step", -1)))
                except Exception:
                    checkpoint_steps.append(-1)
        failure = {
            "schema": "CORELAB_PI0164_TRAINING_FAILURE_SEAL_V1",
            "status": "BLOCKED",
            "arm": arm,
            "error_type": type(exc).__name__,
            "error": str(exc)[-4000:],
            "traceback": traceback.format_exc(),
            "model_forward_calls_observed": int(counter["model_forward"]),
            "checkpoint_global_steps_observed": checkpoint_steps,
            "training_updates_observed": max(checkpoint_steps, default=0),
            "optimizer_steps_observed": max(checkpoint_steps, default=0),
            "execution_boundary": {"gpu": True, "model_construction": 1, "model_forward": int(counter["model_forward"]), "training": bool(counter["model_forward"]), "backward": bool(counter["model_forward"]), "optimizer_steps": max(checkpoint_steps, default=0), "Validation-E": 0, "Holdout-F": 0, "WBC": 0, "simulator": 0, "hardware": False, "PBS": 0},
            "retry_or_step_extension": False,
            "timestamp_utc": now_utc(),
        }
        write_json(output / "FAILURE_SEAL.json", failure)
        raise
    finally:
        Gr00tN1d7.forward = original_forward

    checkpoints = {str(step): _checkpoint_record(output / f"checkpoint-{step}", step) for step in (64, 128, 192, 256)}
    require(all(item["status"] == "PASS" for item in checkpoints.values()), f"{arm} checkpoint state is incomplete: {checkpoints}")
    result = {
        "schema": "CORELAB_PI0164_TRAINING_ARM_RESULT_V1",
        "status": "PASS",
        "arm": arm,
        "starting_checkpoint_tree_sha256": CHECKPOINT_TREE,
        "training_seed": TRAIN_SEED,
        "optimizer_steps": MAX_STEPS,
        "backward_steps": MAX_STEPS,
        "model_forward_calls_observed": int(counter["model_forward"]),
        "checkpoint_writes": len(checkpoints),
        "checkpoints": checkpoints,
        "sampler_overlay": overlay,
        "gpu": dict(gpu),
        "execution_boundary": {"gpu": True, "model_construction": 1, "model_forward": int(counter["model_forward"]), "training": True, "backward": MAX_STEPS, "optimizer_steps": MAX_STEPS, "Validation-E": 0, "Holdout-F": 0, "WBC": 0, "simulator": 0, "hardware": False, "PBS": 0},
        "timestamp_utc": now_utc(),
    }
    write_json(output / "training_result.json", result)
    return result


def static_self_check() -> dict[str, Any]:
    import ast

    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    module_imports: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            module_imports.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            module_imports.append(node.module.split(".")[0])
    require("torch" not in module_imports, "static prefix imports torch")
    require("gr00t" not in module_imports, "static prefix imports GR00T")
    require("transformers" not in module_imports, "static prefix imports transformers")
    slots = _make_slot_sequence()
    require(len(slots) == MAX_STEPS and all(0 <= int(item) < 24 for item in slots), "slot sequence ABI drift")
    require(sha256_bytes((json.dumps(slots, separators=(",", ":")) + "\n").encode()) == sha256_bytes((json.dumps(_make_slot_sequence(), separators=(",", ":")) + "\n").encode()), "slot sequence is not deterministic")
    return {"schema": "CORELAB_PI0164_STATIC_SELF_CHECK_V1", "status": "PASS", "slot_count": len(slots), "sequence_sha256": sha256_bytes((json.dumps(slots, separators=(",", ":")) + "\n").encode()), "model_forward": 0, "training": 0, "gpu": False}


def _repository_binding(args: argparse.Namespace) -> dict[str, str]:
    """Bind the authorized source revision, including node-local materializations.

    A remote runtime may receive this runner in a sealed node-local snapshot
    without the Git checkout itself.  In that case the caller supplies the
    already-admitted source HEAD and the protocol base is recorded alongside
    it.  A normal checkout keeps the stricter branch/clean-tree/ancestry checks
    below.
    """
    if args.repository_head:
        head = str(args.repository_head)
        require(len(head) == 40 and all(character in "0123456789abcdef" for character in head.lower()), f"invalid materialized source HEAD: {head}")
        return {"branch": BRANCH, "head": head, "protocol_base_head": PROTOCOL_BASE_HEAD, "source_mode": "sealed_node_local_materialization"}
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    require(branch == BRANCH, f"branch mismatch: {branch}")
    ancestry = subprocess.run(
        ["git", "-C", str(REPO), "merge-base", "--is-ancestor", PROTOCOL_BASE_HEAD, head],
        check=False,
    )
    require(ancestry.returncode == 0, f"PI0164 source is not descended from protocol base {PROTOCOL_BASE_HEAD}: {head}")
    require(git("status", "--porcelain=v1") == "", "working tree must be clean before training")
    return {"branch": branch, "head": head, "protocol_base_head": PROTOCOL_BASE_HEAD, "source_mode": "git_checkout"}


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.mode == "static":
        return static_self_check()
    require(args.mode == "train", f"unsupported mode: {args.mode}")
    repository = _repository_binding(args)
    head = repository["head"]
    identities = _external_identities(args)
    q0 = _load_q0(args)
    snapshot = gpu_snapshot()
    gpu = select_gpu(snapshot, args.gpu_uuid)
    root = Path(args.output_root).resolve()
    require(not root.exists() and not root.is_symlink(), f"pilot output root must be fresh: {root}")
    root.mkdir(parents=True, exist_ok=False)
    admission = {
        "schema": "CORELAB_PI0164_TRAINING_PILOT_ADMISSION_V1",
        "status": "ADMITTED_FOR_DIRECT_EXECUTION",
        "authority": {"pi_gate": PI_GATE, "issue_comment": PI_COMMENT, "pr_comment": PR_COMMENT},
        "repository": {**repository, "expected_head": EXPECTED_HEAD},
        "identities": identities,
        "q0": {key: value for key, value in q0.items() if key not in {"body", "wrist", "state", "expert", "timestamps", "anchor", "target"}},
        "q0_qualification": q0["qualification"],
        "training": {"arms": ["E", "S"], "updates_per_arm": MAX_STEPS, "total_nominal_updates": 2 * MAX_STEPS, "seed": TRAIN_SEED, "q_policy": "qualified C1 q0; fallback TRAIN-a/expert-0000 only if qualification fails", "q_fallback_used": False},
        "resource": {"selected": gpu, "snapshot": snapshot, "shared_gpu": True, "no_process_interference": True},
        "execution_boundary_before_training": {"gpu": False, "model_construction": 0, "model_forward": 0, "training": False, "backward": False, "optimizer_steps": 0, "Validation-E": 0, "Holdout-F": 0, "WBC": 0, "simulator": 0, "hardware": False, "PBS": 0},
        "timestamp_utc": now_utc(),
    }
    write_json(root / "admission.json", admission)
    arm_results = {}
    # E and S are independent fresh-optimizer arms.  A failure is sealed and
    # stops the pilot; no automatic retry, seed/noise substitution, or step
    # extension is permitted by PI0164.
    for arm in ("E", "S"):
        arm_results[arm] = train_arm(args, arm, root, identities, gpu, q0, repository)
    result = {
        "schema": "CORELAB_PI0164_TRAINING_PILOT_RESULT_V1",
        "status": "TRAINING_COMPLETE_PENDING_FR_EVALUATION",
        "return_token": RETURN_TOKEN,
        "authority": admission["authority"],
        "repository": admission["repository"],
        "admission_sha256": sha256_file(root / "admission.json"),
        "arm_results": arm_results,
        "execution_boundary": {"gpu": True, "model_construction": 2, "model_forward": 2 * MAX_STEPS, "training": True, "backward": 2 * MAX_STEPS, "optimizer_steps": 2 * MAX_STEPS, "Validation-E": 0, "Holdout-F": 0, "WBC": 0, "simulator": 0, "hardware": False, "PBS": 0},
        "timestamp_utc": now_utc(),
    }
    write_json(root / "pilot_training_result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("static", "train"), default="static")
    parser.add_argument("--base-model", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--groot", type=Path)
    parser.add_argument("--cosmos", type=Path)
    parser.add_argument("--modality", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--n2-stats", type=Path)
    parser.add_argument("--q0-npz", type=Path)
    parser.add_argument("--q0-json", type=Path)
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--repository-head", help="Authorized source HEAD when running from a sealed node-local materialization")
    args = parser.parse_args()
    try:
        value = run(args)
    except Exception as exc:
        print(f"PI0164_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(jsonable(value), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
