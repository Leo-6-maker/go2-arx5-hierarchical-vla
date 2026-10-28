#!/usr/bin/env python3
"""PI0244's fixed 48-cell teacher-forced direct-flow sidecar.

Action construction, training-boundary conditioning and metrics are delegated
to the sealed PI0242/Stage26 helpers; this file binds only PI0244 identities.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
import uuid
from pathlib import Path
from typing import Any


MODELS = ("initializer", "pi0243_a5000", "pi0244_r010000")
FAMILIES = ("a", "b", "c", "d")
QUANTILES = (0.0, 0.1, 0.5, 0.9)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def append_event(path: Path, attempt_id: str, event: str, **fields: Any) -> None:
    row = {"attempt_id": attempt_id, "event": event, "unix_time": time.time(), **fields}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def load_bridge(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("pi0242_bridge_for_pi0244", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import sealed PI0242 helpers: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def verify_inventory(root: Path, entries: list[dict[str, Any]]) -> dict[str, Any]:
    expected = {str(row["path"]): row for row in entries}
    actual: set[str] = set()
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs:
            path = Path(directory) / name
            if path.is_symlink():
                raise RuntimeError(f"checkpoint contains a symlink directory: {path}")
        for name in files:
            path = Path(directory) / name
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f"checkpoint contains a non-regular file: {path}")
            relative = path.relative_to(root).as_posix()
            actual.add(relative)
            row = expected.get(relative)
            if row is None or path.stat().st_size != int(row["bytes"]):
                raise RuntimeError(f"checkpoint inventory path/size mismatch: {relative}")
            if sha256(path) != str(row["sha256"]):
                raise RuntimeError(f"checkpoint file hash mismatch: {relative}")
    if actual != set(expected):
        raise RuntimeError(f"checkpoint inventory mismatch: missing={sorted(set(expected)-actual)} extra={sorted(actual-set(expected))}")
    return {"file_count": len(actual), "total_bytes": sum(int(x["bytes"]) for x in entries)}


def cell_name(model: str, row: dict[str, Any], t_index: int) -> str:
    source = str(row["source_episode_id"])
    tag = hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]
    return f"{model}__{row['target_family']}__{tag}__t{t_index}"


def self_test() -> None:
    rows = [{"target_family": family, "source_episode_id": family * 8} for family in FAMILIES]
    cells = [cell_name(model, row, t) for model in MODELS for row in rows for t in range(4)]
    assert len(cells) == 48 and len(set(cells)) == 48
    assert QUANTILES == (0.0, 0.1, 0.5, 0.9)
    print("PI0244_R0_DIRECT_FLOW_SELF_TEST=PASS cells=48 get_action=0")


def make_plan(args: argparse.Namespace, bridge: Any) -> dict[str, Any]:
    import numpy as np

    root = args.output_root.resolve()
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"sidecar output root must be new/empty: {root}")
    root.mkdir(parents=True, exist_ok=True)

    pi_card = read_json(args.pi0241_card)
    r0_card = read_json(args.r0_card)
    r0_closure = read_json(args.r0_closure)
    r0_training = read_json(args.r0_training_manifest)
    if r0_closure.get("status") != "SCIENCE_FAIL_M16_10K_BUDGET" or r0_closure.get("selected_checkpoint") != "NONE":
        raise RuntimeError("sidecar is not bound to the sealed R0 final-only failure")
    if r0_training.get("status") != "PASS_R0_TRAINING_CLOSED" or r0_card.get("selected_checkpoint") != "NONE":
        raise RuntimeError("R0 training/card closure identity is not valid")
    cohort = read_json(args.cohort)
    rows = cohort["rows"]
    if len(rows) != 16 or {str(x["target_family"]) for x in rows} != set(FAMILIES):
        raise RuntimeError("R0 cohort is not the fixed 16-row TRAIN A-D cohort")
    selected = bridge._bridge4_rows(rows)
    selected_ids = [str(x["source_episode_id"]) for x in selected]
    prior_card = read_json(args.p1_run_card)
    if [x["source_episode_id"] for x in prior_card["bridge4_rows"]] != selected_ids:
        raise RuntimeError("current R0 bridge4 differs from the sealed PI0242 bridge4")

    input_by_id = {str(x["source_episode_id"]): x for x in pi_card["selected_inputs"]}
    if not set(selected_ids).issubset(input_by_id):
        raise RuntimeError("bridge4 source missing from sealed PI0241 input inventory")
    for source in selected_ids:
        item = input_by_id[source]
        if sha256(Path(item["tensor_path"])) != item["tensor_sha256"]:
            raise RuntimeError(f"fixed bridge tensor hash mismatch: {source}")

    initializer_entries = list(pi_card["initializer"]["files"].items())
    initializer_inventory = [
        {"path": name, "bytes": int(meta["bytes"]), "sha256": str(meta["sha256"])}
        for name, meta in initializer_entries
    ]
    init_inventory = verify_inventory(args.initializer, initializer_inventory)

    a_train = read_json(args.a5000_manifest)
    a_manifest = a_train["checkpoints"]["5000"]["inventory"]
    a_inventory = verify_inventory(args.a5000, a_manifest["entries"])
    if a_inventory["file_count"] != int(a_manifest["file_count"]):
        raise RuntimeError("PI0243 A5000 file count differs from its sealed manifest")

    r0_checkpoint = r0_training["checkpoints"]["10000"]
    hf_manifest = read_json(args.r010000_manifest)
    if hf_manifest["revision"] != r0_checkpoint["hf_manifest_commit_id"]:
        raise RuntimeError("R0 checkpoint revision differs from the training receipt")
    final_inventory = verify_inventory(args.r010000, hf_manifest["files"])

    p1_records = bridge._load_records(args.p1_records)
    noise_by_source: dict[str, str] = {}
    with np.load(args.p1_arrays, allow_pickle=False) as old_arrays:
        for row in selected:
            source = str(row["source_episode_id"])
            hits = [
                x for x in p1_records
                if x.get("kind") == "direct_flow" and x.get("model") == "initializer"
                and x.get("condition") == "C_T" and x.get("source_episode_id") == source
            ]
            if len(hits) != 4 or len({x["noise_sha256"] for x in hits}) != 1:
                raise RuntimeError(f"sealed fixed-noise coverage mismatch: {source}")
            noise = old_arrays[hits[0]["cell_key"] + "__noise"]
            noise_hash = bridge.sha256_array(noise)
            if noise_hash != hits[0]["noise_sha256"] or noise_hash != prior_card["noise_sha256_by_source_episode"][source]:
                raise RuntimeError(f"sealed paired-noise hash mismatch: {source}")
            noise_by_source[source] = noise_hash

    build_args = argparse.Namespace(code_root=args.code_root, output_root=root)
    conditions = bridge._build_p1_conditions(build_args, pi_card, selected)
    ct = conditions["C_T"]
    if set(ct) != set(selected_ids):
        raise RuntimeError("training-pipeline C_T rows differ from fixed bridge4")
    for row in selected:
        source = str(row["source_episode_id"])
        prior_ct = [
            item for item in p1_records
            if item.get("kind") == "direct_flow" and item.get("model") == "initializer"
            and item.get("condition") == "C_T" and item.get("source_episode_id") == source
        ]
        if len(prior_ct) != 4:
            raise RuntimeError(f"missing sealed prior C_T direct-flow cells: {source}")
        condition_hashes = {item.get("conditioning_hash", item.get("condition_hash")) for item in prior_ct}
        expert_hashes = {item.get("expert_normalized_sha256") for item in prior_ct}
        mask_hashes = {item.get("action_mask_sha256") for item in prior_ct}
        if condition_hashes != {str(ct[source]["condition_hash"])}:
            raise RuntimeError(f"current training-boundary conditioning differs from sealed P1 C_T: {source}")
        if expert_hashes != {bridge.sha256_array(ct[source]["expert_normalized"])}:
            raise RuntimeError(f"current normalized TRAIN action differs from sealed P1 C_T: {source}")
        if mask_hashes != {bridge.sha256_array(ct[source]["action_mask"])}:
            raise RuntimeError(f"current action mask differs from sealed P1 C_T: {source}")
    stage23 = bridge._load_script(
        args.code_root / "scripts" / "stage23_phase_c_flow_field_diagnostic.py",
        "stage23_for_pi0244_r0_direct_plan",
    )
    stage23.assert_protocol_static()
    time_values = [float(stage23.official_t_quantile(q)) for q in QUANTILES]
    cells = [
        {
            "cell_key": cell_name(model, row, ti),
            "model": model,
            "source_episode_id": str(row["source_episode_id"]),
            "target_family": str(row["target_family"]),
            "phase": int(row["phase"]),
            "t_index": ti,
            "t_quantile": q,
            "t_value": time_values[ti],
        }
        for model in MODELS for row in selected for ti, q in enumerate(QUANTILES)
    ]
    if len(cells) != 48 or len({x["cell_key"] for x in cells}) != 48:
        raise RuntimeError("pre-registered direct-flow matrix is not 48 unique cells")

    source_files = {
        "sidecar": Path(__file__).resolve(),
        "pi0242_helper": args.bridge_script,
        "training_adapter": args.code_root / "scripts" / "pi0241_fixed16_learning.py",
        "stage21_inference": args.code_root / "scripts" / "stage21_same_scene_groot_inference.py",
        "stage23_time": args.code_root / "scripts" / "stage23_phase_c_flow_field_diagnostic.py",
        "stage26_flow": args.code_root / "scripts" / "stage26_h1_c1_fixed_noise_local_flow.py",
    }
    plan = {
        "schema": "PI0244_R0_DIRECT_FLOW_RUN_CARD_V1",
        "protocol_commit": "4117f401778067e183385508c668686d1798120f",
        "r0_closure_sha256": sha256(args.r0_closure),
        "r0_training_manifest_sha256": sha256(args.r0_training_manifest),
        "r0_run_card_sha256": sha256(args.r0_card),
        "cohort_sha256": sha256(args.cohort),
        "pi0241_card_sha256": sha256(args.pi0241_card),
        "p1_run_card_sha256": sha256(args.p1_run_card),
        "p1_records_sha256": sha256(args.p1_records),
        "p1_arrays_sha256": sha256(args.p1_arrays),
        "source_sha256": {name: sha256(path) for name, path in source_files.items()},
        "bridge4_rows": [
            {key: row[key] for key in ("source_episode_id", "target_family", "phase", "episode_index", "jitter_family")}
            for row in selected
        ],
        "condition_hashes_C_T": {source: str(ct[source]["condition_hash"]) for source in selected_ids},
        "training_processor_summary": conditions["training_processor_summary"],
        "noise_sha256_by_source_episode": noise_by_source,
        "time_probes": [
            {"quantile": q, "t": value, "t_discretized": int(value * 1000.0)}
            for q, value in zip(QUANTILES, time_values)
        ],
        "checkpoints": {
            "initializer": {"path": str(args.initializer), "inventory": init_inventory, "manifest_sha256": sha256(args.pi0241_card)},
            "pi0243_a5000": {"path": str(args.a5000), "tree_sha256": a_manifest["tree_sha256"], **a_inventory},
            "pi0244_r010000": {
                "path": str(args.r010000), "revision": hf_manifest["revision"],
                "tree_sha256": r0_checkpoint["tree_sha256"], "files_sha256": sha256(args.r010000_manifest),
                **final_inventory,
            },
        },
        "cells": cells,
        "expected_counts": {
            "model_loads": 3, "backbone_forwards": 12,
            "direct_action_head_forwards": 48, "get_action_calls": 0,
            "training": 0, "backward": 0, "optimizer_steps": 0,
            "wbc_steps": 0, "sim_steps": 0,
        },
        "target_xyz_in_model_input": False,
        "teacher_forced_oracle": True,
        "selected_checkpoint": "NONE",
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "device": "cuda:0; all model comparisons in one process/runtime",
        },
    }
    write_json(root / "run_card.json", plan)
    write_json(root / "run_card.seal.json", {"sha256": sha256(root / "run_card.json"), "status": "SEALED"})
    return plan


def _load_saved_cell(root: Path, cell: dict[str, Any], card_sha: str, bridge: Any) -> tuple[dict[str, Any], dict[str, Any]] | None:
    import numpy as np

    directory = root / "cells" / cell["cell_key"]
    if not directory.exists():
        return None
    record_path, arrays_path = directory / "record.json", directory / "arrays.npz"
    if not record_path.is_file() or not arrays_path.is_file():
        raise RuntimeError(f"incomplete first-valid cell: {directory}")
    record = read_json(record_path)
    if record.get("cell_key") != cell["cell_key"] or record.get("run_card_sha256") != card_sha:
        raise RuntimeError(f"cell provenance mismatch: {directory}")
    with np.load(arrays_path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    expected = record.get("array_sha256", {})
    if set(arrays) != set(expected) or any(bridge.sha256_array(arrays[key]) != expected[key] for key in arrays):
        raise RuntimeError(f"first-valid cell array hash mismatch: {directory}")
    return record, arrays


def run_sidecar(args: argparse.Namespace, bridge: Any) -> dict[str, Any]:
    import gc
    import numpy as np
    import torch

    root = args.output_root.resolve()
    run_card_path = root / "run_card.json"
    if not run_card_path.is_file() or read_json(root / "run_card.seal.json").get("sha256") != sha256(run_card_path):
        raise RuntimeError("sealed run card missing or hash mismatch")
    card = read_json(run_card_path)
    card_sha = sha256(run_card_path)
    if not torch.cuda.is_available():
        raise RuntimeError("no allocated CUDA device")
    attempt_id = os.environ.get("PBS_JOBID") or uuid.uuid4().hex
    events_path = root / "execution_events.jsonl"
    append_event(events_path, attempt_id, "attempt_started", host=platform.node(), device="cuda:0")

    pi_card = read_json(args.pi0241_card)
    r0_card = read_json(args.r0_card)
    r0_training = read_json(args.r0_training_manifest)
    r0_closure = read_json(args.r0_closure)
    p1_card = read_json(args.p1_run_card)
    if sha256(args.pi0241_card) != card["pi0241_card_sha256"]:
        raise RuntimeError("sealed PI0241 card changed")
    if sha256(args.r0_card) != card["r0_run_card_sha256"] or r0_card.get("selected_checkpoint") != "NONE":
        raise RuntimeError("sealed R0 run card changed or selection state drifted")
    if sha256(args.r0_closure) != card["r0_closure_sha256"] or r0_closure.get("status") != "SCIENCE_FAIL_M16_10K_BUDGET":
        raise RuntimeError("sealed R0 closure changed")
    if sha256(args.r0_training_manifest) != card["r0_training_manifest_sha256"] or r0_training.get("status") != "PASS_R0_TRAINING_CLOSED":
        raise RuntimeError("sealed R0 training manifest changed")
    if sha256(args.p1_run_card) != card["p1_run_card_sha256"] or sha256(args.p1_records) != card["p1_records_sha256"] or sha256(args.p1_arrays) != card["p1_arrays_sha256"]:
        raise RuntimeError("sealed PI0242 paired-noise artifacts changed")
    if p1_card.get("selected_checkpoint") != "NONE":
        raise RuntimeError("PI0242 paired-noise source unexpectedly selects a checkpoint")
    a_train = read_json(args.a5000_manifest)
    verify_inventory(args.initializer, [
        {"path": name, "bytes": int(meta["bytes"]), "sha256": str(meta["sha256"])}
        for name, meta in pi_card["initializer"]["files"].items()
    ])
    verify_inventory(args.a5000, a_train["checkpoints"]["5000"]["inventory"]["entries"])
    hf_manifest = read_json(args.r010000_manifest)
    if sha256(args.r010000_manifest) != card["checkpoints"]["pi0244_r010000"]["files_sha256"]:
        raise RuntimeError("pinned R0 final-checkpoint manifest changed")
    if hf_manifest["revision"] != card["checkpoints"]["pi0244_r010000"]["revision"]:
        raise RuntimeError("pinned R0 final-checkpoint revision changed")
    verify_inventory(args.r010000, hf_manifest["files"])
    source_files = {
        "sidecar": Path(__file__).resolve(),
        "pi0242_helper": args.bridge_script,
        "training_adapter": args.code_root / "scripts" / "pi0241_fixed16_learning.py",
        "stage21_inference": args.code_root / "scripts" / "stage21_same_scene_groot_inference.py",
        "stage23_time": args.code_root / "scripts" / "stage23_phase_c_flow_field_diagnostic.py",
        "stage26_flow": args.code_root / "scripts" / "stage26_h1_c1_fixed_noise_local_flow.py",
    }
    if {name: sha256(path) for name, path in source_files.items()} != card["source_sha256"]:
        raise RuntimeError("runtime source identity differs from the sealed run card")
    bridge_rows = bridge._bridge4_rows(read_json(args.cohort)["rows"])
    build_args = argparse.Namespace(code_root=args.code_root, output_root=root)
    conditions = bridge._build_p1_conditions(build_args, pi_card, bridge_rows)
    current_ct = {source: str(conditions["C_T"][source]["condition_hash"]) for source in conditions["C_T"]}
    if current_ct != card["condition_hashes_C_T"]:
        raise RuntimeError("runtime training-condition hash differs from immutable run card")

    p1_records = bridge._load_records(args.p1_records)
    with np.load(args.p1_arrays, allow_pickle=False) as prior:
        noise_by_source = {}
        for row in bridge_rows:
            source = str(row["source_episode_id"])
            hits = [
                x for x in p1_records
                if x.get("kind") == "direct_flow" and x.get("model") == "initializer"
                and x.get("condition") == "C_T" and x.get("source_episode_id") == source
                and float(x.get("t_quantile", -1)) == 0.0
            ]
            if len(hits) != 1:
                raise RuntimeError(f"cannot resolve exactly one sealed paired-noise source: {source}")
            noise = prior[hits[0]["cell_key"] + "__noise"].copy()
            if bridge.sha256_array(noise) != card["noise_sha256_by_source_episode"][source]:
                raise RuntimeError(f"fixed noise no longer matches run card: {source}")
            noise_by_source[source] = noise

    stage21 = bridge._load_script(args.code_root / "scripts" / "stage21_same_scene_groot_inference.py", "stage21_for_pi0244_r0_direct_run")
    stage26 = bridge._load_script(args.code_root / "scripts" / "stage26_h1_c1_fixed_noise_local_flow.py", "stage26_for_pi0244_r0_direct_run")
    stage23 = bridge._load_script(args.code_root / "scripts" / "stage23_phase_c_flow_field_diagnostic.py", "stage23_for_pi0244_r0_direct_run")
    stage23.assert_protocol_static()
    target_map = bridge._target_map(pi_card)
    model_paths = {
        "initializer": Path(args.initializer),
        "pi0243_a5000": Path(args.a5000),
        "pi0244_r010000": Path(args.r010000),
    }
    records: dict[str, dict[str, Any]] = {}
    arrays_by_cell: dict[str, dict[str, Any]] = {}
    for cell in card["cells"]:
        saved = _load_saved_cell(root, cell, card_sha, bridge)
        if saved is not None:
            records[cell["cell_key"]], arrays_by_cell[cell["cell_key"]] = saved
    execution = {"model_loads": 0, "backbone_forwards": 0, "direct_action_head_forwards": 0, "get_action_calls": 0}

    for model_label in MODELS:
        missing = [x for x in card["cells"] if x["model"] == model_label and x["cell_key"] not in records]
        if not missing:
            continue
        model = bridge._load_p1_model(model_paths[model_label], Path(pi_card["paths"]["groot_source"]), "cuda:0")
        execution["model_loads"] += 1
        append_event(events_path, attempt_id, "model_loaded", model=model_label)
        model.get_action = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("get_action is forbidden for this sidecar"))
        try:
            for row in bridge_rows:
                source = str(row["source_episode_id"])
                row_cells = [cell for cell in missing if cell["source_episode_id"] == source]
                if not row_cells:
                    continue
                condition = conditions["C_T"][source]
                target_key = (str(row["target_family"]), str(row["jitter_family"]).split("_", 1)[0])
                if target_key not in target_map:
                    raise RuntimeError(f"TRAIN target authority missing: {source}")
                tensor_path = Path(next(x["tensor_path"] for x in pi_card["selected_inputs"] if x["source_episode_id"] == source))
                with np.load(tensor_path, allow_pickle=False) as payload:
                    metric_row = {**row, "expert": np.asarray(payload["expert"], dtype=np.float32), "target": target_map[target_key]}
                prepared_policy, action_input = bridge._p1_prepare_inputs(model, condition, stage21)
                backbone_inputs, action_input = model.prepare_input(prepared_policy)
                with torch.inference_mode():
                    raw_backbone = model.backbone(backbone_inputs)
                    processed = model.action_head.process_backbone_output(raw_backbone)
                execution["backbone_forwards"] += 1
                append_event(events_path, attempt_id, "backbone_forward", model=model_label, source_episode_id=source)
                for cell in row_cells:
                    with torch.inference_mode():
                        record, arrays = bridge._direct_query(
                            model=model,
                            stage26=stage26,
                            stage23=stage23,
                            processed_backbone=processed,
                            action_inputs=action_input,
                            condition=condition,
                            row=metric_row,
                            model_label=model_label,
                            condition_label="C_T",
                            noise=noise_by_source[source],
                            t_quantile=float(cell["t_quantile"]),
                            t_value=float(cell["t_value"]),
                            cell_key=cell["cell_key"],
                        )
                    execution["direct_action_head_forwards"] += 1
                    append_event(events_path, attempt_id, "direct_action_head_forward", model=model_label, source_episode_id=source, cell_key=cell["cell_key"])
                    record["cell_key"] = cell["cell_key"]
                    record["run_card_sha256"] = card_sha
                    record["attempt_id"] = attempt_id
                    record["checkpoint_tree_sha256"] = card["checkpoints"][model_label].get("tree_sha256")
                    record["execution_counters"] = {
                        "backbone_forward": int(cell["cell_key"] == row_cells[0]["cell_key"]),
                        "direct_action_head_forward": 1,
                        "get_action": 0,
                    }
                    record["array_sha256"] = {key: bridge.sha256_array(value) for key, value in arrays.items()}
                    final_dir = root / "cells" / cell["cell_key"]
                    temporary = root / "cells" / f".{cell['cell_key']}.{uuid.uuid4().hex}.tmp"
                    temporary.mkdir(parents=True, exist_ok=False)
                    np.savez_compressed(temporary / "arrays.npz", **arrays)
                    write_json(temporary / "record.json", record)
                    os.replace(temporary, final_dir)
                    records[cell["cell_key"]] = record
                    arrays_by_cell[cell["cell_key"]] = arrays
                    print(json.dumps({"event": "cell_complete", "cell_key": cell["cell_key"], "model": model_label, "source_episode_id": source, "t_index": cell["t_index"]}), flush=True)
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()

    if len(records) != 48:
        raise RuntimeError(f"direct-flow matrix incomplete: {len(records)}/48")
    append_event(events_path, attempt_id, "attempt_completed", valid_cells=len(records))
    ordered = [records[cell["cell_key"]] for cell in card["cells"]]
    merged_arrays = {key: value for cell in card["cells"] for key, value in arrays_by_cell[cell["cell_key"]].items()}
    arrays_tmp = root / ".pi0244_r0_direct_flow_arrays.tmp.npz"
    records_tmp = root / ".pi0244_r0_direct_flow_records.tmp.jsonl"
    np.savez_compressed(arrays_tmp, **merged_arrays)
    records_tmp.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in ordered), encoding="utf-8")
    os.replace(arrays_tmp, root / "pi0244_r0_direct_flow_arrays.npz")
    os.replace(records_tmp, root / "pi0244_r0_direct_flow_records.jsonl")
    summary = {
        "schema": "PI0244_R0_DIRECT_FLOW_SUMMARY_V1",
        "cell_count": len(ordered),
        "model_counts": {name: sum(row["model"] == name for row in ordered) for name in MODELS},
        "flow_medians_by_model": {
            name: {
                metric: float(np.median([row["flow_metrics"][metric] for row in ordered if row["model"] == name]))
                for metric in ("active10_velocity_mse", "xyz3_velocity_mse", "rot6d6_velocity_mse", "gripper1_velocity_mse")
            }
            for name in MODELS
        },
        "physical_medians_by_model": {
            name: {
                metric: float(np.median([row["physical_metrics"][metric] for row in ordered if row["model"] == name]))
                for metric in ("xyz_rmse_m", "so3_rmse_rad", "gripper_rmse", "max_linear_speed_mps", "max_angular_speed_radps")
            }
            for name in MODELS
        },
        "execution_counters": {
            "model_loads": sum(row["event"] == "model_loaded" for row in map(json.loads, events_path.read_text(encoding="utf-8").splitlines())),
            "backbone_forwards": sum(row["event"] == "backbone_forward" for row in map(json.loads, events_path.read_text(encoding="utf-8").splitlines())),
            "direct_action_head_forward_attempts": sum(row["event"] == "direct_action_head_forward" for row in map(json.loads, events_path.read_text(encoding="utf-8").splitlines())),
            "valid_direct_flow_cells": len(ordered),
            "models_with_valid_cells": sorted({row["model"] for row in ordered}),
            "get_action_calls": 0,
            "training": 0,
            "backward": 0,
            "optimizer_steps": 0,
            "wbc_steps": 0,
            "sim_steps": 0,
        },
        "selected_checkpoint": "NONE",
    }
    write_json(root / "pi0244_r0_direct_flow_summary.json", summary)
    closure = {
        "schema": "PI0244_R0_DIRECT_FLOW_CLOSURE_V1",
        "status": "PASS_READ_ONLY",
        "run_card_sha256": card_sha,
        "records_sha256": sha256(root / "pi0244_r0_direct_flow_records.jsonl"),
        "arrays_sha256": sha256(root / "pi0244_r0_direct_flow_arrays.npz"),
        "summary_sha256": sha256(root / "pi0244_r0_direct_flow_summary.json"),
        "execution_events_sha256": sha256(events_path),
        "counts": summary["execution_counters"],
        "selected_checkpoint": "NONE",
    }
    write_json(root / "pi0244_r0_direct_flow_closure.json", closure)
    return closure


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("self-test", "plan", "run"), required=True)
    for name in (
        "bridge-script", "code-root", "pi0241-card", "r0-card", "r0-closure",
        "r0-training-manifest", "cohort", "p1-run-card", "p1-records",
        "p1-arrays", "a5000", "a5000-manifest", "r010000",
        "r010000-manifest", "initializer", "output-root",
    ):
        parser.add_argument(f"--{name}", type=Path, required=name == "output-root")
    args = parser.parse_args()
    if args.mode == "self-test":
        self_test()
        return 0
    bridge = load_bridge(args.bridge_script)
    if args.mode == "plan":
        plan = make_plan(args, bridge)
        print(json.dumps({"status": "SEALED", "cells": len(plan["cells"]), "run_card": str(args.output_root / "run_card.json")}, indent=2))
    else:
        closure = run_sidecar(args, bridge)
        print(json.dumps(closure, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
