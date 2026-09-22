#!/usr/bin/env python3
"""PI0244 thin preset around the sealed PI0243 trainer/screen path.

The mature implementation remains the source of truth.  This wrapper only
binds the new fixed16/10K identity, schedule, family, and terminal closure.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping


PROTOCOL_COMMIT = "d23727c2b7764589573f6c7a7195d091980a9bbc"
AUDITED_PARENT = "69b587fe4edc3a7a0c9ff99240ba0214bc6e6e86"
AUTHORIZATION = "PI0244_M16_10K_GO"
EXPERIMENT = "PI0244_M16_10K_V1"
FAMILY = EXPERIMENT
MAX_STEPS = 10000
CHECKPOINTS = [2500, 5000, 7500, 10000]
SCREENS = [0, 2500, 5000, 7500, 10000]
PREFIX_STEPS = 5000


class PI0244Error(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_base(code_root: Path) -> Any:
    path = code_root / "scripts" / "pi0243_staged_learning_to_sim.py"
    spec = importlib.util.spec_from_file_location("pi0244_pi0243_base", path)
    if spec is None or spec.loader is None:
        raise PI0244Error(f"cannot import mature stage path: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.PROTOCOL_COMMIT = PROTOCOL_COMMIT
    module.AUDITED_PARENT = AUDITED_PARENT
    module.AUTH = dict(module.AUTH)
    module.AUTH["a"] = AUTHORIZATION
    module.stage_spec = stage_spec
    return module


def stage_spec(stage: str) -> dict[str, Any]:
    if str(stage).lower() == "a":
        return {
            "stage": "A",
            "view": "16",
            "max_steps": MAX_STEPS,
            "checkpoints": list(CHECKPOINTS),
            "screens": list(SCREENS),
        }
    raise PI0244Error("PI0244 only permits the fixed16 10K stage")


def _config_differences(parent: Mapping[str, Any], current: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    differences: dict[str, dict[str, Any]] = {}

    def walk(left: Any, right: Any, prefix: str) -> None:
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            for key in sorted(set(left) | set(right)):
                walk(left.get(key), right.get(key), f"{prefix}.{key}" if prefix else str(key))
            return
        if left != right:
            differences[prefix] = {"parent": left, "current": right}

    walk(parent, current, "")
    return differences


ALLOWED_CONFIG_DIFFS = {
    "max_steps",
    "save_total_limit",
    "checkpoint_steps",
    "screen_steps",
}


def validate_schedule_rows(current: list[Mapping[str, Any]], parent: list[Mapping[str, Any]]) -> dict[str, Any]:
    if len(current) != MAX_STEPS:
        raise PI0244Error(f"10K schedule length mismatch: {len(current)}")
    if current[:PREFIX_STEPS] != parent[:PREFIX_STEPS]:
        raise PI0244Error("10K schedule prefix differs from the sealed 5K schedule")
    counts = Counter(str(row["source_episode_id"]) for row in current)
    if len(counts) != 16 or set(counts.values()) != {625}:
        raise PI0244Error(f"fixed16 exposure counts are not 625 each: {dict(counts)}")
    return {"length": len(current), "prefix_steps": PREFIX_STEPS, "source_counts": dict(sorted(counts.items()))}


def _read_rows(path: Path | str) -> list[dict[str, Any]]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise PI0244Error(f"schedule rows missing: {path}")
    return rows


def _patch_card(base: Any, args: argparse.Namespace) -> Path:
    card_path = args.output_root / "pi0243_run_card.json"
    card = base.read_json(card_path)
    parent_card = base.read_json(args.parent_card)
    current_config = card["resolved_config"]
    parent_config = parent_card["resolved_config"]
    differences = _config_differences(parent_config, current_config)
    unexpected = sorted(key for key in differences if key not in ALLOWED_CONFIG_DIFFS)
    if unexpected:
        raise PI0244Error(f"unexpected resolved-config drift: {unexpected}")

    schedule_check = validate_schedule_rows(_read_rows(card["schedule"]["path"]), _read_rows(args.prefix_schedule))
    diff_path = args.output_root / "resolved_diff.json"
    diff = {
        "schema": "PI0244_RESOLVED_DIFF_V1",
        "status": "PASS",
        "protocol_commit": PROTOCOL_COMMIT,
        "experiment": EXPERIMENT,
        "parent_card": str(args.parent_card),
        "parent_card_sha256": sha256_file(args.parent_card),
        "differences": differences,
        "allowed_difference_keys": sorted(ALLOWED_CONFIG_DIFFS),
        "schedule": schedule_check,
    }
    base.write_json(diff_path, diff)

    card["experiment"] = EXPERIMENT
    card["protocol_commit"] = PROTOCOL_COMMIT
    card["audited_parent"] = AUDITED_PARENT
    card["stage_role"] = "PI0244_FIXED16_10K"
    card["resolved_diff"] = {"path": str(diff_path), "sha256": sha256_file(diff_path)}
    card["implementation"]["preset_path"] = str(Path(__file__).resolve())
    card["implementation"]["preset_sha256"] = sha256_file(Path(__file__).resolve())
    card["storage"]["checkpoint_family"] = FAMILY
    card["storage"]["capacity_binding"] = {
        "full_checkpoint_count": 4,
        "retain_all_local_hf_durable": True,
        "checkpoint_steps": list(CHECKPOINTS),
    }
    card["screen_binding"] = {
        "unique_cells": len(SCREENS) * 16,
        "screen_steps": list(SCREENS),
        "final_step": MAX_STEPS,
        "final_only_gate": True,
    }
    base.write_json(card_path, card)
    base.write_json(card_path.with_name("pi0243_run_card.seal.json"), {"schema": "PI0243_RUN_CARD_SEAL_V1", "run_card_sha256": sha256_file(card_path)})
    return card_path


def prepare(args: argparse.Namespace) -> None:
    if args.output_root.exists():
        raise PI0244Error(f"refusing to reuse output root: {args.output_root}")
    base = load_base(args.code_root)
    base_args = SimpleNamespace(
        stage="a",
        code_root=args.code_root,
        groot_source=args.groot_source,
        stage10_root=args.stage10_root,
        train_dataset=args.train_dataset,
        p237_root=args.p237_root,
        modality_config=args.modality_config,
        processor_base_model=args.processor_base_model,
        initializer=args.initializer,
        cosmos_snapshot=args.cosmos_snapshot,
        cosmos_cache_root=args.cosmos_cache_root,
        output_root=args.output_root,
        durable_host=None,
        durable_root=None,
        hf_repo_id=args.hf_repo_id,
        hf_token_file=args.hf_token_file,
        hf_path_prefix=args.hf_path_prefix,
    )
    base.make_card(base_args)
    card_path = _patch_card(base, args)
    print(json.dumps({"status": "SEALED_READY_TO_EXECUTE", "experiment": EXPERIMENT, "run_card": str(card_path), "run_card_sha256": sha256_file(card_path)}, indent=2))


def verify_card(path: Path) -> tuple[Any, dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    code_root = Path(raw["paths"]["code_root"])
    base = load_base(code_root)
    card = base.verify_card(path)
    if card.get("experiment") != EXPERIMENT or card.get("stage_role") != "PI0244_FIXED16_10K":
        raise PI0244Error("PI0244 experiment identity mismatch")
    if card["stage"] != stage_spec("a"):
        raise PI0244Error("PI0244 fixed16/10K stage binding mismatch")
    if card.get("storage", {}).get("checkpoint_family") != FAMILY:
        raise PI0244Error("PI0244 checkpoint family mismatch")
    diff = card.get("resolved_diff", {})
    diff_path = Path(diff.get("path", ""))
    diff_path = base._mapped_stage_path(card, diff_path, "PI0243_METADATA_MIRROR_ROOT")
    if not diff_path.is_file() or sha256_file(diff_path) != diff.get("sha256"):
        raise PI0244Error("resolved_diff is missing or changed")
    payload = json.loads(diff_path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS" or payload.get("experiment") != EXPERIMENT:
        raise PI0244Error("resolved_diff is not PASS")
    return base, card


def train(args: argparse.Namespace) -> None:
    base, _ = verify_card(args.run_card)
    base.train_stage(SimpleNamespace(run_card=args.run_card, stage="a", authorization=AUTHORIZATION))
    manifest_path = Path(json.loads(args.run_card.read_text(encoding="utf-8"))["paths"]["output_root"]) / "pi0243_training_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update({"experiment": EXPERIMENT, "protocol_commit": PROTOCOL_COMMIT, "stage_role": "PI0244_FIXED16_10K"})
        base.write_json(manifest_path, manifest)


def screen(args: argparse.Namespace) -> None:
    base, card = verify_card(args.run_card)
    base.screen_stage(SimpleNamespace(run_card=args.run_card, stage="a", device=args.device))
    root = Path(card["paths"]["output_root"])
    old_path = root / "pi0243_stage_closure.json"
    closure = json.loads(old_path.read_text(encoding="utf-8"))
    passed = all(bool(value) for value in closure.get("criteria", {}).values()) and int(closure.get("final_count", -1)) == 16 and int(closure.get("screen_cells", -1)) == 80
    closure.update(
        {
            "schema": "PI0244_STAGE_CLOSURE_V1",
            "experiment": EXPERIMENT,
            "stage_role": "PI0244_FIXED16_10K",
            "status": "PASS_M16_10K" if passed else "SCIENCE_FAIL_M16_10K_BUDGET",
            "terminal_token": None if passed else "SCIENCE_FAIL_M16_10K_BUDGET — STOP_FOR_PI_OBJECTIVE_OR_OPTIMIZATION_DECISION",
            "selected_checkpoint": "NONE",
        }
    )
    base.write_json(root / "pi0244_stage_closure.json", closure)
    print(json.dumps(closure, indent=2, sort_keys=True))


def self_test() -> None:
    spec = stage_spec("a")
    assert spec["view"] == "16" and spec["max_steps"] == 10000
    assert spec["checkpoints"] == CHECKPOINTS and spec["screens"] == SCREENS
    current = [{"source_episode_id": f"s{i % 16}"} for i in range(10000)]
    parent = current[:5000]
    result = validate_schedule_rows(current, parent)
    assert result["length"] == 10000 and set(result["source_counts"].values()) == {625}
    assert set(_config_differences({"max_steps": 5000, "scheduler": "cosine"}, {"max_steps": 10000, "scheduler": "cosine"})) == {"max_steps"}
    print("PI0244_SELF_TEST_PASS")


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("prepare", "train", "screen", "self-test"), required=True)
    parser.add_argument("--run-card", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--code-root", type=Path)
    parser.add_argument("--groot-source", type=Path)
    parser.add_argument("--stage10-root", type=Path)
    parser.add_argument("--train-dataset", type=Path)
    parser.add_argument("--p237-root", type=Path)
    parser.add_argument("--modality-config", type=Path)
    parser.add_argument("--processor-base-model", type=Path)
    parser.add_argument("--initializer", type=Path)
    parser.add_argument("--cosmos-snapshot", type=Path)
    parser.add_argument("--cosmos-cache-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--parent-card", type=Path)
    parser.add_argument("--prefix-schedule", type=Path)
    parser.add_argument("--hf-repo-id")
    parser.add_argument("--hf-token-file", type=Path)
    parser.add_argument("--hf-path-prefix")
    return parser


def main() -> int:
    args = parser().parse_args()
    if args.phase == "self-test":
        self_test()
        return 0
    if args.phase == "prepare":
        required = (args.code_root, args.groot_source, args.stage10_root, args.train_dataset, args.p237_root, args.modality_config, args.processor_base_model, args.initializer, args.cosmos_snapshot, args.cosmos_cache_root, args.output_root, args.parent_card, args.prefix_schedule, args.hf_repo_id, args.hf_token_file, args.hf_path_prefix)
        if any(value is None for value in required):
            raise PI0244Error("prepare requires all path, parent-card, prefix-schedule, and HF arguments")
        prepare(args)
    elif args.phase == "train":
        if args.run_card is None:
            raise PI0244Error("train requires --run-card")
        train(args)
    else:
        if args.run_card is None:
            raise PI0244Error("screen requires --run-card")
        screen(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
