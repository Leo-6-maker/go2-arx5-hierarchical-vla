#!/usr/bin/env python3
"""CPU-only Stage45E candidate-selection packet builder.

This preparation tool never loads a model, checkpoint, dataset split, Torch,
CUDA, PBS, Validation-E, or Holdout-F.  It consumes one normalized TRAIN24
JSON artifact prepared from already-sealed evidence and produces descriptive
paired summaries for PI review.  It deliberately cannot select a winner.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO / "manifests/STAGE45E_NEUTRAL_CANDIDATE_SELECTION_PREP_V1.json"
FORBIDDEN_TOP_LEVEL_IMPORTS = {
    "torch",
    "transformers",
    "gr00t",
    "jax",
    "tensorflow",
    "isaacgym",
    "omni",
    "subprocess",
}


class CandidateSelectionError(RuntimeError):
    """Raised when normalized evidence violates the prep contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CandidateSelectionError(message)


def read_json(path: Path) -> Any:
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"JSON input is unavailable or unsafe: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    require(not path.exists() and not path.is_symlink(), f"output must be fresh: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest() -> dict[str, Any]:
    value = read_json(MANIFEST_PATH)
    require(value.get("schema") == "STAGE45E_NEUTRAL_CANDIDATE_SELECTION_PREP_V1", "manifest schema drift")
    require(value.get("status") == "PREP_ONLY_NO_EXECUTION", "manifest is not preparation-only")
    require(value.get("claim_boundary", {}).get("automatic_winner") is False, "automatic winner must remain disabled")
    require(value.get("validation_e_plan_only", {}).get("execute_supported") is False, "Validation-E execution must remain disabled")
    return value


def static_guard() -> dict[str, Any]:
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(Path(__file__)))
    imports: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".", 1)[0])
    forbidden = sorted(imports.intersection(FORBIDDEN_TOP_LEVEL_IMPORTS))
    return {
        "schema": "STAGE45E_NEUTRAL_CANDIDATE_SELECTION_STATIC_GUARD_V1",
        "status": "PASS" if not forbidden else "BLOCKED",
        "forbidden_top_level_imports": forbidden,
        "model_construction": 0,
        "model_forward": 0,
        "training": 0,
        "gpu": 0,
        "pbs": 0,
        "validation_e_read": False,
        "holdout_f_read": False,
        "wbc": 0,
        "simulator": 0,
        "hardware": 0,
    }


def plan() -> dict[str, Any]:
    cfg = manifest()
    guard = static_guard()
    require(guard["status"] == "PASS", "static guard failed")
    return {
        "schema": "STAGE45E_NEUTRAL_CANDIDATE_SELECTION_PLAN_V1",
        "status": "PASS_STATIC_NOT_EXECUTED",
        "authority": cfg["authority"],
        "branch": cfg["repository"]["branch"],
        "base_head": cfg["repository"]["base_head"],
        "normalized_input_schema": cfg["normalized_input_schema"],
        "candidate_order": cfg["candidate_order"],
        "primary_metrics": cfg["primary_metrics"],
        "paired_comparisons": cfg["paired_comparisons"],
        "claim_boundary": cfg["claim_boundary"],
        "decision_state": "PI_DECISION_REQUIRED",
        "static_guard": guard,
        "return_token": cfg["return_token"],
    }


def validation_plan() -> dict[str, Any]:
    cfg = manifest()
    return {
        "schema": "STAGE45E_VALIDATION_E_FUTURE_ADMISSION_SKELETON_V1",
        "status": "LOCKED_PLAN_ONLY",
        "validation_e_read": False,
        "holdout_f_read": False,
        "execute_supported": False,
        "required_before_any_future_read": [
            "explicit later PI gate selecting exactly one frozen candidate recipe",
            "exact selected checkpoint/model identity",
            "exact selected normalization/loss/sampler identity",
            "Validation-E dataset/split identity bound without reading Holdout-F",
            "fresh output root and raw-before-derived evidence contract",
            "no post-Validation recipe tuning under the same selection decision",
        ],
        "selection_source": "future PI decision only",
        "current_authorization": cfg["validation_e_plan_only"],
        "decision_state": "PI_DECISION_REQUIRED",
    }


def _finite_number(value: Any, field: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{field} must be numeric")
    numeric = float(value)
    require(math.isfinite(numeric), f"{field} must be finite")
    return numeric


def _canonical_sign(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
        require(math.isfinite(numeric), "progress_sign must be finite")
        if numeric > 0:
            return "positive"
        if numeric < 0:
            return "negative"
        return "zero"
    require(isinstance(value, str) and value.strip(), "progress_sign must be a non-empty scalar")
    return value.strip()


def validate_normalized_input(value: Any) -> dict[str, Any]:
    cfg = manifest()
    require(isinstance(value, Mapping), "normalized evidence must be an object")
    require(value.get("schema") == cfg["normalized_input_schema"], "normalized evidence schema drift")
    require(value.get("split") == "TRAIN24", "only TRAIN24 normalized evidence is accepted")
    require(value.get("validation_e_read") is False, "normalized evidence indicates Validation-E read")
    require(value.get("holdout_f_read") is False, "normalized evidence indicates Holdout-F read")

    window_ids = value.get("window_ids")
    require(isinstance(window_ids, list) and len(window_ids) == cfg["window_count"], "TRAIN24 window count drift")
    require(all(isinstance(item, str) and item for item in window_ids), "window_ids must be non-empty strings")
    require(len(set(window_ids)) == len(window_ids), "window_ids are not unique")

    candidates = value.get("candidates")
    require(isinstance(candidates, Mapping), "candidates must be an object")
    require(list(candidates.keys()) == cfg["candidate_order"], "candidate order/set drift")

    numeric_metrics = tuple(cfg["numeric_metrics"])
    primary_metrics = tuple(cfg["primary_metrics"])
    normalized: dict[str, Any] = {
        "schema": cfg["normalized_input_schema"],
        "split": "TRAIN24",
        "window_ids": list(window_ids),
        "validation_e_read": False,
        "holdout_f_read": False,
        "candidates": {},
    }

    for label in cfg["candidate_order"]:
        candidate = candidates[label]
        require(isinstance(candidate, Mapping), f"candidate {label} must be an object")
        provenance = candidate.get("provenance")
        require(isinstance(provenance, Mapping) and provenance, f"candidate {label} provenance is missing")
        rows = candidate.get("rows")
        require(isinstance(rows, list) and len(rows) == cfg["window_count"], f"candidate {label} row count drift")
        by_window: dict[str, Any] = {}
        for row in rows:
            require(isinstance(row, Mapping), f"candidate {label} row must be an object")
            window_id = row.get("window_id")
            target_id = row.get("target_id")
            require(window_id in window_ids, f"candidate {label} has unknown window_id")
            require(window_id not in by_window, f"candidate {label} duplicates window {window_id}")
            require(isinstance(target_id, str) and target_id, f"candidate {label}:{window_id} target_id missing")
            metrics = row.get("metrics")
            require(isinstance(metrics, Mapping), f"candidate {label}:{window_id} metrics missing")
            require(all(metric in metrics for metric in primary_metrics), f"candidate {label}:{window_id} primary metric missing")
            clean_metrics = {metric: _finite_number(metrics[metric], f"{label}:{window_id}:{metric}") for metric in numeric_metrics}
            clean_metrics["progress_sign"] = _canonical_sign(metrics["progress_sign"])
            by_window[window_id] = {
                "window_id": window_id,
                "target_id": target_id,
                "metrics": clean_metrics,
            }
        require(set(by_window) == set(window_ids), f"candidate {label} window identity mismatch")
        normalized["candidates"][label] = {
            "provenance": dict(provenance),
            "rows": [by_window[window_id] for window_id in window_ids],
        }
    return normalized


def _candidate_summary(candidate: Mapping[str, Any], numeric_metrics: Sequence[str]) -> dict[str, Any]:
    rows = candidate["rows"]
    summary = {
        metric: {
            "count": len(rows),
            "mean": float(statistics.fmean(float(row["metrics"][metric]) for row in rows)),
            "min": float(min(float(row["metrics"][metric]) for row in rows)),
            "max": float(max(float(row["metrics"][metric]) for row in rows)),
        }
        for metric in numeric_metrics
    }
    sign_counts: dict[str, int] = {}
    for row in rows:
        sign = str(row["metrics"]["progress_sign"])
        sign_counts[sign] = sign_counts.get(sign, 0) + 1
    return {"numeric": summary, "progress_sign_counts": dict(sorted(sign_counts.items()))}


def _pair_summary(left: Mapping[str, Any], right: Mapping[str, Any], numeric_metrics: Sequence[str]) -> dict[str, Any]:
    left_rows = {row["window_id"]: row for row in left["rows"]}
    right_rows = {row["window_id"]: row for row in right["rows"]}
    require(set(left_rows) == set(right_rows), "paired candidates do not share the same windows")
    window_ids = list(left_rows)

    deltas: dict[str, Any] = {}
    for metric in numeric_metrics:
        values = [
            float(left_rows[wid]["metrics"][metric]) - float(right_rows[wid]["metrics"][metric])
            for wid in window_ids
        ]
        deltas[metric] = {
            "count": len(values),
            "mean_left_minus_right": float(statistics.fmean(values)),
            "min_left_minus_right": float(min(values)),
            "max_left_minus_right": float(max(values)),
        }

    by_target: dict[str, list[str]] = {}
    for wid in window_ids:
        left_target = left_rows[wid]["target_id"]
        right_target = right_rows[wid]["target_id"]
        require(left_target == right_target, f"target identity mismatch for paired window {wid}")
        by_target.setdefault(left_target, []).append(wid)

    target_summaries: dict[str, Any] = {}
    for target_id, target_windows in sorted(by_target.items()):
        target_summaries[target_id] = {}
        for metric in numeric_metrics:
            values = [
                float(left_rows[wid]["metrics"][metric]) - float(right_rows[wid]["metrics"][metric])
                for wid in target_windows
            ]
            target_summaries[target_id][metric] = {
                "count": len(values),
                "mean_left_minus_right": float(statistics.fmean(values)),
                "min_left_minus_right": float(min(values)),
                "max_left_minus_right": float(max(values)),
            }

    sign_transitions: dict[str, int] = {}
    window_rows: dict[str, Any] = {}
    for wid in window_ids:
        left_sign = str(left_rows[wid]["metrics"]["progress_sign"])
        right_sign = str(right_rows[wid]["metrics"]["progress_sign"])
        transition = f"{right_sign}->{left_sign}"
        sign_transitions[transition] = sign_transitions.get(transition, 0) + 1
        window_rows[wid] = {
            "target_id": left_rows[wid]["target_id"],
            "numeric_delta_left_minus_right": {
                metric: float(left_rows[wid]["metrics"][metric]) - float(right_rows[wid]["metrics"][metric])
                for metric in numeric_metrics
            },
            "progress_sign_transition_right_to_left": transition,
        }

    return {
        "overall_numeric_delta": deltas,
        "by_target_numeric_delta": target_summaries,
        "by_window": window_rows,
        "progress_sign_transition_counts_right_to_left": dict(sorted(sign_transitions.items())),
        "statistical_claim": "DESCRIPTIVE_PAIRED_ONLY_NO_IID_INFERENCE",
    }


def build_packet(normalized: Mapping[str, Any], input_sha256: str) -> dict[str, Any]:
    cfg = manifest()
    candidates = normalized["candidates"]
    numeric_metrics = tuple(cfg["numeric_metrics"])
    candidate_summaries = {
        label: _candidate_summary(candidates[label], numeric_metrics)
        for label in cfg["candidate_order"]
    }
    pairs: dict[str, Any] = {}
    for left, right in cfg["paired_comparisons"]:
        pair_name = f"{left}_MINUS_{right}"
        pairs[pair_name] = {
            "left": left,
            "right": right,
            "summary": _pair_summary(candidates[left], candidates[right], numeric_metrics),
        }

    return {
        "schema": cfg["decision_packet_schema"],
        "status": "PASS_DESCRIPTIVE_TRAIN24_ONLY",
        "decision_state": "PI_DECISION_REQUIRED",
        "winner_declared": False,
        "automatic_selection_performed": False,
        "posthoc_threshold_used": False,
        "iid_significance_test_performed": False,
        "validation_e_authorized": False,
        "holdout_f_authorized": False,
        "normalized_input_sha256": input_sha256,
        "split": "TRAIN24",
        "window_count": cfg["window_count"],
        "candidate_order": cfg["candidate_order"],
        "candidate_provenance": {label: candidates[label]["provenance"] for label in cfg["candidate_order"]},
        "candidate_summaries": candidate_summaries,
        "paired_comparisons": pairs,
        "claim_boundary": cfg["claim_boundary"],
        "execution_counters": {
            "model_construction": 0,
            "model_forward": 0,
            "training": 0,
            "gpu": 0,
            "pbs": 0,
            "validation_e_read": False,
            "holdout_f_read": False,
            "wbc": 0,
            "simulator": 0,
            "hardware": 0,
        },
        "next_action": "PI must review descriptive TRAIN24 evidence and issue a separate gate before any Validation-E read or further experiment.",
    }


def build_from_file(input_path: Path, output_path: Path) -> dict[str, Any]:
    raw = read_json(input_path)
    normalized = validate_normalized_input(raw)
    packet = build_packet(normalized, sha256_file(input_path))
    write_json(output_path, packet)
    return packet


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    modes = value.add_mutually_exclusive_group(required=True)
    modes.add_argument("--plan-only", action="store_true")
    modes.add_argument("--validation-plan-only", action="store_true")
    modes.add_argument("--build-packet", action="store_true")
    value.add_argument("--input", type=Path)
    value.add_argument("--output", type=Path)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.plan_only:
        print(json.dumps(plan(), indent=2, sort_keys=True, allow_nan=False))
        return 0
    if args.validation_plan_only:
        print(json.dumps(validation_plan(), indent=2, sort_keys=True, allow_nan=False))
        return 0
    require(args.input is not None and args.output is not None, "--build-packet requires --input and --output")
    packet = build_from_file(args.input, args.output)
    print(json.dumps(packet, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CandidateSelectionError as exc:
        print(f"STAGE45E_CANDIDATE_SELECTION_BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
