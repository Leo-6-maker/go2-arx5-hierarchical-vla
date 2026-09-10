"""Pure read-only analysis of checkpoint evaluation artifacts.

The analyzer consumes materialized per-window rows.  It does not import or
instantiate a model, call an inference adapter, decode video, or alter an
evaluation artifact.  All comparisons are descriptive; no success threshold
is embedded here.
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ..experiments.reusable import ReusableManifestError, sha256_file, write_json

ANALYSIS_SCHEMA = "GO2ARX5_REUSABLE_CHECKPOINT_ANALYSIS_V1"


class CheckpointAnalysisError(ReusableManifestError):
    """Raised when evaluation evidence cannot be compared deterministically."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckpointAnalysisError(message)


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * (percentile / 100.0)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] + weight * (ordered[upper] - ordered[lower])


def numeric_summary(values: Iterable[float]) -> dict[str, Any]:
    numbers = [float(value) for value in values]
    _require(
        bool(numbers) and all(math.isfinite(value) for value in numbers),
        "metric distribution is empty or non-finite",
    )
    return {
        "count": len(numbers),
        "mean": sum(numbers) / len(numbers),
        "median": _percentile(numbers, 50.0),
        "p95": _percentile(numbers, 95.0),
        "min": min(numbers),
        "max": max(numbers),
    }


def _load_rows_file(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CheckpointAnalysisError(
                    f"invalid row JSON at line {line_number}: {path}"
                ) from exc
            _require(
                isinstance(value, dict), f"row at line {line_number} is not an object"
            )
            rows.append(dict(value))
        return rows
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointAnalysisError(f"cannot read JSON input: {path}") from exc
    value = (
        document
        if isinstance(document, list)
        else document.get("rows", document.get("per_window"))
        if isinstance(document, dict)
        else None
    )
    _require(isinstance(value, list), f"JSON input must contain rows: [...]: {path}")
    _require(
        all(isinstance(row, dict) for row in value), "all input rows must be objects"
    )
    return [dict(row) for row in value]


def load_rows(source: str | Path) -> tuple[list[dict[str, Any]], Path]:
    """Load per-window rows from a JSONL/JSON file or evaluator output root."""

    candidate = Path(source).expanduser()
    if candidate.is_dir():
        for name in ("per_window.jsonl", "per_window.json"):
            possible = candidate / name
            if possible.is_file() and not possible.is_symlink():
                candidate = possible
                break
        else:
            raise CheckpointAnalysisError(
                f"no per-window evidence in output root: {candidate}"
            )
    if candidate.is_symlink() or not candidate.is_file():
        raise CheckpointAnalysisError(
            f"analysis input is not a regular file: {candidate}"
        )
    candidate = candidate.resolve()
    rows = _load_rows_file(candidate)
    _require(bool(rows), "per-window evidence is empty")
    return rows, candidate


def _checkpoint_label(row: Mapping[str, Any]) -> str:
    checkpoint = row.get("checkpoint")
    if isinstance(checkpoint, Mapping):
        value = checkpoint.get("label")
    else:
        value = row.get("checkpoint_label")
    value = str(value or "").strip()
    _require(bool(value), "row checkpoint label is missing")
    return value


def _checkpoint_sha(row: Mapping[str, Any]) -> str | None:
    checkpoint = row.get("checkpoint")
    if isinstance(checkpoint, Mapping) and checkpoint.get("sha256") is not None:
        return str(checkpoint["sha256"])
    return None


def _target_id(row: Mapping[str, Any]) -> str:
    value = str(row.get("target_id", "")).strip()
    if value:
        return value
    window_id = str(row.get("window_id", "")).strip()
    _require(bool(window_id), "row window_id is missing")
    if window_id.startswith("stage22_expert_") and ":" in window_id:
        return window_id[len("stage22_expert_") :].split(":", 1)[0]
    return window_id.split(":", 1)[0]


def _metric_map(row: Mapping[str, Any]) -> dict[str, float]:
    metrics = row.get("metrics")
    _require(isinstance(metrics, Mapping), "row metrics must be an object")
    result: dict[str, float] = {}
    for key, value in metrics.items():
        if _finite_number(value):
            result[str(key)] = float(value)
    target_metrics = metrics.get("target")
    if isinstance(target_metrics, Mapping):
        for key, value in target_metrics.items():
            if _finite_number(value):
                result[f"target.{key}"] = float(value)
    _require(bool(result), "row contains no scalar metrics")
    return result


def _normalize_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        window_id = str(row.get("window_id", "")).strip()
        _require(bool(window_id), "row window_id is missing")
        label = _checkpoint_label(row)
        key = (label, window_id)
        _require(key not in seen, f"duplicate checkpoint/window row: {key}")
        seen.add(key)
        normalized.append(
            {
                "checkpoint_label": label,
                "checkpoint_sha256": _checkpoint_sha(row),
                "window_id": window_id,
                "target_id": _target_id(row),
                "metrics": _metric_map(row),
            }
        )
    return normalized


def _group_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels: list[str] = []
    for row in rows:
        label = str(row["checkpoint_label"])
        if label not in labels:
            labels.append(label)
    result: dict[str, Any] = {}
    for label in labels:
        selected = [row for row in rows if row["checkpoint_label"] == label]
        metric_names = sorted({name for row in selected for name in row["metrics"]})
        result[label] = {
            "checkpoint_sha256": next(
                (
                    row["checkpoint_sha256"]
                    for row in selected
                    if row["checkpoint_sha256"] is not None
                ),
                None,
            ),
            "window_count": len(selected),
            "metrics": {
                name: numeric_summary(
                    row["metrics"][name] for row in selected if name in row["metrics"]
                )
                for name in metric_names
                if any(name in row["metrics"] for row in selected)
            },
        }
    return result


def _parse_pairs(
    pairs: Sequence[Mapping[str, Any]] | None,
    labels: Sequence[str],
) -> list[dict[str, str]]:
    if pairs is None:
        return []
    result: list[dict[str, str]] = []
    for index, pair in enumerate(pairs):
        if isinstance(pair, Mapping):
            name = str(pair.get("name", "")).strip()
            left = str(pair.get("left", "")).strip()
            right = str(pair.get("right", "")).strip()
        else:
            raise CheckpointAnalysisError(f"analysis pair {index} is not an object")
        _require(bool(name and left and right), f"analysis pair {index} is incomplete")
        _require(
            left in labels and right in labels,
            f"analysis pair {name} references an unknown checkpoint",
        )
        result.append({"name": name, "left": left, "right": right})
    return result


def _paired_comparison(
    rows: Sequence[Mapping[str, Any]],
    *,
    name: str,
    left: str,
    right: str,
    window_order: Sequence[str],
) -> dict[str, Any]:
    left_rows = {
        str(row["window_id"]): row for row in rows if row["checkpoint_label"] == left
    }
    right_rows = {
        str(row["window_id"]): row for row in rows if row["checkpoint_label"] == right
    }
    _require(
        set(left_rows) == set(right_rows),
        f"paired checkpoints have different windows: {left} vs {right}",
    )
    ordered_ids = [window_id for window_id in window_order if window_id in left_rows]
    _require(len(ordered_ids) == len(left_rows), "paired window order is incomplete")
    deltas: list[dict[str, Any]] = []
    metric_values: dict[str, list[float]] = {}
    for window_id in ordered_ids:
        left_row = left_rows[window_id]
        right_row = right_rows[window_id]
        common = sorted(set(left_row["metrics"]) & set(right_row["metrics"]))
        row_delta: dict[str, Any] = {
            "window_id": window_id,
            "target_id": left_row["target_id"],
            "left": left,
            "right": right,
            "metrics": {},
        }
        for metric in common:
            delta = float(right_row["metrics"][metric] - left_row["metrics"][metric])
            row_delta["metrics"][metric] = {
                "left": float(left_row["metrics"][metric]),
                "right": float(right_row["metrics"][metric]),
                "delta_right_minus_left": delta,
            }
            metric_values.setdefault(metric, []).append(delta)
        deltas.append(row_delta)
    return {
        "name": name,
        "left": left,
        "right": right,
        "delta_convention": "right_minus_left",
        "window_count": len(deltas),
        "metric_distributions": {
            metric: numeric_summary(values)
            for metric, values in sorted(metric_values.items())
        },
        "per_window": deltas,
    }


def analyze_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    paired_comparisons: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Produce deterministic cohort, target, and paired summaries."""

    normalized = _normalize_rows(rows)
    checkpoint_order: list[str] = []
    window_order: list[str] = []
    target_order: list[str] = []
    for row in normalized:
        if row["checkpoint_label"] not in checkpoint_order:
            checkpoint_order.append(row["checkpoint_label"])
        if row["window_id"] not in window_order:
            window_order.append(row["window_id"])
        if row["target_id"] not in target_order:
            target_order.append(row["target_id"])
    pairs = _parse_pairs(paired_comparisons, checkpoint_order)
    target_summaries = {
        target: _group_summary(
            [row for row in normalized if row["target_id"] == target]
        )
        for target in sorted(target_order)
    }
    cohort_summary = _group_summary(normalized)
    paired = [
        _paired_comparison(
            normalized,
            name=item["name"],
            left=item["left"],
            right=item["right"],
            window_order=window_order,
        )
        for item in pairs
    ]
    return {
        "schema": ANALYSIS_SCHEMA,
        "checkpoint_order": checkpoint_order,
        "window_order": window_order,
        "window_count": len(window_order),
        "target_order": sorted(target_order),
        "per_window": normalized,
        "per_target": target_summaries,
        "cohort": cohort_summary,
        "paired_comparisons": paired,
        "threshold_used": None,
        "inference_performed": False,
        "input_mutated": False,
    }


def _write_per_window_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    metric_names = sorted({name for row in rows for name in row["metrics"]})
    fieldnames = [
        "checkpoint_label",
        "checkpoint_sha256",
        "window_id",
        "target_id",
        *metric_names,
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output = {
                "checkpoint_label": row["checkpoint_label"],
                "checkpoint_sha256": row["checkpoint_sha256"] or "",
                "window_id": row["window_id"],
                "target_id": row["target_id"],
            }
            output.update({name: row["metrics"].get(name, "") for name in metric_names})
            writer.writerow(output)
        handle.flush()
        os.fsync(handle.fileno())


def _markdown_report(result: Mapping[str, Any]) -> str:
    lines = [
        "# Reusable checkpoint evaluation analysis",
        "",
        f"- Windows: `{result['window_count']}`",
        f"- Checkpoints: `{', '.join(result['checkpoint_order'])}`",
        "- Inference performed by analyzer: `false`",
        "- Threshold used: `null`",
        "",
        "## Cohort distributions",
        "",
        "| Checkpoint | Windows | Metric | Mean | Median | P95 | Min | Max |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for label in result["checkpoint_order"]:
        summary = result["cohort"][label]
        for metric, values in summary["metrics"].items():
            lines.append(
                f"| {label} | {summary['window_count']} | {metric} | "
                f"{values['mean']:.9g} | {values['median']:.9g} | {values['p95']:.9g} | "
                f"{values['min']:.9g} | {values['max']:.9g} |"
            )
    lines.extend(["", "## Paired comparisons", ""])
    if not result["paired_comparisons"]:
        lines.append("No pair was requested by the experiment manifest.")
    for pair in result["paired_comparisons"]:
        lines.extend(
            [
                f"### {pair['name']}",
                "",
                f"`{pair['right']} - {pair['left']}` over `{pair['window_count']}` paired windows.",
                "",
                "| Metric | Mean delta | Median delta | P95 delta | Min delta | Max delta |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for metric, values in pair["metric_distributions"].items():
            lines.append(
                f"| {metric} | {values['mean']:.9g} | {values['median']:.9g} | "
                f"{values['p95']:.9g} | {values['min']:.9g} | {values['max']:.9g} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_analysis(
    result: Mapping[str, Any],
    *,
    output_dir: str | Path,
    source_path: str | Path,
) -> dict[str, Any]:
    """Write analysis artifacts without modifying the source evidence."""

    output = Path(output_dir).expanduser()
    if output.exists():
        _require(
            output.is_dir() and not output.is_symlink(),
            "analysis output must be a regular directory",
        )
    else:
        output.mkdir(parents=True, exist_ok=False)
    normalized = list(result["per_window"])
    write_json(output / "per_window.json", {"rows": normalized})
    write_json(output / "per_target_summary.json", result["per_target"])
    write_json(output / "cohort_summary.json", result["cohort"])
    write_json(
        output / "paired_deltas.json", {"comparisons": result["paired_comparisons"]}
    )
    _write_per_window_csv(output / "per_window.csv", normalized)
    report_path = output / "report.md"
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write(_markdown_report(result))
        handle.flush()
        os.fsync(handle.fileno())
    source = Path(source_path).expanduser().resolve(strict=True)
    result_manifest = {
        "schema": ANALYSIS_SCHEMA,
        "source_path": str(source),
        "source_sha256": sha256_file(source) if source.is_file() else None,
        "artifacts": {
            "per_window_json": "per_window.json",
            "per_window_csv": "per_window.csv",
            "per_target_summary": "per_target_summary.json",
            "cohort_summary": "cohort_summary.json",
            "paired_deltas": "paired_deltas.json",
            "report": "report.md",
        },
        "threshold_used": None,
        "inference_performed": False,
        "input_mutated": False,
    }
    write_json(output / "analysis_manifest.json", result_manifest)
    return result_manifest


__all__ = [
    "ANALYSIS_SCHEMA",
    "CheckpointAnalysisError",
    "analyze_rows",
    "load_rows",
    "numeric_summary",
    "write_analysis",
]
