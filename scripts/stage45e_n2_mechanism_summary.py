#!/usr/bin/env python3
"""CPU-only descriptive summary for the sealed PI0130 mechanism phases.

The script consumes only phase seals, JSONL rows, and the already materialized
legacy checkpoint tree.  It never imports Torch/GR00T, constructs a model, or
changes any scientific value.  Every aggregate is descriptive (no iid test,
threshold, or automatic winner) and is bound to the raw JSONL hashes emitted
by ``stage45e_n2_gpu_sprint.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import median, pstdev
from typing import Any, Iterable, Mapping, Sequence


EXPECTED = {
    "G3_D1": {"rows": 864, "conditioning_passes": 96, "action_head_calls": 864, "model_forward": 864, "model_construction": 4},
    "G4_D2A": {"rows": 384, "conditioning_passes": 96, "action_head_calls": 768, "model_forward": 768, "model_construction": 4},
    "G5_D1": {"rows": 648, "conditioning_passes": 72, "action_head_calls": 648, "model_forward": 648, "model_construction": 3},
    "G5_D2A": {"rows": 288, "conditioning_passes": 72, "action_head_calls": 576, "model_forward": 576, "model_construction": 3},
}
N2_SEEDS = (2222, 2223, 2224)
EXPECTED_LEGACY_TREE = "2113a6a3b2a91233a42214dbb017c0136acde282fda2a418066f9317e78981a5"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path) -> Any:
    require(path.is_file() and not path.is_symlink(), f"JSON is not a regular file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def numeric_summary(values: Iterable[float]) -> dict[str, Any]:
    data = [float(value) for value in values]
    require(data and all(math.isfinite(value) for value in data), "empty or non-finite metric group")
    ordered = sorted(data)
    return {
        "count": len(data),
        "mean": sum(data) / len(data),
        "std_population": pstdev(data) if len(data) > 1 else 0.0,
        "min": ordered[0],
        "median": median(ordered),
        "max": ordered[-1],
    }


def tree_entries(root: Path, prefix: str) -> list[dict[str, Any]]:
    root = root.resolve()
    require(root.is_dir() and not root.is_symlink(), f"unsafe legacy tree root: {root}")
    entries: list[dict[str, Any]] = []
    for directory, dirnames, filenames in __import__("os").walk(root, followlinks=False):
        directory_path = Path(directory)
        for dirname in dirnames:
            child = directory_path / dirname
            require(child.is_dir() and not child.is_symlink(), f"unsafe legacy directory: {child}")
        for filename in filenames:
            child = directory_path / filename
            require(child.is_file() and not child.is_symlink(), f"unsafe legacy file: {child}")
            entries.append({"path": f"{prefix}/{child.relative_to(root).as_posix()}", "bytes": child.stat().st_size, "sha256": sha256_file(child)})
    return sorted(entries, key=lambda item: item["path"])


def tree_digest(entries: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda item: str(item["path"])):
        digest.update(f"{entry['path']}\0{entry['bytes']}\0{entry['sha256']}\n".encode("utf-8"))
    return digest.hexdigest()


def _phase_metric_sections(phase: str) -> tuple[str, ...]:
    if phase in {"G3_D1", "G5_D1"}:
        return ("normalized_local_flow",)
    return ("state_path_divergence", "field_sensitivity", "teacher_bridge_local_error", "off_path_diagnostic")


def _metric_fields(section: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(field for field in ("l2", "rmse", "mae", "max_abs", "cosine") if field in section)


def _row_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    identity = row.get("identity", {})
    return (identity.get("checkpoint"), identity.get("window_id"), identity.get("noise_seed"), identity.get("t_bucket"), identity.get("stage_index"))


def _artifact_paths(root: Path, phase: str, identity: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    label = str(identity["checkpoint"]).replace("@", "_at_")
    window = f"w{int(identity['window_index']):02d}"
    seed = f"seed{int(identity['noise_seed'])}"
    timestep = f"t{int(identity['t_bucket']):03d}"
    probe = root / "probes" / label / window / seed / timestep
    return probe / "raw_arrays.npz", probe / "raw_manifest.json", probe / "derived_metrics.json"


def load_phase(root: Path, phase: str) -> dict[str, Any]:
    expected = EXPECTED[phase]
    root = root.resolve()
    seal = read_json(root / "seal.json")
    require(seal.get("status") == "PASS", f"{phase} seal is not PASS")
    require(seal.get("counters", {}).get("_phase") == phase, f"{phase} phase marker drift")
    counters = seal.get("counters", {})
    for field in ("conditioning_passes", "action_head_calls", "model_forward", "model_construction"):
        require(counters.get(field) == expected[field], f"{phase} {field} budget drift")
    require(counters.get("raw_stage_count") == expected["rows"], f"{phase} raw row budget drift")
    require(counters.get("training") is False and counters.get("optimizer_steps") == 0, f"{phase} training boundary drift")
    require(seal.get("WBC") == 0 and seal.get("Validation-E") == 0 and seal.get("Holdout-F") == 0, f"{phase} locked boundary drift")
    raw_path = root / "raw_probe_rows.jsonl"
    require(raw_path.is_file() and not raw_path.is_symlink(), f"{phase} raw JSONL is absent")
    raw_sha = sha256_file(raw_path)
    require(raw_sha == seal.get("raw_probe_rows_jsonl_sha256"), f"{phase} raw JSONL hash mismatch")
    rows: list[dict[str, Any]] = []
    keys: set[tuple[Any, ...]] = set()
    missing_artifacts = 0
    invalid_metrics = 0
    target_xyz_exposure = 0
    with raw_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            require(line.strip(), f"{phase} empty raw JSONL line {line_number}")
            row = json.loads(line)
            require(isinstance(row, dict), f"{phase} row {line_number} is not an object")
            key = _row_key(row)
            require(key not in keys, f"{phase} duplicate row key {key!r}")
            keys.add(key)
            identity = row.get("identity", {})
            require(identity.get("target_xyz_in_model_input") is False, f"{phase} target XYZ exposure drift")
            if identity.get("target_xyz_in_model_input") is not False:
                target_xyz_exposure += 1
            require(row.get("finite") is True, f"{phase} non-finite row {line_number}")
            for section in _phase_metric_sections(phase):
                payload = row.get(section)
                require(isinstance(payload, Mapping), f"{phase} missing {section} row {line_number}")
                for field in _metric_fields(payload):
                    require(finite_number(payload[field]), f"{phase} non-finite {section}.{field} row {line_number}")
                if not _metric_fields(payload):
                    invalid_metrics += 1
            # The row binds both evidence layers; verify the sidecars and the
            # referenced raw archive, but deliberately do not run any model or
            # decode operation here.
            raw_archive, raw_manifest, derived_metrics = _artifact_paths(root, phase, identity)
            for candidate, expected_sha in ((raw_archive, row.get("raw_sha256")), (raw_manifest, row.get("raw_manifest_sha256")), (derived_metrics, row.get("derived_sha256"))):
                if not candidate.is_file() or candidate.is_symlink() or not isinstance(expected_sha, str) or sha256_file(candidate) != expected_sha:
                    missing_artifacts += 1
                require(candidate.is_file() and not candidate.is_symlink(), f"{phase} evidence artifact is absent: {candidate}")
                require(sha256_file(candidate) == expected_sha, f"{phase} evidence artifact hash mismatch: {candidate}")
            raw_doc = read_json(raw_manifest)
            require(raw_doc.get("status") == "RAW_SEALED_BEFORE_DERIVED", f"{phase} raw-before-derived status drift")
            require(raw_doc.get("raw_archive", {}).get("path") == str(raw_archive.relative_to(root)), f"{phase} raw archive path binding drift")
            require(raw_doc.get("raw_archive", {}).get("sha256") == row.get("raw_sha256"), f"{phase} raw archive hash binding drift")
            derived_doc = read_json(derived_metrics)
            require(derived_doc.get("status") == "DERIVED_AFTER_RAW_SEAL", f"{phase} derived status drift")
            rows.append(row)
    require(len(rows) == expected["rows"], f"{phase} row count {len(rows)} != {expected['rows']}")
    labels = sorted({str(row["identity"]["checkpoint"]) for row in rows})
    require(len(labels) == (4 if phase in {"G3_D1", "G4_D2A"} else 3), f"{phase} checkpoint cardinality drift")
    return {"phase": phase, "root": str(root), "seal": seal, "rows": rows, "raw_sha256": raw_sha, "labels": labels, "integrity": {"row_count": len(rows), "unique_row_keys": len(keys), "missing_artifacts": missing_artifacts, "invalid_metric_rows": invalid_metrics, "target_xyz_exposure_rows": target_xyz_exposure}}


def metric_group(rows: Sequence[Mapping[str, Any]], section: str, field: str) -> dict[str, Any]:
    values: list[float] = []
    for row in rows:
        payload = row.get(section)
        if isinstance(payload, Mapping) and finite_number(payload.get(field)):
            values.append(float(payload[field]))
    require(values, f"no values for {section}.{field}")
    return numeric_summary(values)


def grouped(rows: Sequence[Mapping[str, Any]], key_fn: Any) -> dict[str, list[Mapping[str, Any]]]:
    result: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        key = str(key_fn(row))
        result.setdefault(key, []).append(row)
    return result


def summarize_rows(rows: Sequence[Mapping[str, Any]], phase: str) -> dict[str, Any]:
    sections = _phase_metric_sections(phase)
    section_summary: dict[str, Any] = {}
    for section in sections:
        fields = sorted({field for row in rows if isinstance(row.get(section), Mapping) for field in _metric_fields(row[section])})
        section_summary[section] = {field: metric_group(rows, section, field) for field in fields}
    return {"row_count": len(rows), "metric_sections": section_summary}


def per_target_and_window(data: Mapping[str, Any]) -> dict[str, Any]:
    phase = str(data["phase"])
    rows = data["rows"]
    target_groups = grouped(rows, lambda row: row["identity"]["target_id"])
    window_groups = grouped(rows, lambda row: row["identity"]["window_id"])
    return {
        "phase": phase,
        "per_target": {target: summarize_rows(group, phase) for target, group in sorted(target_groups.items())},
        "per_window": {window: summarize_rows(group, phase) for window, group in sorted(window_groups.items())},
    }


def per_checkpoint(data: Mapping[str, Any]) -> dict[str, Any]:
    phase = str(data["phase"])
    rows = data["rows"]
    groups = grouped(rows, lambda row: row["identity"]["checkpoint"])
    return {label: summarize_rows(group, phase) for label, group in sorted(groups.items())}


def seed_level(data: Mapping[str, Any]) -> dict[str, Any]:
    phase = str(data["phase"])
    rows = data["rows"]
    n2 = [row for row in rows if str(row["identity"]["checkpoint"]).startswith("N2(seed")]
    by_seed = grouped(n2, lambda row: row["identity"]["checkpoint"])
    return {label: summarize_rows(group, phase) for label, group in sorted(by_seed.items())}


def exposure_comparison(d1_1000: Mapping[str, Any], d1_2000: Mapping[str, Any], d2_1000: Mapping[str, Any], d2_2000: Mapping[str, Any]) -> dict[str, Any]:
    def compare(low: Mapping[str, Any], high: Mapping[str, Any], phase_low: str, phase_high: str, section_names: Sequence[str]) -> dict[str, Any]:
        result: dict[str, Any] = {"low_phase": phase_low, "high_phase": phase_high, "per_seed": {}}
        low_groups = grouped([row for row in low["rows"] if str(row["identity"]["checkpoint"]).startswith("N2(seed")], lambda row: row["identity"]["checkpoint"])
        high_groups = grouped([row for row in high["rows"] if str(row["identity"]["checkpoint"]).startswith("N2(seed")], lambda row: row["identity"]["checkpoint"])
        for seed in N2_SEEDS:
            label = f"N2(seed{seed})@{1000 if phase_low.startswith('G5') else 2000}"
            high_label = f"N2(seed{seed})@{2000 if phase_high.startswith('G3') or phase_high.startswith('G4') else 1000}"
            require(label in low_groups and high_label in high_groups, f"missing exposure pair {label}/{high_label}")
            fields: dict[str, Any] = {}
            for section in section_names:
                names = sorted({field for row in low_groups[label] + high_groups[high_label] if isinstance(row.get(section), Mapping) for field in _metric_fields(row[section])})
                fields[section] = {}
                for field in names:
                    low_mean = metric_group(low_groups[label], section, field)["mean"]
                    high_mean = metric_group(high_groups[high_label], section, field)["mean"]
                    fields[section][field] = {"low_mean": low_mean, "high_mean": high_mean, "delta_high_minus_low": high_mean - low_mean}
            result["per_seed"][str(seed)] = fields
        return result

    return {
        "D1_N2_1000_to_2000": compare(d1_1000, d1_2000, "G5_D1", "G3_D1", ("normalized_local_flow",)),
        "D2A_N2_1000_to_2000": compare(d2_1000, d2_2000, "G5_D2A", "G4_D2A", ("state_path_divergence", "field_sensitivity", "teacher_bridge_local_error", "off_path_diagnostic")),
        "interpretation_boundary": "Descriptive exposure deltas only; no iid significance test, success threshold, or automatic winner.",
    }


def cross_seed(data: Mapping[str, Any]) -> dict[str, Any]:
    phase = str(data["phase"])
    rows = data["rows"]
    labels = [f"N2(seed{seed})@{2000 if phase in {'G3_D1', 'G4_D2A'} else 1000}" for seed in N2_SEEDS]
    groups = grouped(rows, lambda row: row["identity"]["checkpoint"])
    result: dict[str, Any] = {"phase": phase, "seed_level": {}}
    for section in _phase_metric_sections(phase):
        fields = sorted({field for label in labels for row in groups.get(label, []) if isinstance(row.get(section), Mapping) for field in _metric_fields(row[section])})
        result["seed_level"][section] = {}
        for field in fields:
            seed_means = {label: metric_group(groups[label], section, field)["mean"] for label in labels}
            result["seed_level"][section][field] = {"per_seed_mean": seed_means, "across_seed_mean": sum(seed_means.values()) / len(seed_means), "across_seed_std_population": pstdev(seed_means.values())}
    return result


def outliers(data: Mapping[str, Any], limit: int = 5) -> dict[str, Any]:
    phase = str(data["phase"])
    rows = data["rows"]
    section = "normalized_local_flow" if phase in {"G3_D1", "G5_D1"} else "state_path_divergence"
    field = "rmse" if section == "normalized_local_flow" else "l2"
    ranked = sorted(rows, key=lambda row: abs(float(row[section][field])), reverse=True)[:limit]
    return {"section": section, "field": field, "top": [{"checkpoint": row["identity"]["checkpoint"], "target_id": row["identity"]["target_id"], "window_id": row["identity"]["window_id"], "noise_seed": row["identity"].get("noise_seed"), "t_bucket": row["identity"].get("t_bucket"), "value": row[section][field]} for row in ranked]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g3-root", type=Path, required=True)
    parser.add_argument("--g4-root", type=Path, required=True)
    parser.add_argument("--g5-d1-root", type=Path, required=True)
    parser.add_argument("--g5-d2a-root", type=Path, required=True)
    parser.add_argument("--relay-pass", type=Path, required=True)
    parser.add_argument("--legacy-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    require(not output.exists() and not output.is_symlink(), f"summary output must be fresh: {output}")
    output.mkdir(parents=True, exist_ok=False)

    relay = read_json(args.relay_pass.resolve())
    require(relay.get("status") == "PASS" and relay.get("independent_corelab_tree_rehash") is True, "legacy relay materialization is not an accepted independent PASS")
    relay_checkpoint = relay.get("checkpoints", {}).get("FULL24@2000", {})
    require(relay_checkpoint.get("tree_sha256") == EXPECTED_LEGACY_TREE, "relay manifest FULL24@2000 tree drift")
    actual_tree = tree_digest(tree_entries(args.legacy_checkpoint.resolve(), "checkpoint-2000"))
    require(actual_tree == EXPECTED_LEGACY_TREE, f"independent legacy tree rehash mismatch: {actual_tree}")

    phases = {
        "G3_D1": load_phase(args.g3_root, "G3_D1"),
        "G4_D2A": load_phase(args.g4_root, "G4_D2A"),
        "G5_D1": load_phase(args.g5_d1_root, "G5_D1"),
        "G5_D2A": load_phase(args.g5_d2a_root, "G5_D2A"),
    }
    summary = {
        "schema": "CORELAB_STAGE45E_N2_CPU_MECHANISM_SUMMARY_V1",
        "status": "PASS",
        "stage": "Stage45E",
        "authority": {"pi_gate": "PI_GATE_STATE_0130", "issue_11_comment": 5642258588, "pr_56_comment": 5642259704},
        "scientific_policy": {"descriptive_only": True, "iid_significance": False, "automatic_winner": False, "threshold": None, "model_forward": False, "gpu": False, "training": False, "Validation-E": 0, "Holdout-F": 0, "WBC": 0, "PBS": 0, "hardware": 0},
        "legacy_materialization": {"relay_pass_sha256": sha256_file(args.relay_pass.resolve()), "full24_2000_tree_sha256": actual_tree, "full24_2000_checkpoint_path": str(args.legacy_checkpoint.resolve())},
        "phase_bindings": {name: {"root": data["root"], "seal_sha256": sha256_file(Path(data["root"]) / "seal.json"), "raw_probe_rows_jsonl_sha256": data["raw_sha256"], "repository_head": data["seal"].get("repository_head"), "counters": data["seal"].get("counters")} for name, data in phases.items()},
        "integrity": {name: data["integrity"] for name, data in phases.items()},
        "per_checkpoint": {name: per_checkpoint(data) for name, data in phases.items()},
        "cross_seed": {name: cross_seed(data) for name, data in phases.items()},
        "per_target_and_window": {name: per_target_and_window(data) for name, data in phases.items()},
        "outliers": {name: outliers(data) for name, data in phases.items()},
        "exposure_comparison": exposure_comparison(phases["G5_D1"], phases["G3_D1"], phases["G5_D2A"], phases["G4_D2A"]),
        "mechanism_readout": {
            "G3_D1": "Teacher-forced local-flow error is summarized per @2000 checkpoint and N2 seed; this measures local field agreement, not free-running task success.",
            "G4_D2A": "Matched teacher-forced/free-running path divergence and field sensitivity are summarized at @2000; off-path values remain descriptive and not supervised.",
            "G5_D1": "N2 @1000 local-flow error is summarized across the three independent seeds for exposure comparison.",
            "G5_D2A": "N2 @1000 matched path divergence and field sensitivity are summarized for the same exposure comparison.",
            "boundary": "Results characterize exposure and mechanism signals; they do not declare a generalization winner or causal proof beyond the frozen probes.",
        },
    }
    payload = canonical_json(summary).encode("utf-8")
    report_path = output / "mechanism_summary.json"
    report_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (output / "mechanism_summary.sha256").write_text(hashlib.sha256(report_path.read_bytes()).hexdigest() + "  mechanism_summary.json\n", encoding="utf-8")
    # The canonical payload hash is recorded separately to make formatting
    # changes distinguishable from scientific-content changes.
    (output / "canonical_payload.sha256").write_text(hashlib.sha256(payload).hexdigest() + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "schema": summary["schema"], "output_root": str(output), "mechanism_summary_sha256": sha256_file(report_path), "canonical_payload_sha256": hashlib.sha256(payload).hexdigest()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
