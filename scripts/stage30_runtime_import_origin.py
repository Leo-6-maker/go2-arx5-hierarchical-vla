#!/usr/bin/env python3
"""Import-only Stage30 provenance closure for the Stage29 GR00T runtime."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "STAGE30_RUNTIME_IMPORT_ORIGIN_V1"
PI_GATE = "PI_GATE_STATE_0010"

CRITICAL_MODULES = {
    "gr00t": ("gr00t/__init__.py", "1f5bc8699a273afc0df1601cf9b7ac40e2587bf65907961f6714d6f4f18df185"),
    "gr00t.policy.policy": ("gr00t/policy/policy.py", "b53326aee0c26ea34201d70510ad92608a96929d7f39039f578c3d865446de23"),
    "gr00t.policy.gr00t_policy": ("gr00t/policy/gr00t_policy.py", "ad1183a3ce1ebed6cace4a9cb8c5a123e823a0760ebf46f3694639a5ad1521ec"),
    "gr00t.model.gr00t_n1d7.gr00t_n1d7": ("gr00t/model/gr00t_n1d7/gr00t_n1d7.py", "6bd66ef768f1484ac5a7889c40358a599f1862b1d554d47d357e113f2f96501c"),
    "gr00t.configs.model.gr00t_n1d7": ("gr00t/configs/model/gr00t_n1d7.py", "379ff708496a73b25b12f4656850284a16fde0e205828e2a0c0be0af61463744"),
    "gr00t.model.gr00t_n1d7.processing_gr00t_n1d7": ("gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py", "77fd0e1f6972cdcf0334f18f2e84724629f6746612721b6b27135ccb23f00b3f"),
    "gr00t.data.state_action.state_action_processor": ("gr00t/data/state_action/state_action_processor.py", "0151dd0bb6cb727fd401bf78121efc532e7184f2575e6ef25294085f629ac145"),
    "gr00t.data.utils": ("gr00t/data/utils.py", "fa765b240295d9884ab62326c8a21465ae21e6f10133a4b68ac321e8101e4ab7"),
    "gr00t.eval.open_loop_eval": ("gr00t/eval/open_loop_eval.py", "e15706fac4cd8caff3229a1538eb0288b487919b28c0e26ecf30c32d0f61245f"),
}


class ImportAuditError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise ImportAuditError(f"not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *arguments], text=True).strip()


def classify_import_records(records: Sequence[Mapping[str, Any]]) -> str:
    if any(record.get("byte_exact_to_pinned") is not True for record in records):
        return "MATERIAL_CONFLICT"
    if all(record.get("inside_pinned_source_tree") is True for record in records):
        return "PASS"
    return "PASS_WITH_PROVENANCE_NOTE"


def _source_path(imported_path: Path) -> Path:
    if imported_path.suffix == ".pyc":
        try:
            return Path(importlib.util.source_from_cache(str(imported_path)))
        except ValueError as exc:
            raise ImportAuditError(f"cannot resolve source for imported bytecode: {imported_path}") from exc
    return imported_path


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.authorization_token != PI_GATE:
        raise ImportAuditError("PI gate token mismatch")
    head = _git(args.project_repo, "rev-parse", "HEAD")
    if head != args.expected_project_sha or _git(args.project_repo, "status", "--short"):
        raise ImportAuditError("Stage30 checkout SHA/cleanliness mismatch")
    for path, expected, label in (
        (Path(__file__).resolve(), args.expected_runner_sha256, "P0 runner"),
        (args.wrapper, args.expected_wrapper_sha256, "P0 wrapper"),
    ):
        observed = _sha256_file(path)
        if observed != expected:
            raise ImportAuditError(f"{label} SHA mismatch: {observed}")

    pinned_root = args.pinned_groot_source.resolve(strict=True)
    records = []
    for module_name, (relative_path, expected_sha256) in CRITICAL_MODULES.items():
        pinned_path = (pinned_root / relative_path).resolve(strict=True)
        try:
            pinned_path.relative_to(pinned_root)
        except ValueError as exc:
            raise ImportAuditError(f"pinned source escaped tree: {pinned_path}") from exc
        pinned_sha256 = _sha256_file(pinned_path)
        if pinned_sha256 != expected_sha256:
            raise ImportAuditError(f"pinned source SHA mismatch for {module_name}: {pinned_sha256}")

        module = importlib.import_module(module_name)
        module_file = getattr(module, "__file__", None)
        if not module_file:
            raise ImportAuditError(f"imported module has no __file__: {module_name}")
        imported_path = Path(module_file)
        imported_resolved = imported_path.resolve(strict=True)
        source_resolved = _source_path(imported_resolved).resolve(strict=True)
        imported_sha256 = _sha256_file(imported_resolved)
        source_sha256 = _sha256_file(source_resolved)
        try:
            source_resolved.relative_to(pinned_root)
            inside = True
        except ValueError:
            inside = False
        records.append({
            "module": module_name,
            "module_file": str(imported_path),
            "resolved_import_file": str(imported_resolved),
            "import_file_sha256": imported_sha256,
            "resolved_source_file": str(source_resolved),
            "source_file_sha256": source_sha256,
            "expected_pinned_path": str(pinned_path),
            "expected_pinned_sha256": expected_sha256,
            "inside_pinned_source_tree": inside,
            "byte_exact_to_pinned": source_sha256 == expected_sha256,
        })

    status = classify_import_records(records)
    payload = {
        "schema": SCHEMA,
        "status": status,
        "pi_gate": PI_GATE,
        "project_sha": head,
        "pinned_groot_source": str(pinned_root),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "sys_path": list(sys.path),
            "pythonpath": os.environ.get("PYTHONPATH"),
            "python_no_user_site": os.environ.get("PYTHONNOUSERSITE"),
            "python_dont_write_bytecode": os.environ.get("PYTHONDONTWRITEBYTECODE"),
        },
        "runtime": {
            "container_image": args.container_image,
            "container_image_sha256": _sha256_file(args.container_image),
            "stage29_pythonpath_reproduced_exactly": True,
            "gpu_passthrough": False,
            "imports_only": True,
        },
        "modules": records,
        "module_count": len(records),
        "inside_pinned_count": sum(record["inside_pinned_source_tree"] for record in records),
        "byte_exact_count": sum(record["byte_exact_to_pinned"] for record in records),
        "model_constructed": False,
        "checkpoint_loaded": False,
        "model_forward": False,
        "sampler_calls": 0,
        "gpu_scientific_execution": False,
        "scientific_contract_changed": False,
    }
    _write_json_atomic(args.output, payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-repo", type=Path, required=True)
    parser.add_argument("--expected-project-sha", required=True)
    parser.add_argument("--pinned-groot-source", type=Path, required=True)
    parser.add_argument("--container-image", type=Path, required=True)
    parser.add_argument("--wrapper", type=Path, required=True)
    parser.add_argument("--expected-runner-sha256", required=True)
    parser.add_argument("--expected-wrapper-sha256", required=True)
    parser.add_argument("--authorization-token", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        payload = run(_parser().parse_args(argv))
        print(json.dumps({
            "schema": payload["schema"],
            "status": payload["status"],
            "module_count": payload["module_count"],
            "inside_pinned_count": payload["inside_pinned_count"],
            "byte_exact_count": payload["byte_exact_count"],
            "model_forward": False,
            "sampler_calls": 0,
        }, indent=2, sort_keys=True))
        return 0 if payload["status"] != "MATERIAL_CONFLICT" else 2
    except Exception as exc:
        print(json.dumps({
            "schema": SCHEMA,
            "status": "BLOCKED",
            "error_type": type(exc).__name__,
            "error_summary": str(exc).splitlines()[0][:500],
            "model_forward": False,
            "sampler_calls": 0,
        }, indent=2, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
