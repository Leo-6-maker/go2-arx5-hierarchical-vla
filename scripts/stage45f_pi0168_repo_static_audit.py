#!/usr/bin/env python3
"""CPU-only, standard-library repository audit for PI0168.

The auditor deliberately treats source files, committed metadata, and Git
state as its only inputs.  It does not import project code, inspect experiment
payloads, resolve a model, contact a scheduler, or execute a runtime.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence


AUDIT_SCHEMA = "STAGE45F_PI0168_REPOSITORY_STATIC_AUDIT_V1"
AUDITED_VANDA_HEAD = "940df1ab6f428fd0260fe836e36d4cd98c6161e3"

FORBIDDEN_IMPORT_PREFIXES = (
    "torch",
    "transformers",
    "gr00t",
    "numpy",
    "cv2",
    "isaacgym",
    "go2arx5_vla",
)

RISK_CALL_NAMES = {
    "backward",
    "cuda",
    "from_pretrained",
    "get_action",
    "model_imports_present",
    "qsub",
    "raw_inventory",
    "save_pretrained",
    "train",
    "validate_artifacts",
}

TARGETS = (
    "scripts/stage45f_pi0163_validation_e_recovery.py",
    "scripts/stage45f_pi0159_validation_e_continuation.py",
    "scripts/stage45f_vanda_validation_e_admission.py",
    "scripts/stage45f_vanda_validation_e_gen1.py",
    "scripts/stage45c_h200_eval_runtime_adapter.py",
    "scripts/stage26_h1_c1_fixed_noise_local_flow.py",
    "src/go2arx5_vla/evaluation/checkpoint_eval.py",
)

STATIC_CONTEXT_PATHS = (
    "hpc/pbs/stage45f_pi0163_validation_e_gen1_h200.pbs",
    "hpc/pbs/stage45f_validation_e_gen1_h200.pbs",
    "hpc/pbs/stage45f_pi0159_validation_e_continuation_h200.pbs",
    "scripts/stage45f_pi0163_ffmpeg_capsule_bootstrap.py",
    "scripts/stage45f_pi0163_runtime_import_probe.py",
)


def _module_for_import(node: ast.Import | ast.ImportFrom) -> str:
    if isinstance(node, ast.Import):
        return ",".join(alias.name for alias in node.names)
    prefix = "." * int(node.level)
    return prefix + (node.module or "")


def _is_forbidden_module(module: str) -> bool:
    normalized = module.lstrip(".")
    return any(
        normalized == prefix or normalized.startswith(prefix + ".")
        for prefix in FORBIDDEN_IMPORT_PREFIXES
    )


def _call_name(node: ast.Call) -> str:
    value: ast.AST = node.func
    parts: list[str] = []
    while isinstance(value, ast.Attribute):
        parts.append(value.attr)
        value = value.value
    if isinstance(value, ast.Name):
        parts.append(value.id)
    return ".".join(reversed(parts))


class _SourceVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.scope: list[str] = []
        self.imports: list[dict[str, Any]] = []
        self.calls: list[dict[str, Any]] = []
        self.dynamic_imports: list[dict[str, Any]] = []

    def _scope_name(self) -> str:
        return ".".join(self.scope) if self.scope else "<module>"

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.append(
                {
                    "line": node.lineno,
                    "module": alias.name,
                    "kind": "import",
                    "scope": self._scope_name(),
                    "top_level": not self.scope,
                    "forbidden": _is_forbidden_module(alias.name),
                }
            )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = _module_for_import(node)
        self.imports.append(
            {
                "line": node.lineno,
                "module": module,
                "kind": "from",
                "scope": self._scope_name(),
                "top_level": not self.scope,
                "forbidden": _is_forbidden_module(module),
            }
        )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = _call_name(node)
        if name.rsplit(".", 1)[-1] in RISK_CALL_NAMES:
            self.calls.append({"line": node.lineno, "name": name, "scope": self._scope_name()})
        if name.rsplit(".", 1)[-1] == "import_module":
            module = None
            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                module = node.args[0].value
            self.dynamic_imports.append(
                {"line": node.lineno, "module": module, "scope": self._scope_name()}
            )
        self.generic_visit(node)


def analyze_source(source: str, *, path: str = "<fixture>") -> dict[str, Any]:
    """Return AST-only import and risk evidence for one source string."""

    tree = ast.parse(source, filename=path)
    visitor = _SourceVisitor()
    visitor.visit(tree)
    imports = sorted(visitor.imports, key=lambda item: (int(item["line"]), str(item["module"])))
    calls = sorted(visitor.calls, key=lambda item: (int(item["line"]), str(item["name"])))
    dynamic = sorted(visitor.dynamic_imports, key=lambda item: int(item["line"]))
    top_forbidden = [item for item in imports if item["top_level"] and item["forbidden"]]
    local_forbidden = [item for item in imports if not item["top_level"] and item["forbidden"]]
    return {
        "path": path,
        "top_level_imports": [item for item in imports if item["top_level"]],
        "function_local_imports": [item for item in imports if not item["top_level"]],
        "top_level_forbidden_imports": top_forbidden,
        "function_local_forbidden_imports": local_forbidden,
        "risk_calls": calls,
        "dynamic_imports": dynamic,
        "syntax": "PASS",
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}")
    return completed.stdout.strip()


def _source_record(repo: Path, relative: str) -> dict[str, Any]:
    path = repo / relative
    record: dict[str, Any] = {"path": relative, "exists": path.is_file()}
    if not path.is_file():
        record["syntax"] = "MISSING"
        return record
    source = path.read_text(encoding="utf-8")
    evidence = analyze_source(source, path=relative) if path.suffix == ".py" else {
        "path": relative,
        "top_level_imports": [],
        "function_local_imports": [],
        "top_level_forbidden_imports": [],
        "function_local_forbidden_imports": [],
        "risk_calls": [],
        "dynamic_imports": [],
        "syntax": "NOT_PYTHON_SOURCE",
    }
    record.update(evidence)
    record["sha256"] = _sha256_file(path)
    record["line_count"] = len(source.splitlines())
    return record


def _calls(record: Mapping[str, Any], suffix: str) -> list[dict[str, Any]]:
    return [item for item in record.get("risk_calls", []) if str(item.get("name", "")).endswith(suffix)]


def _known_pi0167_blocker(recovery: Mapping[str, Any]) -> dict[str, Any]:
    local_numpy = [
        item
        for item in recovery.get("function_local_imports", [])
        if item.get("module") == "numpy" and item.get("scope") == "raw_inventory"
    ]
    inventory_calls = _calls(recovery, "raw_inventory")
    admission_calls = _calls(recovery, "validate_artifacts")
    pre_admission_calls = [
        item for item in inventory_calls if admission_calls and int(item["line"]) < int(admission_calls[0]["line"])
    ]
    present = bool(local_numpy and pre_admission_calls and admission_calls)
    return {
        "classification": "KNOWN_PI0167_BLOCKER" if present else "NOT_REPRODUCED_STATICALLY",
        "present": present,
        "import": local_numpy,
        "pre_admission_inventory_calls": pre_admission_calls,
        "first_validate_artifacts_call": admission_calls[:1],
        "mechanical_fix_class": "ISOLATED_CHILD_PROCESS_BEFORE_PARENT_ADMISSION",
        "owner": "VANDA",
        "new_p0": False,
    }


def _boundary_record(records: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    recovery = records["scripts/stage45f_pi0163_validation_e_recovery.py"]
    gen1 = records["scripts/stage45f_vanda_validation_e_gen1.py"]
    recovery_calls = recovery.get("risk_calls", [])
    gen1_calls = gen1.get("risk_calls", [])
    recovery_validation = _calls(recovery, "validate_artifacts")
    recovery_inventory = _calls(recovery, "raw_inventory")
    gen1_validation = _calls(gen1, "validate_artifacts")
    gen1_guard = _calls(gen1, "model_imports_present")
    return {
        "recovery_parent_process": {
            "ordered_static_anchors": [
                {"name": "raw_inventory", "lines": [item["line"] for item in recovery_inventory]},
                {"name": "validate_artifacts", "lines": [item["line"] for item in recovery_validation]},
                {"name": "post_boundary_runtime_imports", "line_hint": 1035},
            ],
            "known_import_contamination_before_admission": _known_pi0167_blocker(recovery),
            "model_calls_executed_by_audit": 0,
        },
        "frozen_gen1_boundary": {
            "model_import_guard_calls": gen1_guard,
            "validate_artifacts_calls": gen1_validation,
            "source_recorded_as": "guard_first_then_payload_and_runtime_helpers",
            "audit_conclusion": "STATIC_ORDER_REQUIRES_RUNTIME_ASSERTION",
        },
        "all_target_risk_calls": {
            path: [
                {"line": item["line"], "name": item["name"], "scope": item["scope"]}
                for item in record.get("risk_calls", [])
            ]
            for path, record in records.items()
        },
    }


def build_report(repo: Path, *, audited_vanda_head: str = AUDITED_VANDA_HEAD) -> dict[str, Any]:
    repo = repo.resolve()
    records = {path: _source_record(repo, path) for path in TARGETS}
    context = {path: _source_record(repo, path) for path in STATIC_CONTEXT_PATHS}
    forbidden_top_level = {
        path: record["top_level_forbidden_imports"]
        for path, record in records.items()
        if record.get("top_level_forbidden_imports")
    }
    missing = [path for path, record in {**records, **context}.items() if not record.get("exists")]
    syntax_failures = [
        path
        for path, record in records.items()
        if record.get("syntax") not in {"PASS", "NOT_PYTHON_SOURCE"}
    ]
    git_status = _git(repo, "status", "--porcelain", "--untracked-files=all")
    runtime_roots = tuple(FORBIDDEN_IMPORT_PREFIXES)
    loaded_runtime = sorted(
        name for name in sys.modules if name == runtime_roots or any(
            name == root or name.startswith(root + ".") for root in runtime_roots
        )
    )
    blocker = _known_pi0167_blocker(records["scripts/stage45f_pi0163_validation_e_recovery.py"])
    return {
        "schema": AUDIT_SCHEMA,
        "status": "PASS_STATIC_AUDIT_WITH_KNOWN_PI0167_BLOCKER" if blocker["present"] else "PASS_STATIC_AUDIT",
        "scope": {
            "source_snapshot": audited_vanda_head,
            "source_snapshot_role": "PR71_REVIEWED_HEAD_AT_PI0168_ISSUANCE",
            "payloads_read": 0,
            "validation_e_payloads_read": 0,
            "holdout_f_payloads_read": 0,
            "model_calls": 0,
            "training_calls": 0,
            "gpu_calls": 0,
            "pbs_submissions": 0,
            "runtime_imports_executed": 0,
            "dependencies_installed": 0,
        },
        "repository": {
            "path": str(repo),
            "branch": _git(repo, "branch", "--show-current"),
            "head": _git(repo, "rev-parse", "HEAD"),
            "clean_at_audit_start": not bool(git_status),
            "git_status_at_audit_start": git_status.splitlines(),
        },
        "targets": records,
        "static_context": context,
        "import_purity": {
            "top_level_forbidden_imports": forbidden_top_level,
            "known_pi0167_blocker": blocker,
            "audit_process_project_runtime_imports": loaded_runtime,
            "audit_process_project_runtime_imports_false": not bool(loaded_runtime),
            "mechanical_fix_classes": [
                "ISOLATED_CHILD_PROCESS_BEFORE_PARENT_ADMISSION",
                "MOVE_RUNTIME_IMPORTS_AFTER_FROZEN_ADMISSION",
                "KEEP_PURE_STATIC_ADMISSION_MODULE_STDLIB_ONLY",
            ],
        },
        "process_boundaries": _boundary_record(records),
        "quality": {
            "missing_targets": missing,
            "python_syntax_failures": syntax_failures,
            "top_level_forbidden_import_count": sum(len(value) for value in forbidden_top_level.values()),
            "audit_process_uses_only_stdlib_and_git_metadata": True,
            "status": "PASS" if not missing and not syntax_failures else "BLOCKED",
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--audited-vanda-head", default=AUDITED_VANDA_HEAD)
    args = parser.parse_args(argv)
    report = build_report(args.repo, audited_vanda_head=args.audited_vanda_head)
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if report["quality"]["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
