#!/usr/bin/env python3
"""Launch the manifest-driven inference backend from an absolute path."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys


# Bootstrap both the repository package root and ``src`` before any repository
# import.  This is required for absolute-path launches from a foreign CWD with
# an empty PYTHONPATH.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for _import_root in (REPO_ROOT, SRC_ROOT):
    _import_root_text = str(_import_root)
    if _import_root_text not in sys.path:
        sys.path.insert(0, _import_root_text)


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False)


def _root_error(selected: str | None) -> dict[str, object]:
    return {
        "status": "BLOCKED",
        "error_type": "RepositoryRootError",
        "error": "repo root does not match launcher checkout",
        "selected_root": str(Path(selected).expanduser().resolve())
        if selected is not None
        else str(REPO_ROOT),
        "launcher_root": str(REPO_ROOT),
        "model_modules_present": [],
    }


def _validate_root(selected: str | None) -> Path:
    candidate = (
        REPO_ROOT
        if selected is None
        else Path(selected).expanduser().resolve()
    )
    if candidate != REPO_ROOT or not SRC_ROOT.is_dir():
        raise ValueError(_json(_root_error(selected)))
    return candidate


def _self_check(root: Path) -> dict[str, object]:
    from go2arx5_vla.evaluation.h200_adapters import registered_adapter_ids

    scripts_spec = importlib.util.find_spec("scripts")
    present = [
        name
        for name in ("torch", "gr00t", "transformers", "safetensors")
        if name in sys.modules
    ]
    return {
        "status": "PASS",
        "repository_root": str(root),
        "src_root": str(SRC_ROOT.resolve()),
        "import_roots": {
            "repository": str(root) in sys.path,
            "src": str(SRC_ROOT.resolve()) in sys.path,
        },
        "scripts_namespace": bool(
            scripts_spec is not None and scripts_spec.submodule_search_locations is not None
        ),
        "registered_adapters": list(registered_adapter_ids()),
        "admission_boundary": "REACHED_FAIL_CLOSED",
        "model_modules_present": present,
        "torch_imported": "torch" in present,
        "gr00t_imported": "gr00t" in present,
        "transformers_imported": "transformers" in present,
        "cuda_entered": False,
        "model_construction": False,
        "model_forward": False,
        "training": False,
        "cwd": str(Path.cwd()),
        "pythonpath": os.environ.get("PYTHONPATH", ""),
    }


def _recover_failure_provenance(args: argparse.Namespace, exc: Exception) -> dict[str, object]:
    """Recover the engine-owned failure state for truthful CLI reporting."""

    provenance = getattr(exc, "provenance", None)
    if isinstance(provenance, dict):
        return dict(provenance)
    candidates: list[Path] = []
    if args.output_dir is not None:
        candidates.append(args.output_dir.expanduser().resolve() / "engine_failure_seal.json")
    if args.manifest is not None:
        try:
            manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
            root_value = manifest.get("output", {}).get("root")
            if root_value:
                output = Path(str(root_value)).expanduser()
                if not output.is_absolute():
                    output = args.manifest.expanduser().resolve().parent / output
                candidates.append(output.resolve() / "engine_failure_seal.json")
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
    for candidate in candidates:
        try:
            value = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and value.get("schema"):
            return value
    return {
        "phase": "UNKNOWN",
        "model_execution_started": "UNKNOWN",
        "recovery": "engine failure seal unavailable",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=str)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--runtime-config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--raw-staging-dir", type=Path)
    args = parser.parse_args()

    try:
        root = _validate_root(args.repo_root)
    except ValueError as exc:
        try:
            payload = json.loads(str(exc))
        except json.JSONDecodeError:
            payload = {"status": "BLOCKED", "error": str(exc)}
        print(_json(payload), file=sys.stderr)
        return 2

    if args.self_check:
        print(_json(_self_check(root)))
        return 0

    if args.manifest is None or args.runtime_config is None:
        parser.error("execution mode requires --manifest and --runtime-config")

    try:
        from go2arx5_vla.evaluation.h200_engine import run_h200_inference_engine

        summary = run_h200_inference_engine(
            args.manifest,
            args.runtime_config,
            repo_root=root,
            output_dir=args.output_dir,
            raw_staging_dir=args.raw_staging_dir,
        )
    except Exception as exc:
        provenance = _recover_failure_provenance(args, exc)
        print(
            _json(
                {
                    "status": "BLOCKED",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "model_execution_started": provenance.get(
                        "model_execution_started", "UNKNOWN"
                    ),
                    "failure_provenance": provenance,
                }
            ),
            file=sys.stderr,
        )
        return 2
    print(_json(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
