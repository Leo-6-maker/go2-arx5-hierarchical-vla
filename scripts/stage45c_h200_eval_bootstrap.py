#!/usr/bin/env python3
"""Stage45C execution bootstrap for the model-owned Cosmos processor path.

GR00T N1.7 keeps the canonical ``nvidia/Cosmos-Reason2-2B`` model name so its
backbone dispatch remains unchanged.  During ``Gr00tN1d7`` construction it also
creates a ``Gr00tN1d7DataCollator``; that collator independently calls
``processing_gr00t_n1d7.build_processor`` and a repo-id based Qwen3 processor
may perform Hugging Face Hub metadata lookup even in offline mode.

Historical Stage38 already qualified the narrow remedy: rewrite only that
processor-side repo-id to the exact sealed local Cosmos snapshot while leaving
model config, backbone dispatch, weights and loading kwargs unchanged.  This
bootstrap installs that process-local shim and then executes the unchanged
Stage45C runtime adapter in the same Python process.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import runpy
import sys
from typing import Any


COSMOS_REPO_ID = "nvidia/Cosmos-Reason2-2B"
COSMOS_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"


def _regular_directory(path: Path, label: str) -> Path:
    path = Path(path)
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError(f"{label} is unavailable or unsafe: {path}")
    return path.resolve()


def _load_execution_paths(runtime_binding: Path) -> tuple[Path, Path]:
    runtime_binding = Path(runtime_binding)
    if runtime_binding.is_symlink() or not runtime_binding.is_file():
        raise RuntimeError(f"runtime binding is unavailable or unsafe: {runtime_binding}")
    payload = json.loads(runtime_binding.read_text(encoding="utf-8"))
    paths = payload.get("paths")
    if not isinstance(paths, dict):
        raise RuntimeError("runtime binding has no paths mapping")
    groot_source = paths.get("groot_source")
    cosmos_root = paths.get("cosmos_root")
    if not isinstance(groot_source, str) or not isinstance(cosmos_root, str):
        raise RuntimeError("runtime binding lacks groot_source/cosmos_root")
    groot_source_path = _regular_directory(Path(groot_source), "GR00T source")
    cosmos_root_path = _regular_directory(Path(cosmos_root), "Cosmos cache root")
    snapshot = _regular_directory(
        cosmos_root_path / "snapshots" / COSMOS_REVISION,
        "pinned Cosmos snapshot",
    )
    refs_main = cosmos_root_path / "refs" / "main"
    if refs_main.is_symlink() or not refs_main.is_file():
        raise RuntimeError(f"Cosmos refs/main unavailable or unsafe: {refs_main}")
    if refs_main.read_text(encoding="utf-8").strip() != COSMOS_REVISION:
        raise RuntimeError("Cosmos refs/main does not match the accepted revision")
    return groot_source_path, snapshot


def _bind_model_owned_processor(snapshot: Path) -> dict[str, Any]:
    """Install the Stage38-proven processor-only local-snapshot binding."""

    snapshot = _regular_directory(snapshot, "pinned Cosmos snapshot")
    from gr00t.model.gr00t_n1d7 import processing_gr00t_n1d7 as processing

    sentinel_original = "_stage45c_original_build_processor"
    sentinel_snapshot = "_stage45c_bound_processor_snapshot"
    original = getattr(processing, sentinel_original, None)
    already_bound = getattr(processing, sentinel_snapshot, None)

    if original is None:
        original = processing.build_processor
        setattr(processing, sentinel_original, original)
        setattr(processing, sentinel_snapshot, str(snapshot))

        def build_local_processor(model_name: str, loading_kwargs: dict) -> Any:
            resolved_name = str(snapshot) if model_name == COSMOS_REPO_ID else model_name
            return original(resolved_name, loading_kwargs)

        processing.build_processor = build_local_processor
        status = "INSTALLED"
    else:
        if already_bound != str(snapshot):
            raise RuntimeError(
                "existing Stage45C processor binding points to a different snapshot: "
                f"{already_bound!r} != {str(snapshot)!r}"
            )
        status = "ALREADY_BOUND_SAME_SNAPSHOT"

    return {
        "schema": "STAGE45C_MODEL_OWNED_PROCESSOR_OFFLINE_BINDING_V1",
        "status": "PASS",
        "binding_state": status,
        "canonical_repo_id": COSMOS_REPO_ID,
        "processor_resolved_snapshot": str(snapshot),
        "cosmos_revision": COSMOS_REVISION,
        "model_config_repo_id_rewritten": False,
        "backbone_dispatch_changed": False,
        "loading_kwargs_changed": False,
        "network_required": False,
    }


def main(argv: list[str] | None = None) -> int:
    forwarded = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--runtime-binding", type=Path, required=True)
    args, _ = parser.parse_known_args(forwarded)

    groot_source, snapshot = _load_execution_paths(args.runtime_binding)
    if str(groot_source) not in sys.path:
        sys.path.insert(0, str(groot_source))

    binding = _bind_model_owned_processor(snapshot)
    print(json.dumps(binding, sort_keys=True), flush=True)

    adapter = Path(__file__).resolve().with_name("stage45c_h200_eval_runtime_adapter.py")
    if adapter.is_symlink() or not adapter.is_file():
        raise RuntimeError(f"Stage45C runtime adapter unavailable or unsafe: {adapter}")

    old_argv = sys.argv
    try:
        sys.argv = [str(adapter), *forwarded]
        runpy.run_path(str(adapter), run_name="__main__")
    finally:
        sys.argv = old_argv
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
