#!/usr/bin/env python3
"""Run the bounded Stage28 recovery evaluation against existing artifacts only.

This entry point exposes no training mode.  It binds the evaluation to
PI_GATE_STATE_0008, the accepted H2 checkpoint tree/seal, the historical
training provenance, and the exact hash of this launcher before delegating to
the Stage28 fixed-probe evaluator.  The evaluator writes only result metadata;
it never writes into either checkpoint directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import stage28_h2_fresh_base_2000_fixed_warmup10 as core


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--expected-project-sha", required=True)
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--stage10-root", type=Path, required=True)
    parser.add_argument("--groot-source", type=Path, default=None)
    parser.add_argument("--modality-config", type=Path, required=True)
    parser.add_argument("--cosmos-cache-root", type=Path, default=None)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--stage26-hash-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--authorization-token", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--expected-h2-checkpoint-tree-sha256", required=True)
    parser.add_argument("--expected-h2-checkpoint-seal-sha256", required=True)
    parser.add_argument("--training-source-sha", required=True)
    parser.add_argument("--training-runner-sha256", required=True)
    parser.add_argument("--expected-evaluation-runner-sha256", required=True)
    parser.add_argument("--expected-evaluation-launcher-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.groot_source is None:
        args.groot_source = args.stage10_root / "src/isaac-gr00t"
    if args.cosmos_cache_root is None:
        args.cosmos_cache_root = (
            args.stage10_root
            / "cache/huggingface/hub/models--nvidia--Cosmos-Reason2-2B"
        )

    try:
        launcher_sha256 = core._sha256_file(Path(__file__).resolve())
        if args.expected_evaluation_launcher_sha256 != launcher_sha256:
            raise core.Stage28Error(
                "evaluation-only launcher SHA mismatch: "
                f"{launcher_sha256} != {args.expected_evaluation_launcher_sha256}"
            )
        args.evaluation_launcher_sha256 = launcher_sha256
        result = core.evaluate_fixed_probes(args)
        print(json.dumps(core._jsonable(result), indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {
            "schema": core.STAGE28_SCHEMA,
            "status": "STOP_OR_BLOCKED",
            "error_type": type(exc).__name__,
            "error_summary": str(exc).splitlines()[0][:500],
            "evaluation_only": True,
            "training": False,
            "backward": False,
            "optimizer_step": False,
            "scheduler_step": False,
            "resume": False,
            "checkpoint_write": False,
            "scientific_contract_changed": False,
        }
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
