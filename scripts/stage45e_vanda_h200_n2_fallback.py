#!/usr/bin/env python3
"""Fail-closed, pre-model H200 fallback admission for Stage45E N2.

The default plan mode is safe and model-free.  The preflight mode requires the
exact Corelab resource-blocker token and exact Corelab-produced N2 science
commit/artifact, but still never submits PBS or launches H200/model work.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import stage45e_n2_support as support  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--science-head", default=None)
    parser.add_argument("--n2-stats", type=Path, default=None)
    parser.add_argument("--blocker-token", type=Path, default=None)
    parser.add_argument("--accepted-full-stats", type=Path, default=support.DEFAULT_FULL_STATS)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.plan_only:
        result = support.static_fallback_template()
        result["entrypoint"] = str(Path(__file__).resolve())
        result["repo_root_resolution"] = str(REPO_ROOT)
    else:
        if not args.science_head or args.n2_stats is None or args.blocker_token is None:
            parser.error("preflight requires --science-head, --n2-stats, and --blocker-token")
        result = support.validate_h200_fallback_preflight(
            args.repo.resolve(),
            args.science_head,
            args.n2_stats.resolve(),
            args.blocker_token.resolve(),
            args.accepted_full_stats.resolve(),
            runtime_profile={"device": "cuda:0", "offline_mode": True, "single_gpu": True},
        )
        result["entrypoint"] = str(Path(__file__).resolve())
        result["repo_root_resolution"] = str(REPO_ROOT)

    if args.output is not None:
        support.write_json(args.output.resolve(), result)
    print(json.dumps(support.jsonable(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
