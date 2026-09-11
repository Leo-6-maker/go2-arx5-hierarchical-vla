#!/usr/bin/env python3
"""Build or validate the static Stage45E N2 TRAIN24 evaluation surface.

This entrypoint is intentionally model-free.  ``--plan-only`` is safe from an
arbitrary CWD with an empty PYTHONPATH.  The non-plan path only validates a
future Corelab science commit and N2 artifact; it never constructs a model or
executes an evaluator.
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
    parser.add_argument("--accepted-full-stats", type=Path, default=support.DEFAULT_FULL_STATS)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.plan_only:
        result = support.build_static_plan()
        result["entrypoint"] = str(Path(__file__).resolve())
        result["repo_root_resolution"] = str(REPO_ROOT)
    else:
        if not args.science_head or args.n2_stats is None:
            parser.error("non-plan validation requires --science-head and --n2-stats")
        commit_audit = support.audit_corelab_science_commit(args.repo.resolve(), args.science_head)
        stats_audit = support.validate_n2_statistics_artifact(
            args.n2_stats.resolve(),
            args.accepted_full_stats.resolve(),
            expected_science_head=args.science_head,
        )
        result = support.static_evaluation_template()
        result["status"] = "PASS_STATIC_N2_EVAL_SURFACE"
        result["science_binding"]["observed_science_head"] = args.science_head
        result["science_binding"]["corelab_commit_audit"] = commit_audit
        result["science_binding"]["n2_stats_audit"] = stats_audit
        result["execution_boundary"]["evaluation_authorization_required"] = True
        result["entrypoint"] = str(Path(__file__).resolve())
        result["repo_root_resolution"] = str(REPO_ROOT)

    if args.output is not None:
        support.write_json(args.output.resolve(), result)
    print(json.dumps(support.jsonable(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
