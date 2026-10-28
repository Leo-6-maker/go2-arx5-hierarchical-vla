#!/usr/bin/env python3
"""Run the strengthened raw inventory verifier in an isolated process.

The recovery execution parent must remain free of the frozen model/runtime
import prefixes until artifact admission.  This child is deliberately the
place where the NumPy-backed ``.npy`` array checks run.  It emits exactly one
JSON object on stdout and never shares its ``sys.modules`` with the parent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args(argv)

    repo = Path(__file__).resolve().parents[1]
    scripts = repo / "scripts"
    src = repo / "src"
    for path in (src, scripts):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    try:
        import stage45f_pi0163_validation_e_recovery as recovery

        inventory = recovery.raw_inventory(args.source_root)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(inventory, sort_keys=True, allow_nan=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
