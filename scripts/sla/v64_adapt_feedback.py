"""Run the unchanged 24-window G7 feedback evaluation for a sealed A250."""

import argparse
from pathlib import Path

from m38_gpu_preflight_v1 import pin, require, sha
from v62_source_native import FEEDBACK_ROSTER, V65, checked_manifest, reference, source_role
from v64_adapt_endpoint import END, PLAN


def evaluate(root, worker, roster, endpoint, output, *, execute, recovery=False):
    import v4_long_watch

    pin(roster, FEEDBACK_ROSTER)
    plan, end = (V65, 1000) if recovery else (PLAN, END)
    package = endpoint / "PROVIDER"
    audit = endpoint / "CPU-ENDPOINT-READBACK.json"
    registration = dict(v4_model_role=source_role(end, plan), source_step=end, v62_plan=plan,
                        source_audit=reference(audit))

    def model_manifest(folder, original, step, supplied_audit):
        require(folder == package and step == end and supplied_audit == audit, "Wrong adaptation evaluation provider")
        provider = dict(path=str(folder), sha256=sha(folder / "CHECKPOINT.json"))
        return checked_manifest(provider, original, registration)

    original = v4_long_watch.candidate
    v4_long_watch.candidate = model_manifest
    try:
        v4_long_watch.watch(root, worker, roster, package, end, audit, output, execute=execute,
                            retained_roster_sha=FEEDBACK_ROSTER)
    finally:
        v4_long_watch.candidate = original


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "worker", "roster", "endpoint", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--recovery", action="store_true")
    evaluate(**vars(parser.parse_args()))
