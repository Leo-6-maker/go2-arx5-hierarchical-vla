"""Run the unchanged 24-window G7 feedback evaluation for a sealed A250."""

import argparse
from pathlib import Path

from m38_gpu_preflight_v1 import pin, require, sha
from v62_source_native import FEEDBACK_ROSTER, V65, checked_manifest, reference, source_role
from v64_adapt_endpoint import END, PLAN


def evaluate(root, worker, roster, endpoint, output, *, execute, recovery=False, step=None):
    import v4_long_watch

    pin(roster, FEEDBACK_ROSTER)
    plan, default_step = (V65, 1000) if recovery else (PLAN, END)
    end = default_step if step is None else step
    require(end == default_step or (recovery and end == 3000), "Wrong fixed offline evaluation endpoint")
    package = endpoint / "PROVIDER"
    audit = endpoint / "CPU-ENDPOINT-READBACK.json"
    endpoint_proof = reference(audit)

    def model_manifest(folder, original, step, supplied_audit):
        require(folder == package and step == end and supplied_audit == audit, "Wrong adaptation evaluation provider")
        provider = dict(path=str(folder), sha256=sha(folder / "CHECKPOINT.json"))
        if recovery and end == 3000:
            from v65_recovery_3k_endpoint import check_offline_provider

            return check_offline_provider(provider, original, endpoint_proof)
        registration = dict(v4_model_role=source_role(end, plan), source_step=end, v62_plan=plan,
                            source_audit=endpoint_proof)
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
    parser.add_argument("--step", type=int)
    evaluate(**vars(parser.parse_args()))
