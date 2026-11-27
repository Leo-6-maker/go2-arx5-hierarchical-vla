"""Reconcile the fixed Recovery pair and seal only complete actual Teacher suffixes."""
import argparse
import json
from pathlib import Path

from v64_recovery_launcher import CUDA_NOISE, SCOPE, checked_ref, ref
from v64_recovery_noise import TRAIN_ROSTER
from v64_recovery_roster import PLAN, QUERIES, ROSTER, TICKS, fixed, sha
from v64_recovery_seal import inspect


def read(path):
    return json.loads(path.read_bytes())


def close(root, output):
    spec = read(root / "EXECUTION_SPEC.json")
    registered = fixed(root / "COMPARISON.json", spec["comparison_registration"]["sha256"])
    roster = checked_ref(registered["v64_recovery_train"], TRAIN_ROSTER)["cases"]
    if (registered["plan"] != PLAN or spec["registry_scope"] != SCOPE
            or registered["gpu_noise_sha256"] != CUDA_NOISE or not registered["zero_training"]
            or tuple(row["id"] for row in registered["schedule"]) != tuple(row[0] for row in ROSTER)
            or not all(read(root / role / "control/LEASE.json")["status"] == "CLOSED" for role in (".", "worker"))):
        raise ValueError("The exact frozen pair and both owner leases must be closed")
    outcomes = read(root / "PAIR_CLOSE.private.json")["cases"]
    index = {row["id"]: row for row in outcomes}
    if len(index) != len(outcomes) or not set(index) <= {row["id"] for row in roster}:
        raise ValueError("Unknown or duplicated Recovery outcome")
    results = []
    output.mkdir(mode=0o700)
    for row in roster:
        name = row["id"]
        case = root / name
        if name not in index:
            # A started case with absent terminal remains unknown and retains evidence.
            results.append(dict(id=name, state="UNRECONCILED_START" if case.exists() else "NOT_STARTED",
                                recovery_admitted=False))
            continue
        terminal = case / "RESULT.private.json"
        result = fixed(terminal, index[name]["result_sha256"])
        detail = result["detail"] or {}
        trace = result["trace_audit"] or {}
        ticks, forwards = result["actual_native_integrations"], result["model_generations"]
        lease = read(case / "control/LEASE.json")
        bound = fixed(case / "EXECUTION_SPEC.json", lease["spec_sha256"])
        if (lease["status"] != "CLOSED" or not 0 <= ticks <= TICKS or not 0 <= forwards <= QUERIES
                or result["WBC_changes"] or result["optimizer_updates"] or result["physical_requests"] != 1
                or bound["collection_split"] != "TRAIN" or bound["evaluation_split"] != "TRAIN_RECOVERY"
                or bound["model_path"] != row["scene_path"] or bound["task_instruction_id"] != row["task"]
                or bound["mobile_condition"]["reset"] != row["reset"]):
            raise ValueError("A terminal does not match the original registered scope and clock")
        stage = case / "native-01"
        intent = stage / "RECOVERY-HANDOFF-INTENT.private.json"
        capture = stage / "LIVE_DATA/MANIFEST.private.json"
        failed_handoff = stage / "RECOVERY-HANDOFF-REJECTED.private.json"
        value = dict(id=name, state="CLOSED", ticks=ticks, forwards=forwards, result=ref(terminal),
                     safety_stop=result["unsafe"], shutdown_clean=result["shutdown_clean"],
                     technical_failure=result["technical_failure"], native_stop=result["physics_stops"],
                     after_first_stop_ticks=trace.get("integrations_after_first_STOP"),
                     teacher_intent=ref(intent) if intent.is_file() else None,
                     failed_handoff=ref(failed_handoff) if failed_handoff.is_file() else None,
                     raw_capture=ref(capture) if capture.is_file() else None,
                     student_task_success=bool(detail.get("task_success")),
                     teacher_task_success=bool(detail.get("recovery_task_success")),
                     confirmed_teacher_atoms=detail.get("consumed_teacher_atoms"),
                     recovery_admitted=False)
        if result["unsafe"] and value["after_first_stop_ticks"] != 0:
            value["state"] = "STOP_RECONCILIATION_REQUIRED"
        if value["teacher_task_success"] and value["state"] == "CLOSED":
            try:
                out = output / name
                inspect(case, out)
                value["recovery_admitted"] = True
                value["recovery_label"] = ref(out / "ACTUAL.json")
            except ValueError as error:
                value["seal_rejection"] = str(error)
        results.append(value)
    worker_terminal = root / "worker/RESULT.private.json"
    worker = read(worker_terminal)
    total_forwards = sum(row.get("forwards", 0) for row in results)
    worker_forwards = worker.get("completed_generations")
    actual = dict(plan=PLAN, comparison=ref(root / "COMPARISON.json"), pair_close=ref(root / "PAIR_CLOSE.private.json"),
                  worker=ref(worker_terminal), denominator=len(roster),
                  terminal_cases=len(index), cases=results, native_ticks=sum(row.get("ticks", 0) for row in results),
                  actual_forwards=total_forwards, worker_actual_forwards=worker_forwards,
                  forward_ledger_reconciled=total_forwards == worker_forwards,
                  train_recovery_count=sum(row["recovery_admitted"] for row in results),
                  complete_reconciliation=total_forwards == worker_forwards
                  and all(row["state"] in ("CLOSED", "NOT_STARTED") for row in results), actual_training_updates=0)
    with (output / "RESULT.json").open("x") as stream:
        json.dump(actual, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "output"):
        parser.add_argument(name, type=Path)
    close(**vars(parser.parse_args()))
