"""Reconcile SOURCE optimizer visits and restore a sealed endpoint on Corelab."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from m38_gpu_preflight_v1 import read, require, sha
from v3_main_place_learn import endpoint_files
from v61_source_bank import BATCH, ENDPOINTS, PLAN, SEED
from v61_source_course import BANK_RECEIPT, MAIN, MICRO, R4, read_bank
from v62_source_admission import V62


VISITS = {**BATCH, "auxiliary": 4}
SOURCE_VISITS = dict(dual_old=40, dual_new=40, single=16)
GROUPS = {"shared", *BATCH}


def ledger(run, first, end, banks, slots, rates):
    path = run / "UPDATES.jsonl"
    rows = path.read_text().splitlines()
    require(len(rows) == end - first, "The optimizer ledger has an incomplete interval")
    seconds = []
    for index, line in enumerate(rows, first):
        row = json.loads(line)
        components = row["components"]
        require(row["step"] == index + 1 and row["lr"] == rates[index]
                and math.isfinite(row["global_gradient_norm"]) and row["global_gradient_norm"] > 0
                and math.isfinite(row["seconds"]) and row["seconds"] > 0
                and {name: part["visits"] for name, part in components.items()} == VISITS
                and all(math.isfinite(part["loss"]) and math.isfinite(part["gradient_l2"])
                        for part in components.values()), "Invalid actual SOURCE update")
        require(all(components[name]["visit_sha256"] == hashlib.sha256(bank[index].tobytes()).hexdigest()
                    for name, bank in banks.items())
                and components["auxiliary"]["slots"] == slots[index].tolist()
                and row["source_visits_from_actual_indices"] == SOURCE_VISITS,
                "Actual sample indices, source membership, or auxiliary slots changed")
        seconds.append(row["seconds"])
    return seconds


def restore(endpoint, step):
    import jax
    import orbax.checkpoint as ocp

    require(jax.default_backend() == "cpu", "The independent restore runs on Corelab CPU")
    with ocp.PyTreeCheckpointer() as checkpointer:
        metadata = checkpointer.metadata(endpoint).tree
        args = jax.tree.map(lambda _: ocp.RestoreArgs(restore_type=np.ndarray), metadata)
        state = checkpointer.restore(endpoint, args=ocp.args.PyTreeRestore(
            item=metadata, restore_args=args, transforms={}))
    require(set(state) == {"parameters", "optimizer", "step", "rng"} and int(state["step"]) == step
            and set(state["parameters"]) == set(state["optimizer"]) == GROUPS
            and np.array_equal(state["rng"], np.asarray(jax.random.PRNGKey(SEED))),
            "SOURCE checkpoint lost its step, RNG, or a trainable partition")
    require(all(np.isfinite(leaf).all() for part in ("parameters", "optimizer") for leaf in jax.tree.leaves(state[part])),
            "SOURCE checkpoint contains nonfinite parameters or optimizer state")
    counters = {}
    for name, part in state["optimizer"].items():
        counters[name] = [int(x) for x in jax.tree.leaves(part)
                          if np.asarray(x).ndim == 0 and np.issubdtype(np.asarray(x).dtype, np.integer)]
    require(all(count == [step] for count in counters.values()), "Restored Adam counters disagree with the actual update clock")
    return dict(rng_restored=True, complete_training_endpoint=True,
                adam_counters={name: count[0] for name, count in counters.items()},
                parameter_leaves=len(jax.tree.leaves(state["parameters"])))


def audit(runs, bank, end):
    record, banks, slots, rates = read_bank(bank)
    probe = end == 6
    require((probe and len(runs) == 1) or end in ENDPOINTS, "Unknown SOURCE milestone")
    expected_course = dict(plan=PLAN, bank_receipt_sha256=BANK_RECEIPT, initial_r4_receipt_sha256=R4,
                           effective_batch=BATCH, auxiliary=4, micro=MICRO, source_rng_seed=SEED)
    first, timings, input_ledgers = 0, [], []
    for run in runs:
        contract_file, terminal_file = (run / name for name in ("RUN_CONTRACT.json", "TERMINAL.json"))
        contract, terminal = read(contract_file), read(terminal_file)
        step = terminal["updates"]
        require(contract["course"] == terminal["course"] == expected_course and contract["start"] == first
                and first < step <= end and contract["science_plan"] == terminal["science_plan"] == V62
                and terminal["interval_updates"] == step - first and terminal["contract_sha256"] == sha(contract_file)
                and contract["input_sha256"] == record["manifest_sha256"]
                and contract["ancestry"]["initial_r4_receipt_sha256"] == R4
                and contract["ancestry"]["parent_sha256"] == MAIN
                and terminal["optimized_source_visits"] == {k: (step - first) * count for k, count in SOURCE_VISITS.items()},
                "The effects do not belong to the registered contiguous SOURCE training course")
        statuses = {"QUALIFICATION_PASS"} if probe else (
            {"FORMAL_SOURCE_BOUNDARY", "PROVISIONAL_SOURCE_BOUNDARY"} if step == end else
            {"SAVED_AT_FINITE_BOUNDARY", "FORMAL_SOURCE_BOUNDARY", "PROVISIONAL_SOURCE_BOUNDARY"})
        require(terminal["status"] in statuses and ((contract["mode"] == "probe") == probe),
                "Checkpoint does not represent a valid closed SOURCE interval")
        if probe:
            require(terminal["restore"]["prediction_equal"], "Hopper recovery did not pass")
        endpoint = run / f"STEP{step:04d}"
        receipt = terminal["checkpoint"]
        require(receipt["step"] == step and Path(receipt["path"]).name == endpoint.name
                and receipt["files"] == endpoint_files(endpoint), "Transferred checkpoint differs from the training receipt")
        timings.extend(ledger(run, first, step, banks, slots, rates))
        input_ledgers.append(dict(start=first, end=step, terminal_sha256=sha(terminal_file),
                                 optimizer_ledger_sha256=sha(run / "UPDATES.jsonl")))
        first = step
    require(first == end, "A registered cumulative milestone is incomplete")
    summary = restore(endpoint, end)
    warm = timings[2:] or timings
    return dict(step=end, endpoint=str(endpoint.resolve()), formal=False, science_plan=V62,
                parent_sha256=MAIN, initial_r4_receipt_sha256=R4, bank_receipt_sha256=BANK_RECEIPT,
                training_terminal_sha256=sha(terminal_file), receipt_sha256=sha(terminal_file),
                contract_sha256=sha(contract_file), provisional=terminal["provisional"], inputs=input_ledgers,
                actual_visits={k: end * v for k, v in VISITS.items()},
                actual_source_visits={k: end * v for k, v in SOURCE_VISITS.items()},
                warm_p50_s=float(np.median(warm)), warm_p95_s=float(np.percentile(warm, 95)), **summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bank", type=Path)
    parser.add_argument("end", type=int, choices=(6, *ENDPOINTS))
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--provider", type=Path)
    args = parser.parse_args()
    from scripts.pi0292.durable import write

    require(bool(args.reference) == bool(args.provider) and (args.end != 6 or args.provider is None),
            "Provide both export paths for a formal course endpoint")
    result = audit(args.runs, args.bank, args.end)
    if args.provider:
        from v4_long_export import export

        result["provider_manifest_sha256"] = export(result, args.reference, args.provider, diagnostic=True)
    write(args.output, result)
