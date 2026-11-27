"""Audit the complete A250 endpoint against actual A74 visits on Corelab CPU."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha
from scripts.pi0292.durable import write
from v3_main_place_learn import endpoint_files
from v61_source_course import MAIN, R4
from v62_source_endpoint import restore
from v63_adapt_course import BATCH, MICRO, SEED, TARGET
from v63_adapt_data import PLAN


BANK = "1ce0abe57d403f93fa25f71ad8268389c80a4cfbb8bbaa5b499e3db3820b0fa0"
END = 250
VISITS = dict(BATCH, auxiliary=4, robodojo_old=16, robodojo_expanded=16)
GO2 = {group: counts[0] for group, counts in TARGET.items()}


def audit(run, bank, qualification_sha):
    record = read(pin(bank / "ACTUAL.json", BANK))
    contract_file = run / "RUN_CONTRACT.json"
    terminal_file = run / "TERMINAL.json"
    contract, terminal = read(contract_file), read(terminal_file)
    course = dict(plan=PLAN, r4_parent=R4, course_seed=SEED, bank_sha256=BANK,
                  data_sha256=record["added_manifest_sha256"], batches=BATCH, micro=MICRO, auxiliary=4)
    require(contract["course"] == terminal["course"] == course
            and contract["start"] == 0 and contract["end"] == END
            and contract["optimizer"] == "NEW_ADAM_FROM_R4" and contract["rng"] == "INDEPENDENT_ADAPT_SEED"
            and contract["lr_schedule"] == "WARM100_COSINE3000"
            and contract["ancestry"]["initial_r4_receipt_sha256"] == R4
            and contract["ancestry"]["parent_sha256"] == MAIN
            and contract["probe_receipt"] == qualification_sha
            and terminal["engineering_updates"] == 0 and terminal["contract_sha256"] == sha(contract_file)
            and terminal["status"] == "FORMAL_ADAPT_A250_BOUNDARY"
            and terminal["updates"] == terminal["bank_cursor"] == END,
            "Expected the full fresh-R4 A250 course with separately accounted engineering qualification")
    ledger = run / "UPDATES.jsonl"
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    require(len(rows) == END, "A250 optimizer ledger is incomplete")
    phases, seconds = Counter(), []
    with np.load(pin(bank / "BANK.npz", record["bank_file_sha256"]), allow_pickle=False) as indices:
        banks = {name: indices[name] for name in BATCH}
        slots, rates = indices["auxiliary_slots"], indices["rates"]
        for index, row in enumerate(rows):
            components = row["components"]
            require(row["step"] == index + 1 and row["lr"] == rates[index]
                    and math.isfinite(row["seconds"]) and row["seconds"] > 0
                    and math.isfinite(row["global_gradient_norm"]) and row["global_gradient_norm"] > 0
                    and {name: value["visits"] for name, value in components.items()} == dict(BATCH, auxiliary=4)
                    and all(math.isfinite(value["loss"]) and math.isfinite(value["gradient_l2"])
                            and value["gradient_l2"] > 0 for value in components.values()),
                    f"Invalid actual ADAPT update {index + 1}")
            require(all(components[name]["visit_sha256"] == hashlib.sha256(values[index].tobytes()).hexdigest()
                        for name, values in banks.items())
                    and components["auxiliary"]["slots"] == slots[index].tolist()
                    and row["actual_go2_population"] == GO2
                    and row["actual_go2_aux_population"] == dict(base=2, legacy=1, newest=1)
                    and row["actual_robodojo_old"] == row["actual_robodojo_expanded"] == 16
                    and sum(row["actual_go2_phase"].values()) == BATCH["target"],
                    f"Actual visit, phase or auxiliary membership changed at update {index + 1}")
            phases.update(row["actual_go2_phase"])
            seconds.append(row["seconds"])
        unique_windows = int(np.unique(banks["target"][:END]).size)
    visits = {name: count * END for name, count in VISITS.items()}
    groups = {name: count * END for name, count in GO2.items()}
    require(terminal["optimizer_visits"] == visits and terminal["go2_population"] == groups
            and terminal["go2_phase"] == phases == record["first250_target_phase"]
            and unique_windows == record["first250_target_unique_windows"],
            "A250 terminal or exposure totals differ from the actual update bank")
    endpoint = run / f"STEP{END:04d}"
    receipt = terminal["checkpoint"]
    require(receipt["step"] == END and Path(receipt["path"]).name == endpoint.name
            and receipt["files"] == endpoint_files(endpoint),
            "Transferred A250 checkpoint differs from the sealed writer receipt")
    restored = restore(endpoint, END, groups={"shared", *BATCH}, seed=SEED)
    return dict(step=END, endpoint=str(endpoint.resolve()), formal=True, science_plan=PLAN,
                parent_sha256=MAIN, initial_r4_receipt_sha256=R4, bank_receipt_sha256=BANK,
                training_terminal_sha256=sha(terminal_file), receipt_sha256=sha(terminal_file),
                contract_sha256=sha(contract_file), optimizer_ledger_sha256=sha(ledger),
                engineering_qualification_sha256=qualification_sha, actual_visits=visits,
                actual_go2_population=groups, actual_go2_phase=dict(phases), unique_go2_windows=unique_windows,
                warm_p50_s=float(np.median(seconds[2:])), warm_p95_s=float(np.percentile(seconds[2:], 95)),
                **restored)


if __name__ == "__main__":
    from v4_long_export import export

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "bank", "qualification", "output", "reference", "provider"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "Preserve the independently sealed A250 receipt")
    proof = audit(args.run, args.bank, sha(args.qualification))
    proof["provider_manifest_sha256"] = export(proof, args.reference, args.provider, diagnostic=True)
    write(args.output, proof)
