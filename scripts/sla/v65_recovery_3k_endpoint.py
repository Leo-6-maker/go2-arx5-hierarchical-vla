"""Independently restore the delivered V6.5 3K endpoint on Corelab CPU."""

import argparse
from pathlib import Path

from m38_gpu_preflight_v1 import pin, read, require, sha
from scripts.pi0292.durable import write
from v3_main_place_learn import endpoint_files
from v61_source_course import MAIN, R4
from v62_source_endpoint import restore
from v65_recovery_chain import ledger
from v65_recovery_course import BATCH, PLAN, SEED
from v65_recovery_endpoint import VISITS, export_go2_endpoint
from v65_recovery_extension import BASE_SEAL, BASE_TERMINAL, COURSE, END


RANGES = ((0, 1000), (1000, 1800), (1800, 2200), (2200, 2500), (2500, 2800), (2800, END))
BANK = "f412a728fdc3fa1b8bf677b6b6b2b9aa107a704e320c563a182dda2b1e0c6a3e"
ORIGINAL_CPU_PROOF = "c422f63baca6b6641c4bc4c0cc5309dd1c5d89202b1c45c4ada5180d9b0f6843"
FINAL_PBS_JOB = "737978.hopper-m-02"


def audit(capsule, hopper_sha, original):
    source = read(pin(capsule / "FINAL-HOPPER-READBACK.json", hopper_sha))
    visits = {role: count * END for role, count in VISITS.items()}
    chain = source["timeline"]
    require(source["pbs"] == {"job": FINAL_PBS_JOB, "state": "F", "exit_status": 0}
            and source["final_3k"] and source["step"] == END
            and source["bank_sha256"] == BANK and source["unique_visits"] == visits
            and source["paid_updates"] == 3100 and source["paid_unsealed_updates"] == 100
            and source["original_1k_fm_visits"] == 128_000
            and tuple((row["start"], row["end"]) for row in chain) == RANGES
            and chain[0]["seal"] == BASE_SEAL and chain[0]["terminal"] == BASE_TERMINAL,
            "Require the completed Hopper allocation and independently read continuation lineage")
    run = capsule / "run"
    endpoint = run / f"STEP{END:04d}"
    terminal = read(pin(run / "TERMINAL.json", chain[-1]["terminal"]))
    seal = read(pin(run / f"STEP{END:04d}_SEALED.json", chain[-1]["seal"]))
    contract = read(pin(run / "RUN_CONTRACT.json", terminal["contract_sha256"]))
    binding = read(pin(capsule / "BINDING.json", contract["binding_sha256"]))
    resume = read(run / "RESUME.json")
    bank = pin(capsule / "BANK-3K/ACTUAL.json", BANK)
    start = RANGES[-1][0]
    slots = ("shared", *BATCH)
    prior = binding["previous_boundary"]
    require(read(capsule / "JOB.json")["job"] == FINAL_PBS_JOB
            and terminal["status"] == "FORMAL_V65_3K_COMPLETE" and terminal["formal_updates"] == END
            and terminal["start"] == contract["start"] == resume["step"] == prior["step"] == start
            and terminal["end"] == contract["end"] == contract["segment_end"] == END
            and terminal["resumed_from_formal_1k"] and terminal["segment_updates"] == END - start
            and terminal["engineering_updates"] == 0 and not contract["new_adam_from_r4"]
            and contract["plan"] == binding["plan"] == PLAN and contract["seed"] == SEED
            and contract["course"] == binding["course"] == COURSE
            and contract["ancestry"]["parent_sha256"] == MAIN
            and contract["ancestry"]["initial_r4_receipt_sha256"] == R4
            and contract["previous_seal"] == resume["parent_seal"] == prior["seal"] == chain[-2]["seal"]
            and contract["previous_terminal"] == prior["terminal"] == chain[-2]["terminal"]
            and resume["rng_restored"] and resume["bank_sha256"] == binding["bank"]["sha256"] == BANK
            and resume["adam_counters"] == dict.fromkeys(slots, start),
            "The final segment retained the exact parent, data bank, optimizer and RNG")
    summary = ledger(run, bank, start, END, settle=True)
    require(terminal["optimizer_visits"] == summary["actual_visits"] and terminal["checkpoint"] == seal
            and seal["step"] == seal["bank_cursor"] == END and seal["adam_counts"] == dict.fromkeys(slots, END)
            and len(seal["files"]) == source["file_count"] and seal["files"] == endpoint_files(endpoint),
            "Received bytes, sample visits or optimizer state differ from the Hopper checkpoint seal")
    proof = read(pin(original / "CPU-ENDPOINT-READBACK.json", ORIGINAL_CPU_PROOF))
    previous = read(pin(original / "run/STEP1000_SEALED.json", BASE_SEAL))
    require(proof["initial_r4_receipt_sha256"] == R4 and proof["endpoint_sha256"] == BASE_SEAL
            and proof["frozen_lora_sha256"] == previous["lora_sha256"] == seal["lora_sha256"]
            and previous["frozen_weight_sha256"] == seal["frozen_weight_sha256"],
            "The continuation must inherit the independently observed frozen 1K model")
    state = restore(endpoint, END, groups=set(slots), seed=SEED)
    require(state["adam_counters"] == seal["adam_counts"], "Corelab cold-restored Adam counts disagree")
    return dict(step=END, plan=PLAN, endpoint_sha256=chain[-1]["seal"], contract_sha256=terminal["contract_sha256"],
                training_terminal_sha256=chain[-1]["terminal"], initial_r4_receipt_sha256=R4,
                frozen_lora_sha256=seal["lora_sha256"], bank_receipt_sha256=sha(bank),
                hopper_readback_sha256=hopper_sha, original_corelab_readback_sha256=ORIGINAL_CPU_PROOF,
                formal_scientific_endpoint=True, actual_visits=visits, fm_visits=128 * END, go2_visits=96 * END,
                paid_unsealed_updates=source["paid_unsealed_updates"], final_segment=summary,
                native_admission=False, hardware_allowed=False, physics_ticks=0, **state)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("capsule", "original", "output"):
        cli.add_argument(name, type=Path)
    cli.add_argument("--hopper-sha", required=True)
    cli.add_argument("--r4-provider", type=Path)
    cli.add_argument("--provider", type=Path)
    args = cli.parse_args()
    require(not args.output.exists() and bool(args.r4_provider) == bool(args.provider), "Preserve original readbacks")
    proof = audit(args.capsule, args.hopper_sha, args.original)
    if args.provider:
        proof["provider_manifest_sha256"] = export_go2_endpoint(
            args.capsule / f"run/STEP{END:04d}", args.r4_provider, args.provider, proof, "V65_RECOVERY_ADAPT_3K")
    write(args.output, proof)
