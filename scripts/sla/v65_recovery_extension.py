"""Append a 2K continuation to the sealed V6.5 course without changing its 1K prefix."""

import argparse
from collections import Counter
from math import cos, pi
from pathlib import Path

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha
from scripts.pi0292.durable import write
from v61_source_course import read_bank
from v65_recovery_course import BATCH, END as START, PEAK, PLAN, SEED, bind, host_readback, schedule, target_schedule, validate
from v6_dual_course import balanced_source


END = 3000
SAMPLING_SEED = 655302611
COURSE = "V65_RECOVERY_1K_TO_3K"
BASE_SEAL = "f69e63fbb818113b2bae659ea67eedc06935bbb12046e4c42b69adc0b41d2e31"
BASE_TERMINAL = "35cc15ae6e2c00ec8871416378d345b188c7404581f7b1b545b1b3f60912bae5"
BASE_LEDGER = "c68c28f01bb657a1b96365a2b910542554ecdea9f5f909e8e8ed3097b508a473"


def rate(step):
    if step <= START:
        return schedule(step)
    if step <= START+50:
        return PEAK * (.1+.4*(step-START)/50)
    return PEAK * (.1+.2*(1+cos(pi*(step-START-50)/(END-START-50))))


def original(capsule):
    run = capsule / "run"
    terminal = read(pin(run / "TERMINAL.json", BASE_TERMINAL))
    seal = read(pin(run / "STEP1000_SEALED.json", BASE_SEAL))
    contract = read(pin(run / "RUN_CONTRACT.json", terminal["contract_sha256"]))
    binding = read(pin(capsule / "BINDING.json", contract["binding_sha256"]))
    pin(run / "UPDATES.jsonl", BASE_LEDGER)
    require(terminal["status"] == "FORMAL_V65_1K_COMPLETE" and terminal["formal_updates"] == START
            and terminal["checkpoint"] == seal and seal["step"] == seal["bank_cursor"] == START
            and contract["plan"] == binding["plan"] == PLAN and contract["seed"] == SEED,
            "The sealed formal V6.5 1K endpoint is required")
    return binding, seal


def cold_bank(capsule):
    binding, _ = original(capsule)
    config = read(pin(binding["data"]["path"], binding["data"]["sha256"]))
    bank = binding["bank"]
    return config, bank, bind(config, Path(bank["path"]).parent, bank["sha256"])


def pack(capsule, output):
    config, bank, data = cold_bank(capsule)
    datasets, _, prefix, slots, _, demographics = data
    rng = np.random.default_rng(SAMPLING_SEED)
    target, aux = target_schedule(datasets["target"], rng, start=START, end=END)
    dual = datasets["robodojo"]
    old, new = (balanced_source(reader, END-START, 8, rng) for reader in dual.members)
    stats, source, _, _ = read_bank(Path(config["parent_paths"]["bank"]))
    require(len(source["libero"]) >= END and np.array_equal(source["libero"][:START], prefix["libero"])
            and stats["reader_order_sha256"]["libero"] == read(bank["path"])["reader_order"]["libero"],
            "The original ordered LIBERO bank must cover the 3K run")
    banks = dict(target=np.concatenate((prefix["target"], target)),
                 robodojo=np.concatenate((prefix["robodojo"], np.column_stack((old, new + dual.base_length)))),
                 libero=source["libero"][:END])
    slots = np.concatenate((slots, aux))
    rates = np.asarray([rate(step) for step in range(1, END+1)])
    validate(datasets, banks, slots, rates, end=END, rate_schedule=rate)
    output.mkdir(mode=0o700)
    with (output / "BANK.npz").open("xb") as stream:
        np.savez_compressed(stream, **banks, auxiliary_slots=slots, rates=rates)
    actual = output / "ACTUAL.json"
    episodes, phases, _ = demographics
    write(actual, dict(course=COURSE, plan=PLAN, start=START, end=END, rng_seed=SEED, sampling_seed=SAMPLING_SEED,
                       original_bank=bank["sha256"], original_seal=BASE_SEAL, bank_sha256=sha(output / "BANK.npz"),
                       recovery_episodes=dict(Counter(episodes[target[:, 94:]].ravel())),
                       recovery_phases=dict(Counter(phases[target[:, 94:]].ravel())), actual_optimizer_visits=0))
    extension = (datasets, data[1], banks, slots, rates, demographics)
    host_readback(extension, output / "FIRST-BATCH.json", step=START, bank_sha256=sha(actual))


def extended_bank(capsule, root, expected):
    _, bank, data = cold_bank(capsule)
    datasets, tokenizer, prefix, original_slots, original_rates, _ = data
    record = read(pin(root / "ACTUAL.json", expected))
    require(record["course"] == COURSE and record["plan"] == PLAN
            and record["original_bank"] == bank["sha256"] and record["original_seal"] == BASE_SEAL
            and record["start"] == START and record["end"] == END
            and record["rng_seed"] == SEED and record["sampling_seed"] == SAMPLING_SEED,
            "The 3K bank must append to the exact 1K bank")
    with np.load(pin(root / "BANK.npz", record["bank_sha256"]), allow_pickle=False) as file:
        banks = {k: file[k] for k in BATCH}
        slots, rates = file["auxiliary_slots"], file["rates"]
    require(all(np.array_equal(banks[k][:START], old) for k, old in prefix.items())
            and np.array_equal(slots[:START], original_slots) and np.array_equal(rates[:START], original_rates),
            "The 3K bank changed the already consumed 1K prefix")
    population = validate(datasets, banks, slots, rates, end=END, rate_schedule=rate)
    return datasets, tokenizer, banks, slots, rates, population


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("capsule", type=Path)
    cli.add_argument("output", type=Path)
    pack(**vars(cli.parse_args()))
