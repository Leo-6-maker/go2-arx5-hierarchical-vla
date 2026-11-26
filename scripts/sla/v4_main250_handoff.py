"""Expose a transferred Hopper endpoint to the original Core V3 evaluator."""

import argparse
import hashlib
import os
from pathlib import Path

from scripts.pi0292.durable import OwnerLock, write
from m38_gpu_preflight_v1 import pin, read, require, sha
from v3_main_place_learn import BATCH, MAIN, UPDATES, data_scope, endpoint_files, sample_bank, update_log


def import_endpoint(staged, destination, reference, originals, commands):
    """Publish the evaluator's terminal marker only after all inputs match."""
    require(staged.stat().st_dev == destination.stat().st_dev,
            "Stage the checkpoint on the Core evaluation filesystem")
    lock = OwnerLock(destination / "HOPPER_IMPORT.lock")
    try:
        reservation = read(destination / "MIGRATION_RESERVED.json")
        require(reservation["started"] is False and not (destination / "TRAIN_TERMINAL.json").exists(),
                "The Core slot was not reserved for this migration")
        def payload(name, imported_name=None):
            incoming = staged / name
            installed = destination / (imported_name or name)
            require(incoming.exists() != installed.exists(),
                    "Import payload needs unambiguous reconciliation")
            return incoming if incoming.exists() else installed

        terminal_path = payload("TRAIN_TERMINAL.json", "HOPPER_TERMINAL.original.json")
        terminal = read(terminal_path)
        contract = read(payload("RUN_CONTRACT.json"))
        reference_contract = read(reference / "V3_RUN_CONTRACT.json")
        require(all(contract[key] == value for key, value in reference_contract.items())
                and contract["parent_checkpoint_sha256"] == MAIN and contract["batch"] == BATCH,
                "Hopper run does not match the original science contract")
        require(terminal["status"] == "FORMAL_250" and terminal["formal_endpoint"]
                and terminal["completed_updates"] == UPDATES
                and Path(terminal["endpoint"]).name == "STEP0250",
                "An exact completed endpoint is required")
        endpoint = payload("STEP0250")
        require(endpoint_files(endpoint) == terminal["files"], "Transferred Orbax files differ from their seal")
        previous = read(reference / "CHECKPOINT.json")
        stats = dict(state=read(pin(reference / "STATE.json", previous["state_sha256"]))["state"],
                     actions=read(pin(reference / "ACTION.json", previous["action_sha256"]))["actions"])
        bank, _ = sample_bank(data_scope(originals, commands, stats))
        ledger = update_log(payload("UPDATES.jsonl").parent)
        require(len(ledger) == UPDATES
                and sha(commands / "MANIFEST.json") == contract["data_manifest_sha256"]
                and hashlib.sha256(bank.tobytes()).hexdigest() == contract["exact_sampler_bank_sha256"]
                and all(row["visit_sha256"] == hashlib.sha256(batch.tobytes()).hexdigest()
                        for row, batch in zip(ledger, bank, strict=True)),
                "The 250 original V3 logical batches must match exactly")
        for name in ("RUN_CONTRACT.json", "UPDATES.jsonl", "STEP0250"):
            if (staged / name).exists():
                (staged / name).rename(destination / name)
        if terminal_path.parent == staged:
            terminal_path.rename(destination / "HOPPER_TERMINAL.original.json")
        write(destination / "HOPPER_IMPORT.json", dict(
            original_terminal_sha256=sha(destination / "HOPPER_TERMINAL.original.json"),
            updates_sha256=sha(destination / "UPDATES.jsonl"), complete_formal_updates=UPDATES))
        write(destination / "TRAIN_TERMINAL.json", dict(
            terminal, endpoint=str((destination / "STEP0250").resolve(strict=True))))
    finally:
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("staged", "destination", "reference", "originals", "commands"):
        parser.add_argument(name, type=Path)
    os.umask(0o077)
    import_endpoint(**vars(parser.parse_args()))
