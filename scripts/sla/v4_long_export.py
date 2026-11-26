"""Audit V4 training-state receipts and export the Go2 parameters for frozen native evaluation."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha
from v3_main_place_learn import endpoint_files


COUNTS = {"target": 64, "robodojo": 32, "libero": 32, "auxiliary": 4}
SCHEMA = "SLA_V4_MAIN_MULTISOURCE_INFERENCE"
FORMAL_ROLES = {1000: "MAIN_MULTISOURCE_1000_V4", 2000: "MAIN_MULTISOURCE_2000_V4"}


def ledger_prefix(result, steps):
    with (result / "UPDATES.jsonl").open() as log:
        rows = [json.loads(log.readline()) for _ in range(steps)]
    require([row["step"] for row in rows] == list(range(1, steps + 1)), "Training ledger has duplicate or missing updates")
    for row in rows:
        require(math.isfinite(row["lr"]) and row["lr"] > 0
                and math.isfinite(row["global_gradient_norm"]) and row["global_gradient_norm"] > 0,
                "Missing finite global update")
        require({k: v["visits"] for k, v in row["components"].items()} == COUNTS,
                "Realized data population differs from the declared per-update counts")
    return rows


def audit(run, receipt_file, custody, bank_root):
    receipt = read(receipt_file)
    step = int(receipt["step"])
    path = Path(receipt["endpoint"]).resolve(strict=True)
    require(path.parent == run.resolve(strict=True) and path.name == f"STEP{step:04d}"
            and receipt_file.parent.resolve(strict=True) == run.resolve(strict=True),
            "Selected checkpoint belongs to another run")
    contract_path = run / "RUN_CONTRACT.json"
    contract = read(contract_path)
    settings, charter = contract["settings"], contract["charter"]
    require(step > 0 and step <= charter["updates"]
            and contract["model"]["parent_sha256"] == charter["parent_sha256"],
            "Wrong declared ancestry or update count")
    if receipt.get("formal_endpoint"):
        require(receipt["completed_updates"] == step == charter["updates"]
                and receipt["contract_sha256"] == sha(contract_path)
                and len((run / "UPDATES.jsonl").read_text().splitlines()) == step,
                "The formal endpoint does not include the full registered update population")

    bank = bank_root / "prepared"
    frozen_bank = read(pin(bank / "ACTUAL.json", charter["target_bank_sha256"]))
    with np.load(pin(bank / "BANK.npz", frozen_bank["archive_sha256"]), allow_pickle=False) as data:
        indices = {"target": data["target_indices"]}
        slots = data["auxiliary_slots"]
    require(hashlib.sha256(slots.tobytes()).hexdigest() == settings["aux_bank_sha256"],
            "The fixed physical auxiliary slots changed")
    parent = Path(charter["parent_path"])
    require(parent.resolve(strict=True).is_relative_to(custody.resolve(strict=True)), "Unknown model custody")
    progress = read(pin(parent / "COMPLETE.json", charter["parent_sha256"]))["progress"]
    for domain in ("robodojo", "libero"):
        start = progress["role_visits"][domain]
        bank_file = custody / "banks.verified" / (domain + ".npy")
        source = np.load(bank_file, mmap_mode="r", allow_pickle=False)
        count = COUNTS[domain]
        indices[domain] = source[start:start + 2000 * count].reshape(2000, count)
    require({domain: hashlib.sha256(mm.tobytes()).hexdigest() for domain, mm in indices.items()}
            == settings["exact_bank_hashes"], "A frozen training sample bank differs")
    rows = ledger_prefix(run, step)
    for idx, row in enumerate(rows):
        for domain, data in indices.items():
            require(row["components"][domain]["visit_sha256"] == hashlib.sha256(data[idx].tobytes()).hexdigest(),
                    f"Consumed {domain} visits disagree at update {idx + 1}")
        require(row["components"]["auxiliary"]["slots"] == slots[idx].tolist(),
                f"Physical auxiliary slots disagree at update {idx + 1}")
    started = time.monotonic()
    require(endpoint_files(path) == receipt["files"], "Saved checkpoint bytes no longer match the writer's receipt")
    return dict(step=step, formal=receipt.get("formal_endpoint", False), contract_sha256=sha(contract_path),
                parent_sha256=charter["parent_sha256"], receipt_sha256=sha(receipt_file),
                files=len(receipt["files"]), tensor_hash_readback_s=time.monotonic() - started,
                target_visits=step * 64, robodojo_visits=step * 32, libero_visits=step * 32, auxiliary_visits=step * 4,
                checkpoint_bytes=sum((path / f).stat().st_size for f in receipt["files"]),
                p50_seconds=float(np.median([row["seconds"] for row in (rows[2:] or rows)])),
                endpoint=str(path))


def export(proof, reference, destination, *, diagnostic=False):
    import jax
    import orbax.checkpoint as ocp
    from flax.traverse_util import flatten_dict
    from scripts.pi0292.durable import write

    require(jax.default_backend() == "cpu", "The read-only exporter must use CPU")
    step = proof["step"]
    require(diagnostic or (proof["formal"] and step in FORMAL_ROLES), "Only the registered complete endpoint is eligible")
    original = read(reference / "CHECKPOINT.json")
    require(len(original["arrays"]) == 29 and original["checkpoint_sha256"] == proof["parent_sha256"],
            "Missing the exact native provider template for this parent")
    endpoint = Path(proof["endpoint"])
    with ocp.PyTreeCheckpointer() as saver:
        metadata = saver.metadata(endpoint).tree
        selected = {key: metadata["parameters"][key] for key in ("shared", "target")}
        types = jax.tree.map(lambda _: ocp.RestoreArgs(restore_type=np.ndarray), selected)
        state = saver.restore(endpoint, args=ocp.args.PyTreeRestore(
            item={"parameters": selected, "step": metadata["step"]},
            restore_args={"parameters": types, "step": ocp.RestoreArgs(restore_type=int)}, transforms={}))
    require(state["step"] == step, "Training-state clock and write receipt disagree")
    values = flatten_dict(state["parameters"])
    staged = destination.with_name(destination.name + f".pending.{os.getpid()}")
    require(not destination.exists(), "Preserve the original exported provider package")
    staged.mkdir(mode=0o700)
    (staged / "parameters").mkdir()
    rows, seen = [], set()
    for previous in original["arrays"]:
        root, domain, *path = re.findall(r"\['([^']+)'\]", previous["key"])
        value = np.asarray(values[tuple([domain, *path])])
        relative = Path(previous["file"])
        require(root == "parameters" and relative.parts == ("parameters", relative.name)
                and domain in ("shared", "target") and list(value.shape) == previous["shape"]
                and str(value.dtype) == previous["dtype"] and np.isfinite(value).all(),
                "Learned array differs from the frozen Go2 provider ABI")
        with (staged / relative).open("xb") as f:
            np.save(f, value, allow_pickle=False)
        digest = sha(staged / relative)
        seen.add(tuple([domain, *path]))
        rows.append(dict(previous, sha256=digest, changed=digest != previous["sha256"]))
    require(set(values) == seen and any(row["changed"] for row in rows),
            "The complete learned Go2 parameter set must be represented")
    identity = hashlib.sha256(json.dumps([r["sha256"] for r in rows], separators=(",", ":")).encode()).hexdigest()
    write(staged / "CHECKPOINT.json", dict(original, schema=SCHEMA, phase=step, arrays=rows,
          role="DIAGNOSTIC" if diagnostic else FORMAL_ROLES[step],
          source_codec="DENSE_REFERENCE_G7_V6", checkpoint_sha256=identity,
          previous_checkpoint_sha256=proof["parent_sha256"], frozen_wbc=True, original_l1=True,
          formal_inference_eligible=not diagnostic, training_receipt_sha256=proof["receipt_sha256"],
          training_contract_sha256=proof["contract_sha256"]))
    staged.rename(destination)
    return sha(destination / "CHECKPOINT.json")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "receipt_file", "custody", "bank_root"):
        cli.add_argument(name, type=Path)
    cli.add_argument("--reference", type=Path)
    cli.add_argument("--export", type=Path)
    cli.add_argument("--diagnostic", action="store_true")
    a = cli.parse_args()
    proof = audit(a.run, a.receipt_file, a.custody, a.bank_root)
    if a.export:
        proof["provider_manifest_sha256"] = export(proof, a.reference, a.export, diagnostic=a.diagnostic)
    print(json.dumps(proof, indent=2))
