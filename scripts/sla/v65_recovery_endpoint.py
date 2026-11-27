"""Independently read a V6.5 checkpoint and account for its exact realized inputs."""

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha
from scripts.pi0292.durable import write
from v3_main_place_learn import endpoint_files
from v61_source_course import MAIN, R4
from v62_source_endpoint import restore
from v65_recovery_course import BATCH, END, PARTS, PLAN, SEED


VISITS = {**PARTS, "robodojo_old": 8, "robodojo_expanded": 8, "libero": 16, "aux": 4}
R4_GO2_PROVIDER = "e825a48ed540b5d4d6c3623da24ac56d5aa3b51f2bf3aa488b313f21e044da5e"


def ledger(run, bank, step):
    bank_id = read(bank / "ACTUAL.json")
    require(bank_id["plan"] == PLAN and bank_id["seed"] == SEED, "The endpoint must bind the immutable 1K input bank")
    with np.load(pin(bank / "BANK.npz", bank_id["bank_sha256"]), allow_pickle=False) as data:
        indices = {name: data[name] for name in BATCH}
        slots, rates = data["auxiliary_slots"], data["rates"]
    source = run / "UPDATES.jsonl"
    with source.open() as stream:
        rows = [json.loads(stream.readline()) for _ in range(step)]
    recovery, phases, moments = Counter(), Counter(), []
    for i, row in enumerate(rows):
        terms = row["components"]
        require(row["step"] == i+1 and row["lr"] == rates[i] and row["go2"] == PARTS
                and row["robodojo_old"] == row["robodojo_expanded"] == 8
                and len(set(row["recovery_ids"])) == 2 and sum(row["go2_phase"].values()) == 96
                and {role: term["visits"] for role, term in terms.items()} == dict(BATCH, auxiliary=4)
                and terms["auxiliary"]["slots"] == slots[i].tolist()
                and all(terms[role]["visit_sha256"] == hashlib.sha256(values[i].tobytes()).hexdigest()
                        for role, values in indices.items())
                and all(math.isfinite(term["gradient_l2"]) and term["gradient_l2"] > 0 and math.isfinite(term["loss"])
                        for term in terms.values())
                and math.isfinite(row["global_gradient_norm"]) and row["global_gradient_norm"] > 0
                and math.isfinite(row["seconds"]) and row["seconds"] > 0,
                f"Incorrect actual inputs or gradients for formal V6.5 update {i+1}")
        recovery.update(row["recovery_ids"])
        phases.update(row["go2_phase"])
        moments.append(row["seconds"])
    if step == END:
        require(len(source.read_text().splitlines()) == END and recovery == bank_id["go2_recovery_episode"],
                "The final update denominator and Recovery exposure must reconcile")
    return dict(actual_visits={role: n*step for role,n in VISITS.items()}, fm_visits=128*step, go2_visits=96*step,
                recovery_episode_visits=dict(recovery), go2_phase_visits=dict(phases),
                warm_p50_s=float(np.median(moments[5:] or moments)), warm_p95_s=float(np.percentile(moments[5:] or moments, 95)))


def audit(capsule, bank, step):
    run = capsule / "run"
    contract_path, binding_path = run / "RUN_CONTRACT.json", capsule / "BINDING.json"
    contract, binding = read(contract_path), read(binding_path)
    qualification = capsule / "qualification/TERMINAL.json"
    require(contract["plan"] == binding["plan"] == PLAN and contract["seed"] == SEED
            and contract["new_adam_from_r4"] and contract["start"] == 0 and contract["end"] == END
            and contract["ancestry"]["parent_sha256"] == MAIN and contract["ancestry"]["initial_r4_receipt_sha256"] == R4
            and contract["qualification_sha256"] == sha(qualification) and read(qualification)["status"] == "QUALIFICATION_PASS"
            and binding["bank"]["sha256"] == sha(bank / "ACTUAL.json") and contract["binding_sha256"] == sha(binding_path),
            "The checkpoint must descend from actual GPU qualification and fresh R4")
    endpoint = run / f"STEP{step:04d}"
    seal = endpoint.with_name(endpoint.name + "_SEALED.json")
    checkpoint = read(seal)
    require(checkpoint["step"] == checkpoint["bank_cursor"] == step and checkpoint["files"] == endpoint_files(endpoint),
            "Pad, cursor, or endpoint bytes differ from the checkpoint seal")
    parameters = read(pin(capsule / "PARAMETERS.json", contract["actual_training_scope_sha256"]))
    require(checkpoint["frozen_weight_sha256"] == parameters["total_frozen_sha256"]
            and checkpoint["lora_sha256"] == parameters["parent_lora_sha256"], "Frozen parent identity changed")
    inputs = ledger(run, bank, step)
    if step == END:
        terminal = read(run / "TERMINAL.json")
        require(terminal["status"] == "FORMAL_V65_1K_COMPLETE" and terminal["formal_updates"] == END
                and terminal["contract_sha256"] == sha(contract_path) and terminal["checkpoint"] == checkpoint
                and terminal["optimizer_visits"] == inputs["actual_visits"], "Formal completion has not been sealed")
    state = restore(endpoint, step, groups={"shared", *BATCH}, seed=SEED)
    require(state["adam_counters"] == checkpoint["adam_counts"], "Independent Adam readback disagrees")
    return dict(step=step, endpoint_sha256=sha(seal), contract_sha256=sha(contract_path), plan=PLAN,
                initial_r4_receipt_sha256=R4, frozen_lora_sha256=checkpoint["lora_sha256"],
                formal_scientific_endpoint=step == END, **inputs, **state)


def r4_provider(root):
    from go2arx5_vla.sla_adapter.scale20k_contract import is_lora_parameter

    manifest = read(pin(root / "CHECKPOINT.json", R4_GO2_PROVIDER))
    require(manifest["training_receipt_sha256"] == R4 and len(manifest["arrays"]) == 29,
            "The original complete R4-Go2 provider is required")
    base, lora = {}, {}
    for row in manifest["arrays"]:
        group, domain, *path = re.findall(r"\['([^']+)'\]", row["key"])
        require(group == "parameters" and domain in ("shared", "target"), "Unknown old provider parameter path")
        key = (domain, *path)
        base[key] = row
        if domain == "shared" and is_lora_parameter(path[-1]):
            lora[key] = np.load(pin(root / row["file"], row["sha256"]), allow_pickle=False)
    digest = hashlib.sha256()
    for (_, *path), value in sorted(lora.items()):
        digest.update(("/".join(path)+":"+str(value.shape)+":"+str(value.dtype)+"\n").encode())
        digest.update(value.tobytes())
    return manifest, base, lora, digest.hexdigest()


def export_go2(capsule, parent, destination, proof):
    import jax
    import orbax.checkpoint as ocp
    from flax.traverse_util import flatten_dict

    require(jax.default_backend() == "cpu" and proof["step"] == END and proof["formal_scientific_endpoint"],
            "Export only from the independently restored final endpoint on CPU")
    original, base, lora, digest = r4_provider(parent)
    require(digest == proof["frozen_lora_sha256"], "R4 evaluation export does not carry the actual frozen LoRA bytes")
    endpoint = capsule / "run/STEP1000"
    with ocp.PyTreeCheckpointer() as reader:
        meta = reader.metadata(endpoint).tree["parameters"]
        groups = {k: meta[k] for k in ("shared", "target")}
        arrays = jax.tree.map(lambda _: ocp.RestoreArgs(restore_type=np.ndarray), groups)
        restored = reader.restore(endpoint, args=ocp.args.PyTreeRestore(
            item={"parameters": groups}, restore_args={"parameters": arrays}, transforms={}))
        active = flatten_dict(restored["parameters"])
    require(not active.keys() & lora.keys() and active.keys() | lora.keys() == base.keys(),
            "Active and inherited arrays do not exactly reconstruct the original Go2 inference ABI")
    require(not destination.exists(), "Preserve prior evaluation weights")
    staging = destination.with_name(destination.name + f".pending.{os.getpid()}")
    (staging / "parameters").mkdir(mode=0o700, parents=True)
    rows = []
    for key, row in base.items():
        target = staging / row["file"]
        if key in lora:
            shutil.copyfile(parent / row["file"], target)
            pin(target, row["sha256"])
        else:
            value = active[key]
            require(list(value.shape) == row["shape"] and str(value.dtype) == row["dtype"], "Incompatible Go2 array")
            with target.open("xb") as stream:
                np.save(stream, value, allow_pickle=False)
        actual = sha(target)
        rows.append(dict(row, sha256=actual, changed=actual != row["sha256"]))
    checkpoint = hashlib.sha256(json.dumps([r["sha256"] for r in rows], separators=(",", ":")).encode()).hexdigest()
    write(staging / "CHECKPOINT.json", dict(original, phase=END, arrays=rows, checkpoint_sha256=checkpoint,
          role="V65_RECOVERY_ADAPT_1K", formal_inference_eligible=False, training_receipt_sha256=proof["endpoint_sha256"],
          training_contract_sha256=proof["contract_sha256"], parent_lora_sha256=digest))
    staging.rename(destination)
    return sha(destination / "CHECKPOINT.json")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("capsule", type=Path)
    cli.add_argument("bank", type=Path)
    cli.add_argument("step", type=int, choices=sorted({*range(100, END+1, 100), 250}))
    cli.add_argument("output", type=Path)
    cli.add_argument("--r4-provider", type=Path)
    cli.add_argument("--provider", type=Path)
    args = cli.parse_args()
    require(not args.output.exists() and bool(args.r4_provider) == bool(args.provider), "Require paired optional provider paths")
    proof = audit(args.capsule, args.bank, args.step)
    if args.provider:
        proof["provider_manifest_sha256"] = export_go2(args.capsule, args.r4_provider, args.provider, proof)
    write(args.output, proof)
