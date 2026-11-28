"""Cold-restore both complete B6 endpoints and export their neutral-input providers."""
import argparse
from contextlib import closing
import gc
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha
from scripts.pi0292.durable import OwnerLock, write

ARMS = ("LAMBDA_ZERO", "SHARED_L2SP")
SCOPE = "V7_B6_SHARED_L2SP_MATCHED_V1"


def export(endpoint, parent, destination, seal, proof):
    import jax
    import orbax.checkpoint as ocp
    from flax.traverse_util import flatten_dict
    from v65_recovery_endpoint import r4_provider

    original, rows, inherited, digest = r4_provider(parent)
    require(digest == seal["lora_sha256"], "Inherited frozen LoRA changed")
    with ocp.PyTreeCheckpointer() as reader:
        metadata = reader.metadata(endpoint).tree["parameters"]
        selected = {key: metadata[key] for key in ("shared", "target")}
        args = jax.tree.map(lambda _: ocp.RestoreArgs(restore_type=np.ndarray), selected)
        saved = reader.restore(endpoint, args=ocp.args.PyTreeRestore(
            item={"parameters": selected}, restore_args={"parameters": args}, transforms={}))
    active = flatten_dict(saved["parameters"])
    projection = active.pop(("target", "previous_executed_gripper_projection"))
    require(projection.shape == (1, 1024) and not projection.any()
            and hashlib.sha256(projection.tobytes()).hexdigest() == seal["projection_sha256"],
            "Only the exactly neutral B6 condition can use the unchanged provider")
    require(not active.keys() & inherited.keys() and active.keys() | inherited.keys() == rows.keys(),
            "B6 arrays cannot reconstruct the original inference ABI")
    (destination / "parameters").mkdir(mode=0o700, parents=True)
    exported = []
    for key, row in rows.items():
        path = destination / row["file"]
        if key in inherited:
            shutil.copyfile(parent / row["file"], path)
            pin(path, row["sha256"])
        else:
            value = active[key]
            require(list(value.shape) == row["shape"] and str(value.dtype) == row["dtype"], "Array ABI changed")
            with path.open("xb") as stream:
                np.save(stream, value, allow_pickle=False)
        exported.append(dict(row, sha256=sha(path)))
    identity = hashlib.sha256(json.dumps([r["sha256"] for r in exported], separators=(",", ":")).encode()).hexdigest()
    write(destination / "CHECKPOINT.json", dict(original, arrays=exported, checkpoint_sha256=identity,
          role="V7_B6_SHARED_PARENT_ANCHOR", phase=seal["step"], b6_arm=proof["arm"],
          training_receipt_sha256=proof["endpoint_sha256"], training_contract_sha256=proof["contract_sha256"],
          parent_lora_sha256=digest, formal_inference_eligible=False, native_admission=False,
          hardware_allowed=False, physics=False))
    return sha(destination / "CHECKPOINT.json")


def run(capsule, terminal_sha, parent, output):
    from v3_main_place_learn import endpoint_files
    from v62_source_endpoint import restore

    require(not output.exists(), "Preserve the existing cold-restore output")
    terminal = read(pin(capsule / "TERMINAL.json", terminal_sha))
    binding = read(capsule / "BINDING.json")
    step = terminal["end_step"]
    require(terminal["scope"] == binding["scope"] == SCOPE and terminal["status"] == "PAIR_COMPLETE"
            and terminal["formal_updates"] == 2 * step and step in (100, 250, 500), "Incomplete matched B6 pair")
    output.mkdir(mode=0o700)
    checks, ledgers = {}, []
    with closing(OwnerLock(output / "OWNER.lock")):
        for arm in ARMS:
            source = capsule / arm
            result = read(pin(source / "TERMINAL.json", terminal["arms"][arm]["terminal_sha256"]))
            seal_path = source / f"STEP{step:04d}_SEALED.json"
            seal = read(seal_path)
            endpoint = source / f"STEP{step:04d}"
            require(result["scope"] == SCOPE and result["arm"] == arm
                    and result["status"] == "FORMAL_ENDPOINT_COMPLETE" and result["formal_updates"] == step
                    and result["updates_not_in_ledger"] == 0 and seal == result["checkpoint"]
                    and seal["files"] == endpoint_files(endpoint) and seal["bank_cursor"] == step,
                    "Transferred endpoint, complete Adam state or actual ledger differs")
            ledger = [json.loads(row) for row in pin(source / "UPDATES.jsonl", result["ledger_sha256"]).read_text().splitlines()]
            require(len(ledger) == step and [row["step"] for row in ledger] == list(range(1, step + 1)),
                    "Missing actual updates")
            proof = restore(endpoint, step, groups={"shared", "target", "robodojo", "libero"}, seed=529202611)
            require(proof["adam_counters"] == seal["adam_counts"], "Adam counters differ from the source")
            proof.update(arm=arm, endpoint_sha256=sha(seal_path), contract_sha256=sha(source / "RUN_CONTRACT.json"))
            proof["provider_sha256"] = export(endpoint, parent, output / arm, seal, proof)
            proof["raw_parent_penalty_before_last_update"] = ledger[-1]["parent_anchor"]["raw_loss"]
            proof["warm_p50_s"] = float(np.median([row["seconds"] for row in ledger[2:]]))
            checks[arm] = proof
            ledgers.append(ledger)
            gc.collect()
        for left, right in zip(*ledgers, strict=True):
            require(left["step"] == right["step"] and left["lr"] == right["lr"] and left["noise"] == right["noise"]
                    and left["go2"] == right["go2"] == dict(old=48, new=46, recovery=2)
                    and all(left["components"][role]["visit_sha256"] == right["components"][role]["visit_sha256"]
                            for role in ("target", "robodojo", "libero")), "Matched pair diverged in its inputs")
        write(output / "ACTUAL.json", dict(scope=SCOPE, step=step, arms=checks, formal_updates=2*step,
              matched_cuda_steps=step, complete_training_endpoint=True, source_terminal_sha256=terminal_sha,
              new_optimizer_updates=0, native_integrations=0))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("capsule", type=Path)
    cli.add_argument("terminal_sha")
    cli.add_argument("parent", type=Path)
    cli.add_argument("output", type=Path)
    run(**vars(cli.parse_args()))
