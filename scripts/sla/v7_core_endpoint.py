"""Reconcile the completed V7-A pair and export its independently restored Go2 providers."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha
from scripts.pi0292.durable import write


PLAN = "67d15fb8fad688a2bbd82e225111bb49ec34559b"
R4 = "f1f5a994005a9fc831934a156ca823d7e4b67794ce2533d72e087c2aeb29b9cb"
BANK = "ec5e7436b839e8452b61fc137f4fe6f659af2aabb522165c7604413b12d20024"
ARRAYS = "a8fde923b6453be823fd8097b8440a7ad8780d6d484bc3d882df8a5b070e64ff"
ARMS = ("CONTROL_NEUTRAL", "TREATMENT_PREVIOUS_EXECUTED_GRIPPER_COMMAND")
STEPS, SEED = 500, 529202611
BATCH = dict(target=96,robodojo=16,libero=16)
TARGET_GROUPS = dict(old=48,new=46,recovery=2)
VISITS = {**BATCH,"auxiliary":4}
PROJECTION = "previous_executed_gripper_projection"


def byte_identity(value):
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def array_list_identity(rows, projection):
    values = [r["sha256"] for r in rows] + [projection["sha256"]]
    return hashlib.sha256(json.dumps(values,separators=(",",":")).encode()).hexdigest()


def hopper(capsule, output):
    from hopper_pbs import native

    job = read(capsule/"JOB.json")["job"]
    state = native(job)
    terminal = read(capsule/"TERMINAL.json")
    require(state["job_state"] == "F" and state["Exit_status"] == 0
            and terminal["status"] == "FORMAL_V7_PAIR_COMPLETE" and terminal["plan"] == PLAN
            and terminal["formal_updates"] == 2*STEPS and terminal["engineering_updates"] == 0
            and tuple(terminal["arms"]) == ARMS, "Require both fixed endpoints and a normal PBS terminal")
    write(output,dict(plan=PLAN,job=job,job_state="F",exit_status=0,formal_updates=2*STEPS,
          binding_sha256=sha(capsule/"BINDING.json"),terminal_sha256=sha(capsule/"TERMINAL.json"),
          arms=terminal["arms"]))


def paired_ledger(capsule, bank):
    paths = [capsule/arm/"UPDATES.jsonl" for arm in ARMS]
    counts = Counter()
    timing = ([], [])
    with np.load(pin(bank/"BANK.npz",ARRAYS),allow_pickle=False) as planned:
        with paths[0].open() as a, paths[1].open() as b:
            for i, (left, right) in enumerate(zip(a,b,strict=True)):
                rows = (json.loads(left),json.loads(right))
                for row, seconds in zip(rows,timing,strict=True):
                    terms = row["components"]
                    require(i < STEPS and row["step"] == i+1 and row["lr"] == planned["rates"][i]
                            and row["go2"] == TARGET_GROUPS
                            and np.isfinite(row["global_gradient_norm"]) and row["global_gradient_norm"] > 0
                            and row["seconds"] > 0 and np.isfinite(row["seconds"])
                            and {role: part["visits"] for role,part in terms.items()} == VISITS
                            and all(np.isfinite(part["loss"]) and np.isfinite(part["gradient_l2"]) for part in terms.values())
                            and all(terms[role]["visit_sha256"] == byte_identity(planned[role][i]) for role in BATCH)
                            and terms["auxiliary"]["slots"] == planned["auxiliary_slots"][i].tolist(),
                            "An actual formal update differs from the fixed step or sample bank")
                    seconds.append(row["seconds"])
                noise = rows[0]["noise"]
                require(noise == rows[1]["noise"] and set(noise) == set(VISITS)
                        and all(len(sha256) == 64 for keys in noise.values() for sha256 in keys)
                        and all(len(noise[role]) == 2 for role in BATCH) and len(noise["auxiliary"]) == 1,
                        "Actual paired CUDA draw identities differ")
                counts.update(TARGET_GROUPS)
                counts.update({role:VISITS[role] for role in ("robodojo","libero","auxiliary")})
    require(all(len(seconds) == STEPS for seconds in timing), "The formal pair is missing updates")
    return dict(formal_updates=2*STEPS,matched_cuda_steps=STEPS,visits_each=dict(counts),
                timing_p50_s={arm:float(np.median(seconds[2:])) for arm,seconds in zip(ARMS,timing,strict=True)})


def export_provider(root, endpoint, parent, arm, proof, step500):
    import jax
    import orbax.checkpoint as ocp
    from flax.traverse_util import flatten_dict
    from v65_recovery_endpoint import r4_provider

    require(jax.default_backend() == "cpu" and proof["complete_training_endpoint"] and
            endpoint.name == "STEP0500" and step500["step"] == STEPS and not root.exists(),
            "Require the independent final CPU restore and a new deployment path")
    old, old_rows, frozen_lora, frozen_hash = r4_provider(parent)
    require(step500["lora_sha256"] == frozen_hash, "The original R4 frozen LoRA does not match")
    with ocp.PyTreeCheckpointer() as reader:
        metadata = reader.metadata(endpoint).tree["parameters"]
        wanted = {domain:metadata[domain] for domain in ("target","shared")}
        args = jax.tree.map(lambda _:ocp.RestoreArgs(restore_type=np.ndarray),wanted)
        restored = reader.restore(endpoint,args=ocp.args.PyTreeRestore(
            item={"parameters":wanted},restore_args={"parameters":args},transforms={}))["parameters"]
    active = flatten_dict(restored)
    condition = active.pop(("target",PROJECTION))
    require(condition.shape == (1,1024) and condition.dtype == np.float32 and np.isfinite(condition).all()
            and bool(condition.any()) == (arm == ARMS[1]) and byte_identity(condition) == step500["projection_sha256"]
            and not active.keys() & frozen_lora.keys() and active.keys() | frozen_lora.keys() == old_rows.keys(),
            "The final condition or original provider array ABI differs from the checkpoint")
    stage = Path(tempfile.mkdtemp(prefix=f".{root.name}-",dir=root.parent))
    (stage/"parameters").mkdir()
    rows = []
    for name, row in old_rows.items():
        target = stage/row["file"]
        if name in frozen_lora:
            shutil.copyfile(parent/row["file"],target)
            pin(target,row["sha256"])
        else:
            value = active[name]
            require(list(value.shape) == row["shape"] and str(value.dtype) == row["dtype"], "Incompatible target array")
            with target.open("xb") as stream:
                np.save(stream,value,allow_pickle=False)
        rows.append(dict(row,sha256=sha(target)))
    path = stage/"previous_command_projection.npy"
    with path.open("xb") as stream:
        np.save(stream,condition,allow_pickle=False)
    projection = dict(file=path.name,sha256=sha(path),array_sha256=byte_identity(condition),shape=[1,1024],dtype="float32",
                      input="previous_actual_consumed_gripper_command",arm=arm)
    write(stage/"CHECKPOINT.json",dict(old,role="V7_A_CONTEXT_500",phase=STEPS,arrays=rows,
          v7_plan=PLAN,v7_arm=arm,v7_previous_command=projection,
          checkpoint_sha256=array_list_identity(rows,projection),training_receipt_sha256=proof["endpoint_sha256"],
          training_contract_sha256=proof["contract_sha256"],parent_lora_sha256=frozen_hash,
          formal_inference_eligible=False,native_admission=False,hardware_allowed=False,physics=False))
    stage.rename(root)
    return sha(root/"CHECKPOINT.json")


def core(capsule, bank, parent, receipt, expected_receipt, output):
    from v3_main_place_learn import endpoint_files
    from v62_source_endpoint import restore

    proof = read(pin(receipt,expected_receipt))
    require(proof["plan"] == PLAN and proof["job_state"] == "F" and proof["exit_status"] == 0
            and proof["formal_updates"] == 2*STEPS and tuple(proof["arms"]) == ARMS, "Invalid completed Hopper readback")
    terminal = read(pin(capsule/"TERMINAL.json",proof["terminal_sha256"]))
    binding = read(pin(capsule/"BINDING.json",proof["binding_sha256"]))
    manifest = read(pin(bank/"MATCHED_INPUTS.json",BANK))
    require(binding["r4_receipt"] == R4 and binding["formal_steps_per_arm"] == STEPS and binding["arms"] == list(ARMS)
            and binding["plan"] == terminal["plan"] == PLAN and binding["manifest"]["sha256"] == BANK
            and binding["bank"]["sha256"] == manifest["bank_sha256"] == ARRAYS
            and terminal["arms"] == proof["arms"] and terminal["formal_updates"] == 2*STEPS,
            "Hopper handoff differs from the original completed pair binding")
    ledger = paired_ledger(capsule,bank)
    checks = []
    for arm in ARMS:
        run = capsule/arm
        result = read(pin(run/"TERMINAL.json",proof["arms"][arm]["terminal_sha256"]))
        contract = read(pin(run/"RUN_CONTRACT.json",result["contract_sha256"]))
        sealed = run/"STEP0500_SEALED.json"
        checkpoint = read(sealed)
        endpoint = run/"STEP0500"
        require(result["status"] == "FORMAL_V7_ARM_500_COMPLETE" and result["step"] == STEPS
                and result["visits"] == ledger["visits_each"] and result["checkpoint"] == checkpoint
                and checkpoint["files"] == endpoint_files(endpoint)
                and contract["new_adam_from_r4"] and contract["plan"] == PLAN and contract["arm"] == arm
                and contract["binding_sha256"] == proof["binding_sha256"] and contract["seed"] == SEED,
                "Unexpected training lineage, visit counts or transferred checkpoint bytes")
        pin(run/"UPDATES.jsonl",result["optimizer_ledger_sha256"])
        restored = restore(endpoint,STEPS,groups={"shared",*BATCH},seed=SEED)
        require(restored["adam_counters"] == checkpoint["adam_counts"], "Adam restore differs from sealed counters")
        checks.append((arm,endpoint,checkpoint,dict(restored,endpoint_sha256=sha(sealed),
                        contract_sha256=result["contract_sha256"],training_terminal_sha256=sha(run/"TERMINAL.json"))))
    require(len({checkpoint["frozen_weight_sha256"] for _,_,checkpoint,_ in checks}) == 1,
            "The two fits must share a frozen model")
    output.mkdir(mode=0o700)
    arms = {}
    for arm,endpoint,sealed,result in checks:
        dest = output/arm
        arms[arm] = dict(result,provider_sha256=export_provider(dest,endpoint,parent,arm,result,sealed))
    write(output/"CORE-PAIR-ACTUAL.json",dict(plan=PLAN,pbs=proof["job"],step=STEPS,arms=arms,
          received_hopper_sha256=expected_receipt,native_integrations=0,**ledger))


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    commands=p.add_subparsers(dest="command",required=True)
    train=commands.add_parser("hopper")
    receive=commands.add_parser("core")
    for inst,names in ((train,("capsule","output")),(receive,("capsule","bank","parent","receipt","output"))):
        for name in names:
            inst.add_argument(name,type=Path)
    receive.add_argument("--expected-receipt",required=True)
    args=vars(p.parse_args())
    {"hopper":hopper,"core":core}[args.pop("command")](**args)
