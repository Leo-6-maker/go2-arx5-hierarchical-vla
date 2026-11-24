"""Separate the three-view packet and right-arm state effects on frozen closing intent."""

import argparse
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from sea_target_t2 import CONDITION as T2_CONDITION
from sea_target_t2 import EVIDENCE, LOCAL, PANEL as T2_PANEL, PLAN_SHA, TL_SHA, anchors_and_payloads, exact_noise
from sea_target_t2_execute import run_owner, run_sampler, scoped_context
from sea1a_panel import UUID, infer
from sea1b_panel import discrete_token_reader, fields_at_device, registration, verified_processor, verify_discrete_prompt
from sea_task_193 import device_fields, load_roster
from source_custody import SEEDS, digest, pin, read, require, write_new
from demonstrated_grip import rows
from policy_query import input_identity, tensor_sha

PANEL = "SEA-T3-193-VISUAL-RIGHT-01"
PAIRS = {"G00": "FRUIT00", "G01": "FRUIT02"}
CONDITIONS = ("GG", "GS", "SG", "SS")
NEW_CAP = 12
T2_COLD_SHA = "3f62c6e3ac79e7ef38da130423b92e0d616302c2ffd1db96d5d13f49c4d1e102"
T2_CUSTODY_SHA = "f05f5bb8debdc50a4ab080717f7309c1df0aaf2fb590cb3af608cac5a5d9d8e2"


def code_binding():
    directory = Path(__file__).parent
    return {name: digest(directory / name) for name in (
        "sea_t3_visual_state.py", "sea_t3_visual_state_score.py", "sea_target_t2_execute.py",
        "sea_target_t2.py", "sea_task_193.py", "sea1a_panel.py", "sea1b_panel.py",
        "sea_observation.py", "sea_three_view.py",
    )}


def assemble(root, prepare):
    target_rows, target, _ = anchors_and_payloads(root, prepare)
    _, sources, _ = load_roster(root)
    original = read(pin(EVIDENCE / "SOURCE_TASK/TL_ACTUAL.json", TL_SHA))
    original_inputs = {cell["anchor"]: cell["input_sha256"] for cell in original["results"]}
    decoder = discrete_token_reader(root)
    payloads, inputs, fields, evidence = {}, {}, {}, {}
    for pair, source in PAIRS.items():
        target_id = pair + "-before_close"
        g, s = target[target_id, T2_CONDITION], sources[source, "TL"]
        require(input_identity(s)[0] == original_inputs[source]
                and input_identity(g)[0] == target_rows[target_id]["input_sha256"], "changed source or target anchor")
        fixed_left = g["state"][:8]
        eyes, right = {"G": g["images"], "S": s["images"]}, {"G": g["state"][8:], "S": s["state"][8:]}
        fields[pair], evidence[pair], processed_left = {}, {}, set()
        for condition in CONDITIONS:
            image, body = condition
            obs = dict(images=eyes[image], state=np.r_[fixed_left, right[body]], prompt=LOCAL)
            sha, raw = input_identity(obs)
            payloads[pair, condition] = obs
            inputs[pair, condition] = sha
            prepared = prepare({**obs, "images": dict(obs["images"]), "state": obs["state"].copy()})
            token_length = verify_discrete_prompt(decoder, prepared, LOCAL)
            processed_left.add(tensor_sha(prepared["state"][:7]))
            fields[pair][condition] = fields_at_device(prepared)
            evidence[pair][condition] = dict(raw_fields=raw, source_anchor=source, target_anchor=target_id,
                                             left_state_sha256=tensor_sha(fixed_left),
                                             right_state_sha256=tensor_sha(right[body]),
                                             prompt_and_state_token_count=token_length,
                                             input_sha256=sha, right_query_aperture=float(right[body][7]))
        require(inputs[pair, "GG"] == target_rows[target_id]["input_sha256"], "GG is not the original T2 condition")
        require(len(processed_left) == 1, "left state changed in the official coordinate conversion")
        verify_factors(fields[pair])
    return payloads, inputs, fields, evidence


def verify_factors(fields):
    require(all(len(row["memory"]) == 9 and all(row["current_masks"].values()) for row in fields.values()),
            "all three real image slots and nine history frames must be retained")
    for a, b in (("GG", "GS"), ("SG", "SS")):
        x, y = fields[a], fields[b]
        require(all(x[k] == y[k] for k in ("current", "memory", "current_masks"))
                and all(x[k] != y[k] for k in ("state", "token_ids")), "state intervention leaks to images")
    for a, b in (("GG", "SG"), ("GS", "SS")):
        x, y = fields[a], fields[b]
        require(all(x[k] == y[k] for k in ("state", "token_ids", "token_mask", "current_masks"))
                and x["memory"] != y["memory"] and all(x["current"][k] != y["current"][k] for k in x["current"]),
                "three-view packet intervention leaks to state or prompt")


def roster(inputs, noises):
    return [dict(id=f"{pair}-{cond}-seed{n}", anchor=pair, condition=cond, seed=seed,
                 input_sha256=inputs[pair, cond], initial_noise_sha256=tensor_sha(noises[n][2]))
            for n, seed in enumerate(SEEDS) for pair in PAIRS for cond in CONDITIONS]


def old_t2(root):
    references = EVIDENCE / "TARGET_T2"
    provenance = read(pin(references / "CUSTODY.json", T2_CUSTODY_SHA))
    actual = read(pin(references / "ACTUAL.json", provenance["actual_report_sha256"]))
    cold = read(pin(references / "COLD.json", T2_COLD_SHA))
    old = root / "v19" / T2_PANEL
    files = {name: pin(old / name, sha) for name, sha in provenance["asset_sha256"].items()}
    start = read(files["START.json"])
    exit_record = read(files["OWNER_EXIT.json"])
    load, = [row for row in rows(files["calls.jsonl"]) if row["event"] == "LOAD_RETURNED"]
    require(provenance["owner_process_absent"] and provenance["project_lock_reacquired_readonly"]
            and start["cold_sha256"] == T2_COLD_SHA and load["gpu"] == UUID
            and exit_record == provenance["terminal_receipt"] and exit_record["project_lock_released"]
            and exit_record["returned"] == exit_record["entered"] == 8
            and actual["counts"] == dict(RETURNED=8, TECHNICAL_ENTERED=0, UNSTARTED=0),
            "original T2 is unclosed or not usable for exact reuse")
    return old, {r["id"]: r for r in actual["rows"]}, cold, load["jax_version"], files


def exact_reuse(root, cells, fields, noises, processor, *, on_gpu):
    import jax

    old, done, cold, jax_version, files = old_t2(root)
    require(cold["processor"] == processor and jax_version == jax.__version__
            and cold["code"]["sea1a_panel.py"] == code_binding()["sea1a_panel.py"]
            and cold["code"]["sea_task_193.py"] == code_binding()["sea_task_193.py"],
            "T2 recorded model or sampler identity differs")
    device_fields_old = read(files["DEVICE_INPUTS.json"])
    prior = {}
    for cell in cells:
        if cell["condition"] != "GG":
            continue
        pair, seed = cell["anchor"], cell["seed"][1]
        original = pair + f"-before_close-seed{seed}"
        at = pair + "-before_close"
        e, r = (read(old / f"{original}.{phase}.json") for phase in ("entered", "returned"))
        raw = done[original]
        trace = device_fields_old[at]
        require(e["fields"] == trace and all(e[key] == cell[key] == raw[key] for key in (
                    "input_sha256", "seed", "initial_noise_sha256"))
                and r["file_sha256"] == raw["output_sha256"]
                and r["raw_sha256"] == raw["raw_sha256"] and r["decoded_sha256"] == raw["decoded_sha256"],
                "archived GG input or return differs")
        want = trace if on_gpu else cold["fields"][at][T2_CONDITION]
        require(fields[pair, "GG"] == want, "GG sampler tensors differ from T2")
        _, key, noise = noises[seed]
        with np.load(files[f"noise-{seed}.npz"], allow_pickle=False) as stored:
            require(np.array_equal(stored["noise"], noise) and np.array_equal(stored["sample_key"], key)
                    and e["sample_key"] == key.tolist(), "GG actual noise or sample key differs")
        path = pin(old / f"{original}.npz", r["file_sha256"])
        with np.load(path, allow_pickle=False) as z:
            require(tensor_sha(z["absolute"]) == r["decoded_sha256"] and tensor_sha(z["actions"]) == r["raw_sha256"],
                    "archived GG tensors differ")
        prior[cell["id"]] = dict(archived_slot=original, output_sha256=r["file_sha256"],
                                 raw_sha256=r["raw_sha256"], decoded_sha256=r["decoded_sha256"])
    require(len(prior) == 4, "all four original GG cells required before new sampling")
    return prior


def cold(root, destination):
    import jax
    import jax.numpy as jnp
    from openpi.policies.policy import Policy

    require(jax.default_backend() == "cpu" and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU proof only")
    frozen, _ = registration()
    prepare, inverse, processor = verified_processor(root, frozen)
    payloads, inputs, fields, evidence = assemble(root, prepare)
    noises = exact_noise(root)
    cells = roster(inputs, noises)
    flat = {(pair, c): tensor for pair, factors in fields.items() for c, tensor in factors.items()}
    reuse = exact_reuse(root, cells, flat, noises, processor, on_gpu=False)
    stub = SimpleNamespace(_input_transform=prepare, _output_transform=inverse, _sample_kwargs={},
                           _sample_actions=lambda _, __, **kw: jnp.zeros_like(kw["noise"]))
    stub.infer = Policy.infer.__get__(stub)
    witness = []
    for cell in cells:
        k = cell["anchor"], cell["condition"]
        out, captured = infer(stub, payloads[k], flat[k], *noises[cell["seed"][1]], witness.append)
        require(out["actions"].shape == (50, 16) and captured["actions"].shape == (50, 32), "official ABI")
    require(len(witness) == 16, "incomplete CPU sampling seam")
    write_new(destination, dict(
        status="T3_3V_FACTORIAL_CPU_COLD_VERIFIED", panel=PANEL, plan_sha256=PLAN_SHA, cells=cells,
        code=code_binding(), pairs=PAIRS, fields=fields, evidence=evidence, processor=processor,
        reuse_cpu_verified=reuse, source_real_left_positive_control=TL_SHA, max_new_calls=NEW_CAP,
        surrogate_calls=len(witness), model_forwards=0, new_native_starts=0, physics_ticks=0, optimizer_updates=0,
    ))


def worker(root, cold_path, descriptor):
    dest, proof, current = scoped_context(root, cold_path, descriptor, PANEL, PLAN_SHA)
    require(proof["code"] == code_binding(), "T3 code changed after proof")
    frozen, _ = registration()
    prepare, _, processor = verified_processor(root, frozen)
    payloads, inputs, _, evidence = assemble(root, prepare)
    noises = exact_noise(root)
    cells = roster(inputs, noises)
    require(cells == proof["cells"] and evidence == proof["evidence"] and processor == proof["processor"],
            "fixed T3 input differs from CPU proof")
    fields = device_fields(prepare, payloads, proof["fields"], set(CONDITIONS))
    reuse = exact_reuse(root, cells, fields, noises, processor, on_gpu=True)
    require(reuse == proof["reuse_cpu_verified"], "historical GG outputs changed after CPU proof")
    new = [cell for cell in cells if cell["id"] not in reuse]
    require(len(new) == NEW_CAP, "T3 maximum new-forward allocation")
    write_new(dest / "REUSE.json", reuse)
    write_new(dest / "DEVICE_INPUTS.json", {"/".join(k): v for k, v in fields.items()})
    run_sampler(root, dest, PANEL, current, new, payloads, fields, noises)


def execute(root, cold_path, allocation_path, allocation_sha):
    proof = read(cold_path)
    grant = read(pin(allocation_path, allocation_sha))
    require(proof["status"] == "T3_3V_FACTORIAL_CPU_COLD_VERIFIED" and proof["panel"] == PANEL
            and proof["plan_sha256"] == PLAN_SHA and proof["code"] == code_binding()
            and proof["max_new_calls"] == NEW_CAP and len(proof["reuse_cpu_verified"]) == 4
            and len({c["id"] for c in proof["cells"]}) == 16 and proof["model_forwards"] == 0,
            "changed or incomplete T3 proof")
    require(grant["status"] == "APPROVED" and grant["panel"] == PANEL and bool(grant["authorization_id"])
            and grant["max_forwards"] == NEW_CAP and grant["native_starts"] == grant["physics_ticks"]
            == grant["optimizer_updates"] == 0 and grant["cold_sha256"] == digest(cold_path)
            and grant["valid_until_epoch"] > time.time(), "missing exact T3 scientific allocation")
    run_owner(root, cold_path, grant, allocation_sha, PANEL, PLAN_SHA, Path(__file__), related_t2_sha256=T2_COLD_SHA)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    modes = parser.add_subparsers(dest="mode", required=True)
    modes.add_parser("cold").add_argument("output", type=Path)
    turn = modes.add_parser("execute")
    turn.add_argument("cold", type=Path)
    turn.add_argument("allocation", type=Path)
    turn.add_argument("sha256")
    child = modes.add_parser("worker")
    child.add_argument("cold", type=Path)
    child.add_argument("descriptor", type=int)
    a = parser.parse_args()
    os.umask(0o077)
    if a.mode == "cold":
        cold(a.root.resolve(), a.output)
    elif a.mode == "execute":
        execute(a.root.resolve(), a.cold.resolve(), a.allocation.resolve(), a.sha256)
    else:
        worker(a.root.resolve(), a.cold.resolve(), a.descriptor)
