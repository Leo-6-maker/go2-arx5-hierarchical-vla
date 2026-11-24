"""Compare two actual revised auxiliary views against the sealed V1 outputs."""

import argparse
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from sea_mount_capture import PANEL as CAPTURE, SLOTS
from sea_mount_ingress import paired_payloads
from sea_target_t2 import EVIDENCE, PLAN_SHA, exact_noise
from sea_target_t2_execute import run_owner, run_sampler, scoped_context
from sea_t3_visual_state import exact_reuse
from sea1a_panel import infer
from sea1b_panel import registration, verified_processor
from sea_task_193 import device_fields
from source_custody import SEEDS, digest, pin, read, require, write_new
from policy_query import input_identity, tensor_sha

PANEL = "SEA193-AUX-V2-PAIRED-02"
INITIAL_ATTEMPT = "SEA193-AUX-V2-PAIRED-01"
ACTUAL_SHA = "243dc623c1f09552a7254f82ebd03ce6e695ca07683a78f86a37a9f99605e637"
INGRESS_SHA = "249bd4ed05228afe2002bdb7ff1fc8bbfd76f020ff7d08eb256bc4101eb8c56e"
CONDITIONS = ("GG", "AUX2")
LAYOUTS = ("G00", "G01")
CAP = 4


def binding():
    scripts = Path(__file__).parent
    return {n: digest(scripts / n) for n in (
        "sea_mount_ab.py", "sea_mount_ab_score.py", "sea_mount_ingress.py", "sea_mount_capture.py",
        "sea_t3_visual_state.py", "sea_target_t2_execute.py", "sea_target_t2.py",
        "sea_task_193.py", "sea1a_panel.py", "sea1b_panel.py",
    )}


def inputs(root, prepare, processor):
    import jax

    old = read(pin(EVIDENCE / "VISUAL_SUPPORT/CAPTURE_INGRESS.json", INGRESS_SHA))
    require(old["capture_sha256"] == ACTUAL_SHA and tuple(old["rows"]) == SLOTS, "unexpected registered RGB custody")
    capture = root / "v19" / CAPTURE
    pin(capture / "CAPTURE_CLOSE.json", ACTUAL_SHA)
    # The ingress receipt uses CPU resize arithmetic. The device seam has a
    # separate exact comparison for the archived GPU inputs.
    with jax.default_device(jax.devices("cpu")[0]):
        rows, payloads = paired_payloads(root, capture, prepare, processor)
    require(rows == old["rows"] and processor == old["processor"], "changed-only V2 readback differs")
    fields = {name[:3]: dict(zip(CONDITIONS, (r["v1_fields"], r["v2_fields"]), strict=True)) for name, r in rows.items()}
    states = {p: float(payloads[p, "GG"]["state"][15]) for p in LAYOUTS}
    return payloads, fields, states


def zero_entry_closure(root):
    folder = root / "v19" / INITIAL_ATTEMPT
    owner, exit_path = read(folder / "START.json"), folder / "OWNER_EXIT.json"
    closed = read(exit_path)
    require(owner["panel"] == INITIAL_ATTEMPT and closed["exit_code"] != 0
            and closed["project_lock_released"] and closed["entered"] == closed["returned"] == 0
            and not (Path("/proc") / str(owner["owner_pid"])).exists()
            and not list(folder.glob("*.entered.json")) and not list(folder.glob("*.returned.json"))
            and not (folder / "calls.jsonl").exists(),
            "initial allocation has an unresolved or consumed sampler opportunity")
    return digest(exit_path)


def cells_for(payloads, noises):
    return [dict(id=f"{p}-{c}-seed{n}", anchor=p, condition=c, seed=seed,
                 input_sha256=input_identity(payloads[p, c])[0], initial_noise_sha256=tensor_sha(noises[n][2]))
            for n, seed in enumerate(SEEDS) for p in LAYOUTS for c in CONDITIONS]


def cold(root, output):
    import jax
    import jax.numpy as jnp
    from openpi.policies.policy import Policy

    require(jax.default_backend() == "cpu" and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU cold only")
    model, _ = registration()
    prepare, inverse, processor = verified_processor(root, model)
    payloads, fields, states = inputs(root, prepare, processor)
    noises = exact_noise(root)
    cells = cells_for(payloads, noises)
    flat = {(p, c): observation for p, v in fields.items() for c, observation in v.items()}
    reuse = exact_reuse(root, cells, flat, noises, processor, on_gpu=False)
    stub = SimpleNamespace(_input_transform=prepare, _output_transform=inverse, _sample_kwargs={},
                           _sample_actions=lambda _, __, **kw: jnp.zeros_like(kw["noise"]))
    stub.infer = Policy.infer.__get__(stub)
    entered = []
    for c in cells:
        key = c["anchor"], c["condition"]
        response, retained = infer(stub, payloads[key], flat[key], *noises[c["seed"][1]], entered.append)
        require(response["actions"].shape == (50, 16) and retained["actions"].shape == (50, 32),
                "official raw and decoded shape")
    require(len(entered) == 8 and len(reuse) == 4, "incomplete candidate CPU population")
    write_new(output, dict(
        status="AUX_V2_PAIRED_CPU_INPUT_VERIFIED", panel=PANEL, code=binding(),
        capture_sha256=ACTUAL_SHA, ingress_sha256=INGRESS_SHA, processor=processor,
        cells=cells, fields=fields, query_aperture=states, cpu_reuse_candidates=reuse,
        initial_zero_entry_exit_sha256=zero_entry_closure(root),
        max_new_forwards=CAP, model_forwards=0, native_starts=0, physics_ticks=0, optimizer_updates=0,
    ))


def worker(root, cold_path, descriptor):
    dest, proof, current = scoped_context(root, cold_path, descriptor, PANEL, PLAN_SHA)
    require(proof["status"] == "AUX_V2_PAIRED_CPU_INPUT_VERIFIED" and proof["code"] == binding(),
            "changed V2 CPU proof")
    model, _ = registration()
    prepare, _, processor = verified_processor(root, model)
    payloads, _, states = inputs(root, prepare, processor)
    noises = exact_noise(root)
    cells = cells_for(payloads, noises)
    require(cells == proof["cells"] and states == proof["query_aperture"] and processor == proof["processor"],
            "GPU input differs from fixed cold registry")
    device = device_fields(prepare, payloads, proof["fields"], set(CONDITIONS))
    reuse = exact_reuse(root, cells, device, noises, processor, on_gpu=True)
    require(reuse == proof["cpu_reuse_candidates"], "exact four-cell V1 reuse differs")
    fresh = [c for c in cells if c["id"] not in reuse]
    require(len(fresh) == CAP and all(c["condition"] == "AUX2" for c in fresh), "V2 sampling cap")
    write_new(dest / "REUSE.json", reuse)
    write_new(dest / "DEVICE_INPUTS.json", {"/".join(k): v for k, v in device.items()})
    run_sampler(root, dest, PANEL, current, fresh, payloads, device, noises)


def execute(root, cold_path, allocation_path, sha):
    proof = read(cold_path)
    grant = read(pin(allocation_path, sha))
    require(proof["status"] == "AUX_V2_PAIRED_CPU_INPUT_VERIFIED" and proof["panel"] == PANEL
            and proof["code"] == binding() and proof["capture_sha256"] == ACTUAL_SHA
            and len({c["id"] for c in proof["cells"]}) == 8
            and proof["max_new_forwards"] == CAP and proof["model_forwards"] == 0,
            "changed paired proof")
    require(grant["status"] == "APPROVED" and grant["panel"] == PANEL
            and bool(grant["authorization_id"]) and grant["max_forwards"] == CAP
            and grant["native_starts"] == grant["physics_ticks"] == grant["optimizer_updates"] == 0
            and grant["cold_sha256"] == digest(cold_path)
            and grant["initial_zero_entry_exit_sha256"] == proof["initial_zero_entry_exit_sha256"] == zero_entry_closure(root)
            and time.time() < grant["valid_until_epoch"],
            "missing independent V2 scientific allocation")
    run_owner(root, cold_path, grant, sha, PANEL, PLAN_SHA, Path(__file__), captured_aux_sha256=ACTUAL_SHA,
              initial_zero_entry_exit_sha256=proof["initial_zero_entry_exit_sha256"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    modes = p.add_subparsers(dest="mode", required=True)
    modes.add_parser("cold").add_argument("output", type=Path)
    runner = modes.add_parser("execute")
    runner.add_argument("cold", type=Path)
    runner.add_argument("allocation", type=Path)
    runner.add_argument("sha256")
    child = modes.add_parser("worker")
    child.add_argument("cold", type=Path)
    child.add_argument("descriptor", type=int)
    a = p.parse_args()
    os.umask(0o077)
    if a.mode == "cold":
        cold(a.root.resolve(), a.output)
    elif a.mode == "execute":
        execute(a.root.resolve(), a.cold.resolve(), a.allocation.resolve(), a.sha256)
    else:
        worker(a.root.resolve(), a.cold.resolve(), a.descriptor)
