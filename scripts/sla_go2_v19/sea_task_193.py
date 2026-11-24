"""Run the separately allocated V1.9.3 source-language panels, without physics."""

import argparse
import fcntl
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from types import SimpleNamespace

import h5py
import numpy as np

from sea1a_panel import UUID, frozen_predictor, gpu_lock, infer, journal, site, verify_files, view_fields
from sea1b_panel import discrete_token_reader, fields_at_device, packed, registration, rng_noise, verified_processor, verify_discrete_prompt
from sea_task_source import ORIGINAL, PARAPHRASE, payload
from source_custody import BASE, digest, pin, read, require, write_new
from policy_query import input_identity, tensor_sha

PLAN = BASE / "docs/protocols/SLA_GO2_V193_THREE_VIEW_FIRST_REGISTRY_20261124.json"
PLAN_SHA = "5161e08283bb2dbb25c13776217c61f17f6e891e422b3e2a59575e4dc900bbbb"
REPORT = BASE / "reports/SLA_GO2_V192"
SOURCE_SHA = "f0beb769430a15b240c601d6fbb4039c5fc3e0e93169c71ffe14a21c7fbc0fed"
OLD_ACTUAL_SHA = "7cb5e38c69318f0d85a927b4aef3db340af1b96619a77cbb02659e7ccaadf1b9"
OLD_CUSTODY_SHA = "bef73a7d551ad364afd4ef8b2ac150195153f33ae1b724c3edb1846fe989978c"
LOCAL = "Pick up the fruit, lift it above the table and keep holding it."
NAMES = {"T0T1": "SEA-TASK-T0T1-193-01", "TL": "SEA-TASK-TL-193-01"}


def load_roster(root):
    protocol = read(pin(PLAN, PLAN_SHA))
    cap = next(s["cap"] for s in protocol["stages"] if s["id"] == "SOURCE_TASK")
    require(cap == dict(t0_t1=16, tl_separate=8, exact_reuse_only=True), "wrong SEA-TASK cap")
    reg = read(pin(REPORT / "SEA_TASK/REGISTRATION.json", SOURCE_SHA))
    require(reg["prompts"] == {"T0": ORIGINAL, "T1": PARAPHRASE}
            and [a["frame"] for a in reg["anchors"]] == [252, 266, 263, 276], "old SEA-TASK population changed")
    obs = {}
    for a in reg["anchors"]:
        with h5py.File(pin(root / "data" / a["episode"], a["source_sha256"])) as h:
            require(h["instruction"].asstr()[()] == ORIGINAL and int(h["additional_info/frequency"][()]) == 30,
                    "source instruction or source clock changed")
            raw = payload(h, a["frame"], ORIGINAL)
        for label, text in (("T0", ORIGINAL), ("T1", PARAPHRASE), ("TL", LOCAL)):
            obs[a["anchor"], label] = {**raw, "prompt": text}
    cells = {}
    for batch, conditions in (("T0T1", ("T0", "T1")), ("TL", ("TL",))):
        cells[batch] = [dict(anchor=a["anchor"], seed=[0,n], condition=c, input_sha256=input_identity(obs[a["anchor"],c])[0])
                        for n in (0, 1) for a in reg["anchors"] for c in conditions]
    require(cells["T0T1"] == reg["t0_t1_cells"], "input bytes in the fixed 16-cell panel changed")
    return reg, obs, cells


def case_id(cell):
    return f"{cell['anchor']}-{cell['condition']}-seed{cell['seed'][1]}"


def code_binding():
    local = Path(__file__).parent
    return {name: digest(local / name) for name in (
        "sea_task_193.py", "sea_task_source.py", "sea_task_193_score.py", "sea1a_panel.py",
        "sea1b_panel.py", "sea1b_run.py", "sea1b_score.py", "sea1_archive_readout.py", "source_custody.py",
    )}


def cold(root, output):
    import jax
    import jax.numpy as jnp
    from openpi.policies.policy import Policy

    require(jax.default_backend() == "cpu" and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU cold only")
    _, obs, cells = load_roster(root)
    frozen, _ = registration()
    prepare, inverse, identity = verified_processor(root, frozen)
    decoder = discrete_token_reader(root)
    prepared = {key: packed(prepare, o, False) for key, o in obs.items()}
    known = {a: {c: fields_at_device(prepared[a,c]) for c in ("T0", "T1", "TL")} for a, _ in obs}
    for (a,c), value in prepared.items():
        verify_discrete_prompt(decoder, value, obs[a,c]["prompt"])
        reference, trial = known[a]["T0"], known[a][c]
        require(all(reference[k] == trial[k] for k in ("current", "memory", "current_masks", "state"))
                and all(trial["current_masks"].values())
                and (reference["token_ids"] == trial["token_ids"]) == (c == "T0"),
                "language condition leaked outside the instruction")
    stub = SimpleNamespace(_input_transform=prepare, _output_transform=inverse, _sample_kwargs={},
                           _sample_actions=lambda _, __, **kw: jnp.zeros_like(kw["noise"]))
    stub.infer = Policy.infer.__get__(stub)
    witness = []
    noise = {n: rng_noise([0,n]) for n in (0,1)}
    for rows in cells.values():
        for row in rows:
            a,c = row["anchor"],row["condition"]
            decoded, captured = infer(stub, obs[a,c], known[a][c], *noise[row["seed"][1]], witness.append)
            require(decoded["actions"].shape == (50,16) and captured["actions"].shape == (50,32),
                    "wrong cold inverse/forward seam")
    require(len(witness) == 24, "incomplete cold panels")
    write_new(output, dict(
        status="COLD_T0T1_AND_TL_VERIFIED", registry_sha256=PLAN_SHA, source_registry_sha256=SOURCE_SHA,
        processor=identity, code=code_binding(), cells=cells, per_payload_fields=known,
        old_reuse_candidates=["FRUIT03-T0-seed0", "FRUIT03-T0-seed1"],
        learned_model_forwards=0, physics_ticks=0, optimizer_updates=0,
    ))


def device_fields(prepare, obs, cold_fields, conditions):
    import jax
    import jax.numpy as jnp
    from openpi.models.model import Observation, preprocess_observation

    def at(value, platform):
        with jax.default_device(jax.devices(platform)[0]):
            batch = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], value)
            return preprocess_observation(None, Observation.from_dict(batch), train=False)

    fields = {}
    for (a,c), o in obs.items():
        if c not in conditions:
            continue
        value = packed(prepare, o, False)
        cpu, gpu = (at(value,p) for p in ("cpu","gpu"))
        ca, ga = map(view_fields, (cpu,gpu))
        require(ca == cold_fields[a][c]
                and all(ca[k] == ga[k] for k in ("current_masks","token_ids","token_mask","state")),
                "GPU observations do not match the cold proof")
        for key in (*cpu.images, "memory"):
            x,y = (np.asarray(v.extra_images if key == "memory" else v.images[key]) for v in (cpu,gpu))
            require(np.max(np.abs(x-y)) <= np.finfo(np.float32).eps
                    and np.array_equal(np.rint((x+1)*127.5), np.rint((y+1)*127.5)),
                    "GPU generated different RGB pixels")
        fields[a,c] = ga
    return fields


def verified_reuse(root, cells, fields, noises, processor, current_jax):
    old = root / "v19/SEA1B-SOURCE-FACTORIAL-01"
    receipt = read(pin(REPORT / "SEA1B/CUSTODY.json", OLD_CUSTODY_SHA))
    cold = read(pin(REPORT / "SEA1B/REGISTRATION.json", receipt["cold_sha256"]))
    score = read(pin(REPORT / "SEA1B/ACTUAL.json", OLD_ACTUAL_SHA))
    require(receipt["score_sha256"] == OLD_ACTUAL_SHA and receipt["jax_version"] == current_jax
            and receipt["gpu_uuid"] == UUID and cold["processor"] == processor,
            "no exact same sampler environment for historical reuse")
    require(receipt["entrypoint_sha256"] == digest(Path(__file__).with_name("sea1b_run.py"))
            and cold["code"]["scripts/sla_go2_v19/sea1a_panel.py"] == digest(Path(__file__).with_name("sea1a_panel.py")),
            "historical sampler or retained inference changed")
    require(read(pin(old / "OWNER_EXIT.json", receipt["seals"]["OWNER_EXIT.json"]))["project_lock_released"],
            "historical GPU owner not closed")
    archived = {r["slot"]: r for r in score["results"]}
    reuse = {}
    for row in cells:
        if (row["anchor"],row["condition"]) != ("FRUIT03","T0"):
            continue
        n = row["seed"][1]
        key = f"SEA1B-S03-S0-seed{n}"
        a = archived[key]
        e = read(old / f"{key}.entered.json")
        _, split, noise = noises[n]
        try:
            with np.load(old / f"noise-{n}.npz", allow_pickle=False) as z:
                require(np.array_equal(z["noise"],noise) and np.array_equal(z["sample_key"],split)
                        and tensor_sha(noise) == e["actual_initial_noise_sha256"] == a["actual_initial_noise_sha256"],
                        "stored actual noise tensor differs")
            require(e["model_input_sha256"] == row["input_sha256"] and e["seed"] == a["seed"] == row["seed"]
                    and e["fields"] == a["fields"] == fields["FRUIT03","T0"],
                    "archived input or sampler fields differ")
            result = pin(old / f"{key}.npz", a["output_sha256"])
        except (OSError, ValueError):
            continue
        reuse[case_id(row)] = dict(archived_slot=key, output_sha256=digest(result),
                                  noise_sha256=a["actual_initial_noise_sha256"])
    return reuse


def scoped_worker(root, batch, cold_path, fd):
    path, _ = site(root)
    a,b = path.stat(),os.fstat(fd)
    require((a.st_dev,a.st_ino) == (b.st_dev,b.st_ino), "wrong project lock")
    with path.open("rb") as probe:
        try:
            fcntl.flock(probe,fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            raise ValueError("missing live project GPU lock")
    dest = root / "v19" / NAMES[batch]
    start = read(dest / "START.json")

    def current():
        require(start["owner_pid"] == os.getppid() and start["run"] == NAMES[batch]
                and start["plan_sha"] == PLAN_SHA and time.time() < start["deadline"]
                and not any((dest / f).exists() for f in ("STOP.json","OWNER_EXIT.json")),
                "stopped or expired experiment owner")
    current()
    worker(root, batch, dest, read(pin(cold_path,start["cold_sha256"])), current)


def worker(root, batch, dest, cold, current):
    import jax
    from p0_o0_runner import gpu_identity
    from XPolicyLab.policy.SLA_Memory.model import Model

    require(jax.default_backend() == "gpu" and len(jax.local_devices()) == 1 and gpu_identity() == UUID,
            "wrong GPU execution device")
    _, obs, batches = load_roster(root)
    cells = batches[batch]
    require(cold["code"] == code_binding() and cold["cells"] == batches, "execute binding changed after cold proof")
    frozen, _ = registration()
    prepare, _, identity = verified_processor(root,frozen)
    fields = device_fields(prepare,obs,cold["per_payload_fields"],{r["condition"] for r in cells})
    write_new(dest / "DEVICE_INPUTS.json", {"/".join(k):v for k,v in fields.items()})
    noises = {n: rng_noise([0,n]) for n in (0,1)}
    reuse, reuse_reason = {}, None
    if batch == "T0T1":
        try:
            reuse = verified_reuse(root,cells,fields,noises,identity,jax.__version__)
        except (OSError, ValueError) as exc:
            reuse_reason = str(exc)
    write_new(dest / "REUSE.json", dict(exact=reuse, refused_global=reuse_reason))
    for n, (_, split, noise) in noises.items():
        with (dest / f"noise-{n}.npz").open("xb") as f:
            np.savez(f,noise=noise,sample_key=split)
    cfg = verify_files(root)
    current()
    log = dest / "calls.jsonl"
    journal(log,"LOAD_ENTERED",batch=batch)
    model = Model(cfg)
    journal(log,"LOAD_RETURNED",jax_version=jax.__version__,gpu=UUID)
    for row in cells:
        current()
        name,c,n = row["anchor"],row["condition"],row["seed"][1]
        slot = case_id(row)
        if slot in reuse:
            journal(log,"EXACT_REUSE",slot=slot,**reuse[slot])
            continue
        model.reset()
        rng,key,noise = noises[n]
        capture = {}
        then = time.monotonic()

        def entering(seen):
            current()
            write_new(dest / f"{slot}.entered.json", dict(
                **row, slot=slot, sample_key=key.tolist(), noise_sha256=tensor_sha(noise), fields=seen,
            ))

        def retained(o):
            response, data = infer(model._policy,o,fields[name,c],rng,key,noise,entering)
            capture.update(data)
            return response

        result = frozen_predictor(model,obs[name,c],infer=retained)
        current()
        require(result.shape == (50,16) and result.dtype == np.float32 and capture["actions"].shape == (50,32)
                and all(np.isfinite(x).all() for x in (result,capture["actions"],capture["sampler_state"])),
                "model output violates official shape or finite check")
        out = dest / f"{slot}.npz"
        with out.open("xb") as f:
            np.savez(f,absolute=result,**capture)
        write_new(dest / f"{slot}.returned.json", dict(file_sha256=digest(out),duration_s=time.monotonic()-then))
    journal(log,"PANEL_RETURNED",registered=len(cells),samples_excluding_reuse=len(cells)-len(reuse),
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def execute(root, batch, cold_path, allocation_path, allocation_sha):
    g = read(pin(allocation_path,allocation_sha))
    cold = read(cold_path)
    require(g["status"] == "APPROVED" and bool(g["authorization_id"]) and g["batch"] == batch
            and g["run"] == NAMES[batch] and g["registered_cap"] == (16 if batch == "T0T1" else 8)
            and g["physics_ticks"] == g["optimizer_updates"] == 0
            and g["cold_sha256"] == digest(cold_path) and g["plan_sha"] == PLAN_SHA
            and g["valid_until_epoch"] > time.time()
            and cold["status"] == "COLD_T0T1_AND_TL_VERIFIED" and cold["code"] == code_binding(),
            "invalid allocation, deadline, or cold proof")
    require(os.environ.get("REACH_SHARED_GPU") == "1", "approved share mode required")
    out = root / "v19" / NAMES[batch]
    started = False
    code, peak = None,0
    clock = time.monotonic()
    try:
        with gpu_lock(root) as (fd, info):
            deadline = min(g["valid_until_epoch"],time.time()+600)
            out.mkdir(mode=0o700)
            write_new(out / "START.json", dict(run=NAMES[batch], owner_pid=os.getpid(), deadline=deadline,
                                              plan_sha=PLAN_SHA, cold_sha256=digest(cold_path), allocation_sha=allocation_sha, site=info))
            started = True
            env = dict(
                os.environ, CUDA_VISIBLE_DEVICES="0", CUDA_DEVICE_ORDER="PCI_BUS_ID", JAX_PLATFORMS="cuda,cpu",
                XLA_PYTHON_CLIENT_PREALLOCATE="false", XLA_PYTHON_CLIENT_MEM_FRACTION="0.80",
                HF_HOME=str(root / "hf-cache"), XDG_CACHE_HOME=str(root / "runtime-cache"),
                OPENPI_DATA_HOME=str(root / "openpi-cache"), JAX_COMPILATION_CACHE_DIR=str(root / "jax-cache/P1_SOURCE_01"),
                PYTHONPATH=f"{root}:{root}/vendor", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
            )
            command = [sys.executable,"-B",str(Path(__file__).resolve()),str(root),"worker",batch,str(cold_path),str(fd.fileno())]
            with (out / "worker.log").open("x") as sink:
                p = subprocess.Popen(command,pass_fds=(fd.fileno(),),cwd=root,env=env,stdout=sink,stderr=subprocess.STDOUT)
                try:
                    while p.poll() is None:
                        require(time.time() < deadline and not (out / "STOP.json").exists(),"deadline or STOP")
                        use = subprocess.check_output(["nvidia-smi","-i","0","--query-gpu=memory.used","--format=csv,noheader,nounits"],timeout=10)
                        peak = max(peak,int(use))
                        time.sleep(.5)
                finally:
                    if p.poll() is None:
                        p.kill()
                    code = p.wait()
    finally:
        if started:
            write_new(out / "OWNER_EXIT.json", dict(
                exit_code=code, project_lock_released=True, sampled_board_peak_mib=peak, lease_wall_s=time.monotonic()-clock,
                entered=len(list(out.glob("*.entered.json"))), returned=len(list(out.glob("*.returned.json"))),
                physics_ticks=0, optimizer_updates=0,
            ))
    require(code == 0, "preserve the incomplete panel and inspect the worker")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("root",type=Path)
    mode = cli.add_subparsers(dest="mode",required=True)
    mode.add_parser("cold").add_argument("out",type=Path)
    for action in ("execute","worker"):
        cmd = mode.add_parser(action)
        cmd.add_argument("batch",choices=NAMES)
        cmd.add_argument("cold",type=Path)
        if action == "execute":
            cmd.add_argument("allocation",type=Path)
            cmd.add_argument("allocation_sha")
        else:
            cmd.add_argument("fd",type=int)
    args = cli.parse_args()
    os.umask(0o077)
    if args.mode == "cold":
        cold(args.root.resolve(),args.out)
    elif args.mode == "execute":
        execute(args.root.resolve(),args.batch,args.cold,args.allocation,args.allocation_sha)
    else:
        scoped_worker(args.root.resolve(),args.batch,args.cold,args.fd)
