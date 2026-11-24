"""Execute the separately allocated eight-cell T2 diagnostic without a native owner."""

import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from sea_target_t2 import CONDITION, PANEL, PLAN_SHA, anchors_and_payloads, cell_list, code_binding, exact_noise
from sea1a_panel import UUID, frozen_predictor, gpu_lock, infer, journal, site, verify_files
from sea1b_panel import registration, verified_processor
from sea_task_193 import device_fields
from source_custody import BASE, digest, pin, read, require, write_new
from policy_query import tensor_sha


def check_proof(cold, control, cold_path):
    require(cold["status"] == "T2_3V_CPU_COLD_VERIFIED" and cold["panel"] == PANEL
            and cold["source_plan_sha256"] == PLAN_SHA and cold["code"] == code_binding()
            and cold["frozen_model_forwards"] == cold["physics_ticks"] == cold["optimizer_updates"] == 0
            and len(cold["cells"]) == len({c["id"] for c in cold["cells"]}) == 8
            and control["status"] == "FOUR_ACTUAL_H0_PATHS_ARCHIVED"
            and control["cold_sha256"] == digest(cold_path) and set(control["rows"]) == set(cold["rows"]),
            "cold T2 or teacher scoring proof differs")


def scoped_context(root, cold_path, descriptor, panel, plan_sha):
    import jax
    from p0_o0_runner import gpu_identity

    lock, _ = site(root)
    info, inherited = lock.stat(), os.fstat(descriptor)
    require((info.st_dev, info.st_ino) == (inherited.st_dev, inherited.st_ino), "wrong GPU lease")
    with lock.open("rb") as probe:
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            raise ValueError("missing active GPU project lock")
    dest = root / "v19" / panel
    start = read(dest / "START.json")

    def current():
        require(start["owner_pid"] == os.getppid() and start["panel"] == panel and start["plan_sha256"] == plan_sha
                and time.time() < start["deadline_epoch"]
                and not any((dest / f).exists() for f in ("STOP.json", "OWNER_EXIT.json")),
                "expired, stopped or foreign frozen panel owner")

    current()
    require(jax.default_backend() == "gpu" and len(jax.local_devices()) == 1 and gpu_identity() == UUID,
            "wrong frozen model device")
    return dest, read(pin(cold_path, start["cold_sha256"])), current


def run_sampler(root, dest, panel, current, cells, payloads, fields, noises):
    import jax
    from XPolicyLab.policy.SLA_Memory.model import Model

    for n, (_, key, noise) in noises.items():
        with (dest / f"noise-{n}.npz").open("xb") as out:
            np.savez(out, noise=noise, sample_key=key)
    config = verify_files(root)
    current()
    log = dest / "calls.jsonl"
    journal(log, "LOAD_ENTERED", panel=panel)
    model = Model(config)
    journal(log, "LOAD_RETURNED", jax_version=jax.__version__, gpu=UUID)
    for cell in cells:
        current()
        name, anchor, condition, seed = cell["id"], cell["anchor"], cell["condition"], cell["seed"][1]
        rng, key, noise = noises[seed]
        model.reset()
        capture = {}
        before = time.monotonic()

        def entering(seen):
            current()
            write_new(dest / f"{name}.entered.json", dict(
                **cell, sample_key=key.tolist(), fields=seen,
            ))

        def retained(observation):
            answer, record = infer(model._policy, observation, fields[anchor, condition], rng, key, noise, entering)
            capture.update(record)
            return answer

        absolute = frozen_predictor(model, payloads[anchor, condition], infer=retained)
        current()
        require(absolute.shape == (50, 16) and absolute.dtype == np.float32 and capture["actions"].shape == (50, 32)
                and all(np.isfinite(x).all() for x in (absolute, capture["actions"], capture["sampler_state"])),
                "frozen panel output ABI")
        path = dest / f"{name}.npz"
        with path.open("xb") as writer:
            np.savez(writer, absolute=absolute, **capture)
        write_new(dest / f"{name}.returned.json", dict(
            file_sha256=digest(path), duration_s=time.monotonic() - before,
            raw_sha256=tensor_sha(capture["actions"]), decoded_sha256=tensor_sha(absolute),
        ))
    journal(log, "PANEL_RETURNED", sampler_enters=len(cells))


def worker(root, cold_path, descriptor):
    dest, cold, current = scoped_context(root, cold_path, descriptor, PANEL, PLAN_SHA)
    require(cold["code"] == code_binding(), "T2 execution changed after preflight")
    frozen, _ = registration()
    prepare, _, processor = verified_processor(root, frozen)
    rows, payloads, _ = anchors_and_payloads(root, prepare)
    noises = exact_noise(root)
    require(rows == cold["rows"] and cell_list(rows, noises) == cold["cells"]
            and processor == cold["processor"], "cold input bytes differ at the GPU")
    fields = device_fields(prepare, payloads, cold["fields"], {CONDITION})
    write_new(dest / "DEVICE_INPUTS.json", {name: fields[name, CONDITION] for name in rows})
    run_sampler(root, dest, PANEL, current, cold["cells"], payloads, fields, noises)


def execute(root, cold_path, control_path, allocation_path, allocation_sha):
    allocation = read(pin(allocation_path, allocation_sha))
    cold, control = read(cold_path), read(control_path)
    check_proof(cold, control, cold_path)
    require(allocation["panel"] == PANEL and allocation["status"] == "APPROVED"
            and bool(allocation["authorization_id"]) and allocation["max_forwards"] == 8
            and allocation["native_starts"] == allocation["physics_ticks"] == allocation["optimizer_updates"] == 0
            and allocation["cold_sha256"] == digest(cold_path) and allocation["teacher_sha256"] == digest(control_path)
            and allocation["valid_until_epoch"] > time.time(), "missing exact T2 scientific allocation")
    run_owner(root, cold_path, allocation, allocation_sha, PANEL, PLAN_SHA, Path(__file__),
              teacher_sha256=digest(control_path))


def run_owner(root, cold_path, allocation, allocation_sha, panel, plan_sha, entrypoint, **binding):
    require(os.environ.get("REACH_SHARED_GPU") == "1", "approved GPU share mode required")
    dest = root / "v19" / panel
    code, peak, started = None, 0, False
    wall_start = time.monotonic()
    try:
        with gpu_lock(root) as (lock, gpu):
            deadline = min(allocation["valid_until_epoch"], time.time() + 600)
            dest.mkdir(mode=0o700)
            write_new(dest / "START.json", dict(
                panel=panel, plan_sha256=plan_sha, deadline_epoch=deadline, owner_pid=os.getpid(),
                allocation_sha256=allocation_sha, cold_sha256=digest(cold_path),
                gpu=gpu, **binding,
            ))
            started = True
            dependencies = [BASE / "scripts" / n for n in
                            ("sla_transfer_v14", "sla_transfer_v15", "sla_transfer_feedback")]
            env = dict(
                os.environ, CUDA_VISIBLE_DEVICES="0", CUDA_DEVICE_ORDER="PCI_BUS_ID", JAX_PLATFORMS="cuda,cpu",
                XLA_PYTHON_CLIENT_PREALLOCATE="false", XLA_PYTHON_CLIENT_MEM_FRACTION="0.80",
                HF_HOME=str(root / "hf-cache"), XDG_CACHE_HOME=str(root / "runtime-cache"),
                OPENPI_DATA_HOME=str(root / "openpi-cache"), JAX_COMPILATION_CACHE_DIR=str(root / "jax-cache/P1_SOURCE_01"),
                PYTHONPATH=os.pathsep.join(map(str, [root, root / "vendor", *dependencies])),
                OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
            )
            command = [sys.executable, "-B", str(entrypoint.resolve()), str(root), "worker",
                       str(cold_path), str(lock.fileno())]
            with (dest / "worker.log").open("x") as log:
                process = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                                           pass_fds=(lock.fileno(),))
                try:
                    while process.poll() is None:
                        require(time.time() < deadline and not (dest / "STOP.json").exists(), "deadline or STOP")
                        usage = subprocess.check_output(
                            ["nvidia-smi", "-i", "0", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                            timeout=10,
                        )
                        peak = max(peak, int(usage))
                        time.sleep(.5)
                finally:
                    if process.poll() is None:
                        process.kill()
                    code = process.wait()
    finally:
        if started:
            write_new(dest / "OWNER_EXIT.json", dict(
                exit_code=code, project_lock_released=True, sampled_board_peak_mib=peak,
                lease_wall_s=time.monotonic() - wall_start, entered=len(list(dest.glob("*.entered.json"))),
                returned=len(list(dest.glob("*.returned.json"))), physics_ticks=0, optimizer_updates=0,
            ))
    require(code == 0, "frozen panel incomplete, preserve its actual allocation denominator")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    modes = parser.add_subparsers(dest="mode", required=True)
    owner = modes.add_parser("execute")
    for label in ("cold", "teacher", "allocation"):
        owner.add_argument(label, type=Path)
    owner.add_argument("allocation_sha")
    child = modes.add_parser("worker")
    child.add_argument("cold", type=Path)
    child.add_argument("descriptor", type=int)
    a = parser.parse_args()
    os.umask(0o077)
    if a.mode == "execute":
        execute(a.root.resolve(), a.cold.resolve(), a.teacher.resolve(), a.allocation.resolve(), a.allocation_sha)
    else:
        worker(a.root.resolve(), a.cold.resolve(), a.descriptor)
