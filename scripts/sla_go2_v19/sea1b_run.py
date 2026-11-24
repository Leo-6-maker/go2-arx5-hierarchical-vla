"""Execute SEA1B with a bounded GPU owner and fresh, paired initial noise."""

import argparse
import fcntl
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np

from sea1b_panel import (
    CONDITIONS, PANEL, PLAN_SHA, code_fingerprints, infer, packed, registration, rng_noise,
    source_inputs, verified_processor,
)
from sea1a_panel import UUID, frozen_predictor, gpu_lock, journal, site, verify_files, view_fields
from source_custody import digest, pin, read, require, write_new
from policy_query import input_identity, tensor_sha


def entrypoint_sha256():
    return digest(Path(__file__))


def expected_fields(prepare, observations, cold_fields):
    import jax
    import jax.numpy as jnp
    from openpi.models.model import Observation, preprocess_observation

    def on(value, platform):
        with jax.default_device(jax.devices(platform)[0]):
            batch = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], value)
            return preprocess_observation(None, Observation.from_dict(batch), train=False)

    bound, evidence = {}, []
    for name, args in observations.items():
        for condition in CONDITIONS:
            value = packed(prepare, args[condition in ("S2", "S3")], condition in ("S1", "S3"))
            a, b = on(value, "cpu"), on(value, "gpu")
            cpu, gpu = map(view_fields, (a, b))
            require(cpu == cold_fields[name][condition]
                    and all(cpu[k] == gpu[k] for k in ("current_masks", "token_ids", "token_mask", "state")),
                    "unexpected processor change at the GPU boundary")
            da, db = ({**q.images, "memory": q.extra_images} for q in (a, b))
            errors = {}
            for name_rgb in da:
                x, y = (np.asarray(arr[name_rgb]) for arr in (da, db))
                err = float(np.max(np.abs(x-y)))
                require(x.dtype == y.dtype == np.float32 and x.shape == y.shape
                        and err <= np.finfo(np.float32).eps
                        and np.array_equal(np.rint((x+1)*127.5), np.rint((y+1)*127.5)),
                        "GPU preprocessor changed RGB pixels")
                errors[name_rgb] = err
            bound[name, condition] = gpu
            evidence.append(dict(anchor=name, condition=condition, gpu=gpu, max_rgb_roundoff=errors))
    return bound, evidence


def live_owner(root, out, fd):
    path, _ = site(root)
    a, b = path.stat(), os.fstat(fd)
    require((a.st_dev, a.st_ino) == (b.st_dev, b.st_ino), "wrong project GPU lease")
    with path.open("rb") as probe:
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            raise ValueError("worker started without its live project lease")
    start = read(out / "START.json")

    def current():
        require(start["owner_pid"] == os.getppid() and start["panel"] == PANEL
                and start["registry_sha256"] == PLAN_SHA
                and time.time() < start["deadline_epoch"]
                and not any((out / f).exists() for f in ("STOP.json", "OWNER_EXIT.json")),
                "worker owner missing, expired or stopped")
    current()
    return current


def run_path(root):
    return root / "v19" / PANEL


def worker(root, fd, cold_path):
    import jax
    from p0_o0_runner import gpu_identity
    from XPolicyLab.policy.SLA_Memory.model import Model

    out = run_path(root)
    current = live_owner(root, out, fd)
    start = read(out / "START.json")
    cold = read(pin(cold_path, start["cold_sha256"]))
    require(cold["code"] == code_fingerprints() and start["entrypoint_sha256"] == entrypoint_sha256(),
            "execution entrypoint changed")
    frozen, panel = registration()
    observations, identity, virtual = source_inputs(root, panel)
    require(identity == cold["inputs"] and virtual == cold["fixed_virtual_state_sha256"],
            "different source payload from signed cold proof")
    require(jax.default_backend() == "gpu" and len(jax.local_devices()) == 1
            and gpu_identity() == UUID, "wrong frozen GPU")
    prepare, _, _ = verified_processor(root, frozen)
    input_fields, witness = expected_fields(prepare, observations, cold["effective_sampler_fields"])
    write_new(out / "DEVICE_INPUTS.json", witness)
    cfg = verify_files(root)
    require(digest(root / "evidence/SOURCE_AND_WEIGHT_MANIFEST.actual.json") == frozen["manifest_sha256"],
            "different official checkpoint inventory")
    current()
    log = out / "calls.jsonl"
    journal(log, "LOAD_ENTERED", panel=PANEL)
    model = Model(cfg)
    journal(log, "LOAD_RETURNED", jax_version=jax.__version__, gpu=UUID)
    noises = {tuple(seed): rng_noise(seed) for seed in panel["seeds"]}
    for seed, (_, key, noise) in noises.items():
        with (out / f"noise-{seed[1]}.npz").open("xb") as f:
            np.savez(f, noise=noise, sample_key=key)
    for row in panel["cells"]:
        name, condition, seed, slot = (row[k] for k in ("anchor", "condition", "seed", "id"))
        payload = observations[name][condition in ("S2", "S3")]
        rng, key, noise = noises[tuple(seed)]
        original = model._policy._input_transform
        model.reset()
        capture = {}
        now = time.monotonic()

        def entering(fields):
            current()
            write_new(out / f"{slot}.entered.json", dict(
                slot=slot, seed=seed, actual_initial_noise_sha256=tensor_sha(noise), sample_key=key.tolist(),
                fields=fields, model_input_sha256=input_identity(payload)[0],
            ))

        def retained(value):
            result, data = infer(model._policy, value, input_fields[name, condition], rng, key, noise, entering)
            capture.update(data)
            return result

        model._policy._input_transform = lambda obs: packed(original, obs, condition in ("S1", "S3"))
        try:
            output = frozen_predictor(model, payload, infer=retained)
        finally:
            model._policy._input_transform = original
        current()
        require(output.shape == (50, 16) and output.dtype == np.float32 and capture["actions"].shape == (50, 32)
                and all(np.isfinite(a).all() for a in (output, capture["actions"], capture["sampler_state"])),
                "invalid model result")
        path = out / f"{slot}.npz"
        with path.open("xb") as f:
            np.savez(f, absolute=output, **capture)
        write_new(out / f"{slot}.returned.json", dict(output_sha256=digest(path),
                                                     duration_s=time.monotonic()-now))
    journal(log, "PANEL_RETURNED", cells=len(panel["cells"]), max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def allocation(root, grant_path, grant_sha256, cold_path, now):
    _, panel = registration()
    grant = read(pin(grant_path, grant_sha256))
    cold = read(cold_path)
    require(grant.get("status") == "APPROVED" and grant.get("panel") == PANEL
            and grant.get("caps") == panel["cap"] and bool(grant.get("authorization_id"))
            and grant.get("registry_sha256") == PLAN_SHA and grant.get("cold_sha256") == digest(cold_path)
            and grant.get("entrypoint_sha256") == entrypoint_sha256()
            and np.isfinite(grant.get("valid_until_epoch", 0)) and now < grant.get("valid_until_epoch", 0),
            "missing matching live scientific allocation")
    require(cold.get("status") == "COLD_32_INPUT_SLOTS_VERIFIED" and cold.get("registry_sha256") == PLAN_SHA
            and cold.get("code") == code_fingerprints() and cold.get("cells") == [c["id"] for c in panel["cells"]]
            and cold.get("verified_exact_archival_reuse") == 0, "different cold proof")
    require(root.is_dir() and not run_path(root).exists(), "already used science identity")
    return grant


def execute(root, grant_path, grant_sha256, cold_path):
    grant = allocation(root, grant_path, grant_sha256, cold_path, time.time())
    require(os.environ.get("REACH_SHARED_GPU") == "1", "site-approved GPU sharing flag required")
    out = run_path(root)
    recorded_start, status, peak = False, None, 0
    opened = time.monotonic()
    try:
        with gpu_lock(root) as (lease, info):
            deadline = min(grant["valid_until_epoch"], time.time()+600)
            require(time.time() < deadline, "allocation expired before the lease")
            out.mkdir(mode=0o700)
            write_new(out / "START.json", dict(
                panel=PANEL, registry_sha256=PLAN_SHA, grant_sha256=grant_sha256, cold_sha256=digest(cold_path),
                entrypoint_sha256=entrypoint_sha256(), owner_pid=os.getpid(), deadline_epoch=deadline, site=info,
            ))
            recorded_start = True
            env = dict(
                os.environ, CUDA_VISIBLE_DEVICES="0", CUDA_DEVICE_ORDER="PCI_BUS_ID", JAX_PLATFORMS="cuda,cpu",
                XLA_PYTHON_CLIENT_PREALLOCATE="false", XLA_PYTHON_CLIENT_MEM_FRACTION="0.80",
                HF_HOME=str(root / "hf-cache"), XDG_CACHE_HOME=str(root / "runtime-cache"),
                OPENPI_DATA_HOME=str(root / "openpi-cache"),
                JAX_COMPILATION_CACHE_DIR=str(root / "jax-cache/P1_SOURCE_01"),
                PYTHONPATH=f"{root}:{root}/vendor", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
            )
            command = [sys.executable, "-B", str(Path(__file__).resolve()), str(root), "worker",
                       str(lease.fileno()), str(cold_path)]
            with (out / "worker.log").open("x") as sink:
                process = subprocess.Popen(command, cwd=root, env=env, pass_fds=(lease.fileno(),),
                                           stdout=sink, stderr=subprocess.STDOUT)
                try:
                    while process.poll() is None:
                        require(time.time() < deadline and not (out / "STOP.json").exists(), "deadline or STOP")
                        usage = subprocess.check_output(
                            ["nvidia-smi", "-i", "0", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                            timeout=10,
                        )
                        peak = max(peak, int(usage))
                        time.sleep(.5)
                finally:
                    if process.poll() is None:
                        process.kill()
                    status = process.wait()
    finally:
        if recorded_start:
            write_new(out / "OWNER_EXIT.json", dict(
                exit_code=status, sampled_board_peak_mib=peak, lease_wall_s=time.monotonic()-opened,
                project_lock_released=True, physics_ticks=0, training_updates=0,
                entered=len(list(out.glob("*.entered.json"))), returned=len(list(out.glob("*.returned.json"))),
            ))
    require(status == 0, "worker stopped; keep the fixed roster and all evidence")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("root", type=Path)
    modes = cli.add_subparsers(dest="mode", required=True)
    child = modes.add_parser("worker")
    child.add_argument("fd", type=int)
    child.add_argument("cold", type=Path)
    parent = modes.add_parser("execute")
    parent.add_argument("grant", type=Path)
    parent.add_argument("grant_sha256")
    parent.add_argument("cold", type=Path)
    args = cli.parse_args()
    os.umask(0o077)
    if args.mode == "worker":
        worker(args.root.resolve(), args.fd, args.cold.resolve())
    else:
        execute(args.root.resolve(), args.grant.resolve(), args.grant_sha256, args.cold.resolve())
