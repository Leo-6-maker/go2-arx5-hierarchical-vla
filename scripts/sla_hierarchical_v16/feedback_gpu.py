"""Site owner entry point for the prospective six-return V4 offline panel."""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET

from feedback_policy_panel import REG, inputs, pin, read, require, write_new

UUID = "GPU-fef928e1-b12e-2f20-4b2e-a868e6e0faad"
ALLOWED = {
    ("1600648", "G", "/usr/lib/xorg/Xorg"),
    ("1600946", "G", "/usr/bin/gnome-shell"),
}
RECEIVER = Path(
    "/var/tmp/sla-v6-core-uid2009-owned-20261118/sla-main3800-core-matched-native-dev6-20261122-01/RECEIVER.json"
)
LIMIT_SECONDS = 1800


def journal(path, event, **fields):
    with path.open("a") as f:
        f.write(
            json.dumps(dict(event=event, time_s=time.time(), **fields), allow_nan=False)
            + "\n"
        )
        f.flush()
        os.fsync(f.fileno())


def field_mib(value):
    a, units = value.split()
    require(units == "MiB", "unexpected memory unit")
    return int(a)


def site(root):
    require(socket.gethostname().split(".")[0] == "corelab-4080s", "wrong host")
    cfg = read(RECEIVER)
    require(cfg["gpu"] == UUID, "GPU0 owner mismatch")
    lock = Path(cfg["gpu_lock"]).resolve(strict=True)
    gpu = ET.fromstring(
        subprocess.check_output(["nvidia-smi", "-i", "0", "-q", "-x"])
    ).find("gpu")
    require(gpu.findtext("uuid") == UUID, "wrong GPU UUID")
    seen = {
        (p.findtext("pid"), p.findtext("type"), p.findtext("process_name"))
        for p in gpu.findall("processes/process_info")
    }
    require(seen <= ALLOWED, "GPU0 has additional users")
    ram = int(
        next(
            line.split()[1]
            for line in Path("/proc/meminfo").read_text().splitlines()
            if line.startswith("MemAvailable:")
        )
    )
    prior = read(root / "v12/FORWARD_RESOURCES.json")
    board = field_mib(gpu.findtext("fb_memory_usage/free"))
    require(
        board >= max(prior[k]["board_peak_mib"] for k in ("O2", "P2_EXTRA"))
        and ram >= max(prior[k]["resident_peak_kib"] for k in ("O2", "P2_EXTRA")),
        "below prior measured resources",
    )
    return lock, dict(
        gpu=UUID,
        allowed_desktops_observed=len(seen),
        board_free_mib=board,
        ram_available_kib=ram,
    )


def verify_files(root):
    manifest = read(root / "evidence/SOURCE_AND_WEIGHT_MANIFEST.actual.json")
    require(
        manifest["model_revision"] == "7f6880569c3be777291e646fad1b4951245e8553"
        and manifest["source_commit"] == "c1b205ec42b4ab2cac9c2159c8b4c2f723e5e454",
        "wrong frozen model",
    )
    vendor = root / "vendor/XPolicyLab"
    for r in manifest["source_files"]:
        pin(vendor / r["path"], r["sha256"])
    pieces = manifest["verified_checkpoint_files"]
    require(
        len(pieces) == 19 and sum(r["bytes"] for r in pieces) == 12441551063,
        "wrong checkpoint population",
    )
    for r in pieces:
        pin(
            vendor / "policy/SLA_Memory/checkpoints" / r["path"],
            r["local_content_sha256"],
        )
    pin(Path(manifest["tokenizer"]["path"]), manifest["tokenizer"]["sha256"])
    return read(
        pin(
            root / "evidence/P0_EFFECTIVE_CONFIG.json",
            manifest["effective_config_sha256"],
        )
    )


def retained_inference(policy, obs, capture):
    import numpy as np

    original = policy._output_transform

    def retain(result):
        require(
            not capture and set(result) >= {"state", "actions"},
            "unexpected sampler result",
        )
        for key, name in (("actions", "actions"), ("state", "sampler_state")):
            a = np.asarray(result[key])
            capture[name + "_dtype"] = np.array(a.dtype.name)
            capture[name] = (
                a.astype(np.float32) if a.dtype.name == "bfloat16" else a.copy()
            )
        return original(result)

    policy._output_transform = retain
    try:
        return policy.infer(obs)
    finally:
        policy._output_transform = original


def run_model(root, out):
    import numpy as np
    import jax
    import jax.numpy as jnp
    from p0_o0_runner import gpu_identity
    from XPolicyLab.policy.SLA_Memory.model import Model
    from policy_query import frozen_predictor

    roster = inputs(root)
    cfg = verify_files(root)
    require(
        jax.default_backend() == "gpu"
        and len(jax.local_devices()) == 1
        and gpu_identity() == UUID,
        "unexpected compute device",
    )
    log = out / "calls.jsonl"
    journal(
        log,
        "LOAD_ENTERED",
        registration_sha256=hashlib.sha256(REG.read_bytes()).hexdigest(),
    )
    model = Model(cfg)
    journal(log, "LOAD_RETURNED", device=UUID)
    seed = read(REG)["seed"]
    for key, (obs, info) in roster.items():
        model.reset()
        model._policy._rng = jax.random.wrap_key_data(
            jnp.asarray(seed, dtype=jnp.uint32)
        )
        capture = {}
        then = time.monotonic()
        journal(
            log, "FORWARD_ENTERED", id=key, input_sha256=info["input_sha256"], seed=seed
        )
        try:
            absolute = frozen_predictor(
                model,
                obs,
                infer=lambda value: retained_inference(model._policy, value, capture),
            )
            require(
                absolute.shape == (50, 16)
                and capture["actions"].shape[0] == 50
                and all(
                    np.isfinite(a).all()
                    for a in (absolute, capture["actions"], capture["sampler_state"])
                ),
                "malformed prediction",
            )
            path = out / (key + ".npz")
            with path.open("xb") as f:
                np.savez(f, absolute=absolute, **capture)
            journal(
                log,
                "FORWARD_RETURNED",
                id=key,
                seed=seed,
                input_sha256=info["input_sha256"],
                file=path.name,
                file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                seconds=time.monotonic() - then,
            )
        except BaseException:
            if capture:
                with (out / (key + "_partial.npz")).open("xb") as f:
                    np.savez(f, **capture)
            journal(log, "FORWARD_FAILED", id=key, error=traceback.format_exc())
            raise
    journal(
        log,
        "PANEL_RETURNED",
        count=len(roster),
        native_ticks=0,
        training_updates=0,
        resident_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("mode", choices=("check", "execute", "worker"))
    args = parser.parse_args()
    root = args.root.resolve()
    out = root / "v16/V4-FEEDBACK-01"
    lock, info = site(root) if args.mode != "worker" else (None, None)
    if args.mode == "worker":
        # This process inherits the canonical lease and a separate watchdog from the owner.
        confirmed = read(RECEIVER)
        expected = Path(confirmed["gpu_lock"]).resolve(strict=True)
        fd = int(os.environ["V4_FEEDBACK_LOCK_FD"])
        st = os.fstat(fd)
        st2 = expected.stat()
        require(
            (st.st_dev, st.st_ino) == (st2.st_dev, st2.st_ino)
            and read(out / "START.json")["pid"]
            == int(os.environ["V4_FEEDBACK_OWNER_PID"]),
            "missing owner lease",
        )
        start = read(out / "START.json")
        require(
            not (out / "OWNER_EXIT.json").exists()
            and time.time() < start["deadline_epoch"]
            and start["registration_sha256"]
            == hashlib.sha256(REG.read_bytes()).hexdigest(),
            "closed or stale panel",
        )
        # A distinct open-file description must see the inherited lock as busy.
        with expected.open("rb") as probe:
            try:
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                raise ValueError("unlocked resource")
        write_new(out / "WORKER_ENTERED.json", dict(pid=os.getpid(), epoch=time.time()))
        run_model(root, out)
        return
    with lock.open("rb") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _, info = site(root)
        roster = inputs(root)
        if args.mode == "check":
            verify_files(root)
            require(not out.exists(), "panel already allocated")
            print(
                json.dumps(
                    dict(
                        **info,
                        inputs=[s[1]["input_sha256"] for s in roster.values()],
                        checkpoint_verified=True,
                        new_forwards=0,
                        lease_probe_only=True,
                        checked_utc=datetime.now(timezone.utc).isoformat(),
                    )
                )
            )
            return
        os.umask(0o077)
        out.mkdir(exist_ok=False, mode=0o700)
        now = time.time()
        write_new(
            out / "START.json",
            dict(
                **info,
                pid=os.getpid(),
                start_utc=datetime.now(timezone.utc).isoformat(),
                deadline_epoch=now + LIMIT_SECONDS,
                registration_sha256=hashlib.sha256(REG.read_bytes()).hexdigest(),
            ),
        )
        env = dict(
            os.environ,
            CUDA_DEVICE_ORDER="PCI_BUS_ID",
            CUDA_VISIBLE_DEVICES="0",
            JAX_PLATFORMS="cuda",
            XLA_PYTHON_CLIENT_PREALLOCATE="false",
            XLA_PYTHON_CLIENT_MEM_FRACTION="0.80",
            HF_HOME=str(root / "hf-cache"),
            XDG_CACHE_HOME=str(root / "runtime-cache"),
            OPENPI_DATA_HOME=str(root / "openpi-cache"),
            JAX_COMPILATION_CACHE_DIR=str(root / "jax-cache/P1_SOURCE_01"),
            OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            PYTHONPATH=str(root) + ":" + str(root / "vendor"),
            V4_FEEDBACK_LOCK_FD=str(lease.fileno()),
            V4_FEEDBACK_OWNER_PID=str(os.getpid()),
        )
        code = None
        peak = 0
        try:
            with (out / "worker.log").open("x") as sink:
                p = subprocess.Popen(
                    [
                        "timeout",
                        "--signal=TERM",
                        "--kill-after=15s",
                        str(LIMIT_SECONDS),
                        sys.executable,
                        str(Path(__file__).resolve()),
                        str(root),
                        "worker",
                    ],
                    cwd=root,
                    env=env,
                    stdout=sink,
                    stderr=subprocess.STDOUT,
                    pass_fds=(lease.fileno(),),
                )
                try:
                    while p.poll() is None:
                        # Monitoring errors must never leave the inference process running after releasing the lease.
                        board = int(
                            subprocess.check_output(
                                [
                                    "nvidia-smi",
                                    "-i",
                                    "0",
                                    "--query-gpu=memory.used",
                                    "--format=csv,noheader,nounits",
                                ],
                                timeout=10,
                            )
                        )
                        peak = max(peak, board)
                        time.sleep(0.5)
                    code = p.returncode
                finally:
                    if p.poll() is None:
                        p.terminate()
                    p.wait()
        finally:
            write_new(
                out / "OWNER_EXIT.json",
                dict(
                    exit_code=code, board_peak_mib_sampled=peak, ended_epoch=time.time()
                ),
            )
        require(code == 0, "inference worker stopped; see sealed partial evidence")


if __name__ == "__main__":
    main()
