"""One genuine frozen-policy call on an active, bounded, student Reach card."""

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sla_hierarchical_v16"))
from demonstrated_grip import read, require
from feedback_gpu import RECEIVER, UUID, journal, retained_inference, site, verify_files
from policy_query import frozen_predictor, input_identity


def worker(root, case, query, fd):
    import jax
    import jax.numpy as jnp
    from p0_o0_runner import gpu_identity
    from XPolicyLab.policy.SLA_Memory.model import Model

    lock = Path(read(RECEIVER)["gpu_lock"])
    st = os.fstat(fd)
    require(
        (st.st_dev, st.st_ino) == (lock.stat().st_dev, lock.stat().st_ino),
        "wrong GPU lease",
    )
    with lock.open("rb") as other:
        try:
            fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            raise ValueError("missing exclusive lock")
    lease = read(case / "control/LEASE.json")

    def current():
        require(
            read(case / "control/LEASE.json") == lease
            and lease["status"] == "ACTIVE"
            and time.time() < lease["deadline_epoch"]
            and not any((p / "STOP.json").exists() for p in (case, case.parent)),
            "closed world lease or STOP",
        )

    current()
    require(
        case.name in ("G00-A", "G01-A")
        and read(case.parent / "REGISTRATION.json")["issue"] == 483
        and read(case.parent / "QUALIFICATION.json")["qualified_for_Z1"],
        "no paired Reach allocation",
    )
    require(
        jax.default_backend() == "gpu"
        and len(jax.local_devices()) == 1
        and gpu_identity() == UUID,
        "GPU mismatch",
    )
    cfg = verify_files(root)
    model = Model(cfg)
    model.reset()
    meta = read(query / "REQUEST.json")
    prior = read(case / "NATIVE_REQUEST.json")
    require(
        meta["episode"] == prior["episode"]
        and 0 <= meta["number"] < 12
        and not (query / "RETURN.json").exists(),
        "request binding",
    )
    with np.load(query / "INPUT.npz", allow_pickle=False) as z:
        obs = {
            "images": {
                "cam_high": z["head"],
                "cam_left_wrist": np.zeros((480, 640, 3), np.uint8),
                "cam_right_wrist": z["wrist"],
            },
            "state": z["state"],
            "prompt": str(z["prompt"]),
        }
    require(
        input_identity(obs)[0] == meta["input_sha256"],
        "input does not match paused owner",
    )
    model._policy._rng = jax.random.wrap_key_data(
        jnp.asarray(meta["seed"], dtype=jnp.uint32)
    )
    capture = {}
    current()
    log = query / "events.jsonl"
    journal(log, "FORWARD_ENTERED", digest=meta["input_sha256"], seed=meta["seed"])
    result = frozen_predictor(
        model, obs, infer=lambda x: retained_inference(model._policy, x, capture)
    )
    require(
        result.shape == (50, 16)
        and capture["actions"].shape == (50, 32)
        and all(
            np.isfinite(x).all()
            for x in (result, capture["actions"], capture["sampler_state"])
        ),
        "invalid official output",
    )
    with (query / "OUTPUT.npz").open("xb") as dest:
        np.savez(dest, absolute=result, **capture)
    journal(log, "FORWARD_RETURNED", seconds=time.time() - meta["epoch"])
    with (query / "RETURN.json").open("x") as f:
        json.dump(
            {
                "episode": meta["episode"],
                "input_sha256": meta["input_sha256"],
                "output_sha256": hashlib.sha256(
                    (query / "OUTPUT.npz").read_bytes()
                ).hexdigest(),
            },
            f,
        )


def predict(root, case, tick, number, obs, sensor):
    from contextlib import ExitStack

    q = case / f"query-{number:02d}"
    q.mkdir(mode=0o700)
    require(number < 12 and sensor["tick"] == tick, "query limit or stale input")
    key, _ = input_identity(obs)
    require(key == sensor["input_sha256"], "changed sensor input")
    np.savez_compressed(
        q / "INPUT.npz",
        head=obs["images"]["cam_high"],
        wrist=obs["images"]["cam_right_wrist"],
        state=obs["state"],
        prompt=obs["prompt"],
    )
    seed = read(
        Path(__file__).resolve().parents[2]
        / "reports/SLA_HIERARCHICAL_V16/FEEDBACK_POLICY_PANEL_REGISTRATION.json"
    )["seed"]
    seed = [int(seed[0]), (int(seed[1]) + number) % 2**32]
    meta = {
        "number": number,
        "tick": tick,
        "input_sha256": key,
        "seed": seed,
        "episode": read(case / "NATIVE_REQUEST.json")["episode"],
        "epoch": time.time(),
    }
    (q / "REQUEST.json").write_text(json.dumps(meta))
    lock, _ = site(root)
    with ExitStack() as stack:
        lease = stack.enter_context(lock.open("rb"))
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _, resources = site(root)
        (q / "GPU_START.json").write_text(json.dumps(resources))
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES="0",
            CUDA_DEVICE_ORDER="PCI_BUS_ID",
            JAX_PLATFORMS="cuda",
            XLA_PYTHON_CLIENT_PREALLOCATE="false",
            XLA_PYTHON_CLIENT_MEM_FRACTION="0.80",
            HF_HOME=str(root / "hf-cache"),
            XDG_CACHE_HOME=str(root / "runtime-cache"),
            OPENPI_DATA_HOME=str(root / "openpi-cache"),
            JAX_COMPILATION_CACHE_DIR=str(root / "jax-cache/P1_SOURCE_01"),
            PYTHONPATH=str(root) + ":" + str(root / "vendor"),
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
        )
        exe = root / "vendor/XPolicyLab/policy/SLA_Memory/openpi/.venv/bin/python"
        log = stack.enter_context((q / "worker.log").open("x"))
        proc = subprocess.Popen(
            [
                str(exe),
                str(Path(__file__).resolve()),
                str(root),
                str(case),
                str(q),
                str(lease.fileno()),
            ],
            env=env,
            cwd=root,
            stdout=log,
            stderr=subprocess.STDOUT,
            pass_fds=(lease.fileno(),),
        )
        try:
            require(proc.wait(timeout=240) == 0, "inference worker interrupted")
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait()
            (q / "GPU_EXIT.json").write_text(
                json.dumps({"code": proc.returncode, "ended": time.time()})
            )
    r = read(q / "RETURN.json")
    require(
        r["input_sha256"] == key
        and r["episode"] == meta["episode"]
        and hashlib.sha256((q / "OUTPUT.npz").read_bytes()).hexdigest()
        == r["output_sha256"],
        "unbound policy return",
    )
    with np.load(q / "OUTPUT.npz", allow_pickle=False) as z:
        return z["absolute"].copy()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("root", "case", "query"):
        p.add_argument(k, type=Path)
    p.add_argument("fd", type=int)
    args = p.parse_args()
    worker(args.root, args.case, args.query, args.fd)
