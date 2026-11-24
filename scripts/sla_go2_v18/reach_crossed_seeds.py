"""Complete the two missing frozen G01 observation/seed cells of issue 486."""

import argparse
from contextlib import redirect_stdout
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from reach_forward import (
    site,
    verify_files,
    retained_inference,
    frozen_predictor,
    input_identity,
    journal,
    UUID,
)
from demonstrated_grip import read, require, pin
from feedback_policy_panel import arm_diff

HERE = Path(__file__).resolve()
REPORT = HERE.parents[2] / "reports/SLA_GO2_V18"
DIGESTS = (
    "999b92327bb93242c82651808f2466d967f17acd05b808f65d5b8701de110095",
    "b16e7d7bdb8f4287d296c6565096ebe6995e13e3af9cdd2ed1534cbb7a93013c",
)


def record(path, obj):
    with path.open("x") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write("\n")


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def case_inputs(root):
    archive = root / "v18/sealed-Z1/members.json"
    members = read(
        pin(archive, read(REPORT / "Z1_NATIVE/ARCHIVE.json")["member_manifest_sha256"])
    )
    case = Path("Z1_REACH_483_G01/G01-A")
    rows = []
    for i in (0, 1):
        q = case / f"query-{i:02d}"

        def sealed(name):
            return pin(root / "v18" / q / name, members[str(q / name)])

        req = read(sealed("REQUEST.json"))
        ret = read(sealed("RETURN.json"))
        require(
            req["tick"] == (6203, 8203)[i]
            and req["seed"] == [0, i]
            and req["input_sha256"] == DIGESTS[i],
            "registered cell",
        )
        with np.load(sealed("INPUT.npz"), allow_pickle=False) as z:
            obs = dict(
                state=z["state"],
                prompt=str(z["prompt"]),
                images=dict(
                    cam_high=z["head"],
                    cam_right_wrist=z["wrist"],
                    cam_left_wrist=np.zeros((480, 640, 3), np.uint8),
                ),
            )
        require(
            input_identity(obs)[0] == ret["input_sha256"] == DIGESTS[i],
            "measured input",
        )
        rows.append((obs, pin(sealed("OUTPUT.npz"), ret["output_sha256"])))
    return rows


def worker(root, out, i, fd):
    import jax
    import jax.numpy as jnp
    from p0_o0_runner import gpu_identity
    from XPolicyLab.policy.SLA_Memory.model import Model

    lock, _ = site(root)
    a, b = os.fstat(fd), lock.stat()
    require((a.st_dev, a.st_ino) == (b.st_dev, b.st_ino), "wrong inherited lease")
    with lock.open("rb") as probe:
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            raise ValueError("not leased")
    started = read(out / "START.json")

    def current():
        require(
            started["issue"] == 486
            and started["owner_pid"] == os.getppid()
            and time.time() < started["deadline"]
            and not any((out / n).exists() for n in ("EXIT.json", "STOP.json")),
            "owner stopped or expired",
        )

    current()
    obs = case_inputs(root)[i][0]
    require(
        len(jax.local_devices()) == 1
        and jax.default_backend() == "gpu"
        and gpu_identity() == UUID,
        "GPU identity",
    )
    cfg = verify_files(root)
    model = Model(cfg)
    model.reset()
    seed = [0, 1 - i]
    model._policy._rng = jax.random.wrap_key_data(jnp.asarray(seed, dtype=jnp.uint32))
    capture = {}
    current()
    # Exclusive marker counts a sampler entry even if this worker is interrupted.
    record(
        out / f"entered-{i}.json",
        dict(obs=i, seed=seed, input_sha256=DIGESTS[i], time=time.time()),
    )
    absolute = frozen_predictor(
        model, obs, infer=lambda x: retained_inference(model._policy, x, capture)
    )
    require(
        absolute.shape == (50, 16)
        and capture["actions"].shape == (50, 32)
        and all(
            np.isfinite(a).all()
            for a in (absolute, capture["actions"], capture["sampler_state"])
        ),
        "invalid result",
    )
    p = out / f"cross-{i}.npz"
    with p.open("xb") as f:
        np.savez(f, absolute=absolute, **capture)
    record(
        out / f"returned-{i}.json",
        dict(
            obs=i,
            seed=seed,
            input_sha256=DIGESTS[i],
            output_sha256=file_sha(p),
            time=time.time(),
        ),
    )


def execute(root, out, allocation):
    grant = read(allocation)
    require(
        grant.get("issue") == 486
        and grant.get("allocated") == 2
        and grant.get("status") == "APPROVED",
        "issue 486 scientific allocation",
    )
    require(os.environ.get("REACH_SHARED_GPU") == "1", "shared mode required")
    case_inputs(root)
    lock, _ = site(root)
    with lock.open("rb") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _, info = site(root)
        out.mkdir(mode=0o700)
        now = time.time()
        deadline = now + 600
        record(
            out / "START.json",
            dict(
                issue=486,
                owner_pid=os.getpid(),
                start=now,
                deadline=deadline,
                site=info,
                allocation_sha256=file_sha(allocation),
            ),
        )
        try:
            for i in (0, 1):
                _, headroom = site(root)
                journal(out / "owner.jsonl", "SPAWN", cell=i, **headroom)
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
                    PYTHONPATH=f"{root}:{root}/vendor",
                    OMP_NUM_THREADS="1",
                    MKL_NUM_THREADS="1",
                    OPENBLAS_NUM_THREADS="1",
                )
                exe = (
                    root / "vendor/XPolicyLab/policy/SLA_Memory/openpi/.venv/bin/python"
                )
                with (out / f"worker-{i}.log").open("x") as log:
                    proc = subprocess.Popen(
                        [
                            str(exe),
                            str(HERE),
                            str(root),
                            "worker",
                            "--index",
                            str(i),
                            "--fd",
                            str(lease.fileno()),
                        ],
                        pass_fds=(lease.fileno(),),
                        cwd=root,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                    try:
                        while proc.poll() is None:
                            require(
                                time.time() < min(deadline, now + 300 * (i + 1))
                                and not (out / "STOP.json").exists(),
                                "deadline or STOP",
                            )
                            time.sleep(0.5)
                    finally:
                        if proc.poll() is None:
                            proc.kill()
                        proc.wait()
                        journal(
                            out / "owner.jsonl",
                            "EXIT",
                            cell=i,
                            pid=proc.pid,
                            code=proc.returncode,
                        )
                require(
                    proc.returncode == 0 and (out / f"returned-{i}.json").exists(),
                    "interrupted cell",
                )
        finally:
            record(
                out / "EXIT.json",
                dict(
                    ended=time.time(),
                    entered=len(list(out.glob("entered-*.json"))),
                    returned=len(list(out.glob("returned-*.json"))),
                    simulation_ticks=0,
                    updates=0,
                ),
            )


def compare(root, paths):
    from feedback_policy_panel import official_action_codec
    from reach_pair_readout import geometry

    _, _, decode = official_action_codec(root)
    case = root / "v18/Z1_REACH_483_G01/G01-A"
    records = [
        json.loads(s) for s in (case / "POLICY_OWNER.jsonl").read_text().splitlines()
    ]
    returns = {r["tick"]: r["sensor"] for r in records if r["kind"] == "RETURN"}
    native = case / read(case / "NATIVE_REQUEST.json")["episode"]
    with (native / "MOBILE_SAMPLES.private.jsonl").open() as f:
        samples = {
            r["native_end_step"]: r
            for r in map(json.loads, f)
            if r["native_end_step"] in returns
        }
    pr = np.asarray(read(case / "REACH_ENTRY.json")["direction"])

    def decoded(state, actions):
        return decode(dict(state=state.copy(), actions=actions.copy()))[
            "actions"
        ].astype(np.float32)

    values, geom = {}, []
    for cell, path in paths.items():
        with np.load(path, allow_pickle=False) as z:
            raw = z["actions"].astype(str(z["actions_dtype"]))
            st = z["sampler_state"].astype(str(z["sampler_state_dtype"]))
            absolute = z["absolute"]
            require(np.array_equal(decoded(st, raw), absolute), "official decode")
            i, seed = cell
            tick = (6203, 8203)[i]
            meta = returns[tick]
            require(meta["input_sha256"] == DIGESTS[i], "native sensor geometry")
            v = geometry(
                absolute,
                np.array(meta["world_origin"]),
                np.array(meta["link6_tcp"]),
                pr,
                np.array(samples[tick]["object_pose"])[:3, 3],
                np.array(samples[tick]["actual_TCP"])[:3, 3],
            )
            geom.append(
                dict(tick=tick, seed=[0, seed], output_sha256=file_sha(path), **v)
            )
            values[cell] = (st, raw)

    def contrast(a, b, state):
        ra = values[a][1]
        rb = values[b][1]
        st = values[state][0]
        return dict(
            raw_equal=bool(np.array_equal(ra, rb)),
            right_arm=arm_diff(decoded(st, ra)[:, 8:], decoded(st, rb)[:, 8:]),
        )

    res = dict(geometry=geom)
    if len(values) == 4:
        for i in (0, 1):
            require(
                np.array_equal(values[i, 0][0], values[i, 1][0]),
                "stored state changed with seed",
            )
        res["same_seed_observation_effect"] = [
            contrast((0, s), (1, s), (1, s)) for s in (0, 1)
        ]
        res["same_observation_seed_effect"] = [
            contrast((i, 0), (i, 1), (i, 0)) for i in (0, 1)
        ]
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("mode", choices=("preflight", "execute", "worker", "score"))
    parser.add_argument("--index", type=int, choices=(0, 1))
    parser.add_argument("--fd", type=int)
    parser.add_argument("--allocation", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    out = root / "v18/Z1_CROSSED_486"
    if args.mode == "worker":
        worker(root, out, args.index, args.fd)
    elif args.mode == "execute":
        execute(root, out, args.allocation)
    else:
        cells = {(i, i): p for i, (_, p) in enumerate(case_inputs(root))}
        if args.mode == "score":
            require(read(out / "EXIT.json")["returned"] == 2, "incomplete cells")
            for i in (0, 1):
                ret = read(out / f"returned-{i}.json")
                require(
                    ret["seed"] == [0, 1 - i] and ret["input_sha256"] == DIGESTS[i],
                    "wrong cell",
                )
                cells[i, 1 - i] = pin(out / f"cross-{i}.npz", ret["output_sha256"])
        with redirect_stdout(sys.stderr):
            result = compare(root, cells)
        print(json.dumps(result, indent=2, allow_nan=False))
