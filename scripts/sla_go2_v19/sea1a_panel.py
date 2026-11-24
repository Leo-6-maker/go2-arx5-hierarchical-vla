"""Run the fixed issue #496 appearance ablation on archived RGB and the frozen policy."""

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np

from source_custody import BASE, digest, pin, read, require, target_case, target_obs, write_new

sys.path.insert(0, str(BASE / "scripts/sla_go2_v18"))
from policy_query import input_identity, tensor_sha
from reach_forward import UUID, frozen_predictor, journal, site, verify_files
from sea1_archive_readout import processor
from sea_observation import AppearanceEdit

REG = BASE / "docs/protocols/SLA_GO2_V191_SEA1A_REGISTRATION_20261124.json"
REG_SHA256 = "3f1ad5a7aa96ae5050f93dc153e3b443c6dca52e6144b26e0cc6dc108b2492ee"
PANEL = "SEA1A-SAME-VIEW-APPEARANCE-01"
BRIDGE = "v19/sea1-target-bridge-02"
RAW, PAINT = "RAW", "SAME_VIEW_CROSSPAINT"
SLOTS = ("G00-00", "G00-01", "G01-00", "G01-01")


def registration():
    reg = read(pin(REG, REG_SHA256))
    expected = [(s, a, [0, n]) for n in range(2) for s in SLOTS for a in (RAW, PAINT)]
    require([(c["state_id"], c["arm"], c["seed"]) for c in reg["cells"]] == expected
            and [s["id"] for s in reg["states"]] == list(SLOTS)
            and reg["panel_id"] == PANEL, "wrong fixed population or sampling order")
    return reg


def run_path(root):
    return root / "v19" / (PANEL + "-02")


def load_inputs(root, reg):
    from sea1_observation_preflight import edit_for, packet

    for entry in reg["source_evidence"]:
        blob = (BASE / entry["path"]).read_bytes()
        h = hashlib.sha1(f"blob {len(blob)}\0".encode() + blob).hexdigest()
        require(h == entry["git_blob_sha"], "registered input evidence changed")

    evidence = BASE / "reports/SLA_GO2_V19/SEA1"
    bridge_path = read(pin(root / BRIDGE / "TARGET_BRIDGE_ACTUAL.json",
                           digest(evidence / "TARGET_BRIDGE_ACTUAL.json")))
    require(bridge_path["source_geometry_asset_sha256"] == reg["source_geometry_asset_sha256"],
            "appearance asset identity")
    edits = {r["id"]: r for r in bridge_path["slots"]}
    ingress = {r["id"]: r for r in read(evidence / "OBSERVATION_INGRESS_ACTUAL.json")["go2"]}
    archive = read(BASE / "reports/SLA_GO2_V18/Z1_NATIVE/ARCHIVE.json")
    members = read(pin(root / "v18/sealed-Z1/members.json", archive["member_manifest_sha256"]))

    def sealed(p):
        return pin(p, members[str(p.relative_to(root / "v18"))])

    payloads, expected_fields = {}, {}
    for state in reg["states"]:
        name = state["id"]
        case = target_case(root, state["layout"])
        native = read(sealed(case / "NATIVE_REQUEST.json"))
        spec_path = sealed(case / "EXECUTION_SPEC.json")
        spec = read(spec_path)
        folder = case / f"query-{name[-2:]}"
        req = read(sealed(folder / "REQUEST.json"))
        src = target_obs(pin(sealed(folder / "INPUT.npz"), state["input_file_sha256"]))
        obs = packet(src, native["episode"], name, state["tick"], "GO2_NATIVE_2MS",
                     state["history_ticks"], left_present=False)
        row, trace = edits[name], ingress[name]
        require(req["input_sha256"] == state["input_sha256"] == row["original_input_sha256"]
                and req["tick"] == state["tick"] == row["tick"] == trace["tick"]
                and req["episode"] == native["episode"]
                and digest(spec_path) == state["owner_spec_sha256"]
                and spec["frozen_files"][spec["model_path"]] == state["scene_sha256"]
                and [a["tick"] for a in row["head"]] == trace["history_ticks"] == state["history_ticks"]
                and all(not a["oracle_edited_nonrobot_pixels"] and not a["oracle_edited_task_object_pixels"]
                        for a in (*row["head"], row["wrist"])),
                "changed scene, capture, history or supported raster")
        paint = pin(root / BRIDGE / (name + ".private.npz"), state["crosspaint_archive_sha256"])
        with np.load(paint, allow_pickle=False) as z:
            edit = AppearanceEdit(
                tuple(edit_for(frame, im, mask) for frame, im, mask in
                      zip(obs.head, z["head"], z["head_paint"], strict=True)),
                edit_for(obs.right, z["wrist"], z["wrist_paint"]),
            )
        raw, coated = obs.legacy_input(), obs.legacy_input()
        edit.apply(obs, coated)
        require(input_identity(raw)[0] == state["input_sha256"] == trace["raw_input_sha256"]
                and input_identity(coated)[0] == state["crosspaint_input_sha256"]
                == row["diagnostic_input_sha256"] == trace["diagnostic_input_sha256"],
                "registered RGB/state/instruction fingerprint")
        payloads[name, RAW], payloads[name, PAINT] = raw, coated
        expected_fields[name, RAW], expected_fields[name, PAINT] = trace["raw_fields"], trace["diagnostic_fields"]

    for cell in reg["cells"]:
        key = cell["state_id"], cell["arm"]
        require(input_identity(payloads[key])[0] == cell["input_sha256"], "cell input digest differs")
    return payloads, expected_fields


def view_fields(view):
    return dict(
        current={k: tensor_sha(v) for k, v in view.images.items()},
        current_masks={k: bool(v[0]) for k, v in view.image_masks.items()},
        memory=[tensor_sha(x) for x in view.extra_images[0]],
        token_ids=tensor_sha(view.tokenized_prompt), token_mask=tensor_sha(view.tokenized_prompt_mask),
        state=tensor_sha(view.state),
    )


def observation_fields(observation):
    from openpi.models.model import preprocess_observation

    return view_fields(preprocess_observation(None, observation, train=False))


def gpu_fields(root, payloads, cpu_fields):
    import jax
    import jax.numpy as jnp
    from openpi.models.model import Observation, preprocess_observation

    prepare, _, identity = processor(root)
    require(identity == registration()["processor_identity"], "official processor identity")

    def processed(value, platform):
        with jax.default_device(jax.devices(platform)[0]):
            batch = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], value)
            return preprocess_observation(None, Observation.from_dict(batch), train=False)

    fields = {}
    for key, payload in payloads.items():
        prepared = prepare(payload)
        cpu, gpu = (processed(prepared, name) for name in ("cpu", "gpu"))
        reference, actual = view_fields(cpu), view_fields(gpu)
        require(reference == cpu_fields[key]
                and all(reference[k] == actual[k] for k in ("state", "token_ids", "token_mask", "current_masks")),
                "platform difference outside RGB arithmetic")
        errors = {}
        for name in (*cpu.images.keys(), "memory"):
            x, y = (np.asarray(v.extra_images if name == "memory" else v.images[name]) for v in (cpu, gpu))
            error = float(np.max(abs(x-y)))
            require(x.shape == y.shape and x.dtype == y.dtype == np.float32
                    and error <= np.finfo(np.float32).eps
                    and np.array_equal(np.rint((x+1)*127.5), np.rint((y+1)*127.5)),
                    "device image preprocessing changed actual pixel values")
            errors[name] = error
        fields[key] = dict(cpu=reference, gpu=actual, max_rgb_float_delta=errors)
    for state in SLOTS:
        check_fields(fields[state, RAW]["gpu"], fields[state, PAINT]["gpu"])
    return fields


def check_fields(a, b):
    require(set(a["current"]) == set(b["current"]) == {"base_0_rgb", "right_wrist_0_rgb", "left_wrist_0_rgb"}
            and a["current"]["left_wrist_0_rgb"] == b["current"]["left_wrist_0_rgb"]
            and all(a["current"][k] != b["current"][k] for k in ("base_0_rgb", "right_wrist_0_rgb"))
            and len(a["memory"]) == len(b["memory"]) == 9
            and all(x != y for x, y in zip(a["memory"], b["memory"], strict=True))
            and all(a[k] == b[k] for k in ("state", "token_ids", "token_mask", "current_masks"))
            and all(a["current_masks"].values()), "not an isolated appearance intervention")


def rng_noise(seed):
    import jax
    import jax.numpy as jnp

    rng = jax.random.wrap_key_data(jnp.asarray(seed, dtype=jnp.uint32))
    sample_key = jax.random.split(rng)[1]
    noise = np.asarray(jax.random.normal(sample_key, (1, 50, 32)))
    require(noise.dtype == np.float32 and np.isfinite(noise).all(), "invalid initial noise")
    return rng, np.asarray(jax.random.key_data(sample_key)), noise


def infer(policy, payload, fields, rng, key, noise, entering):
    import jax

    original = policy._sample_actions
    original_output = policy._output_transform
    capture = {}

    def guarded(sample_key, observation, **kwargs):
        require(np.array_equal(jax.random.key_data(sample_key), key)
                and np.array_equal(kwargs.get("noise"), noise), "wrong actual sampler key or noise")
        observed = observation_fields(observation)
        require(observed == fields, "effective sampler input changed")
        entering(observed)
        return original(sample_key, observation, **kwargs)

    def retain(result):
        for key, name in (("actions", "actions"), ("state", "sampler_state")):
            value = np.asarray(result[key])
            capture[name + "_dtype"] = np.array(value.dtype.name)
            capture[name] = value.astype(np.float32) if value.dtype.name == "bfloat16" else value.copy()
        return original_output(result)

    policy._rng = rng
    policy._sample_actions = guarded
    policy._output_transform = retain
    try:
        result = policy.infer(payload, noise=noise)
    finally:
        policy._sample_actions = original
        policy._output_transform = original_output
    return result, capture


def code_binding():
    scripts = Path(__file__).parent
    files = [
        Path(__file__), scripts / "sea1a_score.py", scripts / "sea1_archive_readout.py",
        scripts / "sea1_observation_preflight.py", scripts / "sea_observation.py",
        scripts / "sea2_archived_action_preflight.py",
        BASE / "scripts/sla_go2_v18/reach_forward.py",
        BASE / "scripts/sla_hierarchical_v16/feedback_gpu.py",
    ]
    return {str(path.relative_to(BASE)): digest(path) for path in files}


def cold(root, dest):
    import jax
    import jax.numpy as jnp
    from openpi.policies.policy import Policy
    from types import SimpleNamespace

    require(jax.default_backend() == "cpu" and os.environ.get("CUDA_VISIBLE_DEVICES") == "",
            "cold input verification requires CPU only")
    reg = registration()
    payloads, fields = load_inputs(root, reg)
    prepare, inverse, identity = processor(root)
    require(identity == reg["processor_identity"], "official processor identity")
    observed = []

    def surrogate(_, obs, **kw):
        return jnp.zeros_like(kw["noise"])

    stub = SimpleNamespace(_input_transform=prepare, _output_transform=inverse,
                           _sample_actions=surrogate, _sample_kwargs={})
    stub.infer = Policy.infer.__get__(stub)
    for state in SLOTS:
        check_fields(fields[state, RAW], fields[state, PAINT])
    noises = {n: rng_noise([0, n]) for n in range(2)}
    for cell in reg["cells"]:
        key = cell["state_id"], cell["arm"]
        result, capture = infer(stub, payloads[key], fields[key], *noises[cell["seed"][1]], observed.append)
        require(np.asarray(result["actions"]).shape == (50, 16)
                and capture["actions"].shape == (50, 32)
                and capture["sampler_state"].shape == (32,), "official transform round trip")
    require(len(observed) == len(reg["cells"]), "cold sampler seam incomplete")
    write_new(dest, dict(
        status="COLD_16_INPUTS_VERIFIED", registration_sha256=REG_SHA256,
        code=code_binding(), processor=identity, slots=[c["slot"] for c in reg["cells"]],
        reuse_verified=0, historical_noise_archived=False, jax_version=jax.__version__,
        model_forwards=0, physics_ticks=0, training_updates=0,
    ))


@contextmanager
def gpu_lock(root):
    path, _ = site(root)
    with path.open("rb") as file:
        fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield file, site(root)[1]


def execute(root, allocation_path, allocation_hash, proof_path):
    grant = read(pin(allocation_path, allocation_hash))
    proof = read(proof_path)
    now = time.time()
    require(grant.get("issue") == 496 and grant.get("status") == "APPROVED"
            and grant.get("panel") == PANEL and grant.get("max_forwards") == 16
            and grant.get("registration_sha256") == REG_SHA256
            and grant.get("cold_sha256") == digest(proof_path)
            and bool(grant.get("authorization_id")) and now < grant.get("valid_until_epoch", 0),
            "missing matching scientific allocation")
    require(proof["status"] == "COLD_16_INPUTS_VERIFIED" and proof["registration_sha256"] == REG_SHA256
            and proof["code"] == code_binding() and proof["reuse_verified"] == 0,
            "cold execution binding changed")
    require(os.environ.get("REACH_SHARED_GPU") == "1", "approved GPU sharing environment required")
    previous = root / "v19" / PANEL
    prior_exit = previous / "OWNER_EXIT.json"
    require(grant.get("prior_exit_sha256") == digest(prior_exit)
            and read(prior_exit)["project_lock_released"]
            and read(prior_exit)["exit_code"] != 0
            and not list(previous.glob("*.entered.json")),
            "prior attempt is not a verified zero-entry engineering interruption")
    out = run_path(root)
    code, peak, started = None, 0, time.monotonic()
    recorded_start = False
    try:
        with gpu_lock(root) as (lease, info):
            out.mkdir(mode=0o700)
            deadline = min(grant["valid_until_epoch"], time.time() + 600)
            write_new(out / "START.json", dict(
                issue=496, owner_pid=os.getpid(), deadline=deadline, site=info,
                registration_sha256=REG_SHA256, allocation_sha256=allocation_hash, cold_sha256=digest(proof_path),
            ))
            recorded_start = True
            env = dict(
                os.environ, CUDA_VISIBLE_DEVICES="0", CUDA_DEVICE_ORDER="PCI_BUS_ID", JAX_PLATFORMS="cuda,cpu",
                XLA_PYTHON_CLIENT_PREALLOCATE="false", XLA_PYTHON_CLIENT_MEM_FRACTION="0.80",
                HF_HOME=str(root / "hf-cache"), XDG_CACHE_HOME=str(root / "runtime-cache"),
                OPENPI_DATA_HOME=str(root / "openpi-cache"),
                JAX_COMPILATION_CACHE_DIR=str(root / "jax-cache/P1_SOURCE_01"),
                PYTHONPATH=f"{root}:{root}/vendor", OMP_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
            )
            try:
                with (out / "worker.log").open("x") as log:
                    p = subprocess.Popen(
                        [sys.executable, str(Path(__file__).resolve()), str(root), "worker", str(lease.fileno())],
                        pass_fds=(lease.fileno(),), cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                    )
                    try:
                        while p.poll() is None:
                            require(time.time() < deadline and not (out / "STOP.json").exists(), "deadline or STOP")
                            use = subprocess.check_output([
                                "nvidia-smi", "-i", "0", "--query-gpu=memory.used",
                                "--format=csv,noheader,nounits",
                            ], timeout=10)
                            peak = max(peak, int(use))
                            time.sleep(.5)
                    finally:
                        if p.poll() is None:
                            p.kill()
                        code = p.wait()
            finally:
                elapsed = time.monotonic() - started
    finally:
        if recorded_start:
            write_new(out / "OWNER_EXIT.json", dict(
                exit_code=code, board_peak_mib_sampled=peak, lease_seconds=elapsed, project_lock_released=True,
                native_starts=0, physics_ticks=0, training_updates=0, render_jobs=0,
            ))
    require(code == 0, "worker stopped; preserve the partial panel")


def worker(root, fd):
    import jax
    from p0_o0_runner import gpu_identity
    from XPolicyLab.policy.SLA_Memory.model import Model

    path, _ = site(root)
    st, lease = path.stat(), os.fstat(fd)
    require((st.st_dev, st.st_ino) == (lease.st_dev, lease.st_ino), "GPU owner lease differs")
    with path.open("rb") as probe:
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            raise ValueError("GPU owner did not hold the project lock")
    out = run_path(root)
    start = read(out / "START.json")

    def current():
        require(start["issue"] == 496 and start["owner_pid"] == os.getppid()
                and start["registration_sha256"] == REG_SHA256 and time.time() < start["deadline"]
                and not (out / "STOP.json").exists(), "expired or stopped GPU owner")

    current()
    reg = registration()
    payloads, fields = load_inputs(root, reg)
    require(jax.default_backend() == "gpu" and len(jax.local_devices()) == 1
            and gpu_identity() == UUID, "GPU identity")
    verified_fields = gpu_fields(root, payloads, fields)
    write_new(out / "DEVICE_INPUTS.json", [
        dict(state=state, arm=condition, **check) for (state, condition), check in verified_fields.items()
    ])
    cfg = verify_files(root)
    current()
    log = out / "CALLS.jsonl"
    journal(log, "LOAD_ENTERED")
    model = Model(cfg)
    journal(log, "LOAD_RETURNED", jax_version=jax.__version__, gpu=UUID)
    noises = {n: rng_noise([0, n]) for n in range(2)}
    for n, (_, key, noise) in noises.items():
        with (out / f"noise-{n}.npz").open("xb") as f:
            np.savez(f, noise=noise, sample_key=key)
    for cell in reg["cells"]:
        key = cell["state_id"], cell["arm"]
        rng, sample_key, noise = noises[cell["seed"][1]]
        actual = {}
        model.reset()
        then = time.monotonic()

        def entering(fields_seen):
            current()
            write_new(out / (cell["slot"] + ".entered.json"), dict(
                slot=cell["slot"], seed=cell["seed"], input_sha256=cell["input_sha256"],
                noise_sha256=tensor_sha(noise), sample_key=sample_key.tolist(), sampler_fields=fields_seen,
            ))

        def retained(payload):
            result, capture = infer(model._policy, payload, verified_fields[key]["gpu"], rng, sample_key, noise, entering)
            actual.update(capture)
            return result

        current()
        absolute = frozen_predictor(model, payloads[key], infer=retained)
        require(absolute.shape == (50, 16) and absolute.dtype == np.float32
                and actual["actions"].shape == (50, 32) and actual["sampler_state"].shape == (32,)
                and all(np.isfinite(x).all() for x in (absolute, actual["actions"], actual["sampler_state"])),
                "invalid sampler or inverse return")
        current()
        output = out / (cell["slot"] + ".npz")
        with output.open("xb") as f:
            np.savez(f, absolute=absolute, **actual)
        write_new(out / (cell["slot"] + ".returned.json"), dict(
            output_sha256=digest(output), decoded_sha256=tensor_sha(absolute), seconds=time.monotonic()-then,
        ))
    journal(log, "PANEL_RETURNED", cells=16, resident_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("root", type=Path)
    sub = cli.add_subparsers(dest="mode", required=True)
    sub.add_parser("cold").add_argument("path", type=Path)
    run = sub.add_parser("execute")
    run.add_argument("allocation", type=Path)
    run.add_argument("allocation_sha256")
    run.add_argument("proof", type=Path)
    sub.add_parser("worker").add_argument("fd", type=int)
    args = cli.parse_args()
    os.umask(0o077)
    root = args.root.resolve()
    if args.mode == "cold":
        cold(root, args.path)
    elif args.mode == "execute":
        execute(root, args.allocation, args.allocation_sha256, args.proof)
    else:
        worker(root, args.fd)


if __name__ == "__main__":
    main()
