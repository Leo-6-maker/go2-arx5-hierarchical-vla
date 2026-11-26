"""Run a single frozen R4 or R5 pass over the sealed teacher observation bank."""

import argparse
from collections import Counter
from contextlib import closing
import hashlib
import os
from pathlib import Path
import sys
import time

import numpy as np

from m38_gpu_preflight_v1 import bind_imports, check_gpu, pin, read, require
from scripts.pi0292.durable import OwnerLock, write
from v4_long_watch import candidate, pinned, ref, score
from v62_r5_teacher_feedback import PLAN


AUDITS = {
    "R4": "820b1ef46077355da86c85c448bd83e2b552ab199f138439d057e82f53f79741",
    "R5": "f63869a7dda6f47a93985becd6811cfb88e80d967cbafb506546b3c24421563c",
}


def run(role, worker, roster, roster_sha, retention, out, *, execute):
    source = read(pin(roster, roster_sha))
    rows = source["rows"]
    require(source["plan"] == PLAN and source["evaluation_only"]
            and len({r["id"] for r in rows}) == len(rows) == 48
            and Counter(r["category"] for r in rows) == {"STILL_CARRYING": 24, "RELEASE_NEIGHBORHOOD": 24}
            and len({r["noise_id"] for r in rows}) == 48,
            "Use the fixed paired teacher roster")
    spec = read(worker)
    bind_imports(spec)
    from scripts.sla.core_hopper_native_scope import accepted

    _, gate, receiver = accepted(spec)
    template = read(pin(Path(gate["package"]["path"]) / "CHECKPOINT.json", gate["package"]["sha256"]))
    package = retention / f"{role}_STEP0200_PROVIDER"
    provider_proof = pin(retention / f"{role}_STEP0200_PROVIDER_READBACK.json", AUDITS[role])
    manifest = candidate(package, template, 200, provider_proof)
    for row in rows:
        with np.load(pinned(row["input"]), allow_pickle=False) as packet:
            require(packet["state_eef10"].shape == (10,) and np.isfinite(packet["state_eef10"]).all()
                    and all(packet[key].shape == (256, 256, 3) and packet[key].dtype == np.uint8
                            for key in ("body_rgb", "wrist_rgb")), "Malformed recorded student-visible teacher fields")
    if not execute:
        write(out, dict(role=role, roster=ref(roster), checkpoint=manifest["checkpoint_sha256"],
                        windows=len(rows), model_forwards=0, physics_ticks=0))
        return

    import jax
    from go2arx5_vla.sla_adapter.adapt5k_data import AdaptTargetDataset
    from openpi_client.image_tools import resize_with_pad
    from scripts.sla.hopper_reference import provider_from_manifest
    from scripts.sla.hopper_site import pinned as site_pinned

    with closing(OwnerLock(Path(spec["neural_owner_lock"]))):
        require(sys.executable == spec["python"] and os.uname().nodename == spec["host"]
                and os.getuid() == spec["uid"] and all(os.environ.get(key) == spec["gpu_environment"][key]
                    for key in ("CUDA_VISIBLE_DEVICES", "JAX_PLATFORMS", "XLA_FLAGS", "XLA_PYTHON_CLIENT_MEM_FRACTION")),
                "Use the qualified local frozen-model execution environment")
        gpu = check_gpu(spec["GPU_uuid"], spec["allocator_limit_bytes"], spec["minimum_host_available_bytes"])
        require(jax.default_backend() == "gpu" and len(jax.local_devices()) == 1, "One Corelab GPU is required")
        out.mkdir(mode=0o700)
        write(out / "START.json", dict(role=role, roster=ref(roster), gpu=gpu,
                                       checkpoint=manifest["checkpoint_sha256"], max_forwards=48))
        jax.config.update("jax_compilation_cache_dir", str(out / "jit-cache"))
        deadline = time.monotonic() + 900
        acquisition = site_pinned(receiver["input"])["target_D1"]
        dataset = AdaptTargetDataset(acquisition["legacy"], acquisition["captures"], resize_with_pad=resize_with_pad)
        dataset.bind_state(acquisition["statistics"])
        model = provider_from_manifest(package, manifest, receiver["model"], dataset, receiver["tokenizer"]["path"])
        infer = model.infer
        seen = []
        active = None

        def verify(*args):
            realized = hashlib.sha256(np.asarray(args[-1]).tobytes()).hexdigest()
            require(active is not None and realized == active["noise_sha256"], "Actual policy noise changed")
            seen.append(realized)
            return infer(*args)

        model.infer = verify
        results = []
        for row in rows:
            require(time.monotonic() < deadline and not (out / "STOP.json").exists(), "Offline lease ended")
            active = row
            stem = out / row["id"]
            write(stem.with_suffix(".ENTERED.json"), row)
            with np.load(pinned(row["input"]), allow_pickle=False) as observation:
                old = row["previous_reference"]
                result = model.predict(observation["state_eef10"], [observation["body_rgb"], observation["wrist_rgb"]],
                                       row["language"], observation["T_policy_from_mount"],
                                       noise_id=row["noise_id"], previous_reference=old)
                actions = result["physical_H40"]
                metrics = score(actions, observation, old)
                width = 2 + 40 * actions[:, 9]
                metrics.update(max_open_p10_mm=float(width[:10].max()), max_open_h40_mm=float(width.max()))
            action = stem.with_suffix(".npz")
            with action.open("xb") as f:
                np.savez(f, normalized=result["normalized"], physical_H40=actions)
            record = dict(row, action=ref(action), metrics=metrics, actual_noise_sha256=seen[-1])
            write(stem.with_suffix(".RETURNED.json"), record)
            results.append(record)
        require(len(seen) == len(results) == 48, "Incomplete actual teacher-feedback sample")
        write(out / "RESULT.json", dict(role=role, checkpoint=manifest["checkpoint_sha256"], roster=ref(roster),
              rows=results, model_forwards=48, physics_ticks=0, training_updates=0))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("role", choices=AUDITS)
    for name in ("worker", "roster", "retention", "out"):
        p.add_argument(name, type=Path)
    p.add_argument("--roster-sha", required=True)
    p.add_argument("--execute", action="store_true")
    run(**vars(p.parse_args()))
