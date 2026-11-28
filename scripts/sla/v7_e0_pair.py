"""Compare both sealed V7-A endpoints with the original frozen TRAIN actions."""

import argparse
from contextlib import closing
import gc
import hashlib
from pathlib import Path

import numpy as np

from m38_gpu_preflight_v1 import bind_imports, check_gpu, pin, read, require, sha
from scripts.pi0292.durable import OwnerLock, write
from v7_core_endpoint import ARMS, PLAN
from v7_core_provider import load_context_provider
from v7_e0_matched_forward import MANIFEST, WORKER
from v7_feedback_eval import check_pair


ORIGINAL_RESULTS = "379c227d7d3e82a13132f631e4b8a24310d58a3e0c6ca779cc952217e40c613d"


def action_delta(current, baseline):
    xyz = 1000 * np.linalg.norm(current[:, :3] - baseline[:, :3], axis=-1)
    return dict(p10_xyz_max_mm=float(xyz[:10].max()), h40_xyz_max_mm=float(xyz.max()),
                first_jaw_delta_mm=float(40 * (current[0, 9] - baseline[0, 9])))


def run(manifest, worker, original, providers, pair_sha, output, safe_manifest=None, safe_sha=None):
    import jax
    from go2arx5_vla.sla_adapter.adapt5k_data import AdaptTargetDataset
    from openpi_client.image_tools import resize_with_pad

    roster = read(pin(manifest, MANIFEST))
    spec = read(pin(worker, WORKER))
    bind_imports(spec)
    from scripts.sla.core_hopper_native_scope import accepted
    from scripts.sla.hopper_site import pinned

    _, pair = check_pair(providers, pair_sha)
    evidence = read(pin(original / "ACTUAL.json", ORIGINAL_RESULTS))
    require(evidence["input_sha256"] == MANIFEST and evidence["actual_model_forwards"] == 15,
            "Wrong original TRAIN action evidence")
    baseline = {}
    for row in evidence["rows"]:
        if row["model"] == "R4":
            with np.load(pin(original / row["arrays_file"], row["arrays_sha256"]), allow_pickle=False) as data:
                baseline[row["frame"], row["task"]] = data["physical_H40"]
    fixed = [row for row in roster["conditions"] if row["model"] == "R4"]
    require(len(fixed) == len(baseline) == 5, "Expected the original five TRAIN states and prompt pairs")
    frames = dict(roster["frames"])
    require((safe_manifest is None) == (safe_sha is None), "A successful-stage manifest requires its sealed digest")
    if safe_manifest is not None:
        success = read(pin(safe_manifest, safe_sha))
        require(success["plan"] == PLAN and success["split"] == "TRAIN_DIAGNOSTIC"
                and set(success["frames"]) == {"R4_SAFE_GRASP", "R4_SAFE_LIFT"}
                and not success["authorized_for_training_labels"], "Expected the fixed successful-stage TRAIN evidence")
        for name, frame in success["frames"].items():
            record = frame["baseline"]
            with np.load(pin(record["path"], record["sha256"]), allow_pickle=False) as data:
                baseline[name, frame["task"]] = data["physical_H40"]
            fixed.append(dict(frame=name, task=frame["task"], language=frame["language"]))
            frames[name] = frame
    count = len(fixed) * len(ARMS)
    with closing(OwnerLock(Path(spec["neural_owner_lock"]))):
        gpu = check_gpu(spec["GPU_uuid"], spec["allocator_limit_bytes"], spec["minimum_host_available_bytes"])
        require(jax.default_backend() == "gpu" and len(jax.local_devices()) == 1, "Expected the original GPU path")
        _, _, receiver = accepted(spec)
        source = pinned(receiver["input"])["target_D1"]
        target = AdaptTargetDataset(source["legacy"], source["captures"], resize_with_pad=resize_with_pad)
        target.bind_state(source["statistics"])
        output.mkdir(mode=0o700)
        write(output / "START.json", dict(plan=PLAN, gpu=gpu, max_forwards=count, pair_sha256=pair_sha))
        measurements = []
        for arm in ARMS:
            digest = pair["arms"][arm]["provider_sha256"]
            model, _ = load_context_provider(providers / arm, digest, receiver["model"], target, receiver["tokenizer"]["path"])
            infer = model.infer
            draws = []

            def check_draw(*args):
                draws.append(hashlib.sha256(np.asarray(args[-1]).tobytes()).hexdigest())
                require(draws[-1] == frame["observed_noise_sha256"], "TRAIN forward changed actual CUDA noise")
                return infer(*args)

            model.infer = check_draw
            for row in fixed:
                frame = frames[row["frame"]]
                path = frame["input"]
                stem = output / f"{arm}-{row['frame']}-{row['task']}"
                write(stem.with_suffix(".ENTERED.json"), dict(arm=arm, frame=row["frame"], task=row["task"]))
                with np.load(pin(path["path"], path["sha256"]), allow_pickle=False) as packet:
                    received = model.predict(packet["state_eef10"], [packet["body_rgb"], packet["wrist_rgb"]], row["language"],
                               packet["T_policy_from_mount"], noise_id=frame["observed_noise_id"],
                               previous_reference=frame["previous_command"])
                current = received["physical_H40"]
                require(current.shape == (40, 10) and np.isfinite(current).all(), "Invalid H40 output")
                artifact = stem.with_suffix(".npz")
                with artifact.open("xb") as stream:
                    np.savez(stream, normalized=received["normalized"], physical_H40=current)
                result = dict(arm=arm, frame=row["frame"], task=row["task"], actual_noise_sha256=draws[-1],
                              compared_to_r4=action_delta(current, baseline[row["frame"], row["task"]]),
                              g7=received["G7_gripper_speed"], action_sha256=sha(artifact))
                write(stem.with_suffix(".RETURNED.json"), result)
                measurements.append(result)
            require(len(draws) == len(fixed), "Missing TRAIN sampler calls")
            del model, infer
            jax.clear_caches()
            gc.collect()
    write(output / "RESULT.json", dict(rows=measurements, model_forwards=count, optimizer_updates=0,
          native_integrations=0, input_sha256=MANIFEST, successful_stage_sha256=safe_sha,
          pair_sha256=pair_sha, authority="TRAIN_FIXED_OBSERVATION"))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "worker", "original", "providers", "output"):
        cli.add_argument(name, type=Path)
    cli.add_argument("pair_sha")
    cli.add_argument("--safe-manifest", type=Path)
    cli.add_argument("--safe-sha")
    run(**vars(cli.parse_args()))
