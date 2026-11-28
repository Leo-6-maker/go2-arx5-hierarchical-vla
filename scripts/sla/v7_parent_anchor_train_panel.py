"""Seal and measure a fixed TRAIN-only B6 action-retention panel."""
import argparse
from contextlib import closing
import gc
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile

import numpy as np

from m38_gpu_preflight_v1 import bind_imports, check_gpu, pin, read, require, sha
from scripts.pi0292.durable import OwnerLock, write

INDEX = "9d9396b503064b9fc94646db1e38c0c3a3ca1a1a24b254310644bd21fca43f56"
TEACHERS = ("CHV1-TRAIN-F2-HOLD-blue-00", "CHV1-TRAIN-F1-HOLD-red-02",
            "MEV1-TRAIN-F1-NEAR_NEG-00-PLACE-blue-left", "EF5_DIVERSE_PLACE_001",
            "V64_RECOVERY_TRAIN_02")


def freeze(site, package, output):
    require(not output.exists(), "Keep the existing fixed panel")
    index = read(pin(package / "index.json", INDEX))
    original = read(pin(site / "v7-e0-matched-train.dLi0Wh/ACTUAL.private.json",
                        "a6246288dd47f003309d058b8288aeea4454c3f63b14ad7ba13d54b1a2504bdb"))
    safe = read(pin(site / "v7-pair-recovery-02/E0-R4-SAFE.private.json",
                    "e5d05461034583e07b53a9b307ac5f0e92b691917cec937607eed2d9c8acb705"))
    output.mkdir(mode=0o700)
    frames, scenes = [], set()
    for number, name in enumerate(TEACHERS):
        entry, = (row for row in index["episodes"] if row["episode"] == name)
        require(entry["split"] in ("TRAIN", "TRAIN_RECOVERY"), "Protected data cannot select a candidate")
        if number < 4:
            scenes.add(entry["scene_group_sha256"])
        with tarfile.open(pin(package / entry["path"], entry["archive_sha256"])) as archive:
            with np.load(io.BytesIO(archive.extractfile("labels.npz").read()), allow_pickle=False) as data:
                labels = {key: data[key] for key in data.files}
            for phase in dict.fromkeys(labels["phase"].tolist()):
                available = np.flatnonzero((labels["phase"] == phase) & labels["valid"][:, :10].all(axis=1))
                if not len(available):
                    continue
                k = int(available[len(available)//2])
                with np.load(io.BytesIO(archive.extractfile(f"frames-{k//16:03d}.npz").read()), allow_pickle=False) as images:
                    slot = k % 16
                    require(np.array_equal(images["state_eef10"][slot], labels["state_eef10"][k])
                            and images["native_step"][slot] == labels["native_step"][k], "Frame/query binding changed")
                    rgb = images["rgb"][slot]
                packet = output / f"TEACHER-{number}-{phase}.npz"
                with packet.open("xb") as stream:
                    np.savez(stream, body_rgb=rgb[0], wrist_rgb=rgb[1], state_eef10=labels["state_eef10"][k],
                             T_policy_from_mount=entry["T_policy_from_mount"], physical_action7=labels["action"][k],
                             valid=labels["valid"][k])
                tick = int(labels["native_step"][k])
                frames.append(dict(id=packet.stem, episode=name, phase=str(phase), query=k+1,
                    input=dict(path=str(packet), sha256=sha(packet)), language=entry["language"],
                    noise_id=0, observed_noise_sha256=None, has_real_teacher_label=True,
                    previous_command=dict(width_m=float(labels["prior_actual_width_m"][k]),
                                          query_native_s=tick*.002, source_native_s=tick*.002-.05)))
    require(len(scenes) == 4, "The ordinary teacher panel must span four scene identities")
    for name, frame in safe["frames"].items():
        frames.append(dict(id=name, episode=frame["case"], phase=frame["phase"], query=frame["query"],
            input=frame["input"], language=frame["language"], noise_id=frame["observed_noise_id"],
            observed_noise_sha256=frame["observed_noise_sha256"], previous_command=frame["previous_command"],
            has_real_teacher_label=False))
    for condition in (row for row in original["conditions"] if row["model"] == "R4"):
        frame = original["frames"][condition["frame"]]
        frames.append(dict(id=condition["frame"]+"-"+condition["task"], episode=frame["case"], phase=frame["phase"],
            query=frame["query"], input=frame["input"], language=condition["language"],
            noise_id=frame["observed_noise_id"], observed_noise_sha256=frame["observed_noise_sha256"],
            previous_command=frame["previous_command"], has_real_teacher_label=False))
    write(output / "PANEL.json", dict(split="TRAIN_DIAGNOSTIC", frames=frames, teacher_scene_count=4,
          selection="ONE_MIDDLE_P10_SUPPORTED_QUERY_PER_AVAILABLE_PHASE_IN_FIXED_EPISODES",
          new_teacher_noise="PREDECLARED_SEED_0_COMMON_TO_ALL_MODELS", repetitions=2,
          max_forwards=2*3*len(frames), protected_conditions=0, optimizer_updates=0, native_integrations=0))


def measure(panel, panel_sha, worker, providers, output, *, reuse_r4=None, reuse_r4_sha=None):
    import jax
    from go2arx5_vla.sla_adapter.adapt5k_data import AdaptTargetDataset
    from go2arx5_vla.sla_adapter.gripper_v2 import GripLogitCodec
    from openpi_client.image_tools import resize_with_pad
    from v7_e0_matched_forward import WORKER
    from v7_parent_anchor_endpoint import ARMS

    roster = read(pin(panel, panel_sha))
    spec = read(pin(worker, WORKER))
    bind_imports(spec)
    from scripts.sla.core_hopper_native_scope import accepted
    from scripts.sla.hopper_site import pinned
    import importlib.util
    require(roster["split"] == "TRAIN_DIAGNOSTIC" and not output.exists(), "Use a new fixed TRAIN evaluation")
    proof = read(providers / "ACTUAL.json")
    original = read(Path(panel).parent.parent / "R4_REFERENCE.json")
    baseline = None
    if reuse_r4 is not None:
        require(reuse_r4_sha is not None, "Pin the immutable reused R4 output")
        prior = read(pin(reuse_r4 / "ACTUAL.json", reuse_r4_sha))
        require(prior["complete"] and prior["panel_sha256"] == panel_sha
                and read(reuse_r4.parent / "R4_REFERENCE.json") == original,
                "The original R4, inputs or fixed TRAIN panel changed")
        selected = [row for row in prior["rows"] if row["model"] == "R4"]
        baseline = {(row["frame"], row["repeat"]): row for row in selected}
        require(len(selected) == len(baseline) == 2 * len(roster["frames"])
                and set(baseline) == {(frame["id"], repeat) for frame in roster["frames"] for repeat in range(2)},
                "The reused R4 panel is incomplete or duplicated")
    module_path = pin(original["provider_source"]["path"], original["provider_source"]["sha256"])
    module = importlib.util.spec_from_file_location("b6_original_provider", module_path)
    provider_module = importlib.util.module_from_spec(module)
    module.loader.exec_module(provider_module)
    with closing(OwnerLock(Path(spec["neural_owner_lock"]))):
        gpu = check_gpu(spec["GPU_uuid"], spec["allocator_limit_bytes"], spec["minimum_host_available_bytes"])
        require(jax.default_backend() == "gpu", "Use the actual CUDA model path")
        _, _, receiver = accepted(spec)
        source = pinned(receiver["input"])["target_D1"]
        target = AdaptTargetDataset(source["legacy"], source["captures"], resize_with_pad=resize_with_pad)
        target.bind_state(source["statistics"])
        output.mkdir(mode=0o700)
        reused = 0 if baseline is None else len(baseline)
        write(output / "START.json", dict(gpu=gpu, panel_sha256=panel_sha,
              max_forwards=roster["max_forwards"] - reused, reused_old_forwards=reused,
              r4_manifest_sha256=original["manifest_sha256"], reuse_r4_sha256=reuse_r4_sha))
        models = {"R4": (Path(original["root"]), original["manifest_sha256"])}
        models.update({arm: (providers / arm, proof["arms"][arm]["provider_sha256"]) for arm in ARMS})
        rows = []
        for name, (root, digest) in models.items():
            manifest = read(pin(root / "CHECKPOINT.json", digest))
            if name == "R4" and baseline is not None:
                for frame in roster["frames"]:
                    noise = np.asarray(jax.random.normal(jax.random.PRNGKey(frame["noise_id"]), (1,40,32)))
                    noise_sha = hashlib.sha256(noise.tobytes()).hexdigest()
                    require(frame["observed_noise_sha256"] in (None, noise_sha), "Archived CUDA noise changed")
                    pin(frame["input"]["path"], frame["input"]["sha256"])
                    for repeat in range(2):
                        row = baseline[frame["id"], repeat]
                        artifact = Path(row["artifact"])
                        require(row["noise_sha256"] == noise_sha and artifact.name == str(artifact),
                                "The reused R4 noise or artifact path differs")
                        source = pin(reuse_r4 / artifact, row["action_sha256"])
                        shutil.copyfile(source, output / artifact)
                        require(sha(output / artifact) == row["action_sha256"], "R4 copy changed")
                        write(output / f"R4-{frame['id']}-R{repeat}.json", row)
                        rows.append(row)
                continue
            model = provider_module.provider_from_manifest(root, manifest, receiver["model"], target, receiver["tokenizer"]["path"])
            for frame in roster["frames"]:
                noise = np.asarray(jax.random.normal(jax.random.PRNGKey(frame["noise_id"]), (1,40,32)))
                noise_sha = hashlib.sha256(noise.tobytes()).hexdigest()
                require(frame["observed_noise_sha256"] in (None, noise_sha), "Archived CUDA noise changed")
                with np.load(pin(frame["input"]["path"], frame["input"]["sha256"]), allow_pickle=False) as packet:
                    previous = None
                    for repeat in range(2):
                        result = model.predict(packet["state_eef10"], [packet["body_rgb"], packet["wrist_rgb"]],
                            frame["language"], packet["T_policy_from_mount"], noise_id=frame["noise_id"],
                            previous_reference=frame["previous_command"])
                        action = result["physical_H40"]
                        require(action.shape == (40,10) and np.isfinite(action).all(), "Invalid actual H40")
                        path = output / f"{name}-{frame['id']}-R{repeat}.npz"
                        payload = dict(normalized=result["normalized"], physical_H40=action)
                        if frame["has_real_teacher_label"]:
                            payload["teacher_H40"] = GripLogitCodec(packet["T_policy_from_mount"]).decode(
                                packet["state_eef10"], np.pad(packet["physical_action7"], ((0,0),(0,25))))
                            payload["valid"] = packet["valid"]
                        with path.open("xb") as stream:
                            np.savez(stream, **payload)
                        row = dict(model=name, frame=frame["id"], phase=frame["phase"], repeat=repeat,
                            noise_sha256=noise_sha, action_sha256=sha(path), artifact=path.name,
                            g7=result["G7_gripper_speed"],
                            p10_up_request_mm=float(1000*(action[:10,2].max()-packet["state_eef10"][2])),
                            repeat_max_abs=0 if previous is None else float(np.max(np.abs(action-previous))))
                        rows.append(row)
                        write(output / f"{name}-{frame['id']}-R{repeat}.json", row)
                        previous = action
            del model
            jax.clear_caches()
            gc.collect()
        require(len(rows) == roster["max_forwards"], "Incomplete fixed panel")
        write(output / "ACTUAL.json", dict(panel_sha256=panel_sha, rows=rows,
              actual_forwards=len(rows) - reused, reused_old_forwards=reused,
              total_measured_outputs=len(rows), reuse_r4_sha256=reuse_r4_sha,
              optimizer_updates=0, native_integrations=0, complete=True))


def summarize(panel, output):
    from scipy.spatial.transform import Rotation
    from scripts.pi0291_arx5_data.native_dataset import matrix_from10

    roster, result = read(panel), read(output / "ACTUAL.json")
    require(result["complete"] and result["panel_sha256"] == sha(panel), "Incomplete or changed TRAIN panel")
    rows = {(r["model"], r["frame"], r["repeat"]): r for r in result["rows"]}
    changes = []
    for frame in roster["frames"]:
        original = rows["R4", frame["id"], 0]
        with np.load(pin(output / original["artifact"], original["action_sha256"]), allow_pickle=False) as data:
            base = data["physical_H40"]
        base_rotation = matrix_from10(base)[:, :3, :3]
        for name in ("R4", "LAMBDA_ZERO", "SHARED_L2SP"):
            row = rows[name, frame["id"], 0]
            repeated = rows[name, frame["id"], 1]
            with np.load(pin(output / row["artifact"], row["action_sha256"]), allow_pickle=False) as data:
                action = data["physical_H40"]
                delta = action-base
                angles = np.degrees(Rotation.from_matrix(matrix_from10(action)[:, :3, :3]
                                    @ np.swapaxes(base_rotation, -1, -2)).magnitude())
                change = dict(frame=frame["id"], episode=frame["episode"], phase=frame["phase"], model=name,
                    p10_xyz_rms_from_r4_mm=float(1000*np.sqrt(np.mean(delta[:10, :3]**2))),
                    p10_rotation_mean_from_r4_deg=float(angles[:10].mean()),
                    p10_jaw_mean_from_r4_mm=float(40*np.abs(delta[:10, 9]).mean()),
                    first_width_mm=float(2+40*action[0,9]),
                    p10_up_request_mm=row["p10_up_request_mm"], g7=row["g7"],
                    repeat_max_abs=repeated["repeat_max_abs"])
                if "teacher_H40" in data:
                    label = data["teacher_H40"]
                    valid = data["valid"][:10]
                    error = action[:10]-label[:10]
                    change["p10_teacher_xyz_rms_mm"] = float(1000*np.sqrt(np.mean(error[valid, :3]**2)))
                    change["p10_teacher_jaw_mae_mm"] = float(40*np.abs(error[valid,9]).mean())
                    change["teacher_p10_up_request_mm"] = float(1000*(label[:10,2].max()-label[0,2]))
                changes.append(change)
    write(output / "SUMMARY.json", dict(panel_sha256=sha(panel), actual_sha256=sha(output/"ACTUAL.json"),
          rows=changes, measurements="RAW_POLICY_ACTIONS_ONLY_NO_DYNAMIC_L1_OR_WBC_EXECUTION",
          actual_forwards=result["actual_forwards"], reused_old_forwards=result.get("reused_old_forwards", 0),
          native_integrations=0, optimizer_updates=0))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("mode", choices=("freeze", "measure", "summarize"))
    cli.add_argument("paths", nargs="+")
    cli.add_argument("--reuse-r4", type=Path)
    cli.add_argument("--reuse-r4-sha")
    args = cli.parse_args()
    if args.mode == "freeze":
        freeze(*(Path(p) for p in args.paths))
    elif args.mode == "measure":
        measure(Path(args.paths[0]), args.paths[1], *(Path(p) for p in args.paths[2:]),
                reuse_r4=args.reuse_r4, reuse_r4_sha=args.reuse_r4_sha)
    else:
        summarize(*(Path(p) for p in args.paths))
