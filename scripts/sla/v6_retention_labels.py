"""Compare nominal and consumed labels over the exact old 52-episode population."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


def same(path, expected):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Label source changed: {path.name}")
    return path


def compare(archive, consumed, new_manifest, output):
    from go2arx5_vla.sla_adapter.gripper_v2 import GripLogitCodec
    from scripts.pi0291_arx5_data.native_dataset import matrix_from10
    from scipy.spatial.transform import Rotation

    seal = json.loads((archive / "SEALED.private.json").read_bytes())
    manifest = same(archive / "MANIFEST.private.json", seal["manifest_sha256"])
    trust = {r["id"]: r for r in json.loads(manifest.read_bytes())["episodes"]}
    data_manifest = json.loads((consumed / "MANIFEST.json").read_bytes())
    used = data_manifest["episodes"]
    if len(used) != 52 or data_manifest["signed_original_manifest_sha256"] != seal["manifest_sha256"]:
        raise ValueError("Expected the actual fixed 52 training episodes")
    phases = defaultdict(list)
    first_rows = {"fit": [], "check": []}
    total = 0
    for episode_index, item in enumerate(used):
        old = trust[item["episode"]]
        nominal_path = same(archive / old["labels"]["file"], old["labels"]["sha256"])
        real_path = same(consumed / item["labels"], item["sha256"])
        codec = GripLogitCodec(old["mount"]["T_policy_from_mount"])
        with np.load(nominal_path, allow_pickle=False) as f, np.load(real_path, allow_pickle=False) as r:
            nominal = f["nominal_policy_H40"]
            anchors = f["state_eef10"]
            target, valid, phase = (r[k] for k in ("action", "valid", "phase"))
            ticks = r["native_step"]
            if (not np.array_equal(f["native_step"], ticks) or not np.array_equal(f["phase"], phase)
                    or not np.array_equal(ticks, 5703 + 250*np.arange(len(nominal))) or not valid[:, :10].all()):
                raise ValueError("Same archived 2 ms query clock or mask changed")
            for i, anchor in enumerate(anchors):
                active = valid[i]
                raw = np.pad(target[i], ((0, 0), (0, 25)))
                real = codec.decode(anchor, raw)[active]
                plan = nominal[i][active]
                first_rows["fit" if episode_index % 2 == 0 else "check"].append(
                    (matrix_from10(plan[:1])[0], matrix_from10(real[:1])[0]))
                xyz = np.linalg.norm(real[:, :3] - plan[:, :3], axis=1) * 1000
                angle = Rotation.from_matrix(matrix_from10(real)[:, :3, :3]
                    @ matrix_from10(plan)[:, :3, :3].swapaxes(-1, -2)).magnitude() * 180 / np.pi
                jaw = np.abs(real[:, 9] - plan[:, 9]) * 40
                for ext, sl in (("first", slice(0, 1)), ("consumed_P10", slice(0, 10)),
                                ("later_queries", slice(10, None))):
                    phases[str(phase[i]), ext].append(np.column_stack((xyz, angle, jaw))[sl])
                total += 1
    stats = []
    for (phase, scope), measurements in sorted(phases.items()):
        v = np.concatenate(measurements, axis=0)
        if not len(v):
            continue
        q = np.percentile(v, [50, 95], axis=0)
        stats.append(dict(phase=phase, scope=scope, actual_atoms=len(v),
                          position_mm=q[:, 0].tolist(), rotation_deg=q[:, 1].tolist(), grip_mm=q[:, 2].tolist()))
    newer = json.loads(new_manifest.read_bytes())
    expanded = Counter(r["task"] for r in newer["episodes"])
    previous = Counter(r["task"] for r in used)
    fit, test = (np.asarray(first_rows[key]) for key in ("fit", "check"))
    plan, real = fit[:, 0], fit[:, 1]
    rotation = Rotation.from_matrix(real[:, :3, :3] @ plan[:, :3, :3].swapaxes(-1, -2)).mean().as_matrix()
    translation = (real[:, :3, 3] - plan[:, :3, 3] @ rotation.T).mean(axis=0)
    plan, real = test[:, 0], test[:, 1]
    raw = np.linalg.norm(real[:, :3, 3] - plan[:, :3, 3], axis=1) * 1000
    adjusted = np.linalg.norm(real[:, :3, 3] - plan[:, :3, 3] @ rotation.T - translation, axis=1) * 1000
    raw_rot = Rotation.from_matrix(real[:, :3, :3] @ plan[:, :3, :3].swapaxes(-1, -2)).magnitude()
    adjusted_rot = Rotation.from_matrix(
        real[:, :3, :3] @ (rotation @ plan[:, :3, :3]).swapaxes(-1, -2)).magnitude()
    check = dict(fit_windows=len(fit), held_episode_windows=len(test), transform_translation_mm=(1000*translation).tolist(),
        transform_angle_deg=float(np.degrees(Rotation.from_matrix(rotation).magnitude())),
        held_raw_position_mm=np.percentile(raw, [50, 95]).tolist(),
        held_adjusted_position_mm=np.percentile(adjusted, [50, 95]).tolist(),
        held_raw_rotation_deg=np.percentile(np.degrees(raw_rot), [50, 95]).tolist(),
        held_adjusted_rotation_deg=np.percentile(np.degrees(adjusted_rot), [50, 95]).tolist())
    with output.open("x") as sink:
        json.dump(dict(windows=total, episodes=len(used), original_tasks=dict(previous),
                       core_64_tasks=dict(expanded), percentile_order=[50, 95], numbers=stats,
                       fixed_transform_hypothesis=check,
                       native_integrations=0, model_forwards=0, optimizer_updates=0), sink, indent=2)
        sink.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("archive", "consumed", "new_manifest", "output"):
        parser.add_argument(arg, type=Path)
    compare(**vars(parser.parse_args()))
