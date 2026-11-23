"""Prepare teacher-intent examples from the six sealed TRAIN episodes. CPU only."""

import argparse
import dataclasses
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
from demonstrated_grip import SOURCE_T, events, pin, read, require, rows
from observation_stream import rigid
from scipy.spatial.transform import Rotation, Slerp
from visual_lift_panel import S0, panel, write_new


def checksum(path):
    with path.open("rb") as f:
        h = hashlib.sha256()
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def raw_arm(world_tcp, origin, tool, width):
    local = np.linalg.inv(origin) @ world_tcp @ np.linalg.inv(tool)
    quat = Rotation.from_matrix(local[..., :3, :3]).as_quat()[..., [3, 0, 1, 2]]
    quat = np.where(quat[..., :1] < 0, -quat, quat)
    return np.concatenate(
        (local[..., :3, 3], quat, np.asarray(width)[..., None]), axis=-1
    )


def task_labels(nominal, origin, tool, left):
    require(nominal.shape == (40, 10) and np.isfinite(nominal).all(), "teacher H40")
    x, y = nominal[:, 3:6], nominal[:, 6:9]
    rot = np.stack([x, y, np.cross(x, y)], axis=1)
    require(
        np.allclose(rot @ rot.transpose(0, 2, 1), np.eye(3), atol=1e-7, rtol=0)
        and np.all((nominal[:, 9] >= 0) & (nominal[:, 9] <= 1)),
        "nominal pose or width",
    )
    # Exact ratio of the two rates, including all coincident 20/30 Hz knots.
    phase = np.arange(50) * 2 / 3
    knots = np.arange(40)
    world = np.broadcast_to(np.eye(4), (50, 4, 4)).copy()
    world[:, :3, 3] = np.stack(
        [np.interp(phase, knots, v) for v in nominal[:, :3].T], axis=1
    )
    world[:, :3, :3] = Slerp(knots, Rotation.from_matrix(rot))(phase).as_matrix()
    width = nominal[(np.arange(50) * 2) // 3, 9]
    right = raw_arm(world, origin, tool, width)
    return np.column_stack((np.broadcast_to(left, (50, 8)), right)).astype(
        np.float32
    ), world


def history(tick):
    return [0, *[max(0, tick - 500 * k) for k in range(8, 0, -1)], tick]


def stereo_bank(ep, info, queries, dense, receipt):
    ticks = [q["native_step"] for q in queries]
    required = {t for tick in ticks for t in history(tick)}
    bank = {}
    offset = 0
    for shard in info["frame_shards"]:
        require(shard["first_frame"] == offset, "original RGB order")
        with np.load(pin(ep.parent.parent / shard["file"], shard["sha256"])) as z:
            ts, rgb = z["native_step"], z["rgb"]
        require(
            list(ts) == ticks[offset : offset + len(ts)]
            and len(ts) == shard["frames"]
            and rgb.dtype == np.uint8
            and rgb.shape == (len(ts), 2, 256, 256, 3),
            "original RGB",
        )
        bank.update(zip(map(int, ts), rgb))
        offset += len(ts)
    require(offset == len(queries), "incomplete stereo recordings")
    missing = required - bank.keys()
    frames = rows(pin(dense / "frames.jsonl", receipt["frames_sha256"]))
    # Check all original stereo overlaps against the pixel digests from the independent replay.
    for f in frames:
        if f["tick"] in bank:
            require(
                f["rgb_sha256"]
                == [hashlib.sha256(x.tobytes()).hexdigest() for x in bank[f["tick"]]],
                "original and dense RGB differ",
            )
    offset = 0
    for shard in receipt["shards"]:
        group = frames[offset : offset + shard["count"]]
        if any(f["tick"] in missing for f in group):
            with np.load(pin(dense / shard["name"], shard["sha256"])) as z:
                ts, rgb = z["tick"], z["rgb"]
            require(list(ts) == [f["tick"] for f in group], "dense shard clock")
            for f, im in zip(group, rgb):
                if f["tick"] in missing:
                    require(
                        f["rgb_sha256"]
                        == [hashlib.sha256(x.tobytes()).hexdigest() for x in im],
                        "dense historical pixels",
                    )
                    bank[f["tick"]] = im.copy()
        offset += len(group)
    require(offset == len(frames) and required == bank.keys(), "missing causal history")
    keys = sorted(bank)
    return np.array(keys), np.stack([bank[k] for k in keys]), len(missing)


class TargetExamples:
    """One prepared episode. Model examples have no reference, contact, phase or outcome fields."""

    def __init__(self, folder):
        with np.load(folder / "sensor_action.npz", allow_pickle=False) as z:
            self.data = {k: z[k] for k in z.files}
        self.meta = read(folder / "episode.json")

    def __len__(self):
        return len(self.data["query_tick"])

    def __getitem__(self, i):
        d = self.data
        im = d["stereo"]
        indices = d["history_indices"][i]
        return {
            "observation.images.cam_high": im[indices, 0],
            "observation.images.cam_right_wrist": im[indices[-1], 1],
            "observation.images.cam_left_wrist": np.zeros((480, 640, 3), np.uint8),
            "observation.state": d["state"][i].copy(),
            "action": d["nominal_action"][i].copy(),
            "prompt": self.meta["language"],
            "task_index": self.meta["task_index"],
        }


def prepare_episode(root, dest, bundle, pins, reg, task_index):
    ep, info, queries, receipt, windows = bundle
    name = info["id"]
    dest.mkdir()
    dense = root / "v12/DENSE" / name
    keys, images, historic = stereo_bank(ep, info, queries, dense, receipt)
    native = ep.parents[3] / "cases" / name / "native-01"
    atomfile = native / "MOBILE_SAMPLES.private.jsonl"
    atoms = rows(pin(atomfile, pins[str(atomfile)]))
    end = {a["native_end_step"]: a for a in atoms}
    require(len(end) == len(atoms), "duplicate completed native interval")
    offset = np.eye(4)
    offset[:3, 3] = reg["source_origin"]
    tool = rigid(reg["T_link6_TCP"])
    left = np.array(next(iter(windows.values()))["state"][:8])
    states, actions, history_indices, ticks = [], [], [], []
    expected_world, nominal, installed, intervals = [], [], [], []
    phases, ev_count = Counter(), Counter()
    latency = []
    checked_old = 0
    with (dest / "native_audit.jsonl").open("x") as audit:
        for i, q in enumerate(queries):
            tick = q["native_step"]
            a = end[tick]
            require(
                q["query"] == i
                and tick == 5703 + 250 * i
                and a["ticks"] == 25
                and a["native_start_step"] == tick - 25
                and abs(q["native_time_s"] - tick * 0.002) < 1e-8,
                "query/previous interval clock",
            )
            tip = rigid(q["actual_state"]["tcp"])
            origin = rigid(a["actual_mount"]) @ offset
            fingers = np.array(q["actual_state"]["finger_q"])
            prev = q["previous_confirmed_command"]
            require(
                np.allclose(tip, a["actual_TCP"], atol=1e-9, rtol=0)
                and fingers.shape == (2,)
                and abs(prev["width_m"] - 0.044 * a["installed_EEF10"][9]) < 1e-9
                and abs(prev["source_native_s"] - (tick - 25) * 0.002) < 1e-8,
                "previous command/measured TCP",
            )
            state = np.r_[
                left, raw_arm(tip, origin, tool, fingers.mean() / 0.044)
            ].astype(np.float32)
            require(0 <= state[15] <= 1, "invalid measured gripper")
            nominal_h = np.asarray(q["nominal_world_H40"])
            installed_h = np.asarray(q["installed_reference_H40"])
            require(
                installed_h.shape == (40, 10)
                and q["supervision_mask"] == [True] * 40
                and np.allclose(tip[:3, 3], nominal_h[0, :3], atol=1e-9, rtol=0)
                and q["object_phase_memory_in_student_inputs"] is False,
                "nominal label identity",
            )
            raw, world = task_labels(nominal_h, origin, tool, left)
            hs = history(tick)
            ix = np.searchsorted(keys, hs)
            require(list(keys[ix]) == hs, "missing history frame")
            if tick in windows:
                w = windows[tick]
                with np.load(
                    pin(
                        root / "v12/STUDENT_INPUT" / (w["id"] + ".npz"),
                        w["input_sha256"],
                    )
                ) as old:
                    require(
                        np.array_equal(images[ix, 0], old["head"])
                        and np.array_equal(images[ix[-1], 1], old["right_wrist"])
                        and np.array_equal(state, old["state"])
                        and str(old["prompt"]) == info["language"]
                        and np.allclose(
                            origin, w["T_W_source_origin"], atol=1e-10, rtol=0
                        ),
                        "old official query differs",
                    )
                checked_old += 1
            confirmed = []
            for step in q["intervals"]:
                j = step["atom"]
                t = tick + 25 * j
                c = end[t + 25]
                require(
                    step["start_native_step"] == c["native_start_step"] == t
                    and step["end_native_step"] == c["native_end_step"] == t + 25
                    and step["confirmed_ticks"] == c["ticks"] == 25
                    and c["safety_valid"]
                    and np.array_equal(installed_h[j], c["installed_EEF10"])
                    and step["installed_reference_sha256"]
                    == c["installed_reference_sha256"],
                    "teacher native interval identity",
                )
                confirmed.append(
                    {
                        "index": j,
                        "start": t,
                        "end": t + 25,
                        "wire": c["installed_reference_sha256"],
                        "solver_target_identity": step["boundary"][
                            "solver_target_identity"
                        ],
                        "installed_EEF10": c["installed_EEF10"],
                        "achieved_tcp": c["actual_TCP"],
                        "gripper_sequence": c["gripper_command"]["sequence"],
                        "controller_selected_reference": c[
                            "controller_selected_reference"
                        ],
                    }
                )
            mask = np.array(q["consumed_mask"], bool)
            require(
                mask.shape == (40,)
                and list(np.flatnonzero(mask)) == [c["index"] for c in confirmed],
                "confirmed interval mask mismatch",
            )
            original = events(np.arange(40) * 0.05, nominal_h[:, 9], step=True)
            sampled = events(SOURCE_T, raw[:, 15].astype(float), step=True)
            inside = [e for e in original if e["s"] <= SOURCE_T[-1]]
            require(
                [e["kind"] for e in sampled] == [e["kind"] for e in inside],
                "lost in-range gripper event",
            )
            latency.extend(b["s"] - a0["s"] for a0, b in zip(inside, sampled))
            ev_count.update(
                in_support=len(inside), after_support=len(original) - len(inside)
            )
            color = info["task"].split("-")[1]
            audit.write(
                json.dumps(
                    {
                        "tick": tick,
                        "phase": q["phase"],
                        "raw_task_intent": q["raw_intent"],
                        "measured_tcp": tip.tolist(),
                        "measured_fingers_m": fingers.tolist(),
                        "previous_command": prev,
                        "previous_wire": a["installed_reference_sha256"],
                        "origin": origin.tolist(),
                        "target_color": color,
                        "object_audit": q["actual_state"]["objects"][color],
                        "confirmed_native": confirmed,
                    },
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
            phases[q["phase"]] += 1
            states.append(state)
            actions.append(raw)
            ticks.append(tick)
            history_indices.append(ix)
            nominal.append(nominal_h)
            installed.append(installed_h)
            intervals.append(mask)
            expected_world.append(world)
    require(checked_old == len(windows), "missing old policy query")
    np.savez_compressed(
        dest / "sensor_action.npz",
        query_tick=ticks,
        stereo_tick=keys,
        stereo=images,
        history_indices=history_indices,
        state=states,
        nominal_action=actions,
    )
    np.savez_compressed(
        dest / "task_audit.npz",
        nominal_world_H40=nominal,
        installed_reference_H40=installed,
        confirmed_native_interval_mask=intervals,
        nominal_world_at_source_times=expected_world,
        source_clock_s=SOURCE_T,
        nominal_clock_s=np.arange(40) * 0.05,
        link6_tcp=tool,
    )
    write_new(
        dest / "episode.json",
        {
            "episode": name,
            "language": info["language"],
            "task_index": task_index,
            "family": info["family"],
            "split": info["split"],
            "seed": info["reset"],
            "scene_sha256": info["scene_sha256"],
            "individual_teacher_pass": info["individual_pass"],
            "episode_sha256": checksum(ep),
            "original_queries_sha256": receipt["source_files"][
                "LIVE_DATA/QUERIES.private.json"
            ],
            "atoms_sha256": pins[str(atomfile)],
            "task": info["task"],
        },
    )
    gap = np.linalg.norm((np.array(nominal) - installed)[:, :, :3], axis=-1) * 1000
    return {
        "queries": len(ticks),
        "historical_extra_stereo": historic,
        "existing_calls_verified": checked_old,
        "phases": dict(phases),
        "events": dict(ev_count),
        "max_gripper_event_delay_s": max(latency, default=0),
        "confirmed_native_intervals": int(np.sum(intervals)),
        "teacher_pass": info["individual_pass"],
        "nominal_to_installed_mm_p50_p95_max": np.percentile(
            gap, [50, 95, 100]
        ).tolist(),
        "artifacts": {p.name: checksum(p) for p in sorted(dest.iterdir())},
    }


def validate_episode(folder, data, stats, decoder):
    from openpi import transforms
    from openpi.training.data_loader import transform_dataset

    ds = TargetExamples(folder)
    transformed = transform_dataset(ds, dataclasses.replace(data, norm_stats=stats))
    physical = transforms.compose(
        [*data.repack_transforms.inputs, *data.data_transforms.inputs]
    )
    low, high = (
        np.asarray(q)[..., 7:14] for q in (stats["actions"].q01, stats["actions"].q99)
    )
    require(low.shape[-1] == high.shape[-1] == 7, "checkpoint right action statistics")
    audit = rows(folder / "native_audit.jsonl")
    with np.load(folder / "task_audit.npz") as z:
        expected, tool = z["nominal_world_at_source_times"], z["link6_tcp"]
    errors = np.zeros(3)
    below_counts = np.zeros(7, dtype=int)
    above_counts = np.zeros(7, dtype=int)
    at_nominal_full_open = 0
    any_out = 0
    for i in range(len(ds)):
        batch = transformed[i]
        act = batch["actions"]
        require(
            act.shape == (50, 32)
            and batch["state"].shape == (32,)
            and np.isfinite(act).all()
            and np.all(act[:, 14:] == 0)
            and batch["extra_images"].shape == (9, 224, 224, 3)
            and all(x.shape == (224, 224, 3) for x in batch["image"].values())
            and all(batch["image_mask"].values())
            and batch["tokenized_prompt_mask"].any(),
            "official training interface",
        )
        labels = physical(ds[i])["actions"][:, 7:14]
        outside = (labels < low) | (labels > high)
        below_counts += (labels < low).sum(axis=0)
        above_counts += (labels > high).sum(axis=0)
        at_nominal_full_open += int(
            np.count_nonzero(ds.data["nominal_action"][i, :, 15] == 1)
        )
        any_out += int(outside.any())
        raw = decoder({"state": batch["state"].copy(), "actions": act.copy()})[
            "actions"
        ]
        r = raw[:, 8:]
        frame = np.broadcast_to(np.eye(4), (50, 4, 4)).copy()
        frame[:, :3, 3] = r[:, :3]
        frame[:, :3, :3] = Rotation.from_quat(r[:, [4, 5, 6, 3]]).as_matrix()
        actual = np.array(audit[i]["origin"]) @ frame @ tool
        pos_mm = (
            np.linalg.norm(actual[:, :3, 3] - expected[i, :, :3, 3], axis=1).max()
            * 1000
        )
        relative = actual[:, :3, :3].transpose(0, 2, 1) @ expected[i, :, :3, :3]
        angle_deg = Rotation.from_matrix(relative).magnitude().max() * 180 / np.pi
        width_mm = abs(r[:, 7] - ds.data["nominal_action"][i, :, 15]).max() * 40
        errors = np.maximum(errors, [pos_mm, angle_deg, width_mm])
    require(
        np.isfinite(errors).all() and np.all(errors < [0.01, 0.001, 0.001]),
        "teacher intent round-trip drift",
    )
    return {
        "official_rows": len(ds),
        "right_axis_below_q01": below_counts.tolist(),
        "right_axis_above_q99": above_counts.tolist(),
        "at_nominal_full_open": at_nominal_full_open,
        "windows_any_outside": any_out,
        "max_round_trip_mm_deg_width_mm": errors.tolist(),
    }


def run(root, out):
    from checkpoint_inputs import configured_transforms
    from openpi import transforms

    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        and os.environ.get("JAX_PLATFORMS") == "cpu",
        "CPU only",
    )
    cfg, data, stats = configured_transforms(root)
    require(
        cfg.model.action_horizon == 50 and not cfg.model.extra_frames_first,
        "wrong model contract",
    )
    decoder = transforms.compose(
        [
            *data.model_transforms.outputs,
            transforms.Unnormalize(stats, use_quantiles=True),
            *data.data_transforms.outputs,
        ]
    )
    pins = read(pin(root / "v13/run01/S0_ACTUAL.json", S0))["inputs"]
    reg_path = root / "v12/TARGET_REGISTRATION.json"
    reg = read(pin(reg_path, pins[str(reg_path)]))
    tasks = ["HOLD-blue", "HOLD-red", "PLACE-red-right"]
    episodes = {}
    require(not out.exists(), "output already exists")
    with tempfile.TemporaryDirectory(dir=out.parent, prefix=out.name + "-") as tmp:
        work = Path(tmp) / "panel"
        work.mkdir()
        for bundle in panel(root):
            info = bundle[1]
            folder = work / info["id"]
            result = prepare_episode(
                root, folder, bundle, pins, reg, tasks.index(info["task"])
            )
            result.update(validate_episode(folder, data, stats, decoder))
            episodes[info["id"]] = result
            print(info["id"], result["official_rows"], flush=True)
        require(
            sum(r["existing_calls_verified"] for r in episodes.values()) == 16,
            "panel identity",
        )
        write_new(
            work / "result.json",
            {
                "id": "V7-DATA-01",
                "episodes": episodes,
                "raw_input": "two-camera/one-arm",
                "right_supervision_model_indices": list(range(7, 14)),
                "checkpoint_right_action_q01": np.asarray(stats["actions"].q01)[
                    ..., 7:14
                ].tolist(),
                "checkpoint_right_action_q99": np.asarray(stats["actions"].q99)[
                    ..., 7:14
                ].tolist(),
                "missing_left": "virtual left state and image; no physical left target; vanilla loss has no right-only mask",
                "camera_history": "reset, t-8..t-1 seconds, current; body at all ten, real wrist at current",
                "label_kind": "current nominal plan, 50 rows at 30Hz; never an executed 50-row sequence",
                "new_forwards": 0,
                "new_integrations": 0,
                "optimizer_updates": 0,
            },
        )
        work.rename(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    run(args.root.resolve(), args.out.resolve())
