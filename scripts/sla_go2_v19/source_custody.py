"""Freeze the SEA-0 measured anchor roster using the original archived inputs."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "scripts/sla_hierarchical_v16"))
from demonstrated_grip import pin, read, require
from policy_query import input_identity
from visual_lift_panel import write_new

REV = "1127bda94f172d757386945af3eb3a6754d1a4e6"
SOURCE_H5 = {
    "episode_0000000.hdf5": "515dcd5a3541ffa4e5563504d2e6746f10732c49aeba866760ba75db7c14c456",
    "episode_0000001.hdf5": "8b7c9e3784cef53a55bf9f2cfe0b3968036e957a622be50bd76665bc708d45ec",
}
O0 = BASE / "reports/SLA_OFFICIAL_GO2_TRANSFER/O0_ACTUAL.json"
VIEWS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
SEEDS = ([0, 0], [0, 1])
SOURCE_ARMS = ("REAL_SOURCE", "SOURCE_TWIN", "GO2_TWIN", "CROSSPAINT_RECOVERY")
TARGET_ARMS = ("RAW", "ORACLE", "DEPLOYABLE", "LEFT_MASK_ONLY")
DATASET = "RoboDojo-Benchmark/RoboDojo"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def array_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def source_history(frame):
    return [0, *(max(0, frame-i*30) for i in range(8, 0, -1)), frame]


def source_motion(h, frame, side, mode):
    # Fixed half-second window with both endpoints. No prediction is read here.
    ee = h[f"{mode}/{side}_ee_poses"][frame:frame+16]
    grip = h[f"{mode}/{side}_ee_joint_states"][frame:frame+16, 0]
    require(ee.shape == (16, 7) and grip.shape == (16,) and np.isfinite(ee).all()
            and np.isfinite(grip).all()
            and np.allclose(np.linalg.norm(ee[:, 3:], axis=1), 1, rtol=0, atol=1e-6),
            "missing source half-second pose or gripper evidence")
    cos = np.clip(np.abs(ee[:, 3:] @ ee[0, 3:]), 0, 1)
    return dict(
        travel_mm=float(1000*np.max(np.linalg.norm(ee[:, :3]-ee[0, :3], axis=1))),
        angle_deg=float(2*np.rad2deg(np.arccos(cos)).max()),
        grip_signal_range=float(np.ptp(grip)),
    )


def real_source_observation(h, row, archived):
    from XPolicyLab.utils.process_data import decode_image_bit

    frame = row["frame"]
    hist = source_history(frame)
    head = np.stack([decode_image_bit(h["vision/cam_head/colors"][i]) for i in hist])
    wrists = [decode_image_bit(h[f"vision/{view}/colors"][frame]) for view in VIEWS[1:]]
    state = np.concatenate([
        h[f"state/{side}_{field}"][frame]
        for side in ("left", "right") for field in ("ee_poses", "ee_joint_states")
    ]).astype(np.float32)
    prompt = h["instruction"].asstr()[()]
    require(
        head.shape == (10, 480, 640, 3) and head.dtype == np.uint8
        and all(x.shape == (480, 640, 3) and x.dtype == np.uint8 for x in wrists)
        and hist == archived["head_frames"] and [array_hash(x) for x in head] == archived["head_sha256"]
        and dict(zip(VIEWS, map(array_hash, (head[-1], *wrists)))) == archived["current_rgb_sha256"]
        and np.array_equal(state, np.asarray(archived["state"], dtype=np.float32))
        and prompt == archived["prompt"], "source cadence, RGB, state or prompt differs from O0")
    return dict(images=dict(cam_high=head, cam_left_wrist=wrists[0], cam_right_wrist=wrists[1]),
                state=state, prompt=prompt)


def replay_real_input(h, row, archived):
    return input_identity(real_source_observation(h, row, archived))[0]


def freeze_source(root):
    import h5py

    o0 = read(O0)
    inputs = root / "evidence/O0_PANEL_01"
    anchors, candidates, datasets, geometry = [], [], {}, {}
    journal = {r["call"]: r for r in (
        json.loads(line) for line in (inputs / "calls.jsonl").read_text().splitlines()
    ) if r["event"] == "FORWARD_RETURNED"}
    for episode, sha in SOURCE_H5.items():
        with h5py.File(pin(root / "data" / episode, sha)) as h:
            frequency = int(h["additional_info/frequency"][()])
            require(frequency == 30, "source data format or FPS changed")
            datasets[episode] = {
                "sha256": sha, "frames": h["state/right_arm_joint_states"].shape[0],
                "fps": frequency,
                "camera_fields": {view: sorted(h[f"vision/{view}"]) for view in VIEWS},
                "recording_calibration": sorted(h["additional_info"]),
            }
            count = 0
            eligible_rows = sorted((x for x in o0["source"] if x["episode"] == episode
                                    and x["mode"] == "primary"), key=lambda x: x["frame"])
            for row in eligible_rows:
                frame = row["frame"]
                right = source_motion(h, frame, "right", "state")
                left = {mode: source_motion(h, frame, "left", mode) for mode in ("state", "action")}
                eligible = right["travel_mm"] >= 5 and all(
                    x["travel_mm"] <= 1 and x["angle_deg"] <= .5
                    and x["grip_signal_range"] <= .005 for x in left.values()
                )
                chosen = eligible and count < 2
                candidates.append(dict(episode=episode, frame=frame, selected=chosen,
                                       eligible_motion_screen=eligible, right=right, left=left))
                if not chosen:
                    continue
                count += 1
                archive_path = inputs / f"call_{row['call']:02d}_input.json"
                archived = read(archive_path)
                require((archived["episode"], archived["frame"], archived["mode"])
                        == (episode, frame, "primary"), "O0 input custody changed")
                observed = replay_real_input(h, row, archived)
                entry = journal[row["call"]]
                result = pin(inputs / f"call_{row['call']:02d}_actions.npz", row["output_sha256"])
                with np.load(result, allow_pickle=False) as z:
                    require(entry["file_sha256"] == row["output_sha256"]
                            and array_hash(z["native_absolute_arms"]) == entry["native_sha256"],
                            "O0 decoded output changed")
                    raw_retained = {"actions", "sampler_state"} <= set(z.files)
                key = f"S{len(anchors):02d}"
                geometry[key] = {
                    "frames": source_history(frame),
                    **{side: {
                        "joint_q": [h[f"state/{side}_arm_joint_states"][i].tolist()
                                    for i in source_history(frame)],
                        "recorded_link6": [h[f"state/{side}_ee_poses"][i].tolist()
                                           for i in source_history(frame)],
                    } for side in ("left", "right")},
                }
                anchors.append(dict(
                    id=key, episode=episode, frame=frame, history_frames=source_history(frame),
                    input_sha256=observed, source_h5_sha256=sha, o0_call=row["call"],
                    record_seed=archived["prng_key_before_split"], o0_input_receipt_sha256=digest(archive_path),
                    o0_archive_sha256=row["output_sha256"], o0_decoded_sha256=entry["native_sha256"],
                    raw_sampler_archive=raw_retained,
                    source_rmse_mm=row["right_position_vector_rms_mm"],
                    original_hold_rmse_mm=row["hold_current_state_rms_mm"],
                    left_local_support="KINEMATICALLY_QUIET_ONLY",
                ))
    return anchors, candidates, datasets, geometry


def target_case(root, layout):
    folder = "Z1_REACH_483" if layout == "G00" else "Z1_REACH_483_G01"
    return root / "v18" / folder / f"{layout}-A"


def target_obs(path):
    with np.load(path, allow_pickle=False) as z:
        return dict(images=dict(cam_high=z["head"], cam_right_wrist=z["wrist"],
                                cam_left_wrist=np.zeros((480, 640, 3), np.uint8)),
                    state=z["state"], prompt=str(z["prompt"]))


def target_anchors(root):
    anchors, private = [], {}
    for layout in ("G00", "G01"):
        case = target_case(root, layout)
        spec = read(case / "EXECUTION_SPEC.json")
        lease = read(case / "control/LEASE.json")
        require(lease["status"] == "CLOSED" and lease["spec_sha256"] == digest(case / "EXECUTION_SPEC.json"),
                "target archived owner identity changed")
        scene = Path(spec["model_path"])
        scene_sha = spec["frozen_files"][str(scene)]
        pin(scene, scene_sha)
        native = case / read(case / "NATIVE_REQUEST.json")["episode"]
        for i in range(2):
            q = case / f"query-{i:02d}"
            request = read(q / "REQUEST.json")
            obs = target_obs(q / "INPUT.npz")
            observed, _ = input_identity(obs)
            tick = request["tick"]
            require(observed == request["input_sha256"] and request["episode"] == native.name
                    and tick == [6203, 8203][i] and request["number"] == i
                    and obs["images"]["cam_high"].shape == (10, 256, 256, 3),
                    "wrong archived paused Go2 capture")
            key = f"{layout}-{i:02d}"
            anchors.append(dict(
                id=key, layout=layout, tick=tick, input_sha256=observed,
                input_file_sha256=digest(q / "INPUT.npz"), scene_sha256=scene_sha,
                owner_spec_sha256=lease["spec_sha256"],
                history_ticks=[0, *range(tick-4000, tick+1, 500)],
                original_return_present=(q / "RETURN.json").is_file(),
            ))
            private[key] = dict(case=str(case), native=str(native), scene=str(scene), query=str(q))
    return anchors, private


def cache_entry(q, kind, seed, expected, receipt, path, *, mask_receipt=None):
    r = read(receipt)
    require(r["input_sha256"] == expected and r.get("seed", seed) == seed,
            "historical cache receipt changed")
    f = pin(path, r["output_sha256"])
    with np.load(f, allow_pickle=False) as sample:
        require(sample["actions"].shape == (50, 32) and sample["absolute"].shape == (50, 16)
                and sample["sampler_state"].shape == (32,)
                and all(np.isfinite(sample[k]).all() for k in ("actions", "absolute", "sampler_state")),
                "cache lacks the actual sampler return")
    result = dict(id=q, arm=kind, seed=seed, input_sha256=expected, output_sha256=r["output_sha256"],
                  receipt_sha256=digest(receipt), exact_archive_candidate=True)
    if kind == "LEFT_MASK_ONLY":
        require(mask_receipt is not None and read(mask_receipt) == {
            "base_0_rgb": True, "left_wrist_0_rgb": False, "right_wrist_0_rgb": True,
        }, "archived sampler lacks its missing-left mask")
        result["sampler_mask_sha256"] = digest(mask_receipt)
    return result


def target_cache(root, anchors, private):
    found = []
    for target in anchors:
        key = target["id"]
        q = Path(private[key]["query"])
        req = read(q / "REQUEST.json")
        if target["original_return_present"]:
            found.append(cache_entry(key, "RAW", req["seed"], target["input_sha256"],
                                     q / "RETURN.json", q / "OUTPUT.npz"))
        if target["layout"] == "G01":
            i = int(key[-2:])
            for folder, arm, seed in (
                ("Z1_CROSSED_486", "RAW", [[0, 1], [0, 0]][i]),
                ("Z1_MASK_489", "LEFT_MASK_ONLY", [0, 0]),
            ):
                path = root / "v18" / folder
                closed = read(path / "EXIT.json")
                require(closed["entered"] == closed["returned"] == 2
                        and closed["simulation_ticks"] == closed["updates"] == 0,
                        "historical sampler batch still unresolved")
                masks = path / f"mask-{i}.json" if arm == "LEFT_MASK_ONLY" else None
                found.append(cache_entry(key, arm, seed, target["input_sha256"],
                                         path / f"returned-{i}.json", path / f"cross-{i}.npz",
                                         mask_receipt=masks))
    return found


def audit(root, out):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU-only archive reader")
    source, candidates, datasets, source_geometry = freeze_source(root)
    target, private = target_anchors(root)
    cache = target_cache(root, target, private)
    plan = [
        dict(id=r["id"], arm=arm, seed=seed,
             planned_status="RENDER_OR_CALIBRATION_PENDING" if arm not in {"RAW", "LEFT_MASK_ONLY", "REAL_SOURCE"}
             else ("ARCHIVE_CANDIDATE" if any(
                 (x["id"], x["arm"], x["seed"]) == (r["id"], arm, seed) for x in cache
             ) else "NO_EXACT_ARCHIVED_SAMPLE"))
        for seed in SEEDS for population, arms in ((source, SOURCE_ARMS), (target, TARGET_ARMS))
        for r in population for arm in arms
    ]
    out.mkdir(mode=0o700)
    write_new(out / "TARGET_GEOMETRY_INPUTS.private.json", private)
    write_new(out / "SOURCE_GEOMETRY_INPUTS.private.json", source_geometry)
    write_new(out / "SEA0_PAIRED_ROSTER.json", dict(
        status="ANCHORS_FROZEN_RENDER_NOT_YET_QUALIFIED", source=source, go2=target,
        expected_source_anchors=4, source_selection=candidates, conditions=plan,
        exact_archive_candidates=cache, independent_confirmation_states=0,
        source_head_history_contract="initial frame + i-8s..i-1s + current, clamp to frame zero",
        new_model_forwards=0, optimizer_updates=0, integration_ticks=0, science_allocated=False,
    ))
    write_new(out / "SEA0_SOURCE_CUSTODY.json", dict(
        dataset=DATASET, revision=REV, license="apache-2.0", source_files=datasets,
        checkpoint_revision=read(O0)["checkpoint"]["revision"],
        left_note="Local motion screen only. The demonstration instruction can require both arms.",
        unknown_source=("real camera intrinsics and time-aligned extrinsics", "real RGB-D and scene poses",
                        "real finger-gap calibration", "real flange-to-common-tool transform"),
        real_source_pose_render_qualified=False, target_heldout_used=False,
        real_hardware_requests=0, new_model_forwards=0, optimizer_updates=0, integration_ticks=0,
    ))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    audit(args.root.resolve(), args.output.resolve())
