"""PI0288 recorded commands to relative-SE3 H40 and official CPU admission."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from ..target_sim import dataset as engine_independent
from ..target_sim.dataset import (POLICY_FIELDS, ROT6D_LAYOUT, build_processor_files,
    configure_cpu, enforce_roundtrip, fit_statistics, image_arrays, legacy_codec_module,
    official_feature, physical_errors, pose10, require, sha, training_population, write_json)
from ..target_sim.recorder import _native_pose
from ..trajectory.so3 import geodesic_angle, quat_to_matrix, rotation_matrix_to_rot6d_rows
from .recorder import IDENTITIES, LABEL_SOURCE, validate_sample
from .scorer import score_episode
from .teacher import require_gripper_spec


def windows(rows):
    """Index i..i+39, offsets0..1.95;641 samples permit602 windows.

    Actual measured TCP anchors relative labels. The official processor uses
    the same raw world pose/state to implement its native relative conversion.
    """
    result = []
    for anchor in range(max(0, len(rows) - 39)):
        group = rows[anchor:anchor + 40]
        first = group[0]
        if any(row["episode_id"] != first["episode_id"] or row["segment_id"] != first["segment_id"] or
                row["identity"] != first["identity"] or row["split"] != first["split"] or
                any(row.get(flag, False) or row["privileged"].get(flag, False) for flag in
                    ("reset", "abort", "native_abort", "takeover", "invalid_reference")) or
                not row["supervision"]["valid"] or not row["supervision"]["actually_sent"] or
                any(row["supervision"].get(k) is None for k in ("accepted", "installed", "committed"))
                for row in group):
            continue
        times = np.asarray([row["observation_time_s"] for row in group])
        commands = [row["supervision"] for row in group]
        if (not np.all(np.abs(np.diff(times) - .05) <= 1e-8) or
                not np.all(np.abs([cmd["command_time_s"] - t for cmd, t in zip(commands, times)]) <= 1e-8)):
            continue
        require(len({cmd["sequence"] for cmd in commands}) == 40, "Duplicated committed setpoint sequence")
        require(all(cmd["action_label_source"] == LABEL_SOURCE for cmd in commands), "Wrong label source")
        actual, _ = _native_pose(first["policy_observation"])
        action = np.stack([pose10(cmd["committed"]) for cmd in commands])
        relative = []
        differences = {name: [] for name in ("requested_accepted", "accepted_installed", "installed_committed")}
        for cmd in commands:
            committed, grip = _native_pose(cmd["committed"])
            rel = np.linalg.inv(actual) @ committed
            relative.append(np.r_[rel[:3, 3], rotation_matrix_to_rot6d_rows(rel[:3, :3]), grip])
            for name, left, right in (("requested_accepted", "requested", "accepted"),
                    ("accepted_installed", "accepted", "installed"), ("installed_committed", "installed", "committed")):
                a, ag = _native_pose(cmd[left]); b, bg = _native_pose(cmd[right])
                differences[name].append([float(np.linalg.norm(a[:3, 3] - b[:3, 3])),
                    float(geodesic_angle(a[:3, :3], b[:3, :3])), abs(ag - bg)])
        result.append(dict(anchor=anchor, state=pose10(first["policy_observation"]), action=action,
            relative_action=np.asarray(relative), valid_mask=np.ones((40, 10), dtype=bool),
            instruction=first["policy_observation"]["instruction"], identity=first["identity"],
            layout_id=first["layout_id"], task_id=first["task_id"], stage=first["privileged"].get("teacher_stage"),
            command_ids=[(cmd["ref_id"], cmd["sequence"]) for cmd in commands],
            command_times_s=times, label_source=LABEL_SOURCE, point_offsets_s=np.arange(40) / 20,
            rot6d_layout=ROT6D_LAYOUT, command_difference_max={k: np.max(v, axis=0).tolist() for k,v in differences.items()}))
    return result


def load_records(path, episode, *, clock_tolerance_s=1e-8):
    from PIL import Image
    path = Path(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    require(len(rows) == 641, "Successful episode must contain641 raw20Hz samples")
    seen = {name: set() for name in ("body_rgb", "wrist_rgb")}
    for i, row in enumerate(rows):
        validate_sample(row)
        require(row["evidence_kind"] == "REAL_MUJOCO_RECORDING", "CPU fixtures cannot qualify real data")
        require(all(row[k] == episode[k] for k in ("episode_id", "layout_id", "split", "identity")), "Episode frozen metadata mismatch")
        require(abs(row["observation_time_s"] - i / 20) <= clock_tolerance_s, "Exact0..32s20Hz recording required")
        require(row["policy_observation"]["instruction"] == rows[0]["policy_observation"]["instruction"], "Language changed")
        if i:
            require(row["supervision"]["sequence"] > rows[i-1]["supervision"]["sequence"], "Setpoint sequence regression")
            for clock in ("simulation", "ros", "wall_monotonic"):
                require(row["times"][clock] >= rows[i-1]["times"][clock], "Raw clock rollback")
        for name in seen:
            image = row["policy_observation"][name]
            require(image["valid"], "Missing frame cannot qualify successful supervision")
            require(image["frame_index"] == i, "Dropped/copied frame cannot fill a20Hz training slot")
            frame = (path.parent / image["path"]).resolve()
            require(frame.is_relative_to(path.parent.resolve()) and frame not in seen[name], "Frame escape/reuse rejected")
            seen[name].add(frame)
            require(sha(frame) == image["sha256"], "Original RGB hash mismatch")
            with Image.open(frame) as rgb:
                require(rgb.mode == "RGB" and rgb.width >= 256 and rgb.height >= 256, "Raw dual RGB must be at least256x256")
                rgb.verify()
    return rows


def score_saved(rows, episode, path):
    require(sha(path) == episode["physics_sha256"], "Saved physics changed")
    data = {key: np.asarray(value) for key,value in json.loads(Path(path).read_text(encoding="utf-8")).items()}
    for row in rows:
        matches = np.flatnonzero(np.abs(data["time_s"] - row["observation_time_s"]) <= 1e-8)
        require(len(matches) == 1, "Missing same-time measured physics")
        actual, _ = _native_pose(row["policy_observation"])
        tcp = data["tcp_world"][matches[0]]
        require(np.linalg.norm(tcp[:3,3] - actual[:3,3]) <= 1e-5 and
            geodesic_angle(tcp[:3,:3], actual[:3,:3]) <= 1e-4, "Policy TCP does not match physical measurement")
    return score_episode(data, goal_center=episode["goal_center"], initial_object_pose=episode["initial_object_pose"],
        table_height_m=episode["table_height_m"], complete=episode["complete"], evidence_source="REAL_MUJOCO_SAVED_PHYSICS")


def validate_manifest(root, *, clock_tolerance_s=1e-8):
    root = Path(root).resolve()
    path = root / "DATA_MANIFEST.json"
    if not path.is_file():
        return dict(status="DATA_NOT_ADMITTED", reasons=["Missing real PI0288 DATA_MANIFEST.json"], episodes=[]), []
    manifest = json.loads(path.read_text(encoding="utf-8"))
    require(manifest["source"] == "target_mujoco" and manifest["simulation_mode"] == "FROZEN_NATIVE_OCS2_MUJOCO", "Only frozen native target MuJoCo data qualify")
    require_gripper_spec(manifest["gripper_spec"])
    require(set(manifest["identity"]) == set(IDENTITIES) and all(manifest["identity"].values()), "Missing frozen dataset identities")
    for gate in ("G1", "G2", "G3"):
        binding = manifest["prerequisites"][gate]
        receipt_path = (root / binding["path"]).resolve()
        require(receipt_path.is_relative_to(root) and sha(receipt_path) == binding["sha256"], "Upstream receipt escape/hash mismatch")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        require(receipt.get("status") == "PASS" and receipt.get("evidence_kind") == "REAL_TARGET_PHYSICS", f"{gate} lacks real physical PASS")
        keys = {"G1": ("core_wbc", "clock"), "G2": ("core_wbc", "task_executor", "clock", "scene"),
                "G3": IDENTITIES}[gate]
        require(all(receipt.get("identity", {}).get(key) == manifest["identity"][key] for key in keys), f"{gate} frozen identity mismatch")
    episodes = manifest["episodes"]
    require(len({ep["layout_id"] for ep in episodes}) == len(episodes), "Repeated layout; no quality retries")
    require(len({ep["episode_id"] for ep in episodes}) == len(episodes), "Repeated episode UUID")
    require(len({ep["initial_state_sha256"] for ep in episodes}) == len(episodes), "TRAIN/DEV initial states collide")
    expected = {"TRAIN": set(range(288100, 288164)), "DEV": set(range(288200, 288216))}
    checks, samples, success = [], [], {"TRAIN": 0, "DEV": 0}
    reasons = []
    for ep in episodes:
        require(ep["split"] in expected and ep["layout_id"] in expected[ep["split"]], "Reserved/nonformal layout rejected")
        require(ep["identity"] == manifest["identity"] and isinstance(ep["complete"], bool), "Episode identity/completion invalid")
        saved = (root / ep["samples"]).resolve(); physics = (root / ep["physics"]).resolve()
        require(saved.is_relative_to(root) and physics.is_relative_to(root), "Episode data escape dataset")
        eligible = []
        try:
            require(ep["evidence_kind"] == "REAL_MUJOCO_RECORDING", "Synthetic episode rejected")
            require(sha(saved) == ep["samples_sha256"], "Saved records changed")
            raw = [json.loads(line) for line in saved.read_text(encoding="utf-8").splitlines() if line.strip()]
            score = score_saved(raw, ep, physics)
            if score["persistent_success"]:
                rows = load_records(saved, ep, clock_tolerance_s=clock_tolerance_s)
                eligible = windows(rows)
                require(bool(eligible) and len(eligible) <= 602, "No valid committed H40 supervision")
                success[ep["split"]] += 1
                samples.extend(dict(item, split=ep["split"], episode_id=ep["episode_id"], source_path=saved,
                    rows=rows, evidence_kind=ep["evidence_kind"]) for item in eligible)
        except (ValueError, KeyError, OSError) as exc:
            score = dict(persistent_success=False, diagnostic_error=f"{type(exc).__name__}: {exc}")
            reasons.append(f"{ep['episode_id']}: corrupt or incomplete evidence")
        checks.append(dict(episode_id=ep["episode_id"], layout_id=ep["layout_id"], split=ep["split"], score=score, valid_H40=len(eligible)))
    for split, minimum in (("TRAIN",56),("DEV",14)):
        layouts = {ep["layout_id"] for ep in episodes if ep["split"] == split}
        if layouts != expected[split]: reasons.append(f"{split} fixed denominator incomplete:{len(layouts)}/{len(expected[split])}")
        if success[split] < minimum: reasons.append(f"{split} physical successes:{success[split]} below{minimum}")
    if not samples: reasons.append("No real successful continuous H40")
    return dict(status="DATA_NOT_ADMITTED" if reasons else "MUJOCO_TARGET_DATA_READY", reasons=reasons,
        manifest_sha256=sha(path), successful=success, episodes=checks, label_source=LABEL_SOURCE,
        rot6d_layout=ROT6D_LAYOUT, maximum_windows_per_episode=602, neural_forwards=0, optimizer_updates=0), samples


def check_official(samples, output, official, tokenizer):
    require(all(sample.get("evidence_kind") == "REAL_MUJOCO_RECORDING" for sample in samples), "CPU/synthetic inputs cannot qualify")
    ids = {sample["episode_id"] for sample in samples}
    stages = {sample["stage"] for sample in samples if sample["stage"]}
    require(len(ids) >= 2 and len(stages) >= 2, "Official admission requires at least2 real episodes and mixed stages")
    report = engine_independent.check_official(samples, output, official, tokenizer)
    legacy = configure_cpu(official, tokenizer)
    codec = legacy.Codec(output)
    a = samples[0]
    b = next(sample for sample in samples if sample["episode_id"] != a["episode_id"] and sample["stage"] != a["stage"])
    features = [official_feature(codec, sample) for sample in (a,b)]
    batch = codec.proc.collator(features)["inputs"]
    import torch
    def move_cpu(value):
        if isinstance(value, torch.Tensor):
            moved = value.to("cpu")
            require(moved.device.type == "cpu" and torch.equal(value, moved), "CPU device transfer mismatch")
            return moved
        if isinstance(value, dict): return {k:move_cpu(v) for k,v in value.items()}
        if isinstance(value, (list,tuple)): return [move_cpu(v) for v in value]
        return value
    move_cpu(batch)
    require(not torch.cuda.is_initialized(), "CPU admission initialized CUDA")
    for kind in ("state", "action"):
        require(batch[kind].shape[0] == 2 and bool((batch[kind+"_mask"][...,:10] == 1).all())
            and not bool(batch[kind+"_mask"][...,10:].any()), "Mixed episode masks invalid")
    report.update(mixed_episode_ids=[a["episode_id"],b["episode_id"]], mixed_stages=[a["stage"],b["stage"]],
        device_transfer="CPU_ACTUAL_BATCH_PASS", real_episode_count=len(ids))
    return report
