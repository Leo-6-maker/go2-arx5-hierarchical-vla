"""Saved target records -> EEF10 windows -> unchanged official CPU processor.

The native xyzw quaternion is authoritative. EEF10 uses FIRST TWO ROWS of R;
the bridge's FIRST TWO COLUMNS representation is never passed to this codec.
No model is constructed, no physics is run, and caches are always local-only.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np

from ..trajectory.so3 import (geodesic_angle, quat_to_matrix,
                              rot6d_rows_to_matrix, rotation_matrix_to_rot6d_rows)

REPO = Path(__file__).resolve().parents[3]
POLICY_FIELDS = frozenset(("body_rgb", "wrist_rgb", "tcp_xyz",
                          "tcp_quaternion_xyzw", "gripper", "instruction"))
ROT6D_LAYOUT = "GR00T_FIRST_TWO_ROWS"
IDENTITIES = ("asset", "teacher", "controller", "codec")


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(path):
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    """Preserve an existing receipt unless its complete content is identical."""
    path = Path(path)
    content = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        require(path.read_text(encoding="utf-8") == content, f"Preserve existing artifact: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def pose10(record):
    xyz = np.asarray(record["tcp_xyz"], dtype=np.float64)
    quat = np.asarray(record["tcp_quaternion_xyzw"], dtype=np.float64)
    grip = np.asarray(record["gripper"], dtype=np.float64).reshape(-1)
    require(xyz.shape == (3,) and quat.shape == (4,) and grip.shape == (1,), "Invalid TCP/gripper shape")
    require(np.isfinite(np.r_[xyz, quat, grip]).all(), "Nonfinite TCP/gripper")
    require(abs(np.linalg.norm(quat) - 1) <= 1e-5, "Quaternion must be nondegenerate and normalized xyzw")
    require(0 <= grip[0] <= 1, "Gripper must be calibrated open fraction in [0,1]")
    return np.r_[xyz, rotation_matrix_to_rot6d_rows(quat_to_matrix(quat)), grip]


def load_records(path, episode, *, clock_tolerance_s=1e-3, camera_max_age_s=.05):
    path = Path(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    require(bool(rows), "No actual target records")
    from PIL import Image
    require(clock_tolerance_s >= 0 and camera_max_age_s >= 0, "Clock limits must be nonnegative")
    previous = None
    seen_images = {key: set() for key in ("body_rgb", "wrist_rgb")}
    seen_commands = set()
    identity = rows[0]["identity"]
    require(all(identity.get(key) for key in IDENTITIES), "Missing asset/teacher/controller/codec identity")
    for row in rows:
        require(all(row[key] == episode[key] for key in ("episode_id", "layout_id", "split")), "Episode/layout/split mismatch")
        require(row["task_id"] == "GO2_PLACE_V1", "Wrong target task")
        require(row["identity"] == identity, "Identity changed inside episode")
        obs = row["policy_observation"]
        require(frozenset(obs) == POLICY_FIELDS, "Illegal or missing policy fields; truth/feedback/BASE must stay isolated")
        require(isinstance(obs["instruction"], str) and bool(obs["instruction"].strip()), "Missing episode language")
        require(obs["instruction"] == rows[0]["policy_observation"]["instruction"], "Language changed within episode")
        pose10(obs)
        time_s = float(row["observation_time_s"])
        require(np.isfinite(time_s), "Nonfinite observation clock")
        if previous is not None:
            require(abs(time_s - previous - .05) <= clock_tolerance_s, "Nonmonotonic or non-20Hz observations")
        previous = time_s
        for camera in ("body_rgb", "wrist_rgb"):
            image = obs[camera]
            require(image["valid"] is True and image["source"] == "target_render", "RGB is not a valid target render")
            stamp = float(image["timestamp_s"])
            require(np.isfinite(stamp) and -clock_tolerance_s <= time_s - stamp <= camera_max_age_s, "Future or stale RGB")
            image_path = (path.parent / image["path"]).resolve()
            require(image_path.is_relative_to(path.parent.resolve()), "RGB escapes episode directory")
            require(image_path.is_file(), f"Missing RGB: {image_path}")
            require(image_path not in seen_images[camera], "Reused RGB path cannot stand in for a new rendered frame")
            seen_images[camera].add(image_path)
            with Image.open(image_path) as rgb:
                require(rgb.mode == "RGB" and rgb.width > 0 and rgb.height > 0, "Camera file must contain real RGB")
                rgb.verify()
            if "sha256" in image:
                require(sha(image_path) == image["sha256"], "RGB bytes changed")
        require(obs["body_rgb"]["path"] != obs["wrist_rgb"]["path"], "Dual cameras cannot alias one saved RGB frame")
        require("execution_feedback" in row and "privileged_diagnostics" in row, "Four record namespaces required")
        command = row["action_target"]
        pose10(command)
        require(command["command_id"] and command["command_id"] not in seen_commands, "Command IDs must be unique")
        seen_commands.add(command["command_id"])
        require(command["frame_id"] == "world" and bool(command["tcp_id"]), "Missing world frame/TCP binding")
        issued = float(command["issue_time_s"])
        require(np.isfinite(issued) and -clock_tolerance_s <= issued - time_s <= .05, "Command must follow current observation")
        feedback = row["execution_feedback"]
        require(feedback.get("command_id") == command["command_id"], "Feedback must bind same command")
        require(isinstance(feedback.get("valid_mask"), dict), "Feedback requires explicit valid masks")
        for field, valid in feedback["valid_mask"].items():
            require(isinstance(valid, bool) and ((feedback.get(field) is not None) == valid), "Missing feedback must be None with invalid mask")
    return rows


def windows(rows):
    """Use later actually sent commands, never a planned future or achieved TCP."""
    out = []
    for anchor in range(len(rows) - 40):
        group = rows[anchor:anchor + 41]
        first = group[0]
        if any(row["segment_id"] != first["segment_id"] or
               row["episode_id"] != first["episode_id"] or
               any(row.get(flag, False) or row["privileged_diagnostics"].get(flag, False)
                   for flag in ("reset", "abort", "native_abort", "takeover")) for row in group):
            continue
        if not all(all(row["action_target"].get(key) is True for key in ("actually_sent", "accepted", "valid")) for row in group):
            continue
        targets = [row["action_target"] for row in group[1:]]
        if not all(all(command.get(key) is True for key in ("actually_sent", "accepted", "valid")) for command in targets):
            continue
        require(len({command["command_id"] for command in targets}) == 40, "Duplicated future command identity")
        require(len({(command["frame_id"], command["tcp_id"]) for command in targets}) == 1, "Future command frame/TCP changed")
        if not np.all(np.abs(np.diff([row["action_target"]["issue_time_s"] for row in group]) - .05) <= 1e-3):
            continue
        out.append(dict(anchor=anchor, state=pose10(first["policy_observation"]),
                        action=np.stack([pose10(command) for command in targets]),
                        instruction=first["policy_observation"]["instruction"],
                        command_ids=[command["command_id"] for command in targets]))
    return out


def physical_errors(expected, decoded):
    expected = np.asarray(expected, dtype=np.float64)
    decoded = np.asarray(decoded, dtype=np.float64)
    require(expected.shape == decoded.shape == (40, 10), "Expected real H40 x EEF10")
    require(np.isfinite(expected).all() and np.isfinite(decoded).all(), "Nonfinite codec output")
    angle = geodesic_angle(rot6d_rows_to_matrix(expected[:, 3:9]), rot6d_rows_to_matrix(decoded[:, 3:9]))
    return dict(XYZ_m=float(np.linalg.norm(expected[:, :3] - decoded[:, :3], axis=-1).max()),
                SO3_rad=float(angle.max()), gripper=float(np.abs(expected[:, 9] - decoded[:, 9]).max()))


def enforce_roundtrip(expected, decoded):
    error = physical_errors(expected, decoded)
    require(error["XYZ_m"] <= 1e-5 and error["SO3_rad"] <= 1e-4 and error["gripper"] <= 1e-5,
            f"Physical codec roundtrip failed: {error}")
    return error


def score_saved(rows, episode, path):
    """Independent offline scorer reads saved physics, never teacher success."""
    from .scorer import score_episode
    path = Path(path)
    require(path.is_file(), "Missing independent saved 0..32s physics stream")
    if "physics_sha256" in episode:
        require(sha(path) == episode["physics_sha256"], "Saved physics bytes changed")
    data = {key: np.asarray(value) for key, value in json.loads(path.read_text(encoding="utf-8")).items()}
    # Cross-check every policy TCP with the independent physical state stream.
    for row in rows:
        matches = np.flatnonzero(np.abs(data["time_s"] - row["observation_time_s"]) <= 1e-8)
        require(len(matches) == 1, "Observation missing from independent saved physics")
        tcp = data["tcp_world"][matches[0]]
        obs = row["policy_observation"]
        require(np.linalg.norm(tcp[:3, 3] - obs["tcp_xyz"]) <= 1e-5 and
                geodesic_angle(tcp[:3, :3], quat_to_matrix(obs["tcp_quaternion_xyzw"])) <= 1e-4,
                "Policy TCP differs from independent actual physics TCP")
    return score_episode(data, goal_center=episode["goal_center"], initial_object_pose=episode["initial_object_pose"],
                         table_height_m=episode["table_height_m"], complete=episode["complete"],
                         evidence_source="TARGET_SIM_SAVED_PHYSICS")


def validate_manifest(root, *, clock_tolerance_s=1e-3):
    root = Path(root)
    manifest = root / "DATA_MANIFEST.json"
    if not manifest.is_file():
        return dict(status="NOT_READY", reasons=["Missing actual target DATA_MANIFEST.json"], episodes=[]), []
    value = json.loads(manifest.read_text(encoding="utf-8"))
    require(value["source"] == "target_sim", "Source data cannot qualify target expert data")
    require(value.get("simulation_mode") == "FROZEN_WBC_TARGET_SIM", "NO_WBC or pinned-base diagnostics cannot qualify")
    prerequisites = value.get("prerequisites", {})
    for gate in ("G1", "G2", "G3"):
        binding = prerequisites.get(gate)
        require(isinstance(binding, dict), f"Missing upstream real {gate} receipt binding")
        gate_path = (root / binding["path"]).resolve()
        require(sha(gate_path) == binding["sha256"], f"Upstream {gate} receipt bytes changed")
        receipt = json.loads(gate_path.read_text(encoding="utf-8"))
        require(receipt.get("status") == "PASS" and receipt.get("evidence_kind") == "REAL_TARGET_PHYSICS",
                f"Upstream {gate} not passed by actual physical evidence")
    episodes = value["episodes"]
    require(len({ep["layout_id"] for ep in episodes}) == len(episodes), "Repeated layout; retain one fixed attempt")
    expected = {"TRAIN": set(range(287100, 287164)), "DEV": set(range(287200, 287216))}
    checks, samples, successful = [], [], {"TRAIN": 0, "DEV": 0}
    for ep in episodes:
        require(ep["split"] in expected and ep["layout_id"] in expected[ep["split"]], "Nonformal or reserved confirmation layout")
        require(isinstance(ep["complete"], bool), "Episode completion must be an explicit boolean")
        path = (root / ep["samples"]).resolve()
        require(path.is_relative_to(root.resolve()), "Episode samples escape dataset")
        if "samples_sha256" in ep:
            require(sha(path) == ep["samples_sha256"], "Episode records changed")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        physics_path = (root / ep.get("physics", str(Path(ep["samples"]).with_name("physics.json")))).resolve()
        require(physics_path.is_relative_to(root.resolve()), "Physics escapes dataset")
        try:
            score = score_saved(rows, ep, physics_path)
        except (ValueError, KeyError, OSError) as exc:
            # Failed attempts stay in the denominator with their original evidence.
            score = dict(persistent_success=False, diagnostic_error=f"{type(exc).__name__}: {exc}")
        passed = bool(score["persistent_success"])
        if passed:
            rows = load_records(path, ep, clock_tolerance_s=clock_tolerance_s)
            eligible = windows(rows)
            require(bool(eligible), "Successful episode has no genuine valid H40")
        else:
            eligible = []
        successful[ep["split"]] += int(passed)
        checks.append(dict(episode_id=ep["episode_id"], layout_id=ep["layout_id"], split=ep["split"],
                           samples_sha256=sha(path), physics_sha256=sha(physics_path) if physics_path.is_file() else None,
                           score=score, valid_H40=len(eligible)))
        samples.extend(dict(item, split=ep["split"], episode_id=ep["episode_id"], source_path=path, rows=rows) for item in eligible)
    reasons = []
    for split, minimum in (("TRAIN", 56), ("DEV", 14)):
        layouts = {ep["layout_id"] for ep in episodes if ep["split"] == split}
        if layouts != expected[split]:
            reasons.append(f"{split} fixed attempt denominator incomplete: {len(layouts)}/{len(expected[split])}")
        if successful[split] < minimum:
            reasons.append(f"{split} physical successes {successful[split]} below {minimum}")
    if not samples:
        reasons.append("No successful genuine H40 supervision")
    return dict(status="NOT_READY" if reasons else "PASS_SAVED_PHYSICS_PENDING_CODEC", reasons=reasons,
                manifest_sha256=sha(manifest), successful=successful, episodes=checks, prerequisites=prerequisites,
                rot6d_layout=ROT6D_LAYOUT, future_execution_prefix=8,
                neural_forwards=0, optimizer_updates=0, physics_integrations=0), samples


def legacy_codec_module():
    # ponytail: reuse the established official encode/decode implementation.
    path = REPO / "scripts/pi0282_execution/codec.py"
    spec = importlib.util.spec_from_file_location("pi0287_native_codec", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_cpu(official, tokenizer):
    official, tokenizer = Path(official).resolve(), Path(tokenizer).resolve()
    require((official / "gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py").is_file(), "Missing actual official GR00T processor source")
    require((tokenizer / "tokenizer_config.json").is_file(), "Missing local Cosmos/Qwen processor/tokenizer cache")
    require(any((tokenizer / name).is_file() for name in ("tokenizer.json", "vocab.json", "tokenizer.model")), "Missing local tokenizer vocabulary")
    os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      PI0282_OFFICIAL_DIR=str(official), PI0282_COSMOS_DIR=str(tokenizer))
    return legacy_codec_module()


def fit_statistics(samples, official, tokenizer):
    """Fit NEW target statistics exclusively on admitted TRAIN H40 population."""
    legacy = configure_cpu(official, tokenizer)
    legacy.native_imports()
    from gr00t.data.state_action.state_action_processor import StateActionProcessor
    from gr00t.data.types import ActionType, ActionFormat
    train = training_population(samples)
    require(bool(train), "No admitted TRAIN population")
    converter = StateActionProcessor({legacy.TAG: legacy.modality()}, use_relative_action=True)
    state = np.stack([item["state"] for item in train])
    action = np.concatenate([item["action"] for item in train])
    relative = np.concatenate([converter._convert_to_relative_action(item["action"][:, :9], item["state"][:9],
                                 ActionType.EEF, ActionFormat.XYZ_ROT6D) for item in train])
    def summary(array):
        require(np.isfinite(array).all(), "Nonfinite statistics population")
        return {key: value.tolist() for key, value in dict(min=array.min(0), max=array.max(0), mean=array.mean(0),
                std=array.std(0), q01=np.quantile(array, .01, axis=0), q99=np.quantile(array, .99, axis=0)).items()}
    return {legacy.TAG: dict(state=dict(eef_9d=summary(state[:, :9]), gripper_position=summary(state[:, 9:])),
             action=dict(eef_9d=summary(action[:, :9]), gripper_position=summary(action[:, 9:])),
             relative_action=dict(eef_9d=summary(relative)))}, dict(fit_split="TRAIN_ONLY",
             TRAIN_episode_ids=sorted({item["episode_id"] for item in train}), state_population=len(state),
             action_population=len(action), DEV_in_fit=0, source_domain_in_fit=0)


def training_population(samples):
    require(all(item["split"] in ("TRAIN", "DEV") for item in samples), "Unexpected statistics split")
    return [item for item in samples if item["split"] == "TRAIN"]


def build_processor_files(output, template, statistics, tokenizer):
    legacy = legacy_codec_module()
    template, output = Path(template), Path(output)
    config = json.loads((template / "processor_config.json").read_text(encoding="utf-8"))
    kwargs = config["processor_kwargs"]
    kwargs["modality_configs"][legacy.TAG] = legacy.modality()
    kwargs.update(use_percentiles=False, clip_outliers=True, shortest_image_edge=256, crop_fraction=1.,
                  image_crop_size=[256, 256], image_target_size=[256, 256], random_rotation_angle=0,
                  color_jitter_params={key: 0. for key in ("brightness", "contrast", "saturation", "hue")},
                  extra_augmentation_config=None, state_dropout_prob=0., exclude_state=False,
                  max_state_dim=132, max_action_dim=132, max_action_horizon=40, use_relative_action=True,
                  model_name=str(Path(tokenizer).resolve()), model_type="qwen")
    # Target-only statistics: old Panda statistics and files are never rewritten.
    for name, value in (("processor_config.json", config), ("statistics.json", statistics),
                        ("embodiment_id.json", json.loads((template / "embodiment_id.json").read_text(encoding="utf-8")))):
        write_json(output / name, value)


def image_arrays(sample):
    from PIL import Image
    obs = sample["rows"][sample["anchor"]]["policy_observation"]
    images = []
    for key in ("body_rgb", "wrist_rgb"):
        with Image.open(sample["source_path"].parent / obs[key]["path"]) as image:
            require(image.mode == "RGB", "Saved camera must be RGB; no silent conversion")
            array = np.asarray(image).copy()
        require(array.ndim == 3 and array.shape[-1] == 3 and array.dtype == np.uint8, "Invalid dual RGB")
        images.append(array)
    return images


def official_feature(codec, sample, *, instruction=None):
    from PIL import Image
    legacy = legacy_codec_module()
    body, wrist = image_arrays(sample)
    codec.proc.train()
    step = codec.step_type(images={"base_camera": [Image.fromarray(body)], "hand_camera": [Image.fromarray(wrist)]},
                           states=legacy.fields(sample["state"].astype(np.float32).reshape(1, 10)),
                           actions=legacy.fields(sample["action"].astype(np.float32)),
                           text=instruction or sample["instruction"], embodiment=codec.tag)
    return codec.proc([{"type": "episode_step", "content": step}])


def check_official(samples, output, official, tokenizer):
    legacy = configure_cpu(official, tokenizer)
    codec = legacy.Codec(output)
    import torch
    torch.set_grad_enabled(False)
    require(not torch.cuda.is_initialized(), "CPU processor check must not initialize CUDA")
    maxima = {split: dict(XYZ_m=0., SO3_rad=0., gripper=0.) for split in ("TRAIN", "DEV")}
    preclip = {split: 0 for split in ("TRAIN", "DEV")}
    for sample in samples:
        decoded = codec.decode_actions(codec.encode_actions(sample["action"], sample["state"]), sample["state"])
        error = physical_errors(sample["action"], decoded)
        if sample["split"] == "TRAIN":
            enforce_roundtrip(sample["action"], decoded)
        for key in error:
            maxima[sample["split"]][key] = max(maxima[sample["split"]][key], error[key])
        # float64 preclip counts are diagnostics; DEV never refits statistics.
        preclip[sample["split"]] += codec.action_audit(sample["action"].astype(np.float64), sample["state"].astype(np.float64))["native_preclip_out_of_range_count"]
    # Every real episode language and dual camera pair goes through the native path.
    checked_episodes = set()
    for sample in samples:
        if sample["episode_id"] in checked_episodes:
            continue
        codec.proc.collator([official_feature(codec, sample)])
        checked_episodes.add(sample["episode_id"])
    require(bool(samples), "No actual processor sample")
    first = samples[0]
    # Diagnostic text changes only collator probe, never saved episode language.
    texts = [first["instruction"], first["instruction"] + " Carefully place the cube in the blue target area and fully release it."]
    features = [official_feature(codec, first, instruction=text) for text in texts]
    single = [codec.proc.collator([feature])["inputs"] for feature in features]
    mixed = codec.proc.collator(features)["inputs"]
    lengths = []
    for index, inputs in enumerate(single):
        require(tuple(inputs["action"].shape) == (1, 40, 132), "Official action tensor ABI mismatch")
        for kind in ("action", "state"):
            mask = inputs[kind + "_mask"]
            require(bool((mask[..., :10] == 1).all()) and not bool(mask[..., 10:].any()), "BASE/feedback padding must be invalid")
            require(torch.equal(inputs[kind][0], mixed[kind][index]), "Single/mixed batch data mismatch")
            require(torch.equal(mask[0], mixed[kind + "_mask"][index]), "Single/mixed batch validity masks mismatch")
        tokens = inputs["input_ids"][0][inputs["attention_mask"][0].bool()]
        lengths.append(len(tokens))
        require(torch.equal(tokens, mixed["input_ids"][index][mixed["attention_mask"][index].bool()]), "Official variable-text padding changed tokens")
    require(lengths[0] != lengths[1], "Variable-language probe must exercise unequal token lengths")
    source_files = ("gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py",
                    "gr00t/data/state_action/state_action_processor.py", "gr00t/data/state_action/pose.py", "gr00t/data/utils.py")
    return dict(status="PASS_ACTUAL_OFFICIAL_CPU_PROCESSOR", identity=codec.identity,
                processor_sources={name: sha(Path(official) / name) for name in source_files},
                reused_codec_sha256=sha(REPO / "scripts/pi0282_execution/codec.py"),
                roundtrip_maxima=maxima, TRAIN_roundtrip_required=dict(XYZ_m=1e-5, SO3_rad=1e-4, gripper=1e-5),
                DEV_out_of_range_disclosed=True, preclip_count=preclip, preclip_zero_is_gate=False,
                variable_language_token_lengths=lengths, actual_episode_languages_checked=len(checked_episodes), dual_RGB=True, hidden_truth_inputs=[],
                BASE_supervision=False, neural_forwards=0, optimizer_updates=0, physics_integrations=0)
