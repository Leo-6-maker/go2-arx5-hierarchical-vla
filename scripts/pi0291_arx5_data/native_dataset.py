"""PI0291 real-RGB project DatasetLoader and fixed official CPU processor."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

OFFICIAL_SHA = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
DOMAIN = "FIXED_BASE_ARX5_TASK_SIM"
TAG = "new_embodiment"
POLICY_KEYS = frozenset(("instruction", "actual_tcp", "actual_gripper", "actual_gripper_speed", "timestamp_s", "valid", "body_rgb", "wrist_rgb"))


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def pose10(matrix, gripper):
    matrix = np.asarray(matrix, dtype=np.float64)
    return np.concatenate((matrix[..., :3, 3], matrix[..., :2, :3].reshape(matrix.shape[:-2] + (6,)), np.asarray(gripper)[..., None]), axis=-1)


def matrix_from10(value):
    value = np.asarray(value, dtype=np.float64)
    x = value[..., 3:6]
    x = x / np.linalg.norm(x, axis=-1, keepdims=True)
    y = value[..., 6:9] - np.sum(value[..., 6:9] * x, axis=-1, keepdims=True) * x
    y = y / np.linalg.norm(y, axis=-1, keepdims=True)
    matrix = np.broadcast_to(np.eye(4), value.shape[:-1] + (4, 4)).copy()
    matrix[..., :3, :3] = np.stack((x, y, np.cross(x, y)), axis=-2)
    matrix[..., :3, 3] = value[..., :3]
    return matrix


def validate_samples(samples, verify_images=None):
    if len(samples) != 641 or samples[-1]["supervision"] is not None:
        raise ValueError("Complete 641 observations / 640 commands required")
    segments = set()
    for index, sample in enumerate(samples):
        obs = sample["policy_obs"]
        if frozenset(obs) != POLICY_KEYS:
            raise ValueError("Illegal or missing policy observation fields")
        if not obs["valid"] or abs(obs["timestamp_s"] - index / 20) > 1e-8:
            raise ValueError("Invalid or unordered observation clock")
        matrix = np.asarray(obs["actual_tcp"])
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError("Actual TCP must be finite SE3")
        if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8) or not np.allclose(matrix[:3, :3] @ matrix[:3, :3].T, np.eye(3), atol=1e-7) or np.linalg.det(matrix[:3, :3]) < .999999:
            raise ValueError("Invalid actual TCP rotation")
        if not 0 <= obs["actual_gripper"] <= 1:
            raise ValueError("Normalized actual gripper must be closed=0/open=1")
        history = sample["causal_history"]
        segments.add((history["episode_id"], history["segment_id"]))
        if history["previous_samples"] != index or history["previous_committed"] != index or any(history[k] for k in ("reset", "abort", "takeover")):
            raise ValueError("Future history or discontinuity in complete episode")
        for name in ("body_rgb", "wrist_rgb"):
            frame = obs[name]
            if not frame["valid"] or frame["frame_index"] != index or frame["snapshot_id"] != sample["snapshot_id"] or abs(frame["timestamp_s"] - index / 20) > 1e-8:
                raise ValueError("RGB/state snapshot mismatch")
            if verify_images is not None:
                path = (Path(verify_images) / frame["path"]).resolve()
                if not path.is_relative_to(Path(verify_images).resolve()) or sha(path) != frame["sha256"]:
                    raise ValueError("RGB path escape or frame hash mismatch")
                with Image.open(path) as image:
                    if image.mode != "RGB" or image.size != (256, 256):
                        raise ValueError("Raw camera image must be actual 256x256 RGB")
        if index < 640:
            command = sample["supervision"]
            if not command["actually_committed"] or not command["valid"] or command["sequence"] != index or abs(command["command_time_s"] - index / 20) > 1e-8:
                raise ValueError("Uncommitted or mistimed action supervision")
            target = np.asarray(command["committed"])
            if target.shape != (4, 4) or not np.isfinite(target).all() or not 0 <= command["committed_gripper"] <= 1:
                raise ValueError("Invalid committed absolute command")
            if not np.allclose(target[3], [0, 0, 0, 1], atol=1e-8) or not np.allclose(target[:3, :3] @ target[:3, :3].T, np.eye(3), atol=1e-7) or np.linalg.det(target[:3, :3]) < .999999:
                raise ValueError("Invalid committed target rotation")
    if len(segments) != 1:
        raise ValueError("Episode or execution identity changed within window population")


def cpu_guard():
    os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false", NO_ALBUMENTATIONS_UPDATE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA must not be initialized")
    torch.set_num_threads(1)
    original = torch.nn.Module.__call__
    if not getattr(original, "pi0291_guard", False):
        def guarded(module, *args, **kwargs):
            if type(module).__module__.startswith("torchvision.transforms"):
                return original(module, *args, **kwargs)
            raise PermissionError("PI0291 forbids neural forward")
        guarded.pi0291_guard = True
        torch.nn.Module.__call__ = guarded


def modality():
    from gr00t.data.types import ModalityConfig, ActionConfig, ActionFormat, ActionType, ActionRepresentation
    return {TAG: {
        "video": ModalityConfig(delta_indices=[0], modality_keys=["external_rgb", "wrist_rgb"]),
        "state": ModalityConfig(delta_indices=[0], modality_keys=["eef_9d", "gripper_position"]),
        "action": ModalityConfig(delta_indices=list(range(40)), modality_keys=["eef_9d", "gripper_position"], action_configs=[ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.EEF, format=ActionFormat.XYZ_ROT6D, state_key="eef_9d"), ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT, state_key="gripper_position")]),
        "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.language.task_description"]),
    }}


def official_import(source):
    source = Path(source)
    actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if actual != OFFICIAL_SHA:
        raise ValueError("Fixed official source revision mismatch")
    for relative in ("gr00t/data/state_action/pose.py", "gr00t/data/state_action/state_action_processor.py", "gr00t/data/utils.py", "gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py"):
        expected = subprocess.check_output(["git", "-C", str(source), "show", OFFICIAL_SHA + ":" + relative])
        if hashlib.sha256(expected).hexdigest() != sha(source / relative):
            raise ValueError("Official processor/codec source locally modified")
    sys.path.insert(0, str(source))


def build_processor(source, cosmos, statistics):
    cpu_guard()
    official_import(source)
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
    processor = Gr00tN1d7Processor(modality_configs=modality(), statistics=statistics, use_percentiles=False, clip_outliers=False, use_relative_action=True, image_crop_size=[256, 256], image_target_size=[256, 256], use_albumentations=False, random_rotation_angle=0, color_jitter_params=None, crop_fraction=1., model_name=str(cosmos), model_type="qwen", max_state_dim=132, max_action_dim=132, max_action_horizon=40, state_dropout_prob=0., exclude_state=False, transformers_loading_kwargs={"local_files_only": True, "trust_remote_code": True})
    processor.train_image_transform = processor.eval_image_transform
    processor.eval()
    if processor.embodiment_id_mapping[TAG] != 10:
        raise ValueError("Native projector route mismatch")
    return processor


class DatasetLoader:
    """The project training entrance: raw absolute data -> official processor."""
    def __init__(self, dataset_root, source, cosmos, split, indices=None):
        if split not in ("TRAIN", "DEV"):
            raise ValueError("Explicit source split required")
        self.root = Path(dataset_root)
        self.manifest = read(self.root / "DATASET_MANIFEST.json")
        if self.manifest["domain"] != DOMAIN:
            raise ValueError("ARX5 domain isolation required")
        self.source, self.cosmos, self.split = str(source), str(cosmos), split
        self.statistics = read(self.root / "statistics.json")
        if sha(self.root / "statistics.json") != self.manifest["statistics_sha256"]:
            raise ValueError("Frozen TRAIN statistics changed")
        self.rows = [(episode, anchor) for episode in self.manifest["episodes"] if episode["split"] == split for anchor in range(episode["windows"])]
        if indices is not None:
            self.rows = [self.rows[index] for index in indices]
        self.processor = None
        self.cached_id, self.cached = None, None

    def __len__(self):
        return len(self.rows)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(processor=None, cached_id=None, cached=None)
        return state

    def initialize(self):
        if self.processor is None:
            self.processor = build_processor(self.source, self.cosmos, self.statistics)

    def raw(self, index):
        episode, anchor = self.rows[index]
        path = self.root / episode["file"]
        if self.cached_id != episode["id"]:
            if sha(path) != episode["sha256"]:
                raise ValueError("Native export changed")
            with np.load(path, allow_pickle=False) as archive:
                self.cached = {key: archive[key].copy() for key in archive.files}
            self.cached_id = episode["id"]
        packet = self.cached
        if not packet["valid_window"][anchor] or packet["label_representation"].item() != "ABSOLUTE_COMMITTED_EEF10":
            raise ValueError("No genuine committed absolute H40")
        images = []
        for camera in range(2):
            frame = Path(episode["raw_root"]) / packet["rgb_paths"][anchor, camera]
            if sha(frame) != packet["rgb_sha256"][anchor, camera]:
                raise ValueError("Original RGB hash mismatch")
            with Image.open(frame) as image:
                images.append(np.asarray(image.convert("RGB")).copy())
        return packet["actual_state10"][anchor], packet["committed_absolute10"][anchor:anchor + 40], images, packet["instruction"].item(), {"episode_id": episode["id"], "anchor": anchor, "split": self.split}

    def __getitem__(self, index):
        self.initialize()
        from gr00t.data.types import VLAStepData, MessageType
        from gr00t.data.embodiment_tags import EmbodimentTag
        state, commands, images, text, metadata = self.raw(index)
        content = VLAStepData(images={"external_rgb": [images[0]], "wrist_rgb": [images[1]]}, states={"eef_9d": state[None, :9], "gripper_position": state[None, 9:]}, actions={"eef_9d": commands[:, :9], "gripper_position": commands[:, 9:]}, masks=None, text=text, embodiment=EmbodimentTag.NEW_EMBODIMENT, metadata={})
        value = self.processor([{"type": MessageType.EPISODE_STEP.value, "content": content}])
        return {"processed": value, "identity": metadata}


def collate_project(rows):
    # The caller owns and invokes the official collator, without metadata leakage.
    return rows
