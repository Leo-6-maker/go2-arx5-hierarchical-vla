"""Bind causal sensor frames and isolated appearance edits to official SLA input."""

from dataclasses import dataclass

import numpy as np

from source_custody import require, source_history
from policy_query import input_identity, tensor_sha

LEFT_MASK = "left_wrist_0_rgb"
CAMERA_KEYS = {"base_0_rgb", LEFT_MASK, "right_wrist_0_rgb"}
MISSING_LEFT_SIZE = (480, 640, 3)
MODEL_HISTORY_LENGTH = 10
THREE_VIEW_V1 = "THREE_VIEW_V1"
AUXILIARY_CAMERA = "aux_left_rgb"


def owned(array):
    value = np.array(array, copy=True, order="C")
    value.setflags(write=False)
    return value


@dataclass(frozen=True)
class Frame:
    episode: str
    camera: str
    tick: int
    rgb: np.ndarray

    def __post_init__(self):
        require(bool(self.episode) and type(self.tick) is int and self.tick >= 0,
                "invalid frame timestamp")
        require(self.rgb.ndim == 3 and self.rgb.shape[-1] == 3 and min(self.rgb.shape[:2]) > 0
                and self.rgb.dtype == np.uint8, "invalid RGB frame")
        object.__setattr__(self, "rgb", owned(self.rgb))


@dataclass(frozen=True)
class SensorObservation:
    episode: str
    query_id: str
    tick: int
    clock: str
    head: tuple[Frame, ...]
    right: Frame
    left: Frame | None
    state: np.ndarray
    prompt: str
    profile: str | None = None

    def __post_init__(self):
        require(bool(self.episode) and bool(self.query_id) and type(self.tick) is int and self.tick >= 0,
                "invalid observation identity")
        require(self.state.shape == (16,) and self.state.dtype == np.float32 and np.isfinite(self.state).all()
                and isinstance(self.prompt, str), "invalid official state or instruction")
        if self.clock == "ROBODOJO_RECORDED_30HZ":
            expected = source_history(self.tick)
            require(self.left is not None and self.profile is None, "real source left camera or profile mismatch")
        elif self.clock == "GO2_NATIVE_2MS":
            expected = [0, *(max(0, self.tick - i * 500) for i in range(8, 0, -1)), self.tick]
        else:
            raise ValueError("unsupported observation clock")
        frames = tuple(self.head)
        require(len(frames) == MODEL_HISTORY_LENGTH, "wrong head history count")
        for frame, tick in zip(frames, expected, strict=True):
            self._verify_frame(frame, "cam_high", tick)
        for earlier, later in zip(frames, frames[1:]):
            require(earlier.tick != later.tick or np.array_equal(earlier.rgb, later.rgb),
                    "conflicting samples from the same head frame")
        require(all(frame.rgb.shape == frames[-1].rgb.shape for frame in frames), "inconsistent head raster")
        self._verify_frame(self.right, "cam_right_wrist", self.tick)
        require(self.profile in (None, THREE_VIEW_V1), "unsupported sensor profile")
        if self.profile == THREE_VIEW_V1:
            require(self.left is not None, "missing actual auxiliary camera")
            self._verify_frame(self.left, AUXILIARY_CAMERA, self.tick)
            require(self.left.rgb.shape == MISSING_LEFT_SIZE and self.left.rgb.any(), "invalid or blank auxiliary RGB")
        elif self.left is not None:
            self._verify_frame(self.left, "cam_left_wrist", self.tick)
        object.__setattr__(self, "head", frames)
        object.__setattr__(self, "state", owned(self.state))

    def _verify_frame(self, frame, camera, tick):
        require(type(frame) is Frame and (frame.episode, frame.camera, frame.tick) == (self.episode, camera, tick),
                "cross-episode, future, reordered or wrong-camera frame")

    def legacy_input(self):
        left = self.left.rgb if self.left is not None else np.zeros(MISSING_LEFT_SIZE, np.uint8)
        return dict(state=self.state.copy(), prompt=self.prompt,
                    images=dict(cam_high=np.stack([frame.rgb for frame in self.head]),
                                cam_right_wrist=self.right.rgb.copy(), cam_left_wrist=left.copy()))


@dataclass(frozen=True)
class RasterEdit:
    episode: str
    camera: str
    tick: int
    input_sha256: str
    rgb: np.ndarray
    support: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "rgb", owned(self.rgb))
        object.__setattr__(self, "support", owned(self.support))

    def apply(self, frame):
        require((self.episode, self.camera, self.tick) == (frame.episode, frame.camera, frame.tick)
                and self.input_sha256 == tensor_sha(frame.rgb), "appearance edit associated with the wrong frame")
        require(self.rgb.shape == frame.rgb.shape and self.rgb.dtype == np.uint8
                and self.support.shape == frame.rgb.shape[:2] and self.support.dtype == bool,
                "appearance edit raster mismatch")
        require(np.array_equal(self.rgb[~self.support], frame.rgb[~self.support]),
                "appearance edit outside its observed support")
        return self.rgb.copy()


@dataclass(frozen=True)
class AppearanceEdit:
    head: tuple[RasterEdit, ...]
    right: RasterEdit

    def apply(self, observation, payload):
        require(len(self.head) == len(observation.head), "appearance edit history is incomplete")
        payload["images"]["cam_high"] = np.stack([
            edit.apply(frame) for edit, frame in zip(self.head, observation.head, strict=True)
        ])
        payload["images"]["cam_right_wrist"] = self.right.apply(observation.right)


def prepare_sensor(observation, official_prepare, *, appearance=None, left_mask_only=False):
    """The deployable path accepts measured sensor packets and same-view RGB edits."""
    require(type(observation) is SensorObservation, "deployable input requires a measured sensor packet")
    require(appearance is None or type(appearance) is AppearanceEdit, "invalid appearance edit provenance")
    require(not left_mask_only or (appearance is None and observation.left is None),
            "left mask control requires a missing left view and unchanged measured input")
    payload = observation.legacy_input()
    if appearance is not None:
        appearance.apply(observation, payload)
    digest = input_identity(payload)[0]
    prepared = official_prepare(payload)
    require(set(prepared["image"]) == set(prepared["image_mask"]) == CAMERA_KEYS,
            "official current camera ABI changed")
    if observation.profile == THREE_VIEW_V1:
        require(all(bool(v) for v in prepared["image_mask"].values()), "three-view cameras must be valid")
    if left_mask_only:
        require(all(bool(mask) for mask in prepared["image_mask"].values()), "legacy processor masks changed")
        prepared["image_mask"] = dict(prepared["image_mask"], **{LEFT_MASK: np.bool_(False)})
    return prepared, digest
