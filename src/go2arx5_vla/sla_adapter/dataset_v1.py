"""Read the sealed TRAIN100 without modifying captures or expert labels."""
import numpy as np
from PIL import Image

from .codec_v1 import Codec, Quantiles, pad32, LABEL_TIMES_S
from ..world_model.native_arx5_dataset_v1 import NativeCameraTransferDataset

CAMERAS = ('base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb')


def images224(images, resize_with_pad):
    """Call the pinned openpi_client resize function, not an approximate resize."""
    if len(images) != 2:
        raise ValueError('BODY_AND_WRIST_REQUIRED')
    rgb, padding = {}, {}
    for name, value in zip(CAMERAS[:2], images):
        a = np.asarray(value)
        if a.dtype != np.uint8 or a.shape != (256, 256, 3):
            raise ValueError('ORIGINAL_RGB256_UINT8_REQUIRED')
        rgb[name], padding[name] = resize_with_pad(a, 224, 224, return_mask=True)
    rgb[CAMERAS[2]] = np.zeros_like(rgb[CAMERAS[0]])
    padding[CAMERAS[2]] = np.zeros((224, 224), dtype=bool)
    return dict(image=rgb, image_mask=dict(zip(CAMERAS, (True, True, False))), image_padding_mask=padding)


class TargetDataset:
    """Data conversion only; this class does not grant training/physics admission."""
    def __init__(self, root, camera_audit, episode_frames, *, resize_with_pad, statistics=None):
        self.original = NativeCameraTransferDataset(root, camera_audit, processor=None)
        self.frames = episode_frames
        self.resize = resize_with_pad
        self.statistics = statistics
        ids = {e['id'] for e in self.original.manifest['episodes']}
        if set(self.frames) != ids:
            raise ValueError('ACTUAL_POST_HOLD_FRAME_FOR_EVERY_EPISODE_REQUIRED')
        self.codecs = {key: Codec(v['T_policy_from_mount']) for key, v in self.frames.items()}

    def __len__(self):
        return len(self.original)

    def active(self, index):
        state, actions, images, text = self.original.raw(index)
        identity = self.original.identity(index)
        packet = self.original._cached[1]
        if (packet['command_period_s'].item() != .05 or packet['native_dt_s'].item() != .002
                or not np.array_equal(packet['native_command_anchor'], np.arange(15)*40)):
            raise ValueError('UNCHANGED_NATIVE_H40_CLOCK_REQUIRED')
        s, a = self.codecs[identity['episode_id']].encode(state, actions)
        return s, a, images, text, state, actions, identity

    def __getitem__(self, index):
        if self.statistics is None:
            raise ValueError('FIT_SLA_TRAIN_STATISTICS_BEFORE_MODEL_INPUT')
        s, a, images, text, _, _, _ = self.active(index)
        result = images224(images, self.resize)
        result.update(state=pad32(self.statistics['state'].normalize(s)).astype(np.float32),
                      actions=pad32(self.statistics['actions'].normalize(a)).astype(np.float32), prompt=text)
        # Audit identities, native clock and stages remain outside model inputs.
        return result

    def fit_statistics(self):
        states, actions = [], []
        for i in range(len(self)):
            s, a, *_ = self.active(i)
            states.append(s)
            actions.append(a)
        s, a = np.stack(states), np.concatenate(actions)
        self.statistics = dict(state=Quantiles.fit(s), actions=Quantiles.fit(a))
        return dict(state=self.statistics['state'].summary(s), actions=self.statistics['actions'].summary(a),
                    unique_state_windows=len(s), action_knots=len(a), fit_before_oversampling=True,
                    label_times_s=LABEL_TIMES_S.tolist(), complete_consumption_s=2.)
