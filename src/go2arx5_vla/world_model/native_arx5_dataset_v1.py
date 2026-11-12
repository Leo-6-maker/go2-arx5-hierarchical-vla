"""TRAIN-only observation version; separate from every historical dataset gate."""
from pathlib import Path
import json
import numpy as np
from PIL import Image

from .native_camera_transfer_v1 import CAMERA_ID
from .storage import digest


class NativeCameraTransferDataset:
    def __init__(self, root, camera_audit, *, processor, for_optimization=False,
                 training_admission=None):
        self.root = Path(root)
        self.manifest_sha = digest(self.root/'DATASET_MANIFEST.json')
        self.statistics_sha = digest(self.root/'statistics.json')
        self.manifest = json.loads((self.root/'DATASET_MANIFEST.json').read_text())
        if self.statistics_sha != self.manifest.get('statistics_sha256'):
            raise ValueError('ORIGINAL_TARGET_STATISTICS_CHANGED')
        pin = Path(camera_audit['path'])
        if digest(pin) != camera_audit['sha256']:
            raise ValueError('ACTUAL_CAMERA_AUDIT_CHANGED')
        q = json.loads(pin.read_text())
        if (q.get('schema') != 'NATIVE_CAMERA_ACTUAL_PROCESSOR_COVERAGE_V1'
                or not q.get('camera_coverage_passes')
                or q.get('actual_Processor_calls') != 1500
                or q.get('dataset_manifest_sha256') != self.manifest_sha
                or self.manifest.get('profile') != CAMERA_ID
                or self.manifest.get('old_DEV_references_copied') is not False):
            raise ValueError('FULL_ACTUAL_BODY_WRIST_INPUT_COVERAGE_REQUIRED')
        # Optimization is unlocked by a distinct official-parent/cache/restore
        # qualification, never by camera success or an old FIT4 receipt alone.
        if for_optimization:
            if training_admission is None:
                raise PermissionError('NEW_OFFICIAL_PARENT_TRAINING_ADMISSION_REQUIRED')
            p = Path(training_admission['path'])
            if digest(p) != training_admission['sha256']:
                raise ValueError('TRAINING_ADMISSION_CHANGED')
            a = json.loads(p.read_text())
            if (a.get('schema') != 'NATIVE_ARX5_OFFICIAL_PARENT_TRAINING_ADMISSION_V1'
                    or not a.get('passes') or a.get('dataset_sha256') != self.manifest_sha
                    or a.get('statistics_sha256') != self.statistics_sha):
                raise PermissionError('NATIVE_CAMERA_TRAINING_QUALIFICATION_NOT_PASSED')
        self.processor = processor
        self.for_optimization = for_optimization
        episodes = self.manifest['episodes']
        if len(episodes) != 100 or any(e['split'] != 'TRAIN' for e in episodes):
            raise ValueError('EXACT_TRAIN100_WITHOUT_DEV_TEST')
        self.rows = [(e, i) for e in episodes for i in e['supervised_anchors']]
        if len(self.rows) != 1450 or len({(e['id'], i) for e, i in self.rows}) != 1450:
            raise ValueError('UNIQUE_1450_SUPERVISED_WINDOWS')
        self._cached = None

    def __len__(self):
        return len(self.rows)

    def identity(self, index):
        e, i = self.rows[index]
        return dict(episode_id=e['id'], split='TRAIN', anchor=i)

    def observation_identity(self,index):
        import hashlib
        e,i=self.rows[index]
        if self._cached is None or self._cached[0]!=e['id']:
            self.raw(index)
        packet=self._cached[1]
        return dict(camera_profile=self.manifest['profile'],RGB_sha256=packet['rgb_sha256'][i].tolist(),
            language_sha256=hashlib.sha256(packet['instruction'].item().encode()).hexdigest())

    def raw(self, index):
        e, i = self.rows[index]
        if self._cached is None or self._cached[0] != e['id']:
            file = self.root/e['file']
            if not file.resolve().is_relative_to(self.root.resolve()) or digest(file) != e['sha256']:
                raise ValueError('PINNED_NEW_OBSERVATION_EPISODE')
            with np.load(file, allow_pickle=False) as z:
                self._cached = e['id'], {k:z[k].copy() for k in z.files}
        packet = self._cached[1]
        if not packet['valid_window'][i] or packet['label_representation'].item() != 'ABSOLUTE_COMMITTED_EEF10':
            raise ValueError('ORIGINAL_NOMINAL_CONSUMPTION_PROVEN_LABELS_ONLY')
        images = []
        for c in range(2):
            path = self.root/str(packet['rgb_paths'][i, c])
            if not path.resolve().is_relative_to(self.root.resolve()) or digest(path) != packet['rgb_sha256'][i, c]:
                raise ValueError('PINNED_NEW_BODY_OR_ORIGINAL_WRIST_BYTES')
            with Image.open(path) as picture:
                if picture.mode != 'RGB' or picture.size != (256, 256):
                    raise ValueError('EXACT_RGB256_REQUIRED')
                images.append(np.asarray(picture).copy())
        atom = int(packet['native_command_anchor'][i])
        actions = packet['committed_absolute10'][atom:atom+40]
        if actions.shape != (40, 10):
            raise ValueError('NO_PARTIAL_H40_OR_CROSS_EPISODE')
        return packet['actual_state10'][i], actions.copy(), images, packet['instruction'].item()

    def __getitem__(self, index):
        from gr00t.data.types import VLAStepData, MessageType
        from gr00t.data.embodiment_tags import EmbodimentTag
        state, actions, images, text = self.raw(index)
        value = VLAStepData(images={'external_rgb':[images[0]], 'wrist_rgb':[images[1]]},
            states={'eef_9d':state[None,:9], 'gripper_position':state[None,9:]},
            actions={'eef_9d':actions[:,:9], 'gripper_position':actions[:,9:]},
            masks=None, text=text, embodiment=EmbodimentTag.NEW_EMBODIMENT, metadata={})
        return dict(processed=self.processor([{'type':MessageType.EPISODE_STEP.value, 'content':value}]))
