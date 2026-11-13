"""Read a named dense target pilot; legacy TRAIN100 remains a separate reader."""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import numpy as np

from .codec_v1 import Quantiles, pad32
from .dataset_v1 import images224
from .gripper_v2 import GripLogitCodec, VERSION

SCHEMA = 'SLA_NAMED_DENSE_TARGET_TRAIN_PILOT_V4_1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def member(root, relative):
    path = Path(root) / relative
    if Path(relative).is_absolute() or not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('OWN_IMMUTABLE_DATASET_MEMBER_REQUIRED')
    return path


def manifest_contract(value):
    if (value.get('schema') != SCHEMA or value.get('state') != 'committed'
            or value.get('dataset_id') != 'T_PICK_HOLD_TRAIN01_P10_2EP_V4_1'
            or value.get('scope') != 'NAMED_PICK_HOLD_TRAIN01_P10_FITTING_PILOT_ONLY'
            or value.get('D100_new') is not False or value.get('PICK_PLACE_admitted') is not False
            or value.get('legacy_engineering_receipts_modified') is not False
            or value.get('codec') != VERSION or value.get('tasks') != ['PICK_HOLD']
            or value.get('split') != 'TRAIN' or value.get('phase_input_to_student') is not False
            or value.get('command_period_ticks') != 25 or value.get('physics_dt_s') != .002
            or value.get('frame_period_ticks') != 50 or value.get('anchor_period_ticks') != 250
            or not value.get('consumption_proven') or not value.get('actual_success_tail_s') == 2.):
        raise ValueError('EXPLICIT_CONSUMED_NAMED_PILOT_NOT_D100_OR_PLACEMENT')
    episodes = value.get('episodes') or []
    if (len(episodes) != 2 or len({e['id'] for e in episodes}) != 2
            or any(e['split'] != 'TRAIN' or e['task'] != 'PICK_HOLD'
                   or e['layout'] != 'TRAIN01' or e['prefix_atoms'] != 10
                   or e['windows'] != 61 or e['frames'] != 321 or e['commands'] != 640
                   or e['native_integrations'] != 21703 for e in episodes)
            or sorted(e['restart'] for e in episodes) != [1, 2]
            or value.get('supervised_windows') != 122):
        raise ValueError('EXACT_TWO_INDEPENDENT_SOURCE_EPISODES_AND_WINDOW_DENOMINATOR')


class DensePilotDataset:
    """Bounded frame-shard cache; no PNG, FK or per-visit source hashing."""
    def __init__(self, root, expected_manifest_sha256, *, resize_with_pad, max_cache_bytes=512*1024**2):
        self.root = Path(root)
        path = self.root / 'MANIFEST.private.json'
        if sha(path) != expected_manifest_sha256:
            raise ValueError('FIXED_NAMED_PILOT_MANIFEST_REQUIRED')
        self.manifest_sha = expected_manifest_sha256
        self.manifest = json.loads(path.read_bytes())
        manifest_contract(self.manifest)
        self.resize = resize_with_pad
        self.cache_limit = max_cache_bytes
        self.cache = OrderedDict()
        self.cache_bytes = 0
        self.labels, self.frames, self.codecs, self.rows = {}, {}, {}, []
        stat = self.manifest['statistics']
        if sha(member(self.root, stat['file'])) != stat['sha256']:
            raise ValueError('FROZEN_PILOT_TRAIN_NORMALIZER_REQUIRED')
        stats = json.loads(member(self.root, stat['file']).read_bytes())
        self.statistics = {k: Quantiles(stats[k]['q01'], stats[k]['q99']) for k in ('state', 'actions')}
        for episode in self.manifest['episodes']:
            eid = episode['id']
            labels = episode['labels']
            p = member(self.root, labels['file'])
            if sha(p) != labels['sha256']:
                raise ValueError('SEALED_PILOT_LABEL_BYTES_REQUIRED')
            with np.load(p, allow_pickle=False) as f:
                data = {key: f[key].copy() for key in f.files}
            if (data['state7'].shape != (61, 7) or data['latent7'].shape != (61, 40, 7)
                    or data['state_eef10'].shape != (61, 10)
                    or data['nominal_policy_H40'].shape != (61, 40, 10)
                    or not np.array_equal(data['native_step'], 5703+np.arange(61)*250)
                    or not np.array_equal(data['frame_index'], np.arange(61)*5)
                    or data['phase'].tolist() != episode['expert_stage_audit_only']):
                raise ValueError('COMPLETE_CONTIGUOUS_PILOT_ANCHOR_CONTRACT')
            if set(data['phase']) != {'PREGRASP', 'DESCEND', 'CLOSE', 'LIFT', 'HOLD'}:
                raise ValueError('COMPLETE_GRASP_LIFT_HOLD_PHASE_COVERAGE_REQUIRED')
            for key in ('state7', 'latent7', 'state_eef10', 'nominal_policy_H40'):
                if not np.isfinite(data[key]).all():
                    raise ValueError('FINITE_PILOT_TRAIN_ARRAYS_REQUIRED')
            for shard in episode['frame_shards']:
                if sha(member(self.root, shard['file'])) != shard['sha256']:
                    raise ValueError('SEALED_PILOT_RGB_SHARD_BYTES_REQUIRED')
            self.labels[eid] = data
            self.frames[eid] = episode['mount']
            self.codecs[eid] = GripLogitCodec(episode['mount']['T_policy_from_mount'])
            self.rows.extend((episode, anchor) for anchor in range(61))
        self.original = self

    def __len__(self):
        return len(self.rows)

    def identity(self, index):
        episode, anchor = self.rows[index]
        return dict(episode_id=episode['id'], anchor=anchor,
                    native_step=int(self.labels[episode['id']]['native_step'][anchor]))

    def _frame(self, episode, frame):
        descriptor = next(s for s in episode['frame_shards']
                          if s['first_frame'] <= frame < s['first_frame']+s['frames'])
        name = descriptor['file']
        if name not in self.cache:
            with np.load(member(self.root, name), allow_pickle=False) as f:
                value = {key: f[key].copy() for key in ('rgb', 'native_step', 'state_eef10')}
            size = sum(v.nbytes for v in value.values())
            if size > self.cache_limit:
                raise MemoryError('ONE_FRAME_SHARD_EXCEEDS_DECLARED_CACHE')
            while self.cache and self.cache_bytes+size > self.cache_limit:
                _, removed = self.cache.popitem(last=False)
                self.cache_bytes -= sum(v.nbytes for v in removed.values())
            self.cache[name] = value;self.cache_bytes += size
        self.cache.move_to_end(name)
        value = self.cache[name];offset = frame-descriptor['first_frame']
        rgb = value['rgb'][offset]
        if rgb.shape != (2, 256, 256, 3) or rgb.dtype != np.uint8:
            raise ValueError('ORIGINAL_DUAL_RGB256_REQUIRED')
        return value, offset

    def raw(self, index):
        episode, anchor = self.rows[index]
        data = self.labels[episode['id']]
        frame = int(data['frame_index'][anchor])
        values, offset = self._frame(episode, frame)
        if (values['native_step'][offset] != data['native_step'][anchor]
                or not np.array_equal(values['state_eef10'][offset], data['state_eef10'][anchor])):
            raise ValueError('SAME_ACTUAL_CAPTURE_AND_ACTION_ANCHOR_REQUIRED')
        return (data['state_eef10'][anchor].copy(), data['nominal_policy_H40'][anchor].copy(),
                [image.copy() for image in values['rgb'][offset]], episode['language'])

    def __getitem__(self, index):
        episode, anchor = self.rows[index]
        _, _, images, text = self.raw(index)
        data = self.labels[episode['id']]
        item = images224(images, self.resize)
        item.update(state=pad32(self.statistics['state'].normalize(data['state7'][anchor])).astype(np.float32),
                    actions=pad32(self.statistics['actions'].normalize(data['latent7'][anchor])).astype(np.float32),
                    prompt=text)
        return item

    def catalog_rows(self):
        return [dict(split='TRAIN', task=ep['task'], episode=ep['id'],
                     phase=ep['expert_stage_audit_only'][a], anchor=a) for ep, a in self.rows]
