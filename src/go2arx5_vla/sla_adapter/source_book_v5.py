"""Typed conversion of existing full BOOK source commands; no target admission.

The source arm uses mechanical44mm fractions. Only desired commands are mapped
to the target nominal2–42mm convention. Measured state grip stays unchanged.
"""
from collections import OrderedDict
import json
from pathlib import Path

import numpy as np

from .codec_v1 import Quantiles
from .dense_dataset_v4 import DensePilotDataset, member, sha
from .gripper_v2 import GripLogitCodec, VERSION

SOURCE_MANIFEST_SHA = '1dcb555894a3400b0ec1a032c60a039e2b40ac38705f339ff2d327b10a1c3c9d'
SCHEMA = 'SLA_EXISTING_FULL_BOOK_SOURCE21_V5'
DATASET_ID = 'S_BOOK_FULL21_COMMAND_2HZ_V5'
IDS = tuple(f'BOOK_SOURCE_SEED{i:03}' for i in range(15)) + tuple(
    f'BOOK_GEOMETRY_TRAIN{i:03}' for i in (1, 3, 2, 4, 5, 6))
ANCHORS = np.arange(0, 601, 10)


def full_source_rows(manifest):
    rows = [r for r in manifest['episodes'] if r['task'] == 'BOOK'
            and r.get('complete_commands') == 640 and r.get('actual_observations') == 641]
    if (tuple(r['id'] for r in rows) != IDS
            or any(r['split'] != 'TRAIN' or r['windows'] != 601 for r in rows)):
        raise ValueError('EXACT21_EXISTING_COMPLETE_TRAIN_BOOK_EPISODES_NO_SUFFIX_OR_DEV')
    return rows


def nominal_commands(source10):
    value = np.array(source10, dtype=np.float64, copy=True)
    if value.shape != (640, 10) or not np.isfinite(value).all():
        raise ValueError('FULL640_FINITE_ACTUALLY_CONSUMED_SOURCE_COMMANDS_REQUIRED')
    value[:, 9] = (value[:, 9]-.002/.044)/(.042/.044-.002/.044)
    if ((value[:, 9] < 0) | (value[:, 9] > 1)).any():
        raise ValueError('SOURCE_DESIRED_WIDTH_OUTSIDE_NATIVE2_TO42MM_NO_CLIPPING')
    if not np.allclose(.002+.04*value[:, 9], .044*np.asarray(source10)[:, 9], atol=1e-12, rtol=0):
        raise ValueError('EXACT_SOURCE_DESIRED_MECHANICAL_WIDTH_ROUNDTRIP_REQUIRED')
    return value


def manifest_contract(value):
    episodes = value.get('episodes') or []
    if (value.get('schema') != SCHEMA or value.get('state') != 'committed'
            or value.get('dataset_id') != DATASET_ID or value.get('domain') != 'SOURCE_FIXED_ARM'
            or value.get('source_manifest_sha256') != SOURCE_MANIFEST_SHA
            or value.get('codec') != VERSION or value.get('split') != 'TRAIN'
            or value.get('tasks') != ['BOOK_FROM_TABLE_TO_SLOT']
            or value.get('target_WBC_qualified') is not False
            or value.get('public_RoboTwin_S_cmd_added') != 0
            or value.get('supervised_windows') != 1281
            or value.get('phase_or_object_truth_in_student') is not False
            or value.get('state_gripper_transformed') is not False
            or value.get('command_period_ticks') != 25 or value.get('physics_dt_s') != .002
            or value.get('anchor_period_ticks') != 250 or value.get('source_frame_period_ticks') != 25
            or tuple(e['id'] for e in episodes) != IDS
            or any(e['split'] != 'TRAIN' or e['task'] != 'BOOK_FROM_TABLE_TO_SLOT'
                   or e['windows'] != 61 or e['frames'] != 61 or e['source_frames'] != 641
                   or e['commands'] != 640 or e['source_native_integrations'] != 18500
                   or e['expert_suffix'] is not False for e in episodes)):
        raise ValueError('EXACT_EXISTING_BOOK21_SOURCE_ONLY_COMMAND_AND_SPARSE_RGB_CONTRACT')


class SourceBookDataset(DensePilotDataset):
    """Reuse the lossless bounded shard reader and ordinary pinned RGB resize."""
    validate_manifest = staticmethod(manifest_contract)

    def __init__(self, root, expected_manifest_sha256, *, resize_with_pad, max_cache_bytes=512*1024**2):
        self.root = Path(root)
        path = self.root/'MANIFEST.private.json'
        if sha(path) != expected_manifest_sha256:
            raise ValueError('PINNED_SLA_BOOK21_SOURCE_MANIFEST_REQUIRED')
        self.manifest_sha = expected_manifest_sha256
        self.manifest = json.loads(path.read_bytes());self.validate_manifest(self.manifest)
        self.resize = resize_with_pad;self.cache_limit = max_cache_bytes
        self.cache = OrderedDict();self.cache_bytes = 0
        self.labels, self.frames, self.codecs, self.rows = {}, {}, {}, []
        stat = self.manifest['statistics'];p = member(self.root, stat['file'])
        if sha(p) != stat['sha256']:
            raise ValueError('PINNED_SOURCE_ONLY_BOOK_TRAIN_STATISTICS_REQUIRED')
        stats = json.loads(p.read_bytes())
        self.statistics = {k: Quantiles(stats[k]['q01'], stats[k]['q99']) for k in ('state', 'actions')}
        for ep in self.manifest['episodes']:
            eid = ep['id'];label = ep['labels'];p = member(self.root, label['file'])
            if sha(p) != label['sha256']:
                raise ValueError('PINNED_BOOK_EXECUTED_SOURCE_LABELS_REQUIRED')
            with np.load(p, allow_pickle=False) as f:data = {k: f[k].copy() for k in f.files}
            if (data['state7'].shape != (61, 7) or data['latent7'].shape != (61, 40, 7)
                    or data['state_eef10'].shape != (61, 10) or data['nominal_policy_H40'].shape != (61, 40, 10)
                    or not np.array_equal(data['native_step'], 2500+ANCHORS*25)
                    or not np.array_equal(data['frame_index'], np.arange(61))
                    or data['phase'].tolist() != ep['expert_stage_audit_only']
                    or any(not np.isfinite(data[k]).all() for k in ('state7', 'latent7', 'state_eef10', 'nominal_policy_H40'))):
                raise ValueError('BOOK_SOURCE2HZ_COMPLETE_CONSUMED_H40_AND_ACTUAL_IMAGE_CLOCK')
            shards = ep['frame_shards']
            covered = [i for s in shards for i in range(s['first_frame'], s['first_frame']+s['frames'])]
            if covered != list(range(61)) or any(sha(member(self.root,s['file'])) != s['sha256'] for s in shards):
                raise ValueError('EXACT61_PINNED_LOSSLESS_SOURCE_CAMERA_ANCHORS')
            self.labels[eid] = data;self.frames[eid] = ep['mount']
            self.codecs[eid] = GripLogitCodec(ep['mount']['T_policy_from_mount'])
            self.rows.extend((ep, anchor) for anchor in range(61))
        self.original = self
