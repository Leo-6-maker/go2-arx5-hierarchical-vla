"""One target TRAIN population and one visit bank for both V6 G7 arms."""
from collections import Counter, OrderedDict
import hashlib
from pathlib import Path

import numpy as np

from .checkpoint_v1 import sha
from .codec_v1 import Quantiles
from .config_v1 import SEED
from .dense_dataset_v4 import DensePilotDataset, member
from .dense_g7_v6 import VERSION, previous_command_context
from .gripper_v2 import GripLogitCodec
from .mfree_training_v1 import MobileDataset
from .mixed_visits_v4 import Catalog
from .runtime_v1 import read_pin
from go2arx5_vla.world_model.multitask_score_v6 import INSTRUCTIONS

C20_MANIFEST = '4bcc5ffcd8920d874599a54cca73792c50a49ea6013d65ca7826817a9f80d7de'
C20_STATISTICS = 'f28d50e6eeea09b332d4bd159eca4333bbe23b185f95f1d336d45311e364ab3e'
SCHEMA = 'SLA_V6_G7_TARGET_TRAIN_V1'
CLOCK_FIELDS = ('width_m', 'source_native_s', 'query_native_s')


def reference(values):
    return dict(zip(CLOCK_FIELDS, map(float, values), strict=True))


def legacy_contexts(dataset, time_audit):
    audit = read_pin(time_audit)
    if (audit['schema'] != 'SLA_REFERENCE_CONTINUED_TIME_CODEC_C20_LABEL_AUDIT_V1'
            or not audit['passes'] or audit['dataset_manifest_sha256'] != C20_MANIFEST
            or audit['statistics_sha256'] != C20_STATISTICS or audit['windows'] != 1376
            or dataset.manifest_sha != C20_MANIFEST or len(dataset) != 1376):
        raise ValueError('EXACT_C20_ACTUAL_PREVIOUS_COMMAND_LEDGER_REQUIRED')
    contexts = read_pin(audit['reference_contexts'])['contexts']
    if len(contexts) != len(dataset):
        raise ValueError('C20_REFERENCE_DENOMINATOR_CHANGED')
    output = []
    for i, (episode, anchor) in enumerate(dataset.rows):
        row = contexts[i]
        values = [row[key] for key in CLOCK_FIELDS]
        previous_command_context(reference(values))
        if ((row['index'], row['episode'], row['anchor']) != (i, episode['id'], anchor)
                or abs(values[2] - dataset.identity(i)['native_step']*.002) > 1e-8):
            raise ValueError('C20_REFERENCE_MUST_MATCH_NATIVE_QUERY_AND_ANCHOR')
        output.append(values)
    return np.asarray(output, dtype=np.float64)


class TeacherDataset(DensePilotDataset):
    """Use the same RGB/state join and bounded shard reader as C20."""
    def __init__(self, pin, *, resize_with_pad, cache_bytes):
        self.manifest = read_pin(pin)
        self.manifest_sha = pin['sha256']
        self.root = Path(pin['path']).parent
        manifest = self.manifest
        if (manifest['schema'] != SCHEMA or manifest['state'] != 'committed'
                or manifest['split'] != 'TRAIN' or manifest['representation'] != VERSION
                or any(manifest[key] is not False for key in (
                    'phase_input_to_student', 'teacher_memory_input', 'unconsumed_suffix_is_executed'))):
            raise ValueError('SEALED_V6_G7_TRAIN_MANIFEST_REQUIRED')
        audit = read_pin(dict(path=str(self.root/'ACTUAL.private.json'), sha256=manifest['audit_sha256']))
        if audit['status'] != 'G7_TARGET_TRAIN_READY' or audit['teacher_windows'] != manifest['windows']:
            raise ValueError('ACTUAL_TEACHER_DATA_AND_FP32_AUDIT_REQUIRED')
        stats = manifest['statistics']
        fit = read_pin(dict(path=str(member(self.root, stats['file'])), sha256=stats['sha256']))
        if (fit['split'] != 'TRAIN' or fit['legacy_manifest_sha256'] != C20_MANIFEST
                or fit['teacher_windows'] != manifest['windows'] or fit['legacy_windows'] != 1376):
            raise ValueError('ONE_SHARED_TRAIN_STATISTICS_POPULATION_REQUIRED')
        self.statistics = {key: Quantiles(fit[key]['q01'], fit[key]['q99']) for key in ('state', 'actions')}
        self.cache, self.cache_bytes, self.cache_limit = OrderedDict(), 0, cache_bytes
        self.resize = resize_with_pad
        self.labels, self.frames, self.codecs, self.rows, self.contexts = {}, {}, {}, [], []
        cells = Counter()
        for episode in manifest['episodes']:
            eid, n = episode['id'], episode['windows']
            label = member(self.root, episode['labels']['file'])
            if (episode['split'] != 'TRAIN' or eid in self.labels or not 1 <= n <= 180
                    or sha(label) != episode['labels']['sha256']):
                raise ValueError('SEALED_UNIQUE_COMPLETE_TEACHER_EPISODE_REQUIRED')
            with np.load(label, allow_pickle=False) as stream:
                data = {key: stream[key].copy() for key in stream.files}
            if (data['state7'].shape != (n, 7) or data['latent7'].shape != (n, 40, 7)
                    or data['state_eef10'].shape != (n, 10) or data['nominal_policy_H40'].shape != (n, 40, 10)
                    or not np.array_equal(data['native_step'], 5703 + 250*np.arange(n))
                    or not np.array_equal(data['frame_index'], np.arange(n))
                    or not np.array_equal(data['supervision_mask'], np.ones((n, 40), dtype=bool))
                    or not np.array_equal(data['consumed_mask'], np.tile([True]*10 + [False]*30, (n, 1)))
                    or data['phase'].tolist() != episode['expert_stage_audit_only']
                    or data['previous_actual_command'].shape != (n, 3)):
                raise ValueError('COMPLETE_G7_H40_WITH_SEPARATE_CONFIRMED_P10_REQUIRED')
            if any(not np.isfinite(data[key]).all() for key in ('state7', 'latent7', 'state_eef10', 'nominal_policy_H40')):
                raise ValueError('FINITE_TEACHER_G7_LABELS_REQUIRED')
            for ctx, step in zip(data['previous_actual_command'], data['native_step'], strict=True):
                previous_command_context(reference(ctx))
                if abs(ctx[2] - step*.002) > 1e-8:
                    raise ValueError('EXACT_TEACHER_COMMAND_CLOCK_REQUIRED')
            start = 0
            for shard in episode['frame_shards']:
                if (shard['first_frame'] != start or not 1 <= shard['frames'] <= 16
                        or sha(member(self.root, shard['file'])) != shard['sha256']):
                    raise ValueError('CONTIGUOUS_VERIFIED_CURRENT_RGB_SHARDS_REQUIRED')
                start += shard['frames']
            if start != n:
                raise ValueError('ONE_PAUSED_RGB_FOR_EACH_FULL_TRAIN_LABEL_REQUIRED')
            self.labels[eid] = data
            self.frames[eid] = episode['mount']
            self.codecs[eid] = GripLogitCodec(episode['mount']['T_policy_from_mount'])
            self.rows.extend((episode, anchor) for anchor in range(n))
            self.contexts.extend(data['previous_actual_command'])
            cells[episode['task'], episode['family']] += 1
        expected = {(task, family) for task in INSTRUCTIONS for family in ('F1', 'F2')}
        if set(cells) != expected or min(cells.values()) < 2 or len(self) != manifest['windows']:
            raise ValueError('SIX_TASK_TWO_FAMILY_COMPLETE_TEACHER_COVERAGE_REQUIRED')


class MultitaskDataset:
    def __init__(self, pin, *, resize_with_pad, cache_bytes):
        self.new = TeacherDataset(pin, resize_with_pad=resize_with_pad, cache_bytes=cache_bytes//2)
        bindings = self.new.manifest['legacy_C20']
        self.legacy = MobileDataset(bindings['dataset'], C20_MANIFEST,
            resize_with_pad=resize_with_pad, max_cache_bytes=cache_bytes//2)
        self.legacy.contexts = legacy_contexts(self.legacy, bindings['time_audit'])
        self.statistics = self.legacy.statistics = self.new.statistics
        self.rows = self.legacy.rows + self.new.rows
        self.manifest, self.manifest_sha = self.new.manifest, self.new.manifest_sha
        self.codecs = self.legacy.codecs | self.new.codecs
        if len(self.codecs) != len(self.legacy.codecs)+len(self.new.codecs):
            raise ValueError('LEGACY_AND_NEW_EPISODE_IDS_MUST_BE_DISJOINT')
        self.original = self

    def __len__(self):
        return len(self.rows)

    def _entry(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        return (self.legacy, index) if index < len(self.legacy) else (self.new, index-len(self.legacy))

    def __getitem__(self, index):
        dataset, local = self._entry(index)
        return dataset[local]

    def raw(self, index):
        dataset, local = self._entry(index)
        return dataset.raw(local)

    def identity(self, index):
        dataset, local = self._entry(index)
        return dataset.identity(local)

    def reference(self, index):
        dataset, local = self._entry(index)
        return reference(dataset.contexts[local])


def visit_bank(dataset):
    """Exactly 200 blocks of 32 old and 128 new slots; only task counts are forced."""
    legacy = Catalog('T', dataset.legacy.manifest_sha, dataset.legacy.catalog_rows())
    teacher = Catalog('T', dataset.new.manifest_sha, dataset.new.catalog_rows())
    if set(teacher.tree) != set(INSTRUCTIONS):
        raise ValueError('ALL_FIXED_TEACHER_INSTRUCTIONS_REQUIRED')
    rows = np.empty((200, 160), dtype='<i8')
    order = np.random.default_rng(SEED+60).permutation(INSTRUCTIONS).tolist()
    counts = Counter()
    for block, values in enumerate(rows):
        for i in range(32):
            values[i] = legacy.sample(SEED, block*32+i).row_index
        for i in range(128):
            ordinal = block*128+i
            task = order[ordinal % len(order)]
            values[32+i] = len(legacy.rows)+teacher.sample(SEED+61, ordinal, task=task).row_index
            counts[task] += 1
        np.random.default_rng(SEED+62+block).shuffle(values)
    values = rows.reshape(-1)
    return values, dict(schema='SLA_V6_G7_FIXED_TARGET_VISIT_BANK', seed=SEED,
        visit_map_sha256=hashlib.sha256(values.tobytes()).hexdigest(),
        legacy_catalog_sha256=legacy.sha256, teacher_catalog_sha256=teacher.sha256,
        legacy_visits=6400, teacher_visits=25600, teacher_task_visits=dict(counts),
        slot_count=32000, updates_per_superblock=5, visited_unique_windows=len(set(values)))
