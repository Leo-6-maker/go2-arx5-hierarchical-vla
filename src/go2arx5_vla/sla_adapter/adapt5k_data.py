"""ADAPT5K reads complete, independently admitted target plans with frozen actions."""
from pathlib import Path

import numpy as np

from .codec_v1 import Quantiles
from .dense_g7_v6 import VERSION
from .multitask_data_v6 import C20_STATISTICS, TeacherDataset, reference
from .runtime_v1 import read_pin
from .scale20k_target import ScaleTargetDataset

CAPTURE_SCHEMA = 'SLA_TARGET_ADAPTATION_NEW_EXPERTS_V1'
STATS_SCHEMA = 'SLA_ADAPT5K_TARGET_STATE_STATISTICS_V2'


class CompletePlanBatch(TeacherDataset):
    def __init__(self, manifest, receipt, *, statistics, resize_with_pad, cache_bytes):
        self.manifest = read_pin(manifest)
        self.manifest_sha = manifest['sha256']
        self.root = Path(manifest['path']).parent
        proof = read_pin(receipt)
        episodes = self.manifest['episodes']
        admitted = {row['id']: row for row in proof['cases'] if row['admitted'] is True}
        if (self.manifest['schema'] != CAPTURE_SCHEMA or self.manifest['representation'] != VERSION
                or self.manifest['statistics_sha256'] != C20_STATISTICS
                or self.manifest['live_SCALE_training_data_changed'] is not False
                or proof['dataset_sha256'] != self.manifest_sha
                or not episodes or len(admitted) != proof['admitted_episodes'] or len(admitted) != len(episodes)
                or set(admitted) != {episode['id'] for episode in episodes}
                or self.manifest['windows'] != proof['admitted_windows']):
            raise ValueError('ADAPT5K_REQUIRES_ACTUAL_ADMISSION_OF_NEW_TRAIN_MEMBERS')
        for ep in episodes:
            row = admitted[ep['id']]
            if (ep['label_kind'] != 'expert_query_plan_H40' or ep['individual_pass'] is not True
                    or ep['task'] != row['instruction'] or ep['windows'] != row['windows']
                    or row['confirmed_P10_atoms'] != 10 * ep['windows']
                    or any(ep[k] != row[k] for k in ('scene_sha256', 'reset', 'native_result_sha256'))):
                raise ValueError('ADAPT5K_REQUIRES_A_COMPLETE_PLAN_AND_THE_ORIGINAL_P10_RECEIPT')
        self.statistics = statistics
        self._load_episodes(resize_with_pad, cache_bytes)
        if any(data[key].dtype != np.bool_ for data in self.labels.values()
               for key in ('supervision_mask', 'consumed_mask')):
            raise ValueError('ADAPT5K_REQUIRES_EXPLICIT_BOOLEAN_H40_AND_P10_MASKS')


def fit_target_state(datasets, action_statistics):
    """Fit native seven-axis state from training labels, never source or eval rows."""
    state = np.concatenate([data['state7'] for dataset in datasets for data in dataset.labels.values()])
    fit = Quantiles.fit(state)
    normalized = fit.normalize(state)
    bins = np.digitize(normalized, np.linspace(-1, 1, 257)[:-1]) - 1
    outside_action = np.zeros(7, dtype=np.int64)
    action_count = 0
    for dataset in datasets:
        for data in dataset.labels.values():
            actions = data['latent7'].reshape(-1, 7)
            outside_action += np.count_nonzero((actions < action_statistics.q01)
                                               | (actions > action_statistics.q99), axis=0)
            action_count += len(actions)
    return dict(schema=STATS_SCHEMA, split='TRAIN', fit_windows=len(state), state=fit.summary(state),
                state_token_low_boundary_fraction=(bins <= 0).mean(axis=0).tolist(),
                state_token_high_boundary_fraction=(bins == 255).mean(axis=0).tolist(),
                frozen_action_statistics_sha256=C20_STATISTICS, action_points=action_count,
                action_outside_frozen_quantiles_fraction=(outside_action / action_count).tolist())


class AdaptTargetDataset:
    """Use the original Go2 reader plus verified new episodes. Partial stays excluded."""
    def __init__(self, legacy_spec, captures, *, resize_with_pad, cache_bytes=512*1024**2):
        captures = tuple(captures)
        if not captures:
            raise ValueError('ADAPT5K_REQUIRES_AT_LEAST_ONE_ADMITTED_NEW_EPISODE')
        legacy = ScaleTargetDataset(legacy_spec, resize_with_pad=resize_with_pad, cache_bytes=cache_bytes//2)
        original_stats = legacy.statistics
        new = [CompletePlanBatch(c['manifest'], c['admission'], statistics=original_stats,
                                 resize_with_pad=resize_with_pad, cache_bytes=cache_bytes//(2*len(captures)))
               for c in captures]
        self.members = [legacy, *new]
        all_episodes = [ep for member in self.members for ep in member.manifest['episodes']]
        old_scenes = {ep['scene_sha256'] for ep in legacy.manifest['episodes'] if 'scene_sha256' in ep}
        new_episodes = [ep for member in new for ep in member.manifest['episodes']]
        new_scenes = {ep['scene_sha256'] for ep in new_episodes}
        if (len({ep['id'] for ep in all_episodes}) != len(all_episodes)
                or len(new_scenes) != len(new_episodes) or new_scenes & old_scenes
                or len({ep['native_result_sha256'] for ep in new_episodes}) != len(new_episodes)):
            raise ValueError('ADAPT5K_REQUIRES_DISTINCT_NEW_TRAIN_EPISODES_AND_SCENES')
        self.manifest = dict(episodes=all_episodes)
        self.original_windows = len(legacy)
        self.new_episodes = len(new_episodes)
        self.rows = [row for member in self.members for row in member.rows]
        self.labels = {eid: labels for member in self.members for eid, labels in member.labels.items()}
        self.codecs = {eid: codec for member in self.members for eid, codec in member.codecs.items()}
        self.statistics = {'state': original_stats['state'], 'actions': original_stats['actions']}

    def fit_state(self):
        return dict(fit_target_state(self.members, self.statistics['actions']),
                    training_manifest_sha256=[member.manifest_sha for member in self.members])

    def bind_state(self, pin):
        value = read_pin(pin)
        manifests = {member.manifest_sha: len(member) for member in self.members}
        fit_members = set(value['training_manifest_sha256'])
        if (value['schema'] != STATS_SCHEMA or value['split'] != 'TRAIN'
                or value['frozen_action_statistics_sha256'] != C20_STATISTICS
                or self.members[0].manifest_sha not in fit_members or len(fit_members) < 2
                or not fit_members <= manifests.keys()
                or value['fit_windows'] != sum(manifests[key] for key in fit_members)):
            raise ValueError('ADAPT5K_REQUIRES_FROZEN_D1_STATE_STATISTICS_ACROSS_STAGES')
        stats = value['state']
        self.statistics['state'] = Quantiles(stats['q01'], stats['q99'])
        for member in self.members:
            if isinstance(member, ScaleTargetDataset):
                member.new.statistics = member.legacy.statistics = self.statistics
            member.statistics = self.statistics

    def __len__(self):
        return len(self.rows)

    def _entry(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        for member in self.members:
            if index < len(member):
                return member, index
            index -= len(member)

    def __getitem__(self, index):
        member, local = self._entry(index)
        return member[local]

    def raw(self, index):
        member, local = self._entry(index)
        return member.raw(local)

    def identity(self, index):
        member, local = self._entry(index)
        return member.identity(local)

    def reference(self, index):
        member, local = self._entry(index)
        if member is self.members[0]:
            return member.reference(local)
        return reference(member.contexts[local])
