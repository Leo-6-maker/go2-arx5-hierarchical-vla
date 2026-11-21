"""Adopt cumulative Core TRAIN admissions only at a registered H200 phase boundary."""
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from go2arx5_vla.sla_adapter.adapt5k_data import AdaptTargetDataset, CompletePlanBatch
from go2arx5_vla.sla_adapter.scale20k_contract import bank_sha256
from scripts.pi0292.durable import OwnerLock
from scripts.sla.hopper_checkpoint import BOUNDARIES
from scripts.sla.hopper_inputs import D2_SPEC, PLAN, STEPS, BATCH, stage_target
from scripts.sla.hopper_parent_custody import digest, sync_dir, write_once

POPULATION = '9e483c85eb90cc2e9d1da66b9f3f8dff6606c6173b66755bcffcc4c7e5f91279'


def pinned_file(path, expected):
    if digest(path) != expected:
        raise ValueError('Changed target curriculum input '+path.name)
    return json.loads(path.read_bytes())


def load_snapshot(folder, seal_sha, population):
    rows = pinned_file(population, POPULATION)['populations']['TRAIN']
    permitted = {r['condition_id']: r for r in rows}
    seal = pinned_file(folder/'SEALED.private.json', seal_sha)
    manifest = pinned_file(folder/'MANIFEST.private.json', seal['manifest_sha256'])
    proof = pinned_file(folder/'ACTUAL.private.json', seal['admission_sha256'])
    if (seal['plan'] != PLAN or manifest['plan'] != PLAN or seal['split'] != 'TRAIN'
            or manifest['split'] != 'TRAIN' or proof['dataset_sha256'] != seal['manifest_sha256']
            or not 0 < seal['count'] == len(manifest['episodes']) == len(proof['cases']) <= 240
            or seal['windows'] != sum(ep['windows'] for ep in manifest['episodes'])):
        raise ValueError('Incomplete cumulative TRAIN snapshot')
    ids, native, content, scenes = set(), set(), set(), {}
    for ep, admission, pin in zip(manifest['episodes'], proof['cases'], seal['episode_sha256'], strict=True):
        eid = ep['id']
        card = permitted[eid]
        members = [ep['labels'], *ep['frame_shards']]
        signature = tuple(member['sha256'] for member in members)
        scene = scenes.setdefault(card['scene_family'], ep['scene_sha256'])
        if (not admission['admitted'] or admission['id'] != eid or admission['episode_sha256'] != pin
                or ep['split'] != 'TRAIN' or ep['family'] != card['family'] or ep['task'] != card['instruction']
                or ep['reset'] != card['reset_seed'] or ep['scene_sha256'] != scene or eid in ids
                or ep['native_result_sha256'] in native or signature in content
                or not 0 < admission['native_ticks'] <= card['native_max_ticks']):
            raise ValueError('Unregistered, duplicate or unqualified Core capture')
        for member in members:
            rel = Path(member['file'])
            if (rel.parent != Path(eid) or not (folder/rel).resolve(strict=True).is_relative_to(folder.resolve())
                    or digest(folder/rel) != member['sha256']):
                raise ValueError('TRAIN payload membership or content mismatch')
        ids.add(eid)
        native.add(ep['native_result_sha256'])
        content.add(signature)
    if len(set(scenes.values())) != len(scenes):
        raise ValueError('Registered scene families unexpectedly alias')
    return seal, manifest


class WithCoreTrain(AdaptTargetDataset):
    """Retain all D2 indices and its bound normalization, append independently admitted captures."""
    def __init__(self, d2, folder, seal_sha, population):
        from openpi_client.image_tools import resize_with_pad

        seal, manifest = load_snapshot(folder, seal_sha, population)
        old = d2.manifest['episodes']
        if len(old) != 26 or len(d2) != 2031 or d2.new_episodes != 3:
            raise ValueError('Construct cumulative views on the original D2 only')
        old_ids = {ep['id'] for ep in old}
        old_scenes = {ep.get('scene_sha256') for ep in old}
        old_results = {ep.get('native_result_sha256') for ep in old}
        new = manifest['episodes']
        if any(ep['id'] in old_ids or ep['scene_sha256'] in old_scenes or
               ep['native_result_sha256'] in old_results for ep in new):
            raise ValueError('Core capture overlaps a D2 training member')
        self.statistics = d2.statistics
        extra = CompletePlanBatch(dict(path=str(folder/'MANIFEST.private.json'), sha256=seal['manifest_sha256']),
                                  dict(path=str(folder/'ACTUAL.private.json'), sha256=seal['admission_sha256']),
                                  statistics=self.statistics, resize_with_pad=resize_with_pad, cache_bytes=256*1024**2)
        self.members = [*d2.members, extra]
        self.original_windows = d2.original_windows
        self.new_episodes = d2.new_episodes + len(new)
        self.rows = [*d2.rows, *extra.rows]
        self.manifest = dict(episodes=[*old, *new])
        self.labels = {**d2.labels, **extra.labels}
        self.codecs = {**d2.codecs, **extra.codecs}
        self.seal = seal


def bank_at_boundary(folder, view, phase, seed):
    if phase not in BOUNDARIES:
        raise ValueError('Unregistered target switch phase')
    folder.mkdir(parents=True, exist_ok=True)
    lock = OwnerLock(folder/'OWNER.lock')
    try:
        final = folder/'bank.verified'
        if not final.exists():
            incoming = Path(tempfile.mkdtemp(prefix='bank.incoming.', dir=folder))
            arr, counts = stage_target(view, phase, seed)
            with (incoming/'TARGET.npy').open('xb') as f:
                np.save(f, arr, allow_pickle=False)
                f.flush()
                os.fsync(f.fileno())
            write_once(incoming/'BANK.private.json', dict(phase=phase, seed=seed, initial_data=D2_SPEC,
                       target_manifest=view.seal['manifest_sha256'], counts=counts,
                       bank_sha256=bank_sha256(arr), file_sha256=digest(incoming/'TARGET.npy')))
            sync_dir(incoming)
            incoming.rename(final)
            sync_dir(folder)
        proof = json.loads((final/'BANK.private.json').read_bytes())
        path = final/'TARGET.npy'
        if (proof['phase'] != phase or proof['seed'] != seed or proof['target_manifest'] != view.seal['manifest_sha256']
                or digest(path) != proof['file_sha256'] or proof['counts']['new_quota_per_update'] != min(48, 4*view.new_episodes)):
            raise ValueError('Conflicting target bank at this curriculum boundary')
        actual = np.load(path, allow_pickle=False, mmap_mode='r')
        if (actual.dtype != np.int64 or actual.shape != ((STEPS-phase)*BATCH['target'],)
                or actual.min() < 0 or actual.max() >= len(view) or bank_sha256(actual) != proof['bank_sha256']):
            raise ValueError('Incomplete switched target bank')
        return actual, proof
    finally:
        lock.close()
