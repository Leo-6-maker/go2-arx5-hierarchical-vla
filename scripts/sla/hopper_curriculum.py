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

from scripts.sla.hopper_snapshot_relay import verified


class WithCoreTrain(AdaptTargetDataset):
    """Retain all D2 indices and its bound normalization, append independently admitted captures."""
    def __init__(self, d2, folder, seal_sha, population):
        from openpi_client.image_tools import resize_with_pad

        seal, manifest, _ = verified(folder, population, seal_sha)
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
