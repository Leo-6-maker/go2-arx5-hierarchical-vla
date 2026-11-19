"""Freeze 4,000 existing SOURCE_NATIVE TRAIN visits without opening TEST."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from .audit_d4_r0 import checked, sha

S1000 = 'c08e919b5151ccafdfd5b427e02710f1a8e9f9189ca227314f72e5ec0296ab6d'
SOURCE_DATA = '7954a6637564ad39a88f3e173c47687d66c39c50bbaccd9a2ad5f1a361667ab5'
SOURCE_STATS = 'a94e1567bb002621d5841150a8e677e38fa6d02b51af22e4ab6f301317ebd521'
SOURCE_VISITS = 4000
SOURCE_RNG_SEED = 2026111907


def freeze(dataset, checkpoint, output):
    dataset, checkpoint, output = map(Path, (dataset, checkpoint, output))
    manifest = checked(dataset/'MANIFEST.private.json', SOURCE_DATA)
    endpoint = checked(checkpoint/'COMPLETE.json', S1000)
    if (endpoint['bindings']['source_manifest_sha256'] != SOURCE_DATA
            or endpoint['bindings']['statistics_sha256'] != SOURCE_STATS
            or manifest['statistics']['sha256'] != SOURCE_STATS
            or (manifest['horizon'], manifest['nominal_frequency_Hz'], manifest['action_dimensions']) != (50, 30, 14)
            or manifest['label_kind'] != 'FUTURE_LOGGED_STATE_REFERENCE'):
        raise ValueError('J_REQUIRES_THE_ORIGINAL_SEALED_SOURCE_NATIVE_TRAIN')
    checked(dataset/manifest['statistics']['file'], SOURCE_STATS)
    visits_file = dataset/manifest['visit_map']['file']
    if sha(visits_file) != manifest['visit_map']['sha256']:
        raise ValueError('SOURCE32000_VISIT_BANK_CHANGED')
    episodes = [e for e in manifest['episodes'] if e['valid'] and e['split'] == 'TRAIN']
    if len(episodes) != 36:
        raise ValueError('EXPECTED_FROZEN_SOURCE36_TRAIN')
    counts = np.cumsum([ep['windows'] for ep in episodes])
    for ep in episodes:
        audit = ep['audit']
        if (audit['valid_H50_windows'] != ep['windows'] or not audit['last_copied_action_excluded']
                or audit['original_frames']-50 != ep['windows']):
            raise ValueError('SOURCE_WINDOW_CROSSES_ORIGINAL_COPIED_TAIL')
    original = np.load(visits_file, allow_pickle=False)
    if original.shape != (32000,) or original.dtype.kind not in 'iu' or original.min() < 0 or original.max() >= counts[-1]:
        raise ValueError('ORIGINAL_TRAIN_VISITS_NOT_VALID_FOR_SEALED_H50_ROWS')
    # Keep a declared prefix of the previously frozen source sampler. The
    # optimizer never consumes that old run's RNG or absolute target cursor.
    selected = original[:SOURCE_VISITS].astype('<i8', copy=True)
    visit_tasks = Counter(episodes[i]['task'] for i in np.searchsorted(counts, selected, side='right'))
    output.mkdir(mode=0o700)
    bank = output/'VISITS4000.npy'
    with bank.open('xb') as stream:
        np.save(stream, selected, allow_pickle=False)
    result = dict(schema='SLA_J_FROZEN_H50_SOURCE_WINDOW_BANK_V1', scheme='FIRST4000_OF_SEALED_SOURCE32000',
        train_manifest_sha256=SOURCE_DATA, train_statistics_sha256=SOURCE_STATS, source_endpoint_sha256=S1000,
        visits=len(selected), unique_windows=len(set(selected.tolist())), train_episodes=len(episodes),
        by_task=dict(visit_tasks), visit_bank=dict(file=bank.name, sha256=sha(bank)),
        indices_sha256=hashlib.sha256(selected.tobytes()).hexdigest(), source_RNG_seed=SOURCE_RNG_SEED,
        source_TEST_data_reads=0, new_training_updates=0, new_model_generations=0, new_native_ticks=0)
    (output/'J_SOURCE_BANK.private.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result
