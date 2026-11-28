"""Verify same-scene ordinary Teacher pools before a two-start early-Recovery contrast."""
import argparse
from collections import Counter
import io
from pathlib import Path
import tarfile

import numpy as np

from m38_gpu_preflight_v1 import pin, read, require, sha
from scripts.pi0292.durable import write

INDEX = '9d9396b503064b9fc94646db1e38c0c3a3ca1a1a24b254310644bd21fca43f56'
EARLY = ('PREGRASP', 'DESCEND', 'CLOSE', 'LIFT')
CONTROLS = ('MEV1-TRAIN-F1-NEAR_NEG-01-PLACE-red-right', 'MEV1-TRAIN-F1-NEAR_NEG-00-PLACE-blue-left')


def audit(package, recovery, output):
    require(not output.exists(), 'Preserve the prior cohort inventory')
    source = read(pin(package/'index.json', INDEX))
    cold = read(pin(recovery/'MANIFEST.json', '34e4d7432c4372e3eb8d4ff6b8cf4757c99c409f732d2c848cef7ad117f5a029'))
    admitted = read(pin(recovery/'READER-ACTUAL.json', '9d483063886171681b594e244be71fcf64f51dc25aad763f7cc6c4698da0e7bb'))
    require(source['training_episodes'] == 93 and source['training_windows'] == 7916
            and admitted['windows'] == 100 and admitted['current_training_consumption'] == 0,
            'Use the original A93 and the independently reader-verified red suffix')
    pools = []
    for name in CONTROLS:
        entry, = (r for r in source['episodes'] if r['episode'] == name)
        require(entry['split'] == 'TRAIN', 'The control cannot use protected data')
        with tarfile.open(pin(package/entry['path'], entry['archive_sha256'])) as archive:
            with np.load(io.BytesIO(archive.extractfile('labels.npz').read()), allow_pickle=False) as f:
                phases, support = f['phase'], f['valid']
                require(dict(Counter(phases.tolist())) == entry['phases']
                        and len(phases) == entry['windows'] and support.shape == (len(phases), 40),
                        'Archive population or phase metadata changed')
                selected = np.flatnonzero(np.isin(phases, EARLY) & support[:, :10].all(axis=1))
                require(set(phases[selected]) == set(EARLY) and support[selected].all(),
                        'Ordinary early phases must retain full original H40 support')
                pools.append(dict(episode=name, task=entry['task'], scene_sha256=entry['scene_group_sha256'],
                    archive_sha256=entry['archive_sha256'], eligible_rows=selected.tolist(),
                    phase_windows=dict(Counter(phases[selected].tolist())), full_H40_windows=len(selected)))
    red, = cold['episodes']
    require(red['scene_sha256'] == pools[0]['scene_sha256'] and red['task'] == pools[0]['task'],
            'The red contrast must match the actual scene and task')
    with np.load(pin(recovery/red['labels']['file'], red['labels']['sha256']), allow_pickle=False) as f:
        selected = np.flatnonzero(np.isin(f['phase'], EARLY) & f['valid'][:, :10].all(axis=1))
        require(f['valid'][selected].all() and set(f['phase'][selected]) == set(EARLY),
                'The admitted early suffix lacks complete phase support')
        treatment = dict(episode=red['id'], scene_sha256=red['scene_sha256'],
                         eligible_rows=selected.tolist(), phase_windows=dict(Counter(f['phase'][selected].tolist())),
                         full_H40_windows=len(selected), source_manifest_sha256=sha(recovery/'MANIFEST.json'))
    write(output, dict(scope='C1_SAME_SCENE_EARLY_CONTROL_POOL_PREPARATION', data_index_sha256=INDEX,
          ordinary_teacher_pools=pools, admitted_red_early_pool=treatment,
          early_takeover_starts_admitted=1, required_early_takeover_starts=2,
          blue_recovery_admission_pending=True, formal_training_admitted=False,
          intended_matching='ONE_SLOT_PER_EPISODE; EXACT_SCENE_TASK_AND_PHASE; COMMON_OLD_AND_AUX_SLOTS',
          future_fm_batch=dict(old=48, common_new=44, original_R3=2, contrast=2, robodojo=16, libero=16),
          bank_built=False, model_forwards=0, native_integrations=0, optimizer_updates=0))
    print(output, sha(output), flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('package', 'recovery', 'output'):
        cli.add_argument(name, type=Path)
    audit(**vars(cli.parse_args()))
