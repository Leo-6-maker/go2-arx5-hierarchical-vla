"""Seal all identities before the first pilot, after actual commissioning."""
import argparse
import json
from pathlib import Path
import time

from .config import PARAMETERS, PILOTS, TRAIN, DEV, AUTHORITY, layout, canonical_hash
from .engine import DRIVER_VERSION
from go2arx5_vla.task_data.io import file_hash, write


def seal(root, source):
    root = Path(root)
    target = root / 'TEACHER_AND_LAYOUT_SEAL.json'
    if target.exists():
        raise RuntimeError('Existing seal is immutable')
    code = json.loads((root / 'CODE_SEAL.json').read_text())
    assert code['source_sha'] == source
    for name, digest in code['files'].items():
        assert file_hash(root / 'code' / name) == digest, name
    ledger = json.loads((root / 'RESOURCE_ACCOUNTING.json').read_text())
    assert not any(row['id'].startswith(('P', 'T', 'D')) for row in ledger['starts'])
    evidence = {}
    for identifier in ('C29100', 'C29101', 'C29102', 'C29103'):
        folder = root / 'episodes' / identifier
        result = json.loads((folder / 'RESULT.json').read_text())
        assert result['status'] == 'VALID_COMPLETE' and result['physics_steps'] == 16000
        assert result['score']['safety_valid']
        assert result['score']['persistent_success'] == (identifier in ('C29100', 'C29103'))
        if identifier == 'C29103':
            assert result['source_sha'] == source
        evidence[identifier] = {'source_sha': result['source_sha'],
            'result_sha256': file_hash(folder / 'RESULT.json'),
            'physics_sha256': file_hash(folder / 'physics.npz')}
    cards = {identifier: layout(identifier) for identifier in PILOTS + TRAIN + DEV}
    assert len({card['seed'] for card in cards.values()}) == 88
    assert len({card['group_id'] for card in cards.values()}) == 88
    value = dict(authority=AUTHORITY, source_sha=source, code_files=code['files'],
        parameter_sha256=canonical_hash(PARAMETERS), parameters=PARAMETERS,
        driver_version=DRIVER_VERSION, sealed_epoch=time.time(), layouts=cards,
        layout_sha256=canonical_hash(cards), commissioning=evidence,
        formal_order=list(TRAIN + DEV), pilot_order=list(PILOTS),
        statistics_rule=dict(population='ADMITTED_FORMAL_TRAIN_ONLY',
            state='601 legal observation anchors per admitted episode, equal anchor weight',
            action='601 H40 windows per admitted episode, equal window/timestep weight; repeated overlap intentional',
            normalizer='stock min/max', use_percentiles=False, clip_outliers=False,
            DEV_refit=False, inverse_clipping='stock decoder clipping must be measured'),
        forbidden=dict(neural=0, training=0, WBC=0, hardware=0, Hopper_PBS=0),
        selected='NONE', deployment='NONE')
    write(target, value)
    print(json.dumps(dict(status='SEALED_BEFORE_FIRST_PILOT', source_sha=source,
                         seal_sha256=file_hash(target), layouts=88)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--source-sha', required=True)
    args = parser.parse_args()
    seal(args.root, args.source_sha)
