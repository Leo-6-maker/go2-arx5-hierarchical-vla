"""Freeze the verified RoboDojo ARX X5 expansion over the original splits."""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from go2arx5_vla.sla_adapter.joint_source_d4 import SOURCE_MANIFEST
from go2arx5_vla.sla_adapter.source_native_v6 import COORDINATES, HORIZON, SCHEMA, anchor_strata, load_numeric
from scripts.pi0292.durable import read, write


REVISION = '1127bda94f172d757386945af3eb3a6754d1a4e6'


def prepare(old_root, expansion_root, expansion_receipt, output):
    old_manifest = old_root/'MANIFEST.private.json'
    if sha(old_manifest) != SOURCE_MANIFEST:
        raise ValueError('SCALE20K_ROBODOJO_ORIGINAL_SPLITS_MUST_MATCH_CUSTODY')
    prior = read(old_manifest)
    if prior['TEST_labels_read'] is not False or prior['HF_revision'] != REVISION:
        raise ValueError('SCALE20K_ROBODOJO_TEST_LABELS_STAY_CLOSED')
    added = read(expansion_receipt)
    if (added['revision'] != REVISION or added['new_episodes'] != 12 or len(added['rows']) != 12
            or added['nominal_hz'] != 30 or added['native_horizon'] != HORIZON):
        raise ValueError('SCALE20K_EXACT_REGISTERED_ARX_X5_EXPANSION_REQUIRED')
    originals = [row for row in prior['episodes'] if row['valid'] and row['split'] in ('TRAIN','SOURCE_DEV')]
    if (sum(row['split']=='TRAIN' for row in originals),sum(row['split']=='SOURCE_DEV' for row in originals)) != (36,12):
        raise ValueError('SCALE20K_ROBODOJO_REQUIRES_UNCHANGED_ORIGINAL_36_TRAIN_12_DEV')

    numeric = []
    for row in originals:
        pin = row['arrays']
        path = old_root/pin['file']
        if sha(path) != pin['sha256']:
            raise ValueError('SCALE20K_ROBODOJO_VERIFIED_ORIGINAL_NUMERIC_BYTES_CHANGED')
        with np.load(path,allow_pickle=False) as f:
            state,actions = f['state'],f['actions']
        numeric.append((row,state,actions))
    for row in added['rows']:
        path = expansion_root/row['filename']
        if sha(path) != row['source_sha256'] or row['split'] != 'TRAIN' or not row['admitted']:
            raise ValueError('SCALE20K_ROBODOJO_EXPANSION_FILE_OR_TRAIN_ADMISSION_CHANGED')
        state,actions,instruction,audit = load_numeric(path)
        if audit['valid_H50_windows'] != row['window_count']:
            raise ValueError('SCALE20K_ROBODOJO_NATIVE_WINDOWS_CHANGED')
        numeric.append((dict(id=row['episode'],task=row['task'],instruction=instruction,valid=True,split='TRAIN',
                             raw_path=str(path.resolve()),raw_sha256=row['source_sha256'],windows=len(actions)),state,actions))
    if (len({row['id'] for row,_,_ in numeric}) != 60 or len({row['raw_sha256'] for row,_,_ in numeric}) != 60
            or any(state.shape != (row['windows'],14) or actions.shape != (row['windows'],HORIZON,14)
                   for row,state,actions in numeric)):
        raise ValueError('SCALE20K_ROBODOJO_TYPED_EPISODES_REQUIRE_DISTINCT_IDENTITIES')

    output.mkdir(mode=0o700)
    storage = output/'numeric'
    storage.mkdir()
    train = [(s,a) for row,s,a in numeric if row['split']=='TRAIN']
    states = np.concatenate([s for s,_ in train],axis=0)
    actions = np.concatenate([a.reshape(-1,14) for _,a in train],axis=0)
    statistics = {}
    for field,values in (('state',states),('actions',actions)):
        q = np.quantile(values,[.01,.25,.75,.99],axis=0)
        statistics[field] = dict(q01=q[0].tolist(),q99=q[3].tolist(),IQR=(q[2]-q[1]).tolist(),samples=len(values))
    stats = output/'STATISTICS.json'
    write(stats,dict(**statistics,split='TRAIN',episodes=48,horizon=HORIZON,nominal_frequency_Hz=30))
    rows = []
    for index,(row,state,action) in enumerate(numeric):
        path = storage/f'{index:03d}.npz'
        threshold = .01*np.asarray(statistics['state']['IQR'])
        dynamic = anchor_strata(state,action,statistics['state']['IQR'])
        delta = action[:,-1].copy()
        delta[:,[6,13]] -= state[:,[6,13]]
        future_span_dynamic = np.any(abs(delta)>threshold,axis=-1)
        with path.open('xb') as handle:
            np.savez(handle,state=state,actions=action,dynamic=dynamic,future_span_dynamic=future_span_dynamic)
        rows.append(dict(row, arrays=dict(file=str(path.relative_to(output)),sha256=sha(path)),
                         admission='SCALE20K_ARX_X5_NATIVE_H50_VERIFIED'))
    summary = dict(schema=SCHEMA,recipe='SLA_SCALE20K_ROBODOJO_TRAIN48_DEV12_V1',coordinates=COORDINATES,
        HF_revision=REVISION,horizon=HORIZON,nominal_frequency_Hz=30,TEST_labels_read=False,
        original_manifest_sha256=SOURCE_MANIFEST,expansion_manifest_sha256=sha(expansion_receipt),
        episodes=rows,statistics=dict(file=stats.name,sha256=sha(stats)),target_commands=False)
    write(output/'MANIFEST.private.json',summary)
    return dict(TRAIN=48,SOURCE_DEV=12,protected_SOURCE_TEST=12,statistics_sha256=sha(stats),
                manifest_sha256=sha(output/'MANIFEST.private.json'),
                train_windows=sum(row['windows'] for row in rows if row['split']=='TRAIN'),new_optimizer_updates=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('old_root','expansion_root','expansion_receipt','output'):
        parser.add_argument(name,type=Path)
    args = parser.parse_args()
    start = time.monotonic()
    proof = prepare(args.old_root,args.expansion_root,args.expansion_receipt,args.output)
    write(args.output/'ACTUAL.private.json',dict(proof,elapsed_s=time.monotonic()-start))
    print(json.dumps(proof))


if __name__ == '__main__':
    main()
