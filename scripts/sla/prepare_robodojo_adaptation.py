"""Freeze the verified raw archives for the next RoboDojo TRAIN200 version."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from go2arx5_vla.sla_adapter.checkpoint_v1 import sha
from go2arx5_vla.sla_adapter.runtime_v1 import read_pin
from go2arx5_vla.sla_adapter.source_native_v6 import COORDINATES, HORIZON, SCHEMA, load_numeric
from scripts.pi0292.durable import write
from scripts.sla.prepare_robodojo_scale20k import REVISION, write_numeric


BASE_MANIFEST = '57e9836d62bf0775bca7c7b7396ab4251d299a6d23116de66771af0fd3960991'
INVENTORY = 'e7d944cdeb738b2df090f3593cd4b9b574a1f3ce80babccb123fb2e3f3603343'
TRANCHES = (48,104)


def inventory_files(value):
    if isinstance(value,dict):
        if isinstance(value.get('path'),str) and value['path'].endswith('.hdf5'):
            yield value
        for child in value.values():
            yield from inventory_files(child)
    elif isinstance(value,list):
        for child in value:
            yield from inventory_files(child)


def prepare(spec):
    root = Path(spec['output'])
    base = Path(spec['base_numeric_custody'])
    active = read_pin(dict(path=str(base/'MANIFEST.private.json'),sha256=BASE_MANIFEST))
    receipt = read_pin(dict(path=spec['registered_metadata']['path'],sha256=INVENTORY))
    historical = list(inventory_files(read_pin(spec['original_protected_file_inventory'])))
    historical_paths = {row['path'] for row in historical}
    historical_hashes = {row['sha256'] for row in historical}
    if (len(historical_paths) != 60 or len(historical_hashes) != 60
            or active['HF_revision'] != REVISION or receipt['revision'] != REVISION
            or active['TEST_labels_read'] is not False or active['target_commands'] is not False):
        raise ValueError('EXPECTED_THE_FROZEN_ROBODOJO_SOURCE_IDENTITY_AND_PROTECTED_SPLITS')
    original_rows = active['episodes']
    if Counter(row['split'] for row in original_rows) != {'TRAIN':48,'SOURCE_DEV':12}:
        raise ValueError('EXPECTED_CURRENT_TRAIN48_AND_THE_ORIGINAL_DEV12')
    prior_hashes = {row['raw_sha256'] for row in original_rows}
    candidates = receipt['selection']['152']['rows']
    if (len(candidates) != sum(TRANCHES) or len({row['sha256'] for row in candidates}) != sum(TRANCHES)
            or len({row['file'] for row in candidates}) != sum(TRANCHES)
            or any(row['sha256'] in historical_hashes or row['sha256'] in prior_hashes
                   or row['file'] in historical_paths for row in candidates)):
        raise ValueError('TRAIN200_CANDIDATES_MUST_BE_DISJOINT_FROM_ALL_FROZEN_SPLITS')

    numeric = []
    for row in original_rows:
        p = base/row['arrays']['file']
        if sha(p) != row['arrays']['sha256']:
            raise ValueError('Frozen original source numeric array changed')
        with np.load(p,allow_pickle=False) as arrays:
            numeric.append((row,arrays['state'],arrays['actions']))
    offset = 0
    for pin, count in zip(spec['tranches'],TRANCHES,strict=True):
        result = read_pin(pin)
        batch = candidates[offset:offset+count]
        if (result['revision'] != REVISION or result['registered'] != count or result['passed'] != count
                or [r['sha256'] for r in result['cases']] != [r['sha256'] for r in batch]):
            raise ValueError('Expected the complete exact-stage raw download audit')
        for registered, audited in zip(batch,result['cases'],strict=True):
            filename = registered['task']+'-'+Path(registered['file']).name
            raw = Path(pin['path']).parent/filename
            if (audited['file'] != filename or audited['task'] != registered['task']
                    or audited['numeric_and_RGB_pass'] is not True or audited['split'] != 'TRAIN_CANDIDATE'
                    or raw.stat().st_size != registered['bytes'] or sha(raw) != registered['sha256']):
                raise ValueError('Verified raw archive changed before composition: '+filename)
            state, actions, instruction, audit = load_numeric(raw)
            if audit['valid_H50_windows'] != audited['windows']:
                raise ValueError('Verified source H50 window count changed')
            row = dict(id='ADAPT-RD-'+registered['sha256'],split='TRAIN',valid=True,
                       task=registered['task'],instruction=instruction,windows=len(state),
                       raw_path=str(raw.resolve()),raw_sha256=registered['sha256'],
                       public_archive=registered['file'],admission='SOURCE_NATIVE_H50_ADAPTATION_V1')
            numeric.append((row,state,actions))
        offset += count
    if len({r['id'] for r,_,_ in numeric}) != 212 or any(
            state.shape != (r['windows'],14) or actions.shape != (r['windows'],HORIZON,14)
            for r,state,actions in numeric):
        raise ValueError('TRAIN200 requires 212 typed distinct numeric episodes')
    rows,stats = write_numeric(root,numeric)
    data = dict(schema=SCHEMA,recipe='SLA_TARGET_ADAPTATION_ROBODOJO_TRAIN200_DEV12_V1',
                HF_revision=REVISION,coordinates=COORDINATES,horizon=HORIZON,nominal_frequency_Hz=30,
                episodes=rows,statistics=dict(file=stats.name,sha256=sha(stats)),
                original_manifest_sha256=BASE_MANIFEST,registered_metadata_sha256=INVENTORY,
                TEST_labels_read=False,target_commands=False,data_site_binding='PENDING')
    manifest = root/'MANIFEST.private.json'
    write(manifest,data)
    result = dict(train_count=200,unchanged_dev_raw_identities=12,protected_test_count=12,
                  train_windows=sum(row['windows'] for row in rows if row['split']=='TRAIN'),
                  first48_native_prior_windows=sum(row['windows'] for row in original_rows if row['split']=='TRAIN'),
                  newly_admitted_train_episodes=152,new_episodes_bytes=sum(row['bytes'] for row in candidates),
                  source_manifest_sha256=sha(manifest),statistics_sha256=sha(stats),
                  TEST_labels_read=False,model_updates=0,new_native_requests=0,site_assets_bound=False)
    write(root/'ACTUAL.private.json',result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('sha256')
    args = parser.parse_args()
    print(prepare(read_pin(dict(path=args.config,sha256=args.sha256))))


if __name__ == '__main__':
    main()
