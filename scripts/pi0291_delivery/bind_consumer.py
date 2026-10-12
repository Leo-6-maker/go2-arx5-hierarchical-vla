"""Explicit hash-checked consumer path rebinding; never refit statistics."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def bind(native, raw, output, selected=None):
    native, raw, output = Path(native), Path(raw), Path(output)
    original = read(native/'DATASET_MANIFEST.json')
    assert original['domain'] == 'FIXED_BASE_ARX5_TASK_SIM'
    assert sha(native/'statistics.json') == original['statistics_sha256']
    assert not output.exists(), 'Preserve prior consumer binding'
    records = [row.copy() for row in original['episodes'] if selected is None or row['id'] in selected]
    assert records and (selected is None or {row['id'] for row in records} == set(selected))
    output.mkdir(parents=True)
    (output/'episodes').mkdir()
    mappings = []
    for row in records:
        assert re.fullmatch(r'[TD]291[0-9]{3}',row['id'])
        folder = raw/row['id']
        assert folder.resolve().is_relative_to(raw.resolve())
        assert sha(folder/'samples.json') == row['samples_sha256']
        assert sha(folder/'physics.npz') == row['physics_sha256']
        assert sha(folder/'FILE_MANIFEST.json') == row['raw_FILE_MANIFEST_sha256']
        for item in read(folder/'FILE_MANIFEST.json'):
            path = folder/item['path']
            assert path.resolve().is_relative_to(folder.resolve()) and not path.is_symlink()
            assert sha(path) == item['sha256']
        assert row['file'] == 'episodes/'+row['id']+'.npz'
        assert sha(native/row['file']) == row['sha256']
        shutil.copy2(native/row['file'],output/row['file'])
        assert sha(output/row['file']) == row['sha256']
        mappings.append(dict(id=row['id'],producer_raw_root=row['raw_root'],consumer_raw_root=str(folder.resolve()),
                             samples_sha256=row['samples_sha256'],physics_sha256=row['physics_sha256']))
        row['raw_root'] = str(folder.resolve())
    for name in ('statistics.json','STATISTICS_PROVENANCE.json','modality.json','task_map.json'):
        shutil.copy2(native/name,output/name)
        assert sha(output/name) == sha(native/name)
    assert sha(output/'statistics.json') == original['statistics_sha256']
    rebound = dict(original,episodes=records,
        TRAIN_windows=sum(row['windows'] for row in records if row['split']=='TRAIN'),
        DEV_windows=sum(row['windows'] for row in records if row['split']=='DEV'),
        producer_manifest_sha256=sha(native/'DATASET_MANIFEST.json'),
        consumer_rebinding='Only raw_root paths/selected subset; source hashes, absolute commands and frozen TRAIN statistics unchanged')
    (output/'DATASET_MANIFEST.json').write_text(json.dumps(rebound,indent=2)+'\n')
    (output/'SPLIT.json').write_text(json.dumps({split:[row['id'] for row in records if row['split']==split] for split in ('TRAIN','DEV')},indent=2)+'\n')
    receipt = dict(status='PASS_HASH_CHECKED_CONSUMER_REBINDING',producer_manifest_sha256=sha(native/'DATASET_MANIFEST.json'),
        consumer_manifest_sha256=sha(output/'DATASET_MANIFEST.json'),statistics_sha256=sha(output/'statistics.json'),
        statistics_refit=False,mappings=mappings,neural_forward=0,physics_steps=0,
        loader_readback='NOT_RUN_BY_BINDING; caller must exercise fixed DatasetLoader on this new directory')
    (output/'CONSUMER_BINDING_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(status=receipt['status'],episodes=len(records),statistics_refit=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--native-root',required=True)
    parser.add_argument('--raw-root',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--episode-ids',nargs='+')
    args = parser.parse_args()
    bind(args.native_root,args.raw_root,args.output,args.episode_ids)
