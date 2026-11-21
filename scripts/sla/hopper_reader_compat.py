"""Verify the exact change in a cold training capsule that can read repaired TRAIN captures."""
from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path

SCIENCE = 'b9581196d0acd02fb748222988e5a5543ce1f232'
READER = 'b69a924e64e964d8bcc4c257403860344a8d19ee'
ORIGINAL_MANIFEST = '411702baacce2d3c3ba5b362c47246328d1e2e98b25148323e287db72140afdf'
READER_MANIFEST = '5f71601cee21571d7d66ec598f0e1a49a8973f88a4103b98c1fd7e324e75cc20'
ADDED = 'scripts/sla/hopper_snapshot_producers.py'
CHANGED = 'scripts/sla/hopper_snapshot_relay.py'


def sha(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError('Expected a regular capsule member')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(folder, expected, revision):
    path = folder/'SOURCE_BINDING.private.json'
    if sha(path) != expected:
        raise ValueError('Wrong registered source manifest')
    manifest = json.loads(path.read_bytes())
    if manifest['new_revision'] != revision:
        raise ValueError('Unexpected capsule source revision')
    rows = {n:v['sha256'] for n,v in manifest['modules'].items()}
    allowed = {*rows, 'SOURCE_BINDING.private.json', 'population.json', 'code.sha256'}
    if any(p.is_symlink() or (p.is_file() and p.relative_to(folder).as_posix() not in allowed)
           for p in folder.rglob('*')):
        raise ValueError('Unregistered file in the cold execution source')
    if any(sha(folder/n) != h for n,h in rows.items()):
        raise ValueError('Module content differs from the registered source manifest')
    return rows


def compare(original, candidate):
    old = inventory(original, ORIGINAL_MANIFEST, SCIENCE)
    new = inventory(candidate, READER_MANIFEST, READER)
    if (len(old)!=954 or len(new)!=955 or set(new)!=set(old)|{ADDED}
            or {k for k,v in old.items() if new[k]!=v} != {CHANGED}):
        raise ValueError('Reader extension changed another training module')
    return dict(qualified_training_revision=SCIENCE, actual_reader_runtime_revision=READER,
                original_manifest_sha256=ORIGINAL_MANIFEST, reader_manifest_sha256=READER_MANIFEST,
                original_modules=len(old), runtime_modules=len(new), identical_modules=len(old)-1,
                added={ADDED:new[ADDED]}, changed={CHANGED:dict(original=old[CHANGED],candidate=new[CHANGED])})


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('original', type=Path)
    p.add_argument('candidate', type=Path)
    a = p.parse_args()
    print(json.dumps(compare(a.original,a.candidate),sort_keys=True))
