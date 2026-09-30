"""Package only the original source frames named by the frozen PI0261 plan."""
import hashlib
import json
import tarfile
from pathlib import Path

SOURCE = Path('/var/cache/users/liuyu-nfs/Projects/go2-arx5/PI0260/R2')
ROOT = Path('/home/liuyu-nfs/Projects/go2-arx5/PI0261')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    plan = json.loads((ROOT / 'PAIRING_PLAN.json').read_text())
    refs = {}
    for pair in plan['common_start_bank']:
        for row in pair['commands']:
            refs[row['frame']] = row['frame_sha256']
    for unit in plan['units']:
        for row in unit['ordinary']:
            refs[row['frame']] = row['frame_sha256']
    members = []
    for name, digest in sorted(refs.items()):
        path = SOURCE / name
        assert path.resolve().is_relative_to(SOURCE.resolve())
        assert path.is_file() and sha(path) == digest
        members.append(dict(path=name, bytes=path.stat().st_size, sha256=digest))
    manifest = ROOT / 'SOURCE_FRAME_MANIFEST.json'
    rendered = json.dumps(dict(schema='PI0261_EXACT_FROZEN_SOURCE_FRAMES_V1',
                               status='PASS_SOURCE_BOUND_FRAME_INVENTORY', files=len(members),
                               total_bytes=sum(x['bytes'] for x in members), members=members),
                          indent=2, sort_keys=True) + '\n'
    if manifest.exists():
        assert manifest.read_text() == rendered
    else:
        manifest.write_text(rendered)
    archive = ROOT / 'SOURCE_FRAMES.tar'
    if not archive.exists():
        with tarfile.open(archive, 'w') as tar:
            for row in members:
                tar.add(SOURCE / row['path'], arcname=row['path'], recursive=False)
    print(json.dumps(dict(status='PASS_FROZEN_FRAME_PACKAGE', files=len(members),
                          total_bytes=sum(x['bytes'] for x in members), archive_bytes=archive.stat().st_size,
                          archive_sha256=sha(archive), manifest_sha256=sha(manifest))))


if __name__ == '__main__':
    main()
