"""Extract only fixed original PI0261 frames into the isolated working set."""
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

ROOT = Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0261')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    archive = ROOT / 'SOURCE_FRAMES.tar'
    manifest = json.loads((ROOT / 'SOURCE_FRAME_MANIFEST.json').read_text())
    assert manifest['status'] == 'PASS_SOURCE_BOUND_FRAME_INVENTORY'
    assert sha(archive) == '4a3b5c294bdd9efa689333f763582bef5056d1e590a6823e30079f8aaae58b08'
    expected = {row['path']: row for row in manifest['members']}
    assert len(expected) == manifest['files'] == 2565
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        assert {member.name for member in members} == set(expected)
        for member in members:
            assert member.isfile() and member.size == expected[member.name]['bytes']
            target = ROOT / 'source' / member.name
            assert target.resolve().is_relative_to((ROOT / 'source').resolve())
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as input_stream, target.open('xb') as output_stream:
                    shutil.copyfileobj(input_stream, output_stream)
            assert target.stat().st_size == member.size and sha(target) == expected[member.name]['sha256']
    result = dict(status='PASS_PI0261_EXACT_ORIGINAL_FRAMES_STAGED', files=len(members),
                  source_archive_sha256=sha(archive), source_manifest_sha256=sha(ROOT / 'SOURCE_FRAME_MANIFEST.json'),
                  total_bytes=manifest['total_bytes'], root=str(ROOT / 'source'))
    target = ROOT / 'SOURCE_FRAME_STAGE_RECEIPT.json'
    rendered = json.dumps(result, indent=2, sort_keys=True) + '\n'
    if target.exists():
        assert target.read_text() == rendered
    else:
        target.write_text(rendered)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
