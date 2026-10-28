"""Materialize and verify the fixed full Base6000 Trainer donor."""
import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0261')
RECEIPT = ROOT / 'Base_checkpoint-6000.full_readback.json'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    receipt = json.loads(RECEIPT.read_text())
    assert (receipt['status'], receipt['complete_trainer'], receipt['source_tree_sha256']) == (
        'PASS_FULL_FILE_READBACK', True,
        '24ed9d1b5a0e42e0aff6d62c54d8438d2f202d765b006d90c62d8623604e4ce5')
    assert len(receipt['files']) == 16
    for item in receipt['files']:
        name = receipt['prefix'] + '/' + item['path']
        target = ROOT / 'donor' / name
        if not target.exists() or target.stat().st_size != item['bytes'] or sha(target) != item['sha256']:
            target = Path(hf_hub_download(repo_id=receipt['repo_id'], filename=name,
                                          revision=receipt['revision'], local_dir=ROOT / 'donor',
                                          local_dir_use_symlinks=False))
        assert target.stat().st_size == item['bytes'] and sha(target) == item['sha256'], name
        print(json.dumps(dict(status='DONOR_FILE_VERIFIED', path=item['path'], bytes=item['bytes'])), flush=True)
    result = dict(status='PI0261_FIXED_DONOR_FULL_READBACK_PASS', repo_id=receipt['repo_id'],
                  revision=receipt['revision'], prefix=receipt['prefix'],
                  source_tree_sha256=receipt['source_tree_sha256'], files=16,
                  donor_path=str(ROOT / 'donor' / receipt['prefix']))
    target = ROOT / 'DONOR_LOCAL_READBACK.json'
    rendered = json.dumps(result, indent=2, sort_keys=True) + '\n'
    if target.exists():
        assert target.read_text() == rendered
    else:
        target.write_text(rendered)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
