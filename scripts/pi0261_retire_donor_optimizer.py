"""Retire only the fixed-HF donor optimizer, unused by fresh PI0261 AdamW."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0261')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    original = json.loads((ROOT / 'Base_checkpoint-6000.full_readback.json').read_text())
    local = json.loads((ROOT / 'DONOR_LOCAL_READBACK.json').read_text())
    assert original['revision'] == local['revision'] == '52abc8d151b67bb667e43fcf874dfe2cfa253364'
    item = next(x for x in original['files'] if x['path'] == 'optimizer.pt')
    target = ROOT / 'donor' / original['prefix'] / item['path']
    assert target.resolve().is_relative_to(ROOT.resolve()) and not target.is_symlink()
    assert target.stat().st_nlink == 1 and target.stat().st_size == item['bytes']
    assert sha(target) == item['sha256']
    target.unlink()
    result = dict(status='DONOR_OLD_OPTIMIZER_LOCAL_COPY_RETIRED', path=str(target), bytes=item['bytes'],
                  sha256=item['sha256'], hf_repo=original['repo_id'], hf_revision=original['revision'],
                  hf_prefix=original['prefix'], reason='PI0261 uses fresh AdamW; fixed full Trainer HF readback passed')
    receipt = ROOT / 'DONOR_OPTIMIZER_LOCAL_RETIREMENT.json'
    with receipt.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
