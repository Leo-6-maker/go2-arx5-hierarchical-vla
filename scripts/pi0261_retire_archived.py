"""Retire one complete local Trainer only after fixed HF full-file readback."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

ROOT = Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0261')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main(arm, step):
    checkpoint = ROOT / arm / 'main/training' / f'checkpoint-{step}'
    assert checkpoint.is_dir() and not checkpoint.is_symlink()
    assert checkpoint.resolve().is_relative_to(ROOT.resolve())
    evidence = ROOT / 'evidence' / arm
    uploaded = json.loads((evidence / f'checkpoint-{step}.archive_receipt.json').read_text())
    readback = json.loads((evidence / f'checkpoint-{step}.full_readback.json').read_text())
    assert (readback['status'], readback['arm'], readback['step'], readback['complete_trainer']) == (
        'PASS_FULL_FILE_READBACK', arm, step, True)
    assert (uploaded['revision'], uploaded['tree_sha256'], uploaded['entries']) == (
        readback['revision'], readback['tree_sha256'], readback['files'])
    entries = {row['path']: row for row in readback['files']}
    assert len(entries) >= 16 and {'optimizer.pt', 'scheduler.pt', 'rng_state.pth',
                                    'trainer_state.json'} <= set(entries)
    assert all(not path.is_symlink() for path in checkpoint.rglob('*'))
    actual = {path.relative_to(checkpoint).as_posix(): path for path in checkpoint.rglob('*')
              if path.is_file()}
    assert set(actual) == set(entries)
    for name, path in actual.items():
        stat = path.stat()
        assert not path.is_symlink() and stat.st_nlink == 1
        assert stat.st_size == entries[name]['bytes'] and sha(path) == entries[name]['sha256']
    receipt = evidence / f'checkpoint-{step}.local_retirement.json'
    assert not receipt.exists()
    result = dict(status='RETIRED_FIXED_HF_READBACK_TRAINER_LOCAL_COPY', arm=arm, step=step,
                  local_path=str(checkpoint), repo_id=readback['repo_id'],
                  prefix=readback['prefix'], revision=readback['revision'],
                  tree_sha256=readback['tree_sha256'], files=len(entries),
                  logical_bytes=sum(row['bytes'] for row in entries.values()))
    temp = receipt.with_suffix('.json.partial')
    temp.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    os.replace(temp, receipt)
    shutil.rmtree(checkpoint)
    assert not checkpoint.exists()
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=('C', 'P'), required=True)
    parser.add_argument('--step', type=int, choices=(1000, 2000), required=True)
    args = parser.parse_args()
    main(args.arm, args.step)
