"""Retire only verified PI0264 canary Trainer copies; keep all audit logs."""
import argparse
import json
import os
import shutil
from pathlib import Path

ROOT = Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0264')


def main(arm):
    parent = ROOT / arm
    assert json.loads((parent / 'canary_compare.json').read_text())['status'] == (
        'PASS_PI0264_REAL_TWO_MICROBATCH_EXACT_RESUME_CANARY')
    targets = [parent / phase / 'training' / f'checkpoint-{step}'
               for phase, step in (('continuous', 8), ('split', 4), ('split', 8))]
    inventory = parent / 'CANARY_CHECKPOINT_RETIREMENT.json'
    assert not inventory.exists()
    rows = []
    for target in targets:
        assert target.is_dir() and not target.is_symlink()
        assert target.resolve().is_relative_to(parent.resolve())
        files = []
        for path in target.rglob('*'):
            assert not path.is_symlink()
            if path.is_file():
                stat = path.stat()
                assert stat.st_nlink == 1
                files.append(dict(path=path.relative_to(target).as_posix(), bytes=stat.st_size))
        assert len(files) >= 16
        rows.append(dict(path=str(target), files=files,
                         logical_bytes=sum(file['bytes'] for file in files)))
    result = dict(status='PASS_CANARY_COPIES_RETIRED_AFTER_EXACT_COMPARISON',
                  arm=arm, targets=rows, total_logical_bytes=sum(row['logical_bytes'] for row in rows))
    temp = inventory.with_suffix('.json.partial')
    temp.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    os.replace(temp, inventory)
    for target in targets:
        shutil.rmtree(target)
    assert all(not target.exists() for target in targets)
    print(json.dumps(dict(status=result['status'], arm=arm,
                          total_logical_bytes=result['total_logical_bytes'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=('B', 'L'), required=True)
    main(parser.parse_args().arm)
