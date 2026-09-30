"""Remove redundant canary output-root model views when full step-8 states exist."""
import argparse
import json
import os
from pathlib import Path

ROOT = Path('/scratch/e1192548/Projects/go2-arx5/experiments/PI0261')
NAMES = ('model-00001-of-00002.safetensors',
         'model-00002-of-00002.safetensors', 'model.safetensors.index.json')


def main(arm):
    parent = ROOT / arm
    receipt = parent / 'REDUNDANT_CANARY_MODEL_VIEW_RETIREMENT.json'
    assert not receipt.exists()
    rows = []
    for phase in ('continuous', 'split'):
        training = parent / phase / 'training'
        checkpoint = training / 'checkpoint-8'
        if checkpoint.exists():
            state = json.loads((checkpoint / 'trainer_state.json').read_text())
            assert (state['global_step'], state['max_steps']) == (8, 2000)
            assert all((checkpoint / name).is_file() for name in
                       ('optimizer.pt', 'scheduler.pt', 'rng_state.pth', *NAMES))
        else:
            assert json.loads((parent / 'canary_compare.json').read_text())['status'] == (
                'PASS_PI0261_REAL_TWO_MICROBATCH_EXACT_RESUME_CANARY')
            assert json.loads((parent / 'CANARY_CHECKPOINT_RETIREMENT.json').read_text())['status'] == (
                'PASS_CANARY_COPIES_RETIRED_AFTER_EXACT_COMPARISON')
        for name in NAMES:
            path = training / name
            stat = path.stat()
            assert path.is_file() and not path.is_symlink() and stat.st_nlink == 1
            assert path.resolve().is_relative_to(parent.resolve())
            rows.append(dict(path=str(path), logical_bytes=stat.st_size))
    result = dict(status='RETIRED_REDUNDANT_CANARY_OUTPUT_ROOT_MODEL_VIEWS',
                  arm=arm, files=rows, total_logical_bytes=sum(x['logical_bytes'] for x in rows),
                  full_step8_states_preserved_or_compared_and_retired=True)
    temp = receipt.with_suffix('.json.partial')
    temp.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    os.replace(temp, receipt)
    for row in rows:
        Path(row['path']).unlink()
    assert all(not Path(row['path']).exists() for row in rows)
    print(json.dumps(dict(status=result['status'], arm=arm,
                          total_logical_bytes=result['total_logical_bytes'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=('C', 'P'), required=True)
    main(parser.parse_args().arm)
