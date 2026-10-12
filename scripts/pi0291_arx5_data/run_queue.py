"""Serial first-attempt collection; existing evidence is never retried."""
import argparse
import json
from pathlib import Path
import time

from .config import PILOTS, TRAIN, DEV
from .run_episode import run
from .native_dataset import validate_samples
from .resource_guard import PHYSICS_CUTOFF
from go2arx5_vla.task_data.io import write, file_hash


def main(root, source, phase, start, count):
    root = Path(root)
    bank = dict(pilot=PILOTS, train=TRAIN, dev=DEV)[phase]
    queue = bank[start:start + count]
    assert len(queue) == count and count > 0
    if phase != 'pilot':
        gate = json.loads((root / 'PILOT_GATE.json').read_text())
        assert gate['status'] == 'PASS' and gate['source_sha'] == source
        assert gate['all_eight_independent_raw_audits_pass']
    receipt = root / 'evidence' / f'QUEUE_{phase}_{start:03d}_{count:03d}.json'
    if receipt.exists():
        raise RuntimeError('Existing queue receipt: choose only unattempted IDs')
    rows = []
    began = time.time()
    for identifier in queue:
        if time.time() >= PHYSICS_CUTOFF:
            break
        if (root / 'episodes' / identifier).exists():
            raise RuntimeError('Existing episode: no quality retry')
        tick = time.time()
        result = run(root, identifier, source, 'normal', 'mesa_cpu')
        data_error = None
        try:
            assert result['status'] == 'VALID_COMPLETE'
            validate_samples(json.loads((root / 'episodes' / identifier / 'samples.json').read_text()),
                             root / 'episodes' / identifier)
        except Exception as exc:
            data_error = f'{type(exc).__name__}: {exc}'
        row = dict(id=identifier, source_sha=source, physics_status=result['status'],
            physics_success=result.get('score', {}).get('persistent_success', False),
            structure_valid=data_error is None, structure_error=data_error,
            paired_samples=result['paired_samples'], physics_steps=result['physics_steps'],
            physics_sha256=file_hash(root / 'episodes' / identifier / 'physics.npz')
                if (root / 'episodes' / identifier / 'physics.npz').exists() else None,
            wall_s=time.time() - tick)
        rows.append(row)
        value = dict(phase=phase, source_sha=source, requested=list(queue), attempted=rows,
            unattempted=list(queue[len(rows):]), began_epoch=began, updated_epoch=time.time(),
            physics_cutoff_epoch=PHYSICS_CUTOFF, no_retries=True,
            independent_raw_admission='PENDING_SEPARATE_AUDIT')
        write(receipt, value)
        print(json.dumps(dict(queue_progress=row, remaining=len(queue)-len(rows))), flush=True)
    if not rows:
        write(receipt, dict(phase=phase, requested=list(queue), attempted=[],
                           unattempted=list(queue), status='PHYSICS_CUTOFF'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--source-sha', required=True)
    parser.add_argument('--phase', choices=['pilot', 'train', 'dev'], required=True)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--count', type=int, required=True)
    args = parser.parse_args()
    main(args.root, args.source_sha, args.phase, args.start, args.count)
