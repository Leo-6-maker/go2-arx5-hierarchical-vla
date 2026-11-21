"""Exercise the receiver's relative storage layout and fixed observations in a CPU batch."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

from scripts.sla.hopper_panels import FixedPanels
from scripts.sla.hopper_parent_custody import PLAN, digest, write_once
from scripts.sla.hopper_source_restore_probe import cpu_allocation


def mirror(base, dest, inventory, expected):
    if digest(inventory) != expected:
        raise ValueError('Wrong cross-site input inventory')
    rows = json.loads(inventory.read_bytes())['entries']
    dest.mkdir()
    for name, row in rows.items():
        path = (base/name).resolve(strict=True)
        if (Path(name).is_absolute() or '..' in Path(name).parts or not path.is_relative_to(base)
                or path.stat().st_size != row['bytes']):
            raise ValueError('Cross-site source member differs from its inventory')
        member = dest/name
        member.parent.mkdir(parents=True, exist_ok=True)
        os.link(path, member)
    return len(rows)


def run(root):
    from scripts.pi0292.durable import OwnerLock

    cfg = json.loads((root/'MOUNT.private.json').read_bytes())
    lock = OwnerLock(root/'OWNER.lock')
    try:
        job, _ = cpu_allocation(root, cfg)
        began = time.monotonic()
        origin = Path(cfg['base']).resolve(strict=True)
        counts = {}
        with tempfile.TemporaryDirectory(dir=root, prefix='private-test-view-') as tmp:
            base = Path(tmp)
            for key in ('train', 'eval'):
                inv = cfg[key]
                counts[key] = mirror(origin, base/key, Path(inv['path']), inv['sha256'])
            panels = FixedPanels(base/'train', base/'eval')
            panels_read = {}
            for mode in ('engineering', 'formal', 'formal-restore'):
                visited = {}
                hashes = hashlib.sha256()
                for kind, _, ordinal, row, _, noise in panels.examples(mode):
                    if time.time() >= cfg['deadline_epoch'] or (root/'STOP').exists():
                        raise TimeoutError('Panel read stopped')
                    visited[kind] = visited.get(kind, 0) + 1
                    hashes.update(json.dumps([kind, ordinal, row['identity']], sort_keys=True).encode())
                    hashes.update(noise.tobytes())
                panels_read[mode] = dict(rows=visited, identity_and_noise_sha256=hashes.hexdigest())
            expected = {'engineering': 48, 'formal': 376, 'formal-restore': 48}
            if {mode: sum(v['rows'].values()) for mode, v in panels_read.items()} != expected:
                raise ValueError('Unexpected fixed-panel input counts')
        write_once(root/'RESULT.private.json', dict(plan=PLAN, job=job, source=cfg['source_revision'],
                   status='SOURCE_CPU_RECEIVER_LAYOUT_INPUTS_VERIFIED', inputs=panels_read,
                   mirrored_members=counts, wall_s=time.monotonic()-began, model_loads=0,
                   optimizer_updates=0, optimization_visits=0, model_generations=0, native=0))
    except Exception as exc:
        write_once(root/'FAILURE.private.json', dict(error=str(exc), job=os.environ.get('PBS_JOBID')))
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    run(Path.cwd())
