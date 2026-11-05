"""Finite original-owner worker wrapper; close only this job's actual lease."""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    args = parser.parse_args(argv)
    from go2arx5_vla.world_model.storage import digest
    from scripts.pi0292.durable import write
    spec = json.loads(args.spec.read_text())
    root = Path(spec['registered_root'])
    lease_path = Path(spec['lease'])
    lease = json.loads(lease_path.read_text())
    if lease.get('status') != 'ACTIVE' or lease.get('spec_sha256') != digest(args.spec) or lease.get('run_uuid') != root.name:
        raise PermissionError('EXACT_FRESH_PILOT1_WORKER_REQUIRED')
    if (root / 'SUPERVISOR.private.json').exists():
        raise FileExistsError('DO_NOT_REDISPATCH_EXISTING_OR_UNKNOWN_WORKER')
    write(root / 'SUPERVISOR.private.json', dict(pid=os.getpid(), source_sha=spec['source_sha'], started_epoch=time.time()))
    command = [sys.executable, '-B', '-m', 'scripts.wm02.train_pilot1_actor', '--spec', str(args.spec), '--output', str(root / 'fit-01')]
    started = time.monotonic()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    failure, exit_code = None, 1
    try:
        with (root / 'TRAIN_STDOUT.private.log').open('wb') as stdout, (root / 'TRAIN_STDERR.private.log').open('wb') as stderr:
            result = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=min(spec['wall_limit_s'], spec['deadline_epoch']-time.time()))
        exit_code = result.returncode
    except Exception as error:
        failure = type(error).__name__ + ': ' + str(error)
    finally:
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        write(root / 'WORKER_EXIT.private.json', dict(exit_code=exit_code, argv=command, failure=failure,
              wall_s=time.monotonic()-started, child_CPU_s=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime))
        lease = json.loads(lease_path.read_text())
        if lease.get('status') != 'ACTIVE' or lease.get('spec_sha256') != digest(args.spec):
            raise PermissionError('WORKER_LEASE_CHANGED_DURING_EXECUTION')
        lease.update(status='CLOSED', closed_epoch=time.time(), worker_exit_code=exit_code)
        write(lease_path, lease)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
