"""Use explicit local bindings and the existing registry/Journal, no scheduler."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time


@contextmanager
def admitted_run(spec_path, run_id, profile, output, *, physics=False):
    from scripts.pi0292.durable import OwnerLock
    from .storage import digest, write_json
    spec = json.loads(Path(spec_path).read_text())
    root = Path(spec['registered_root']).resolve()
    output = Path(output).resolve()
    if (spec['profile'] != profile or spec['host'] != os.uname().nodename or spec['uid'] != os.getuid()
            or output.parent != root or output.name != run_id or time.time() >= spec['deadline_epoch']
            or (root/'STOP.json').exists()):
        raise PermissionError('Exact WM01 host/profile/run and finite endpoint required')
    if physics and spec.get('physics') is not True:
        raise PermissionError('Profile does not admit native source physics')
    for name, pin in spec['source_files'].items():
        if digest(Path(spec['source_root'])/name) != pin:
            raise PermissionError('Source identity changed:'+name)
    registry = json.loads(Path(spec['registry_snapshot']).read_text())
    if not any(e.get('role') == spec['role'] and e.get('path') == str(root)
               and e.get('source_sha') == spec['source_sha'] for e in registry['entries']):
        raise PermissionError('Existing host-aware registry binding missing')
    lease = json.loads(Path(spec['lease']).read_text())
    if lease['status'] != 'ACTIVE' or lease['run_uuid'] != root.name or lease['spec_sha256'] != digest(spec_path):
        raise PermissionError('Exact active WM01 lease required')
    cpus = set(spec['cpus'])
    if not cpus or not cpus <= os.sched_getaffinity(0):
        raise PermissionError('Actual CPU affinity not available')
    lock = OwnerLock(root/'OWNER.lock')
    try:
        os.sched_setaffinity(0, cpus)
        output.mkdir(exist_ok=False)
        write_json(output/'STARTED.json', dict(run_id=run_id, source_sha=spec['source_sha'],
            spec_sha256=digest(spec_path), pid=os.getpid(), epoch=time.time(), profile=profile))
        yield spec
    finally:
        lock.close()
