"""Use explicit local bindings and the existing registry/Journal, no scheduler."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time


def cuda_uuid(value):
    """PyTorch returns a bare UUID; NVML/nvidia-smi require the GPU prefix."""
    from uuid import UUID
    value = str(value)
    if value.startswith('GPU-'):
        value = value[4:]
    try:
        return 'GPU-'+str(UUID(value))
    except (ValueError, AttributeError) as error:
        raise ValueError('Invalid native CUDA device UUID') from error


@contextmanager
def admitted_run(spec_path, run_id, profile, output, *, physics=False, gpu=False):
    from scripts.pi0292.durable import OwnerLock
    from .storage import digest, write_json
    spec = json.loads(Path(spec_path).read_text())
    if not Path(__file__).resolve().is_relative_to(Path(spec['source_root']).resolve()):
        raise PermissionError('Interpreter imported a different WM01 source capsule')
    root = Path(spec['registered_root']).resolve()
    output = Path(output).resolve()
    if (spec['profile'] != profile or spec['host'] != os.uname().nodename or spec['uid'] != os.getuid()
            or output.parent != root or output.name != run_id or time.time() >= spec['deadline_epoch']
            or (root/'STOP.json').exists()):
        raise PermissionError('Exact WM01 host/profile/run and finite endpoint required')
    if physics and spec.get('physics') is not True:
        raise PermissionError('Profile does not admit native source physics')
    if gpu:
        if spec.get('execution_site') != 'VANDA_A40':
            raise PermissionError('W1 requires the explicitly bound Vanda single-GPU allocation')
        import subprocess
        from scripts.pi0294.multitask.shared_site import validate_vanda, project_quota
        key = os.environ.get('PBS_JOBID')
        if not key:
            raise PermissionError('Actual PBS allocation required before model import')
        response = subprocess.run(['/opt/pbs/bin/qstat', '-f', '-F', 'json', key],
                                  capture_output=True, text=True, check=True, timeout=20)
        jobs = json.loads(response.stdout)['Jobs']
        matches = [v for k,v in jobs.items() if k.split('.')[0] == key.split('.')[0]]
        if len(matches) != 1:
            raise PermissionError('Exact single PBS allocation required')
        context = dict(spec['site_context'])
        quota_path = Path('/home/svu/vanda/quotas/CFP04-CF-007.proj_scratch.txt')
        now = time.time()
        context['quota_receipt'] = project_quota(quota_path.read_text(), user=os.environ['USER'],
            report_epoch=quota_path.stat().st_mtime, observed_epoch=now)
        validate_vanda(context, matches[0], now=now, uid=os.getuid(), user=os.environ['USER'],
                       hostname=os.uname().nodename, job_id=key)
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
