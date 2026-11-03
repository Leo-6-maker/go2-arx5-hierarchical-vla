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


def validate_core_smoke(spec, observation):
    """The directly authorized GPU1 sharing is a finite model-only diagnostic."""
    share = spec.get('sharing', {})
    if (spec.get('host') != 'corelab-4080s' or spec.get('uid') != 2009
            or spec.get('profile') != 'core-wm01-shared-smoke'
            or spec.get('physics') is not False
            or share.get('direct_user_answer') != '允许GPU1共享'
            or spec.get('GPU_uuid') != 'GPU-845348f1-1511-dc19-0877-cc97658e3edd'
            or share.get('endpoint') != 200
            or share.get('allocator_limit_bytes') != 512*1024**2
            or observation.get('uuid') != spec['GPU_uuid']
            or observation.get('name') != 'NVIDIA GeForce RTX 4080 SUPER'
            or observation.get('total_bytes', 0) < 15*1024**3
            or observation.get('free_bytes', 0) < 2*1024**3):
        raise PermissionError('Exact authorized Core GPU1 model-only smoke and live headroom required')


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
    if gpu and spec.get('execution_site') == 'CORE_SHARED_GPU1':
        import subprocess
        from scripts.pi0294.e1.smoke import node_local
        for name in ('source_root', 'registered_root', 'python', 'data', 'cache', 'logs', 'tmp'):
            node_local(spec[name])
        response = subprocess.run(['nvidia-smi', '--id='+spec['GPU_uuid'],
            '--query-gpu=uuid,name,memory.total,memory.free', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True, timeout=10)
        fields = [value.strip() for value in response.stdout.strip().split(',')]
        validate_core_smoke(spec, dict(uuid=fields[0], name=fields[1],
            total_bytes=int(fields[2])*1024**2, free_bytes=int(fields[3])*1024**2))
        if os.environ.get('CUDA_VISIBLE_DEVICES') != spec['GPU_uuid']:
            raise PermissionError('Core smoke must expose only the authorized GPU1 UUID')
    elif gpu:
        if spec.get('execution_site') != 'VANDA_A40':
            raise PermissionError('Explicit Vanda allocation or authorized Core smoke required')
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
