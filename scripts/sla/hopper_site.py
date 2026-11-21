"""Bind one H200 process to its real, finite PBS allocation and registered science budget."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import time

from scripts.sla.hopper_parent_custody import PLAN, digest

T0 = 1795236112
DEADLINE = T0 + 14*86400


def wall_seconds(value):
    h, m, s = map(int, value.split(':'))
    if min(h, m, s) < 0 or max(m, s) >= 60:
        raise ValueError('Invalid PBS walltime')
    return h*3600 + m*60 + s


def site_epoch(value):
    if isinstance(value, (int, float)):
        return value
    return datetime.datetime.strptime(value, '%a %b %d %H:%M:%S %Y').timestamp()


def pinned(pin):
    p = Path(pin['path'])
    if digest(p) != pin['sha256']:
        raise ValueError('Changed site or budget receipt')
    return json.loads(p.read_bytes())


def live_h200(mount, group):
    import jax

    job = os.environ.get('PBS_JOBID')
    if not job or job != mount['job'] or mount['plan'] != PLAN or mount['T0_epoch'] != T0:
        raise PermissionError('Expected the registered compute job and original campaign start')
    grant = pinned(mount['site_grant'])
    budget = pinned(mount['gpu_reservation'])
    now = time.time()
    if (grant['project'] != mount['project'] or not grant['valid_from_epoch'] <= now < grant['expiry_epoch']
            or budget['plan'] != PLAN or budget['job'] != job or budget['group'] != group
            or budget['reserved_gpu_count'] != 1 or not 0 < budget['reserved_seconds'] <= 21600
            or budget['all_prior_seconds'] < 0 or budget['engineering_prior_seconds'] < 0
            or budget['all_prior_seconds'] + budget['reserved_seconds'] > 320*3600
            or (group == 'engineering' and budget['engineering_prior_seconds'] + budget['reserved_seconds'] > 8*3600)):
        raise PermissionError('Project grant or shared reservation is not current')
    pbs = json.loads(subprocess.check_output([mount['qstat'], '-f', '-F', 'json', job], text=True, timeout=20))['Jobs'][job]
    selection = pbs['Resource_List']
    host = os.uname().nodename.split('.')[0]
    allocation_host = pbs['exec_host'].split('/')[0].split('.')[0]
    wall = wall_seconds(selection['walltime'])
    if (pbs['job_state'] != 'R' or pbs['project'] != grant['project'] or host != allocation_host
            or pbs['Job_Owner'].split('@')[0] != grant['account'] or os.getuid() != mount['uid']
            or int(selection.get('ngpus', 0)) != 1 or wall != budget['reserved_seconds']):
        raise PermissionError('This is not the bound PBS compute allocation')
    limit = min(DEADLINE, site_epoch(pbs['stime']) + wall, grant['expiry_epoch'], mount['deadline_epoch'])
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
    if ',' in visible or not visible or len(jax.local_devices()) != 1 or jax.default_backend() != 'gpu':
        raise PermissionError('Expected one scheduler-assigned visible GPU')
    device = jax.local_devices()[0]
    if 'H200' not in device.device_kind:
        raise PermissionError('Unexpected JAX compute device')
    info = subprocess.check_output(['nvidia-smi', '-i', visible, '--query-gpu=uuid,name',
                                   '--format=csv,noheader'], text=True, timeout=10).strip()
    uid, name = [v.strip() for v in info.split(',')]
    if 'H200' not in name or not uid.startswith('GPU-') or now >= limit:
        raise PermissionError('H200 UUID or current allocation deadline missing')
    return dict(job=job, host=host, gpu_uuid=uid, gpu_name=name, deadline=limit, wall_s=wall)
