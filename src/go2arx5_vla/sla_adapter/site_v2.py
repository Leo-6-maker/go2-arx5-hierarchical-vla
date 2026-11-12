"""Bind a finite single-GPU operation to an existing actual allocation.

No scheduler submission, owner registration, allocation renewal or GPU process
control is implemented here. The existing executor supplies the pinned site.
"""
import json
import os
from pathlib import Path
import subprocess
import time

from .runtime_v1 import read_pin

NETWORK_FILESYSTEMS = {'nfs', 'nfs4', 'cifs', 'lustre', 'gpfs', 'beegfs', 'ceph', 'fuse.sshfs'}


def contract(binding, spec, actual, *, now):
    if (binding.get('schema') != 'SLA_SINGLE_GPU_EXISTING_SITE_V2'
            or binding['host'] != actual['host'] or binding['uid'] != actual['uid']
            or binding['GPU_uuid'] != spec['GPU_uuid'] or actual['GPU_uuid'] != binding['GPU_uuid']
            or actual['GPU_name'] != binding['GPU_name']
            or actual['CUDA_VISIBLE_DEVICES'] != binding['GPU_uuid']):
        raise PermissionError('EXACT_EXISTING_SINGLE_GPU_SITE_REQUIRED')
    wall = spec['max_walltime_s']
    if (type(wall) is not int or wall <= 0 or not now < spec['deadline_epoch'] <= now + wall
            or spec['deadline_epoch'] + 60 > binding['allocation_deadline_epoch']
            or spec['max_gpu_hours'] <= 0 or wall > spec['max_gpu_hours'] * 3600
            or not spec['save_and_exit_on_limit']
            or not 0 < spec['model_service_timeout_s'] <= wall
            or not 0 < spec['RPC_timeout_s'] <= wall):
        raise PermissionError('FINITE_ALLOCATED_WINDOW_AND_SAVE_EXIT_REQUIRED')
    if (binding['GPU_headroom_bytes'] < 1024**3
            or actual['GPU_free_bytes'] <= binding['allocator_limit_bytes'] + binding['GPU_headroom_bytes']
            or actual['host_available_bytes'] <= binding['host_headroom_bytes']
            or binding['allocator_limit_bytes'] < spec['JAX_memory_fraction'] * actual['GPU_total_bytes']):
        raise PermissionError('ACTUAL_SINGLE_GPU_AND_HOST_HEADROOM_REQUIRED')
    if actual['cache_filesystem'] in NETWORK_FILESYSTEMS:
        raise PermissionError('ACTUAL_NODE_LOCAL_COMPILATION_CACHE_REQUIRED')
    for key in ('parent_params', 'dataset', 'compilation_cache_dir'):
        if actual['paths'][key] != binding['paths'][key] or actual['paths'][key] != str(Path(spec[key]).resolve()):
            raise PermissionError('ACTUAL_PINNED_SITE_PATHS_REQUIRED')
    allocation = binding['allocation']
    if allocation['kind'] == 'PBS':
        if (actual['PBS_JOBID'] != allocation['job_id'] or not actual['PBS_JOBID']
                or actual['allocation_project'] != allocation['project']
                or not actual['allocation_running']):
            raise PermissionError('OWN_EXISTING_RUNNING_PBS_ALLOCATION_REQUIRED')
    elif allocation['kind'] == 'EXISTING_USER_SERVICE':
        if (not actual['campaign_active'] or actual['campaign_STOP']
                or actual['campaign_owner'] != 'EXISTING_LOCAL_PC'
                or actual['campaign_deadline_epoch'] < spec['deadline_epoch']
                or spec.get('neural_owner_lock') != allocation['neural_owner_lock']):
            raise PermissionError('ORIGINAL_ACTIVE_USER_SERVICE_OWNER_REQUIRED')
    else:
        raise PermissionError('NO_INVENTED_ALLOCATION_KIND')


def validate(spec):
    binding = read_pin(spec['site_binding'])
    row = subprocess.check_output(['nvidia-smi', '--id=' + binding['GPU_uuid'],
                                   '--query-gpu=uuid,name,memory.total,memory.free',
                                   '--format=csv,noheader,nounits'], text=True, timeout=15)
    uuid, name, total, free = [value.strip() for value in row.strip().split(',')]
    available = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:'))
    paths = {key: str(Path(spec[key]).resolve(strict=True))
             for key in ('parent_params', 'dataset', 'compilation_cache_dir')}
    mount = subprocess.check_output(['findmnt', '-J', '-T', paths['compilation_cache_dir'],
                                     '-o', 'SOURCE,FSTYPE,TARGET'], text=True, timeout=15)
    filesystem = json.loads(mount)['filesystems'][0]
    actual = dict(host=os.uname().nodename, uid=os.getuid(), GPU_uuid=uuid, GPU_name=name,
                  GPU_total_bytes=int(total) * 1024**2, GPU_free_bytes=int(free) * 1024**2,
                  host_available_bytes=available, paths=paths, cache_filesystem=filesystem['fstype'],
                  cache_mount=filesystem, CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  PBS_JOBID=os.environ.get('PBS_JOBID'), epoch=time.time())
    allocation = binding['allocation']
    if allocation['kind'] == 'PBS':
        evidence = read_pin(allocation['actual_allocation'])
        actual.update(allocation_project=evidence['project'],
                      allocation_running=evidence['state'] == 'R' and evidence['job_id'] == actual['PBS_JOBID']
                      and evidence['uid'] == os.getuid() and evidence['host'] == os.uname().nodename
                      and binding['GPU_uuid'] in evidence['allocated_GPU_uuids'])
    elif allocation['kind'] == 'EXISTING_USER_SERVICE':
        campaign = Path(allocation['campaign_root'])
        lease = json.loads((campaign / 'control/LEASE.json').read_bytes())
        actual.update(campaign_active=lease['status'] == 'ACTIVE', campaign_owner=lease['owner'],
                      campaign_STOP=(campaign / 'STOP.json').exists(), campaign_deadline_epoch=lease['deadline_epoch'])
    contract(binding, spec, actual, now=time.time())
    return actual
