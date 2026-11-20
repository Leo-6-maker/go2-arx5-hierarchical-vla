"""Verify a running same-node PBS A40 pair before initializing JAX work."""
import ctypes
import json
import os
from pathlib import Path
import re
import subprocess
import time
import uuid

from .runtime_v1 import read_pin
from .site_v2 import NETWORK_FILESYSTEMS


GIB = 1024**3


def visible_cuda_uuids():
    driver = ctypes.CDLL('libcuda.so.1')
    count = ctypes.c_int()
    if driver.cuInit(0) or driver.cuDeviceGetCount(ctypes.byref(count)) or count.value!=2:
        raise PermissionError('ADAPT_PAIR_EXACTLY_TWO_SCHEDULER_VISIBLE_CUDA_DEVICES_REQUIRED')
    result = []
    for rank in range(count.value):
        device = ctypes.c_int()
        value = (ctypes.c_ubyte*16)()
        if driver.cuDeviceGet(ctypes.byref(device),rank) or driver.cuDeviceGetUuid(ctypes.byref(value),device):
            raise PermissionError('ADAPT_PAIR_CANNOT_IDENTIFY_ALLOCATED_CUDA_DEVICE')
        result.append('GPU-'+str(uuid.UUID(bytes=bytes(value))))
    return result


def validate(spec):
    from scripts.wm02.multisource_d4_vanda import allocation_window

    binding = read_pin(spec['pair_site_binding'])
    job_id = os.environ.get('PBS_JOBID')
    now = time.time()
    if (binding['job_id']!=job_id or binding['uid']!=os.getuid() or binding['host']!=os.uname().nodename
            or binding['schema']!='SLA_ADAPT_TWO_LOCAL_A40_PBS_V1' or not job_id
            or binding['deadline_epoch']!=spec['deadline_epoch'] or not now<spec['deadline_epoch']<=now+10800
            or binding['parent_checkpoint_sha256']!=spec['expected_pair_parent_sha256']):
        raise PermissionError('ADAPT_PAIR_OWN_FINITE_PBS_BINDING_REQUIRED')

    def run(*args):
        return subprocess.check_output(args,text=True,stderr=subprocess.PIPE,timeout=20)

    job = json.loads(run('/opt/pbs/bin/qstat','-f','-F','json',job_id))['Jobs'][job_id]
    hosts = {item.split('/')[0].split('.')[0].lower() for item in job['exec_host'].split('+')}
    gpu_grants = sum(map(int,re.findall(r':ngpus=(\d+)',job['exec_vnode'])))
    window = allocation_window(job,campaign_deadline=binding['deadline_epoch'])
    if (job['job_state']!='R' or job['project']!=binding['project'] or job['Job_Owner'].split('@')[0]!=binding['user']
            or hosts!={binding['host'].split('.')[0].lower()} or int(job['Resource_List']['ngpus'])!=2
            or gpu_grants!=2 or window['PBS_wall_s']>10800 or spec['deadline_epoch']>window['deadline_epoch']):
        raise PermissionError('ADAPT_PAIR_RUNNING_SAME_NODE_TWO_GPU_PBS_GRANT_REQUIRED')
    visible = visible_cuda_uuids()
    if visible!=binding['GPU_uuids'] or len(set(visible))!=2 or binding['GPU_uuids'][0]!=binding['reference_GPU_uuid']:
        raise PermissionError('ADAPT_PAIR_SAME_REFERENCE_PRIMARY_GPU_REQUIRED')
    hardware = []
    for ident in visible:
        row = run('nvidia-smi','--id='+ident,'--query-gpu=name,memory.total,memory.free,pci.bus_id',
                  '--format=csv,noheader,nounits')
        name,total,free,pci = [value.strip() for value in row.split(',')]
        total_bytes,free_bytes = int(total)*1024**2,int(free)*1024**2
        if name!='NVIDIA A40' or total_bytes<40*GIB or free_bytes<=.8*total_bytes+2*GIB:
            raise PermissionError('ADAPT_PAIR_REQUIRES_REAL_A40_HEADROOM_ON_BOTH_GPUS')
        hardware.append(dict(UUID=ident,PCI_bus_id=pci,GPU_bytes=total_bytes,free_bytes=free_bytes))
    if len({row['PCI_bus_id'] for row in hardware})!=2:
        raise PermissionError('ADAPT_PAIR_TWO_DISTINCT_PCI_DEVICES_REQUIRED')
    cache = Path(spec['compilation_cache_dir']).resolve(strict=True)
    mount = json.loads(run('findmnt','-J','-T',str(cache),'-o','FSTYPE'))['filesystems'][0]['fstype']
    mem = next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
               if line.startswith('MemAvailable:'))
    if mount in NETWORK_FILESYSTEMS or mem<64*GIB:
        raise PermissionError('ADAPT_PAIR_LOCAL_CACHE_AND_HOST_HEADROOM_REQUIRED')
    topology = run('nvidia-smi','topo','-m')
    return dict(GPU_uuid=visible[0],GPU_name='NVIDIA A40',host=binding['host'],GPU_uuids=visible,
                topology=topology,devices=hardware,allocation=window)
