"""Bounded native readback of the already-bound Core neural process.

The resident/RPC actor observes the same kernel PID; it does not impersonate
that PID, refresh GLOBAL, acquire a GPU or authorize model/physics effects.
"""
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import time

from . import evaluation_core_identity as identity, mix_core_delegation as contract


def need(ok, why):
    if not ok:raise PermissionError('CORE_DELEGATED_PROCESS_'+why)


def observe(root, body, row):
    root=Path(root);request=body['request'];pid=row['pid'];proc=Path('/proc')/str(pid)
    need(os.uname().nodename=='corelab-4080s' and os.getuid()==2009
         and os.environ.get('USER')=='liuyu-nfs' and not os.environ.get('PBS_JOBID'),
         'ACTUAL_REGISTERED_CORE_ACCOUNT')
    need(root.is_absolute() and not root.is_symlink() and root.stat().st_uid==2009,
         'ACTUAL_OWN_REGISTERED_ROOT')
    if pid==os.getpid():
        return identity.observe_current(root,dict(request,request_sha256=row['request_sha256']),
                                        allow_model_context=row.get('charged_model_loads',0)>0)
    need(proc.stat().st_uid==2009 and 0<len(os.sched_getaffinity(pid))<=4,'OWN_BOUND_PROCESS_AND_ORIGINAL_CPU_AFFINITY')
    stat=(proc/'stat').read_text().rsplit(')',1)[1].split()
    need(stat[0] not in ('Z','X','x') and int(stat[19])==row['proc_start'],
         'SAME_LIVE_KERNEL_PROCSTART_NOT_PID_REUSE')
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    need(boot==row['boot_id'],'SAME_KERNEL_BOOT')
    group=(proc/'cgroup').read_text().strip().split('::',1)
    need(len(group)==2 and group[0]=='0','ACTUAL_UNIFIED_CGROUP')
    cg=Path('/sys/fs/cgroup')/group[1].lstrip('/')
    limits=identity.kernel_limits((cg/'cpu.max').read_text(),(cg/'memory.max').read_text(),
                                 (cg/'pids.max').read_text())
    result=subprocess.run(['systemctl','--user','show',request['native_unit'],'-p','MainPID',
        '-p','ActiveState','-p','ControlGroup','-p','RuntimeMaxUSec','-p','ExecMainStartTimestampMonotonic'],
        capture_output=True,text=True,timeout=8)
    need(result.returncode==0,'BOUNDED_ACTUAL_UNIT_READ')
    properties=dict(line.split('=',1) for line in result.stdout.splitlines() if '=' in line)
    need(properties.get('MainPID')==str(pid) and properties.get('ActiveState')=='active'
         and properties.get('ControlGroup')==group[1],'SAME_ACTIVE_UNIT_MAINPID')
    lock=root.parent.parent/'control/global-native-GPU-locks'/(request['GPU_uuid']+'.lock')
    need(lock.is_file() and not lock.is_symlink() and lock.stat().st_uid==2009,'OWN_DEVICE_LOCK')
    held=False
    for fd in (proc/'fd').iterdir():
        try:
            if fd.resolve()==lock.resolve() and fd.stat().st_ino==lock.stat().st_ino:
                lines=(proc/'fdinfo'/fd.name).read_text().splitlines()
                # flock ownership follows the inherited open-file description.
                held=held or any('FLOCK' in line and ' WRITE ' in line
                                for line in lines if line.startswith('lock:'))
        except FileNotFoundError:continue
    need(held,'ACTUAL_BOUND_PROCESS_HOLDS_INHERITED_DEVICE_LOCK')
    raw={}
    for name,args in [('devices',['--query-gpu=index,uuid,name,pci.bus_id,driver_version,memory.total,memory.used,utilization.gpu']),
                      ('processes',['--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory'])]:
        value=subprocess.run(['nvidia-smi',*args,'--format=csv,noheader,nounits'],
                             capture_output=True,text=True,timeout=8)
        need(value.returncode==0,'COMPLETE_ACTUAL_GPU_READ_NO_UNKNOWN_DEVICE')
        raw[name]=dict(stdout=value.stdout,stderr=value.stderr,returncode=value.returncode)
    device=identity.parse_GPU_rows(raw['devices']['stdout'],raw['processes']['stdout'],
        GPU_uuid=request['GPU_uuid'],own_pid=pid if row.get('charged_model_loads',0)>0 else None,
        desktop_request=request if 'Core_desktop_coexistence_grant_sha256' in request else None)
    value=deepcopy(body['admission']['allocation']['native_process'])
    value.update(observed_epoch=time.time(),kernel_limits=limits,native_unit_properties=properties,
                 native_GPU_raw=raw,device=device)
    contract._live_process(body,row,value)
    return value
