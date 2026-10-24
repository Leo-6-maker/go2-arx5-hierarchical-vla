"""Real Core native process, GPU and kernel-limit identity; no CUDA/model init."""
import csv
import fcntl
import io
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from . import evaluation_core_accounting as c

STATUS='PASS_PV1_OWN_NATIVE_CORELAB_GPU_ALLOCATION'
HANDLE_ENV='PV1_CORE_NATIVE_GPU_HANDLE'


def need(ok,why):
    if not ok:raise PermissionError('PV1_CORE_NATIVE_'+why)


def current_handle():
    """A Core handle never enters PBS_JOBID; legacy callers retain PBS exactly."""
    value=os.environ.get(HANDLE_ENV)
    if value is None:return os.environ.get('PBS_JOBID')
    need(not os.environ.get('PBS_JOBID') and os.uname().nodename=='corelab-4080s'
         and os.getuid()==2009,'CORE_HANDLE_CANNOT_ALIAS_PBS_OR_OTHER_HOST')
    parts=value.split(':')
    need(len(parts)==4 and parts[0]=='coregpu','TYPED_CORE_HANDLE')
    try:pid,start=int(parts[2]),int(parts[3])
    except ValueError:raise PermissionError('PV1_CORE_NATIVE_CANONICAL_PROCESS_HANDLE')
    need(c.process_handle(parts[1],pid,start)==value,'CANONICAL_PROCESS_HANDLE')
    return value


def parse_GPU_rows(raw,process_raw,*,GPU_uuid,own_pid=None):
    """Parse raw nvidia-smi output; an empty/failed read is not a free device."""
    devices=list(csv.reader(io.StringIO(raw)));matches=[]
    need(devices and all(len(r)==8 for r in devices),'COMPLETE_NATIVE_DEVICE_READ')
    for line in devices:
        index,uuid,name,pci,driver,total,used,util=[v.strip() for v in line]
        if uuid==GPU_uuid:
            need(name=='NVIDIA GeForce RTX 4080 SUPER','ACTUAL_RTX_DEVICE_NOT_RELABELED_A40')
            try:index,total,used,util=int(index),int(total),int(used),int(util)
            except ValueError:raise PermissionError('PV1_CORE_NATIVE_FINITE_GPU_METADATA')
            need(index>=0 and 0<=used<=total and 0<=util<=100 and total*1024**2>=30*1024**3,
                 'REAL_MINIMUM_30GIB_DEVICE_NOT_THE_16GIB_CARD')
            matches.append(dict(index=index,GPU_uuid=uuid,name=name,pci_bus_id=pci,driver_version=driver,
                total_bytes=total*1024**2,used_bytes=used*1024**2))
    need(len(matches)==1,'ONE_EXACT_SELECTED_GPU_UUID')
    processes=list(csv.reader(io.StringIO(process_raw)))
    for line in processes:
        need(len(line)==4,'COMPLETE_COMPUTE_PROCESS_READ')
        uuid,pid,_,_=[v.strip() for v in line]
        try:pid=int(pid)
        except ValueError:raise PermissionError('PV1_CORE_NATIVE_REAL_COMPUTE_PID')
        need(uuid!=GPU_uuid or own_pid is not None and pid==own_pid,'NO_FOREIGN_COMPUTE_ON_SELECTED_DEVICE')
    return matches[0]


def kernel_limits(cpu_max,memory_max,pids_max):
    parts=cpu_max.split()
    need(len(parts)==2 and parts[0]!='max','FINITE_KERNEL_CPU_QUOTA')
    try:quota,period=int(parts[0]),int(parts[1]);memory=int(memory_max);tasks=int(pids_max)
    except ValueError:raise PermissionError('PV1_CORE_NATIVE_FINITE_KERNEL_LIMITS')
    need(0<quota<=4*period and period>0 and memory==48*1024**3 and 0<tasks<=64,
         'KERNEL_FOUR_CPU_48GIB_BOUNDED_TASKS')
    return dict(CPU_quota=quota,CPU_period=period,memory_max_bytes=memory,pids_max=tasks)


def observe_current(root,request,*,allow_model_context=False):
    """Read the current process and unit, not a supplied native PASS object."""
    need(type(allow_model_context) is bool,'EXPLICIT_MODEL_CONTEXT_PHASE')
    r=request;root=Path(root)
    need(os.uname().nodename=='corelab-4080s' and os.getuid()==2009
         and os.environ.get('USER')=='liuyu-nfs' and not os.environ.get('PBS_JOBID')
         and os.environ.get('CUDA_VISIBLE_DEVICES')==r['GPU_uuid'],
         'NATIVE_CORE_ACCOUNT_AND_SINGLE_REAL_VISIBLE_UUID')
    need(root.is_absolute() and not root.is_symlink() and root.stat().st_uid==2009
         and len(os.sched_getaffinity(0))<=4,'OWN_REGISTERED_ROOT_AND_CPU_AFFINITY')
    need(type(r.get('allocation_number')) is int and 1<=r['allocation_number']<=2,
         'ORIGINAL_STAGE_NATIVE_NAMESPACE')
    request_path=root/'pbs'/('alloc-'+str(r['allocation_number']).zfill(2))/'REQUEST.json'
    need(request_path.is_file() and not request_path.is_symlink()
         and request_path.stat().st_uid==2009 and not request_path.stat().st_mode&0o222,
         'OWN_IMMUTABLE_REQUEST_FILE')
    request_raw=request_path.read_bytes()
    need(json.loads(request_raw)=={k:v for k,v in r.items() if k!='request_sha256'}
         and hashlib.sha256(request_raw).hexdigest()==r.get('request_sha256')
         and hashlib.sha256((root/'RUN_MANIFEST.json').read_bytes()).hexdigest()==r['run_manifest_sha256'],
         'REAL_REQUEST_BYTES_NOT_A_CALLER_HANDLE_ASSERTION')
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip();pid=os.getpid()
    stat=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split();start=int(stat[19])
    handle=c.process_handle(boot,pid,start)
    need(os.environ.get(HANDLE_ENV) in (None,handle),'NO_PID_REUSE_OR_REBIND')
    cg=Path('/proc/self/cgroup').read_text().strip().split('::',1)
    need(len(cg)==2 and cg[0]=='0','NATIVE_UNIFIED_CONTROL_GROUP')
    group=Path('/sys/fs/cgroup')/cg[1].lstrip('/')
    limits=kernel_limits((group/'cpu.max').read_text(),(group/'memory.max').read_text(),(group/'pids.max').read_text())
    unit=r['native_unit']
    need(unit==root.name+'-neural.service','EXACT_OWN_NEURAL_UNIT')
    result=subprocess.run(['systemctl','--user','show',unit,'-p','MainPID','-p','ActiveState',
        '-p','ControlGroup','-p','RuntimeMaxUSec','-p','ExecMainStartTimestampMonotonic'],
        capture_output=True,text=True,timeout=8)
    need(result.returncode==0,'REAL_NATIVE_UNIT_QUERY_NOT_TIMEOUT')
    properties=dict(x.split('=',1) for x in result.stdout.splitlines() if '=' in x)
    need(properties.get('MainPID')==str(pid) and properties.get('ActiveState')=='active'
         and properties.get('ControlGroup')==cg[1] and properties.get('RuntimeMaxUSec')=='8h',
         'OWN_ACTIVE_MAIN_PID_AND_ORIGINAL_EIGHT_HOUR_UNIT_BOUND')
    lock=root.parent.parent/'control/global-native-GPU-locks'/(r['GPU_uuid']+'.lock')
    try:fd=int(os.environ['PV1_CORE_GPU_LOCK_FD'])
    except (KeyError,ValueError):raise PermissionError('PV1_CORE_NATIVE_INHERITED_DEVICE_LOCK_FD_REQUIRED')
    need(fd>=3 and lock.is_file() and not lock.is_symlink() and lock.stat().st_uid==2009
         and Path('/proc/self/fd/'+str(fd)).resolve()==lock.resolve()
         and os.fstat(fd).st_ino==lock.stat().st_ino,'ACTUAL_SHARED_DEVICE_LOCK_NOT_A_MARKER_FILE')
    fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    raw={}
    for name,args in [('devices',['--query-gpu=index,uuid,name,pci.bus_id,driver_version,memory.total,memory.used,utilization.gpu']),
            ('processes',['--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory'])]:
        q=subprocess.run(['nvidia-smi',*args,'--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=8)
        need(q.returncode==0,'NATIVE_NVIDIA_READ_FAILED_UNKNOWN_DEVICE_USE')
        raw[name]=dict(stdout=q.stdout,stderr=q.stderr,returncode=q.returncode)
    device=parse_GPU_rows(raw['devices']['stdout'],raw['processes']['stdout'],GPU_uuid=r['GPU_uuid'],
        own_pid=pid if allow_model_context else None)
    # Kernel process start is tied to the actual boot; no caller timestamp may
    # reset the allocation wall when exec switches into the neural interpreter.
    boot_epoch=float(next(x.split()[1] for x in Path('/proc/stat').read_text().splitlines() if x.startswith('btime ')))
    started=boot_epoch+start/os.sysconf('SC_CLK_TCK');now=time.time()
    need(r['created_epoch']-1<=started<c.old.STOP_NEW and started<=now<c.old.STOP_PHYSICS,
         'ACTUAL_PROCESS_START_INSIDE_ORIGINAL_WINDOW')
    return dict(schema='PV1_CORE_GPU_NATIVE_PROCESS_STARTED_V1',hostname='corelab-4080s',uid=2009,
        GPU_uuid=r['GPU_uuid'],request_id=r['request_id'],request_sha256=r['request_sha256'],
        submission_nonce=r['submission_nonce'],run_manifest_sha256=r['run_manifest_sha256'],source_sha=r['source_sha'],
        process_state='RUNNING',observed_epoch=now,started_epoch=started,boot_id=boot,pid=pid,proc_start=start,
        native_handle=handle,kernel_limits=limits,native_unit_properties=properties,native_GPU_raw=raw,
        device=device,native_model_permission=False,physics_authorized=False)
