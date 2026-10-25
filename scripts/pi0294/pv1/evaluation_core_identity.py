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

from . import evaluation_core_accounting as c,evaluation_core_B24_1000 as late

STATUS='PASS_PV1_OWN_NATIVE_CORELAB_GPU_ALLOCATION'
HANDLE_ENV='PV1_CORE_NATIVE_GPU_HANDLE'


def need(ok,why):
    if not ok:raise PermissionError('PV1_CORE_NATIVE_'+why)


def cold_profile(manifest):
    """Reject partial Core metadata before a legacy allocation can see it."""
    m=manifest
    typed=(m.get('platform_evaluation_version')==c.VERSION or m.get('scope')==c.ROLE
        or m.get('execution',{}).get('profile')==c.PROFILE)
    if not typed:return False
    need(m.get('platform_evaluation_version')==c.VERSION and m.get('scope')==c.ROLE
         and m.get('execution',{}).get('profile')==c.PROFILE
         and 'platform_training_version' not in m,'COMPLETE_DISTINCT_CORE_ROLE')
    return True


def sealed_identity(manifest,request,allocation,*,run_uuid,manifest_sha256,request_sha256):
    """Join a saved native admission; never create one from a typed PASS flag."""
    from .evaluation_core_admission import request_identity
    need(cold_profile(manifest),'EXPLICIT_CORE_ROLE')
    request_identity(manifest,request,manifest_sha256)
    r,a=request,allocation
    need(a.get('status')==STATUS and a.get('platform_evaluation_version')==c.VERSION
         and manifest['run_uuid']==r['run_uuid']==a.get('run_uuid')==run_uuid
         and a.get('manifest_sha256')==manifest_sha256
         and a.get('request_sha256')==request_sha256
         and all(a.get(k)==r[k] and type(a[k]) is type(r[k]) for k in (
             'source_sha','request_id','submission_nonce','stage','evaluation_candidate',
             'checkpoint_manifest_sha256','statistics_sha256','native_platform','allocation_number'))
         and a.get('model_loaded') is False and type(a.get('new_optimizer_updates')) is int
         and a['new_optimizer_updates']==0
         and not any(k in a for k in ('project','submit_count','PBS_JOBID','platform_training_version')),
         'EXACT_CORE_ADMISSION_REQUEST_JOIN_NO_PBS_ALIAS')
    p=a.get('native_process');need(type(p) is dict,'ACTUAL_NATIVE_PROCESS_EVIDENCE_REQUIRED')
    handle=c.process_handle(p.get('boot_id'),p.get('pid'),p.get('proc_start'))
    exact=dict(schema='PV1_CORE_GPU_NATIVE_PROCESS_STARTED_V1',hostname='corelab-4080s',uid=2009,
        GPU_uuid=r['GPU_uuid'],request_id=r['request_id'],request_sha256=request_sha256,
        submission_nonce=r['submission_nonce'],run_manifest_sha256=manifest_sha256,source_sha=r['source_sha'],
        process_state='RUNNING',native_handle=handle,native_model_permission=False,physics_authorized=False)
    queued='queue_dispatch_mode' in r
    if queued:
        from .evaluation_core_queue import accepted_request
        need(type(a.get('native_queue_acceptance')) is dict and type(a.get('native_queue_plan')) is dict,
             'ACTUAL_QUEUE_RECEIPT_AND_PLAN_REQUIRED')
        accepted=accepted_request(r,a['native_queue_acceptance'],a['native_queue_plan'],request_sha256=request_sha256)
        need(accepted<=p['started_epoch'],'PROCESS_ONLY_AFTER_ACTUAL_QUEUE_ACCEPTANCE')
    need(all(p.get(k)==v and type(p[k]) is type(v) for k,v in exact.items())
         and a.get('native_process_sha256')==c.digest(p)
         and a.get('job_id')==handle and a.get('native_handle')==handle
         and a.get('hostname')==p['hostname'] and a.get('allocation_start_epoch')==p.get('started_epoch')
         and r['created_epoch']-1<=p['started_epoch']<c.old.STOP_PHYSICS
         and (queued or p['started_epoch']<late.start_limit(r))
         and p['started_epoch']<=p['observed_epoch']<=p['started_epoch']+120
         and a.get('allocation_end_epoch')==min(p['started_epoch']+r['walltime_s'],c.old.STOP_PHYSICS),
         'SAME_REAL_BOOT_PID_START_AND_ORIGINAL_WALL_NOT_A_PREPARED_HANDLE')
    limits=p['kernel_limits'];kernel_limits(str(limits['CPU_quota'])+' '+str(limits['CPU_period']),
        str(limits['memory_max_bytes']),str(limits['pids_max']))
    properties=p['native_unit_properties']
    need(properties.get('MainPID')==str(p['pid']) and properties.get('ActiveState')=='active'
         and properties.get('RuntimeMaxUSec')=='8h' and bool(properties.get('ControlGroup')),
         'RAW_ORIGINAL_UNIT_PID_LIMITS')
    raw=p['native_GPU_raw']
    need(set(raw)=={'devices','processes'} and all(type(v.get('returncode')) is int
         and v['returncode']==0 for v in raw.values()),'SUCCESSFUL_NATIVE_NVIDIA_READS')
    device=parse_GPU_rows(raw['devices']['stdout'],raw['processes']['stdout'],GPU_uuid=r['GPU_uuid'],shared_request=r if 'shared_GPU_grant_sha256' in r else None)
    need(device==p['device'],'RAW_SELECTED_GPU_EQUALS_NATIVE_OBSERVATION')
    return a


def current_identity(root,manifest,request,allocation,*,qualified_model_readback=False):
    """Re-observe the caller's kernel process/unit/device, not only old receipts."""
    need(type(qualified_model_readback) is bool,'EXPLICIT_CURRENT_MODEL_CONTEXT_PHASE')
    root=Path(root);directory=root/'pbs'/('alloc-'+str(request['allocation_number']).zfill(2))
    sealed_identity(manifest,request,allocation,run_uuid=root.name,
        manifest_sha256=hashlib.sha256((root/'RUN_MANIFEST.json').read_bytes()).hexdigest(),
        request_sha256=hashlib.sha256((directory/'REQUEST.json').read_bytes()).hexdigest())
    path=directory/'ALLOCATION_ADMISSION.json'
    need(path.is_file() and not path.is_symlink() and not path.stat().st_mode&0o222
         and path.stat().st_uid==2009 and json.loads(path.read_bytes())==allocation,
         'ACTUAL_IMMUTABLE_SAME_PROCESS_ADMISSION')
    for path in (directory/'NATIVE_PROCESS_FINAL.json',directory/'PBS_PROCESS_FINAL.json'):
        need(not path.exists(),'PREMODEL_ONLY_BEFORE_TERMINAL_OR_MODEL_STATE')
    state=root/'pbs/MODEL_STATE.json'
    if qualified_model_readback:
        # Reading a previously acquired proof after model load must still join
        # the complete raw 600s/80-cycle receipt. It cannot acquire that proof
        # after CUDA/model effects. Capture/Pump always use the default phase.
        from .pre_model_qualification import require
        require(root,directory,allocation)
        need(not state.exists() or json.loads(state.read_bytes()).get('job_id')==allocation['native_handle'],
             'SAME_ADMITTED_RESIDENT_MODEL_STATE')
    else:need(not state.exists(),'PREMODEL_ONLY_BEFORE_MODEL_STATE')
    current=observe_current(root,dict(request,request_sha256=allocation['request_sha256']),
        allow_model_context=qualified_model_readback)
    if 'queue_dispatch_mode' in request:
        ack,plan=current_queue(root,request,request_sha256=allocation['request_sha256'])
        need(ack==allocation['native_queue_acceptance'] and plan==allocation['native_queue_plan'],
             'SAME_ACTUAL_IMMUTABLE_QUEUE_ADMISSION')
    need(current['native_handle']==allocation['native_handle']
         and current['started_epoch']==allocation['allocation_start_epoch']
         and current_handle()==current['native_handle']
         and current['observed_epoch']<allocation['allocation_end_epoch']-2700,
         'FRESH_SAME_CALLER_PROCESS_AND_CLOSEOUT_RESERVE')
    return current


def current_queue(root,request,*,request_sha256):
    """Verify the actual resident owner unit/process and its native queue ACK."""
    from .evaluation_core_queue import accepted_request
    root=Path(root);m=json.loads((root/'RUN_MANIFEST.json').read_bytes())
    def sealed(name):
        path=root/name
        need(path.is_file() and not path.is_symlink() and path.stat().st_uid==2009
             and not path.stat().st_mode&0o222 and hashlib.sha256(path.read_bytes()).hexdigest()==
             m['execution']['sealed_inputs'].get(name),'PRESEALED_ACTUAL_QUEUE_INPUT:'+name)
        return json.loads(path.read_bytes())
    plan=sealed('plans/CORE_RESIDENT_STAGE_QUEUE.json')
    binding=sealed('plans/CORE_RESIDENT_QUEUE_BINDINGS.private.json')
    parent=root.parent.parent;control=Path(binding['queue_root'])
    need(control.parent==parent/'control' and control.name=='v4a-core-resident-'+request['queue_id']
         and not control.is_symlink() and control.stat().st_uid==2009,
         'OWN_EXISTING_CAMPAIGN_REGISTERED_CONTROL_QUEUE')
    paths=[root/'NATIVE_CORE_QUEUE_ACCEPTED.json',control/'NATIVE_CORE_QUEUE_ACCEPTED.json',control/'QUEUE_PLAN.json']
    need(all(p.is_file() and not p.is_symlink() and p.stat().st_uid==2009
         and not p.stat().st_mode&0o222 for p in paths),'ACTUAL_OWN_IMMUTABLE_NATIVE_QUEUE_RECORDS')
    ack=json.loads(paths[0].read_bytes())
    need(paths[0].read_bytes()==paths[1].read_bytes() and json.loads(paths[2].read_bytes())==plan
         and hashlib.sha256(paths[2].read_bytes()).hexdigest()==request['queue_plan_sha256'],
         'EXACT_NATIVE_ACCEPTANCE_AND_PLAN_BYTES_NOT_PREPARED_MARKERS')
    accepted_request(request,ack,plan,request_sha256=request_sha256)
    need(Path('/proc/sys/kernel/random/boot_id').read_text().strip()==ack['boot_id'],
         'SAME_OWNER_BOOT_NO_STALE_QUEUE_REUSE')
    process=Path('/proc')/str(ack['pid']);stat_raw=(process/'stat').read_text().rsplit(')',1)[1].split()
    need(process.stat().st_uid==2009 and stat_raw[0] not in ('Z','X')
         and int(stat_raw[19])==ack['proc_start'],'LIVE_NATIVE_OWNER_PID_NOT_REUSED_OR_TERMINAL')
    result=subprocess.run(['systemctl','--user','show',ack['native_unit'],'-p','MainPID','-p','ActiveState',
        '-p','ControlGroup'],capture_output=True,text=True,timeout=8)
    props=dict(x.split('=',1) for x in result.stdout.splitlines() if '=' in x)
    cg=(process/'cgroup').read_text().strip().split('::',1)
    need(result.returncode==0 and props.get('MainPID')==str(ack['pid']) and props.get('ActiveState')=='active'
         and len(cg)==2 and cg[0]=='0' and props.get('ControlGroup')==cg[1],
         'FRESH_ACTUAL_OWNER_UNIT_AND_KERNEL_PROCESS_GROUP')
    return ack,plan


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


def require_device(request,allocation,*,visible,count,name,memory_bytes):
    """CUDA metadata after real admission and qualification; no new precision."""
    r,a=request,allocation
    need(r.get('platform_evaluation_version')==a.get('platform_evaluation_version')==c.VERSION
         and a.get('status')==STATUS and r.get('stage')==a.get('stage')=='P4_DEV'
         and r.get('evaluation_candidate') in c.stage.CANDIDATES
         and a.get('evaluation_candidate')==r['evaluation_candidate']
         and r.get('checkpoint_manifest_sha256')==a.get('checkpoint_manifest_sha256')==c.old.ENDPOINTS[r['evaluation_candidate']]
         and r.get('statistics_sha256')==a.get('statistics_sha256')==c.old.R0
         and type(r.get('optimizer_updates')) is int and r['optimizer_updates']==0
         and r.get('native_platform')=='CORELAB_RTX4080'
         and visible==r.get('GPU_uuid') and type(count) is int and count==1
         and name=='NVIDIA GeForce RTX 4080 SUPER'
         and type(memory_bytes) is int and memory_bytes>=30*1024**3
         and not any('platform_training_version' in v or 'training_block' in v for v in (r,a)),
         'ONE_REAL_VISIBLE_MINIMUM_30GIB_CORE_DEVICE_ONLY_FOR_FROZEN_AB_DEV')
    return True


def parse_GPU_rows(raw,process_raw,*,GPU_uuid,own_pid=None,shared_request=None):
    """Parse raw nvidia-smi output; an empty/failed read is not a free device."""
    from .evaluation_core_shared_B24 import allowed as shared_allowed
    shared=shared_request is not None and shared_allowed(shared_request)
    need(not shared or GPU_uuid==shared_request['GPU_uuid'],'SHARED_ONLY_EXACT_REQUEST_GPU')
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
        if shared and uuid==GPU_uuid:
            try:used_memory=int(line[3].strip())
            except ValueError:raise PermissionError('PV1_CORE_NATIVE_COMPLETE_SHARED_PROCESS_MEMORY')
            need(0<=used_memory<=matches[0]['total_bytes']//1024**2 and bool(line[2].strip()),'ACTUAL_SHARED_PROCESS_MEMORY_AND_IDENTITY')
        need(shared or uuid!=GPU_uuid or own_pid is not None and pid==own_pid,'NO_FOREIGN_COMPUTE_ON_SELECTED_DEVICE')
    if shared:matches[0]['human_authorized_shared_GPU_grant_sha256']=shared_request['shared_GPU_grant_sha256']
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
        own_pid=pid if allow_model_context else None,shared_request=r if 'shared_GPU_grant_sha256' in r else None)
    # Kernel process start is tied to the actual boot; no caller timestamp may
    # reset the allocation wall when exec switches into the neural interpreter.
    boot_epoch=float(next(x.split()[1] for x in Path('/proc/stat').read_text().splitlines() if x.startswith('btime ')))
    started=boot_epoch+start/os.sysconf('SC_CLK_TCK');now=time.time()
    queued='queue_dispatch_mode' in r
    if queued:
        ack,_=current_queue(root,r,request_sha256=r['request_sha256'])
        need(ack['accepted_epoch']<=started,'EXISTING_NATIVE_QUEUE_BEFORE_PROCESS_START')
    need(r['created_epoch']-1<=started<c.old.STOP_PHYSICS and (queued or started<late.start_limit(r))
         and started<=now<c.old.STOP_PHYSICS,
         'ACTUAL_PROCESS_START_INSIDE_ORIGINAL_WINDOW')
    return dict(schema='PV1_CORE_GPU_NATIVE_PROCESS_STARTED_V1',hostname='corelab-4080s',uid=2009,
        GPU_uuid=r['GPU_uuid'],request_id=r['request_id'],request_sha256=r['request_sha256'],
        submission_nonce=r['submission_nonce'],run_manifest_sha256=r['run_manifest_sha256'],source_sha=r['source_sha'],
        process_state='RUNNING',observed_epoch=now,started_epoch=started,boot_id=boot,pid=pid,proc_start=start,
        native_handle=handle,kernel_limits=limits,native_unit_properties=properties,native_GPU_raw=raw,
        device=device,native_model_permission=False,physics_authorized=False)
