"""Registered finite service helpers; no scientific retries or mutable source."""
import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
from scripts.pi0292.durable import OwnerLock,write
from scripts.pi0294.e1.journal import Journal,process_is_same
from .mfree_unattended_ledger_v1 import Ledger,read,sha
GIB=1024**3

def dirs(path):
    path=Path(path);path.mkdir()
    for name in ('control','logs','tmp','cache','data'):(path/name).mkdir()

def fields(spec):return {k:spec[k]for k in('source_sha','source_root','source_files','trusted_source_parent','source_seal','registry','owner')}

def register(parent,entries):
    lock=OwnerLock(Path(parent['world_control'])/'registration.lock')
    try:
        registry=read(parent['registry'])
        for path,spec in entries:
            path=Path(path)
            if (path/'EXECUTION_SPEC.json').exists():raise FileExistsError('NEW_LEASE_ONLY_NO_OVERWRITE')
            write(path/'EXECUTION_SPEC.json',spec)
            write(path/'control/LEASE.json',dict(status='ACTIVE',owner=spec['owner'],spec_sha256=sha(path/'EXECUTION_SPEC.json'),deadline_epoch=spec['deadline_epoch']))
            registry['entries'].append(dict(path=str(path),run_uuid=path.name+'-'+Path(parent['campaign_root']).name,role=spec['role'],source_sha=spec['source_sha'],owner=spec['owner'],executor_node='corelab-4080s',scope='MFREE_UNATTENDED_72H_FINITE',local_lease=str(path/'control/LEASE.json'),ROS_DOMAIN_ID=spec['ROS_DOMAIN_ID']))
        write(parent['registry'],registry)
    finally:lock.close()

def unit_state(unit):
    p=subprocess.run(['systemctl','--user','show',unit,'-p','ActiveState','-p','SubState','-p','MainPID','-p','ControlGroup','-p','MemoryCurrent','-p','MemoryPeak','-p','InvocationID'],capture_output=True,text=True,timeout=15)
    if p.returncode:raise RuntimeError('UNIT_IDENTITY_UNKNOWN:'+unit+':'+p.stderr)
    value=dict(line.split('=',1)for line in p.stdout.splitlines()if '='in line)
    if 'MainPID'not in value:raise RuntimeError('UNIT_MAINPID_UNKNOWN')
    pid=int(value['MainPID']);value['pid']=pid;value['unit']=unit
    if pid:
        proc=Path('/proc')/str(pid)
        value['proc_start']=proc.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]
        value['proc_cgroup']=proc.joinpath('cgroup').read_text()
        if value['ControlGroup']not in value['proc_cgroup']:raise RuntimeError('UNIT_CGROUP_IDENTITY_MISMATCH')
    cg=value.get('ControlGroup')
    if cg:
        group=Path('/sys/fs/cgroup')/cg.lstrip('/')/'cgroup.procs'
        value['cgroup_pids']=[int(x)for x in group.read_text().split()]if group.exists()else[]
    else:value['cgroup_pids']=[]
    return value

def started_unit(unit):
    until=time.monotonic()+15
    while True:
        state=unit_state(unit)
        if state['pid']:return state
        if time.monotonic()>until:raise RuntimeError('UNIT_NOT_STARTED_REQUIRES_RECONCILIATION:'+unit)
        time.sleep(.2)

def launch_service(spec,unit,operation):
    root=Path(spec['registered_root']);duration=max(1,int(spec['deadline_epoch']-time.time()))
    cmd=['systemd-run','--user','--unit='+unit,'--property=Restart=no','--property=RuntimeMaxSec='+str(duration),'--property=KillMode=control-group','--property=TimeoutStopSec=180','--property=StandardInput=null','--property=CPUAffinity='+' '.join(map(str,spec['cpus'])),'--property=StandardOutput=append:'+str(root/'logs/supervisor.log'),'--property=StandardError=append:'+str(root/'logs/supervisor.log')]
    env=dict(spec['native_environment']);env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    for key in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy','SSH_AUTH_SOCK','DISPLAY'):env[key]=''
    for key,value in env.items():cmd.append('--setenv='+key+'='+str(value))
    cmd.extend([spec['python'],'-B',str(Path(spec['source_root'])/'scripts/wm02/mfree_unattended_bootstrap_v1.py'),operation,str(root/'EXECUTION_SPEC.json')])
    p=subprocess.run(cmd,capture_output=True,text=True,timeout=30)
    value=dict(unit=unit,exit=p.returncode,out=p.stdout,err=p.stderr,epoch=time.time())
    write(root/'LAUNCH_ACTUAL.private.json',value)
    if p.returncode:raise RuntimeError('MANAGED_LAUNCH_REQUIRES_RECONCILIATION:'+str(value))
    return value

def disk_admission(spec,extra):
    root=Path(spec['campaign_root']);usage=shutil.disk_usage(root);p=root/'control/DATA_BYTES.private.json'
    if not p.exists()or time.time()-read(p)['epoch']>360:raise RuntimeError('DISK_USAGE_MEASUREMENT_STALE')
    used=read(p)['bytes'];free_floor=max(50*GIB,int(.15*usage.total))
    value=dict(epoch=time.time(),free=usage.free,total=usage.total,floor=free_floor,required_extra=extra,new_bytes=used)
    write(Path(spec['registered_root'])/'DISK_LAST_ACTUAL.private.json',value)
    return usage.free>=free_floor+extra and used+extra<=512*GIB

def memory_admission(spec,worlds,*,neural=False):
    available=int(next(x.split()[1]for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:')))*1024
    required=spec['model_host_peak_bound_bytes']+spec['host_margin_bytes']+spec['frame_cache_bytes']
    current_neural=0;measurements=[]
    for unit in spec['resource_units']:
        actual=unit_state(unit);measurements.append(actual)
        if actual['pid']:
            raw=actual.get('MemoryCurrent','')
            if not raw.isdigit():raise RuntimeError('MEMORY_MEASUREMENT_UNKNOWN:'+unit)
            if unit==spec['old_pipeline_unit']or '-neural-'in unit:current_neural+=int(raw)
    required=max(spec['host_margin_bytes'],required-current_neural)+worlds*spec['native_peak_bound_bytes']
    # MemAvailable already excludes live RSS: subtract measured native use only
    # after bounding it by the immutable per-world admission envelope.
    for actual in measurements:
        if '-collector-'in actual['unit']and actual['pid']:required-=min(int(actual['MemoryCurrent']),spec['native_peak_bound_bytes'])
    value=dict(epoch=time.time(),available=available,required=max(required,0),worlds=worlds,units=measurements)
    write(Path(spec['registered_root'])/'MEMORY_LAST_ACTUAL.private.json',value)
    return available>=required

def close(spec,code,**detail):
    root=Path(spec['registered_root']);write(root/'WORKER_EXIT.private.json',dict(exit_code=code,**detail))
    write(root/'control/LEASE.json',dict(read(root/'control/LEASE.json'),status='CLOSED',worker_exit_code=code,closed_epoch=time.time()))
