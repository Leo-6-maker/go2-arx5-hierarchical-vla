"""Cold V2 recovery, same finite budget, no automatic neural training."""
import copy,time
from pathlib import Path
from scripts.pi0292.durable import write
from .durable_control_v2 import short_lock,read_control
from .mfree_unattended_ledger_v1 import Ledger,sha

def register(parent,entries):
    # Lock order: no claim lock is held here. The short registration transaction
    # contains file/registry publication only, never a Popen or a model load.
    with short_lock(Path(parent['world_control'])/'registration.lock'):
        registry=read_control(parent['registry'])
        for path,spec in entries:
            path=Path(path);sp=path/'EXECUTION_SPEC.json';expected=__import__('hashlib').sha256((__import__('json').dumps(spec,indent=2,allow_nan=False)+'\n').encode()).hexdigest()
            if sp.exists():
                if sha(sp)!=expected:raise ValueError('PARTIAL_REGISTRATION_SPEC_MISMATCH')
            else:write(sp,spec)
            lease=path/'control/LEASE.json'
            if lease.exists():
                value=read_control(lease)
                if value['spec_sha256']!=expected or value['status']!='ACTIVE':raise ValueError('PARTIAL_REGISTRATION_LEASE_MISMATCH')
            else:write(lease,dict(status='ACTIVE',owner=spec['owner'],spec_sha256=expected,deadline_epoch=spec['deadline_epoch']))
            matches=[e for e in registry['entries']if e.get('path')==str(path)]
            if len(matches)>1:raise ValueError('DUPLICATE_REGISTERED_OWNER')
            if matches:
                if any(matches[0][k]!=spec[k]for k in ('source_sha','owner','role')):raise ValueError('PARTIAL_REGISTRY_OWNER_MISMATCH')
            else:registry['entries'].append(dict(path=str(path),run_uuid=path.name+'-'+Path(parent['campaign_root']).name,role=spec['role'],source_sha=spec['source_sha'],owner=spec['owner'],executor_node='corelab-4080s',scope=parent.get('registry_scope','MFREE_COLD_RECOVERY_V2_NO_AUTOTRAIN'),local_lease=str(lease),ROS_DOMAIN_ID=spec['ROS_DOMAIN_ID']))
        write(parent['registry'],registry)
    return dict(phase='REGISTERED',paths=[str(p)for p,_ in entries],epoch=time.time())

class RecoveryLedger(Ledger):
    def claim(self,worker,*,now=None):
        now=time.time()if now is None else now
        with self.transaction()as v:
            if now>=v['deadline_epoch']-3600 or 'C60'in v['snapshots']:return None
            if any(r.get('worker')==worker and r['state']in ('CLAIMED','REGISTERED','LAUNCHED','RUNNING','UNKNOWN')for r in v['attempts']):raise PermissionError('EXISTING_OWNER_NO_REPLAY')
            pending=next((r for r in v['attempts']if r['state']=='CONFIRMED_NOT_STARTED'and not r.get('technical_successor')),None)
            if pending:
                if len(v['attempts'])>=200:raise PermissionError('TOTAL_BUDGET_INCLUDES_TECHNICAL_OVERHEAD')
                row=copy.deepcopy(pending);row.update(id=pending['id']+'-TECH1',state='CLAIMED',worker=worker,carried_over=False,technical_successors=1,technical_parent=pending['id'],claimed_epoch=now,physical_requests=0,native_integrations=0)
                for k in ('failure','reconciliation','reconciliation_history','process','spec','result'):row.pop(k,None)
                pending['technical_successor']=row['id'];v['attempts'].append(row);return copy.deepcopy(row)
        return super().claim(worker,now=now)

def launch_service(spec,unit,operation,*,bootstrap='mfree_recovery_bootstrap_v2.py',memory_max_bytes=None):
    import os,subprocess
    root=Path(spec['registered_root']);duration=max(1,int(spec['deadline_epoch']-time.time()))
    cmd=['systemd-run','--user','--unit='+unit,'--property=Restart=no','--property=RuntimeMaxSec='+str(duration),'--property=KillMode=control-group','--property=TimeoutStopSec=180','--property=StandardInput=null','--property=CPUAffinity='+' '.join(map(str,spec['cpus'])),'--property=StandardOutput=append:'+str(root/'logs/supervisor.log'),'--property=StandardError=append:'+str(root/'logs/supervisor.log')]
    if memory_max_bytes is not None:
        cmd.append('--property=MemoryMax='+str(memory_max_bytes))
    env=dict(spec['native_environment']);env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    for key in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy','SSH_AUTH_SOCK','DISPLAY'):env[key]=''
    if operation=='--diagnostic':env.update(spec['diagnostic_environment'])
    for key,value in env.items():cmd.append('--setenv='+key+'='+str(value))
    cmd.extend([spec['python'],'-B',str(Path(spec['source_root'])/'scripts/wm02'/bootstrap),operation,str(root/'EXECUTION_SPEC.json')])
    r=subprocess.run(cmd,capture_output=True,text=True,timeout=30);receipt=dict(unit=unit,exit=r.returncode,out=r.stdout,err=r.stderr,epoch=time.time());write(root/'LAUNCH_ACTUAL.private.json',receipt)
    if r.returncode:raise RuntimeError('LAUNCH_UNKNOWN_REQUIRES_RECONCILIATION:'+str(receipt))
    return receipt
