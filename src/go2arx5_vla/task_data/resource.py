from __future__ import annotations

import fcntl
import json
from pathlib import Path
import resource
import time
from .config import COMMISSIONING,PILOTS,AUTHORITY
from .io import write

LIMITS=dict(starts=12,commissioning=4,pilot=8,physics_steps=200000,pairs=7692,GPU_s=3600.,CPU_s=24*3600,output_bytes=10*1024**3)


def owned_bytes(root):
    return sum(p.stat().st_size for p in Path(root).rglob('*') if p.is_file() and not p.is_symlink())


class Budget:
    def __init__(self,root):
        self.root=Path(root);self.path=self.root/'RESOURCE_ACCOUNTING.json'
        self.last_storage_check=-float('inf')
        self.lease=(self.root/'PHYSICS_OWNER.lock').open('a')
        fcntl.flock(self.lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if self.path.exists():self.value=json.loads(self.path.read_text())
        else:
            d0=json.loads((self.root/'evidence/D0_RUNTIME.json').read_text())
            self.value=dict(schema='PI0290_RESOURCE_ACCOUNTING_V1',authority=AUTHORITY,limits=LIMITS,
                starts=[],physics_steps=0,pairs=0,GPU_s=d0['GPU_service_s'],CPU_s=d0['CPU_s'],
                unmetered_dependency_setup_CPU_s_upper=7200.,setup_steps_registered_max=8000,setup_steps_actual=0,
                setup_CPU_accounting='Conservative 6-minute bootstrap interval x20 host CPUs plus metadata margin, not measured utilization',
                events=[],GR00T=0,WBC=0,Hopper_PBS=0,hardware=0)
            self.save()
        self.began=time.monotonic();self.cpu0=resource.getrusage(resource.RUSAGE_SELF)
        self.child0=resource.getrusage(resource.RUSAGE_CHILDREN)

    def save(self):write(self.path,self.value)

    def start(self,identifier,source):
        self.guard()
        if identifier not in COMMISSIONING+PILOTS or any(x['id']==identifier for x in self.value['starts']):
            raise RuntimeError('Unregistered or already attempted ID: no quality retry')
        if len(self.value['starts'])>=LIMITS['starts']:raise RuntimeError('BUDGET_STOP_STARTS')
        self.value['starts'].append(dict(id=identifier,source_sha=source,status='STARTED',steps=0,pairs=0,epoch=time.time()))
        self.save()

    def guard(self):
        cpu=resource.getrusage(resource.RUSAGE_SELF);child=resource.getrusage(resource.RUSAGE_CHILDREN)
        elapsed_cpu=cpu.ru_utime+cpu.ru_stime-self.cpu0.ru_utime-self.cpu0.ru_stime+child.ru_utime+child.ru_stime-self.child0.ru_utime-self.child0.ru_stime
        if self.value['physics_steps']>LIMITS['physics_steps'] or self.value['pairs']>LIMITS['pairs']:raise RuntimeError('BUDGET_STOP_COUNTS')
        if self.value['GPU_s']+time.monotonic()-self.began>=LIMITS['GPU_s']:raise RuntimeError('BUDGET_STOP_GPU')
        if self.value['CPU_s']+self.value['unmetered_dependency_setup_CPU_s_upper']+elapsed_cpu>=LIMITS['CPU_s']:raise RuntimeError('BUDGET_STOP_CPU')
        if time.monotonic()-self.last_storage_check>=2:
            if owned_bytes(self.root)>LIMITS['output_bytes']-16*1024**2:raise RuntimeError('BUDGET_STOP_STORAGE')
            self.last_storage_check=time.monotonic()

    def progress(self,identifier,steps,pairs):
        event=next(x for x in self.value['starts'] if x['id']==identifier)
        self.value['physics_steps']+=steps-event['steps'];self.value['pairs']+=pairs-event['pairs']
        event.update(steps=steps,pairs=pairs);self.save()

    def finish(self,identifier,status,GPU_s):
        self.progress(identifier,*[(next(x for x in self.value['starts'] if x['id']==identifier))[k] for k in ['steps','pairs']])
        cpu=resource.getrusage(resource.RUSAGE_SELF);child=resource.getrusage(resource.RUSAGE_CHILDREN)
        CPU_s=cpu.ru_utime+cpu.ru_stime-self.cpu0.ru_utime-self.cpu0.ru_stime+child.ru_utime+child.ru_stime-self.child0.ru_utime-self.child0.ru_stime
        self.value['GPU_s']+=GPU_s;self.value['CPU_s']+=CPU_s
        next(x for x in self.value['starts'] if x['id']==identifier).update(status=status,GPU_s=GPU_s,CPU_s=CPU_s)
        self.value['owned_output_bytes']=owned_bytes(self.root);self.save()
        fcntl.flock(self.lease,fcntl.LOCK_UN);self.lease.close()

    def finish_cpu(self,name):
        cpu=resource.getrusage(resource.RUSAGE_SELF);child=resource.getrusage(resource.RUSAGE_CHILDREN)
        used=cpu.ru_utime+cpu.ru_stime-self.cpu0.ru_utime-self.cpu0.ru_stime+child.ru_utime+child.ru_stime-self.child0.ru_utime-self.child0.ru_stime
        self.value['CPU_s']+=used;self.value['events'].append(dict(stage=name,CPU_s=used,GPU_s=0))
        self.save();fcntl.flock(self.lease,fcntl.LOCK_UN);self.lease.close()
