"""Persistent bounded owner accounting; stdlib resource must not be shadowed."""
from __future__ import annotations

import fcntl
import json
from pathlib import Path
import resource
import time
from .config import COMMISSIONING,PILOTS,TRAIN,DEV,AUTHORITY
from go2arx5_vla.task_data.io import write

LIMITS=dict(starts=100,commissioning=12,pilot=8,train=64,dev=16,physics_steps=1608000,pairs=64100,GPU_s=4*3600.,CPU_s=48*3600,output_bytes=40*1024**3)
PHYSICS_CUTOFF=1791849137.0
DEADLINE=1791851837.0


def owned_bytes(root):
    return sum(p.stat().st_size for p in Path(root).rglob('*') if p.is_file() and not p.is_symlink())


class Budget:
    def __init__(self,root,gpu_service=False):
        self.gpu_service=gpu_service
        self.root=Path(root);self.path=self.root/'RESOURCE_ACCOUNTING.json'
        self.last_storage_check=-float('inf')
        self.lease=(self.root/'PHYSICS_OWNER.lock').open('a')
        fcntl.flock(self.lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if self.path.exists():self.value=json.loads(self.path.read_text())
        else:
            d0={'GPU_service_s':0.,'CPU_s':0.}
            self.value=dict(schema='PI0291_RESOURCE_ACCOUNTING_V1',authority=AUTHORITY,limits=LIMITS,
                starts=[],physics_steps=0,pairs=0,GPU_s=d0['GPU_service_s'],CPU_s=d0['CPU_s'],
                unmetered_dependency_setup_CPU_s_upper=600.,setup_steps_registered_max=8000,setup_steps_actual=0,
                setup_CPU_accounting='Conservative bounded metadata/bootstrap reserve 600 CPU seconds; runtime reused',
                events=[],GR00T=0,WBC=0,Hopper_PBS=0,hardware=0)
            self.save()
        self.began=time.monotonic();self.cpu0=resource.getrusage(resource.RUSAGE_SELF)
        self.child0=resource.getrusage(resource.RUSAGE_CHILDREN)

    def save(self):write(self.path,self.value)

    def start(self,identifier,source):
        self.guard()
        if time.time()>=PHYSICS_CUTOFF:raise RuntimeError('BUDGET_STOP_PHYSICS_DEADLINE')
        if identifier not in COMMISSIONING+PILOTS+TRAIN+DEV or any(x['id']==identifier for x in self.value['starts']):
            raise RuntimeError('Unregistered or already attempted ID: no quality retry')
        if len(self.value['starts'])>=LIMITS['starts']:raise RuntimeError('BUDGET_STOP_STARTS')
        self.value['starts'].append(dict(id=identifier,source_sha=source,status='STARTED',steps=0,pairs=0,epoch=time.time()))
        self.save()

    def guard(self):
        if time.time()>=DEADLINE:raise RuntimeError('BUDGET_STOP_DEADLINE')
        cpu=resource.getrusage(resource.RUSAGE_SELF);child=resource.getrusage(resource.RUSAGE_CHILDREN)
        elapsed_cpu=cpu.ru_utime+cpu.ru_stime-self.cpu0.ru_utime-self.cpu0.ru_stime+child.ru_utime+child.ru_stime-self.child0.ru_utime-self.child0.ru_stime
        if self.value['physics_steps']>LIMITS['physics_steps'] or self.value['pairs']+self.value.get('diagnostic_pairs_actual',0)>LIMITS['pairs']:raise RuntimeError('BUDGET_STOP_COUNTS')
        if self.value['GPU_s']+(time.monotonic()-self.began if self.gpu_service else 0)>=LIMITS['GPU_s']:raise RuntimeError('BUDGET_STOP_GPU')
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
