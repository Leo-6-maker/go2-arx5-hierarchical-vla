"""Strict E1 entry and atomic shared resource reservations, not scientific gates."""
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import socket
import time

from scripts.pi0292.durable import read, write
from scripts.pi0294.e0_scope import inside, STATS, DATA, S1_PROOF
from scripts.pi0294.native_dataset import sha

AUTHORITY='c851cc791e1fbb3efb293f7075f0e771518cf5c0'
CARD_SHA='644dd104291221aaec840e49d2c72bb0c90019740bebe7b06b53e1d5d6c0cec4'
CAPS=dict(main_starts=32,recovery_forks=16,live_and_recovery_integrations=800000,
          shadow_integrations=1200000,generation_queries=3200,optimizer_updates=2048)


def validate(root, profile, *, physical=False, now=None, verify_source=True):
    root=Path(root);now=time.time() if now is None else now
    a=read(root/'E1_ADOPTION_AND_BINDING.json')
    manifest_path=inside(os.environ.get('E1_RUN_MANIFEST',root/'RUN_MANIFEST.json'),root)
    m=read(manifest_path)
    card_path=root/'evidence/E1_TASK_CARD.json'
    if sha(card_path)!=CARD_SHA:raise PermissionError('E1_EXACT_ADOPTED_MACHINE_CARD')
    card=read(card_path)
    if (a['authority_release']!=AUTHORITY or a['machine_card_sha256']!=CARD_SHA
            or a['owner']!='EXISTING_SINGLE_LOCAL_PC_OWNER' or a['run_uuid']!=root.name
            or not root.name.endswith('-localpc-e1') or a['reset_on_resume'] is not False
            or a['allocation_admission_cutoff_epoch']!=a['T0_epoch']+86400
            or a['hard_closeout_epoch']!=a['T0_epoch']+129600
            or a['budget']!=card['budget'] or a['frozen']!=card['frozen']
            or a['training_recipe']!=card['training_recipe']
            or not a['T0_epoch']<=now<a['hard_closeout_epoch']):
        raise PermissionError('E1_EXACT_NEW_AUTHORITY_CLOCK_AND_OWNER')
    if (m['schema']!='RUN_MANIFEST_V1' or m['run_uuid']!=root.name
            or m['authority']['protocol_ref']!=AUTHORITY or m['limits']!=a['budget']
            or m['data']['manifest_sha256']!=DATA
            or m['interfaces']['statistics_sha256']!=STATS):
        raise PermissionError('E1_EXACT_MANIFEST_SOURCE_DATA_AND_STATISTICS')
    b=read(inside(root/'OUTPUT_BINDING.private.json',root));e=m['execution']
    if sha(root/'OUTPUT_BINDING.private.json')!=e['output_binding_sha256']:
        raise PermissionError('E1_OUTPUT_BINDING_CHANGED')
    inside(root,b['registered_parent']);host=socket.gethostname()
    compute=profile=='hopper-neural' and os.environ.get('PBS_JOBID') and host.startswith('hopper-')
    if (e['profile']!=profile or e['uid']!=os.getuid()
            or not (host in e['hostnames'] or compute)
            or Path(b['run_root']).resolve()!=root.resolve() or root.is_symlink()
            or root.stat().st_dev!=Path(b['registered_parent']).stat().st_dev):
        raise PermissionError('E1_EXACT_HOST_PROFILE_OWNER_AND_REGISTERED_OUTPUT')
    lease=read(b['lease']);registry=read(b['registry'])
    if (lease['status']!='ACTIVE' or lease['run_uuid']!=root.name
            or lease['protocol_ref']!=AUTHORITY or lease['deadline_epoch']!=a['hard_closeout_epoch']
            or not any(r.get('run_uuid')==root.name and r['role']==b['role']
                and Path(r['path']).resolve()==root.resolve() for r in registry['entries'])):
        raise PermissionError('E1_ACTUAL_ACTIVE_REGISTERED_LEASE_REQUIRED')
    if sha(root/'evidence/S1_TEACHER_QUALIFIED.json')!=S1_PROOF:
        raise PermissionError('E1_ORIGINAL_S1_QUALIFICATION_REQUIRED')
    if physical and (root/'SOURCE_PHYSICS_STOP.json').exists():
        raise PermissionError('E1_ACTUAL_SAFETY_OR_STATE_INTEGRITY_STOP')
    if verify_source:
        code=inside(root/m['code']['capsule_relative'],root)
        if 'E1_RUN_MANIFEST' in os.environ and manifest_path.resolve()!=(code.parent/'WORKER_RUN_MANIFEST.json').resolve():
            raise PermissionError('E1_IMMUTABLE_PER_WORKER_MANIFEST_REQUIRED')
        seal_path=inside(root/m['code']['source_seal_relative'],root);seal=read(seal_path)
        if (Path(__file__).resolve()!=(code/'scripts/pi0294/e1/scope.py').resolve()
                or sha(seal_path)!=m['code']['source_seal_sha256']
                or seal['source_sha']!=m['code']['vla_commit']):
            raise PermissionError('E1_EXACT_SEALED_RUNTIME_REQUIRED')
        for name,digest in seal['files'].items():
            if sha(inside(code/name,code))!=digest:raise PermissionError('E1_SOURCE_CHANGED:'+name)
        for name,digest in m['code']['unchanged_numerical_files'].items():
            if seal['files'].get(name)!=digest:raise PermissionError('E1_FROZEN_NUMERICAL_SOURCE_CHANGED:'+name)
    return m


class Ledger:
    def __init__(self,root,role):
        self.root=Path(root);self.role=role
        self.path=self.root/('RESOURCE_LEDGER.json' if role=='core' else 'HOPPER_RESOURCE_LEDGER.json')

    @contextmanager
    def transaction(self):
        with self.path.with_suffix('.owner.lock').open('a') as f:
            fcntl.flock(f,fcntl.LOCK_EX)
            value=read(self.path)
            yield value
            write(self.path,value)

    def charge(self,key,count):
        if type(count) is not int or count<0 or key not in CAPS:raise ValueError('INVALID_RESOURCE_CHARGE')
        with self.transaction() as l:
            if l.get(key,0)+count>CAPS[key]:raise PermissionError('E1_RESOURCE_CAP:'+key)
            if key in ('live_and_recovery_integrations','shadow_integrations'):
                other='shadow_integrations' if key=='live_and_recovery_integrations' else 'live_and_recovery_integrations'
                if l.get(key,0)+count+l.get(other,0)>2000000:raise PermissionError('E1_ALL_INTEGRATIONS_CAP')
            l[key]=l.get(key,0)+count

    @contextmanager
    def steps(self,kind,maximum,identifier):
        if kind not in ('shadow_integrations','live_and_recovery_integrations') or type(maximum) is not int or maximum<1:
            raise ValueError('EXACT_STEP_RESERVATION_REQUIRED')
        with self.transaction() as l:
            pending=l.setdefault('reservations',{})
            if identifier in pending:raise PermissionError('PRESERVE_PRIOR_PHYSICAL_RESERVATION')
            total=sum(v['maximum'] for v in pending.values())
            same=sum(v['maximum'] for v in pending.values() if v['kind']==kind)
            if l.get(kind,0)+same+maximum>CAPS[kind] or l.get('shadow_integrations',0)+l.get('live_and_recovery_integrations',0)+total+maximum>2000000:
                raise PermissionError('E1_INTEGRATION_RESERVATION_CAP')
            pending[identifier]=dict(kind=kind,maximum=maximum,started_epoch=time.time())
        meter=dict(actual=0)
        try:yield meter
        finally:
            count=meter['actual']
            if type(count) is not int or not 0<=count<=maximum:raise ValueError('UNKNOWN_NATIVE_STEP_COUNT_PRESERVE_RESERVATION')
            with self.transaction() as l:
                reservation=l['reservations'].pop(identifier)
                if reservation['kind']!=kind:raise ValueError('STEP_RESERVATION_CHANGED')
                l[kind]=l.get(kind,0)+count

    def stop_physics(self,reason,detail):
        p=self.root/'SOURCE_PHYSICS_STOP.json'
        if not p.exists():write(p,dict(status='STOP_ALL_NEW_E1_SOURCE_PHYSICS',reason=reason,detail=detail,observed_epoch=time.time()))

    def guard(self, *, closeout=False):
        a=read(self.root/'E1_ADOPTION_AND_BINDING.json')
        if time.time()>=a['hard_closeout_epoch']:raise TimeoutError('E1_HARD_CLOSEOUT')
        l=read(self.path);key='Core_CPU_s' if self.role=='core' else 'CPU_s'
        cap=172800-(0 if closeout else 21600) if self.role=='core' else 115200
        if l[key]>=cap:raise PermissionError('E1_CPU_OR_CLOSEOUT_RESERVE')

    def cpu(self,cost):
        if not 0<=cost<float('inf'):raise ValueError('INVALID_CPU_COST')
        with self.transaction() as l:
            key='Core_CPU_s' if self.role=='core' else 'CPU_s';l[key]+=cost
