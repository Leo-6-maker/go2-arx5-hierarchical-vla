"""A separately leased MP1 source study cannot inherit or expand E1 authority."""
from contextlib import contextmanager
import os
from pathlib import Path
import socket
import time

from scripts.pi0292.durable import write
from scripts.pi0294.e1.control_read import read
from scripts.pi0294.e1.scope import Ledger as FileLedger,DATA,STATS,S1_PROOF
from scripts.pi0294.native_dataset import sha
from .scope import AUTHORITY,CARD_SHA

CAPS=dict(main_starts=64,recovery_forks=0,live_and_recovery_integrations=2048000,shadow_integrations=2048000,
    generation_queries=4000,optimizer_updates=4096)
CPU_CAPS=dict(core=216000,hopper=172800)


def validate_runtime(root,profile,*,now=None,closeout=False):
    root=Path(root);now=time.time() if now is None else now
    stage=read(root/'MP1_STAGE_BINDING.json');parent=root.parent.parent;a=read(parent/'MP1_ADOPTION_AND_INCREMENTAL_BUDGET.json')
    card=parent/'evidence/MP1_TASK_CARD.json'
    if (sha(card)!=CARD_SHA or a['machine_card_sha256']!=CARD_SHA or a['authority_release']!=AUTHORITY
            or a['run_uuid']!=parent.name or a['E2_additive_budget'] or a['E2_separate_T0'] or a['reset_on_resume']
            or a['incremental_budget']!=read(card)['incremental_budget']
            or a['new_submission_and_candidate_cutoff_epoch']!=a['T0_epoch']+144*3600
            or a['new_experiment_cutoff_epoch']!=a['T0_epoch']+160*3600 or a['hard_closeout_epoch']!=a['T0_epoch']+168*3600
            or not a['T0_epoch']<=now<(a['hard_closeout_epoch'] if closeout else a['new_experiment_cutoff_epoch'])):raise PermissionError('MP1_ACTUAL_SINGLE_INCREMENTAL_AUTHORITY_AND_CLOCK')
    if (stage['schema']!='MP1_SOURCE_SHARED_STAGE_V1' or stage['resource_family']!='MP1' or stage['parent_run_uuid']!=parent.name
            or stage['run_uuid']!=root.name or stage['authority_release']!=AUTHORITY or stage['stage_caps']!=CAPS
            or stage['source_data_sha256']!=DATA or stage['statistics_sha256']!=STATS
            or stage['models']!=['S-BC','S-CORR'] or stage['updates_per_branch']!=2048
            or stage['training_seed']!=20261017 or stage['teacher_takeover'] is not False
            or stage['new_DEV_labels'] is not False or stage['CPU_core_s']!=CPU_CAPS
            or stage['allocated_H200_s_max']!=43200 or stage['PBS_submissions_max']!=4):raise PermissionError('MP1_FIXED_FIRST_SOURCE_PAIR_SCOPE')
    manifest=Path(os.environ.get('MP1_RUN_MANIFEST',root/'RUN_MANIFEST.json'));m=read(manifest);b=read(root/'OUTPUT_BINDING.private.json')
    compute=profile=='hopper-neural' and os.environ.get('PBS_JOBID') and socket.gethostname().startswith('hopper-')
    if (m['schema']!='RUN_MANIFEST_V1' or m['run_uuid']!=root.name or m['owner']!='EXISTING_LOCAL_PC'
            or m['authority']['protocol_ref']!=AUTHORITY or m['scope']!='MP1_SOURCE_SHARED_STUDY'
            or m['execution']['profile']!=profile or m['execution']['uid']!=os.getuid()
            or not (socket.gethostname() in m['execution']['hostnames'] or compute)
            or m['execution']['stage_binding_sha256']!=sha(root/'MP1_STAGE_BINDING.json')
            or m['execution']['output_binding_sha256']!=sha(root/'OUTPUT_BINDING.private.json')
            or root.is_symlink() or Path(b['run_root']).resolve()!=root.resolve()
            or not root.resolve().is_relative_to(Path(b['registered_parent']).resolve())
            or root.stat().st_dev!=Path(b['registered_parent']).stat().st_dev):raise PermissionError('MP1_NATIVE_SOURCE_STAGE_HOST_AND_OUTPUT_BINDING')
    lease=read(b['lease']);registry=read(b['registry'])
    if (lease['status']!='ACTIVE' or lease['authority_release']!=AUTHORITY or lease['run_uuid']!=root.name
            or lease['MP1_parent_run']!=parent.name or lease['deadline_epoch']!=a['new_experiment_cutoff_epoch']
            or not any(v.get('run_uuid')==root.name and v['role']==b['role'] and Path(v['path']).resolve()==root.resolve() for v in registry['entries'])):
        raise PermissionError('MP1_NATIVE_REGISTERED_SOURCE_CHILD_LEASE')
    code=root/m['code']['capsule_relative'];seal=root/m['code']['source_seal_relative'];s=read(seal)
    if (not code.resolve().is_relative_to(root.resolve()) or code.is_symlink() or sha(seal)!=m['code']['source_seal_sha256']
            or s['source_sha']!=m['code']['vla_commit'] or Path(__file__).resolve()!=(code/'scripts/pi0294/mp1/science.py').resolve()):
        raise PermissionError('MP1_EXACT_SEALED_SOURCE_STUDY_CAPSULE')
    if 'MP1_RUN_MANIFEST' in os.environ and (manifest.resolve()!=(code.parent/'WORKER_RUN_MANIFEST.json').resolve() or manifest.stat().st_mode&0o222):raise PermissionError('MP1_IMMUTABLE_WORKER_MANIFEST')
    for name,digest in s['files'].items():
        p=code/name
        if not p.resolve().is_relative_to(code.resolve()) or p.is_symlink() or sha(p)!=digest:raise PermissionError('MP1_SOURCE_MEMBER_CHANGED:'+name)
    if any(s['files'].get(n)!=v for n,v in m['code']['unchanged_numerical_files'].items()):raise PermissionError('MP1_ORIGINAL_FROZEN_NUMERICS_CHANGED')
    if m['data']['manifest_sha256']!=DATA or m['interfaces']['statistics_sha256']!=STATS:raise PermissionError('MP1_FIXED_SOURCE_STATS')
    if sha(root/'evidence/S1_TEACHER_QUALIFIED.json')!=S1_PROOF:raise PermissionError('MP1_ORIGINAL_SOURCE_S1_QUALIFICATION')
    return m


class Ledger(FileLedger):
    def __init__(self,root,role):
        super().__init__(root,role)
        if read(self.root/'MP1_STAGE_BINDING.json')['stage_caps']!=CAPS:raise PermissionError('MP1_EXACT_SOURCE_STUDY_COUNTERS')

    def charge(self,key,count):
        if type(count) is not int or count<0 or key not in CAPS:raise ValueError('MP1_EXACT_POSITIVE_RESOURCE_CHARGE')
        with self.transaction() as l:
            if l.get(key,0)+count>CAPS[key]:raise PermissionError('MP1_SOURCE_STAGE_CAP:'+key)
            l[key]=l.get(key,0)+count

    @contextmanager
    def steps(self,kind,maximum,identifier):
        if kind not in ['live_and_recovery_integrations','shadow_integrations'] or type(maximum) is not int or maximum<=0:raise ValueError('MP1_EXACT_PHYSICS_RESERVATION')
        with self.transaction() as l:
            pending=l.setdefault('reservations',{});same=sum(v['maximum'] for v in pending.values() if v['kind']==kind)
            if identifier in pending or l.get(kind,0)+same+maximum>CAPS[kind]:raise PermissionError('MP1_PRESERVE_PRIOR_OR_EXCESS_PHYSICS_INTENT')
            pending[identifier]=dict(kind=kind,maximum=maximum,started_epoch=time.time())
        meter=dict(actual=0)
        try:yield meter
        finally:
            count=meter['actual']
            if type(count) is not int or not 0<=count<=maximum:raise ValueError('MP1_UNKNOWN_ACTUAL_STEP_COST_PRESERVE_INTENT')
            with self.transaction() as l:
                reservation=l['reservations'].pop(identifier)
                if reservation['kind']!=kind:raise ValueError('MP1_PHYSICAL_RESERVATION_CHANGED')
                l[kind]=l.get(kind,0)+count

    def guard(self,*,closeout=False):
        stage=read(self.root/'MP1_STAGE_BINDING.json');a=read(self.root.parent.parent/'MP1_ADOPTION_AND_INCREMENTAL_BUDGET.json')
        if time.time()>=(a['hard_closeout_epoch'] if closeout else a['new_experiment_cutoff_epoch']):raise TimeoutError('MP1_SOURCE_STUDY_CLOCK_CLOSED')
        l=read(self.path);key='Core_CPU_s' if self.role=='core' else 'CPU_s';cap=stage['CPU_core_s'][self.role]
        if l[key]>=cap:raise PermissionError('MP1_SOURCE_STAGE_CPU_CAP')

    def stop_physics(self,reason,detail):
        p=self.root/'SOURCE_PHYSICS_STOP.json'
        if not p.exists():write(p,dict(status='STOP_ALL_NEW_MP1_SOURCE_PHYSICS',reason=reason,detail=detail,observed_epoch=time.time()))
