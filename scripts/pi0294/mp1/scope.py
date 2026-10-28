"""MP1's separate authority and clock; registration alone never admits science."""
import fcntl
import os
from pathlib import Path
import socket
import time

from scripts.pi0292.durable import read,write
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.e1.scope import DATA,STATS

AUTHORITY='c8cdf60e41e31da5796b44a064ef2f22d594a491'
CARD_SHA='6b1b0ed8aa96a1ee0188bd9a95c5eb2cca273787e6ce5f928a457334ddf629f6'


def validate(root,profile,*,kind='cpu-data',now=None):
    root=Path(root);now=time.time() if now is None else now
    a=read(root/'MP1_ADOPTION_AND_INCREMENTAL_BUDGET.json');p=root/'evidence/MP1_TASK_CARD.json'
    if sha(p)!=CARD_SHA:raise PermissionError('MP1_FIXED_ADOPTED_TASK_CARD')
    card=read(p)
    if (a['authority_release']!=AUTHORITY or a['machine_card_sha256']!=CARD_SHA or a['run_uuid']!=root.name
            or a['owner']!='EXISTING_SINGLE_LOCAL_PC' or a['E2_additive_budget'] is not False or a['E2_separate_T0'] is not False
            or a['reset_on_resume'] is not False or a['incremental_budget']!=card['incremental_budget']
            or a['new_submission_and_candidate_cutoff_epoch']!=a['T0_epoch']+144*3600
            or a['new_experiment_cutoff_epoch']!=a['T0_epoch']+160*3600
            or a['hard_closeout_epoch']!=a['T0_epoch']+168*3600 or not a['T0_epoch']<=now<a['hard_closeout_epoch']):
        raise PermissionError('MP1_ONE_BUDGET_OWNER_AND_CLOCK')
    if kind!='cpu-data':raise PermissionError('MP1_SCIENTIFIC_STAGE_REQUIRES_SEPARATE_NATIVE_ADMISSION')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('MP1_CPU_PREPARATION_WITHOUT_VISIBLE_CUDA')
    manifest=Path(os.environ['MP1_RUN_MANIFEST']);m=read(manifest);binding=read(root/'OUTPUT_BINDING.private.json')
    code=root/m['code']['capsule_relative'];lease=read(binding['lease'])
    if (m['run_uuid']!=root.name or m['authority']['protocol_ref']!=AUTHORITY
            or m['execution']['profile']!=profile or m['execution']['uid']!=os.getuid()
            or socket.gethostname() not in m['execution']['hostnames'] or root.is_symlink()
            or not root.resolve().is_relative_to(Path(binding['registered_parent']).resolve())
            or Path(binding['run_root']).resolve()!=root.resolve()
            or m['execution']['output_binding_sha256']!=sha(root/'OUTPUT_BINDING.private.json')
            or lease['status']!='ACTIVE_CONTROL_PLANE_ONLY' or lease['run_uuid']!=root.name
            or lease['authority_release']!=AUTHORITY or lease['deadline_epoch']!=a['hard_closeout_epoch']
            or manifest.resolve()!=(code.parent/'WORKER_RUN_MANIFEST.json').resolve() or manifest.stat().st_mode&0o222):
        raise PermissionError('MP1_CPU_HOST_OWNER_IMMUTABLE_MANIFEST_AND_CONTROL_LEASE')
    registry=read(binding['registry'])
    if not any(x.get('run_uuid')==root.name and x['role']==binding['role'] and Path(x['path']).resolve()==root.resolve() for x in registry['entries']):
        raise PermissionError('MP1_REGISTERED_OUTPUT_REQUIRED')
    seal=root/m['code']['source_seal_relative']
    if sha(seal)!=m['code']['source_seal_sha256'] or Path(__file__).resolve()!=(code/'scripts/pi0294/mp1/scope.py').resolve():raise PermissionError('MP1_ACTUAL_FIXED_CAPSULE')
    s=read(seal)
    if s['source_sha']!=m['code']['vla_commit']:raise PermissionError('MP1_SEALED_SOURCE_IDENTITY')
    for name,digest in s['files'].items():
        p=code/name
        if not p.resolve().is_relative_to(code.resolve()) or p.is_symlink() or sha(p)!=digest:raise PermissionError('MP1_SOURCE_CHANGED:'+name)
    if any(s['files'].get(name)!=digest for name,digest in m['code']['unchanged_numerical_files'].items()):raise PermissionError('MP1_FROZEN_NUMERICAL_SOURCE_CHANGED')
    return m


def charge_cpu(root,cost):
    if not 0<=cost<float('inf'):raise ValueError('MP1_FINITE_POSITIVE_CPU_COST')
    p=Path(root)/'HOST_CPU_USED.json'
    with p.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);v=read(p);v['CPU_s']+=cost;write(p,v)
