"""The adopted R4 run boundary; frozen R3 execution is never redirected."""
import contextlib
import os
from pathlib import Path
import resource
import socket
import time

from scripts.pi0292.durable import OwnerLock,read,write
from .continuation import within,NUMERICAL_PATHS
from .native_dataset import sha

RUN='20261014T170930Z-incoming-pi0294-r4'
AUTHORITY='ea5dee757e4e3f88f50eee0ce27a5dfefe175f93'
AUTHORITY_SHA='91e94578f7049d9031bdf773a427119fdb8aa318c8c6ec8ec26a0934096e75ec'
GO_COMMENT=6173988141
T0,CUTOFF,DEADLINE,AUG_FREEZE=1791997770,1792037370,1792040970,1792008570


def validate_mode(root,request):
    root=Path(root)
    if request['stage']=='main':
        expected=Path(read(root/'PARENT_BINDING.private.json')['trained_checkpoint'])
        if (request['mode']!='main' or request['allowed_modes']!=['main']
                or request.get('arm')!='C0_V30_M4096' or request['max_steps']!=4096
                or request['stop_at']!=4096 or Path(request['resume'])!=expected):
            raise PermissionError('R4_ONLY_FROZEN_C0_FROM_COMPLETE_M2048')
        proof=read(root/'evidence/R0_MODEL_BEHAVIOR_GATE.json')
        if (proof['status']!='COMPLETE_ACTUAL_M2048_TRAIN4_DEV8_INPUT_CONTRACT_VALID'
                or proof['M2048_TRAIN_attempts']!=4 or proof['M2048_DEV_attempts']!=8):
            raise PermissionError('R4_ACTUAL_M2048_PANEL_BEFORE_TRAINING')
        ready=within(root/request['local_M2048_ready_relative'],root)
        if sha(ready)!=request['local_M2048_ready_sha256']:
            raise PermissionError('R4_FULL_M2048_LOCAL_INVENTORY_CHANGED')
        value=read(ready)
        if (value['status']!='LOCAL_CHECKPOINT_READY' or value['global_step']!=2048
                or value['consumed_samples']!=65536 or not value['full_member_SHA_verified']
                or Path(value['source_directory'])!=expected):
            raise PermissionError('R4_COMPLETE_ORIGINAL_M2048_TRAINER_REQUIRED')
        acceleration=read(root/'evidence/DATALOADER_ACCELERATION.json')
        if acceleration['selected_workers']!=request['workers'] or acceleration['selected_workers']!=0:
            raise PermissionError('R4_CURRENT_C0_NATIVE_ZERO_WORKER_PROFILE_ONLY')
    elif request['stage']=='resident':
        path=within(root/request['S1_qualification_relative'],root)
        proof=read(path)
        if (sha(path)!=request['S1_qualification_sha256'] or proof['status']!='S1_TEACHER_QUALIFIED'
                or proof['teacher_complete_successes']!=4 or proof['teacher_rejects']
                or proof['live_native_failures'] or proof['clone_contexts_passed']<8):
            raise PermissionError('R4_ORIGINAL_ACTUAL_S1_QUALIFICATION_REQUIRED')
        if (root/'evidence/S1_REVOKED.json').exists():raise PermissionError('R4_S1_REVOKED')
        if not 0<request['max_queries']<=6000:raise PermissionError('R4_GLOBAL_GENERATION_CAP')
    else:raise PermissionError('R4_FIXED_MAIN_OR_RESIDENT_ONLY')


def validate_entry(root,*,profile,manifest_path=None,now=None,closeout=False,hostname=None,uid=None,verify_source=True):
    root=Path(root);now=time.time() if now is None else now
    authority=within(root/'AUTHORITY_ADOPTION.json',root)
    if sha(authority)!=AUTHORITY_SHA:raise PermissionError('EXACT_R4_HUMAN_ADOPTION_REQUIRED')
    a=read(authority);m=read(within(manifest_path or root/'RUN_MANIFEST.json',root))
    if (root.name!=RUN or m['run_uuid']!=RUN or m['pi']!='PI0294' or m['owner']!='EXISTING_LOCAL_PC'
            or m['authority']['protocol_ref']!=AUTHORITY or m['authority']['go_comment']!=GO_COMMENT
            or m['limits']!=a['limits'] or not a['authorizes_execution']
            or (a['T0_epoch'],a['new_work_cutoff_epoch'],a['deadline_epoch'],a['AUG_FREEZE_epoch'])!=(T0,CUTOFF,DEADLINE,AUG_FREEZE)
            or not T0<=now<(DEADLINE if closeout else CUTOFF)):
        raise PermissionError('R4_AUTHORITY_OR_IMMUTABLE_CLOCK_MISMATCH')
    host=socket.gethostname() if hostname is None else hostname;uid=os.getuid() if uid is None else uid
    e=m['execution'];native=(profile=='hopper-neural' and os.environ.get('PBS_JOBID') and host.startswith('hopper-') and host.split('.')[0].removeprefix('hopper-').isdigit())
    if e['profile']!=profile or e['uid']!=uid or not(host in e['hostnames'] or native):raise PermissionError('R4_EXACT_HOST_PROFILE_UID')
    binding=within(root/'OUTPUT_BINDING.private.json',root);b=read(binding)
    if sha(binding)!=e['output_binding_sha256'] or Path(b['run_root']).resolve()!=root.resolve() or b['run_uuid']!=RUN:
        raise PermissionError('R4_OUTPUT_BINDING_CHANGED')
    within(root,Path(b['registered_parent']))
    registry,lease=read(b['registry']),read(b['lease'])
    if not any(r['role']==b['role'] and Path(r['path']).resolve()==root.resolve() for r in registry['entries']):raise PermissionError('R4_ROLE_NOT_REGISTERED')
    if (lease['status']!='ACTIVE' or lease['run_uuid']!=RUN or lease['protocol_ref']!=AUTHORITY or lease['deadline_epoch']!=DEADLINE):
        raise PermissionError('R4_ACTIVE_OWNER_LEASE_REQUIRED')
    if verify_source:
        code=within(root/m['code']['capsule_relative'],root);seal_path=within(root/m['code']['source_seal_relative'],root);seal=read(seal_path)
        if (Path(__file__).resolve()!=(code/'scripts/pi0294/r4.py').resolve() or sha(seal_path)!=m['code']['source_seal_sha256']
                or seal['source_sha']!=m['code']['vla_commit'] or seal['protocol_release']!=AUTHORITY):raise PermissionError('R4_SEALED_SOURCE_REQUIRED')
        for name,digest in seal['files'].items():
            if sha(within(code/name,code))!=digest:raise PermissionError('R4_SOURCE_MEMBER_CHANGED:'+name)
        if not set(NUMERICAL_PATHS)<=set(m['code']['unchanged_numerical_files']):raise PermissionError('R4_NUMERICAL_CORE_PROOF_MISSING')
        for name,digest in m['code']['unchanged_numerical_files'].items():
            if seal['files'][name]!=digest:raise PermissionError('R4_FROZEN_NUMERICAL_CORE_CHANGED:'+name)
    return m


def validate_cpu_admission(manifest,admission,plan,workers,now):
    if (manifest['run_uuid']!=RUN or (admission['T0_epoch'],admission['cutoff_epoch'],admission['deadline_epoch'])!=(T0,CUTOFF,DEADLINE)
            or admission['lease_expires_epoch']!=DEADLINE or not T0<=now<CUTOFF):raise PermissionError('R4_EXACT_COLLECTION_CLOCK')
    caps=dict(physics_starts=160,physics_steps=2560000,rgb_pairs=102560,cpu_s=96*3600,output_bytes=96*1024**3)
    if workers not in (1,2,4) or len(admission['allowed_cpus'])>16 or any(admission['limits'][k]>v for k,v in caps.items()):
        raise PermissionError('R4_COLLECTION_RESOURCE_LIMIT')
    if plan.get('kind') not in ('R4_CONTACT_PILOT','R4_CONTACT_TRAIN','R4_MODEL_EVALUATION'):raise PermissionError('R4_FIXED_AUTHORIZED_LANE_ONLY')
    expected=4 if plan['kind']=='R4_CONTACT_PILOT' else 48 if plan['kind']=='R4_CONTACT_TRAIN' else None
    if expected is not None and len(plan['episodes'])!=expected:raise PermissionError('R4_EXACT_FIXED_COLLECTION_COUNT')
    root=Path(admission['root']);budget=read(root/'R4_RESOURCE_LEDGER.json')
    other=[r for r in budget['CPU_reservations'].values() if not(r.get('kind')=='COLLECTION' and r['pid']==os.getpid())]
    reserved={c for r in other for c in r['cpus']}
    if reserved&set(admission['allowed_cpus']):raise PermissionError('R4_CPU_TICKETS_MUST_BE_DISJOINT')
    if len(admission['allowed_cpus'])+len(reserved)>16:raise PermissionError('R4_UNION_CPU_CAP')
    outside_collection=budget['Core_CPU_s']-budget.get('collection_CPU_s',0.)
    if admission['limits']['cpu_s']+outside_collection+sum(r['cpu_s_cap'] for r in other)>96*3600:
        raise PermissionError('R4_ALL_LANES_SHARE_CPU_BUDGET')


@contextlib.contextmanager
def cpu_stage(root,stage,*,cores=1,cpu_s_cap=7200):
    """One bounded CPU ticket and actual cost; no hidden physics/model budget."""
    root=Path(root);m=validate_entry(root,profile='4080-simulation');start=time.process_time()
    children=resource.getrusage(resource.RUSAGE_CHILDREN)
    affinity=sorted(os.sched_getaffinity(0))[-16:]
    with_lock=OwnerLock(root/'R4_RESOURCE_OWNER.lock');p=root/'R4_RESOURCE_LEDGER.json';ledger=read(p)
    active=ledger['CPU_reservations']
    if stage in active or any(s['stage']==stage for s in ledger['CPU_stages']):raise FileExistsError('R4_CPU_STAGE_ALREADY_ATTEMPTED')
    if not 1<=cores<=4 or sum(r['cores'] for r in active.values())+cores>16 or ledger['Core_CPU_s']+sum(r['cpu_s_cap'] for r in active.values())+cpu_s_cap>96*3600:
        raise PermissionError('R4_CPU_RESERVATION_CAP')
    used={c for r in active.values() for c in r['cpus']};cpus=[c for c in affinity if c not in used][:cores]
    if len(cpus)!=cores:raise PermissionError('R4_ALLOWED_CPU_CAPACITY')
    ticket=dict(stage=stage,pid=os.getpid(),proc_start=Path('/proc/self/stat').read_text().split()[21],cores=cores,cpus=cpus,cpu_s_cap=cpu_s_cap,started_epoch=time.time())
    active[stage]=ticket;write(p,ledger);with_lock.close();os.sched_setaffinity(0,set(cpus));status='FAILED_PRESERVED'
    soft,hard=resource.getrlimit(resource.RLIMIT_CPU)
    limit=int(time.process_time()+cpu_s_cap)
    resource.setrlimit(resource.RLIMIT_CPU,(min(limit,hard) if hard!=resource.RLIM_INFINITY else limit,hard))
    write(root/'evidence'/f'{stage}-CPU_TICKET.json',ticket)
    try:
        yield m
        status='COMPLETE'
    finally:
        used=resource.getrusage(resource.RUSAGE_CHILDREN)
        cost=time.process_time()-start+used.ru_utime+used.ru_stime-children.ru_utime-children.ru_stime
        owner=OwnerLock(root/'R4_RESOURCE_OWNER.lock');ledger=read(p)
        ledger['CPU_reservations'].pop(stage);ledger['Core_CPU_s']+=cost
        receipt=dict(ticket,status=status,finished_epoch=time.time(),actual_CPU_s=cost,new_neural_queries=0,new_physics=0)
        ledger['CPU_stages'].append(receipt);write(p,ledger);write(root/'evidence'/f'{stage}-CPU_RECEIPT.json',receipt);owner.close()
