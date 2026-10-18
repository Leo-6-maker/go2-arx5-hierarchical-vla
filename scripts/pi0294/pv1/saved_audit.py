"""Re-audit terminal saved bytes in a new view; original failed audits stay intact."""
import os
from pathlib import Path
import resource
import signal
import time
from types import SimpleNamespace

from scripts.pi0292.durable import OwnerLock, read, write
from scripts.pi0294.fd1 import put, sha
from .capture import counted_model,ensure_legacy_attempt_plan
from .scope import REAUDIT_ROLE,validate


def raw_view(source,destination,expected_manifest):
    source,destination=Path(source),Path(destination)
    if sha(source/'FILE_MANIFEST.json')!=expected_manifest:
        raise ValueError('PV1_ORIGINAL_RAW_MANIFEST_CHANGED')
    if destination.exists():raise FileExistsError('PV1_SAVED_AUDIT_VIEW_ONCE_ONLY')
    destination.mkdir(parents=True)
    members=read(source/'FILE_MANIFEST.json')
    if any(r['path'] in ['AUDIT.json','READY.json'] for r in members):
        raise PermissionError('PV1_DERIVED_AUDIT_OUTPUT_CANNOT_ALIAS_ORIGINAL')
    for row in members:
        p=source/row['path']
        if (p.is_symlink() or not p.resolve().is_relative_to(source.resolve())
                or p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']):
            raise ValueError('PV1_ORIGINAL_RAW_MEMBER_CHANGED_OR_ESCAPED')
        target=destination/row['path']
        target.parent.mkdir(parents=True,exist_ok=True)
        os.link(p,target)
    for name in ['FILE_MANIFEST.json','WORKER_RECEIPT.json']:
        if not (destination/name).exists():os.link(source/name,destination/name)
    return dict(original_raw_manifest_sha256=expected_manifest,members=len(members),
        files_copied=0,immutable_terminal_raw_hardlink_view=True,
        original_derived_audit_not_linked=True)


def run(root):
    import mujoco
    from scripts.pi0294.audit import audit_episode
    from scripts.pi0294.e1.audit import video
    root=Path(root)
    lock=OwnerLock(root/'PV1_SAVED_AUDIT_OWNER.lock')
    m,inputs=validate(root,role=REAUDIT_ROLE)
    ledger=read(root/'RESOURCE_LEDGER.json')
    plan=read(root/'plans/SAVED_AUDIT_PLAN.private.json')
    if (ledger['status']!='RESERVED' or ledger['CPU_reserved_s']!=900
            or ledger['unknown_effects'] or len(plan['cases'])!=2
            or any(c['task']!='T1' for c in plan['cases'])):
        raise PermissionError('PV1_EXACT_TWO_SAVED_T1_AUDITS_FOUR_CONSTRUCTORS_ZERO_INTEGRATION')
    started=time.process_time()
    child=resource.getrusage(resource.RUSAGE_CHILDREN)
    rows=[]
    failure=None
    starts=[]
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_CPU,(900,900))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    def expire(*a):raise TimeoutError('PV1_SAVED_AUDIT_FINITE_TIME')
    def forbidden(*a,**k):raise PermissionError('PV1_SAVED_AUDIT_FORBIDS_ALL_INTEGRATION')
    def start():
        if len(starts)>=4 or time.time()>=m['limits']['operation_deadline_epoch']:
            raise PermissionError('PV1_SAVED_AUDIT_CONSTRUCTOR_OR_TIME_CAP')
        starts.append(1)
        ledger.update(engine_starts=len(starts),status='RUNNING')
        write(root/'RESOURCE_LEDGER.json',ledger)
    original_model=mujoco.MjModel
    original_steps=[mujoco.mj_step,mujoco.mj_step1,mujoco.mj_step2]
    try:
        signal.signal(signal.SIGALRM,expire)
        signal.alarm(max(1,int(m['limits']['operation_deadline_epoch']-time.time())))
        mujoco.MjModel=counted_model(original_model,SimpleNamespace(start=start))
        mujoco.mj_step=mujoco.mj_step1=mujoco.mj_step2=forbidden
        cards=[]
        for case in plan['cases']:
            source=Path(case['source_episode'])
            card=read(source/'CARD.json')
            if card['id']!=case['id'] or card['task']!='T1' or sha(source/'AUDIT.json')!=case['original_failed_audit_sha256']:
                raise ValueError('PV1_ORIGINAL_ATTEMPT_AND_FAILED_AUDIT_IDENTITY')
            binding=read(source.parent.parent/'OUTPUT_BINDING.private.json')
            if read(binding['lease'])['status']!='CLOSED':
                raise PermissionError('PV1_TERMINAL_CAPTURE_LEASE_MUST_BE_CLOSED_BEFORE_VIEW')
            raw_view(source,root/'episodes'/card['id'],case['raw_manifest_sha256'])
            ticket=read(source.parent.parent/'reservations'/(card['id']+'.json'))
            ticket.update(root=str(root),output=str(root/'episodes'/card['id']))
            put(root/'reservations'/(card['id']+'.json'),ticket)
            cards.append(card)
        ensure_legacy_attempt_plan(root,cards)
        for case,card in zip(plan['cases'],cards):
            source=Path(case['source_episode'])
            s1=read(source/'S1_RESULT.json')
            forecasts=read(source/'S1_PREFLIGHTS.json')
            if (s1['actions_modified'] or s1['live_native_failures'] or s1['pre_execution_rejections']
                    or s1['maximum_live_shadow_difference']>1e-10 or len(forecasts)!=80
                    or any(not p['live_state_unchanged'] or p['decision']!='PASS' for p in forecasts)
                    or set(s1['negatives']['rejected'])!={'nonce','action','state','decision'}):
                raise ValueError('PV1_ORIGINAL_S1_FULL_TASK_EVIDENCE_UNQUALIFIED')
            q=audit_episode(root,card['id'],inputs['asset_root'])
            if q['status']!='PASS_INDEPENDENT_RAW_AUDIT':
                raise ValueError('PV1_SAVED_AUDIT_STILL_UNQUALIFIED_NO_PHYSICS_RETRY')
            folder=root/'episodes'/card['id']
            out=root/'evidence/audits'/card['id']
            out.mkdir(parents=True,exist_ok=False)
            result=read(folder/'RESULT.json')
            movie=video(folder,out/'actual-body-wrist.mp4',result['rgb_pairs'])
            movie['relative_path']=str((out/'actual-body-wrist.mp4').relative_to(root))
            if sha(source/'AUDIT.json')!=case['original_failed_audit_sha256'] or sha(source/'FILE_MANIFEST.json')!=case['raw_manifest_sha256']:
                raise ValueError('PV1_ORIGINAL_FAILED_AUDIT_OR_RAW_MODIFIED_BY_VIEW')
            row=dict(id=card['id'],task='T1',scene_group=card['group_id'],audit_status=q['status'],
                task_success=q['admitted'],new_episodes=0,integrations=0,GR00T_queries=0,
                original_raw_manifest_sha256=case['raw_manifest_sha256'],original_failed_audit_preserved=True,
                original_physics_source=result['source_sha'],saved_audit_source=m['code']['vla_commit'],
                full_audit_sha256=sha(folder/'AUDIT.json'),video=movie,
                original_score=result['score'],independent_score=q['detail']['independent_score'])
            put(out/'SAVED_AUDIT_RESULT.json',row)
            rows.append(row)
        put(root/'evidence/SAVED_AUDIT_RESULT.private.json',dict(status='PASS_TWO_ORIGINAL_SAVED_ATTEMPTS_INDEPENDENTLY_AUDITED',
            rows=rows,integrations=0,new_episodes=0,original_failed_audits_preserved=True))
        return dict(status='PASS_TWO_ORIGINAL_SAVED_ATTEMPTS_INDEPENDENTLY_AUDITED',rows=len(rows),new_episodes=0)
    except BaseException as exc:
        failure=type(exc).__name__+': '+str(exc)
        put(root/'evidence/FAILURE.private.json',dict(error=failure,new_physics=0,engine_starts=len(starts)))
        raise
    finally:
        signal.alarm(0)
        mujoco.MjModel=original_model
        mujoco.mj_step,mujoco.mj_step1,mujoco.mj_step2=original_steps
        after=resource.getrusage(resource.RUSAGE_CHILDREN)
        ledger.update(status='FAILED_PRESERVED' if failure else 'TERMINAL_SAVED_AUDIT_PASS',CPU_reserved_s=0,
            CPU_s=time.process_time()-started+after.ru_utime+after.ru_stime-child.ru_utime-child.ru_stime,
            engine_starts=len(starts),new_episodes=0,integrations=0,failure=failure)
        write(root/'RESOURCE_LEDGER.json',ledger)
        lock.close()
