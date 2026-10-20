"""Bounded CPU workers reuse the original native Teacher, S1 and raw audit."""
import os
from pathlib import Path
import resource
import signal
import time

from scripts.pi0292.durable import OwnerLock,read,write
from scripts.pi0294.fd1 import put,sha
from scripts.pi0294.scene_cards import digest,near_duplicate_report,task_card
from scripts.pi0294.pv1.capture import Meter as OriginalMeter,ensure_legacy_attempt_plan,qualify_counter,run_cell
from .plan import candidate,capture_cards,stage_groups,validate_group
from .scope import dispatch_priority,sealed_file,validate


def rendering_identity():
    from OpenGL.GL import glGetString,GL_RENDERER,GL_VENDOR,GL_VERSION
    values={name:glGetString(code) for name,code in [('renderer',GL_RENDERER),('vendor',GL_VENDOR),('version',GL_VERSION)]}
    if any(v is None for v in values.values()):
        raise PermissionError('DATA_COV1_ACTUAL_RENDERING_BACKEND_UNVERIFIED')
    values={k:v.decode(errors='replace') for k,v in values.items()}
    software=any(v in values['renderer'].lower() for v in ('llvmpipe','softpipe','swrast'))
    if not software:
        raise PermissionError('DATA_COV1_HARDWARE_RENDERER_NEEDS_EXPLICIT_VALID_GRAPHICS_BINDING')
    return dict(values,software_renderer_verified=True,neural_GPU=False,
        driver_context_not_CUDA_model_work=True)


class Meter(OriginalMeter):
    def __init__(self,root):
        self.root=Path(root);self.started=time.process_time();self.children=resource.getrusage(resource.RUSAGE_CHILDREN)
        self.m,self.inputs=validate(root,'CAPTURE')
        self.ledger=read(self.root/'RESOURCE_LEDGER.json');self.plan=read(self.root/'plans/WORK_PLAN.private.json')
        static=read(self.plan['static_plan_path'])
        sealed_file(self.plan['static_plan_path'],self.plan['static_plan_sha256'])
        if static['status']!='SEALED_STATIC_QUALIFIED_BEFORE_TASK_RESULTS':
            raise PermissionError('DATA_COV1_CAPTURE_REQUIRES_ORIGINAL_NATIVE_STATIC_QUALIFICATION')
        population=read(self.root.parent.parent/'plans/DATA_COV1_CANDIDATE_POOL.private.json')
        sealed_file(self.root.parent.parent/'plans/DATA_COV1_CANDIDATE_POOL.private.json',static['candidate_pool_sha256'])
        full=capture_cards(static['groups'],self.m['stage'])
        indexed={c['id']:c for c in full}
        if (self.ledger['status']!='RESERVED' or self.ledger['unknown_effects'] or self.ledger['attempted_cells']
                or not self.plan['cards'] or self.plan['cards_sha256']!=digest(self.plan['cards'])
                or any(indexed.get(c['id'])!=c for c in self.plan['cards'])
                or len({c['id'] for c in self.plan['cards']})!=len(self.plan['cards'])):
            raise PermissionError('DATA_COV1_EXACT_DISJOINT_STATIC_TRAIN_FIRST_ATTEMPTS')
        for group in static['groups']:
            validate_group(group)
            entry=next(s for s in population['slots'] if s['split']=='TRAIN'
                and s['candidates'][0]['group_id']==group['group_id'])
            expected=entry['candidates'][group['layout_candidate_index']]
            if group!=expected:
                raise PermissionError('DATA_COV1_NO_POST_STATIC_CARD_OR_TEACHER_CHANGE')
        self.ticks=self.live_ticks=self.shadow_ticks=self.starts=0
        self.geometry_rows=[]
        self.render_times=[]
        ensure_legacy_attempt_plan(self.root,self.plan['cards'])
        os.sched_setaffinity(0,set(self.m['limits']['cpus']))
        resource.setrlimit(resource.RLIMIT_CPU,(self.m['limits']['CPU_s'],self.m['limits']['CPU_s']))
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))
        self.ledger['status']='RUNNING';self.progress()

    def step(self,engine,original,target,opening,live):
        before=engine.steps
        try:return super().step(engine,original,target,opening,live)
        finally:
            if live and engine.steps-before==1:
                from scripts.pi0294.fd1_shadow import pad_clearance
                gaps=pad_clearance(engine)
                self.geometry_rows.append([float(engine.data.time),
                    *[r['vertical_box_bottom_minus_table_top_m'] for r in gaps],
                    *[int(r['projected_box_overlaps_table']) for r in gaps]])

    def instrument_renderer(self,engine):
        cls=type(engine);original=cls.render
        def render(native,*args,**kwargs):
            began=time.monotonic()
            try:return original(native,*args,**kwargs)
            finally:self.render_times.append([float(native.data.time),time.monotonic()-began])
        cls.render=render


def capture(root):
    root=Path(root);lock=OwnerLock(root/'DATA_COV1_WORKER.lock');meter=None;rows=[];failure=None
    try:
        meter=Meter(root);dispatch_priority(root)
        def expire(*args):raise TimeoutError('DATA_COV1_FINITE_CAPTURE_WALL')
        signal.signal(signal.SIGALRM,expire)
        signal.alarm(max(1,int(meter.m['limits']['operation_deadline_epoch']-time.time())))
        qualify_counter(meter)
        for card in meter.plan['cards']:
            dispatch_priority(root);meter.guard()
            if (card['id'] in meter.ledger['attempted_cells'] or meter.ticks+32000>meter.m['limits']['physics_ticks']
                    or meter.starts+88>meter.m['limits']['engine_starts']):
                raise PermissionError('DATA_COV1_ONE_FIRST_ATTEMPT_FULL_RESERVE_NO_REPLACEMENTS')
            meter.ledger['attempted_cells'].append(card['id']);meter.progress(current_cell=card['id'])
            meter.geometry_rows=[]
            meter.render_times=[]
            try:row=run_cell(meter,card,engine_observer=meter.instrument_renderer)
            finally:
                import numpy as np
                from scripts.pi0294.e1.artifacts import atomic_npz
                path=root/'evidence/geometry'/(card['id']+'.npz');path.parent.mkdir(exist_ok=True)
                atomic_npz(path,rows=np.asarray(meter.geometry_rows,dtype=float).reshape(-1,5))
                path.chmod(0o400)
                put(path.with_suffix('.json'),dict(schema='PI0294_DATA_COV1_ACTUAL_PAD_PLANE_ROWS_V1',
                    id=card['id'],rows=len(meter.geometry_rows),sha256=sha(path),
                    columns=['actual_time_s','left_box_bottom_minus_table_top_m','right_box_bottom_minus_table_top_m',
                        'left_projected_box_overlaps_table','right_projected_box_overlaps_table'],
                    source='READ_ONLY_NATIVE_GEOM_FIELDS_AFTER_ORIGINAL_TICK_NO_EXTRA_FORWARD_OR_STEP',
                    metric_is_vertical_plane_gap_not_full_geometric_distance=True,
                    Teacher_or_S1_or_score_changed=False))
                put(root/'evidence/geometry'/(card['id']+'-render.json'),dict(
                    calls=len(meter.render_times),actual_camera_pairs_per_call=1,
                    duration_sum_s=sum(r[1] for r in meter.render_times),times_and_call_durations=meter.render_times,
                    inclusive_CPU_render_call_wall_time_not_GPU_kernel_time=True,
                    original_Mesa_environment_unchanged=True))
            rows.append(row)
            meter.progress(completed_cells=len(rows),current_cell=None)
            if (root/'STOP.json').exists():break
        put(root/'evidence/CAPTURE_RESULT.private.json',dict(status='TERMINAL_CAPTURE_BATCH',rows=rows,
            planned=len(meter.plan['cards']),attempted=len(rows),valid_failures_not_replayed=True,
            independent_DATA_COV1_scope=True,PV1_manifest_unchanged=True))
        return dict(completed=len(rows),successes=sum(r['task_success'] for r in rows))
    except BaseException as exc:
        failure=type(exc).__name__+': '+str(exc)
        put(root/'evidence/CAPTURE_FAILURE.private.json',dict(error=failure,completed_cells=len(rows)))
        raise
    finally:
        signal.alarm(0)
        if meter is not None:
            child=resource.getrusage(resource.RUSAGE_CHILDREN)
            child_cpu=child.ru_utime+child.ru_stime-meter.children.ru_utime-meter.children.ru_stime
            meter.progress(status='FAILED_PRESERVED' if failure else 'TERMINAL_CAPTURE_BATCH',
                CPU_s=time.process_time()-meter.started+child_cpu,child_CPU_s=child_cpu,CPU_reserved_s=0,
                completed_cells=len(rows),failure=failure)
        lock.close()


def static(root):
    import mujoco
    from scripts.pi0294.scene import make_engine,qualify_engine
    root=Path(root);lock=OwnerLock(root/'DATA_COV1_WORKER.lock')
    m,inputs=validate(root,'STATIC');work=read(root/'plans/WORK_PLAN.private.json')
    pool=read(work['candidate_pool_path']);sealed_file(work['candidate_pool_path'],work['candidate_pool_sha256'])
    exclusions=read(work['exclusions_path']);sealed_file(work['exclusions_path'],work['exclusions_sha256'])
    prior=[] if work.get('prior_static_path') is None else read(work['prior_static_path'])['groups']
    if work.get('prior_static_path'):sealed_file(work['prior_static_path'],work['prior_static_sha256'])
    ledger=read(root/'RESOURCE_LEDGER.json');assert ledger['status']=='RESERVED' and not ledger['unknown_effects']
    accepted=list(prior);rejected=[];receipts=[];starts=0;began=time.process_time();failure=None;graphics=None
    os.sched_setaffinity(0,set(m['limits']['cpus']))
    resource.setrlimit(resource.RLIMIT_CPU,(m['limits']['CPU_s'],m['limits']['CPU_s']))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    def guard(*args):
        if (time.time()>=m['limits']['operation_deadline_epoch'] or time.process_time()-began>=m['limits']['CPU_s']-30
                or (root/'STOP.json').exists() or (root.parent.parent/'STOP.json').exists()):
            raise TimeoutError('DATA_COV1_STATIC_CPU_WALL_OR_STOP')
    def forbidden(*args,**kwargs):raise PermissionError('DATA_COV1_STATIC_ZERO_INTEGRATION_ONLY')
    def expire(*args):raise TimeoutError('DATA_COV1_STATIC_WALL')
    originals=[mujoco.mj_step,mujoco.mj_step1,mujoco.mj_step2]
    try:
        dispatch_priority(root)
        mujoco.mj_step=mujoco.mj_step1=mujoco.mj_step2=forbidden
        signal.signal(signal.SIGALRM,expire);signal.alarm(max(1,int(m['limits']['operation_deadline_epoch']-time.time())))
        selected_slots=[s for s in pool['slots'] if s['split']=='TRAIN' and s['replicate']<dict(P24=2,P72=6,P144=12)[m['stage']]]
        for slot in selected_slots:
            if slot['candidates'][0]['group_id'] in {g['group_id'] for g in accepted}:continue
            dispatch_priority(root)
            for group in slot['candidates']:
                guard();validate_group(group)
                conflicts=near_duplicate_report([group],[*exclusions,*accepted])['conflicts']
                if conflicts:
                    rejected.append(dict(group_id=group['group_id'],candidate_index=group['layout_candidate_index'],
                        reason='GEOMETRIC_NEAR_DUPLICATE',conflicts=conflicts));continue
                pair=[]
                for task in ('T1','T2'):
                    guard()
                    if starts>=m['limits']['engine_starts']:raise PermissionError('DATA_COV1_STATIC_CONSTRUCTOR_BOUND')
                    starts+=1;ledger.update(engine_starts=starts,worker_CPU_s=time.process_time()-began)
                    write(root/'RESOURCE_LEDGER.json',ledger)
                    card=task_card(group,task);engine=None
                    try:
                        engine=make_engine(card,inputs['asset_root']);q=qualify_engine(engine,render=True,guard=guard)
                        if graphics is None:
                            graphics=rendering_identity();put(root/'evidence/ACTUAL_RENDERER_IDENTITY.json',graphics)
                    except ValueError as exc:
                        if not str(exc).startswith('Initial native IK unreachable:'):raise
                        q=dict(status='STATIC_FAIL',reason=str(exc),integration_steps=0,task_success_claim=False,object_pose_modified=False)
                    finally:
                        if engine is not None:engine.close()
                    if q['integration_steps'] or q['task_success_claim'] or q['object_pose_modified']:
                        raise RuntimeError('DATA_COV1_STATIC_QUALIFIER_NOT_PHYSICS_OR_TEACHER_SUCCESS')
                    q.update(scene_group=group['group_id'],task=task,card_sha256=digest(card),candidate_index=group['layout_candidate_index'])
                    p=root/'evidence/static'/('S'+str(len(receipts)).zfill(5)+'.json');p.parent.mkdir(exist_ok=True);put(p,q)
                    record=dict(scene_group=group['group_id'],task=task,status=q['status'],card_sha256=q['card_sha256'],
                        receipt_relative=str(p.relative_to(root)),receipt_sha256=sha(p))
                    receipts.append(record);pair.append(record)
                if all(q['status']=='STATIC_PASS_PENDING_LIVE_CONTACT' for q in pair):
                    accepted.append(group);break
                rejected.append(dict(group_id=group['group_id'],candidate_index=group['layout_candidate_index'],
                    reason='ORIGINAL_STATIC_IK_OR_COLLISION_OR_FOV_FAILURE',receipts=pair))
            else:raise PermissionError('DATA_COV1_STATIC_POOL_EXHAUSTED_NO_NEW_SEEDS')
        stage_groups(accepted,m['stage'])
        result=dict(schema='PI0294_DATA_COV1_STATIC_POPULATION_V1',status='SEALED_STATIC_QUALIFIED_BEFORE_TASK_RESULTS',
            stage=m['stage'],groups=accepted,static_receipts=receipts,rejected_candidates=rejected,
            candidate_pool_sha256=work['candidate_pool_sha256'],exclusions_sha256=work['exclusions_sha256'],
            Teacher_episodes=0,integrations=0,neural_queries=0,renderer=graphics)
        put(root/'evidence/STATIC_RESULT.private.json',result)
        return dict(status=result['status'],groups=len(accepted),engine_starts=starts)
    except BaseException as exc:
        failure=type(exc).__name__+': '+str(exc)
        put(root/'evidence/STATIC_FAILURE.private.json',dict(error=failure,accepted_groups=len(accepted),engine_starts=starts,integrations=0))
        raise
    finally:
        signal.alarm(0);mujoco.mj_step,mujoco.mj_step1,mujoco.mj_step2=originals
        ledger.update(status='FAILED_PRESERVED' if failure else 'TERMINAL_STATIC_PLAN_SEALED',CPU_s=time.process_time()-began,
            CPU_reserved_s=0,engine_starts=starts,episode_starts=0,live_ticks=0,shadow_ticks=0,ticks=0,
            failure=failure,accepted_groups=len(accepted))
        write(root/'RESOURCE_LEDGER.json',ledger);lock.close()
