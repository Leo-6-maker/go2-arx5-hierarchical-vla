"""Finite TRAIN-only Teacher collection with unchanged S1 and full raw audit."""
import os
from pathlib import Path
import resource
import signal
import time
import uuid

import numpy as np

from scripts.pi0292.durable import OwnerLock, read, write
from scripts.pi0294.fd1 import put, sha
from scripts.pi0294.scene_cards import digest, task_card
from .scope import CAPTURE_ROLE, validate


def capture_cards(groups, route):
    if route not in ('COVERAGE_FIRST','CLEARANCE_PAIR'):
        raise PermissionError('PV1_ONLY_ADOPTED_DATA_INTERVENTIONS')
    cards=[]
    for index,group in enumerate(groups):
        if group['split']!='TRAIN' or not group['group_id'].startswith('PI0294/PV1/TRAIN/'):
            raise PermissionError('PV1_NO_EVALUATION_OR_OPENED_CONFIRM_CAPTURE')
        order=[('T1',.004),('T2',.004),('T2',.012)] if route=='CLEARANCE_PAIR' else [('T1',.004),('T2',.004)]
        if route=='CLEARANCE_PAIR' and index%2:
            order=[order[0],order[2],order[1]]
        for task,margin in order:
            card=task_card(group,task)
            card.update(id=card['id']+'-C'+str(round(margin*1000)).zfill(2),nominal_clearance_m=margin,
                origin='PV1_ACTUAL_FEEDBACK_TEACHER_WITH_ORIGINAL_S1')
            cards.append(card)
    if len({c['id'] for c in cards})!=len(cards):
        raise PermissionError('PV1_CAPTURE_IDENTITIES_MUST_BE_UNIQUE')
    return cards


class Meter:
    def __init__(self,root):
        self.root=Path(root)
        self.started=time.process_time()
        self.children=resource.getrusage(resource.RUSAGE_CHILDREN)
        self.m,self.inputs=validate(root,role=CAPTURE_ROLE)
        self.ledger=read(self.root/'RESOURCE_LEDGER.json')
        self.plan=read(self.root/'plans/CAPTURE_PLAN.json')
        parent=self.root.parent.parent
        master_path=parent/'plans/PV1_CAPTURE_MASTER_PLAN.json'
        master=read(master_path)
        static_path=parent/'children/population-01/plans/PV1_DATA_PLAN_AND_SPLIT_SEAL.json'
        static=read(static_path)
        groups=[g for g in static['groups'] if g['split']=='TRAIN']
        indices=self.plan['master_card_indices']
        if (sha(master_path)!=self.m['data']['master_capture_plan_sha256']
                or sha(static_path)!=master['static_plan_sha256']
                or static['status']!='SEALED_STATIC_QUALIFIED_BEFORE_TASK_RESULTS'
                or len(groups)!=96 or master['cards']!=capture_cards(groups,master['route'])
                or len(indices)!=len(set(indices)) or any(type(i) is not int or not 0<=i<len(master['cards']) for i in indices)
                or self.plan['cards']!=[master['cards'][i] for i in indices]):
            raise PermissionError('PV1_FROZEN_MASTER_POPULATION_AND_DISJOINT_INDEX_BINDING')
        self.ticks=self.live_ticks=self.shadow_ticks=self.starts=0
        if (self.ledger['status']!='RESERVED' or self.ledger['unknown_effects']
                or self.ledger['attempted_cells'] or len(self.plan['cards'])<1
                or self.plan['capture_cards_sha256']!=digest(self.plan['cards'])
                or any(c['split']!='TRAIN' or c.get('relay_request_relative') for c in self.plan['cards'])):
            raise PermissionError('PV1_FRESH_EXACT_TRAIN_TEACHER_BATCH_ONLY')
        cpus=self.m['limits']['cpus']
        if len(cpus)!=1 or not set(cpus)<=os.sched_getaffinity(0):
            raise PermissionError('PV1_INDEPENDENT_SINGLE_CORE_WORKER_ASSIGNMENT')
        os.sched_setaffinity(0,set(cpus))
        resource.setrlimit(resource.RLIMIT_CPU,(self.m['limits']['CPU_s'],self.m['limits']['CPU_s']))
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))
        self.ledger['status']='RUNNING'
        self.progress()

    def guard(self):
        if (time.time()>=self.m['limits']['operation_deadline_epoch']
                or time.process_time()-self.started>=self.m['limits']['CPU_s']-30
                or (self.root/'STOP.json').exists() or (self.root.parent.parent/'STOP.json').exists()):
            raise TimeoutError('PV1_CAPTURE_ORIGINAL_BOUND_OR_STOP')

    def start(self):
        self.guard()
        if self.starts>=self.m['limits']['engine_starts']:
            raise PermissionError('PV1_RESERVED_CONSTRUCTOR_OR_CLONE_LIMIT')
        self.starts+=1

    def step(self,engine,original,target,opening,live):
        self.guard()
        if self.ticks>=self.m['limits']['physics_ticks']:
            raise PermissionError('PV1_RESERVED_LIVE_SHADOW_TICKS')
        before=engine.steps
        try:return original(engine,target,opening)
        finally:
            count=engine.steps-before
            if count not in (0,1):
                self.ledger['unknown_effects'].append('NATIVE_TICK_ACCOUNTING_UNCERTAIN')
                raise RuntimeError('PV1_UNKNOWN_NATIVE_STEP_COUNT')
            self.ticks+=count
            if live:self.live_ticks+=count
            else:self.shadow_ticks+=count

    def progress(self,**values):
        self.ledger.update(ticks=self.ticks,live_ticks=self.live_ticks,shadow_ticks=self.shadow_ticks,
            engine_starts=self.starts,worker_CPU_s=time.process_time()-self.started,**values)
        write(self.root/'RESOURCE_LEDGER.json',self.ledger)


def run_cell(meter,card):
    import mujoco
    from scripts.pi0294 import episode,teacher as teachers,s1_preflight,r3_teacher,scene,audit
    from scripts.pi0294.fd2 import adapt_teacher
    from scripts.pi0294.e1.audit import video
    root=meter.root
    folder=root/'episodes'/card['id']
    old_factory=episode.make_teacher
    old_static=teachers.make_teacher
    old_engine=episode.make_engine
    old_scene_engine=scene.make_engine
    old_clone=s1_preflight.clone
    native_model=mujoco.MjModel

    class ModelProxy:
        @staticmethod
        def from_xml_string(*args,**kwargs):
            meter.start()
            return native_model.from_xml_string(*args,**kwargs)

    def clone(engine):
        meter.start()
        return old_clone(engine)

    def engine(c,assets):
        value=old_engine(c,assets)
        cls=type(value)
        original=cls.step
        def step(self,target,opening):
            return meter.step(self,original,target,opening,self is value)
        cls.step=step
        return value

    def ordinary(c,initial,native):
        value=old_static(c,initial,native)
        if c['task']=='T2':value=adapt_teacher(value,c['nominal_clearance_m'])
        elif c['nominal_clearance_m']!=.004:raise PermissionError('PV1_T1_REFERENCE_UNCHANGED')
        return value

    class Protected(r3_teacher.ProtectedTeacher):
        def plan(self,state):
            self.guard()
            value=self.teacher.plan(state)
            self.poses,self.opening=value['poses'][:8],value['opening'][:8]
            self.nonce=str(uuid.uuid4())
            forecast,self.expected=s1_preflight.predict(self.engine,self.poses,self.opening,
                nonce=self.nonce,guard=meter.guard,charge=lambda *a,**k:meter.guard())
            forecast.update(sequence=self.engine.steps//25,stage=value['stage'])
            self.records.append(forecast)
            write(self.folder/'S1_PREFLIGHTS.json',self.records)
            meter.progress(current_cell=card['id'])
            if forecast['decision']!='PASS':raise PermissionError('PRE_EXECUTION_SAFETY_REJECT')
            self.index=0
            self.forecast=forecast
            self.start_tick=self.engine.steps
            if self.negatives is None:self.check_negatives()
            return value

    def protected(c,initial,native):
        return Protected(ordinary(c,initial,native),native,root,folder,{},meter.guard,card['id'])

    reservation=dict(schema='PI0294_LIVE_RESERVATION_V1',fixture=False,id=card['id'],card=card,
        card_sha256=digest(card),source_sha=meter.m['code']['vla_commit'],root=str(root),output=str(folder),
        asset_root=meter.inputs['asset_root'],deadline_epoch=meter.m['limits']['operation_deadline_epoch'],
        cpus=meter.m['limits']['cpus'],reservation=dict(physics_starts=1,physics_steps=16000,rgb_pairs=641,
            cpu_s=3600,memory_bytes=8*1024**3,output_bytes=768*1024**2))
    put(root/'reservations'/(card['id']+'.json'),reservation)
    result=None
    try:
        mujoco.MjModel=ModelProxy
        episode.make_engine=engine
        episode.make_teacher=protected
        teachers.make_teacher=ordinary
        s1_preflight.clone=r3_teacher.clone=clone
        result=episode.run_episode(card,folder,meter.inputs['asset_root'],meter.m['code']['vla_commit'],reservation)
        s1=read(folder/'S1_RESULT.json') if (folder/'S1_RESULT.json').exists() else None
        expected=(result['status']=='VALID_COMPLETE' or 'PRE_EXECUTION_SAFETY_REJECT' in (result.get('error') or '')
            or 'STATIC_QUALIFICATION_FAILED_NO_INTEGRATION' in (result.get('error') or ''))
        if (not expected or (s1 and (s1['live_native_failures'] or s1['maximum_live_shadow_difference']>1e-10))):
            put(root/'STOP.json',dict(reason='NATIVE_SAFETY_OR_CAPTURE_INTEGRITY',cell=card['id'],error=result.get('error')))
        # Original FK/contact/calibration verifier forbids integration internally.
        episode.make_engine=old_engine
        scene.make_engine=old_scene_engine
        if result['status']=='VALID_COMPLETE':
            audited=audit.audit_episode(root,card['id'],meter.inputs['asset_root'])
            audit_file=folder/'AUDIT.json'
            if audited['status']!='PASS_INDEPENDENT_RAW_AUDIT':
                raise ValueError('PV1_INDEPENDENT_RAW_AUDIT_FAILED_PRESERVED')
        else:
            from .negative_audit import audit as negative_audit
            audited=negative_audit(root,card,result,s1)
            audit_file=folder/'PARTIAL_NEGATIVE_AUDIT.json'
        out=root/'evidence/audits'/card['id']
        out.mkdir(parents=True,exist_ok=False)
        movie=video(folder,out/'actual-body-wrist.mp4',result['rgb_pairs'])
        if movie:movie['relative_path']=str((out/'actual-body-wrist.mp4').relative_to(root))
        if s1 and (s1['actions_modified'] or s1['maximum_live_shadow_difference']>1e-10
                or any(not f['live_state_unchanged'] for f in read(folder/'S1_PREFLIGHTS.json'))):
            raise ValueError('PV1_ORIGINAL_S1_NONMUTATION_OR_PARITY')
        receipt=dict(id=card['id'],scene_group=card['group_id'],task=card['task'],split='TRAIN',
            nominal_clearance_m=card['nominal_clearance_m'],status=result['status'],
            task_success=bool(audited['admitted']),audit_status=audited['status'],
            raw_manifest_sha256=audited['manifest_sha256'],audit_sha256=sha(audit_file),
            live_ticks=result['physics_steps'],camera_pairs=result['rgb_pairs'],video=movie,
            S1_rejections=0 if not s1 else s1['pre_execution_rejections'],
            native_faults=0 if not s1 else s1['live_native_failures'],
            initial_state=read(folder/'INITIAL_STATE.json'),
            initial_camera_sha256={c:sha(folder/c/'0000.png') for c in ['body_rgb','wrist_rgb']},
            original_R3_S1_metadata_does_not_supply_PV1_data_authority=True,
            new_PV1_scope_and_independent_audit_required_for_training=True,
            label_semantics='ACTUAL_COMMITTED_EEF10_NOT_FUTURE_ACHIEVED_STATE',GR00T_queries=0)
        put(out/'CAPTURE_AUDIT.json',receipt)
        return receipt
    finally:
        mujoco.MjModel=native_model
        episode.make_engine=old_engine
        episode.make_teacher=old_factory
        teachers.make_teacher=old_static
        scene.make_engine=old_scene_engine
        s1_preflight.clone=r3_teacher.clone=old_clone


def run(root):
    root=Path(root)
    lock=OwnerLock(root/'PV1_CAPTURE_OWNER.lock')
    meter=None
    rows=[]
    failure=None
    def expire(*args):raise TimeoutError('PV1_FINITE_CAPTURE_WORKER_WALL')
    try:
        meter=Meter(root)
        signal.signal(signal.SIGALRM,expire)
        signal.alarm(max(1,int(meter.m['limits']['operation_deadline_epoch']-time.time())))
        for card in meter.plan['cards']:
            meter.guard()
            if (card['id'] in meter.ledger['attempted_cells']
                    or meter.ticks+32000>meter.m['limits']['physics_ticks']
                    or meter.starts+88>meter.m['limits']['engine_starts']):
                raise PermissionError('PV1_ONE_ATTEMPT_WITH_FULL_CELL_RESERVE')
            meter.ledger['attempted_cells'].append(card['id'])
            meter.progress(current_cell=card['id'])
            row=run_cell(meter,card)
            rows.append(row)
            meter.progress(completed_cells=len(rows),current_cell=None)
            if (root/'STOP.json').exists():break
        put(root/'evidence/CAPTURE_RESULT.private.json',dict(status='TERMINAL_CAPTURE_BATCH',rows=rows,
            planned=len(meter.plan['cards']),attempted=len(rows),valid_failures_not_replayed=True))
        return dict(completed=len(rows),planned=len(meter.plan['cards']),successes=sum(r['task_success'] for r in rows))
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
