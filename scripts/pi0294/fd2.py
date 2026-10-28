"""Finite paired Teacher pilot. Import performs no native/model work."""
import copy
import json
import os
from pathlib import Path
import resource
import time

import numpy as np

from scripts.pi0292.durable import OwnerLock, read, write
from scripts.pi0294.fd1 import put, sha
from scripts.pi0294.scene_cards import digest, task_card, validate_groups

SCOPE = 'reports/PI0294/design/FD2_DYNAMIC_EXPERT_CLEARANCE_SCOPE_V1.json'
PROTOCOL = 'docs/protocols/PI0294_FD2_DYNAMIC_EXPERT_CLEARANCE_PILOT_V1.md'
ROLE = 'PI0294_FD2_DYNAMIC_EXPERT_PILOT'
BASELINE = 'ba40c713b483a02be5387b5254d045174dc6493b'


def pilot_plan(exclusions):
    """Exact geometry, no outcome/static/FOV-dependent candidate search."""
    visual = dict(table_rgba=[.69,.70,.66,1.], floor_rgba=[.72,.74,.76,1.], light_scale=1.,
        body_camera_translation=[0.,0.,0.], body_camera_rpy=[0.,0.,0.],
        wrist_camera_translation=[0.,0.,0.], wrist_camera_rpy=[0.,0.,0.])
    groups, cards = [], []
    for i,(x,center,gap,sign,yaw) in enumerate([
            (.438,-.004,.174,1,.037), (.466,.006,.178,1,-.051),
            (.440,.009,.182,-1,.061), (.464,-.007,.168,-1,-.029)]):
        group=dict(group_id=f'PI0294/FD2/TRAIN/G{i}', scene_group=f'PI0294/FD2/TRAIN/G{i}',
            split='TRAIN', family='FD2_FIXED', seed=2940200+i,
            source_xy=[x,center-sign*gap/2], goal_xy=[x,center+sign*gap/2], cube_yaw=yaw,
            table_top_m=.44, visual=copy.deepcopy(visual), direction='+Y' if sign==1 else '-Y',
            qualification='NOT_STATIC_OR_CONTACT_QUALIFIED')
        groups.append(group)
        for margin in ([.004,.012] if i%2==0 else [.012,.004]):
            card=task_card(group,'T2')
            card.update(id=f'PI0294-FD2-G{i}-C{round(margin*1000):02d}', nominal_clearance_m=margin)
            cards.append(card)
    report=validate_groups(groups, exclusions)
    return dict(schema='PI0294_FD2_FIXED_PAIRED_PLAN_V1', groups=groups, cards=cards,
        exclusions_sha256=digest(exclusions), exclusion_count=len(exclusions), uniqueness=report,
        denominator=8, replacements=0, training_authorized=False)


def adapt_teacher(teacher, margin):
    if margin not in (.004,.012):
        raise PermissionError('FD2_ONLY_TWO_PREREGISTERED_REFERENCE_CLEARANCES')
    if margin == .012:
        teacher.source_tcp[2] += .008
        teacher.goal_tcp[2] += .008
        teacher.geometry=dict(teacher.geometry, tcp_height_m=teacher.geometry['tcp_height_m']+.008)
    return teacher


def validate(root):
    root=Path(root)
    m=read(root/'RUN_MANIFEST.json');b=read(root/'OUTPUT_BINDING.private.json')
    inputs=read(root/'INPUT_BINDINGS.private.json')
    code=root/m['code']['capsule_relative'];seal=root/m['code']['source_seal_relative']
    if (m['schema']!='PI0294_FD2_RUN_MANIFEST_V1' or m['scope']!=ROLE
            or m['owner']!='EXISTING_LOCAL_PC' or m['run_uuid']!=root.name
            or m['execution']['host']!=os.uname().nodename or m['execution']['uid']!=os.getuid()
            or m['execution']['profile']!='4080-simulation'
            or os.environ.get('FD2_RUN_MANIFEST')!=str(code.parent/'WORKER_RUN_MANIFEST.json')
            or read(code.parent/'WORKER_RUN_MANIFEST.json')!=m):
        raise PermissionError('FD2_EXACT_NATIVE_OWNER_HOST_WORKER_MANIFEST')
    if (root.is_symlink() or root.resolve()!=Path(b['run_root']).resolve()
            or root.parent.resolve()!=Path(b['registered_parent']).resolve()
            or b['role']!=ROLE or b['uid']!=os.getuid()
            or sha(root/'OUTPUT_BINDING.private.json')!=m['execution']['output_binding_sha256']
            or sha(root/'INPUT_BINDINGS.private.json')!=m['execution']['input_binding_sha256']):
        raise PermissionError('FD2_EXACT_REGISTERED_INPUT_OUTPUT')
    spec=read(code/SCOPE)
    if (spec['baseline_fd1_release']!=BASELINE or m['limits']['caps']!=spec['limits']
            or sha(code/SCOPE)!=m['authority']['scope_sha256']
            or sha(code/PROTOCOL)!=m['authority']['protocol_sha256']
            or m['limits']['hard_close_epoch']!=m['limits']['T0_epoch']+21600
            or m['limits']['stop_new_cells_epoch']!=m['limits']['T0_epoch']+18000):
        raise PermissionError('FD2_EXACT_FRESH_BOUNDED_AUTHORITY')
    lease=read(b['lease'])
    if (lease['status']!='ACTIVE' or lease['run_uuid']!=root.name
            or lease['authority_release']!=m['authority']['protocol_ref']
            or lease['deadline_epoch']!=m['limits']['hard_close_epoch']
            or not m['limits']['T0_epoch']<=time.time()<m['limits']['hard_close_epoch']):
        raise PermissionError('FD2_FRESH_LEASE_ORIGINAL_CLOCK')
    matches=[x for x in read(b['registry'])['entries'] if x.get('role')==ROLE]
    if len(matches)!=1 or Path(matches[0]['path']).resolve()!=root.resolve():
        raise PermissionError('FD2_UNIQUE_ROLE_AND_OWNER')
    if sha(seal)!=m['code']['source_seal_sha256']:
        raise ValueError('FD2_SOURCE_SEAL')
    s=read(seal)
    if s['source_sha']!=m['code']['vla_commit'] or any(sha(code/n)!=h for n,h in s['files'].items()):
        raise ValueError('FD2_COLD_SOURCE_MEMBERS')
    if Path(__file__).resolve()!=(code/'scripts/pi0294/fd2.py').resolve():
        raise ValueError('FD2_REAL_IMPORTED_SOURCE')
    for n,h in m['code']['frozen_numerical_files'].items():
        if sha(code/n)!=h:
            raise ValueError('FD2_FROZEN_NUMERICAL_HASH')
    pre=read(root/'plans/ADMISSION.private.json')
    for x in pre['leases']:
        if sha(x['path'])!=x['sha256']:
            raise ValueError('FD2_HISTORICAL_LEASE_CHANGED')
    if sha(Path(pre['fd1_root'])/'RESOURCE_LEDGER.json')!=pre['fd1_ledger_sha256']:
        raise ValueError('FD2_CLOSED_FD1_CHANGED')
    if sha(root/'plans/PILOT_PLAN.json')!=m['data']['manifest_sha256']:
        raise ValueError('FD2_FIXED_EIGHT_CELL_PLAN')
    infra=Path(b['registry']).parent
    if any(sha(infra/n)!=m['execution'][k] for n,k in [('enter.sh','entry_sha256'),('doctor.py','doctor_sha256')]):
        raise ValueError('FD2_APPROVED_ENTRY_IDENTITY')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or 'PBS_JOBID' in os.environ:
        raise PermissionError('FD2_NO_GPU_NO_PBS')
    assets=read(root/'plans/ASSET_IDENTITY.private.json')
    if len(assets['absolute_files'])!=26 or any(sha(p)!=h for p,h in assets['absolute_files'].items()):
        raise ValueError('FD2_ORIGINAL_26_ASSETS_CHANGED')
    return m,inputs


class Meter:
    def __init__(self,root):
        self.root=Path(root);self.m,self.inputs=validate(root)
        self.path=self.root/'RESOURCE_LEDGER.json';self.ledger=read(self.path)
        self.started=time.process_time();self.children=resource.getrusage(resource.RUSAGE_CHILDREN)
        self.ticks=self.starts=self.live_ticks=self.shadow_ticks=0
        if (self.ledger['status']!='RESERVED' or self.ledger['unknown_effects']
                or self.ledger['operations']!=['pilot']):
            raise PermissionError('FD2_SINGLE_RESERVED_OPERATION_ONLY')
        self.ledger['status']='RUNNING';write(self.path,self.ledger)
        self.cpu_cap=self.ledger['CPU_reserved_s']
        os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
        resource.setrlimit(resource.RLIMIT_CPU,(self.cpu_cap,self.cpu_cap))
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))

    def guard(self,new_cell=False):
        elapsed=time.process_time()-self.started
        if (time.time()>=self.m['limits']['hard_close_epoch'] or elapsed>=self.cpu_cap
                or (new_cell and time.time()>=self.m['limits']['stop_new_cells_epoch'])):
            raise TimeoutError('FD2_ORIGINAL_TIME_CPU_BOUND')
        if self.ticks>300000 or self.starts>700:
            raise PermissionError('FD2_PHYSICS_OR_START_CAP')
        if (self.root/'STOP.json').exists():
            raise PermissionError('FD2_NEW_SCOPE_STOP_RETAINED')

    def start(self):
        self.guard()
        if self.starts>=700:
            raise PermissionError('FD2_START_CAP')
        self.starts+=1

    def step(self,engine,original,target,opening,live):
        self.guard()
        if self.ticks>=300000:
            raise PermissionError('FD2_ALL_TICKS_CAP')
        before=engine.steps
        try:
            return original(engine,target,opening)
        finally:
            count=engine.steps-before
            if count not in (0,1):
                self.ledger['unknown_effects'].append('NATIVE_STEP_COUNT')
                raise RuntimeError('FD2_NATIVE_STEP_ACCOUNTING')
            self.ticks+=count
            if live:self.live_ticks+=count
            else:self.shadow_ticks+=count

    def progress(self,**values):
        self.ledger.update(ticks=self.ticks,live_ticks=self.live_ticks,shadow_ticks=self.shadow_ticks,
            engine_starts=self.starts,worker_CPU_s=time.process_time()-self.started,**values)
        write(self.path,self.ledger)


def geometry_row(engine,state,target,opening,stage):
    """No snapshot, forward, control mutation or integration is allowed here."""
    import mujoco
    from .fd1_shadow import pad_clearance
    from .teacher import PUSH_STAGES
    from go2arx5_vla.trajectory.so3 import geodesic_angle
    gaps=pad_clearance(engine);distances=[];forces=[]
    for side in ['left','right']:
        pad=engine.geoms[side+'_pad'];table=engine.geoms['table']
        distances.append(float(mujoco.mj_geomDistance(engine.model,engine.data,pad,table,2.,np.zeros(6))))
        forces.append(sum(float(c['normal_force_N']) for c in state['contacts']
            if {c['geom0'],c['geom1']}=={pad,engine.geoms['cube_geom']}))
    error=np.zeros(3) if target is None else state['tcp'][:3,3]-target[:3,3]
    angle=0. if target is None else float(geodesic_angle(state['tcp'][:3,:3],target[:3,:3]))
    return np.r_[state['time_s'],[x['vertical_box_bottom_minus_table_top_m'] for x in gaps],
        distances,forces,error,angle,0. if opening is None else state['gripper_fraction']-opening,
        PUSH_STAGES.index(stage),state['object_pose'][:3,3],state['tcp'][:3,3]]


def run_cell(meter,card):
    from . import episode,teacher as teacher_module,s1_preflight,r3_teacher
    from .s1_preflight import fingerprint
    root=meter.root;folder=root/'episodes'/card['id']
    # Frozen original routines are wrapped externally, never edited or hot-switched.
    original_factory=episode.make_teacher;original_static=teacher_module.make_teacher
    original_engine=episode.make_engine;original_clone=s1_preflight.clone
    engine_holder=[];measurements=[];shadow_measurements=[];latest={};seen={}
    reference=dict(stage='approach');mutations=[];active_targets={}

    def native(c,assets):
        meter.start();engine=original_engine(c,assets);engine_holder.append(engine)
        cls=type(engine);step0=cls.step;snapshot0=cls.snapshot
        def step(self,target,opening):
            active_targets[id(self)]=(target,opening)
            return meter.step(self,step0,target,opening,self is engine)
        def snapshot(self):
            state=snapshot0(self)
            if self.steps and self.steps!=seen.get(id(self)):
                before=fingerprint(self)
                target,opening=active_targets.get(id(self),(None,None))
                row=geometry_row(self,state,target,opening,reference['stage'])
                if fingerprint(self)!=before:
                    mutations.append('PASSIVE_OBSERVER_CHANGED_NATIVE_STATE')
                    raise RuntimeError(mutations[-1])
                (measurements if self is engine else shadow_measurements).append(row)
                seen[id(self)]=self.steps
            latest[id(self)]=state
            return state
        cls.step,cls.snapshot=step,snapshot
        return engine

    def metered_clone(engine):
        meter.start()
        return original_clone(engine)

    class Protected(r3_teacher.ProtectedTeacher):
        def plan(self,state):
            import uuid
            reference['stage']=self.teacher.stage
            self.guard();value=self.teacher.plan(state)
            self.poses,self.opening=value['poses'][:8],value['opening'][:8]
            self.nonce=str(uuid.uuid4())
            forecast,self.expected=s1_preflight.predict(self.engine,self.poses,self.opening,
                nonce=self.nonce,guard=meter.guard,charge=lambda *a,**k:meter.guard())
            forecast.update(sequence=self.engine.steps//25,stage=value['stage'])
            self.records.append(forecast);write(self.folder/'S1_PREFLIGHTS.json',self.records)
            meter.progress(current_cell=card['id'])
            if forecast['decision']!='PASS':raise PermissionError('PRE_EXECUTION_SAFETY_REJECT')
            self.index=0;self.forecast=forecast;self.start_tick=self.engine.steps
            if self.negatives is None:self.check_negatives()
            return value

    def factory(c,initial,engine):
        t=adapt_teacher(original_static(c,initial,engine),card['nominal_clearance_m'])
        return Protected(t,engine,root,folder,{},meter.guard,card['id'])

    def static_factory(c,initial,engine):
        return adapt_teacher(original_static(c,initial,engine),card['nominal_clearance_m'])

    reservation=dict(schema='PI0294_LIVE_RESERVATION_V1',fixture=False,id=card['id'],card=card,
        card_sha256=digest(card),source_sha=meter.m['code']['vla_commit'],root=str(root),output=str(folder),
        asset_root=meter.inputs['asset_root'],deadline_epoch=meter.m['limits']['hard_close_epoch'],
        cpus=sorted(os.sched_getaffinity(0)),reservation=dict(physics_starts=1,physics_steps=16000,rgb_pairs=641,
            cpu_s=min(3600,meter.cpu_cap-int(time.process_time()-meter.started)),memory_bytes=8*1024**3,output_bytes=768*1024**2))
    put(root/'reservations'/(card['id']+'.json'),reservation)
    result=None
    try:
        episode.make_engine=native;episode.make_teacher=factory;teacher_module.make_teacher=static_factory
        s1_preflight.clone=r3_teacher.clone=metered_clone
        result=episode.run_episode(card,folder,meter.inputs['asset_root'],meter.m['code']['vla_commit'],reservation)
    finally:
        episode.make_engine=original_engine;episode.make_teacher=original_factory
        teacher_module.make_teacher=original_static;s1_preflight.clone=r3_teacher.clone=original_clone
        if folder.exists():
            np.savez_compressed(folder/'DYNAMIC_CLEARANCE.npz',live=np.asarray(measurements).reshape(-1,19),
                shadow=np.asarray(shadow_measurements).reshape(-1,19))
            put(folder/'DYNAMIC_OBSERVER.json',dict(status='PASS_UNCHANGED_FINGERPRINT' if not mutations else 'FAIL',
                live_rows=len(measurements),shadow_rows=len(shadow_measurements),extra_snapshot_forward_step_calls=0,
                columns=['time','left_vertical_gap','right_vertical_gap','left_signed_distance','right_signed_distance',
                    'left_object_force','right_object_force','TCP_minus_ref_X','TCP_minus_ref_Y','TCP_minus_ref_Z',
                    'SO3_error','gripper_error','Teacher_stage_index','object_X','object_Y','object_Z','TCP_X','TCP_Y','TCP_Z']))
    s1=read(folder/'S1_RESULT.json') if (folder/'S1_RESULT.json').exists() else {}
    result.update(S1_rejections=s1.get('pre_execution_rejections',0),
        live_native_failures=s1.get('live_native_failures',0),nominal_clearance_m=card['nominal_clearance_m'])
    expected_error=(result['status']=='VALID_COMPLETE' or 'PRE_EXECUTION_SAFETY_REJECT' in (result.get('error') or '')
        or 'STATIC_QUALIFICATION_FAILED_NO_INTEGRATION' in (result.get('error') or ''))
    if (s1.get('live_native_failures') or mutations or not expected_error
            or (s1 and s1['maximum_live_shadow_difference']>1e-10)):
        put(root/'STOP.json',dict(reason='ACTUAL_NATIVE_SAFETY_OR_S1_INTEGRITY',cell=card['id']))
    return result


def run(root):
    root=Path(root);lock=OwnerLock(root/'FD2_OWNER.lock');meter=None;rows=[];failure=None
    try:
        meter=Meter(root)
        plan=read(root/'plans/PILOT_PLAN.json')
        for card in plan['cards']:
            meter.guard(new_cell=True)
            if card['id'] in meter.ledger['attempted_cells']:
                raise PermissionError('FD2_NO_CELL_RETRY')
            if meter.ticks+32000>300000 or meter.starts+85>700:
                raise PermissionError('FD2_FULL_CELL_RESERVATION_REQUIRED')
            meter.ledger['attempted_cells'].append(card['id']);meter.progress()
            result=run_cell(meter,card);rows.append(result)
            put(root/'evidence'/(card['id']+'_RESULT.private.json'),result)
            meter.progress(completed_cells=len(rows),current_cell=None)
            from .fd2_audit import audit
            audit(root,card['id'])
            if (root/'STOP.json').exists():break
        return rows
    except BaseException as exc:
        failure=type(exc).__name__+': '+str(exc)
        put(root/'evidence/PILOT_FAILURE.private.json',dict(error=failure))
        raise
    finally:
        if meter is not None:
            child=resource.getrusage(resource.RUSAGE_CHILDREN)
            child_cpu=child.ru_utime+child.ru_stime-meter.children.ru_utime-meter.children.ru_stime
            meter.progress(status='TERMINAL_PARTIAL' if failure or len(rows)!=8 else 'TERMINAL_EIGHT_CELLS',
                Core_CPU_s=meter.ledger['Core_CPU_s']+time.process_time()-meter.started+child_cpu,
                CPU_reserved_s=0,child_CPU_s=child_cpu,failure=failure)
            put(root/'evidence/PILOT_RESULT.private.json',dict(status=meter.ledger['status'],rows=rows,
                ticks=meter.ticks,engine_starts=meter.starts,planned_cells=8,attempted_cells=meter.ledger['attempted_cells'],
                new_GPU_PBS_neural_training=0))
        lock.close()


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
    print(json.dumps(dict(completed=len(run(a.root)))),flush=True)
