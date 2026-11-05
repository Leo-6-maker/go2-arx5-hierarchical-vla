"""Same-process observation-gated NEXT2 contact commissioning, original safety."""
from pathlib import Path
import json,math
import numpy as np
from .storage import digest
VELOCITY_LIMIT=.015


def validate(spec,previous):
    p=Path(spec['episode_manifest']);e=json.loads(p.read_text())
    if(digest(p)!=spec['episode_manifest_sha256']or e.get('schema')!='WM02_NEXT2_PIECEWISE_CONTACT_V1'or e.get('phase')!='collect-target-next2-contact'or e.get('run')!=spec['registered_root']or e.get('scene_id')!='P01'or e.get('split')!='TRAIN'or e.get('task_atoms')!=400 or e.get('screen_atom')!=40 or e.get('prehold_atoms')!=32 or e.get('reference_rule')!='FIXED_HOLD32_BURN8_CONTACT40_V2'or e.get('purpose')not in('PRECONTACT_CONTINUITY','GRADED_CONTACT')or e.get('velocity_limit_m_s')!=VELOCITY_LIMIT or e.get('required_hold_s')!=1. or e.get('further_hold_s')!=.5 or spec.get('parallel_batch_spec')or not spec.get('next2_binding')or spec.get('next1_binding')):raise PermissionError('EXACT_FRESH_SINGLE_PROCESS20S_CONTACT_SCOPE_REQUIRED')
    assets=json.loads(Path(spec['scene_allowlist']).read_text());a=next(a for a in assets['scenes']if a['scene_id']=='P01')
    if digest(spec['scene_allowlist'])!=spec['scene_allowlist_sha256']or digest(spec['model_path'])!=a['model_sha256']or spec['model_path']!=a['model_path']:raise PermissionError('ORIGINAL_P01_SCENE_REQUIRED')
    if not previous or previous.get('status')!='NATIVE_STAGE_PASSED'or not previous.get('shutdown_clean')or previous.get('native_binary_sha256')!=digest(spec['native_binary']):raise PermissionError('PRIOR_SAFE_C08_ENDPOINT_REQUIRED')
    for key in('contact_static','contact_reference_check'):
        if digest(spec[key])!=spec[key+'_sha256']:raise PermissionError('ORIGINAL_STATIC_PATH_IDENTITY_REQUIRED')
    static=json.loads(Path(spec['contact_reference_check']).read_text())
    if static.get('grade')!='PRECONTACT_80'or not static.get('all_static_chunks_feasible')or static.get('new_physical_integrations')!=0:raise PermissionError('ORIGINAL_PRECONTACT80_STATIC_REFERENCE_REQUIRED')
    proof=Path(spec['extended_trace_build']);trace=json.loads(proof.read_text())
    if digest(proof)!=spec['extended_trace_build_sha256']or trace.get('status')!='TRACE_ONLY_131072_CAPACITY_COMPILED_LOADER_VERIFIED_NOT_NATIVE_QUALIFIED'or trace.get('trace_capacity')!=131072 or trace.get('trace_library_sha256')!=digest(spec['trace_library'])or trace.get('native_binary_sha256')!=digest(spec['native_binary'])or trace.get('observer_sha256')!=digest(spec['observer_library'])or trace.get('new_target_integrations')!=0:raise PermissionError('ACTUAL_TRACE_ONLY_CAPACITY_SUCCESSOR_REQUIRED')
    if e['purpose']=='GRADED_CONTACT':
        proof=Path(spec['precontact_qualification']);v=json.loads(proof.read_text());native=json.loads((proof.parent/'RESULT.json').read_text())
        if digest(proof)!=spec['precontact_qualification_sha256']or not v.get('precontact_continuity_verified')or not v.get('same_native_process')or native.get('status')!='NATIVE_STAGE_PASSED'or native.get('physics_stops')or not native.get('shutdown_clean'):raise PermissionError('COMPLETE_NATIVE_AND_SCIENTIFIC_C1_QUALIFICATION_REQUIRED')
    return 12503


class Continuity:
    """Twenty complete50ms intervals plus0.5s further hold; no sample-count shortcut."""
    def __init__(self):self.since=None;self.longest_s=0.;self.losses=[];self.was_ready=False
    def observe(self,time_s,geometry,speed):
        ready=bool(geometry and speed<=VELOCITY_LIMIT)
        if ready:
            if self.since is None:self.since=time_s
            elapsed=time_s-self.since;self.longest_s=max(self.longest_s,elapsed)
        else:
            if self.since is not None:self.losses.append(dict(time_s=time_s,continuous_s=time_s-self.since,geometry=geometry,speed_m_s=speed))
            self.since=None;elapsed=0.
        self.was_ready=ready;return elapsed


def gripper_memory(adapter):
    """Read this adapter's child wrapper globals; no injection or memory writes."""
    import os,subprocess,struct
    from .target_sequential import owned_simulator_pid
    process=next(p for label,p in adapter.processes if label=='sim');pid=owned_simulator_pid(process.pid,adapter.spec['native_binary']);library=adapter.spec['outer_library']
    if not hasattr(adapter,'next2_gripper_symbols'):
        output=subprocess.check_output(['nm','-anC',library],text=True);wanted={'target':('2d',16),'steps':('Q',8),'sequence':('Q',8),'opening':('d',8),'expiry':('d',8),'initialized':('?',1),'has_command':('?',1)};symbols={}
        for line in output.splitlines():
            parts=line.split(maxsplit=2)
            if len(parts)==3:
                for key,fmt in wanted.items():
                    if parts[2]=='(anonymous namespace)::'+key:symbols[key]=(int(parts[0],16),*fmt)
        if set(symbols)!=set(wanted):raise RuntimeError('ACTUAL_GRIPPER_MEMORY_SYMBOLS_NOT_AVAILABLE')
        adapter.next2_gripper_symbols=symbols
    maps=[line.split()for line in Path(f'/proc/{pid}/maps').read_text().splitlines()if len(line.split())>=6 and line.split()[-1]==library];base=min(int(line[0].split('-')[0],16)-int(line[2],16)for line in maps);values={}
    with open(f'/proc/{pid}/mem','rb',buffering=0)as memory:
        for key,(offset,fmt,size)in adapter.next2_gripper_symbols.items():memory.seek(base+offset);value=struct.unpack('<'+fmt,memory.read(size));values[key]=list(value)if len(value)>1 else value[0]
    return dict(status='ACTUAL_OWNED_CHILD_GLOBALS_READ_ONLY',pid=pid,library_sha256=digest(library),values=values)


def advance_atom(a,installed_step):
    from .target_reference import require_live_window,trajectory_identity,installed_target_identities
    from scripts.pi0292.eval_contract import matrix10
    before=a.observe();step=before['step'];clock=require_live_window(a.active_reference['source_mpc_s'],before['time_s'],25);event_start=len(a.ros_evidence);after=a.advance_to(step+25);a.spin(.15)
    if a.latest['mode'].data!='WBC'or any(r.get('reason')=='PREFIX_TTL_EXPIRED'for r in a.reference_events()):raise RuntimeError('ORIGINAL_REFERENCE_MODE_OR_TTL_FAILED')
    installed=[r for r in a.active_reference['receiver_events']if r['kind']=='installed'][-1];identity=trajectory_identity(installed['times'],installed['poses'],installed['inputs']);sent=installed_target_identities(a.reference_events(),a.active_reference['wire_sha256']);matched=[]
    for row in a.ros_evidence[event_start:]:
        if row['topic_key']!='policy':continue
        msg=row['message'];target=msg['plan_target_trajectories'];key=trajectory_identity(target['time_trajectory'],[s['value']for s in target['state_trajectory']],[s['value']for s in target['input_trajectory']])
        if key in sent and sent[key]['monotonic_s']<=row['received_monotonic_s']:matched.append(dict(received_monotonic_s=row['received_monotonic_s'],policy_init_time=msg['init_observation']['time'],trajectory_sha256=key,receiver_send_monotonic_s=sent[key]['monotonic_s']))
    r=dict(native_start_step=step,native_end_step=after['step'],actual_ticks=25,clock=clock,reference_sha256=a.active_reference['wire_sha256'],target_trajectory_sha256=identity,solver_policies_with_exact_installed_target=matched,controller_selected_reference='DERIVED_ONLY_NO_SELECTED_POLICY_ID_EXPORTED',mode=a.latest['mode'].data)
    a.command_event(dict(phase='controller-consumed'if matched else'consumption_unconfirmed',**r));a.reference_history.append(dict(end_native_step=after['step'],pose=matrix10(a.active_reference['action_EEF10'])[(step-installed_step)//25]));return r


def stage_path(start,end,n=50):
    """Persistent planned suffix; same trapezoid and unchanged certificates."""
    from go2arx5_vla.trajectory.so3 import matrix_to_quat,quat_to_matrix,quat_slerp
    t=np.arange(n+1,dtype=float);b=8.;area=n-b;weight=np.where(t<b,t*t/(2*b),np.where(t<=n-b,t-b/2,area-(n-t)**2/(2*b)))/area;path=np.repeat(np.asarray(start)[None],n+1,axis=0);path[:,:3,3]=start[:3,3]+weight[:,None]*(end[:3,3]-start[:3,3]);q0=matrix_to_quat(start[:3,:3]);q1=matrix_to_quat(end[:3,:3])
    for i,w in enumerate(weight):path[i,:3,:3]=quat_to_matrix(quat_slerp(q0,q1,float(w)))
    return path


def enough_time(remaining_s,stage):
    # Original max rates are certified independently. These are conservative
    # finite-chain allocations including settling, opening and safe withdrawal.
    return remaining_s+1e-9>={'DESCEND':10.,'CLOSE':6.25,'WITHDRAW':3.}[stage]


def run(a):
    from scripts.pi0292.durable import write
    from .next1_episode import assess
    from .next1_contact import planned_path,chunk
    from go2arx5_vla.task_data.codec import pose10
    from scripts.pi0294.task_gripper_contract import teacher_physical_fraction
    from scripts.pi0294.persistent_teacher import geometry
    import mujoco
    e=json.loads(Path(a.spec['episode_manifest']).read_text());a.prepare_dynamic();initial=a.measured_task_state();a.next1_samples=[initial];path=np.repeat(initial['tcp'][None],2,axis=0);path_start=0;opening=1.;stage='STEADY';stage_since=initial['time_s'];events=[dict(stage='INIT',native_step=0),dict(stage=stage,native_step=2503)];windows=[];samples=[];continuity=Continuity();installed_step=2503;previous=initial;descended=False;first_blocker=None;pending=False;grasp_since=None;support_qualified=False;released=False;withdrawn=False;above=None
    table=mujoco.mj_name2id(a.actual_state.model,mujoco.mjtObj.mjOBJ_GEOM,'task_table');cube=a.actual_state.cube_geom
    def install(atom):
        a.next2_contact_stage=stage;a.install_reference(np.stack([pose10(p,float(teacher_physical_fraction(opening)))for p in chunk(path,max(0,atom-path_start))]))
    def change(new,state,reason,remaining):
        nonlocal stage,stage_since,pending
        events.append(dict(previous=stage,stage=new,native_step=state['native_step'],actual_state_gate=reason,remaining_task_s=remaining));stage=new;stage_since=state['time_s'];pending=True
    install(0)
    for atom in range(400):
        force=pending;pending=False
        if atom==32:force=True
        if atom==40:
            screen=assess(a,e)
            if not screen['qualified']:raise RuntimeError('UNSTABLE_CURRENT_CONTACT_PREFIX')
            path=planned_path(initial['tcp'],a.measured_task_state()['object_pose'],'PRECONTACT_80');above=path[-1].copy();path_start=atom;stage='APPROACH';events.append(dict(stage=stage,native_step=a.observe()['step'],actual_state_gate='QUALIFIED_CURRENT_STEADY_PREFIX'));write(a.output/'CONTACT_PLANNED_REFERENCE.private.json',dict(path=path.tolist(),source='ORIGINAL_PRECONTACT80_NO_EMPIRICAL_BIAS_COMPENSATION',planned_only=True));force=True
        if force or a.observe()['step']-installed_step>=175:install(atom);installed_step=a.observe()['step']
        w=advance_atom(a,installed_step);windows.append(w);state=a.measured_task_state();a.next1_samples.append(state);g=geometry(a.actual_state,state,'T1');speed=float(np.linalg.norm(state['tcp'][:3,3]-previous['tcp'][:3,3])/.05);previous=state;elapsed=continuity.observe(state['time_s'],g['precontact_geometry_ready'],speed)if stage in('APPROACH','PRECONTACT_HOLD')else 0.;memory=gripper_memory(a);remaining=(399-atom)*.05;path_done=atom+1-path_start>=len(path)-1
        support=any({c['geom0'],c['geom1']}=={table,cube}and c['wrench'][0]>=.05 for c in a.observe()['contacts'])
        if state['native_grasp']:
            if grasp_since is None:grasp_since=state['time_s']
        else:grasp_since=None
        if stage=='APPROACH'and path_done and continuity.was_ready:change('PRECONTACT_HOLD',state,'ACTUAL_ORIGINAL_PAD_GEOMETRY_AND_PREREGISTERED_LOW_SPEED',remaining)
        if stage=='PRECONTACT_HOLD'and not continuity.was_ready:change('APPROACH',state,'ACTUAL_PRECONTACT_OR_LOW_SPEED_LOST',remaining)
        if e['purpose']=='GRADED_CONTACT':
            if stage=='PRECONTACT_HOLD'and elapsed>=1.5-1e-9:
                if enough_time(remaining,'DESCEND'):
                    target=above.copy();target[:3,3]=state['object_pose'][:3,3]+[0,0,.02];path=stage_path(above,target);path_start=atom+1;descended=True;change('DESCEND',state,'CURRENT1S_PLUS0.5S_PRECONTACT_AND10S_CHAIN_BUDGET',remaining)
                elif first_blocker is None:first_blocker='INSUFFICIENT_REMAINING_TIME_FOR_DESCENT_AND_SAFE_EXIT'
            elif stage=='DESCEND'and path_done:
                if g['actual_contact_alignment']and speed<=VELOCITY_LIMIT and enough_time(remaining,'CLOSE'):
                    path=np.repeat(path[-1:,:,:],2,axis=0);path_start=atom+1;opening=0.;change('CLOSE',state,'CURRENT_ACTUAL_PAD_ALIGNMENT_LOW_SPEED_AND6.25S_EXIT_BUDGET',remaining)
                elif remaining<6.25:
                    first_blocker=first_blocker or'NO_ACTUAL_CONTACT_ALIGNMENT_BEFORE_CLOSE_EXIT_RESERVE';path=stage_path(path[-1],above);path_start=atom+1;change('WITHDRAW',state,'OPEN_GRIPPER_NO_FORCED_CLOSE_AT_RESERVE',remaining)
            elif stage=='CLOSE':
                if grasp_since is not None and state['time_s']-grasp_since>=.25 and support:
                    change('SUPPORTED_HOLD',state,'ACTUAL_OPPOSING_NATIVE_GRASP_AND_TABLE_SUPPORT',remaining)
                elif state['time_s']-stage_since>=1.25:
                    first_blocker=first_blocker or'NO_ACTUAL_OPPOSING_GRASP_AFTER_BOUNDED_CLOSE';opening=1.;change('RELEASE',state,'BOUNDED_UNSUCCESSFUL_CLOSE_RELEASES',remaining)
            elif stage=='SUPPORTED_HOLD':
                if not state['native_grasp']or not support:first_blocker=first_blocker or'ACTUAL_SUPPORTED_GRASP_LOST';opening=1.;change('RELEASE',state,'ACTUAL_SUPPORT_OR_GRASP_LOSS',remaining)
                elif state['time_s']-stage_since>=.5:support_qualified=True;opening=1.;change('RELEASE',state,'ACTUAL_SUPPORTED_GRASP_HELD0.5S',remaining)
            elif stage=='RELEASE'and state['time_s']-stage_since>=1.25 and np.min(state['finger_q'])>=.040 and not state['robot_object_contact']:
                released=True
                if enough_time(remaining,'WITHDRAW'):path=stage_path(path[-1],above);path_start=atom+1;change('WITHDRAW',state,'ACTUAL_OPEN_FINGERS_AND_CONTACT_RELEASE_WITH3S_EXIT_BUDGET',remaining)
                else:first_blocker=first_blocker or'INSUFFICIENT_REMAINING_TIME_AFTER_ACTUAL_RELEASE'
            elif stage=='WITHDRAW'and path_done and state['time_s']-stage_since>=3. and g['precontact_geometry_ready']and speed<=VELOCITY_LIMIT:withdrawn=True
        contacts=[dict(geom0=c['geom0'],geom1=c['geom1'],distance_m=c['distance_m'],normal_force_N=float(c['wrench'][0]))for c in a.observe()['contacts']]
        row=dict(native_step=state['native_step'],time_s=state['time_s'],stage=stage,precontact=bool(g['precontact_geometry_ready']),alignment=bool(g['actual_contact_alignment']),TCP_speed_m_s=speed,continuous_low_speed_precontact_s=elapsed,TCP=state['tcp'].tolist(),object_pose=state['object_pose'].tolist(),finger_q=state['finger_q'].tolist(),left_object_contact=state['left_object_contact'],right_object_contact=state['right_object_contact'],native_grasp=state['native_grasp'],table_support=support,contacts=contacts,actual_gripper_internal_memory=memory,active_reference_sha256=w['reference_sha256'],command_path_offset_atom=max(0,atom-path_start));samples.append(row);a.command_event(dict(phase='piecewise-contact-achieved',**row))
    verified=bool(e['purpose']=='PRECONTACT_CONTINUITY'and continuity.was_ready and samples[-1]['continuous_low_speed_precontact_s']>=1.5-1e-9);proof=dict(schema='WM02_NEXT2_SAME_PROCESS_PRECONTACT_CONTINUITY_V1',source_sha=a.spec['source_sha'],same_native_process=True,physics_resets=0,state_teleports=0,completed_task_atoms=len(windows),original_pad_geometry_unchanged=True,velocity_limit_m_s=VELOCITY_LIMIT,velocity_preregistration='ceil(NEXT1 TRAIN precontact max0.013406993555705382/0.005)*0.005',required_hold_s=1.,further_hold_s=.5,precontact_continuity_verified=verified,longest_continuous_low_speed_hold_s=continuity.longest_s,losses=continuity.losses,stage_events=events,descended=descended,contact_positives=sum(s['left_object_contact']or s['right_object_contact']for s in samples),supported_grasp_qualified=support_qualified,actual_released=released,actual_withdrawn=withdrawn,first_blocker=first_blocker or(None if verified or(support_qualified and released and withdrawn)else'ACTUAL_PRECONTACT_CONTINUITY_NOT_ATTAINED'if not descended else'STAGED_CONTACT_NOT_COMPLETED'),samples=samples,full_task_success=False)
    write(a.output/'PRECONTACT_CONTINUITY.private.json',proof);a.capture_loaded_identity();cancel=a.cancel_reference();a.seal_trace(a.spec['max_ticks']);value=dict(schema='WM02_TARGET_DYNAMIC_DIAGNOSTIC_V1',phase=a.spec['phase'],actual_collection_start_step=2503,actual_collection_start_s=initial['time_s'],task_native_ticks=10000,complete_task_atoms=len(windows),windows=windows,operator_cancel=cancel,controller_selected_reference='DERIVED_ONLY_NO_SELECTED_POLICY_ID_EXPORTED',full_task_success=False,piecewise_references=True,recorded_gripper_memory=True,reference_rule=e['reference_rule']);write(a.output/'DYNAMIC_DIAGNOSTIC.json',value);a.reference_qualification=value
    if not all(w['solver_policies_with_exact_installed_target']for w in windows):raise RuntimeError('SOLVER_TARGET_NOT_CONFIRMED_FOR_EVERY_ATOM')
    return value
