"""Same-process observation-gated NEXT2 contact commissioning, original safety."""
from pathlib import Path
import json,math
import numpy as np
from .storage import digest
VELOCITY_LIMIT=.015


def validate(spec,previous):
    p=Path(spec['episode_manifest']);e=json.loads(p.read_text())
    if(digest(p)!=spec['episode_manifest_sha256']or e.get('schema')!='WM02_NEXT2_PIECEWISE_CONTACT_V1'or e.get('phase')!='collect-target-next2-contact'or e.get('run')!=spec['registered_root']or e.get('scene_id')!='P01'or e.get('split')!='TRAIN'or e.get('task_atoms')!=400 or e.get('screen_atom')!=40 or e.get('prehold_atoms')!=32 or e.get('reference_rule')!='FIXED_HOLD32_BURN8_CONTACT40_V2'or e.get('purpose')!='PRECONTACT_CONTINUITY'or e.get('velocity_limit_m_s')!=VELOCITY_LIMIT or e.get('required_hold_s')!=1. or e.get('further_hold_s')!=.5 or spec.get('parallel_batch_spec')or not spec.get('next2_binding')or spec.get('next1_binding')):raise PermissionError('EXACT_FRESH_SINGLE_PROCESS20S_CONTACT_SCOPE_REQUIRED')
    assets=json.loads(Path(spec['scene_allowlist']).read_text());a=next(a for a in assets['scenes']if a['scene_id']=='P01')
    if digest(spec['scene_allowlist'])!=spec['scene_allowlist_sha256']or digest(spec['model_path'])!=a['model_sha256']or spec['model_path']!=a['model_path']:raise PermissionError('ORIGINAL_P01_SCENE_REQUIRED')
    if not previous or previous.get('status')!='NATIVE_STAGE_PASSED'or not previous.get('shutdown_clean')or previous.get('native_binary_sha256')!=digest(spec['native_binary']):raise PermissionError('PRIOR_SAFE_C08_ENDPOINT_REQUIRED')
    for key in('contact_static','contact_reference_check'):
        if digest(spec[key])!=spec[key+'_sha256']:raise PermissionError('ORIGINAL_STATIC_PATH_IDENTITY_REQUIRED')
    static=json.loads(Path(spec['contact_reference_check']).read_text())
    if static.get('grade')!='PRECONTACT_80'or not static.get('all_static_chunks_feasible')or static.get('new_physical_integrations')!=0:raise PermissionError('ORIGINAL_PRECONTACT80_STATIC_REFERENCE_REQUIRED')
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


def run(a):
    from scripts.pi0292.durable import write
    from .next1_episode import assess
    from .next1_contact import planned_path,chunk
    from go2arx5_vla.task_data.codec import pose10
    from scripts.pi0294.task_gripper_contract import teacher_physical_fraction
    from scripts.pi0294.persistent_teacher import geometry
    e=json.loads(Path(a.spec['episode_manifest']).read_text());a.prepare_dynamic();initial=a.measured_task_state();a.next1_samples=[initial];path=np.repeat(initial['tcp'][None],2,axis=0);path_start=0;opening=1.;stage='STEADY';events=[dict(stage='INIT',native_step=0),dict(stage=stage,native_step=2503)];windows=[];samples=[];continuity=Continuity();installed_step=2503;previous=initial;descended=False;first_blocker=None
    def install(atom):a.install_reference(np.stack([pose10(p,float(teacher_physical_fraction(opening)))for p in chunk(path,max(0,atom-path_start))]))
    install(0)
    for atom in range(400):
        force=False
        if atom==32:force=True
        if atom==40:
            screen=assess(a,e)
            if not screen['qualified']:raise RuntimeError('UNSTABLE_CURRENT_CONTACT_PREFIX')
            path=planned_path(initial['tcp'],a.measured_task_state()['object_pose'],'PRECONTACT_80');path_start=atom;stage='APPROACH';events.append(dict(stage=stage,native_step=a.observe()['step'],actual_state_gate='QUALIFIED_CURRENT_STEADY_PREFIX'));write(a.output/'CONTACT_PLANNED_REFERENCE.private.json',dict(path=path.tolist(),source='ORIGINAL_PRECONTACT80_NO_EMPIRICAL_BIAS_COMPENSATION',planned_only=True));force=True
        if force or a.observe()['step']-installed_step>=175:install(atom);installed_step=a.observe()['step']
        w=advance_atom(a,installed_step);windows.append(w);state=a.measured_task_state();a.next1_samples.append(state);g=geometry(a.actual_state,state,'T1');speed=float(np.linalg.norm(state['tcp'][:3,3]-previous['tcp'][:3,3])/.05);previous=state;elapsed=continuity.observe(state['time_s'],g['precontact_geometry_ready'],speed)if atom>=40 else 0.;memory=gripper_memory(a)
        if stage=='APPROACH'and atom+1-path_start>=len(path)-1 and continuity.was_ready:stage='PRECONTACT_HOLD';events.append(dict(stage=stage,native_step=state['native_step'],basis='ACTUAL_ORIGINAL_PAD_GEOMETRY_AND_PREREGISTERED_LOW_SPEED'))
        if stage=='PRECONTACT_HOLD'and not continuity.was_ready:stage='APPROACH';events.append(dict(stage=stage,native_step=state['native_step'],reason='ACTUAL_PRECONTACT_OR_LOW_SPEED_LOST'))
        # C1 deliberately stops progression here. C2 is a separate finite identity,
        # beginning again at INIT and proving current attainment in this process.
        contacts=[dict(geom0=c['geom0'],geom1=c['geom1'],distance_m=c['distance_m'],normal_force_N=float(c['wrench'][0]))for c in a.observe()['contacts']]
        row=dict(native_step=state['native_step'],time_s=state['time_s'],stage=stage,precontact=bool(g['precontact_geometry_ready']),alignment=bool(g['actual_contact_alignment']),TCP_speed_m_s=speed,continuous_low_speed_precontact_s=elapsed,TCP=state['tcp'].tolist(),object_pose=state['object_pose'].tolist(),finger_q=state['finger_q'].tolist(),left_object_contact=state['left_object_contact'],right_object_contact=state['right_object_contact'],native_grasp=state['native_grasp'],contacts=contacts,actual_gripper_internal_memory=memory,active_reference_sha256=w['reference_sha256'],command_path_offset_atom=max(0,atom-path_start));samples.append(row);a.command_event(dict(phase='piecewise-contact-achieved',**row))
    verified=bool(continuity.was_ready and samples[-1]['continuous_low_speed_precontact_s']>=1.5-1e-9);proof=dict(schema='WM02_NEXT2_SAME_PROCESS_PRECONTACT_CONTINUITY_V1',source_sha=a.spec['source_sha'],same_native_process=True,physics_resets=0,state_teleports=0,completed_task_atoms=len(windows),original_pad_geometry_unchanged=True,velocity_limit_m_s=VELOCITY_LIMIT,velocity_preregistration='ceil(NEXT1 TRAIN precontact max0.013406993555705382/0.005)*0.005',required_hold_s=1.,further_hold_s=.5,precontact_continuity_verified=verified,longest_continuous_low_speed_hold_s=continuity.longest_s,losses=continuity.losses,stage_events=events,descended=descended,contact_positives=sum(s['left_object_contact']or s['right_object_contact']for s in samples),first_blocker=first_blocker or(None if verified else'ACTUAL_PRECONTACT_CONTINUITY_NOT_ATTAINED'),samples=samples,full_task_success=False)
    write(a.output/'PRECONTACT_CONTINUITY.private.json',proof);a.capture_loaded_identity();cancel=a.cancel_reference();a.seal_trace(a.spec['max_ticks']);value=dict(schema='WM02_TARGET_DYNAMIC_DIAGNOSTIC_V1',phase=a.spec['phase'],actual_collection_start_step=2503,actual_collection_start_s=initial['time_s'],task_native_ticks=10000,complete_task_atoms=len(windows),windows=windows,operator_cancel=cancel,controller_selected_reference='DERIVED_ONLY_NO_SELECTED_POLICY_ID_EXPORTED',full_task_success=False,piecewise_references=True,recorded_gripper_memory=True,reference_rule=e['reference_rule']);write(a.output/'DYNAMIC_DIAGNOSTIC.json',value);a.reference_qualification=value
    if not all(w['solver_policies_with_exact_installed_target']for w in windows):raise RuntimeError('SOLVER_TARGET_NOT_CONFIRMED_FOR_EVERY_ATOM')
    return value
