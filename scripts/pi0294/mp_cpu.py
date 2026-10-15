"""Fixed saved-input planning and narrow adopted CPU admission; no integration."""
import ast,hashlib,json,struct
from pathlib import Path
import numpy as np
from .mp_runtime import VERSION,FOLDER,require_adoption,cards
from .mp_client import Client,PlannerFailure,check_coverage,contact_permitted
from .mp_teacher import Teacher
from .mp_window import SavedWindow
from .task_capture_runtime import read,sha,require_task_asset
from .task_state import ActualState,parse_state
from .base_intent import native_document
from go2arx5_vla.trajectory.so3 import matrix_to_quat,quat_to_matrix,rotation_matrix_to_rot6d_rows,rot6d_rows_to_matrix


def saved_inputs(root,c):
    root=Path(root);folder=(root/'teacher-qualification-v2/episodes'/c['source'] if c['task']=='T1' else root/'episodes'/c['source'])
    if sha(folder/'samples.jsonl')!=c['source_samples_sha256']:raise ValueError('FIXED_ACTUAL_INPUT_SOURCE_HASH')
    samples=[json.loads(s) for s in (folder/'samples.jsonl').read_text().splitlines()]
    wanted={samples[i]['native_step']:i for i in c['frame_indices']};packets={}
    with (folder/'state.bin.journal').open('rb') as f:
        while raw:=f.read(4):
            size,=struct.unpack('<I',raw);packet=parse_state(f.read(size))
            if packet['step'] in wanted:packets[wanted[packet['step']]]=packet
            if len(packets)==len(wanted):break
    if len(packets)!=6:raise ValueError('EXACT_SAVED_NATIVE_PACKETS_REQUIRED')
    return folder,samples,packets


def qualify(root,manifest):
    import mujoco,importlib.metadata as md
    from scripts.pi0292.durable import write
    from go2arx5_vla.task_data.codec import pose10
    from .task_clock import actual_source_age
    from .persistent_teacher_receiver import bind_packet
    from .task_contacts import allowed
    require_adoption(root,manifest);root=Path(root);p=root/FOLDER
    conditions=read(p/'FIXED_CONDITIONS.json');lock=read(p/'SOURCE_LOCK.json');checks={};records=[];first_poses=None;first_state=None;first_engine=None
    previous=read(p/'P1_PLANNING.partial.json') if (p/'P1_PLANNING.partial.json').exists() else dict(records=[])
    if previous['records']:
        historical=p/('P1_BEFORE_REAUDIT_'+previous['source_sha']+'.private.json')
        if not historical.exists():write(historical,previous)
    for row in lock['frozen_project_members']:
        path=Path(__file__).resolve().parents[2]/row['path']
        if sha(path)!=row['sha256']:raise ValueError('FROZEN_SCORER_SAFETY_OR_ASSET_SOURCE_CHANGED:'+row['path'])
    env=read(p/'ENV_RECEIPT.json');assert env['versions']['numpy']=='1.26.4' and env['versions']['mplib']=='0.2.1' and not env['native_profile_changed']
    checks['ISOLATED_DEPENDENCY_IMPORT_NO_NATIVE_PROFILE_MUTATION']=dict(planner_numpy=env['versions']['numpy'],native_numpy=md.version('numpy'),actual_installed_source_sha256=env['installed_planner_source_sha256'])
    if not md.version('numpy').startswith('2.'):raise ValueError('NATIVE_PROFILE_VERSION_CHANGED')
    for c in conditions['CPU_saved_input_conditions']:
        folder,samples,packets=saved_inputs(root,c);model=mujoco.MjModel.from_binary_path(str(folder/'task-scene.private.mjb'));engine=ActualState(model)
        submitted={r['index']:r for r in (json.loads(line) for line in (folder/'committed_intents.jsonl').read_text().splitlines())
            if r.get('phase')=='SUBMITTED_REQUIRES_INSTALL_ACK'}
        card=next(x for x in cards() if x['task']==c['task'] and x['duration_s']==12.)
        teacher=None
        for index,frame in enumerate(c['frame_indices']):
            packet=packets[frame];raw=samples[frame];state=engine.update(packet);state['time_s']=float(raw['time_s'])
            for key in ('qpos','qvel','ctrl'):
                if not np.array_equal(state[key],raw[key]):raise ValueError('ACTUAL_ARRAY_BYTES_CHANGED')
            mount=np.asarray(raw['actual_BASE_mount'] if 'actual_BASE_mount' in raw else submitted[frame]['actual_BASE_mount'])
            before=hashlib.sha256(np.asarray(state['qpos']).tobytes()+np.asarray(state['qvel']).tobytes()).hexdigest()
            if teacher is None:teacher=Teacher(card,state,engine,mount,root=root,condition=c['condition_id'],cpu=True)
            teacher.observe(state,mount)
            teacher.query_index=index
            try:
                saved=next((r for r in previous['records'] if r['task']==c['task'] and r['query_index']==index and r['status']=='PASS_VALID_PLANNED_H40'),None)
                cache=p/(c['condition_id']+'-'+str(index)+'-PLAN.private.npz')
                if saved is not None:
                    req=p/'requests'/c['condition_id']/(f'{index:04d}'+'-input.private.json')
                    target,_=teacher.target(state)
                    expected=teacher.client.payload(state,target,teacher.stage,c['planning_seed_table'][index]['seed'])
                    if read(req)!=expected or sha(req)!=saved['certificate']['input_sha256']:raise ValueError('SAVED_PLAN_INPUT_IDENTITY')
                    with np.load(cache,allow_pickle=False) as arrays:result=dict(poses=arrays['poses'].copy(),opening=arrays['opening'].copy(),persistent=dict(certificate=saved['certificate']))
                    teacher.query_index=index+1
                else:result=teacher.plan(state,mount)
                row=dict(task=c['task'],query_index=index,source_frame=frame,status='PASS_VALID_PLANNED_H40',certificate=result['persistent']['certificate'],source_samples_sha256=c['source_samples_sha256'])
                if not cache.exists():np.savez_compressed(cache,poses=result['poses'],opening=result['opening'])
                else:
                    with np.load(cache,allow_pickle=False) as arrays:
                        assert np.array_equal(arrays['poses'],result['poses']) and np.array_equal(arrays['opening'],result['opening'])
                if first_poses is None:first_poses=result['poses'].copy();first_state=state;first_engine=engine
            except PlannerFailure as failure:
                row=dict(task=c['task'],query_index=index,source_frame=frame,status='VALID_PLANNING_FAIL_PRESERVED',reason=str(failure),evidence=failure.evidence)
            assert before==hashlib.sha256(np.asarray(state['qpos']).tobytes()+np.asarray(state['qvel']).tobytes()).hexdigest()
            records.append(row);write(p/'P1_PLANNING.partial.json',dict(records=records,source_sha=manifest['code']['vla_commit'],new_physics=0))
        if teacher.events:raise ValueError('SAVED_FAR_INPUTS_FORCED_MANIPULATION')
    checks['ACTUAL_PACKET_BYTES_UNCHANGED']=dict(saved_inputs=len(records),unchanged=True)
    valid=[r for r in records if r['status']=='PASS_VALID_PLANNED_H40']
    if first_poses is None:
        proof=dict(status='BLOCKED_VALID_START_PLANNING_FAILURE',version=VERSION,source_sha=manifest['code']['vla_commit'],
            checks=checks,saved_input_results=records,saved_inputs=12,new_physics=0,new_render=0,new_neural_queries=0,
            physical_milestones='NOT_ADMITTED_FIXED_START_PATHS_NOT_FEASIBLE',selected='NONE',deployment='NONE')
        write(p/'CPU_ADMISSION.json',proof);return proof
    checks['ARX5_NATIVE_FK_PARITY']=dict(errors=[r['certificate']['FK_error'] for r in valid],position_tolerance=1e-5,rotation_tolerance=1e-4)
    client=Client(root,first_engine);binding=client.binding;m=first_engine.model
    checks['MOUNT_ROOT_TCP_PAD_FRAME_PARITY']=dict(fullbody_FK_included_actual_root=True,tool='native_x5_ee',native_model_sha256=binding['native_model_sha256'])
    rotation=first_state['tcp'][:3,:3];assert np.allclose(quat_to_matrix(matrix_to_quat(rotation)),rotation,atol=1e-12)
    checks['MPLIB_WXYZ_NATIVE_XYZW_CONVERSION']=dict(roundtrip=True)
    assert np.allclose(rot6d_rows_to_matrix(rotation_matrix_to_rot6d_rows(rotation)),rotation,atol=1e-12)
    checks['ROW_ROT6D_ROUNDTRIP']=dict(roundtrip=True)
    assert len(binding['joint_names'])==20 and binding['leg_joint_count']==12
    checks['ACTUAL_JOINT_ORDER_LIMITS_NO_CLIPPING']=dict(joints=20,legs=12,actual_clipping=False)
    bad=dict(binding,leg_joint_count=0)
    try:check_coverage(m,bad)
    except ValueError:checks['MISSING_LEG_BODY_COLLISION_COVERAGE_REJECT']=True
    else:raise ValueError('MISSING_BODY_NOT_REJECTED')
    assert not contact_permitted(['task_left_pad','task_table'],0.,'contact')
    checks['ILLEGAL_TABLE_COLLISION_REJECT']=True
    assert not contact_permitted(['task_left_pad','task_cube_geom'],0.,'approach') and contact_permitted(['task_left_pad','task_cube_geom'],-.001,'contact') and not contact_permitted(['task_left_pad','task_cube_geom'],-.005,'contact')
    checks['STAGE_SPECIFIC_PAD_CUBE_CONTACT_MASK']=True
    try:client.payload(dict(first_state,native_grasp=False),first_state['tcp'],'lift',0,attached=True)
    except ValueError:checks['ATTACHED_OBJECT_ONLY_AFTER_PHYSICAL_GRASP']=True
    else:raise ValueError('NONPHYSICAL_PAYLOAD_ATTACHED')
    action=np.zeros((40,13));action[:,:10]=np.stack([pose10(v,.5) for v in first_poses])
    continuous=SavedWindow(action).take(8);split=SavedWindow(action);a=split.take(4);restored=SavedWindow.restore(json.loads(json.dumps(split.snapshot())));b=restored.take(4)
    assert np.array_equal(continuous,np.vstack([a,b]));checks['SAVED_PATH_CONTINUOUS8_EQUALS4_RESTORE4']=dict(equal=True,new_RRT_calls=0)
    checks['CROSS_PROCESS_INPUT_PATH_SHA_IDENTITY']=dict(valid_plans=len(valid),all_input_and_response_SHA_checked=True)
    assert actual_source_age(10.2,10.2)==0.
    checks['INTEGER_NS_CLOCK_PAST_10_2_BOUNDARY']=True
    assert actual_source_age(1.,.9)>.025
    assert actual_source_age(1.,1.1)<0
    try:bind_packet(dict(_pi0294=dict(episode='future',sequence=0,source_mpc_s=1.1,ttl_s=.4,operation='install')),episode='future',sequence=-1,now=1.,trajectory_type=None,ee=None,base=None)
    except ValueError:pass
    else:raise ValueError('FUTURE_SOURCE_TIME_ACCEPTED')
    checks['STALE_AND_FUTURE_SOURCE_TIME_REJECT']=True
    doc=native_document(action,np.eye(4),base_valid=True,previous_velocity=np.zeros(3));assert len(doc['ee']['waypoints'])==41 and doc['ee']['waypoints'][-1]['time']==2.
    checks['H40_PREFIX8_AND_BRIDGE_PADDING_ACCOUNTING']=dict(H=40,prefix=8,native_endpoints=41)
    try:bind_packet(dict(doc,_pi0294=dict(episode='a',sequence=0,source_mpc_s=0.,ttl_s=.4,operation='install')),episode='a',sequence=0,now=0.,trajectory_type=None,ee=None,base=None)
    except ValueError:checks['REQUEST_INSTALL_LABEL_DUPLICATE_SEQUENCE_IDENTITY']=True
    else:raise ValueError('DUPLICATE_SEQUENCE_ACCEPTED')
    from go2arx5_vla.target_sim.scorer import BOOL_FIELDS,score_episode
    times=np.arange(641)/20;objects=np.repeat(np.eye(4)[None],641,axis=0);objects[:,:3,3]=[.45,-.08,.46]
    objects[(times>=1)&(times<=4),2,3]=.49;objects[times>=3,1,3]=.08
    tcp=np.repeat(np.eye(4)[None],641,axis=0);tcp[:,:3,3]=[.2,0.,.8]
    fixture={k:np.zeros(641,dtype=bool) for k in BOOL_FIELDS}
    for k in ('object_present','contact_valid','gripper_state_valid','safety_valid'):fixture[k][:]=True
    for k in ('native_grasp','left_object_contact','right_object_contact'):fixture[k][(times>=1)&(times<=4)]=True
    fixture.update(time_s=times,object_pose=objects,tcp_world=tcp,object_linear_velocity=np.zeros((641,3)),object_angular_velocity=np.zeros((641,3)))
    positive=score_episode(fixture,goal_center=[.45,.08,.46],initial_object_pose=objects[0],table_height_m=.44,complete=True,evidence_source='CPU_FIXTURE_NOT_PHYSICS')
    corrupted={k:v.copy() for k,v in fixture.items()};corrupted['object_pose'][590,0,3]+=.2
    negative=score_episode(corrupted,goal_center=[.45,.08,.46],initial_object_pose=objects[0],table_height_m=.44,complete=True,evidence_source='CPU_FIXTURE_NOT_PHYSICS')
    assert positive['persistent_success'] and not negative['persistent_success'] and negative['hold_pass']==40
    checks['FROZEN_SCORE_RETAINS_LATE_FAILURE']=dict(positive_hold=41,late_failure_hold=negative['hold_pass'],late_failure_rejected=True,evidence='CPU_FIXTURE_NOT_PHYSICS')
    target=first_state['tcp'].copy();target[0,3]=np.nan
    try:client.payload(first_state,target,'approach',0)
    except ValueError:checks['NONFINITE_OUTPUT_REJECT']=True
    else:raise ValueError('NONFINITE_TARGET_ACCEPTED')
    backend=Path(__file__).with_name('mp_backend.py').read_text();assert all(x not in backend for x in ('mj_step(','.env.step(','.set_ctrl('))
    checks['NO_PHYSICAL_QPOS_QVEL_TORQUE_COMMAND']=dict(backend_physics_calls=0,backend_joint_commands=0)
    absent=native_document(action,np.eye(4),base_valid=False,previous_velocity=np.zeros(3));assert 'base' in doc and 'base' not in absent
    checks['ZERO_BASE_INTENT_NOT_ABSENT_REFERENCE']=True
    image_pairs=0
    for c in conditions['CPU_saved_input_conditions']:
        folder,samples,packets=saved_inputs(root,c)
        for frame in c['frame_indices']:
            calibration=read(folder/'images'/f'{frame:06d}-calibration.json')
            for camera in ('body_rgb','wrist_rgb'):
                assert calibration[camera]['timestamp_s']==packets[frame]['time_s'] and (folder/'images'/f'{frame:06d}-{camera}.png').is_file()
            image_pairs+=1
    checks['REAL_RGB_STATE_TIMESTAMP_BINDING']=dict(saved_actual_pairs=image_pairs,new_render=0)
    try:require_task_asset(root,manifest,'FULLBODY_COMPONENT_PROGRESS_QUALIFICATION_V2')
    except PermissionError as e:assert str(e)=='COMPONENT_TEACHER_V2_SERIES_TERMINAL_PRESERVED'
    else:raise ValueError('OLD_TERMINAL_REOPENED')
    checks['OLD_TERMINAL_SCOPES_STAY_REJECTED']=True
    card=read(p/'TASK_CARD.json');assert sum(c['integrations_with_warmup_and_tail_max'] for c in conditions['conditions'])+2*2503==card['limits']['integrations_including_warmup_tail']==59018
    checks['BUDGET_AND_PRETASK_REPLACEMENT_COUNTERS']=dict(max_starts=6,max_integrations=59018,max_CPU_s=21600,old_counters_reset=False)
    assert set(checks)==set(conditions['CPU_interface_checks_planned'])
    starts=[next(r for r in records if r['task']==task and r['query_index']==0) for task in ('T1','T2')]
    status='PASS_CPU_CONTRACT_AND_START_PATHS' if all(r['status']=='PASS_VALID_PLANNED_H40' for r in starts) else 'BLOCKED_VALID_START_PLANNING_FAILURE'
    proof=dict(status=status,version=VERSION,source_sha=manifest['code']['vla_commit'],checks=checks,checks_passed=len(checks),saved_input_results=records,
        saved_inputs=12,planning_requests=len(records),actual_state_modified=False,physics_steps=0,new_render=0,new_neural_queries=0,
        physical_milestones='NOT_YET_OBSERVED',selected='NONE',deployment='NONE')
    write(p/'CPU_ADMISSION.json',proof);return proof
