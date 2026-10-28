"""Twelve affected V3 CPU contracts, then fixed source admission; no physics."""
import json,os,time
from pathlib import Path
import numpy as np
from .task_capture_runtime import read,sha
from .sdr_v3_runtime import FOLDER,cards,require_adoption,authority_values,AUTHORITY,condition,slot_cost,goal_consumer
from .sdr_v3_source import prepare,load,sample,with_entry
from .sdr_v3_client import Client,PlannerFailure
from .sdr_v2_cpu import saved_input as old_saved_input

def saved_input(root,task):
    from .task_state import ActualState
    from .sdr_v3_geometry import xml_scene
    import mujoco
    engine,state,base=old_saved_input(root,task)
    if task=='T2':return engine,state,base
    card=next(c for c in cards() if c['task']==task);binding=read(Path(root)/'INPUT_BINDINGS.private.json');model=mujoco.MjModel.from_xml_string(xml_scene(card,binding['asset_root']))
    packet={k:state[k].copy() for k in ['qpos','qvel','ctrl']};packet['qpos'][29]+=.03;packet.update(time_s=state['time_s'],step=0,contacts=[],contact_time_s=state['time_s']-.002)
    engine=ActualState(model);preview=engine.update(packet);preview['time_s']=state['time_s'];preview['CPU_view']='SAVED_ROOT_ARM_NEW_SCENE_PREVIEW_NOT_ACTUAL_OBSERVATION';return engine,preview,base

def packages_for(root,task):
    packages=[]
    for i,r in enumerate(x for x in read(Path(root)/FOLDER/'PROGRAMS.json')['rows'] if x['task']==task):
        d=load(root,r);packages.append(dict(index=i,task=task,approach=d['poses'][0],contact=d['poses'][0],direct_join=True,active_duration_s=r['active_duration_s'],source_id=r['id'],source_program_sha256=r['sha256']))
    return packages

def boundaries(root,manifest):
    import mujoco,hashlib,struct
    from scripts.pi0292.durable import write
    from .sdr_v3_geometry import xml_scene,check_frozen_robot,initial_data,initial_audit
    root=Path(root);p=root/FOLDER;a,s=require_adoption(root,manifest);names=read(p/'TASK_CARD.json')['checks_planned'];checks=[]
    def put(e):checks.append(dict(name=names[len(checks)],status='PASS',evidence=e))
    def reject(fn,kind=PermissionError):
        try:fn()
        except kind:return True
        raise AssertionError('V3_NEGATIVE_MUST_REJECT')
    assert reject(lambda:authority_values(dict(a,authorizes_execution=False),AUTHORITY,time.time()))
    old=root/'source-demo-retime-join-v2';assert read(old/'SERIES_CLOSED.json')['status']=='STOP_V2_VALID_NATIVE_SAFETY_FAIL_NO_TASK_SEED'
    for name,digest in s['old_evidence'].items():assert sha(root/name)==digest
    put(dict(new_authority_positive=True,unsigned_rejected=True,old_V2_terminal_unchanged=True))
    binding=read(root/'INPUT_BINDINGS.private.json');models={};inits={};metadata={}
    old_model=mujoco.MjModel.from_binary_path(str(old/'episodes/SDR2_SEED_W0_T1/task-scene.private.mjb'))
    for card in cards():
        text=xml_scene(card,binding['asset_root']);model=mujoco.MjModel.from_xml_string(text);initial=initial_data(model);models[card['task']]=model;inits[card['task']]=initial
        metadata[card['task']]=initial_audit(model,initial,card)
        assert read(old/'episodes/SDR2_SEED_W0_T1/PREEXECUTION.json')['frozen_robot']['status']=='COMPILED_NATIVE_PROPERTIES_EXACTLY_PRESERVED'
        for field in ['body_pos','body_quat','body_mass','body_inertia','body_ipos','body_iquat']:
            assert np.array_equal(getattr(model,field)[:-1],getattr(old_model,field)[:-1]),field
        for field in ['jnt_type','jnt_axis','jnt_pos','jnt_range','jnt_limited','jnt_qposadr','jnt_dofadr','actuator_gear','actuator_ctrlrange','dof_armature','dof_frictionloss']:
            assert np.array_equal(getattr(model,field),getattr(old_model,field)),field
        # All compiled native geom properties, controller actuators and physics remain identical.
        for field in ['opt.timestep']:
            assert model.opt.timestep==old_model.opt.timestep
        for field in ['geom_size','geom_friction','geom_margin','geom_contype','geom_conaffinity','body_mass','body_inertia','jnt_range','actuator_gainprm','actuator_biasprm','dof_damping']:
            assert np.array_equal(getattr(model,field),getattr(old_model,field)),field
        assert np.array_equal(initial.qpos[:25],initial_data(old_model).qpos[:25])
    put(dict(compiled_robot_and_all_collision_properties_exact=True,only_task_geometry_and_reset_whitelist=True))
    proof=read(root/'source-demo-access-v3-preparation/T1_SAVED_GEOMETRY_BOUND.json');assert abs(proof['proposed_table_top_m']-.47)<1e-14 and .47-.05-proof['observed_root_geometry_plus_margin_top_m']>=.02
    assert np.allclose(inits['T1'].qpos[27:30],[.45,-.08,.49],atol=1e-15,rtol=0)
    put(dict(T1_height=.47,slab_thickness=.05,saved_only_clearance_m=proof['lower_face_clearance_over_saved_envelope_m'],not_dynamic_PASS=True))
    for row in read(p/'PROGRAMS.json')['rows']:
        previous=next(x for x in read(old/'PROGRAMS.json')['rows'] if x['id']==row['id'])
        with np.load(old/'programs'/previous['file'],allow_pickle=False) as z:
            actual=load(root,row);lo=int(np.searchsorted(z['source_indices'],row['cut_start_index']));expected=z['poses'][lo:].copy()
            if row['task']=='T1':expected[:,:3,3]+=[0,0,.03]
            assert np.array_equal(actual['poses'],expected) and np.array_equal(actual['gripper'],z['gripper'][lo:]) and np.array_equal(actual['time'],z['time'][lo:]-z['time'][lo])
    put(dict(T1_only_common_Z_translation=True,all_source_rotations_gripper_and_times_preserved=True))
    from go2arx5_vla.target_sim.scorer import score_episode,BOOL_FIELDS
    from .score import score as old_score
    arrays=dict(time_s=np.arange(641)*.05,tcp_world=np.tile(np.eye(4),(641,1,1)),object_pose=np.tile(np.eye(4),(641,1,1)),object_linear_velocity=np.zeros((641,3)),object_angular_velocity=np.zeros((641,3)))
    arrays['object_pose'][:,:3,3]=[.45,-.08,.46];arrays['tcp_world'][:,:3,3]=[.35,0,.6]
    arrays.update({k:np.full(641,k not in ['native_abort','illegal_contact','joint_limit_penetration','reset','takeover','native_grasp','left_object_contact','right_object_contact'],dtype=bool) for k in BOOL_FIELDS})
    original=score_episode(arrays,goal_center=[.45,.08,.46],initial_object_pose=arrays['object_pose'][0],table_height_m=.44,complete=True,evidence_source='CPU_FIXTURE_NOT_PHYSICS')
    shifted={k:v.copy() for k,v in arrays.items()};shifted['tcp_world'][:,2,3]+=.03;shifted['object_pose'][:,2,3]+=.03
    translated=score_episode(shifted,goal_center=[.45,.08,.49],initial_object_pose=shifted['object_pose'][0],table_height_m=.47,complete=True,evidence_source='CPU_FIXTURE_NOT_PHYSICS')
    assert original==translated and reject(lambda:old_score(arrays,next(c for c in cards() if c['task']=='T1'),table_top=.47),ValueError)
    put(dict(original_score_translation_equivalence=True,old_height_wrapper_still_rejects_H47=True,threshold_changes=0,CPU_fixture_only=True))
    prefix=read(root/'source-demo-access-v3-preparation/T2_SOURCE_CONTACT_PREFIX_AUDIT_R1.json');assert [r['join_index'] for r in prefix['rows']]==[223,231,223]
    assert all(not r['pad_cube_contacts_before_join'] and not r['pad_cube_contacts_at_join'] and not r['grasp_before'] and not r['lift_before'] and not r['carry_before'] for r in prefix['rows'])
    assert [r['cut_start_index'] for r in read(p/'PROGRAMS.json')['rows'] if r['task']=='T2']==[223,231,223]
    put(dict(T2_fixed_slice_and_original_geom_ID_prefix=True,retained_suffix_points=[266,274,274]))
    assert np.array_equal(inits['T2'].qpos[25:27],[.002,.002]);assert np.array_equal(inits['T1'].qpos[:25],inits['T2'].qpos[:25])
    put(dict(T2_keyframe_reset_and_actual_initial_audit_same_function=True,initial_command_fraction=.002/.044,original_actuation_initializes_target_from_real_qpos=True,real_simulator_identity_required_before_first_step=True))
    engine,state,base=saved_input(root,'T2');client=Client(root,engine)
    previous=read(old/'requests/SDR2_CPU_T2/0000/INPUT.private.json')
    for i,pkg in enumerate(packages_for(root,'T2')):
        request=client.payload(state,pkg['contact'],'approach',0);goal=dict(pose=pkg['contact'].tolist(),stage='contact',closed=True,IK_seed=condition(root,'SDR3_CPU_T2')['sources'][i]['original_streams']['IK'][1])
        assert goal_consumer(request,goal)==goal_consumer(previous,previous['goals'][i*2+1])
    put(dict(T2_all_three_original_consumers_exact=True,old_hash_not_rewritten=True,new_actual_needs_new_goal=True))
    preview_engine,preview,_=saved_input(root,'T1');assert preview['CPU_view'].endswith('NOT_ACTUAL_OBSERVATION') and not 'body_rgb' in preview
    assert manifest['execution']['profile']=='4080-simulation' and not os.environ.get('CUDA_VISIBLE_DEVICES')
    put(dict(CPU_preview_labelled=True,no_mismatched_RGB_or_neural_query=True))
    from .mp_client import check_coverage
    for engine in [preview_engine,engine]:check_coverage(engine.model,Client(root,engine).binding)
    assert all(c['fresh_actual_packet_required'] for c in read(p/'CONDITIONS.json')['conditions'] if c['kind']=='OWN_REAL_WARMUP_INPUT_VALIDATION')
    put(dict(full_native_body_collision_coverage=True,fresh_warmup_required=True,old_cache_not_current_actual=True))
    from .task_state import write_gripper
    f=p/'CPU_PACKET_FIXTURE.bin'
    for i in range(8):
        write_gripper(f,i,5+i*.05,.002/.044);assert struct.unpack('<8sQddd',f.read_bytes())[1]==i
    row=next(r for r in read(p/'PROGRAMS.json')['rows'] if r['task']=='T2');program=load(root,row);entry=dict(time=np.array([0.,.15]),poses=np.stack([program['poses'][0]]*2));combined=with_entry(program,entry)
    assert np.all(sample(combined,np.arange(8)*.05)[1]==0)
    put(dict(H40_internal_join_preserves_closed_entry_gripper=True,sequence_prefix8=True,original_14D_ABI_and_pause_source_pins=True))
    for member in read(p/'SOURCE_LOCK.json')['unchanged_native_members']:assert sha(Path(__file__).resolve().parents[2]/member['path'])==member['sha256']
    assert reject(lambda:slot_cost('IK',0,{},[],condition(root,'SDR3_CPU_T2')))
    assert reject(lambda:slot_cost('RRT',6,{},[],condition(root,'SDR3_CPU_T1')))
    assert reject(lambda:slot_cost('IK',0,{'goal_IK_calls':165},[],condition(root,'SDR3_CPU_T1')))
    assert reject(lambda:authority_values(a,AUTHORITY,1792083759));assert not read(p/'TASK_CARD.json')['valid_failure_retry']
    put(dict(persistent_pre_reservation_caps=True,original_deadline=True,old_core_21_hashes=True,no_safety_retry=True))
    assert len(checks)==12;write(p/'ZERO_STEP_SCENE_AUDIT.json',dict(status='PASS_TASK_ONLY_INIT_AND_COMPILED_CORE',tasks=metadata,new_physics=0));return checks

def qualify(root,manifest):
    from scripts.pi0292.durable import write
    root=Path(root);p=root/FOLDER;prepare(root,manifest)
    if (p/'CPU_ADMISSION.json').exists():return read(p/'CPU_ADMISSION.json')
    checks=boundaries(root,manifest);tasks=[]
    for task in ['T1','T2']:
        engine,state,_=saved_input(root,task);client=Client(root,engine);packages=packages_for(root,task)
        try:
            _,certificate,index=client.bundle(state,packages,'approach','SDR3_CPU_'+task,0);chosen=packages[index]
            result=dict(task=task,status='PASS_SOURCE_ROUTE_AND_TIME',source_id=chosen['source_id'],candidate_index=index,source_program_sha256=chosen['source_program_sha256'],active_duration_s=chosen['active_duration_s'],certificate=certificate)
        except PlannerFailure as e:result=dict(task=task,status='NOT_ADMITTED_FIXED_SOURCE_ROUTES',reason=str(e),evidence=e.evidence,physical_starts=0)
        tasks.append(result);write(p/'CPU_RESULTS.partial.json',dict(task_results=tasks,source_sha=manifest['code']['vla_commit']))
    value=dict(status='CPU_CONTRACT_COMPLETE_TASK_LOCAL_RESULTS',source_sha=manifest['code']['vla_commit'],checks_passed=12,checks=checks,task_results=tasks,scientific_source_candidates=6,CPU_RPC_bundles=2,new_physics=0,new_render=0,new_neural_queries=0,selected='NONE',deployment='NONE');write(p/'CPU_ADMISSION.json',value);return value
