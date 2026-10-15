"""Twenty actual CPU contracts, then two finite saved-input source batches."""
import json,os,subprocess,time
from pathlib import Path
import numpy as np
from .task_capture_runtime import read,sha
from .sdr_v2_runtime import FOLDER,AUTHORITY,require_adoption,authority_values,goal_consumer,condition,slot_cost
from .sdr_v2_source import prepare,load,sample,with_entry
from .sdr_v2_client import Client,PlannerFailure
from .mps2_cpu import first_input

def packages_for(root,task):
    p=Path(root)/FOLDER;old=Path(root)/'source-demo-retarget-v1';records=read(old/'SOURCE_EXTRACTION.json')['records'];packages=[]
    for i,row in enumerate(r for r in read(p/'PROGRAMS.json')['rows'] if r['task']==task):
        d=load(root,row);record=next(r for r in records if r['id']==row['id'])
        event=next(e for e in record['events'] if e['stage']==('close' if task=='T1' else 'contact'))
        index=int(np.searchsorted(d['source_indices'],np.searchsorted(np.arange(640)*.05,event['time_s'])))
        packages.append(dict(index=i,task=task,approach=d['poses'][0],contact=d['poses'][0 if task=='T1' else index],direct_join=task=='T1',
            active_duration_s=row['active_duration_s'],source_id=row['id'],source_program_sha256=row['sha256']))
    return packages

def saved_input(root,task):
    old=read(Path(root)/'maniskill-search-v2/CONDITIONS.json')['conditions'];entry=next(e for e in old if e.get('phase')=='CPU_SAVED_INPUT' and e['task']==task)
    return first_input(root,entry)

def boundaries(root,manifest):
    root=Path(root);p=root/FOLDER;a,s=require_adoption(root,manifest);expected=read(p/'TASK_CARD.json')['CPU_checks'];checks=[]
    def put(evidence):checks.append(dict(name=expected[len(checks)],status='PASS',evidence=evidence))
    def reject(fn):
        try:fn()
        except PermissionError:return True
        raise AssertionError('NEGATIVE_CONTRACT_MUST_REJECT')
    assert sha(root/'source-demo-retarget-v1/SERIES_CLOSED.json')==s['old_evidence']['source-demo-retarget-v1/SERIES_CLOSED.json']
    assert reject(lambda:authority_values(dict(a,authorizes_execution=False),AUTHORITY,time.time()))
    put(dict(V1_terminal_retained=True,unsigned_execution_rejected=True))
    source=read(root/'source-demo-retarget-v1/SOURCE_EXTRACTION.json');rows=read(p/'PROGRAMS.json')['rows']
    assert len(rows)==6 and all(r['split']=='TRAIN' and r['sample_command_parity']=='EXACT_ALL_640' for r in source['records'])
    put(dict(sealed_TRAIN_sources=6,original_640_prefix_reconstruction_reused=True))
    for row in rows:
        d=load(root,row);old=next(x for x in read(root/'source-demo-retarget-v1/RETARGET_TIME_GATE.json')['rows'] if x['id']==row['id'])
        with np.load(root/'source-demo-retarget-v1/sources'/old['retarget_file'],allow_pickle=False) as f:
            lo=0 if row['task']=='T2' else 30
            assert np.array_equal(d['poses'],f['poses'][lo:]) and np.array_equal(d['gripper'],f['gripper'][lo:])
    put(dict(all_pose_and_gripper_values_exact=True,no_geometry_smoothing=True))
    from .source_retime_v2_analysis import so3_angles
    for row in [r for r in rows if r['task']=='T2']:
        d=load(root,row);dt=np.diff(d['time']);v=np.linalg.norm(np.diff(d['poses'][:,:3,3],axis=0),axis=1)/dt;w=so3_angles(d['poses'][:-1,:3,:3],d['poses'][1:,:3,:3])/dt
        assert np.all(dt>=.05-1e-12) and np.max(v)<=.06*(1+1e-10) and np.max(w)<=.3*(1+1e-10)
    put(dict(local_rate_intervals_pass=3,original_speed_bounds_unchanged=True))
    for row in rows:
        d=load(root,row);poses,grip=sample(d,d['time']);assert np.allclose(poses,d['poses'],atol=1e-12,rtol=0) and np.array_equal(grip,d['gripper'])
    put(dict(all_knots_and_gripper_mapped_without_deletion=True))
    from go2arx5_vla.contracts.trajectory import TrajectoryChunkV1
    from go2arx5_vla.safety.trajectory_gate import TrajectorySafetyGateV1
    q=np.tile([0.,0.,0.,1.],(40,1));pos=np.tile([10.,0.,0.],(40,1));chunk=TrajectoryChunkV1(chunk_id='CPU-NEGATIVE',source='CPU',observation_timestamp=1.,point_timestamps=1+np.arange(40)*.05,eef_position=pos,eef_orientation=q,source_frame='world',tcp_semantic_id='native_x5_ee',validity_start=1.,validity_end=3.,gripper=np.ones(40)*.5)
    assert TrajectorySafetyGateV1().evaluate(chunk,now=1.,achieved_position=np.zeros(3),achieved_orientation=q[0]).decision.value=='REJECT'
    put(dict(original_gate_reject_fixture=True,local_cartesian_table_is_not_joint_admission=True))
    assert all(r['cut_start_index']==30 and r['cut_end_index']==257 for r in rows if r['task']=='T1');put(dict(T1_fixed_cut=[30,257]))
    prefix=read(p/'PREP_SOURCE_PREFIX_AUDIT.json');lock=read(p/'SOURCE_LOCK.json');assert sha(p/'PREP_SOURCE_PREFIX_AUDIT.json')==lock['source_prefix_audit_sha256']
    assert all(not r['pad_cube_contact_frames'] and not any(r[k] for k in ['any_grasp','any_opposing_contact','any_lifted','any_carried']) for r in prefix['rows'])
    assert all(load(root,r)['gripper'][0]==1. for r in rows if r['task']=='T1');put(dict(source_prefix_no_pad_contact_or_manipulation=True,join_open=True))
    for r in rows:
        if r['task']=='T1':
            d=load(root,r);assert np.allclose(d['time'],np.arange(228)*.05*r['original_uniform_factor'],atol=1e-12,rtol=0)
    put(dict(T1_uniform_suffix_preserved=True))
    engine,state,_=saved_input(root,'T1');client=Client(root,engine);before=state['qpos'].copy()
    for i,package in enumerate(packages_for(root,'T1')):
        old=read(root/f'source-demo-retarget-v1/requests/SDR_CPU_T1_{i}/0000/INPUT.private.json');old_goal=dict(old['goals'][1],IK_seed=old['streams']['IK'][1])
        fresh=client.payload(state,package['contact'],'approach',0);new_goal=dict(pose=package['contact'].tolist(),stage='align',closed=False,IK_seed=old['streams']['IK'][1])
        assert goal_consumer(fresh,new_goal)==goal_consumer(old,old_goal)
    put(dict(actual_saved_input_equivalent_old_goal_consumers=3,old_hash_not_rewritten=True))
    changed=client.payload(dict(state,qpos=state['qpos']+1e-6),packages_for(root,'T1')[0]['contact'],'approach',0)
    assert changed['actual_packet_arrays_sha256']!=old['actual_packet_arrays_sha256'] and np.array_equal(before,state['qpos'])
    put(dict(changed_actual_packet_not_cache_equivalent=True,actual_state_unchanged=True))
    for task in ('T2','T1'):
        c=condition(root,'SDR2_CPU_'+task);assert [x['source_priority'] for x in c['sources']]==[0,1,2]
    order=sorted([(1.,2,0),(1.,1,1),(2.,0,0)]);assert order[0]==(1.,1,1)
    put(dict(source_order_fixed=True,within_source_cost_index_tie_break=True))
    c=condition(root,'SDR2_CPU_T2');assert reject(lambda:slot_cost('RRT',8,{},[],c)) and reject(lambda:slot_cost('DENSE',2,{},[],c))
    assert reject(lambda:slot_cost('IK',0,{},[],condition(root,'SDR2_CPU_T1')))
    put(dict(shared_RRT_and_dense_caps_reject=True,cached_T1_has_zero_new_IK_capacity=True))
    assert reject(lambda:slot_cost('IK',0,{},['IK/0'],c));assert reject(lambda:slot_cost('IK',0,{'CLIK_reserved':2704},[],c))
    put(dict(restore_no_redraw=True,global_initialization_cap_rejects=True))
    d=load(root,rows[-1]);entry=dict(time=np.array([0.,.15]),poses=np.stack([d['poses'][0],d['poses'][0]]));combined=with_entry(d,entry)
    whole=sample(combined,np.arange(8)*.05);split=tuple(np.concatenate([sample(combined,np.arange(4)*.05)[k],sample(combined,np.arange(4,8)*.05)[k]]) for k in (0,1))
    assert all(np.array_equal(x,y) for x,y in zip(whole,split));put(dict(chunk_internal_join_and_prefix8_restore_equivalent=True))
    assert np.array_equal(before,state['qpos']);put(dict(actual_inputs_not_modified=True,source_grasp_not_used_for_live_grasp=True,physics_steps=0))
    capsule=Path(__file__).resolve().parents[2]
    for member in lock['unchanged_native_members']:assert sha(capsule/member['path'])==member['sha256']
    put(dict(frozen_public_members=len(lock['unchanged_native_members']),original_gate_and_score_unchanged=True))
    assert a['window']['original_T0_epoch']==1792044600 and not a['budget_reset'] and reject(lambda:authority_values(a,AUTHORITY,1792083759))
    put(dict(original_T0_and_deadline_retained=True,expired_start_rejected=True))
    card=read(p/'TASK_CARD.json');assert card['task_independent'] and not card['valid_failure_retry'] and not card['post_lock_source_switch'];put(dict(tasks_independent=True,valid_failure_not_engineering_retry=True))
    assert os.environ.get('DISPLAY') and subprocess.run(['xdpyinfo'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0
    from .task_clock import actual_source_age
    assert actual_source_age(10.2,10.2)==0 and actual_source_age(1.,1.1)<0
    put(dict(parent_DISPLAY_responds=True,source_time_representation_boundary=True,new_render=0))
    assert len(checks)==20;return checks

def qualify(root,manifest):
    from scripts.pi0292.durable import write
    root=Path(root);p=root/FOLDER;prepare(root,manifest)
    if (p/'CPU_ADMISSION.json').exists():return read(p/'CPU_ADMISSION.json')
    checks=boundaries(root,manifest);tasks=[]
    for task in ('T2','T1'):
        engine,state,_=saved_input(root,task);client=Client(root,engine);packages=packages_for(root,task)
        try:
            _,certificate,index=client.bundle(state,packages,'approach','SDR2_CPU_'+task,0)
            selected=next(x for x in packages if x['index']==index)
            result=dict(task=task,status='PASS_SOURCE_ROUTE_AND_TIME',source_id=selected['source_id'],candidate_index=index,
                source_program_sha256=selected['source_program_sha256'],active_duration_s=selected['active_duration_s'],certificate=certificate)
        except PlannerFailure as e:result=dict(task=task,status='NOT_ADMITTED_FIXED_SOURCE_ROUTES',reason=str(e),evidence=e.evidence,physical_starts=0)
        tasks.append(result);write(p/'CPU_RESULTS.partial.json',dict(task_results=tasks,source_sha=manifest['code']['vla_commit']))
    value=dict(status='CPU_CONTRACT_COMPLETE_TASK_LOCAL_RESULTS',source_sha=manifest['code']['vla_commit'],checks_passed=20,checks=checks,task_results=tasks,
        scientific_source_candidates=6,CPU_RPC_bundles=2,new_physics=0,new_render=0,new_neural_queries=0,selected='NONE',deployment='NONE')
    write(p/'CPU_ADMISSION.json',value);return value
