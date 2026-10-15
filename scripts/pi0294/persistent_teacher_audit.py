"""Independent CPU audit of first-valid adopted Teacher qualification arrays.

Count actual integrations and object contacts, reconcile submitted/installed
references and record prefix labels. Never re-execute Teacher or physics.
"""
import hashlib,json,struct
from pathlib import Path
import numpy as np
from .task_capture_runtime import read,sha
from .task_state import ActualState,parse_state
from .persistent_teacher import VERSION,cards,geometry
from .task_clock import FROZEN_FIELDS
from .base_intent import native_document
from go2arx5_vla.trajectory.so3 import quat_slerp,geodesic_angle


def rows(path):return [json.loads(s) for s in Path(path).read_text().splitlines()]


def installed_parity(planner):
    accepts={r['ref_id']:r for r in planner if r['kind']=='accept'};covered=set();messages=0;xyzmax=0.;qmax=0.
    for row in planner:
        if row['kind']!='installed' or row.get('ref_id') not in accepts:continue
        a=accepts[row['ref_id']];doc=json.loads(a['raw']);times=np.asarray(row['times'])
        poses=np.asarray(row['poses']);inputs=np.asarray(row['inputs'])
        if poses.shape!=(len(times),14) or inputs.shape!=(len(times),12) or inputs.any():raise ValueError('NATIVE_14D_REFERENCE_ZERO_12D_INPUT')
        for part,offset in (('ee',0),('base',7)):
            waypoints=doc[part]['waypoints'];grid=np.asarray([w['time'] for w in waypoints])
            xyz=np.asarray([w['position'] for w in waypoints]);q=np.asarray([w['quaternion_xyzw'] for w in waypoints])
            for t,p in zip(times,poses[:,offset:offset+7]):
                dt=t-a['source_mpc_s'];expected=np.array([np.interp(dt,grid,xyz[:,k]) for k in range(3)])
                i=int(np.clip(np.searchsorted(grid,dt,side='right')-1,0,len(grid)-2))
                fraction=float(np.clip((dt-grid[i])/(grid[i+1]-grid[i]),0,1))
                qe=quat_slerp(q[i],q[i+1],fraction);qe/=np.linalg.norm(qe);got=p[3:]/np.linalg.norm(p[3:])
                xyzmax=max(xyzmax,float(np.max(np.abs(p[:3]-expected))))
                qmax=max(qmax,float(min(np.max(np.abs(qe-got)),np.max(np.abs(qe+got)))))
        covered.add(row['ref_id']);messages+=1
    if covered!=set(accepts) or xyzmax>1e-6 or qmax>1e-6:raise ValueError('NATIVE_INSTALLATION_PARITY')
    return dict(transactions=len(covered),installed_messages=messages,max_xyz_abs_m=xyzmax,
        max_sign_invariant_unit_quaternion_component=qmax,tolerance=1e-6)


def run(root,episode):
    import mujoco
    from scripts.pi0292.durable import write
    folder=Path(root)/'teacher-qualification-v1/episodes'/episode
    output=folder/'INDEPENDENT_CPU_AUDIT.json'
    if output.exists():raise FileExistsError('FIRST_AUDIT_PRESERVED')
    result=read(folder/'RESULT.json');card=read(folder/'CARD.json')
    if card not in [dict(c,task_asset_version=VERSION) for c in cards()]:raise ValueError('FIXED_CARD_IDENTITY')
    engineering=result['status']=='ENGINEERING_OR_NATIVE_SAFETY_INCOMPLETE'
    if engineering and result.get('error')!='ValueError: ACTUAL_OBSERVATION_SOURCE_CLOCK_NOT_BOUND':
        raise ValueError('UNREVIEWED_ENGINEERING_INCOMPLETE')
    if result['status'] not in ('VALID_TEACHER_NO_PROGRESS_FAIL','RAW_TASK_COMPLETE_REQUIRES_INDEPENDENT_CPU_AUDIT','ENGINEERING_OR_NATIVE_SAFETY_INCOMPLETE'):
        raise ValueError('INCOMPLETE_OR_NATIVE_SAFETY_RESULT_SEPARATE_FROM_QUALITY')
    samples=rows(folder/'samples.jsonl');commands=rows(folder/'committed_intents.jsonl');planner=rows(folder/'planner.jsonl')
    intents=[r for r in commands if r['phase']=='SUBMITTED_REQUIRES_INSTALL_ACK']
    acks=[r for r in commands if r['phase']=='INSTALL_ACK']
    accepted={r['ref_id']:r for r in planner if r['kind']=='accept'}
    n=len(samples);expected_queries=(n-2)//8+1
    if len(intents)!=expected_queries or len(acks)!=expected_queries or len(accepted)!=expected_queries:
        raise ValueError('PREFIX8_TRANSACTION_COUNT')
    if result['native_stops'] or result.get('recorder_error'):raise ValueError('SAFETY_OR_RECORDER_INTEGRITY_FAILURE')
    times=np.asarray([s['time_s'] for s in samples])
    if not np.allclose(times,np.arange(n)*.05,atol=1e-10,rtol=0):raise ValueError('ACTUAL_20HZ_OBSERVATIONS')
    start=read(folder/'TASK_START.json');model=mujoco.MjModel.from_binary_path(str(folder/'task-scene.private.mjb'))
    actual=ActualState(model);packets={};native_count=0;micro=[];grasp_ticks=0;longest_grasp=0;illegal=0
    with (folder/'state.bin.journal').open('rb') as f:
        while length:=f.read(4):
            size,=struct.unpack('<I',length);p=parse_state(f.read(size));native_count+=1
            if p['step']!=native_count:raise ValueError('HIGH_RATE_STATE_SEQUENCE')
            relative=p['step']-start['native_step']
            if relative<0:continue
            state=actual.update(p)
            if relative%25==0:packets[relative//25]=p
            if not relative:continue
            grasp_ticks=grasp_ticks+1 if state['native_grasp'] else 0;longest_grasp=max(longest_grasp,grasp_ticks)
            illegal+=not state['safety_valid']
            micro.append([relative*.002,int(state['robot_object_contact']),int(state['native_grasp']),
                state['object_pose'][0,3],state['object_pose'][1,3],state['object_pose'][2,3]])
    expected_end=start['native_step']+(n if engineering else n-1)*25
    if native_count!=expected_end or len(packets)!=(n+1 if engineering else n) or illegal:raise ValueError('EXACT_PHYSICS_CONTACT_CLOCK_OR_SAFETY')
    hashes={};geometry_ready=0;aligned=0
    for i,s in enumerate(samples):
        p=packets[i]
        if any(not np.array_equal(s[k],p[k]) for k in ('qpos','qvel','ctrl')) or s['native_step']!=p['step']:
            raise ValueError('SAMPLE_NOT_ACTUAL_NATIVE_BYTES')
        actual.update(p)
        direction=np.asarray(card['goal_xy'])-np.asarray(card['source_xy']);direction/=np.linalg.norm(direction)
        recomputed=geometry(actual,dict(s,tcp=np.asarray(s['tcp']),object_pose=np.asarray(s['object_pose'])),card['task'],direction)
        if recomputed['precontact_geometry_ready']!=s['teacher_geometry']['precontact_geometry_ready']:
            raise ValueError('GEOMETRY_NOT_ACTUAL_PAD_OBJECT_FK')
        geometry_ready+=recomputed['precontact_geometry_ready'];aligned+=recomputed['actual_contact_alignment']
        calibration=read(folder/'images'/f'{i:06d}-calibration.json')
        for camera in ('body_rgb','wrist_rgb'):
            if calibration[camera]['timestamp_s']!=p['time_s']:raise ValueError('ACTUAL_RGB_STATE_TIME')
            image=folder/'images'/f'{i:06d}-{camera}.png';hashes[image.name]=sha(image)
        if i:
            previous=intents[(i-1)//8]
            if s['active_intent_sequence']!=previous['sequence'] or s['executed_prefix_index']!=(i-1)%8:
                raise ValueError('PREFIX_ACTION_LABEL_SLICE')
    previous=np.zeros(3)
    for seq,(intent,ack) in enumerate(zip(intents,acks)):
        i=8*seq;p=packets[i];action=np.asarray(intent['action_EEF13']);accept=accepted[intent['wire_sha256']]
        doc=json.loads(accept['raw']);rebuilt=native_document(action,np.asarray(intent['actual_BASE_mount']),base_valid=True,previous_velocity=previous)
        if (intent['index']!=i or intent['sequence']!=seq or ack['native_step']!=p['step']
                or ack['wire_sha256']!=intent['wire_sha256']
                or hashlib.sha256(accept['raw'].encode()).hexdigest()!=intent['wire_sha256']
                or intent['source_mpc_s']!=(round(p['time_s']*1e9)/1e9 if 'source_clock' in intent else p['time_s']) or any(doc[k]!=rebuilt[k] for k in ('ee','base'))
                or not 0<=accept['receiver_mpc_s']-accept['source_mpc_s']<=.025):
            raise ValueError('ACTUAL_INPUT_SUBMITTED_LABEL_OR_SOURCE_IDENTITY')
        previous=action[7,10:13].copy()
    pauses=rows(folder/'pause_intervals.jsonl')
    if len(pauses)!=(n if engineering else n-1) or any(any(p['before'][k]!=p['after'][k] for k in FROZEN_FIELDS) for p in pauses):
        raise ValueError('NONREALTIME_WAIT_CHANGED_PHYSICAL_CONTROLLER_HISTORY')
    lifecycle=[r for r in planner if r['kind']=='lifecycle']
    if lifecycle:raise ValueError('UNEXPECTED_NATIVE_HOLD_OR_REFERENCE_EXPIRY')
    parity=installed_parity(planner)
    target=np.asarray(intents[0]['persistent']['target']);tcp=np.asarray([s['tcp'] for s in samples]);obj=np.asarray([s['object_pose'] for s in samples])
    distances=np.linalg.norm(tcp[:,:3,3]-target[:3,3],axis=1)
    angles=np.asarray([geodesic_angle(p[:3,:3],target[:3,:3]) for p in tcp])
    micro=np.asarray(micro);displacement=np.linalg.norm(micro[:,3:5]-obj[0,:2,3],axis=1)
    lift=micro[:,5]-obj[0,2,3];contact=micro[:,1].astype(bool)
    manipulation=longest_grasp*.002>=.3 if card['task']=='T1' else bool(np.any(contact&(displacement>=.01)&(lift<=.005)&(micro[:,2]==0)))
    milestones=dict(position_progress=bool(distances[0]-distances.min()>=.02),safe_precontact=bool(geometry_ready),
        actual_contact=bool(contact.any()),manipulation=bool(manipulation))
    all_milestones=all(milestones.values())
    proof=dict(status='ENGINEERING_INCOMPLETE_VALID_PREFIX_INTEGRITY_PASS' if engineering else 'PASS_SHORT_MILESTONES_AND_INTEGRITY' if all_milestones else 'VALID_SHORT_MILESTONE_FAIL_INTEGRITY_PASS',
        version=VERSION,id=episode,source_sha=result['source_sha'],new_physics=0,new_model_queries=0,
        observations=n,submitted_and_installed=len(intents),native_integrations=native_count,task_duration_s=float(times[-1]),
        native_safety_stops=0,raw_state_byte_identity=True,RGB_state_time_identity=True,submitted_label_identity=True,
        explicit_query_period_s=.4,record_period_s=.05,pause_intervals=len(pauses),unexpected_lifecycle_events=lifecycle,
        installation=parity,selected_MPC_policy_consumption='UNKNOWN',
        milestones=milestones,geometry_ready_observations=int(geometry_ready),aligned_observations=int(aligned),
        actual_position_goal_distance_initial_min_final_m=[float(distances[0]),float(distances.min()),float(distances[-1])],
        actual_orientation_error_initial_min_final_rad=[float(angles[0]),float(angles.min()),float(angles[-1])],
        actual_TCP_initial_final_world_xyz=tcp[[0,-1],:3,3].tolist(),stage_target_world_xyz=target[:3,3].tolist(),
        longest_actual_grasp_s=longest_grasp*.002,actual_contact_ticks=int(contact.sum()),
        max_actual_object_displacement_xy_m=float(displacement.max()),max_actual_lift_m=float(lift.max()),
        stop_reason=result.get('stop_reason',result.get('error')),final_stage=result.get('final_stage',samples[-1]['teacher_stage']),teacher_events=result.get('teacher_events','UNOBSERVED_TERMINATED_BEFORE_RESULT_SNAPSHOT'),
        formal_task_success='NOT_CLAIMED_SHORT_DEVELOPMENT',full_confirmation_eligible=bool(all_milestones and not engineering),
        engineering_incomplete=engineering,unrecorded_20Hz_final_interval=bool(engineering),
        old_pilot_denominator_unchanged=True,RGB_pairs=n,selected='NONE',deployment='NONE')
    array=folder/'SHORT_ACTUAL_PHYSICS.npz'
    if array.exists():raise FileExistsError('FIRST_SCIENTIFIC_ARRAY_PRESERVED')
    np.savez_compressed(array,time_s=times,tcp=tcp,object_pose=obj,goal_distance_m=distances,angle_rad=angles,contact_micro=micro)
    write(folder/'RGB_HASHES.json',hashes)
    proof['raw_evidence']={n:sha(folder/n) for n in ('samples.jsonl','state.bin.journal','committed_intents.jsonl','planner.jsonl','physics.jsonl','pause_intervals.jsonl','RGB_HASHES.json','SHORT_ACTUAL_PHYSICS.npz')}
    write(output,proof)
    return proof
