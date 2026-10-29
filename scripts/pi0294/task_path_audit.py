"""Independent saved S1 EEF10 audit: actual physics, submitted wire and score.

This module never steps a model, creates a Teacher, publishes a command or
writes an admission/PASS file. The existing owner supplies an already-paid CPU
stage and persists the returned proof. Engineering prefixes have no task score.
"""
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

from .base_intent import eef_document
from .task_clock import require_pause,actual_source_age
from .task_state import ActualState,parse_state
from .task_path_teacher import serial


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_bytes())
def rows(path):return [json.loads(s)for s in Path(path).read_text().splitlines()]


def need(ok,reason):
    if not ok:raise ValueError('S1_INDEPENDENT_'+reason)


def check_installed_values(installed,document,source):
    """ROS native pose payloads are float32; timestamps retain double precision."""
    points=document['ee']['waypoints'];times=np.asarray(installed['times'],dtype=float)
    poses=np.asarray(installed['poses'],dtype=float)
    expected=np.asarray([p['position']+p['quaternion_xyzw']for p in points])
    need(times.shape==(41,)and poses.shape==(41,7)and np.isfinite(poses).all()
         and np.allclose(times,source+np.arange(41)*.05,atol=1e-8,rtol=0),'INSTALLED_41_POINT_TIME_GRID')
    # This is a representation tolerance, not a task/safety threshold. Keep the
    # actual submitted EEF10 values exact in check_transactions above.
    tolerance=8*np.finfo(np.float32).eps
    need(np.allclose(poses[:,:3],expected[:,:3],atol=tolerance,rtol=0),'INSTALLED_POSITION_DIFFERS_FROM_SUBMITTED')
    q=poses[:,3:];target=expected[:,3:]
    need(np.all(np.abs(np.linalg.norm(q,axis=1)-1)<=tolerance),'INSTALLED_QUATERNION')
    delta=np.minimum(np.max(np.abs(q-target),axis=1),np.max(np.abs(q+target),axis=1))
    need(np.all(delta<=tolerance),'INSTALLED_ORIENTATION_DIFFERS_FROM_SUBMITTED')


def check_transactions(samples,records,planner,*,episode,initial_step=2500):
    """80 exact installed H40 references account for 640 executed intervals.

    The next eight physical intervals use the corresponding submitted prefix,
    never future achieved TCP or an unexecuted H40 suffix. An installation ACK
    remains distinct from the unknown selected MPC policy identity.
    """
    need(len(samples)==641,'COMPLETE_32S_OBSERVATIONS')
    submitted=[v for v in records if v['phase']=='SUBMITTED_REQUIRES_INSTALL_ACK']
    durable=[v for v in records if v['phase']=='SUBMISSION_INTENT_BEFORE_EFFECT']
    acks=[v for v in records if v['phase']=='INSTALL_ACK']
    need(len(submitted)==len(durable)==len(acks)==80,'EIGHTY_PREFIX8_TRANSACTIONS')
    need(len(records)==240,'NO_UNKNOWN_COMMAND_JOURNAL_PHASES')
    need([v['phase']for v in records]==[
        phase for _ in range(80)for phase in ('SUBMISSION_INTENT_BEFORE_EFFECT','SUBMITTED_REQUIRES_INSTALL_ACK','INSTALL_ACK')],
        'DURABLE_PUBLICATION_ACK_ORDER')
    accepted={};installed={}
    for row in planner:
        if row.get('kind')=='accept':
            ref=row['ref_id'];need(ref not in accepted,'DUPLICATE_RECEIVER_ACCEPT');accepted[ref]=row
        elif row.get('kind')=='installed'and row.get('ref_id'):
            ref=row['ref_id'];need(ref not in installed,'DUPLICATE_NATIVE_INSTALL');installed[ref]=row
        elif row.get('kind')=='reject':raise ValueError('S1_INDEPENDENT_NATIVE_REJECT_PRESENT')
    need(len(accepted)==len(installed)==81,'EIGHTY_TASK_PLUS_ONE_QUALIFICATION_REFERENCE')
    intervals=[];task_refs=set();ages=[]
    for j,(intent,pre,ack)in enumerate(zip(submitted,durable,acks)):
        index=j*8;step=initial_step+index*25
        need({k:v for k,v in pre.items()if k!='phase'}=={k:v for k,v in intent.items()if k!='phase'},
             'PRE_EFFECT_AND_PUBLISHED_VALUES_DIFFER')
        need(intent['index']==index and intent['sequence']==j+2 and intent['gripper_sequence']==j,
             'GLOBAL_REFERENCE_AND_SEPARATE_GRIPPER_SEQUENCE')
        ref=intent['wire_sha256'];need(ref not in task_refs and ref in accepted and ref in installed,'EXACT_ACCEPT_AND_INSTALL')
        task_refs.add(ref);wire=accepted[ref];text=wire['raw'];document=json.loads(text)
        need(hashlib.sha256(text.encode()).hexdigest()==ref,'RAW_SUBMITTED_WIRE_DIGEST')
        action=np.asarray(intent['action_EEF10'],dtype=float)
        expected=eef_document(action)
        need(set(document)=={'version','ee','_pi0294'}and {k:document[k]for k in ('version','ee')}==expected,
             'ACTUAL_EEF10_WIRE_NO_BASE_OR_LABEL_SUBSTITUTION')
        metadata=document['_pi0294'];source=round(float(samples[index]['raw_time_s'])*1e9)/1e9
        need(metadata==dict(episode=episode,sequence=j+2,source_mpc_s=source,ttl_s=.4,operation='install'),
             'RAW_WIRE_EPISODE_SEQUENCE_AND_SOURCE_CLOCK')
        need(intent['base_valid'] is False and intent['prefix_points']==8
             and intent['execute_physics_steps']==200 and intent['record_every_physics_steps']==25,
             'EEF10_PREFIX8_PHYSICAL_INTERVAL_CONTRACT')
        need(intent['actual_observation_time_s']==samples[index]['raw_time_s']
             and intent['source_mpc_s']==source and wire['source_mpc_s']==source,
             'SAME_ACTUAL_STATE_SOURCE_NOT_REFRESHED')
        age=wire['receiver_mpc_s']-source;ages.append(age)
        need(0<=age<=.025 and 0<=actual_source_age(samples[index]['raw_time_s'],source)<=.025,'SOURCE_AGE')
        need(ack['index']==index and ack['wire_sha256']==ref and ack['native_step']==step,
             'EXACT_SAME_PAUSED_STATE_ACK')
        need(abs(installed[ref]['bound_start']-source)<=1e-8
             and abs(installed[ref]['observation_time_s']-source)<=1e-8,'INSTALL_BOUND_TO_SOURCE')
        check_installed_values(installed[ref],document,source)
        for k in range(8):
            start,end=samples[index+k],samples[index+k+1]
            need(start['native_step']==step+k*25 and end['native_step']==step+(k+1)*25,
                 'ACTUAL_EXECUTED_25_TICK_INTERVAL')
            need(end['active_intent_sequence']==j+2 and end['executed_prefix_index']==k,
                 'OBSERVATION_TO_INSTALLED_PREFIX_IDENTITY')
            intervals.append(dict(interval_index=index+k,start_native_step=start['native_step'],
                end_native_step=end['native_step'],wire_sha256=ref,prefix_index=k,
                action_EEF10=action[k].tolist(),selected_MPC_policy_consumption='UNKNOWN'))
    return dict(submitted_and_installed=80,completed_intervals=640,source_age_max_s=max(ages),
                task_refs=sorted(task_refs),intervals=intervals)


def check_qualification(folder,planner,ros):
    q=read(folder/'CURRENT_NATIVE_QUALIFICATION.json')
    need(q['status']=='PASS_CURRENT_FEEDBACK_INSTALL_CANCEL_HOLD_AT_EXACT_PAUSE'
         and q['native_step']==2500 and q['physics_integrations_during_qualification']==0
         and q['ROS_sequence_offset']==2 and q['first_gripper_sequence']==0,'CURRENT_NATIVE_IO_PREFIX')
    ref=q['hold_ref_id'];accept=[r for r in planner if r['kind']=='accept'and r.get('ref_id')==ref]
    install=[r for r in planner if r['kind']=='installed'and r.get('ref_id')==ref]
    need(len(accept)==len(install)==1,'QUALIFICATION_RAW_ACCEPT_AND_INSTALL')
    text=accept[0]['raw'];metadata=json.loads(text)['_pi0294']
    need(hashlib.sha256(text.encode()).hexdigest()==ref and metadata['sequence']==0
         and metadata['operation']=='install'and metadata['source_mpc_s']==5.,'QUALIFICATION_WIRE_IDENTITY')
    cancels=[r for r in planner if r['kind']=='lifecycle'and r.get('reason')=='CANCEL']
    need(len(cancels)==1 and cancels[0]['monotonic_s']>install[0]['monotonic_s'],'QUALIFICATION_CANCEL_AFTER_INSTALL')
    modes=[r for r in ros if r['kind']=='mode'];hold=[r for r in modes if r['msg']['data']=='HOLD'
        and r['monotonic_s']>cancels[0]['monotonic_s']]
    need(hold and any(r['msg']['data']=='WBC'and r['monotonic_s']>hold[0]['monotonic_s']for r in modes),
         'FRESH_ACTUAL_HOLD_AND_WBC_ACKNOWLEDGEMENTS')
    return ref


def audit_episode(folder,*,expected_source_sha,frozen_model=None):
    """Inspect only existing files; return proof and actual arrays, or no score.

    Complete scoring requires the independently pinned original full-body model
    so frozen robot properties are rechecked rather than trusted from a flag.
    The original scorers retain their exact 32s clock and physical predicates.
    """
    folder=Path(folder);result=read(folder/'RESULT.json');card=read(folder/'CARD.json')
    need(result['source_sha']==expected_source_sha,'PINNED_ACTUAL_RUNNING_SOURCE')
    need(card['task']in('T1','T2')and card['id']==folder.name,'ACTUAL_TASK_AND_EPISODE')
    hashes={'RESULT.json':sha(folder/'RESULT.json'),'CARD.json':sha(folder/'CARD.json')}
    if result['status']!='RAW_TASK_COMPLETE_REQUIRES_INDEPENDENT_CPU_AUDIT':
        return dict(status='INCOMPLETE_NATIVE_EPISODE_NO_TASK_SCORE',source_sha=expected_source_sha,
            native_status=result['status'],native_error=result.get('error'),quality=None,
            publication_count_from_empty_log='UNKNOWN_NOT_INFERRED',
            demonstration_admitted=False,raw_evidence=hashes,new_physics=0),None
    need(card['duration_s']==32 and card['warmup_sim_s']==5,'ORIGINAL_FULL_32S_CLOCK')
    need(result['observations']==641 and result['submitted_intents']==80
         and result['actual_plugin_steps']==18500 and not result['native_stops']
         and not result.get('recorder_error')and result['saved_RGB_pairs']==641,'COMPLETE_NATIVE_RESULT')
    import mujoco
    from PIL import Image
    from .task_plant import check_frozen_robot
    from .score import score_push
    from go2arx5_vla.target_sim.scorer import BOOL_FIELDS,score_episode
    need(isinstance(frozen_model,dict)and set(frozen_model)=={'path','sha256'},'PINNED_ORIGINAL_NATIVE_MODEL')
    need(sha(frozen_model['path'])==frozen_model['sha256'],'ORIGINAL_NATIVE_MODEL_BYTES')
    need(sha(folder/'task-scene.private.mjb')==read(folder/'PREEXECUTION.json')['model_sha256'],'ACTUAL_TASK_MODEL_BYTES')
    model=mujoco.MjModel.from_binary_path(str(folder/'task-scene.private.mjb'))
    original=mujoco.MjModel.from_binary_path(str(frozen_model['path']))
    need(model.nq==34 and model.nv==32 and model.nu==20
         and model.jnt_type[0]==mujoco.mjtJoint.mjJNT_FREE,'FREE_WHOLE_BODY_PLANT')
    frozen=check_frozen_robot(original,model)
    samples=rows(folder/'samples.jsonl');records=rows(folder/'committed_intents.jsonl');planner=rows(folder/'planner.jsonl')
    start=read(folder/'TASK_START.json');need(start['native_step']==2500 and start['time_s']==5.,'NATIVE_TASK_START')
    transactions=check_transactions(samples,records,planner,episode=folder.name)
    qualification_ref=check_qualification(folder,planner,rows(folder/'ros.jsonl'))
    need(qualification_ref not in transactions['task_refs'],'IO_PREFIX_IS_NOT_A_TASK_ACTION')
    t=np.asarray([s['time_s']for s in samples]);need(np.allclose(t,np.arange(641)*.05,atol=1e-8,rtol=0),'20HZ_TASK_GRID')
    pauses=rows(folder/'pause_intervals.jsonl');need(len(pauses)==640,'ALL_PHYSICAL_PAUSE_INTERVALS')
    for index,p in enumerate(pauses):
        need(p['index']==index,'PAUSE_INDEX');require_pause(p['before'],p['after'],2500+index*25)
    actual=ActualState(model);states=[];micro=[];source_clock=[];endpoint=[];native_rows=0;image_hashes={}
    journal=folder/'state.bin.journal';need(journal.stat().st_size<=256*1024**2,'FINITE_NATIVE_JOURNAL')
    with journal.open('rb')as f:
        while header:=f.read(4):
            need(len(header)==4,'JOURNAL_HEADER');size,=struct.unpack('<I',header)
            need(40<=size<=1024**2,'NATIVE_PACKET_BOUND');packet=parse_state(f.read(size));native_rows+=1
            need(packet['step']==native_rows and native_rows<=18500,'EVERY_ACTUAL_INTEGRATION_IN_ORDER')
            need(abs(packet['time_s']-native_rows*.002)<=1e-8,'500HZ_RAW_CLOCK')
            if native_rows<2500:continue
            state=actual.update(packet)
            if native_rows>2500:
                micro.append([int(state['robot_object_contact']),int(state['native_grasp']),
                    int(state['left_object_contact']and state['right_object_contact']),float(state['object_pose'][2,3]),int(not state['safety_valid'])])
                source_clock.append(packet['contact_time_s']-5.);endpoint.append(packet['time_s']-5.)
            if(native_rows-2500)%25:continue
            index=(native_rows-2500)//25;sample=samples[index]
            need(sample['native_step']==native_rows and sample['raw_time_s']==packet['time_s'],'ACTUAL_OBSERVATION_RAW_CLOCK')
            for key in ('qpos','qvel','ctrl'):need(np.array_equal(sample[key],packet[key]),'EXACT_NATIVE_'+key.upper())
            for key in ('tcp','object_pose','object_linear_velocity','object_angular_velocity'):
                need(np.allclose(sample[key],state[key],atol=1e-10,rtol=0),'RECOMPUTED_ACTUAL_'+key.upper())
            for key in BOOL_FIELDS:need(type(sample[key])is bool and sample[key]==bool(state[key]),'RECOMPUTED_ACTUAL_'+key.upper())
            need(bool(sample['robot_object_contact'])==state['robot_object_contact'],'ACTUAL_OBJECT_CONTACT')
            states.append(state);calibration=read(folder/'images'/f'{index:06d}-calibration.json')
            for camera in ('body_rgb','wrist_rgb'):
                path=folder/'images'/f'{index:06d}-{camera}.png';image_hashes[path.name]=sha(path)
                with Image.open(path)as image:need(image.size==(256,256)and image.mode=='RGB','DOUBLE_RGB_FORMAT')
                c=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_CAMERA,camera);need(c>=0,'ORIGINAL_CAMERA')
                transform=np.eye(4);transform[:3,:3]=actual.data.cam_xmat[c].reshape(3,3)@np.diag([1,-1,-1]);transform[:3,3]=actual.data.cam_xpos[c]
                focal=128/np.tan(np.deg2rad(model.cam_fovy[c])/2);cal=calibration[camera]
                need(cal['timestamp_s']==packet['time_s']and np.allclose(cal['T_world_optical'],transform,atol=1e-10,rtol=0)
                     and np.allclose(cal['K'],[[focal,0,128],[0,focal,128],[0,0,1]],atol=1e-10,rtol=0),'RGB_ACTUAL_STATE_CALIBRATION')
    need(native_rows==18500 and len(states)==641 and len(micro)==16000,'COMPLETE_HIGH_RATE_AND_CAMERA_STATE')
    need(len(list((folder/'images').iterdir()))==641*3,'NO_MISSING_OR_EXTRA_TASK_IMAGES')
    arrays={k:np.asarray([bool(s[k])for s in states])for k in BOOL_FIELDS}
    for k in ('object_pose','object_linear_velocity','object_angular_velocity','robot_object_contact'):
        arrays[k]=np.asarray([s[k]for s in states])
    arrays.update(time_s=t,tcp_world=np.asarray([s['tcp']for s in states]),contact_time_s=np.asarray(endpoint),
        native_contact_source_time_s=np.asarray(source_clock),contact_audit=np.asarray(micro))
    need(np.allclose(source_clock,np.arange(16000)*.002,atol=1e-8,rtol=0)
         and np.allclose(endpoint,np.arange(1,16001)*.002,atol=1e-8,rtol=0),'CONTACT_SOURCE_AND_ENDPOINT_CLOCKS')
    if card['task']=='T2':quality=score_push(arrays,card,complete=True)
    else:quality=score_episode(arrays,goal_center=[*card['goal_xy'],.46],initial_object_pose=arrays['object_pose'][0],
        table_height_m=.44,complete=True,evidence_source='REAL_MUJOCO_FULL_GO2_ARX5')
    quality['evidence_source']='REAL_MUJOCO_FULL_GO2_ARX5'
    highrate_safe=bool((arrays['contact_audit'][:,4]==0).all())
    need(highrate_safe,'EVERY_ACTUAL_NATIVE_STATE_SAFE')
    for name in ('samples.jsonl','committed_intents.jsonl','planner.jsonl','physics.jsonl','state.bin.journal',
            'pause_intervals.jsonl','CURRENT_NATIVE_QUALIFICATION.json','PREEXECUTION.json'):
        hashes[name]=sha(folder/name)
    proof=dict(status='COMPLETE_TASK_QUALITY_PASS'if quality['persistent_success']else'COMPLETE_TASK_QUALITY_FAIL',
        source_sha=expected_source_sha,quality=quality,observations=641,native_integrations=18500,task_integrations=16000,
        submitted_and_installed=80,executed_intervals=640,RGB_pairs=641,actual_state_byte_identity=True,
        independent_state_predicates=True,submitted_EEF10_wire_identity=True,BASE_supervision=False,
        frozen_native_robot=frozen,source_age_max_s=transactions['source_age_max_s'],
        image_hashes=image_hashes,raw_evidence=hashes,completed_interval_evidence=transactions['intervals'],
        installed_ROS_float32_absolute_tolerance=8*float(np.finfo(np.float32).eps),
        selected_MPC_consumption='UNKNOWN',demonstration_admitted=bool(quality['persistent_success']),new_physics=0,new_model_queries=0)
    return json.loads(json.dumps(proof,default=serial,allow_nan=False)),arrays
