"""Independent original task scoring of saved actual native arrays; never plan."""
import hashlib,json,struct
from pathlib import Path
import numpy as np
from .task_capture_runtime import read,sha
from .task_state import ActualState,parse_state
from .task_clock import FROZEN_FIELDS
from .base_intent import native_document
from .component_teacher_audit import installed_parity
from .mps2_runtime import FOLDER

def rows(path):return [json.loads(s) for s in Path(path).read_text().splitlines()] if Path(path).exists() else []

def run(root,episode):
    import mujoco
    from scripts.pi0292.durable import write
    from go2arx5_vla.target_sim.scorer import BOOL_FIELDS,score_episode
    from .score import score_push
    root=Path(root);folder=root/FOLDER/'episodes'/episode;output=folder/'INDEPENDENT_CPU_AUDIT.json'
    if output.exists():return read(output)
    result=read(folder/'RESULT.json');card=read(folder/'CARD.json');samples=rows(folder/'samples.jsonl')
    if result['native_stops']:
        proof=dict(status='VALID_NATIVE_SAFETY_FAIL',stops=result['native_stops'],source_sha=result['source_sha'],new_physics=0)
        write(output,proof);return proof
    if not samples:
        proof=dict(status='ENGINEERING_PRETASK_INCOMPLETE',source_sha=result['source_sha'],result_status=result['status'],
            reason=result.get('error'),observations=0,submitted_intents=0,native_integrations=result['actual_plugin_steps'],new_physics=0)
        write(output,proof);return proof
    model=mujoco.MjModel.from_binary_path(str(folder/'task-scene.private.mjb'));actual=ActualState(model)
    start=read(folder/'TASK_START.json');sample_packets={};micro=[];xyz_micro=[];endpoint=[];native_rows=0
    with (folder/'state.bin.journal').open('rb') as f:
        while raw:=f.read(4):
            size,=struct.unpack('<I',raw);p=parse_state(f.read(size));native_rows+=1
            if p['step']!=native_rows:raise ValueError('MPS2_NATIVE_SEQUENCE')
            relative=p['step']-start['native_step']
            if relative<0:continue
            state=actual.update(p)
            if relative%25==0:sample_packets[relative//25]=p
            if relative:
                micro.append([int(state['robot_object_contact']),int(state['native_grasp']),int(state['left_object_contact'] and state['right_object_contact']),
                    float(state['object_pose'][2,3]),int(not state['safety_valid'])]);xyz_micro.append(state['object_pose'][:3,3].copy());endpoint.append(relative*.002)
    commands=rows(folder/'committed_intents.jsonl');intents=[r for r in commands if r['phase']=='SUBMITTED_REQUIRES_INSTALL_ACK'];acks=[r for r in commands if r['phase']=='INSTALL_ACK']
    planner=rows(folder/'planner.jsonl');accepted={r['ref_id']:r for r in planner if r['kind']=='accept'}
    complete=result['status']=='RAW_TASK_COMPLETE_REQUIRES_INDEPENDENT_CPU_AUDIT' and len(samples)==round(card['duration_s']*20)+1
    if result.get('recorder_error'):raise ValueError('MPS2_RECORDER_ERROR')
    n=len(samples);t=np.asarray([s['time_s'] for s in samples])
    if not np.allclose(t,np.arange(n)/20,atol=1e-10,rtol=0):raise ValueError('MPS2_ACTUAL_20HZ')
    expected=(n-2)//8+1 if n>1 else 0
    if len(intents)!=expected or len(acks)!=expected or len(accepted)!=expected:raise ValueError('MPS2_PREFIX8_TRANSACTION_COUNTS')
    if complete and native_rows!=start['native_step']+(n-1)*25:raise ValueError('MPS2_COMPLETE_NATIVE_CLOCK')
    hashes={}
    for i,s in enumerate(samples):
        p=sample_packets[i]
        if s['native_step']!=p['step'] or any(not np.array_equal(s[k],p[k]) for k in ('qpos','qvel','ctrl')):raise ValueError('MPS2_ACTUAL_NATIVE_BYTES')
        state=actual.update(p)
        for key in ('tcp','object_pose','object_linear_velocity','object_angular_velocity'):
            if not np.allclose(state[key],s[key],atol=1e-12,rtol=0):raise ValueError('MPS2_NATIVE_FK_REPLAY:'+key)
        for key in BOOL_FIELDS:
            if bool(state[key])!=bool(s[key]):raise ValueError('MPS2_NATIVE_FLAG_REPLAY:'+key)
        calibration=read(folder/'images'/f'{i:06d}-calibration.json')
        for camera in ('body_rgb','wrist_rgb'):
            if calibration[camera]['timestamp_s']!=p['time_s']:raise ValueError('MPS2_IMAGE_SOURCE_CLOCK')
            image=folder/'images'/f'{i:06d}-{camera}.png';hashes[image.name]=sha(image)
        if i and (s['active_intent_sequence']!=intents[(i-1)//8]['sequence'] or s['executed_prefix_index']!=(i-1)%8):raise ValueError('MPS2_PREFIX_LABEL')
    previous=np.zeros(3)
    for seq,(intent,ack) in enumerate(zip(intents,acks)):
        p=sample_packets[seq*8];a=np.asarray(intent['action_EEF13']);wire=accepted[intent['wire_sha256']]
        if intent['index']!=seq*8 or intent['sequence']!=seq or ack['native_step']!=p['step'] or ack['wire_sha256']!=intent['wire_sha256']:raise ValueError('MPS2_INSTALL_TRANSACTION')
        if hashlib.sha256(wire['raw'].encode()).hexdigest()!=intent['wire_sha256']:raise ValueError('MPS2_WIRE_IDENTITY')
        actual_doc=json.loads(wire['raw']);rebuilt=native_document(a,np.asarray(intent['actual_BASE_mount']),base_valid=True,previous_velocity=previous)
        if any(actual_doc[k]!=rebuilt[k] for k in ('ee','base')):raise ValueError('MPS2_SUBMITTED_LABEL')
        previous=a[7,10:13].copy()
        if intent['source_mpc_s']!=round(p['time_s']*1e9)/1e9 or not 0<=wire['receiver_mpc_s']-wire['source_mpc_s']<=.025:raise ValueError('MPS2_SOURCE_AGE')
        if not np.array_equal(a[:8],np.asarray(intent['scheduled_prefix_EEF13'])):raise ValueError('MPS2_EXACT_PREFIX8')
        if intent['canonical_safety_gate']['decision']=='REJECT':raise ValueError('MPS2_REJECTED_REFERENCE_INSTALLED')
        certificate=intent['persistent']['certificate'];reqhash=certificate['input_sha256']
        qfolder=root/FOLDER/'requests'/episode/f'{seq:04d}'
        if sha(qfolder/'INPUT.private.json')!=reqhash:raise ValueError('MPS2_PLANNING_INPUT_SHA')
        with np.load(qfolder/'H40.private.npz',allow_pickle=False) as saved:
            from go2arx5_vla.task_data.codec import pose10
            expected_pose=np.stack([pose10(p,g) for p,g in zip(saved['poses'],a[:,9])])
            if not np.allclose(expected_pose,a[:,:10],atol=1e-12,rtol=0):raise ValueError('MPS2_FIRST_H40_LABEL_REPLAY')
    pauses=rows(folder/'pause_intervals.jsonl')
    if complete and len(pauses)!=n-1:raise ValueError('MPS2_ALL_PAUSES_REQUIRED')
    if any(any(r['before'][k]!=r['after'][k] for k in FROZEN_FIELDS) for r in pauses):raise ValueError('MPS2_PAUSE_HISTORY_CHANGED')
    if any(r['kind']=='lifecycle' for r in planner):raise ValueError('MPS2_UNEXPECTED_REFERENCE_LIFECYCLE')
    parity=installed_parity(planner) if intents else dict(transactions=0)
    micro=np.asarray(micro).reshape(-1,5);xyz_micro=np.asarray(xyz_micro).reshape(-1,3);longest=run_ticks=0
    for flag in micro[:,1]:run_ticks=run_ticks+1 if flag else 0;longest=max(longest,run_ticks)
    obj=np.asarray([s['object_pose'] for s in samples]);tcp=np.asarray([s['tcp'] for s in samples]);disp=np.linalg.norm(xyz_micro[:,:2]-obj[0,:2,3],axis=1);rise=xyz_micro[:,2]-obj[0,2,3]
    target=np.asarray(intents[0]['persistent']['target']) if intents else tcp[0]
    distances=np.linalg.norm(tcp[:,:3,3]-target[:3,3],axis=1)
    geometry=[s['teacher_geometry'] for s in samples]
    milestones=dict(position_progress=bool(distances[0]-distances.min()>=.02),safe_precontact=any(g['precontact_geometry_ready'] for g in geometry),
        actual_contact=bool(micro[:,0].any()),manipulation=bool(longest*.002>=.3) if card['task']=='T1' else bool(np.any((micro[:,0]>0)&(disp>=.01)&(rise<=.005)&(micro[:,1]==0))))
    quality=None
    arrays={k:np.asarray([s[k] for s in samples]) for k in BOOL_FIELDS}
    arrays.update(time_s=t,tcp_world=tcp,object_pose=obj,object_linear_velocity=np.asarray([s['object_linear_velocity'] for s in samples]),
        object_angular_velocity=np.asarray([s['object_angular_velocity'] for s in samples]),robot_object_contact=np.asarray([s['robot_object_contact'] for s in samples]),
        contact_time_s=np.asarray(endpoint),native_contact_source_time_s=np.asarray(endpoint)-.002,contact_audit=micro)
    if card['duration_s']==32. and complete:
        quality=score_push(arrays,card,complete=True) if card['task']=='T2' else score_episode(arrays,goal_center=[*card['goal_xy'],.46],initial_object_pose=obj[0],table_height_m=.44,complete=True,evidence_source='REAL_MUJOCO_FULL_GO2_ARX5')
        quality['evidence_source']='REAL_MUJOCO_FULL_GO2_ARX5'
    status=('COMPLETE_TASK_QUALITY_PASS' if quality['persistent_success'] else 'COMPLETE_TASK_QUALITY_FAIL') if quality is not None else (
        ('PASS_SHORT_MILESTONES_AND_INTEGRITY' if all(milestones.values()) else 'VALID_SHORT_MILESTONE_FAIL_INTEGRITY_PASS') if complete else 'INCOMPLETE_ACTUAL_PREFIX_PRESERVED')
    proof=dict(status=status,id=episode,source_sha=result['source_sha'],demonstration_source=card['demonstration_source'],
        observations=n,submitted_and_installed=len(intents),native_integrations=native_rows,task_duration_s=float(t[-1]),complete=complete,
        quality=quality,milestones=milestones,actual_contact_ticks=int(micro[:,0].sum()),longest_grasp_s=longest*.002,
        max_actual_object_displacement_m=float(disp.max(initial=0.)),max_actual_object_rise_m=float(rise.max(initial=0.)),
        approach_distance_initial_min_final_m=[float(distances[0]),float(distances.min()),float(distances[-1])],
        actual_TCP_initial_final_xyz=tcp[[0,-1],:3,3].tolist(),RGB_pairs=n,installation=parity,source_labels_exact=True,
        selected_MPC_policy_consumption='UNKNOWN',stop_reason=result.get('stop_reason',result.get('error')),new_physics=0,new_model=0,selected='NONE',deployment='NONE')
    np.savez_compressed(folder/'ACTUAL_PHYSICS.npz',**arrays,tcp_goal_distance=distances,actual_object_xyz_500Hz=xyz_micro)
    write(folder/'RGB_HASHES.json',hashes)
    proof['raw_evidence']={n:sha(folder/n) for n in ('samples.jsonl','state.bin.journal','committed_intents.jsonl','planner.jsonl','pause_intervals.jsonl','physics.jsonl','ACTUAL_PHYSICS.npz','RGB_HASHES.json')}
    write(output,proof);return proof
