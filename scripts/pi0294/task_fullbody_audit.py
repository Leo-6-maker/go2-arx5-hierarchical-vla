"""Saved actual full-body task audit. No model, physics integration or Teacher.

Reuse the frozen object-task predicates. High-rate contacts retain both the
constraint source time and the endpoint of the actual integration they belong to.
"""
import hashlib
import json
from pathlib import Path
import struct
import numpy as np
from .task_capture_runtime import cpu_stage,read,sha
from .task_state import ActualState,parse_state
from .task_clock import FROZEN_FIELDS


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines()]


def run(root,episode):
    import mujoco
    from scripts.pi0292.durable import write
    from .base_intent import native_document
    from .score import score_push
    from go2arx5_vla.target_sim.scorer import BOOL_FIELDS,score_episode
    root=Path(root);folder=root/'episodes'/episode
    with cpu_stage(root,'AUDIT_'+episode,cores=1,cpu_s_cap=1800):
        if (folder/'FULLBODY_CPU_AUDIT.json').exists():raise FileExistsError('FIRST_AUDIT_PRESERVED')
        result=read(folder/'RESULT.json');card=read(folder/'CARD.json')
        if result['status']!='RAW_TASK_COMPLETE_REQUIRES_INDEPENDENT_CPU_AUDIT':raise ValueError('COMPLETE_NATIVE_TASK_REQUIRED')
        model=mujoco.MjModel.from_binary_path(str(folder/'task-scene.private.mjb'))
        if model.jnt_type[0]!=mujoco.mjtJoint.mjJNT_FREE or model.nq!=34 or model.nv!=32 or model.nu!=20:
            raise ValueError('ACTUAL_FULLBODY_PLANT_REQUIRED')
        start=read(folder/'TASK_START.json');samples=rows(folder/'samples.jsonl')
        intents=[r for r in rows(folder/'committed_intents.jsonl') if r['phase']=='SUBMITTED_REQUIRES_INSTALL_ACK']
        acks=[r for r in rows(folder/'committed_intents.jsonl') if r['phase']=='INSTALL_ACK']
        planner=rows(folder/'planner.jsonl');accepted={r['ref_id']:r for r in planner if r['kind']=='accept'}
        installed={r['ref_id'] for r in planner if r['kind']=='installed' and r.get('ref_id')}
        if (len(samples)!=641 or len(intents)!=640 or len(acks)!=640 or len(accepted)!=640
                or result['native_stops'] or result.get('recorder_error')):raise ValueError('COMPLETE_OBSERVATION_REFERENCE_IMAGE_COUNTS_REQUIRED')
        t=np.asarray([s['time_s'] for s in samples])
        if not np.allclose(t,np.arange(641)/20,atol=1e-8,rtol=0):raise ValueError('ACTUAL_20HZ_GRID')
        exact=True;native_rows=0;task_rows=0;micro=[];contact_source=[];endpoints=[];sample_packets={}
        actual=ActualState(model)
        with (folder/'state.bin.journal').open('rb') as f:
            while raw:=f.read(4):
                if len(raw)!=4:raise ValueError('JOURNAL_LENGTH_TRUNCATED')
                size,=struct.unpack('<I',raw);packet=parse_state(f.read(size));native_rows+=1
                if packet['step']!=native_rows:raise ValueError('ACTUAL_NATIVE_INTEGRATION_SEQUENCE')
                if packet['step']<start['native_step']:continue
                if (packet['step']-start['native_step'])%25==0:
                    index=(packet['step']-start['native_step'])//25
                    if index<641:sample_packets[index]=packet
                if packet['step']>start['native_step']:
                    state=actual.update(packet);task_rows+=1
                    micro.append([int(state['robot_object_contact']),int(state['native_grasp']),int(state['left_object_contact'] and state['right_object_contact']),float(state['object_pose'][2,3]),int(not state['safety_valid'])])
                    contact_source.append(packet['contact_time_s']-start['time_s'])
                    endpoints.append(packet['time_s']-start['time_s'])
        if (native_rows!=18500 or task_rows!=16000 or len(sample_packets)!=641
                or not np.allclose(contact_source,np.arange(16000)*.002,atol=1e-8,rtol=0)
                or not np.allclose(endpoints,np.arange(1,16001)*.002,atol=1e-8,rtol=0)):
            raise ValueError('EXACT_HIGH_RATE_NATIVE_CONTACT_AND_INTEGRATION_CLOCKS')
        image_hashes={};previous=np.zeros(3);source_ages=[]
        for i,sample in enumerate(samples):
            packet=sample_packets[i]
            if (sample['native_step']!=packet['step'] or not np.array_equal(sample['qpos'],packet['qpos'])
                    or not np.array_equal(sample['qvel'],packet['qvel']) or not np.array_equal(sample['ctrl'],packet['ctrl'])):
                raise ValueError('OBSERVATION_NOT_EXACT_ACTUAL_NATIVE_STATE')
            calibration=read(folder/'images'/f'{i:06d}-calibration.json')
            for camera in ('body_rgb','wrist_rgb'):
                if calibration[camera]['timestamp_s']!=packet['time_s']:raise ValueError('RGB_STATE_SOURCE_TIME_MISMATCH')
                path=folder/'images'/f'{i:06d}-{camera}.png';image_hashes[path.name]=sha(path)
            if i==640:continue
            item=intents[i];action=np.asarray(item['action_EEF13']);wire=accepted[item['wire_sha256']]
            if (item['index']!=i or item['sequence']!=i or item['wire_sha256'] not in installed
                    or acks[i]['wire_sha256']!=item['wire_sha256'] or acks[i]['native_step']!=packet['step']
                    or hashlib.sha256(wire['raw'].encode()).hexdigest()!=item['wire_sha256']):
                raise ValueError('SAME_TRANSACTION_INSTALL_IDENTITY')
            actual_wire=json.loads(wire['raw'])
            rebuilt=native_document(action,np.asarray(item['actual_BASE_mount']),base_valid=True,previous_velocity=previous)
            if any(actual_wire[k]!=rebuilt[k] for k in ('ee','base')):raise ValueError('ACTION_LABEL_DIFFERS_FROM_ACTUALLY_SUBMITTED_REFERENCE')
            previous=action[0,10:13];age=wire['receiver_mpc_s']-wire['source_mpc_s'];source_ages.append(age)
            if not 0<=age<=.025:raise ValueError('SOURCE_AGE_BOUND')
        pauses=rows(folder/'pause_intervals.jsonl')
        if len(pauses)!=640:raise ValueError('MISSING_NONREALTIME_WAIT_EVIDENCE')
        for p in pauses:
            if any(p['before'][k]!=p['after'][k] for k in FROZEN_FIELDS):raise ValueError('WAIT_CHANGED_PHYSICS_FSM_HISTORY')
        arrays={k:np.asarray([s[k] for s in samples]) for k in BOOL_FIELDS}
        for key in ('object_pose','object_linear_velocity','object_angular_velocity','robot_object_contact'):
            arrays[key]=np.asarray([s[key] for s in samples])
        arrays.update(time_s=t,tcp_world=np.asarray([s['tcp'] for s in samples]),
            contact_time_s=np.asarray(endpoints),native_contact_source_time_s=np.asarray(contact_source),contact_audit=np.asarray(micro))
        # score_push's frozen grid names integration endpoints contact_time_s;
        # preserve the separate, original pre-step constraint timestamps above.
        if card['task']=='T2':quality=score_push(arrays,card,complete=True)
        else:quality=score_episode(arrays,goal_center=[*card['goal_xy'],.46],initial_object_pose=arrays['object_pose'][0],table_height_m=.44,complete=True,evidence_source='REAL_MUJOCO_FULL_GO2_ARX5')
        quality['evidence_source']='REAL_MUJOCO_FULL_GO2_ARX5'
        base=np.asarray([s['qpos'][:3] for s in samples]);base_displacement=base-base[0]
        achieved=float(np.max(base_displacement[:,0]));base_ok=achieved>=card['minimum_actual_BASE_translation_m']
        quality.update(BASE_translation_ok=bool(base_ok),persistent_success=bool(quality['persistent_success'] and base_ok),domain='FULL_GO2_ARX5_FROZEN_NATIVE_WBC_TASK_SIM')
        object_xyz=arrays['object_pose'][:,:3,3]
        quality.update(maximum_actual_lift_m=float(np.max(object_xyz[:,2]-object_xyz[0,2])),maximum_actual_carry_xy_m=float(np.max(np.linalg.norm(object_xyz[:,:2]-object_xyz[0,:2],axis=1))))
        target=folder/'FULLBODY_PHYSICS.npz'
        if target.exists():raise FileExistsError('FIRST_ARRAY_PRESERVED')
        np.savez_compressed(target,**arrays)
        write(folder/'FULLBODY_RGB_HASHES.json',image_hashes)
        proof=dict(status='COMPLETE_TASK_QUALITY_PASS' if quality['persistent_success'] else 'COMPLETE_TASK_QUALITY_FAIL',
            quality=quality,source_sha=result['source_sha'],observations=641,submitted_and_installed=640,
            native_integrations=native_rows,task_integrations=task_rows,actual_RGB_pairs=641,
            native_safety_stops=0,raw_state_byte_identity=True,submitted_label_identity=True,
            source_age_max_s=max(source_ages),pause_intervals=len(pauses),non_realtime=True,
            frozen_native_robot_properties=read(folder/'PREEXECUTION.json')['frozen_robot'],
            actual_BASE_max_forward_m=achieved,actual_BASE_final_translation_m=base_displacement[-1].tolist(),
            native_leg_ctrl_max=np.max(np.abs(np.asarray([s['ctrl'][:12] for s in samples])),axis=0).tolist(),
            contact_clock_semantics='INTEGRATION_ENDPOINT_GRID_WITH_SEPARATE_ACTUAL_PRESTEP_CONSTRAINT_SOURCE',
            physics_sha256=sha(target),RGB_index_sha256=sha(folder/'FULLBODY_RGB_HASHES.json'),
            raw_evidence={n:sha(folder/n) for n in ('samples.jsonl','state.bin.journal','planner.jsonl','committed_intents.jsonl','physics.jsonl','pause_intervals.jsonl')},
            no_new_physics=0,new_model_queries=0,selected='NONE',deployment='NONE')
        write(folder/'FULLBODY_CPU_AUDIT.json',proof);print(json.dumps(proof))
