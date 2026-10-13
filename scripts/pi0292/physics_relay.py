"""Fresh CoreLab observation and one-time prefix8 execution using sealed Engine.

This entry is conditional S6 work, not a collector restart. It creates no model,
opens no network listener and does not modify the teacher, driver or physics.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback
import uuid

import numpy as np
from PIL import Image

from .durable import read,write
from .eval_contract import FRAME,TCP,gate_action,matrix10
from .neural_relay import REQUEST_KEYS,require_case_phase,require_teacher_proof
from .relay_protocol import PROTOCOL,digest,exact_keys,finite_vector,sealed,unseal,uuid_string,validate_identity,validate_observation
from .resource_guard import Budget
from .train_adapter import sha

COMMAND_KEYS=('protocol','kind','run_uuid','episode_uuid','sequence','tick','query_uuid','reference_uuid',
              'observation_sha256','anchor_sha256','prefix_eef10','prefix_sha256','advance_integrations','frame','tcp','expires')


def accept_prefix(command,action,observation,*,request,identity,now):
    cmd=unseal(command);exact_keys(cmd,COMMAND_KEYS)
    obs=unseal(observation);act=unseal(action)
    query=unseal(request);exact_keys(query,REQUEST_KEYS)
    exact_keys(obs,('protocol','kind','run_uuid','episode_uuid','sequence','tick','payload'))
    validate_observation(obs['payload'])
    exact_keys(act,('protocol','kind','run_uuid','episode_uuid','sequence','tick','payload'))
    value=act['payload'];exact_keys(value,('query_uuid','reference_uuid','observation_sha256','anchor_sha256',
                                      'identity','trajectory_eef10','offsets_ns'))
    for key in ('run_uuid','episode_uuid','sequence','tick'):
        if cmd[key]!=obs[key] or act[key]!=obs[key] or query[key]!=obs[key]:raise ValueError('COMMAND_OBSERVATION_IDENTITY_CHANGED')
    if (query['protocol']!=PROTOCOL or query['kind']!='student_request' or query['identity']!=identity
            or query['payload']!=obs['payload'] or cmd['expires']!=query['expires']):
        raise ValueError('ORIGINAL_QUERY_PAYLOAD_IDENTITY_OR_DEADLINE_CHANGED')
    if (obs['protocol']!=PROTOCOL or obs['kind']!='observation' or type(cmd['sequence']) is not int
            or type(cmd['tick']) is not int or cmd['tick']!=cmd['sequence']*200):
        raise ValueError('EXACT_PHYSICS_TICK_AND_SEQUENCE_REQUIRED')
    if (cmd['protocol']!=PROTOCOL or act['protocol']!=PROTOCOL or cmd['kind']!='install_prefix8'
            or act['kind']!='action' or cmd['frame']!=FRAME or cmd['tcp']!=TCP
            or type(cmd['advance_integrations']) is not int or cmd['advance_integrations']!=200
            or not now<cmd['expires']<=now+60):
        raise ValueError('COMMAND_PROTOCOL_CLOCK_FRAME_OR_PREFIX_CHANGED')
    if value['identity']!=identity:raise ValueError('COMMAND_MODEL_OR_TEACHER_IDENTITY_CHANGED')
    if cmd['observation_sha256']!=observation['sha256'] or cmd['anchor_sha256']!=digest(obs['payload']['state_eef10']):
        raise ValueError('COMMAND_FRESH_OBSERVATION_ANCHOR_CHANGED')
    for key in ('query_uuid','reference_uuid','observation_sha256','anchor_sha256'):
        if value[key]!=cmd[key] or query[key]!=cmd[key]:raise ValueError('ACTION_COMMAND_BINDING_CHANGED')
    uuid_string(cmd['query_uuid']);uuid_string(cmd['reference_uuid'])
    trajectory=value['trajectory_eef10']
    if not isinstance(trajectory,list) or len(trajectory)!=40:raise ValueError('FULL_GENERATED_H40_REQUIRED_FOR_GATE')
    for row in trajectory:finite_vector(row,10)
    if (not isinstance(value['offsets_ns'],list) or any(type(x) is not int for x in value['offsets_ns'])
            or value['offsets_ns']!=[i*50000000 for i in range(40)] or cmd['prefix_eef10']!=trajectory[:8]):
        raise ValueError('PREFIX_NOT_THE_FIRST_EIGHT_NATIVE_POINTS')
    if cmd['prefix_sha256']!=digest(cmd['prefix_eef10']):raise ValueError('PREFIX_HASH_CHANGED')
    gate=gate_action(trajectory,obs['payload']['state_eef10'],obs['tick']*.002,cmd['reference_uuid'])
    if gate['decision'] not in ('ACCEPT','WARN'):raise ValueError('FROZEN_SAFETY_GATE_REJECT')
    return cmd,gate


def require_s6_admission(value,*,run,case):
    if (value['status']!='ALLOW_ORIGINAL_S6_GATED_PHASE' or value['run_uuid']!=run
            or value['release']!='261bc5b37f20aaff1ad7d5742252d3dc1a91fa65'
            or value['original_T0']!=1791860304 or value['original_deadline']!=1791946704
            or value['main_status']!='MAIN_COMPLETED_FIXED_S4096'
            or value['offline_status']!='PAIRED_OFFLINE_RECORDS_ANALYZED_NOT_TASK_SUCCESS'
            or value['authority_kind']!='ACTUAL_HOPPER_RECEIPTS_READBACK_BY_EXISTING_LOCAL_PC'):
        raise PermissionError('ACTUAL_POST_MAIN_AND_OFFLINE_S6_AUTHORITY_REQUIRED')
    if case['profile']=='TEACHER_REFERENCE':
        if case['kind']=='RELAY_QUALIFICATION':
            if case['layout_id'] not in ('T291000','T291001'):raise PermissionError('REGISTERED_TEACHER_RELAY_LAYOUT_REQUIRED')
        elif case['kind']=='CONFIRM':
            require_teacher_proof(value['teacher_relay_proof'],run)
            if case['layout_id'] not in {f'E292{i:03d}' for i in range(8)}:raise PermissionError('REGISTERED_TEACHER_CONFIRM_LAYOUT_REQUIRED')
            # The confirmation teacher follows the safe B1 training canaries.
            canaries=value['phase_proofs'].get('B1_canaries',[])
            if (len(canaries)!=2 or {r['layout_id'] for r in canaries}!= {'T291000','T291001'}
                    or not all(r['safety_valid'] is True and r['attempted'] is True for r in canaries)
                    or sum(r['persistent_success'] is True for r in canaries)<1):
                raise PermissionError('B1_TRAIN_CANARY_GATE_REQUIRED_BEFORE_CONFIRM_TEACHER')
        else:raise PermissionError('NO_EXTRA_TEACHER_COLLECTION')
    else:
        require_teacher_proof(value['teacher_relay_proof'],run)
        require_case_phase(case,value['phase_proofs'])


def run(args):
    root=Path(args.root);request_path=Path(args.request);request=read(request_path)
    if (root.name!='20261013T025824Z-localpc-pi0292' or not request_path.resolve().is_relative_to(root.resolve())
            or sha(__file__)!=request['endpoint_sha256'] or time.time()>=1791943104):
        raise PermissionError('REGISTERED_ORIGINAL_RUN_AND_FIXED_ENDPOINT_REQUIRED')
    manifest_path=request_path.parent/'RUN_MANIFEST.json'
    if sha(manifest_path)!=request['run_manifest_sha256']:raise PermissionError('PRELAUNCH_S6_MANIFEST_REQUIRED')
    manifest=read(manifest_path)
    if (manifest['run_uuid']!=root.name or manifest['authority']['go_comment']!=6137102979
            or manifest['execution']['profile']!='4080-simulation'):
        raise PermissionError('OWN_EXISTING_CPU_SIMULATION_PROFILE_REQUIRED')
    if (os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.environ.get('LIBGL_ALWAYS_SOFTWARE')!='1'
            or os.environ.get('__EGL_VENDOR_LIBRARY_FILENAMES')!='/usr/share/glvnd/egl_vendor.d/50_mesa.json'
            or 'MUJOCO_EGL_DEVICE_ID' in os.environ):
        raise PermissionError('CORELAB_CPU_MESA_RENDERING_ONLY')
    case=request['case'];identity=request['identity'];validate_identity(identity)
    if identity['profile']!=case['profile']:raise PermissionError('CASE_AND_POLICY_IDENTITY_CHANGED')
    admission_path=Path(request['admission'])
    if not admission_path.resolve().is_relative_to(root.resolve()) or sha(admission_path)!=request['admission_sha256']:
        raise PermissionError('BOUND_LOCAL_PC_S6_ADMISSION_REQUIRED')
    require_s6_admission(read(admission_path),run=root.name,case=case)
    code_seal=read(root/'CODE_SEAL.json')
    if code_seal['source_sha']!='35290803cd82e10a164daaa53af5743dd43054ee':raise ValueError('ORIGINAL_PHYSICAL_SOURCE_REQUIRED')
    for name,digest_value in code_seal['files'].items():
        if sha(root/'code'/name)!=digest_value:raise ValueError('SEALED_PHYSICS_OR_DRIVER_CHANGED')
    import scripts
    scripts.__path__.append(str(root/'code/scripts'));sys.path.insert(0,str(root/'code/src'))
    from scripts.pi0291_arx5_data.config import PARAMETERS,canonical_hash,DOMAIN
    from scripts.pi0291_arx5_data.engine import Engine
    from scripts.pi0291_arx5_data.teacher import Teacher
    from scripts.pi0291_arx5_data.score import BOOL_FIELDS,saved_score
    from go2arx5_vla.task_data.codec import pose10
    from go2arx5_vla.task_data.io import jsonable
    plan=read(root/'ATTEMPT_PLAN.json');identifier=case['attempt_id']
    if (identifier not in plan['attempts'] or plan['attempts'][identifier]['category']!='S6_ONLY_NOT_COLLECTION'
            or plan['parameter_sha256']!=canonical_hash(PARAMETERS)):
        raise PermissionError('REGISTERED_S6_ATTEMPT_AND_UNCHANGED_PARAMETERS_REQUIRED')
    card=dict(plan['attempts'][identifier]['card'])
    card['asset_root']=read(root/'PRIVATE_ASSET_BINDING.json')['native_asset_root']
    folder=root/'episodes'/identifier
    if folder.exists():raise FileExistsError('Never repeat or overwrite an existing physical attempt')
    budget=Budget(root);budget.start(identifier,code_seal['source_sha']);folder.mkdir()
    relay=folder/'relay';relay.mkdir()
    for name in ('observations','commands','acks','intents'): (relay/name).mkdir()
    write(folder/'CARD.json',{**card,'mode':'normal','domain':DOMAIN,'source_sha':code_seal['source_sha'],
        'policy_profile':case['profile'],'S6_not_training_data':True})
    state=dict(status='STARTED_PAUSE_ON_OBSERVATION',pid=os.getpid(),
        proc_start=Path('/proc/self/stat').read_text().split()[21],run_uuid=root.name,
        episode_uuid=request['episode_uuid'],sequence=0,tick=0,commands=0,profile=case['profile'])
    uuid_string(state['episode_uuid']);write(relay/'STATE.json',state)
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]))
    def stop(signum,frame):raise TimeoutError('OWN_PHYSICAL_EPISODE_900_SECOND_OR_ORIGINAL_CUTOFF')
    signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop)
    signal.alarm(max(1,min(880,int(1791943104-time.time()))))
    engine=None;teacher=None;physical=[];samples=[];highrate=[];references=[]
    def capture():
        s=engine.snapshot();step=engine.steps//25
        if abs(s['time_s']-engine.steps*.002)>1e-8:raise RuntimeError('ACTUAL_PHYSICS_CLOCK_CHANGED')
        teacher.observe(s);images,geometry=engine.render(s)
        if not samples:
            from OpenGL import GL
            vendor=GL.glGetString(GL.GL_VENDOR).decode();renderer=GL.glGetString(GL.GL_RENDERER).decode()
            if vendor!='Mesa' or 'llvmpipe' not in renderer:raise RuntimeError('CPU_RENDER_BACKEND_NOT_VERIFIED')
            write(folder/'RENDER_PROCESS.json',dict(pid=os.getpid(),backend='mesa_cpu',GL_VENDOR=vendor,GL_RENDERER=renderer))
        obs=dict(instruction=card['instruction'],actual_tcp=s['tcp'],actual_gripper=s['gripper_fraction'],
            actual_gripper_speed=s['gripper_speed'],timestamp_s=s['time_s'],valid=True)
        for camera,pixels in images.items():
            path=folder/camera/f'{step:04d}.png';path.parent.mkdir(exist_ok=True);Image.fromarray(pixels).save(path)
            obs[camera]=dict(path=path.relative_to(folder).as_posix(),sha256=sha(path),frame_index=step,
                timestamp_s=s['time_s'],snapshot_id=f'{identifier}:{engine.steps}',source='actual_mujoco_renderer',valid=True)
        physical.append({k:s[k] for k in ['time_s','object_pose','tcp','object_linear_velocity','object_angular_velocity',*BOOL_FIELDS]})
        samples.append(dict(snapshot_id=f'{identifier}:{engine.steps}',policy_obs=obs,geometry=geometry,
            causal_history=dict(previous_samples=step,reset=False,abort=False,takeover=False),supervision=None,
            execution=dict(actual_tcp=s['tcp'],actual_finger_q=s['finger_q'],actual_finger_qd=s['finger_qd'],
                actuator_force=s['actuator_force'],actual_joint_q=s['arm_q'],actual_joint_qd=s['arm_qd']),
            privileged=dict(teacher_stage=teacher.stage,object_pose=s['object_pose'],native_grasp=s['native_grasp'])))
        budget.progress(identifier,engine.steps,len(samples));budget.guard()
        if not s['safety_valid']:raise RuntimeError('REAL_SAFETY_VIOLATION')
        return s
    try:
        engine=Engine(card,'normal');first=engine.snapshot();teacher=Teacher(card,first['object_pose'],'normal');current=capture()
        for sequence in range(80):
            if engine.steps!=sequence*200:raise RuntimeError('FRESH_OBSERVATION_TICK_NOT_AFTER_LAST_PREFIX')
            latest=samples[-1]['policy_obs'];image_rows=[]
            for wire,raw_camera in [('external_rgb','body_rgb'),('wrist_rgb','wrist_rgb')]:
                path=folder/latest[raw_camera]['path'];raw=path.read_bytes()
                image_rows.append(dict(camera=wire,png_base64=base64.b64encode(raw).decode(),sha256=sha(path)))
            payload=dict(snapshot_uuid=str(uuid.uuid4()),language=card['instruction'],
                state_eef10=pose10(current['tcp'],current['gripper_fraction']).tolist(),images=image_rows,frame=FRAME,tcp=TCP)
            observation=sealed(dict(protocol=PROTOCOL,kind='observation',run_uuid=root.name,episode_uuid=state['episode_uuid'],
                sequence=sequence,tick=engine.steps,payload=payload))
            write(relay/'observations'/f'{sequence:03d}.json',observation)
            if case['profile']=='TEACHER_REFERENCE':
                feedback=teacher.plan(current)
                write(relay/f'teacher-{sequence:03d}.json',dict(observation_sha256=observation['sha256'],
                    trajectory_eef10=np.stack([pose10(p,g) for p,g in zip(feedback['poses'],feedback['opening'])]).tolist(),
                    source='ACTUAL_CURRENT_CORELAB_FEEDBACK_TEACHER_NOT_MODEL'))
            state.update(status='WAIT_FRESH_ACTION_PHYSICS_PAUSED',sequence=sequence,tick=engine.steps,
                observation_sha256=observation['sha256'],observation_published_epoch=time.time())
            write(relay/'STATE.json',state);deadline=time.time()+60
            command_path=relay/'commands'/f'{sequence:03d}.json'
            while not command_path.exists():
                budget.guard()
                if (relay/'STOP.json').exists():raise RuntimeError('LOCAL_PC_STOPPED_THIS_ATTEMPT')
                if time.time()>=deadline:raise TimeoutError('PHYSICS_WAIT_ACTION_60_SECOND_TIMEOUT')
                time.sleep(.02)
            bundle=read(command_path);exact_keys(bundle,('command','action','request'))
            command,gate=accept_prefix(bundle['command'],bundle['action'],observation,
                request=bundle['request'],identity=identity,now=time.time())
            intent_path=relay/'intents'/f'{sequence:03d}.json'
            if intent_path.exists():raise RuntimeError('COMMAND_ALREADY_RESERVED_DO_NOT_REPLAY')
            write(intent_path,dict(status='RESERVED_BEFORE_FIRST_INTEGRATION',command_sha256=bundle['command']['sha256'],
                tick_before=engine.steps,requested_integrations=200,query_uuid=command['query_uuid'],gate=gate))
            prefix=np.asarray(command['prefix_eef10'],dtype=np.float64);targets=matrix10(prefix)
            tick_before=engine.steps;segment_start=len(highrate)
            for point in range(8):
                samples[-1]['supervision']=dict(ref_id=command['reference_uuid'],point_index=point,
                    committed=targets[point],committed_gripper=float(prefix[point,9]),actually_committed=True)
                for _ in range(25):
                    if time.time()>=command['expires']:raise TimeoutError('PREFIX_COMMAND_EXPIRED_NO_MORE_INTEGRATION')
                    try:engine.step(targets[point],float(prefix[point,9]))
                    finally:highrate.append(np.r_[engine.data.time,engine.data.qpos,engine.data.qvel,engine.data.ctrl,
                        engine.data.xfrc_applied[engine.ids['tool']],float(bool(engine.safety))])
                current=capture()
            if engine.steps!=tick_before+200:raise RuntimeError('ACTUAL_PREFIX_INTEGRATION_COUNT_CHANGED')
            if time.time()>=command['expires']:raise TimeoutError('PREFIX_ACK_60_SECOND_TIMEOUT')
            segment=relay/f'physics-{sequence:03d}.npz'
            np.savez_compressed(segment,rows=np.asarray(highrate[segment_start:]),tick_before=tick_before,tick_after=engine.steps)
            with segment.open('rb') as handle:os.fsync(handle.fileno())
            references.append(dict(command_sha256=bundle['command']['sha256'],prefix_sha256=command['prefix_sha256'],
                reference_uuid=command['reference_uuid'],actual_committed_points=8,actual_integrations=200,
                unused_H40_tail_not_committed=True))
            ack=sealed(dict(protocol=PROTOCOL,kind='ack',run_uuid=root.name,episode_uuid=state['episode_uuid'],sequence=sequence,
                tick=tick_before,payload={key:command[key] for key in ('query_uuid','reference_uuid','prefix_sha256')}|
                    dict(command_sha256=bundle['command']['sha256'],tick_after=engine.steps,integrations=200,physics_receipt_sha256=sha(segment))))
            write(relay/'acks'/f'{sequence:03d}.json',ack)
            state.update(commands=sequence+1,tick=engine.steps,status='PREFIX_ACKNOWLEDGED_NEXT_OBSERVATION_ALLOWED');write(relay/'STATE.json',state)
        if engine.steps!=16000 or len(samples)!=641:raise RuntimeError('FULL_32_SECONDS_AND_641_ACTUAL_FRAMES_REQUIRED')
        state['status']='VALID_COMPLETE_REAL_S6_PHYSICS_NOT_AUTOMATIC_TASK_SUCCESS'
    except BaseException as error:
        state.update(status='PHYSICAL_ATTEMPT_STOPPED_PRESERVED',error_type=type(error).__name__,error=str(error),
            real_safety_violation=bool(engine is not None and engine.safety))
        write(folder/'ERROR.json',dict(error=str(error),traceback=traceback.format_exc()))
        if engine is not None and engine.safety:
            abort=engine.snapshot()
            write(folder/'ACTUAL_SAFETY_ABORT_SNAPSHOT.json',dict(time_s=abort['time_s'],reasons=engine.safety,
                safety_valid=abort['safety_valid'],paired_RGB=False,no_extra_integrations=True))
            if not physical or abs(physical[-1]['time_s']-abort['time_s'])>1e-10:
                physical.append({k:abort[k] for k in ['time_s','object_pose','tcp','object_linear_velocity','object_angular_velocity',*BOOL_FIELDS]})
    finally:
        signal.alarm(0)
        steps=engine.steps if engine is not None else 0
        if physical:
            arrays={key:np.asarray([s[key] for s in physical]) for key in physical[0]};arrays['tcp_world']=arrays.pop('tcp')
            np.savez_compressed(folder/'physics.npz',**arrays)
        if highrate:np.savez_compressed(folder/'highrate.npz',rows=np.asarray(highrate),nq=engine.model.nq,nv=engine.model.nv,
            layout=np.asarray(['time','qpos','qvel','finite_joint_motor_ctrl','zero_external_wrench','safety_flag']))
        for name,value in [('samples.json',samples),('references.json',references)]:
            (folder/name).write_text(json.dumps(jsonable(value),indent=2)+'\n')
        if teacher is not None:(folder/'TEACHER_EVENTS.json').write_text(json.dumps(jsonable(teacher.events),indent=2)+'\n')
        if engine is not None:
            write(folder/'scene.xml.json',dict(xml=engine.xml,sha256=hashlib.sha256(engine.xml.encode()).hexdigest()));engine.close()
        result=dict(id=identifier,status=state['status'],profile=case['profile'],physics_steps=steps,paired_samples=len(samples),
            acknowledged_commands=state['commands'],real_safety_violation=state.get('real_safety_violation',False),
            S6_only_not_training_data=True,selected='NONE',deployment='NONE')
        if physical:result['score']=saved_score(folder)
        write(folder/'RESULT.json',result);state.update(finished_epoch=time.time(),result=result);write(relay/'STATE.json',state)
        budget.progress(identifier,steps,len(samples));budget.finish(identifier,state['status'],0.)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--request',required=True)
    print(json.dumps(run(parser.parse_args())))
