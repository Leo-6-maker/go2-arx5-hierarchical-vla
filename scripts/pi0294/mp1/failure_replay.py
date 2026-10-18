"""Exact recorded-command reconstruction, not another autonomous policy rollout."""
import argparse
import base64
import hashlib
import os
from pathlib import Path
import resource
import signal
import time

import numpy as np

from go2arx5_vla.task_data.codec import pose10
from go2arx5_vla.trajectory.so3 import geodesic_angle
from scripts.pi0292.durable import OwnerLock,write
from scripts.pi0292.eval_contract import matrix10
from scripts.pi0292.relay_protocol import unseal
from scripts.pi0294.e1.control_read import read
from scripts.pi0294.e1.collision import inspect_prefix,save_state
from scripts.pi0294.e1.initial_history import qualify_recorded_initial
from scripts.pi0294.episode import _robot_contact
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.scene import make_engine
from scripts.pi0294.s1_preflight import fingerprint
from scripts.pi0294.teacher import make_teacher
from . import failure_config as config
from .science import validate_runtime,Ledger


def verify_images(body,folder):
    images=body['payload']['images'];cameras={'external_rgb':'body_rgb','wrist_rgb':'wrist_rgb'}
    if len(images)!=2 or {v['camera'] for v in images}!=set(cameras):
        raise ValueError('EXACT_ORIGINAL_DOUBLE_CAMERA_INPUT')
    frame=body['tick']//25
    for row in images:
        raw=base64.b64decode(row['png_base64'],validate=True)
        if hashlib.sha256(raw).hexdigest()!=row['sha256']:
            raise ValueError('ORIGINAL_OBSERVATION_INLINE_CAMERA_HASH')
        path=Path(folder)/cameras[row['camera']]/f'{frame:04d}.png'
        if path.is_symlink() or sha(path)!=row['sha256']:
            raise ValueError('ORIGINAL_RENDERED_CAMERA_EQUALS_MODEL_INPUT')


def bound_inputs(root):
    root=Path(root);plan=read(root/'plans/FAILURE_ANCHORS.json');cases=config.validate_plan(plan)
    binding=read(root/'plans/SOURCE_FAILURE_INPUTS.private.json');source=Path(binding['source_root'])
    if source.resolve()!=(root.parent/'source-pair-01').resolve() or source.is_symlink():
        raise PermissionError('CURRENT_SOURCE_PAIR_READ_ONLY_INPUTS_ONLY')
    if sha(source/'RUN_MANIFEST.json')!=binding['source_manifest_sha256']:
        raise ValueError('SOURCE_PAIR_SCIENTIFIC_MANIFEST_CHANGED')
    for name,digest in binding['files'].items():
        p=source/name
        if (Path(name).is_absolute() or '..' in Path(name).parts or p.is_symlink()
                or not p.resolve().is_relative_to(source.resolve()) or sha(p)!=digest):
            raise ValueError('FIXED_FAILURE_RECONSTRUCTION_INPUT_CHANGED')
    if (source/'SOURCE_PHYSICS_STOP.json').exists():
        raise PermissionError('ORIGINAL_SOURCE_SAFETY_STOP_NOT_BYPASSED')
    return source,cases,binding


def observe_prefix(engine,sequence,poses,opening,observer):
    if observer is None:return
    before=fingerprint(engine)
    action=hashlib.sha256(poses.tobytes()+opening.tobytes()).hexdigest()
    observer(engine,sequence,poses,opening)
    if (fingerprint(engine)!=before
            or hashlib.sha256(poses.tobytes()+opening.tobytes()).hexdigest()!=action):
        raise RuntimeError('READONLY_REPLAY_OBSERVER_CHANGED_SOURCE_OR_ACTION')


def replay_case(root,source,case,ledger,guard,*,prefix_observer=None):
    identifier=case['attempt_id'];expected=config.ATTEMPTS[identifier]
    folder=source/'episodes'/identifier;card=read(folder/'CARD.json')
    adjudication=read(folder/'MP1_ADJUDICATION.json')
    if (card['split']!='TRAIN' or adjudication['status']!='VALID_TASK_FAIL'
            or adjudication['failure_stage']!='S1_PRE_EXECUTION_REJECT'
            or adjudication['live_native_failures'] or sha(folder/'RESULT.json')!=expected['result_sha256']):
        raise PermissionError('ONLY_FOUR_ALREADY_ADJUDICATED_SAFE_NATIVE_FAILURES')
    observations=sorted((folder/'relay/observations').glob('*.json'));last=expected['last_sequence']
    if [p.name for p in observations]!=[f'{i:03d}.json' for i in range(last+1)]:
        raise ValueError('EXACT_EXISTING_OBSERVATION_PREFIX_NO_REPLAY_EXTENSION')
    reference=read(folder/'references.json');forecasts=read(folder/'S1_PREFLIGHTS.json')
    if len(reference)!=last or len(forecasts)!=last+1:
        raise ValueError('EXACT_ACCEPTED_PREFIXES_AND_SINGLE_LAST_REJECT')
    with np.load(folder/'highrate.npz',allow_pickle=False) as z:recorded=z['rows'].copy()
    if len(recorded)!=expected['main_steps'] or not np.isfinite(recorded).all():
        raise ValueError('COMPLETE_SAVED_ACTUAL_HIGHRATE_REQUIRED')
    ledger.charge('main_starts',1)
    engine=make_engine(card,read(root/'INPUT_BINDINGS.private.json')['asset_root'])
    maximum=0.;states=[];comparison=None;reference_difference=0.
    try:
        first=engine.snapshot();teacher=make_teacher(card,first['object_pose'],engine)
        initial=read(folder/'INITIAL_STATE.json')
        if (hashlib.sha256(engine.xml.encode()).hexdigest()!=initial['model_xml_sha256']
                or not np.array_equal(first['qpos'],initial['qpos'])
                or not np.array_equal(first['qvel'],initial['qvel'])):
            raise ValueError('EXACT_ORIGINAL_INITIAL_MODEL_AND_STATE_REQUIRED')
        request=read(source/card['relay_request_relative'])
        original_case=dict(attempt_id=identifier,episode_uuid=request['episode_uuid'],card=card)
        for sequence in range(last+1):
            guard();state=engine.snapshot();state['robot_object_contact']=_robot_contact(engine,state)
            if sequence==0:qualify_recorded_initial(engine,folder,guard)
            teacher.observe(state)
            packet=read(folder/'relay/observations'/f'{sequence:03d}.json');body=unseal(packet)
            actual=pose10(state['tcp'],state['gripper_fraction'])
            if (body['sequence']!=sequence or body['tick']!=engine.steps or body['episode_uuid']!=request['episode_uuid']
                    or body['payload']['language']!=card['instruction']
                    or np.max(np.abs(actual-np.asarray(body['payload']['state_eef10'])))>1e-12):
                raise ValueError('RECONSTRUCTED_FRESH_OBSERVATION_STATE_MISMATCH')
            if sequence in case['sequences']:
                verify_images(body,folder)
                states.append(save_state(root/'evidence/TRAIN_STATES'/(identifier+'-S'+str(sequence).zfill(3)),engine,teacher,original_case,sequence,packet))
            bundle=read(folder/'relay/commands'/f'{sequence:03d}.json')
            if sequence in case['sequences']:
                target=root/'evidence/TRAIN_STATES'/states[-1]['snapshot_relative']/'ORIGINAL_ACTION_BUNDLE.private.json'
                write(target,bundle)
                states[-1].update(original_query_uuid=unseal(bundle['request'])['query_uuid'],
                    original_action_sha256=bundle['action']['sha256'],
                    original_action_bundle_sha256=sha(target),
                    original_noise_identity='REQUIRES_MATCHING_EXISTING_HOPPER_RECEIPT_NO_NEW_QUERY')
            action=np.asarray(unseal(bundle['action'])['payload']['trajectory_eef10'],dtype=np.float64)
            poses,opening=matrix10(action),action[:,9]
            fingerprint=hashlib.sha256(poses[:8].tobytes()+opening[:8].tobytes()).hexdigest()
            original=forecasts[sequence]
            if original['action_sha256']!=fingerprint or original['sequence']!=sequence*8:
                raise ValueError('ORIGINAL_S1_EXACT_INSTALLED_PREFIX_IDENTITY')
            if sequence in case['sequences']:
                observe_prefix(engine,sequence,poses[:8],opening[:8],prefix_observer)
            ack=folder/'relay/acks'/f'{sequence:03d}.json'
            if sequence==last:
                if ack.exists() or original['decision']!='REJECT':
                    raise ValueError('ONLY_ORIGINAL_ZERO_LIVE_REJECTED_PREFIX_INSPECTED')
                check,rows=inspect_prefix(engine,poses[:8],opening[:8],ledger,'LAST-REJECT-'+identifier)
                if (check['decision']!='REJECT' or check['first_violation_offset_ticks']!=expected['offset']
                        or check['predicted_native_safety']!=original['predicted_native_safety']
                        or check['shadow_integrations']!=original['shadow_integrations']):
                    raise ValueError('ORIGINAL_PREDICTED_REJECTION_NOT_REPRODUCED')
                np.savez_compressed(root/'evidence'/(identifier+'-rejected-shadow.npz'),observed_rows=rows)
                comparison=dict(original_S1=original,reconstructed_shadow=check,
                    source_actual_TCP=state['tcp'].tolist(),source_actual_gripper_fraction=float(state['gripper_fraction']),
                    source_actual_object_pose=state['object_pose'].tolist(),
                    requested_vs_shadow_end_XYZ_m=float(np.linalg.norm(np.asarray(check['requested_TCP_at_end'])[:3,3]-np.asarray(check['actual_TCP_at_end'])[:3,3])),
                    requested_vs_shadow_end_SO3_rad=float(geodesic_angle(np.asarray(check['requested_TCP_at_end'])[:3,:3],np.asarray(check['actual_TCP_at_end'])[:3,:3])),
                    actual_source_teacher_stage=teacher.stage,teacher_stage_is_policy_phase=False,
                    actual_source_native_grasp=bool(state['native_grasp']),
                    original_live_native_safety_faults=0,shadow_prediction_is_actual_live_collision=False,
                    same_state_expert_or_hybrids_run=False)
                break
            if not ack.is_file() or original['decision']!='PASS':
                raise ValueError('NO_REPLAY_OF_UNACKNOWLEDGED_OR_REJECTED_ACTION')
            wire_ack=unseal(read(ack));wire_command=unseal(bundle['command'])
            if (wire_ack['sequence']!=sequence or wire_ack['payload']['tick_after']!=(sequence+1)*200
                    or wire_ack['payload']['integrations']!=200
                    or wire_ack['payload']['command_sha256']!=bundle['command']['sha256']
                    or wire_command['tick']!=sequence*200
                    or sha(folder/'relay'/f'physics-{sequence:03d}.npz')!=wire_ack['payload']['physics_receipt_sha256']):
                raise ValueError('EXACT_ACTUAL_ACK_AND_NATIVE_PROGRESS_CHAIN')
            encoded=np.stack([pose10(p,g) for p,g in zip(poses,opening)])
            difference=float(np.max(np.abs(encoded-np.asarray(reference[sequence]['rows10']))))
            reference_difference=max(reference_difference,difference)
            if reference[sequence]['committed_prefix_count']!=8 or difference>1e-12:
                raise ValueError('WIRE_ACTION_EQUALS_ORIGINAL_INSTALLED_REFERENCE')
            with ledger.steps('shadow_integrations',200,'HISTORY-'+identifier+'-'+str(sequence)) as meter:
                for i in range(200):
                    guard()
                    if i and i%25==0:
                        s=engine.snapshot();s['robot_object_contact']=_robot_contact(engine,s);teacher.observe(s)
                    before=engine.steps
                    try:engine.step(poses[i//25],float(opening[i//25]))
                    finally:meter['actual']+=engine.steps-before
                    s=engine.snapshot()
                    observed=np.r_[engine.data.time,engine.data.qpos,engine.data.qvel,engine.data.ctrl,
                        engine.data.xfrc_applied[engine.ids['tool']],float(not s['safety_valid'])]
                    difference=float(np.max(np.abs(observed-recorded[engine.steps-1])));maximum=max(maximum,difference)
                    if difference>1e-10 or engine.safety:
                        raise ValueError('ORIGINAL_ACTUAL_DYNAMICS_CONTROL_HISTORY_NOT_REPRODUCED')
        if engine.steps!=expected['main_steps'] or len(states)!=2 or comparison is None:
            raise ValueError('EXACT_TWO_SELECTED_STATES_AND_COMPLETE_SAVED_PROGRESS')
        return dict(attempt_id=identifier,task=card['task'],original_main_steps=expected['main_steps'],
            reconstruction_shadow_steps=engine.steps,maximum_original_highrate_difference=maximum,
            maximum_original_wire_installed_reference_difference=reference_difference,
            states=states,first_rejected_prefix_diagnosis=comparison,
            initial_static_IK_qualifier_reproduced=True,
            state_restoration='FROM_INITIAL_MODEL_STATIC_IK_AND_ALL_ACTUAL_ACKED_COMMANDS_NO_QPOS_INJECTION',
            original_process_RNG_not_recorded=True,frozen_D0_engine_uses_no_random_sampling=True,
            reconstructed_full_live_IK_control_python_teacher_history_saved=True,
            original_hidden_state_bitwise_or_RNG_fingerprint_identity_claimed=False)
    finally:engine.close()


def run(root):
    root=Path(root);m=validate_runtime(root,'4080-simulation');stage=read(root/'MP1_STAGE_BINDING.json')
    if not config.is_diagnostic(stage):raise PermissionError('EXACT_CURRENT_FAILURE_HISTORY_DIAGNOSTIC_SCOPE')
    source,cases,binding=bound_inputs(root);ledger=Ledger(root,'core');ledger.guard();lock=OwnerLock(root/'FAILURE_HISTORY_OWNER.lock')
    start=time.process_time();deadline=stage['diagnostic_deadline_epoch'];results=[]
    remaining=int(config.CPU_CAPS['core']-read(ledger.path)['Core_CPU_s'])
    if remaining<=0:raise PermissionError('ORIGINAL_DIAGNOSTIC_CPU_REMAINING_REQUIRED')
    if not stage['diagnostic_created_epoch']<=time.time()<deadline<=stage['diagnostic_created_epoch']+config.WALL_S:
        raise PermissionError('SINGLE_ORIGINAL_DIAGNOSTIC_WINDOW_NO_RESET')
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});resource.setrlimit(resource.RLIMIT_CPU,(remaining,remaining))
    def guard():
        ledger.guard()
        if time.time()>=deadline or time.process_time()-start>=remaining:raise TimeoutError('BOUNDED_CURRENT_FAILURE_RECONSTRUCTION')
    def expire(signum,frame):raise TimeoutError('ORIGINAL_CURRENT_FAILURE_RECONSTRUCTION_WALL')
    signal.signal(signal.SIGALRM,expire);signal.alarm(max(1,int(deadline-time.time())))
    try:
        for case in cases:
            result=replay_case(root,source,case,ledger,guard);results.append(result)
            write(root/'evidence/RECONSTRUCTION_PROGRESS.json',dict(completed=len(results),planned=4,cases=results))
        bound_inputs(root)
        value=dict(status='PASS_FOUR_CURRENT_FAILURES_EXACT_HISTORY_AND_FIRST_ILLEGAL_GEOMETRY',
            source_sha=m['code']['vla_commit'],original_scientific_source_sha=binding['source_code_sha'],
            cases=results,selected_states=sum(len(x['states']) for x in results),
            new_neural_queries=0,new_live_episodes=0,new_expert_labels=0,
            original_results_modified=False,valid_negatives_repeated_as_policy_evaluation=False,
            frozen_WBC_or_S1_or_plant_modified=False)
        write(root/'evidence/FAILURE_HISTORY_RESULT.json',value);return value
    except BaseException as exc:
        detail=dict(status='DIAGNOSTIC_RECONSTRUCTION_GAP_PRESERVED',error_type=type(exc).__name__,error=str(exc),completed=len(results))
        write(root/'evidence/FAILURE_HISTORY_ERROR.private.json',detail);ledger.stop_physics('DIAGNOSTIC_RECONSTRUCTION_UNRESOLVED',detail);raise
    finally:signal.alarm(0);ledger.cpu(time.process_time()-start);lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();v=run(a.root);print(v['status'],flush=True)
