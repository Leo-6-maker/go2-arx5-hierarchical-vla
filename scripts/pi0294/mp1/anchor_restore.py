"""Zero-integration saved-state reconstruction checks, not policy qualification."""
import base64
import argparse
import copy
import hashlib
from pathlib import Path
import pickle
import random
import os
import resource
import signal
import time

import numpy as np

from go2arx5_vla.task_data.codec import pose10
from scripts.pi0292.durable import read,write,OwnerLock
from scripts.pi0292.relay_protocol import unseal
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.s1_preflight import data_state,observed_state
from .failure_config import ATTEMPTS,selected_sequences
from .scope import validate,charge_cpu

EXCLUDED={'model','data','ik_data','renderer'}


def load_owned_snapshot(root,record):
    """Only unpickle bytes already sealed by this owner's closed B1 receipt."""
    root=Path(root)
    if (record['original_attempt_id'] not in ATTEMPTS
            or record['sequence'] not in selected_sequences(record['original_attempt_id'])
            or record['tick']!=record['sequence']*200 or record['split']!='TRAIN'
            or not record['physically_safety_valid']):
        raise PermissionError('EXACT_EIGHT_OWNED_CURRENT_FAILURE_NONINITIAL_ANCHORS')
    relative=Path(record['snapshot_relative'])
    if relative.is_absolute() or len(relative.parts)!=1 or str(relative) in ['.','..']:
        raise PermissionError('SINGLE_EXACT_OWNED_ANCHOR_DIRECTORY')
    folder=root/'evidence/TRAIN_STATES'/relative
    if folder.is_symlink():raise PermissionError('NO_SNAPSHOT_DIRECTORY_SYMLINK')
    for name,digest in [('integration.npz',record['full_integration_state_sha256']),
                        ('history.private.pkl',record['history_sha256']),
                        ('ORIGINAL_ACTION_BUNDLE.private.json',record['original_action_bundle_sha256'])]:
        p=folder/name
        if p.is_symlink() or not p.resolve().is_relative_to(folder.resolve()) or sha(p)!=digest:
            raise ValueError('OWNED_SNAPSHOT_MEMBER_HASH_BEFORE_DESERIALIZATION')
    with np.load(folder/'integration.npz',allow_pickle=False) as z:
        if set(z.files)!={'live','ik','observed'}:raise ValueError('COMPLETE_LIVE_IK_AND_CONTROLLER_OBSERVATION')
        arrays={k:z[k].copy() for k in z.files}
    if any(a.ndim!=1 or not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError('FINITE_COMPLETE_SNAPSHOT_STATE')
    history=pickle.loads((folder/'history.private.pkl').read_bytes())
    if set(history)!={'engine_python','teacher','python_rng','numpy_rng'} or EXCLUDED&set(history['engine_python']):
        raise ValueError('COMPLETE_OWNED_HISTORY_NO_MODEL_OR_RUNTIME_OBJECT_INJECTION')
    packet=read(folder/'observation.json');body=unseal(packet)
    if (packet['sha256']!=record['observation_sha256'] or body['sequence']!=record['sequence']
            or body['tick']!=record['tick'] or body['episode_uuid']!=record['original_episode_uuid']):
        raise ValueError('EXACT_ORIGINAL_ANCHOR_OBSERVATION_PACKET')
    images=body['payload']['images']
    if len(images)!=2 or {i['camera'] for i in images}!={'external_rgb','wrist_rgb'}:
        raise ValueError('EXACT_OWNED_ANCHOR_DOUBLE_CAMERA_INPUT')
    for image in images:
        if hashlib.sha256(base64.b64decode(image['png_base64'],validate=True)).hexdigest()!=image['sha256']:
            raise ValueError('ORIGINAL_ANCHOR_INLINE_RGB_HASH')
    return arrays,history,body


def restore_and_check(engine,record,arrays,history,*,mj=None):
    if mj is None:import mujoco as mj
    if hashlib.sha256(engine.xml.encode()).hexdigest()!=record['model_xml_sha256']:
        raise ValueError('ORIGINAL_ANCHOR_XML_IDENTITY')
    fields=set(engine.__dict__)-EXCLUDED
    if fields!=set(history['engine_python']):
        raise ValueError('EXACT_FROZEN_ENGINE_PYTHON_FIELDS_NO_MISSING_HISTORY')
    for name,value in history['engine_python'].items():setattr(engine,name,copy.deepcopy(value))
    kind=mj.mjtState.mjSTATE_INTEGRATION
    for name,key in [('data','live'),('ik_data','ik')]:
        if arrays[key].shape!=(mj.mj_stateSize(engine.model,kind),):
            raise ValueError('EXACT_NATIVE_INTEGRATION_STATE_DIMENSION')
        mj.mj_setState(engine.model,getattr(engine,name),arrays[key],kind)
        if not np.array_equal(data_state(engine.model,getattr(engine,name)),arrays[key]):
            raise ValueError('NATIVE_LIVE_AND_IK_STATE_RESTORE_EXACT')
    if engine.steps!=record['tick'] or engine.safety or not np.array_equal(observed_state(engine),arrays['observed']):
        raise ValueError('SAVED_CONTROLLER_CACHE_TARGETS_CLOCK_AND_STATE_RESTORE')
    original_python=random.getstate();original_numpy=np.random.get_state()
    try:
        random.setstate(history['python_rng']);np.random.set_state(history['numpy_rng'])
        if (random.getstate()!=history['python_rng']
                or pickle.dumps(np.random.get_state())!=pickle.dumps(history['numpy_rng'])):
            raise ValueError('DIAGNOSTIC_SAVED_RNG_ROUNDTRIP')
        teacher=copy.deepcopy(history['teacher'])
        if pickle.dumps(teacher)!=pickle.dumps(history['teacher']):raise ValueError('SAVED_TEACHER_HISTORY_ROUNDTRIP')
        state=engine.snapshot()
        maximum=float(np.max(np.abs(observed_state(engine)-arrays['observed'])))
        if engine.steps!=record['tick'] or maximum>1e-10 or not state['safety_valid']:
            raise ValueError('ZERO_INTEGRATION_FORWARD_SNAPSHOT_CHANGED_SAVED_STATE')
        return dict(actual_state_eef10=pose10(state['tcp'],state['gripper_fraction']).tolist(),
            maximum_saved_observed_difference=maximum,full_live_IK_controller_history_restored=True,
            teacher_history_restored=True,diagnostic_saved_RNG_roundtrip=True,
            original_historical_RNG_fingerprint_identity_claimed=False,
            restored_tick=engine.steps,physics_integrations=0,
            dynamic_future_prefix_parity_proven=False)
    finally:random.setstate(original_python);np.random.set_state(original_numpy)


def run(root,binding_relative):
    root=Path(root);relative=Path(binding_relative)
    if relative.is_absolute() or '..' in relative.parts or (root/relative).is_symlink():
        raise PermissionError('SINGLE_SEALED_ANCHOR_RESTORE_BINDING')
    m=validate(root,'4080-cpu-data');binding=root/relative;q=read(binding)
    if (sha(binding)!=m['execution']['anchor_restore_binding_sha256']
            or q['parent_run']!=root.name or q['maximum_CPU_s']!=600
            or q['new_neural_queries'] or q['new_physics']
            or q['engine_zero_step_starts_max']!=8
            or not q['created_epoch']<=time.time()<q['deadline_epoch']<=q['created_epoch']+600):
        raise PermissionError('ORIGINAL_MP1_ZERO_INTEGRATION_RESTORE_CPU_SCOPE')
    child=root/'children/sbc512-failure-history-01';source=read(child/'RUN_MANIFEST.json')
    result=child/'evidence/FAILURE_HISTORY_RESULT.json';closure=child/'evidence/CLOSED_RECEIPT.private.json'
    if (sha(result)!=q['result_sha256'] or sha(closure)!=q['closed_receipt_sha256']
            or read(closure)['status']!='CURRENT_FAILURE_HISTORY_CHILD_CLOSED_READBACK'
            or read(read(child/'OUTPUT_BINDING.private.json')['lease'])['status']!='CLOSED'):
        raise PermissionError('ONLY_HASH_BOUND_CLOSED_ACTUAL_B1_INPUTS')
    value=read(result)
    if value['status']!='PASS_FOUR_CURRENT_FAILURES_EXACT_HISTORY_AND_FIRST_ILLEGAL_GEOMETRY':
        raise PermissionError('ACTUAL_QUALIFIED_CURRENT_FAILURE_RECONSTRUCTION_REQUIRED')
    states=[s for c in value['cases'] for s in c['states']]
    if (len(states)!=8 or len({s['id'] for s in states})!=8
            or {(s['original_attempt_id'],s['sequence']) for s in states}
            !={(i,s) for i in ATTEMPTS for s in selected_sequences(i)}):
        raise PermissionError('EXACT_EIGHT_SEALED_STATES_NO_NEW_ANCHOR_SELECTION')
    frozen=source['code']['unchanged_numerical_files'];seal=read(root/m['code']['source_seal_relative'])
    if any(seal['files'].get(n)!=v for n,v in frozen.items()):
        raise PermissionError('FROZEN_D0_S1_NUMERICAL_SOURCE_UNCHANGED')
    lock=OwnerLock(root/'ZERO_STEP_ANCHOR_RESTORE_OWNER.lock');start=time.process_time()
    output=root/'evidence/ANCHOR_ZERO_STEP_RESTORE_01';output.mkdir(mode=0o700)
    rows=[];starts=0
    import mujoco
    from scripts.pi0294.scene import make_engine
    original_step=mujoco.mj_step;original_step1=mujoco.mj_step1;original_step2=mujoco.mj_step2
    def forbidden(*args,**kwargs):raise PermissionError('NO_PHYSICAL_INTEGRATION_IN_ZERO_STEP_RESTORE_AUDIT')
    mujoco.mj_step=mujoco.mj_step1=mujoco.mj_step2=forbidden
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});resource.setrlimit(resource.RLIMIT_CPU,(600,600))
    def expire(signum,frame):raise TimeoutError('ORIGINAL_ANCHOR_RESTORE_600S_BOUND')
    signal.signal(signal.SIGALRM,expire);signal.alarm(max(1,int(q['deadline_epoch']-time.time())))
    try:
        assets=read(child/'INPUT_BINDINGS.private.json')['asset_root']
        for record in states:
            if time.time()>=q['deadline_epoch']:raise TimeoutError('ORIGINAL_RESTORE_DEADLINE')
            arrays,history,body=load_owned_snapshot(child,record);starts+=1
            engine=make_engine(record['card'],assets)
            try:
                witness=restore_and_check(engine,record,arrays,history)
                if np.max(np.abs(np.asarray(witness.pop('actual_state_eef10'))-body['payload']['state_eef10']))>1e-12:
                    raise ValueError('RESTORED_NATIVE_OBSERVATION_EQUALS_ORIGINAL_MODEL_INPUT')
                if body['payload']['language']!=record['card']['instruction']:
                    raise ValueError('SAME_ORIGINAL_LANGUAGE_BINDING')
                rows.append(dict(id=record['id'],task=record['task'],sequence=record['sequence'],
                    source_observation_sha256=record['observation_sha256'],
                    full_integration_state_sha256=record['full_integration_state_sha256'],
                    history_sha256=record['history_sha256'],**witness))
            finally:engine.close()
        if sha(result)!=q['result_sha256'] or sha(closure)!=q['closed_receipt_sha256']:
            raise ValueError('CLOSED_B1_EVIDENCE_CHANGED_DURING_ZERO_STEP_AUDIT')
        report=dict(status='PASS_EIGHT_SAVED_FULL_STATE_ZERO_INTEGRATION_RESTORES',
            rows=rows,states=8,engine_zero_step_starts=starts,new_physics=0,new_neural_queries=0,
            original_b1_result_sha256=q['result_sha256'],source_sha=m['code']['vla_commit'],
            saved_snapshot_static_restore_only=True,future_prefix_dynamic_parity='STILL_REQUIRED_IN_BOUND_SHADOW_SCOPE',
            original_historical_unrecorded_hidden_state_or_RNG_identity_claimed=False)
        write(output/'RESULT.json',report);return report
    except BaseException as exc:
        write(output/'FAILURE.private.json',dict(status='ZERO_STEP_RESTORE_GAP_PRESERVED',error_type=type(exc).__name__,error=str(exc),completed=len(rows),engine_zero_step_starts=starts,new_physics=0));raise
    finally:
        mujoco.mj_step,mujoco.mj_step1,mujoco.mj_step2=original_step,original_step1,original_step2
        signal.alarm(0);charge_cpu(root,time.process_time()-start);lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--binding',required=True)
    a=p.parse_args();print(run(a.root,a.binding)['status'],flush=True)
