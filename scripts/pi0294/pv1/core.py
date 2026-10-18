"""Autonomous PV1 source DEV through the original episode, relay and native S1."""
import argparse
import json
import os
from pathlib import Path
import time

from scripts.pi0292.durable import OwnerLock,read,write
from scripts.pi0294 import episode,relay_teacher,s1_preflight,r3_teacher
from scripts.pi0294.e0_core import classify
from scripts.pi0294.e1.core import E1ProtectedTeacher
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.scene_cards import digest
from .capture import counted_model,qualify_counter
from .core_ledger import Ledger
from .evaluation_plan import session_spec,remaining_panel
from .runtime import validate_runtime


class Counter:
    def __init__(self,root,ledger,inputs,card):
        self.root,self.ledger,self.inputs=Path(root),ledger,inputs
        self.plan=dict(cards=[card])
    def start(self):self.ledger.charge('engine_starts',1)


def qualify(root,label):
    root=Path(root);validate_runtime(root,'4080-simulation');ledger=Ledger(root);ledger.guard()
    spec=session_spec(root,label);inputs=read(root/'INPUT_BINDINGS.private.json')
    lock=OwnerLock(root/'PV1_SOURCE_OWNER.lock');began=time.process_time()
    try:
        if (root/'evidence/COUNTER_NATIVE_QUALIFICATION.json').exists():
            raise FileExistsError('PV1_PRESERVE_FIRST_ZERO_INTEGRATION_COUNTER_QUALIFICATION')
        qualify_counter(Counter(root,ledger,inputs,spec['cards'][0]))
        return read(root/'evidence/COUNTER_NATIVE_QUALIFICATION.json')
    finally:ledger.cpu(time.process_time()-began);lock.close()


def run(root,card_path):
    root=Path(root);m=validate_runtime(root,'4080-simulation');ledger=Ledger(root);ledger.guard()
    inputs=read(root/'INPUT_BINDINGS.private.json');card_path=Path(card_path)
    if not card_path.is_absolute():card_path=root/card_path
    if card_path.is_symlink() or not card_path.resolve().is_relative_to(root.resolve()):
        raise PermissionError('PV1_REGISTERED_AUTONOMOUS_CARD_ONLY')
    card=read(card_path);session=card['pv1_session_label'];spec=session_spec(root,session)
    prior=read(root/'plans/PRIOR_EVALUATION_OUTCOMES.json')
    if m['execution']['sealed_inputs'].get('plans/PRIOR_EVALUATION_OUTCOMES.json')!=sha(root/'plans/PRIOR_EVALUATION_OUTCOMES.json'):
        raise PermissionError('PV1_SETTLED_PRIOR_CONDITIONS_REQUIRED_BEFORE_ANY_REPLACEMENT')
    prior=prior+[read(p) for p in (root/'episodes').glob('*/PV1_ADJUDICATION.json')]
    expected=next((c for c in remaining_panel(prior,spec) if c['id']==card['id']),None)
    if (expected is None or {k:v for k,v in card.items() if not k.startswith('relay_request_')}!=expected
            or card.get('e1_intervention_collection') or card['teacher_takeover'] is not False):
        raise PermissionError('PV1_EXACT_PREDECLARED_AUTONOMOUS_CONDITION_NOT_A_TEACHER_ROLLOUT')
    qualification=read(root/'evidence/COUNTER_NATIVE_QUALIFICATION.json')
    if (qualification['status']!='PASS_ORIGINAL_VS_COUNTED_NATIVE_MODEL_STATE_XML_AND_TWO_RGB'
            or qualification['integrations'] or qualification['model_constructor_attempts']!=2):
        raise PermissionError('PV1_ZERO_TIME_NATIVE_COUNTER_PARITY_BEFORE_POLICY_PHYSICS')
    ready=read(root/'plans'/session/'REMOTE_READY.json');registry=read(root/'plans'/session/'REGISTRY.json')
    case=next((c for c in registry['cases'].values() if c['attempt_id']==card['id']),None)
    if (case is None or case['card']!=card or ready['status']!='PV1_NATIVE_MODEL_READY'
            or ready['label']!=session or ready['identity']['global_step']!=spec['checkpoint_step']
            or ready['identity']['relay_identity']!=read(root/card['relay_request_relative'])['identity']
            or ready['native_pre_model_qualification']['status']!='PASS_SAME_JOB_NATIVE_QUALIFICATION_READBACK'
            or registry['native_pre_model_receipt_sha256']!=ready['native_pre_model_qualification']['receipt_sha256']
            or ready['job_id']!=registry['job_id']):
        raise PermissionError('PV1_CURRENT_NATIVE_MODEL_IDENTITY_AND_REAL_PREMODEL_PROOF')
    deadline=min(read(root.parent.parent/'PV1_ADOPTION_AND_RESOURCE_BINDING.json')['stop_new_physics_epoch'],
        ready['allocation_end_epoch']-2700,m['limits']['operation_deadline_epoch'])
    if time.time()+900>=deadline:raise TimeoutError('PV1_KEEP_NATIVE_45_MIN_CLOSEOUT_AND_CASE_BOUNDARY')
    lock=OwnerLock(root/'PV1_SOURCE_OWNER.lock');started=time.process_time();folder=root/'episodes'/card['id']
    original,original_relay=episode.make_teacher,relay_teacher.RelayTeacher
    native_model=None;old_clone=s1_preflight.clone;old_r3_clone=r3_teacher.clone;engines=[];result=None
    reservation=dict(schema='PI0294_LIVE_RESERVATION_V1',fixture=False,id=card['id'],card=card,
        card_sha256=digest(card),source_sha=m['code']['vla_commit'],root=str(root),output=str(folder),deadline_epoch=deadline,
        cpus=m['limits']['cpus'],reservation=dict(physics_starts=1,physics_steps=16000,rgb_pairs=641,
            cpu_s=3600,memory_bytes=8*1024**3,output_bytes=768*1024**2))
    def guard():
        ledger.guard()
        if time.time()>=deadline or time.process_time()-started>=3600:raise TimeoutError('PV1_REAL_EPISODE_CPU_AND_WALL_BOUND')
    def factory(c,initial,engine):engines.append(engine);return original(c,initial,engine)
    def relay(t,r,f,c):
        return E1ProtectedTeacher(original_relay(t,r,f,c),engines[0],root,folder,ledger,guard,c['family']+'-'+c['task'])
    def clone(engine):ledger.charge('engine_starts',1);return old_clone(engine)
    try:
        import mujoco
        native_model=mujoco.MjModel
        if len(m['limits']['cpus'])>3 or not set(m['limits']['cpus'])<=os.sched_getaffinity(0):
            raise PermissionError('PV1_BOUND_CORE_CPU_AFFINITY')
        ledger.charge('main_starts',1)
        with ledger.steps('live_and_recovery_integrations',16000,card['id']) as meter:
            episode.make_teacher,relay_teacher.RelayTeacher=factory,relay
            mujoco.MjModel=counted_model(native_model,Counter(root,ledger,inputs,card))
            s1_preflight.clone=r3_teacher.clone=clone
            try:result=episode.run_episode(card,folder,inputs['asset_root'],m['code']['vla_commit'],reservation)
            finally:
                if engines:meter['actual']=engines[0].steps
                elif result:meter['actual']=result['physics_steps']
                else:raise RuntimeError('PV1_UNKNOWN_NATIVE_LIVE_EFFECT_RETAIN_RESERVATION')
        s1=read(folder/'S1_RESULT.json') if (folder/'S1_RESULT.json').exists() else {}
        rejected=read(folder/'MODEL_REJECT.json') if (folder/'MODEL_REJECT.json').exists() else None
        outcome,stage,trust=classify(result,s1,rejected)
        if outcome=='ACTUAL_SAFETY_FAIL' or stage=='S1_INTEGRITY':
            ledger.stop_physics(stage,dict(attempt_id=card['id'],result=result))
        row=dict(status=outcome,failure_stage=stage,pipeline_trustworthy=trust,attempt_id=card['id'],
            condition_id=card['condition_id'],model=card['policy_candidate'],task=card['task'],split=card['split'],
            scene_group=card['scene_group'],noise_bank=card['noise_bank'],family=card['family'],
            duration_s=result['physics_steps']*.002,main_integrations=result['physics_steps'],RGB_pairs=result['rgb_pairs'],
            S1_rejections=s1.get('pre_execution_rejections',0),live_native_failures=s1.get('live_native_failures',0),
            maximum_live_shadow_difference=s1.get('maximum_live_shadow_difference'),result_sha256=sha(folder/'RESULT.json'),
            teacher_takeover=False,selected='NONE',real_robot_deployment='NONE')
        write(folder/'PV1_ADJUDICATION.json',row);return row
    finally:
        episode.make_teacher,relay_teacher.RelayTeacher=original,original_relay
        s1_preflight.clone=old_clone;r3_teacher.clone=old_r3_clone
        if native_model is not None:mujoco.MjModel=native_model
        ledger.cpu(time.process_time()-started);lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    group=parser.add_mutually_exclusive_group(required=True);group.add_argument('--card');group.add_argument('--qualify-session')
    args=parser.parse_args();value=qualify(args.root,args.qualify_session) if args.qualify_session else run(args.root,args.card)
    print(json.dumps(value),flush=True)
