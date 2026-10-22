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
from .evaluation_plan import session_spec,final_session_spec,remaining_panel
from .runtime import validate_runtime
from .scope import SOURCE_COMPONENT_ROLE


class Counter:
    def __init__(self,root,ledger,inputs,card):
        self.root,self.ledger,self.inputs=Path(root),ledger,inputs
        self.plan=dict(cards=[card])
    def start(self):self.ledger.charge('engine_starts',1)


def qualify(root,label):
    root=Path(root);m=validate_runtime(root,'4080-simulation');ledger=Ledger(root);ledger.guard()
    component=m.get('scope')==SOURCE_COMPONENT_ROLE
    if component:
        if label!='DEC1_SOURCE_COMPONENT_COUNTER':raise PermissionError('DEC1_EXACT_SOURCE_COUNTER_OPERATION')
        plan=read(root/'plans/DEC1_SOURCE_CPU_COMPONENT_PLAN.json')
        spec=dict(cards=[plan['conditions'][0]['card']])
    else:spec=(final_session_spec if 'LOCKBOX' in label else session_spec)(root,label)
    inputs=read(root/'INPUT_BINDINGS.private.json')
    lock=OwnerLock(root/'PV1_SOURCE_OWNER.lock');began=time.process_time()
    try:
        if (root/'evidence/COUNTER_NATIVE_QUALIFICATION.json').exists():
            raise FileExistsError('PV1_PRESERVE_FIRST_ZERO_INTEGRATION_COUNTER_QUALIFICATION')
        if component:
            intent=root/'evidence/SOURCE_COMPONENT_COUNTER_INTENT.json'
            if intent.exists() or (root/'evidence/SOURCE_COMPONENT_COUNTER_BINDING.json').exists():
                raise FileExistsError('DEC1_SOURCE_COUNTER_ALREADY_ATTEMPTED_NO_RETRY')
            write(intent,dict(run_uuid=root.name,source_sha=m['code']['vla_commit'],
                manifest_sha256=sha(root/'RUN_MANIFEST.json'),operation=label))
        qualify_counter(Counter(root,ledger,inputs,spec['cards'][0]))
        if component:
            receipt=root/'evidence/COUNTER_NATIVE_QUALIFICATION.json'
            write(root/'evidence/SOURCE_COMPONENT_COUNTER_BINDING.json',dict(
                schema='DEC1_SOURCE_COMPONENT_NATIVE_COUNTER_BINDING_V1',run_uuid=root.name,
                source_sha=m['code']['vla_commit'],manifest_sha256=sha(root/'RUN_MANIFEST.json'),
                qualification_sha256=sha(receipt),new_integrations=0,new_Teacher_episodes=0))
        return read(root/'evidence/COUNTER_NATIVE_QUALIFICATION.json')
    finally:ledger.cpu(time.process_time()-began);lock.close()


def run(root,card_path):
    root=Path(root);m=validate_runtime(root,'4080-simulation')
    if m.get('scope')==SOURCE_COMPONENT_ROLE:return _source_component(root,card_path,m)
    ledger=Ledger(root);ledger.guard()
    inputs=read(root/'INPUT_BINDINGS.private.json');card_path=Path(card_path)
    if not card_path.is_absolute():card_path=root/card_path
    if card_path.is_symlink() or not card_path.resolve().is_relative_to(root.resolve()):
        raise PermissionError('PV1_REGISTERED_AUTONOMOUS_CARD_ONLY')
    card=read(card_path);session=card['pv1_session_label']
    spec=(final_session_spec if card.get('split')=='LOCKBOX' else session_spec)(root,session)
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


def _source_component(root,card_path,m):
    """Explicit CPU component branch; the existing episode still owns all steps."""
    from scripts.pi0294.dec1.source_cpu_intake import PLAN_SHA256,PLAN_OWNER_RELEASE,ORIGINAL_13
    from scripts.pi0294.dec1.source_component_factory import SourceComponentFactory,source_component_adapter
    ledger=Ledger(root);ledger.guard();inputs=read(root/'INPUT_BINDINGS.private.json')
    card_path=Path(card_path)
    if not card_path.is_absolute():card_path=root/card_path
    if card_path.is_symlink() or not card_path.resolve().is_relative_to(root.resolve()):
        raise PermissionError('DEC1_EXACT_REGISTERED_COMPONENT_CARD')
    card=read(card_path);plan_path=root/'plans/DEC1_SOURCE_CPU_COMPONENT_PLAN.json'
    plan=read(plan_path);matches=[c for c in plan['conditions'] if c['card']==card]
    if len(matches)!=1 or 'relay_request_relative' in card:
        raise PermissionError('DEC1_PRESEALED_SOURCE_TEACHER_COMPONENT_ONLY')
    condition=matches[0];folder=root/'episodes'/card['id']
    if folder.exists() or folder.is_symlink():raise FileExistsError('DEC1_PRESERVE_FIRST_COMPONENT_ATTEMPT_NO_REPLAY')
    qpath=root/'evidence/COUNTER_NATIVE_QUALIFICATION.json'
    q=read(qpath);bound=read(root/'evidence/SOURCE_COMPONENT_COUNTER_BINDING.json')
    if (q.get('status')!='PASS_ORIGINAL_VS_COUNTED_NATIVE_MODEL_STATE_XML_AND_TWO_RGB'
            or type(q.get('integrations')) is not int or q['integrations']!=0
            or type(q.get('model_constructor_attempts')) is not int or q['model_constructor_attempts']!=2
            or bound.get('schema')!='DEC1_SOURCE_COMPONENT_NATIVE_COUNTER_BINDING_V1'
            or bound.get('run_uuid')!=root.name or bound.get('source_sha')!=m['code']['vla_commit']
            or bound.get('manifest_sha256')!=sha(root/'RUN_MANIFEST.json')
            or bound.get('qualification_sha256')!=sha(qpath)):
        raise PermissionError('DEC1_ACTUAL_CURRENT_SOURCE_ZERO_TIME_COUNTER_QUALIFICATION')
    deadline=m['limits']['operation_deadline_epoch']
    if time.time()+900>=deadline:raise TimeoutError('DEC1_COMPONENT_KEEP_ORIGINAL_CLOSEOUT_MARGIN')
    lock=OwnerLock(root/'PV1_SOURCE_OWNER.lock');started=time.process_time()
    original_factory=episode.make_teacher;old_clone=s1_preflight.clone;native_model=None;factory=None
    result=None
    def guard():
        ledger.guard()
        if time.time()>=deadline or time.process_time()-started>=3600:
            raise TimeoutError('DEC1_ORIGINAL_COMPONENT_CPU_AND_WALL_BOUND')
    def clone(engine):
        ledger.charge('engine_starts',1)
        return old_clone(engine)
    reservation=dict(schema='PI0294_LIVE_RESERVATION_V1',fixture=False,id=card['id'],card=card,
        card_sha256=digest(card),source_sha=m['code']['vla_commit'],root=str(root),output=str(folder),
        asset_root=inputs['asset_root'],deadline_epoch=deadline,cpus=m['limits']['cpus'],
        reservation=dict(physics_starts=1,physics_steps=16000,rgb_pairs=641,cpu_s=3600,
            memory_bytes=8*1024**3,output_bytes=768*1024**2))
    try:
        # These are the original registered native primitives. They are imported
        # only after typed admission and this child's actual zero-time proof.
        import mujoco
        from scripts.pi0291_arx5_data.engine import Engine as BaseEngine
        native_model=mujoco.MjModel
        ledger.charge('main_starts',1)
        with ledger.steps('live_and_recovery_integrations',16000,card['id']) as live:
            factory=SourceComponentFactory(folder=folder,ledger=ledger,outer_live_meter=live,
                original_make_teacher=original_factory,
                adapter=source_component_adapter(condition['interface'],card['id']),guard=guard,
                plan_bytes=plan_path.read_bytes(),expected_plan_sha256=PLAN_SHA256,
                population_bytes=(root/'plans/PV1_DATA_PLAN_AND_SPLIT_SEAL.json').read_bytes(),
                plan_owner_release=PLAN_OWNER_RELEASE,runtime_release=m['code']['vla_commit'],
                condition_id=condition['condition_id'],episode_id=card['id'],executor_id='DEC1_E_SOURCE_SINGLE_POSE',
                source_step_sha256=ORIGINAL_13['scripts/pi0291_arx5_data/engine.py'],
                expected_step=BaseEngine.step,expected_snapshot=BaseEngine.snapshot,
                fingerprint=s1_preflight.fingerprint,event_maximum_bytes=128*1024**2)
            errors=[]
            try:
                episode.make_teacher=factory
                mujoco.MjModel=counted_model(native_model,Counter(root,ledger,inputs,card))
                s1_preflight.clone=clone
                result=episode.run_episode(card,folder,inputs['asset_root'],m['code']['vla_commit'],reservation)
            except BaseException as error:errors.append(error)
            finally:
                episode.make_teacher=original_factory;s1_preflight.clone=old_clone;mujoco.MjModel=native_model
                status='FAILED_PRESERVED' if result is None else result['status']
                detail=str(errors[0]) if errors else None if result is None else result.get('error')
                # All three closeout operations are attempted exactly once.
                # Reconcile writes the actual native lower bound even when an
                # earlier persistence/close callback was interrupted.
                for callback in (lambda:factory.persist_if_unfinished(status,detail),
                                 factory.close_evidence,factory.reconcile_outer):
                    try:callback()
                    except BaseException as error:errors.append(error)
                if errors:
                    primary=next((e for e in errors if not isinstance(e,Exception)),errors[0])
                    secondary=next((e for e in reversed(errors) if e is not primary),None)
                    if secondary is not None and primary.__cause__ is None:raise primary from secondary
                    raise primary
            # Capture AFTER real close+reconcile, while this original live
            # context still retains its bound if any readback is unknown.
            terminal_anchor=factory.terminal_anchor(manifest_sha256=sha(root/'RUN_MANIFEST.json'))
        s1=read(folder/'S1_RESULT.json')
        unknown=(factory.unknown or s1.get('engineering_unknown') is not False
            or s1.get('events_readback_valid') is not True or s1.get('terminal_stop_error') is not None)
        if unknown:raise PermissionError('DEC1_COMPONENT_EVIDENCE_UNKNOWN_NOT_QUALIFIED')
        rejected=read(folder/'MODEL_REJECT.json') if (folder/'MODEL_REJECT.json').exists() else None
        outcome,stage,trust=classify(result,s1,rejected)
        if outcome=='ACTUAL_SAFETY_FAIL' or stage=='S1_INTEGRITY':
            ledger.stop_physics(stage,dict(attempt_id=card['id'],result=result))
        complete=(result['status']=='VALID_COMPLETE' and trust and result['physics_steps']==16000
            and result['rgb_pairs']==641 and s1['shadow_integrations']==16000
            and len(s1['actual_completed_prefixes'])==80 and len(s1['actual_25tick_ACKs'])==640)
        component_status=('COMPONENT_NATIVE_SAFETY_FAIL' if outcome=='ACTUAL_SAFETY_FAIL'
            else 'COMPONENT_COMPLETE_NOT_STUDENT_EVALUATION' if complete
            else 'KNOWN_COMPONENT_SAFETY_OR_REFERENCE_REJECT' if trust and stage in (
                'S1_PRE_EXECUTION_REJECT','ORIGINAL_REFERENCE_GATE_REJECT')
            else 'COMPONENT_ENGINEERING_INCOMPLETE')
        row=dict(schema='DEC1_SOURCE_COMPONENT_ATTEMPT_V1',attempt_id=card['id'],condition_id=condition['condition_id'],
            status=component_status,
            diagnostic_original_classification=outcome,failure_stage=stage,pipeline_trustworthy=trust,
            policy='ORIGINAL_FEEDBACK_TEACHER_NO_GR00T',task=card['task'],interface=condition['interface'],
            main_integrations=result['physics_steps'],shadow_integrations=s1['shadow_integrations'],
            completed_prefixes=len(s1['actual_completed_prefixes']),RGB_pairs=result['rgb_pairs'],
            original_Teacher_task_success_diagnostic=result.get('score',{}).get('persistent_success'),
            component_plan_sha256=PLAN_SHA256,result_sha256=sha(folder/'RESULT.json'),S1_sha256=sha(folder/'S1_RESULT.json'),
            native_STOP_return_and_owner_event_confirmed=True,independent_terminal_audit='PENDING_OWNER_READBACK',
            terminal_anchor_relative=terminal_anchor['relative'],terminal_anchor_sha256=terminal_anchor['sha256'],
            student_neural_ACK_qualification=False,new_training_data=False,GR00T_queries=0,
            selected='NONE',deployment='NONE')
        write(folder/'DEC1_SOURCE_COMPONENT_ATTEMPT.json',row)
        return row
    finally:
        episode.make_teacher=original_factory;s1_preflight.clone=old_clone
        if native_model is not None:mujoco.MjModel=native_model
        try:ledger.cpu(time.process_time()-started)
        finally:lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    group=parser.add_mutually_exclusive_group(required=True);group.add_argument('--card');group.add_argument('--qualify-session')
    args=parser.parse_args();value=qualify(args.root,args.qualify_session) if args.qualify_session else run(args.root,args.card)
    print(json.dumps(value),flush=True)
