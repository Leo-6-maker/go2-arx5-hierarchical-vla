"""Actual source-only policy evaluation through frozen original episode and S1."""
import argparse
import json
import os
from pathlib import Path
import time

from scripts.pi0292.durable import OwnerLock,write
from scripts.pi0294.e1.control_read import read
from scripts.pi0294 import episode,relay_teacher
from scripts.pi0294.e0_core import classify
from scripts.pi0294.e1.core import E1ProtectedTeacher
from scripts.pi0294.scene_cards import digest
from scripts.pi0294.native_dataset import sha
from .science import validate_runtime,Ledger


def run(root,card_path):
    root=Path(root);m=validate_runtime(root,'4080-simulation');ledger=Ledger(root,'core');ledger.guard()
    if (root/'SOURCE_PHYSICS_STOP.json').exists():raise PermissionError('MP1_SOURCE_SAFETY_STOP_PRESERVED')
    card_path=Path(card_path);card_path=card_path if card_path.is_absolute() else root/card_path
    if not card_path.resolve().is_relative_to(root.resolve()):raise PermissionError('MP1_REGISTERED_CARD_ONLY')
    card=read(card_path);label=card['mp1_model_label'];session=card['mp1_session_label'];plan=read(root/'plans/SOURCE_EVALUATION_PLAN.json')
    from .plans import session_spec,remaining_panel
    spec=session_spec(plan,session)
    prior=[read(p) for p in (root/'episodes').glob('*/MP1_ADJUDICATION.json')]
    allowed=remaining_panel(prior,spec)
    expected=next((c for c in allowed if c['id']==card['id']),None)
    if (card.get('e1_intervention_collection') or card['split']!=spec['split'] or expected is None
            or {k:v for k,v in card.items() if k not in ['mp1_session_label','relay_request_relative','relay_request_sha256']}!=expected):
        raise PermissionError('MP1_PREDECLARED_AUTONOMOUS_PANEL_ONLY')
    ready=read(root/('plans/'+session+'/REMOTE_READY.json'));registry=read(root/('plans/'+session+'/REGISTRY.json'))
    case=next(c for c in registry['cases'].values() if c['attempt_id']==card['id'])
    if (case['card']!=card or ready['status']!='MP1_NATIVE_MODEL_READY' or ready['label']!=session
            or ready['identity']['global_step']!=spec['checkpoint_step']
            or ready['identity']['relay_identity']!=read(root/card['relay_request_relative'])['identity']):raise PermissionError('MP1_EXACT_PREBOUND_SOURCE_MODEL')
    deadline=min(read(root/'MP1_STAGE_BINDING.json')['new_experiment_cutoff_epoch'],ready['allocation_end_epoch']-2700)
    if time.time()+60>=deadline:raise TimeoutError('MP1_PRESERVE_ALLOCATION_CLOSEOUT_RESERVE')
    lock=OwnerLock(root/'MP1_SOURCE_OWNER.lock');started=time.process_time();folder=root/'episodes'/card['id'];engines=[];result=None
    original,original_relay=episode.make_teacher,relay_teacher.RelayTeacher
    reservation=dict(schema='PI0294_LIVE_RESERVATION_V1',fixture=False,id=card['id'],card=card,card_sha256=digest(card),source_sha=m['code']['vla_commit'],
        root=str(root),output=str(folder),deadline_epoch=deadline,cpus=sorted(os.sched_getaffinity(0))[:3],
        reservation=dict(physics_starts=1,physics_steps=16000,rgb_pairs=641,cpu_s=3600,memory_bytes=8*1024**3,output_bytes=512*1024**2))
    def guard():
        ledger.guard()
        if (root/'SOURCE_PHYSICS_STOP.json').exists():raise PermissionError('MP1_SOURCE_SAFETY_STOP_DURING_CASE')
        if time.time()>=deadline or time.process_time()-started>=3600:raise TimeoutError('MP1_SOURCE_EPISODE_BOUNDARY')
    def factory(c,initial,engine):engines.append(engine);return original(c,initial,engine)
    def relay(t,r,f,c):return E1ProtectedTeacher(original_relay(t,r,f,c),engines[0],root,folder,ledger,guard,c['family']+'-'+c['task'])
    try:
        ledger.charge('main_starts',1)
        with ledger.steps('live_and_recovery_integrations',16000,card['id']) as meter:
            episode.make_teacher,relay_teacher.RelayTeacher=factory,relay
            try:result=episode.run_episode(card,folder,read(root/'INPUT_BINDINGS.private.json')['asset_root'],m['code']['vla_commit'],reservation)
            finally:
                if engines:meter['actual']=engines[0].steps
                elif result:meter['actual']=result['physics_steps']
        s1=read(folder/'S1_RESULT.json') if (folder/'S1_RESULT.json').exists() else {}
        rejected=read(folder/'MODEL_REJECT.json') if (folder/'MODEL_REJECT.json').exists() else None
        outcome,stage,trust=classify(result,s1,rejected)
        if outcome=='ACTUAL_SAFETY_FAIL' or stage=='S1_INTEGRITY':ledger.stop_physics(stage,dict(attempt_id=card['id'],result=result))
        value=dict(status=outcome,failure_stage=stage,pipeline_trustworthy=trust,attempt_id=card['id'],model=label,task=card['task'],split=card['split'],scene_group=card['scene_group'],
            duration_s=result['physics_steps']*.002,main_integrations=result['physics_steps'],RGB_pairs=result['rgb_pairs'],S1_rejections=s1.get('pre_execution_rejections',0),
            live_native_failures=s1.get('live_native_failures',0),maximum_live_shadow_difference=s1.get('maximum_live_shadow_difference'),
            result_sha256=sha(folder/'RESULT.json'),teacher_takeover=False,selected='NONE',real_robot_deployment='NONE')
        write(folder/'MP1_ADJUDICATION.json',value);return value
    finally:episode.make_teacher,relay_teacher.RelayTeacher=original,original_relay;ledger.cpu(time.process_time()-started);lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--card',required=True);a=p.parse_args();print(json.dumps(run(a.root,a.card)),flush=True)
