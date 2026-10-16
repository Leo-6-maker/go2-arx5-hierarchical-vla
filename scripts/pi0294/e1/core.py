"""Autonomous TRAIN-only evaluation with original episode, relay and S1 logic."""
import argparse
import json
import os
from pathlib import Path
import time
import uuid

from scripts.pi0292.durable import OwnerLock,read,write
from scripts.pi0294 import episode,relay_teacher
from scripts.pi0294.e0_core import classify
from scripts.pi0294.r3_teacher import ProtectedTeacher
from scripts.pi0294.s1_preflight import predict
from scripts.pi0294.scene_cards import digest
from scripts.pi0294.native_dataset import sha
from .scope import validate,Ledger


class E1ProtectedTeacher(ProtectedTeacher):
    def plan(self,state):
        self.guard();plan=self.teacher.plan(state);self.poses,self.opening=plan['poses'][:8],plan['opening'][:8]
        self.nonce=str(uuid.uuid4())
        with self.ledger.steps('shadow_integrations',200,self.folder.name+'-S1-'+str(self.engine.steps)) as meter:
            def charge(count,*,invocation):
                if not invocation:meter['actual']+=count
            forecast,self.expected=predict(self.engine,self.poses,self.opening,nonce=self.nonce,guard=self.guard,charge=charge)
        forecast.update(sequence=self.engine.steps//25,stage=plan['stage']);self.records.append(forecast)
        write(self.folder/'S1_PREFLIGHTS.json',self.records)
        if forecast['decision']!='PASS':
            if hasattr(self.teacher,'intervene') and self.teacher.intervene(state,forecast):return self.plan(state)
            raise PermissionError('PRE_EXECUTION_SAFETY_REJECT')
        self.index=0;self.forecast=forecast;self.start_tick=self.engine.steps
        if self.negatives is None:self.check_negatives()
        return plan


def run(root,card_path):
    root=Path(root);m=validate(root,'4080-simulation',physical=True);ledger=Ledger(root,'core');ledger.guard()
    card_path=Path(card_path)
    if not card_path.is_absolute():card_path=root/card_path
    if not card_path.resolve().is_relative_to(root.resolve()):raise PermissionError('E1_REGISTERED_CARD_PATH')
    card=read(card_path);ready=read(root/('plans/'+card['e1_model_label']+'/REMOTE_READY.json'))
    if card['split']!='TRAIN' or ready['status']!='E1_NATIVE_MODEL_READY':raise PermissionError('E1_TRAIN_ACTUAL_MODEL_ONLY')
    if ready['label']!=card['e1_model_label'] or ready['identity']['relay_identity']!=read(root/card['relay_request_relative'])['identity']:
        raise PermissionError('E1_EXACT_FIXED_CHECKPOINT_REQUIRED')
    registry=read(root/('plans/'+card['e1_model_label']+'/REGISTRY.json'))
    case=next(c for c in registry['cases'].values() if c['attempt_id']==card['id'])
    if case['card']!=card:raise PermissionError('E1_EXACT_PREBOUND_TRAIN_CARD')
    deadline=min(read(root/'E1_ADOPTION_AND_BINDING.json')['hard_closeout_epoch'],ready['allocation_end_epoch']-2700)
    if time.time()+60>=deadline:raise TimeoutError('E1_KEEP_45_MIN_ALLOCATION_CLOSEOUT_RESERVE')
    lock=OwnerLock(root/'E1_SOURCE_OWNER.lock');start=time.process_time();folder=root/'episodes'/card['id']
    original,original_relay=episode.make_teacher,relay_teacher.RelayTeacher;engines=[];result=None
    reservation=dict(schema='PI0294_LIVE_RESERVATION_V1',fixture=False,id=card['id'],card=card,card_sha256=digest(card),
        source_sha=m['code']['vla_commit'],root=str(root),output=str(folder),deadline_epoch=deadline,
        cpus=sorted(os.sched_getaffinity(0))[:3],reservation=dict(physics_starts=1,physics_steps=16000,rgb_pairs=641,
            cpu_s=3600,memory_bytes=8*1024**3,output_bytes=512*1024**2))
    def guard():
        ledger.guard()
        if time.time()>=deadline or time.process_time()-start>=3600:raise TimeoutError('E1_EPISODE_BOUNDARY')
    def factory(c,initial,engine):engines.append(engine);return original(c,initial,engine)
    def relay(t,r,f,c):
        teacher=original_relay(t,r,f,c)
        if c.get('e1_intervention_collection'):
            from .intervention import InterventionTeacher
            teacher=InterventionTeacher(teacher,engines[0],root,folder,c)
        return E1ProtectedTeacher(teacher,engines[0],root,folder,ledger,guard,c['family']+'-'+c['task'])
    try:
        ledger.charge('main_starts',1)
        with ledger.steps('live_and_recovery_integrations',16000,card['id']) as meter:
            episode.make_teacher,relay_teacher.RelayTeacher=factory,relay
            try:result=episode.run_episode(card,folder,read(root/'INPUT_BINDINGS.private.json')['asset_root'],m['code']['vla_commit'],reservation)
            finally:
                if engines:meter['actual']=engines[0].steps
                elif result:meter['actual']=result['physics_steps']
        s1=read(folder/'S1_RESULT.json') if (folder/'S1_RESULT.json').exists() else {}
        rejected=read(folder/'E1_MODEL_REJECT.json') if (folder/'E1_MODEL_REJECT.json').exists() else None
        outcome,stage,trust=classify(result,s1,rejected)
        if 'INTERVENTION_' in (result.get('error') or '') and 'MISMATCH' in result['error']:
            ledger.stop_physics('INTERVENTION_CRITICAL_STATE_MISMATCH',result['error']);outcome='ACTUAL_SAFETY_FAIL';stage='S1_INTEGRITY';trust=False
        if outcome=='ACTUAL_SAFETY_FAIL' or stage=='S1_INTEGRITY':ledger.stop_physics(stage,dict(attempt_id=card['id'],result=result))
        if card.get('e1_intervention_collection') and outcome!='ACTUAL_SAFETY_FAIL' and stage!='S1_INTEGRITY':
            intervention=read(folder/'INTERVENTION_RESULT.json') if (folder/'INTERVENTION_RESULT.json').exists() else {}
            complete=result['status']=='VALID_COMPLETE'
            valid_reject=bool(intervention.get('invalid_expert'))
            outcome='COLLECTION_COMPLETE' if complete else ('VALID_EXPERT_REJECT' if valid_reject else 'ENGINEERING_INCOMPLETE')
            stage='INTERVENTION_COLLECTION_NOT_AUTONOMOUS_EVALUATION';trust=complete or valid_reject
        row=dict(status=outcome,failure_stage=stage,pipeline_trustworthy=trust,attempt_id=card['id'],model=card['e1_model_label'],
            task=card['task'],split='TRAIN',scene_group=card['scene_group'],duration_s=result['physics_steps']*.002,
            main_integrations=result['physics_steps'],RGB_pairs=result['rgb_pairs'],S1_rejections=s1.get('pre_execution_rejections',0),
            live_native_failures=s1.get('live_native_failures',0),maximum_live_shadow_difference=s1.get('maximum_live_shadow_difference'),
            result_sha256=sha(folder/'RESULT.json'),teacher_takeover=False,selected='NONE',deployment='NONE')
        if card.get('e1_intervention_collection'):
            row.update(teacher_takeover=bool(intervention.get('switches')),intervention=intervention,
                assisted_task_success=bool(result.get('score',{}).get('persistent_success')),autonomous_evaluation=False)
        write(folder/'E1_ADJUDICATION.json',row);return row
    finally:
        episode.make_teacher,relay_teacher.RelayTeacher=original,original_relay;ledger.cpu(time.process_time()-start);lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--card',required=True);a=p.parse_args();print(json.dumps(run(a.root,a.card)),flush=True)
