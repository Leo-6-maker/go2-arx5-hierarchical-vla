"""Original teacher episodes with D0+S1 rejection and actual clone parity."""
import argparse,json,os,sys,time,uuid
from pathlib import Path
if not __package__:
    sys.path[:0]=[str(Path(__file__).resolve().parents[2]),str(Path(__file__).resolve().parents[2]/'src')]
    __package__='scripts.pi0294'
import numpy as np
from scripts.pi0292.durable import OwnerLock,read,write
from . import episode
from .continuation import validate_entry,CUTOFF
from .r3_s1 import ENGINEERING_DEADLINE
from .scene_cards import digest
from .s1_preflight import predict,bind_install,observed_state,fingerprint,clone,VERSION

CONTEXTS={'F0-T1':(0,4,7,11),'F0-T2':(0,21,28,29)}


class ProtectedTeacher:
    def __init__(self,teacher,engine,root,folder,ledger,guard,key):
        self.teacher,self.engine,self.root,self.folder=teacher,engine,root,folder
        self.ledger,self.guard,self.key=ledger,guard,key
        self.records=[];self.index=0;self.expected=None;self.maximum=0.;self.contexts=[];self.negatives=None
    def __getattr__(self,name):return getattr(self.teacher,name)
    def charge(self,count,*,invocation):
        if invocation:
            if self.ledger['shadow_advance_invocations']+1>2500 or self.ledger['shadow_integrations']+200>600000:
                raise PermissionError('S1_SHADOW_CAP')
            self.ledger['shadow_advance_invocations']+=1;self.ledger['shadow_integrations']+=200
            self.ledger['shadow_pending_reservation']=200;write(self.root/'RESOURCE_LEDGER.json',self.ledger)
    def plan(self,state):
        self.guard();plan=self.teacher.plan(state)
        self.poses,self.opening=plan['poses'][:8],plan['opening'][:8]
        self.nonce=str(uuid.uuid4())
        forecast,self.expected=predict(self.engine,self.poses,self.opening,nonce=self.nonce,guard=self.guard,charge=self.charge)
        self.ledger['shadow_integrations']-=200-forecast['shadow_integrations']
        self.ledger.pop('shadow_pending_reservation');write(self.root/'RESOURCE_LEDGER.json',self.ledger)
        forecast.update(sequence=self.engine.steps//25,stage=plan['stage']);self.records.append(forecast)
        write(self.folder/'S1_PREFLIGHTS.json',self.records)
        if forecast['decision']!='PASS':raise PermissionError('PRE_EXECUTION_SAFETY_REJECT')
        self.index=0;self.forecast=forecast
        if self.negatives is None:self.check_negatives()
        return plan
    def check_negatives(self):
        before=fingerprint(self.engine);checks=[]
        for kind in ('nonce','action','state','decision'):
            test=clone(self.engine);p=self.poses.copy();o=self.opening.copy();r=dict(self.forecast);nonce=self.nonce
            if kind=='nonce':nonce+='-stale'
            elif kind=='action':o[0]+=1e-6
            elif kind=='state':test.data.qpos[0]+=1e-6
            else:r['decision']='REJECT'
            try:
                bind_install(test,p,o,r,nonce)
            except PermissionError:checks.append(kind)
            else:raise ValueError('S1_NEGATIVE_WAS_ACCEPTED:'+kind)
            finally:test.close()
        if fingerprint(self.engine)!=before:raise ValueError('S1_NEGATIVES_CHANGED_LIVE_STATE')
        self.negatives=dict(status='PASS_ACTUAL_CLONED_STATE_NEGATIVES_ZERO_INTEGRATION',rejected=checks)
    def compare_previous(self):
        error=float(np.max(np.abs(observed_state(self.engine)-self.expected[self.index-1])))
        self.maximum=max(self.maximum,error)
        if error>1e-10:raise RuntimeError('S1_ACTUAL_LIVE_SHADOW_PARITY_MISMATCH')
    def guard_prefix(self):
        self.guard()
        if self.index==0:bind_install(self.engine,self.poses,self.opening,self.forecast,self.nonce)
        else:self.compare_previous()
        if self.index>=200:raise PermissionError('S1_PREFIX_REPLAY_REJECT')
        self.index+=1
        if hasattr(self.teacher,'guard_prefix'):self.teacher.guard_prefix()
    def acknowledge(self,engine,rows):
        if engine is not self.engine or self.index!=200:raise ValueError('S1_EXACT_PREFIX200_REQUIRED')
        self.compare_previous()
        sequence=self.forecast['sequence']
        if sequence in CONTEXTS.get(self.key,()):
            self.contexts.append(dict(task=self.key,sequence=sequence,stage=self.forecast['stage'],
                actual_ticks_compared=200,maximum_live_shadow_difference=self.maximum,pre_execution_rejections=0))
        if hasattr(self.teacher,'acknowledge'):self.teacher.acknowledge(engine,rows)
    def finish(self,status,error):
        if hasattr(self.teacher,'finish'):self.teacher.finish(status,error)
        write(self.folder/'S1_RESULT.json',dict(version=VERSION,status=status,error=error,
            shadow_invocations=len(self.records),shadow_integrations=sum(r['shadow_integrations'] for r in self.records),
            pre_execution_rejections=sum(r['decision']=='REJECT' for r in self.records),
            live_native_failures=int(bool(self.engine.safety)),maximum_live_shadow_difference=self.maximum,
            contexts=self.contexts,negatives=self.negatives,not_formal_training_data=True,
            privileged_simulation_state=True,actions_modified=False,selected='NONE',deployment='NONE'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--card',required=True);a=p.parse_args()
    root=Path(a.root);manifest=validate_entry(root,profile='4080-simulation')
    lock=OwnerLock(root/'S1_OWNER.lock');ledger=read(root/'RESOURCE_LEDGER.json');card=read(a.card)
    if ledger.get('reservations'):raise PermissionError('OUTSTANDING_MAIN_RESERVATION')
    if ledger['main_or_diagnostic_starts']+1>32 or ledger['main_or_diagnostic_integrations']+16000>512000 or ledger['RGB_pairs']+641>20512:
        raise PermissionError('R3_MAIN_BUDGET_CAP')
    folder=root/'episodes'/card['id'];folder.parent.mkdir(exist_ok=True)
    deadline=min(ENGINEERING_DEADLINE,CUTOFF)
    def guard():
        if time.time()>=deadline:raise TimeoutError('S1_ENGINEERING_TIMEBOX_CLOSED')
    guard();cpu_start=time.process_time()
    reservation=dict(schema='PI0294_LIVE_RESERVATION_V1',fixture=False,id=card['id'],card=card,
        card_sha256=digest(card),source_sha=manifest['code']['vla_commit'],root=str(root),output=str(folder),
        asset_root=read(root/'INPUT_BINDINGS.private.json')['asset_root'],deadline_epoch=deadline,
        cpus=[min(os.sched_getaffinity(0))],reservation=dict(physics_starts=1,physics_steps=16000,rgb_pairs=641,
            cpu_s=min(3600,86400-ledger['Core_CPU_s']),memory_bytes=8*1024**3,output_bytes=512*1024**2))
    ledger['reservations'][card['id']]=reservation;write(root/'RESOURCE_LEDGER.json',ledger)
    original=episode.make_teacher
    def factory(c,initial,engine):
        return ProtectedTeacher(original(c,initial,engine),engine,root,folder,ledger,guard,c['family']+'-'+c['task'])
    episode.make_teacher=factory;result=None
    try:
        result=episode.run_episode(card,folder,reservation['asset_root'],reservation['source_sha'],reservation)
    finally:
        episode.make_teacher=original
        ledger['main_or_diagnostic_starts']+=1
        ledger['main_or_diagnostic_integrations']+=result['physics_steps'] if result else 16000
        ledger['RGB_pairs']+=result['rgb_pairs'] if result else 641
        ledger['Core_CPU_s']+=time.process_time()-cpu_start
        ledger['reservations'].pop(card['id'])
        ledger['stages'].append(dict(phase=card['id'],status=result['status'] if result else 'ENGINEERING_FAILURE_PRESERVED'))
        write(root/'RESOURCE_LEDGER.json',ledger);lock.close()
    print(json.dumps(result,allow_nan=False),flush=True)


if __name__=='__main__':main()
