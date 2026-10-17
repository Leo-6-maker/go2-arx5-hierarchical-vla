"""Single-owner fresh-observation routes for predeclared source development."""
import hashlib
import json
import time
import uuid

from scripts.pi0292.durable import write
from scripts.pi0292.eval_contract import gate_action
from scripts.pi0292.relay_protocol import unseal
from scripts.pi0294.e1.journal import Journal
from scripts.pi0294.relay_transport import run_episode
from .plans import remaining_panel,session_spec


def stop_engineering_branch(remote,session):
    for role in ['core','hopper']:
        remote(role,r'''
import json,sys
from pathlib import Path
r=Path(ROOT);m=json.loads((r/'RUN_MANIFEST.json').read_text());code=r/m['code']['capsule_relative'];sys.path[:0]=[str(code),str(code/'src')]
from scripts.pi0294.mp1.cold_boundary import stop_target
p=stop_target(r,int(SESSION.rsplit('-',1)[1]))
value=dict(run_uuid=RUN,session=SESSION,reason='POISONED_FILE_RPC_UNKNOWN_EFFECT_NO_NEXT_CASE_OR_MODEL',
 native_safety_stop=False,scientific_policy_failure=False)
if p.exists():assert json.loads(p.read_text())==value
else:p.write_text(json.dumps(value,indent=2)+'\n');p.chmod(0o400)
assert json.loads(p.read_text())==value
print(json.dumps(dict(status='RELAY_ENGINEERING_BRANCH_STOP_READBACK',role=ROLE)))
''',dict(ROLE=role,SESSION=session),label='MP1-poisoned-relay-stop-all-next-model-sessions')


def bind(remote,session,ready):
    facts={}
    for role in ['core','hopper']:
        facts[role]=remote(role,r'''
import hashlib,json,sys
from pathlib import Path
r=Path(ROOT);raw=(r/'RUN_MANIFEST.json').read_bytes();m=json.loads(raw);rows=[];active=[]
code=r/m['code']['capsule_relative'];sys.path[:0]=[str(code),str(code/'src')]
from scripts.pi0294.mp1.cold_boundary import blocked
if ROLE=='core':
 for p in (r/'worker_processes').glob('*.json'):
  w=json.loads(p.read_text());s=Path('/proc',str(w['pid']),'stat')
  if s.exists() and s.read_text().split()[2]!='Z' and s.read_text().split()[21]==str(w['proc_start']):active.append(p.stem)
 assert not active and not json.loads((r/'RESOURCE_LEDGER.json').read_text())['reservations']
 for folder in (r/'episodes').iterdir():
  if folder.is_dir():
   p=folder/'MP1_ADJUDICATION.json'
   assert p.is_file(),'MP1_UNKNOWN_PRIOR_PHYSICAL_OUTCOME_NO_BLIND_RESTART'
   rows.append(json.loads(p.read_text()))
 if 'prior_recovery_usage' in json.loads((r/'MP1_STAGE_BINDING.json').read_text()):
  rows=json.loads((r/'plans/PRIOR_RECOVERY_OUTCOMES.json').read_text())+rows
print(json.dumps(dict(source_sha=m['code']['vla_commit'],manifest_sha256=hashlib.sha256(raw).hexdigest(),
 plan=json.loads((r/'plans/SOURCE_EVALUATION_PLAN.json').read_text()),rows=rows,
 engineering_stop=blocked(r,int(SESSION.rsplit('-',1)[1])),
 source_stop=(r/'SOURCE_PHYSICS_STOP.json').exists() if ROLE=='core' else False)))
''',dict(ROLE=role,SESSION=session),label='MP1-fixed-panel-boundary-'+role)
    if facts['core']['plan']!=facts['hopper']['plan']:raise PermissionError('MP1_IDENTICAL_FIXED_DEVELOPMENT_PLAN_ON_BOTH_HOSTS')
    if facts['core']['source_stop'] or any(v['engineering_stop'] for v in facts.values()):return None
    spec=session_spec(facts['core']['plan'],session)
    if ready['status']!='MP1_NATIVE_MODEL_READY' or ready['identity']['global_step']!=spec['checkpoint_step'] or ready['label']!=session:
        raise PermissionError('MP1_FIXED_NATIVE_CHECKPOINT_ONLY')
    selected=remaining_panel(facts['core']['rows'],spec)
    if not selected:return None
    cases={}
    for index,base in enumerate(selected):
        card=dict(base,mp1_session_label=session);episode=str(uuid.uuid4());relative='plans/'+session+'/requests/'+card['id']+'.json'
        request=dict(run_uuid=ready['run_uuid'],attempt_id=card['id'],source_sha=facts['core']['source_sha'],run_manifest_sha256=facts['core']['manifest_sha256'],
            initial_state_sha256=card['initial_state_sha256'],profile='B1',category='TRAIN_DEVELOPMENT' if spec['split']=='TRAIN' else 'DEV',
            identity=ready['identity']['relay_identity'],episode_uuid=episode,deadline_epoch=ready['allocation_end_epoch']-2700,
            live_readiness_sha256='c2ca87bda6942f17b037d47bdacf33f069f2bd1ecb2f8d57126d87ba8a69ed5c')
        raw=json.dumps(request,indent=2,allow_nan=False)+'\n';card.update(relay_request_relative=relative,relay_request_sha256=hashlib.sha256(raw.encode()).hexdigest())
        cases[episode]=dict(run_uuid=ready['run_uuid'],episode_uuid=episode,attempt_id=card['id'],identity=request['identity'],deadline_epoch=request['deadline_epoch'],
            layout_id=card['scene_group'],core_request_relative=relative,core_request_sha256=card['relay_request_sha256'],core_request=request,card=card,panel_index=index,
            hopper_session_relative='evidence/'+session+'/resident')
    registry=dict(run_uuid=ready['run_uuid'],deadline_epoch=ready['allocation_end_epoch']-2700,cases=cases,checkpoint_identity=ready['identity'],
        **{role+'_'+k:info[k] for role,info in facts.items() for k in ['source_sha','manifest_sha256']})
    for role in ['core','hopper']:
        remote(role,r'''
import json
from pathlib import Path
r=Path(ROOT);d=r/'plans'/SESSION;d.mkdir(parents=True,exist_ok=True)
def put(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);raw=json.dumps(v,indent=2,allow_nan=False)+'\n'
 if p.exists():assert p.read_text()==raw
 else:p.write_text(raw);p.chmod(0o400)
if ROLE=='core':
 assert not (r/'SOURCE_PHYSICS_STOP.json').exists()
 for c in REGISTRY['cases'].values():
  put(r/c['core_request_relative'],c['core_request']);put(d/'cards'/(c['attempt_id']+'.json'),c['card'])
 put(d/'REMOTE_READY.json',READY)
put(d/'REGISTRY.json',REGISTRY);print(json.dumps(dict(status='MP1_ACTUAL_REMAINING_FIXED_PANEL_BOUND',count=len(REGISTRY['cases']))))
''',dict(ROLE=role,SESSION=session,REGISTRY=registry,READY=ready),label='MP1-bind-fixed-session-'+role)
    return registry


def routes(remote,base,session,ready,start_worker,transport):
    registry=bind(remote,session,ready)
    if registry is None:return dict(status='NO_NEW_CASES_TERMINAL_OR_SOURCE_OR_ENGINEERING_STOP',session=session,rows=[])
    journal=Journal(base/'mutation-journal',ready['run_uuid']);rows=[]
    def rpc(role,args):
        v=transport.call(role,args)
        if not v['ok']:raise RuntimeError('MP1_UNKNOWN_FILE_RPC_OUTCOME_NO_RESEND:'+v['error'])
        value=v['value']
        if isinstance(value,dict) and value.get('known_intervention'):raise PermissionError('MP1_TEACHER_OFF_AUTONOMOUS_EVALUATION')
        if isinstance(value,dict) and value.get('known_terminal'):raise RuntimeError('MP1_KNOWN_CORE_TERMINAL:'+value['known_terminal']['failure_stage'])
        return value
    class Core:
        def observation(self,c,s):return rpc('core',dict(operation='observation',episode=c['episode_uuid'],sequence=s))
        def ack(self,c,s):return rpc('core',dict(operation='ack',episode=c['episode_uuid'],sequence=s))
        def install(self,c,s,b):
            args=dict(operation='install',episode=c['episode_uuid'],sequence=s,bundle=b)
            return journal.once('install-'+unseal(b['command'])['query_uuid'],args,lambda:rpc('core',args))
    class Hopper:
        last=None
        def query(self,c,q,teacher):
            if teacher is not None:raise PermissionError('MP1_NO_TEACHER_IN_STUDENT_INPUT')
            self.last=q;args=dict(operation='query',episode=c['episode_uuid'],query=q)
            return journal.once('query-'+unseal(q)['query_uuid'],args,lambda:rpc('hopper',args))
        def response(self,c,q):return rpc('hopper',dict(operation='response',episode=c['episode_uuid'],query_uuid=q))
    core,hopper=Core(),Hopper()
    for case in sorted(registry['cases'].values(),key=lambda c:c['panel_index']):
        if any(getattr(s,'failed',False) for s in getattr(transport,'sessions',{}).values()):
            stop_engineering_branch(remote,session)
            return dict(status='ENGINEERING_BRANCH_STOP_NO_NEXT_CASE_OR_MODEL',session=session,rows=rows,planned=len(registry['cases']),abort_remaining_sessions=True)
        if time.time()+900>=registry['deadline_epoch']:break
        args=['--card','plans/'+session+'/cards/'+case['attempt_id']+'.json']
        journal.once('start-'+case['attempt_id'],dict(attempt_id=case['attempt_id']),lambda:start_worker('scripts.pi0294.mp1.core',case['attempt_id'],args))
        failure=None;rejection=None;hopper.last=None
        try:run_episode(base/'routes'/case['episode_uuid'],case,core,hopper)
        except BaseException as exc:
            failure=dict(type=type(exc).__name__,message=str(exc))
            if hopper.last:
                q=unseal(hopper.last)
                try:
                    action=hopper.response(case,q['query_uuid'])
                    if action:
                        gate=gate_action(unseal(action)['payload']['trajectory_eef10'],q['payload']['state_eef10'],q['tick']*.002,q['reference_uuid'])
                        if gate['decision']=='REJECT':rejection=dict(original_gate=gate,action_sha256=action['sha256'],query_uuid=q['query_uuid'])
                except Exception:pass
            remote('core',r'''
import json
from pathlib import Path
folder=Path(ROOT)/'episodes'/ATTEMPT
def put(p,v):
 if not p.exists():p.write_text(json.dumps(v,indent=2)+'\n');p.chmod(0o600)
if REJECTION is not None:put(folder/'MODEL_REJECT.json',REJECTION)
if (folder/'relay').is_dir():put(folder/'relay/STOP.json',dict(run_uuid=RUN,episode_uuid=EPISODE,reason='MP1_OWN_CASE_TERMINAL_UNKNOWN_EFFECT_NOT_REPLAYED'))
print(json.dumps(dict(status='MP1_OWN_CASE_STOP_REQUESTED')))
''',dict(ATTEMPT=case['attempt_id'],EPISODE=case['episode_uuid'],REJECTION=rejection),label='MP1-own-case-stop')
        end=time.time()+90;terminal={}
        while time.time()<end:
            terminal=remote('core',r'''
import json
from pathlib import Path
r=Path(ROOT);p=r/'episodes'/ATTEMPT/'MP1_ADJUDICATION.json';w=json.loads((r/'worker_processes'/(ATTEMPT+'.json')).read_text());s=Path('/proc',str(w['pid']),'stat')
alive=s.exists() and s.read_text().split()[2]!='Z' and s.read_text().split()[21]==str(w['proc_start'])
print(json.dumps(dict(terminal=not alive,row=json.loads(p.read_text()) if p.exists() else None,source_stop=(r/'SOURCE_PHYSICS_STOP.json').exists())))
''',dict(ATTEMPT=case['attempt_id']),label='MP1-actual-case-terminal')
            if terminal['terminal']:break
            time.sleep(1)
        row=terminal.get('row') or dict(attempt_id=case['attempt_id'],status='ENGINEERING_OUTCOME_UNKNOWN_NOT_RETRIED',pipeline_trustworthy=False)
        row=dict(row,route_failure=failure);rows.append(row);write(base/(session+'-RESULTS.private.json'),dict(session=session,rows=rows,planned=len(registry['cases'])))
        print(json.dumps(dict(session=session,covered=len(rows),status=row['status'])),flush=True)
        if any(getattr(s,'failed',False) for s in getattr(transport,'sessions',{}).values()):
            stop_engineering_branch(remote,session)
            return dict(status='ENGINEERING_BRANCH_STOP_NO_NEXT_CASE_OR_MODEL',session=session,rows=rows,planned=len(registry['cases']),abort_remaining_sessions=True)
        if terminal.get('source_stop') or not terminal.get('terminal'):break
    return dict(session=session,rows=rows,planned=len(registry['cases']))
