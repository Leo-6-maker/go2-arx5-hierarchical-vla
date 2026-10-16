"""Finite single local-PC owner. SSH discovery/credentials stay outside source Git."""
import hashlib
import json
from pathlib import Path
import time
import uuid

from scripts.pi0292.durable import OwnerLock,read,write
from scripts.pi0292.relay_protocol import unseal
from scripts.pi0292.eval_contract import gate_action
from scripts.pi0294.relay_transport import run_episode
from .journal import Journal


RPC=r'''
import json,os,resource,subprocess,time
from pathlib import Path
root=Path(ROOT);m=json.loads((root/'RUN_MANIFEST.json').read_text());code=root/m['code']['capsule_relative']
os.sched_setaffinity(0,{sorted(os.sched_getaffinity(0))[3 if ROLE=='core' else 0]})
if ROLE=='hopper':
 p=root/'OWNER_HEARTBEAT.json';t=p.with_name(p.name+'.pending.'+str(os.getpid()));t.write_text(json.dumps(dict(run_uuid=RUN,observed_epoch=time.time()))+'\n');t.chmod(0o600);os.replace(t,p)
b=json.loads((root/('INPUT_BINDINGS.private.json' if ROLE=='core' else 'NEURAL_BINDINGS.private.json')).read_text())
python=b['runtime_python'] if ROLE=='core' else b['python']
program='import sys,resource,runpy;sys.path[:0]='+repr([str(code),str(code/'src')])+";resource.setrlimit(resource.RLIMIT_CPU,(8,8));runpy.run_module('scripts.pi0294.e1.rpc',run_name='__main__')"
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
before=time.process_time();child=resource.getrusage(resource.RUSAGE_CHILDREN)
r=subprocess.run([python,'-c',program,'--root',str(root),'--role',ROLE,'--label',LABEL],input=json.dumps(ARGS,allow_nan=False),capture_output=True,text=True,env=env,cwd=code,timeout=20)
after=resource.getrusage(resource.RUSAGE_CHILDREN);cpu=time.process_time()-before+after.ru_utime+after.ru_stime-child.ru_utime-child.ru_stime
import fcntl
ledger=root/('RESOURCE_LEDGER.json' if ROLE=='core' else 'HOPPER_RESOURCE_LEDGER.json')
with ledger.with_suffix('.owner.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);l=json.loads(ledger.read_text());key='Core_CPU_s' if ROLE=='core' else 'CPU_s';l[key]+=cpu;l['RPC_calls']=l.get('RPC_calls',0)+1
 temp=ledger.with_name(ledger.name+'.pending.'+str(os.getpid()));temp.write_text(json.dumps(l,indent=2)+'\n');temp.chmod(0o600);os.replace(temp,ledger)
if r.returncode:print(json.dumps(dict(ok=False,exit=r.returncode,error=r.stderr[-1500:])))
else:print(json.dumps(dict(ok=True,value=json.loads(r.stdout))))
'''


def encode(value):return json.dumps(value,indent=2,allow_nan=False)+'\n'


def recoverable_256_conditions(rows):
    expected={(f'PI0294/TRAIN/{family}/000',task) for family in ['F0','F1'] for task in ['T1','T2']}
    if len(rows)!=4 or {(r['scene_group'],r['task']) for r in rows}!=expected:raise PermissionError('E1_EXACT_PRIOR_256_PANEL_REQUIRED')
    selected=[]
    for row in rows:
        if row['live_native_failures']:raise PermissionError('E1_PRIOR_ACTUAL_SAFETY_STOP')
        if row['status']=='VALID_TASK_FAIL' and row['pipeline_trustworthy'] and row['failure_stage']=='S1_PRE_EXECUTION_REJECT':continue
        if row['status']=='ENGINEERING_INCOMPLETE' and not row['pipeline_trustworthy']:selected.append((row['scene_group'],row['task']))
        else:raise PermissionError('E1_NO_REPEAT_VALID_POLICY_OUTCOME')
    if selected!=[('PI0294/TRAIN/F1/000','T2')]:raise PermissionError('E1_ONLY_OBSERVED_ENGINEERING_INCOMPLETE_RECOVERY')
    return set(selected)


def bind(remote,label,ready,*,collection_cards=None):
    facts={}
    for role in ['core','hopper']:
        facts[role]=remote(role,r'''
import hashlib,json
from pathlib import Path
root=Path(ROOT);raw=(root/'RUN_MANIFEST.json').read_bytes();m=json.loads(raw);prior=[]
if ROLE=='core' and LABEL=='M1-256-CANONICAL':
 original=json.loads((root/'plans/M1-256/REGISTRY.json').read_text())
 prior=[json.loads((root/'episodes'/c['attempt_id']/'E1_ADJUDICATION.json').read_text()) for c in original['cases'].values()]
print(json.dumps(dict(source_sha=m['code']['vla_commit'],manifest_sha256=hashlib.sha256(raw).hexdigest(),
 prior_256=prior,cards=[c['card'] for c in json.loads((Path(E0_ROOT)/'evidence/model-M2048/REGISTRY.json').read_text())['cases'].values() if c['card']['split']=='TRAIN'] if ROLE=='core' else [])))
''',dict(ROLE=role,LABEL=label),label='E1-stage-bind-read-'+label)
    cases={}
    originals=collection_cards if collection_cards is not None else facts['core']['cards']
    expected_count=4
    if label=='M1-256-CANONICAL':
        recoverable=recoverable_256_conditions(facts['core']['prior_256'])
        originals=[c for c in originals if (c['scene_group'],c['task']) in recoverable];expected_count=1
    for index,original in enumerate(sorted(originals,key=lambda c:(c['scene_group'],c['task']))):
        card={k:v for k,v in original.items() if not k.startswith('relay_request_')}
        card['id']='PI0294-E1-'+label+'-'+card['scene_group'].split('/')[-2]+'-'+card['scene_group'].split('/')[-1]+'-'+card['task'];card['e1_model_label']=label
        if collection_cards is not None:card['e1_intervention_collection']=True
        episode=str(uuid.uuid4());relative='plans/'+label+'/requests/'+card['id']+'.json'
        request=dict(run_uuid=ready['run_uuid'],attempt_id=card['id'],source_sha=facts['core']['source_sha'],
            run_manifest_sha256=facts['core']['manifest_sha256'],initial_state_sha256=card['initial_state_sha256'],
            profile='B1',category='TRAIN_DEVELOPMENT',identity=ready['identity']['relay_identity'],episode_uuid=episode,
            deadline_epoch=ready['allocation_end_epoch']-2700,live_readiness_sha256='c2ca87bda6942f17b037d47bdacf33f069f2bd1ecb2f8d57126d87ba8a69ed5c')
        card['relay_request_relative']=relative;card['relay_request_sha256']=hashlib.sha256(encode(request).encode()).hexdigest()
        cases[episode]=dict(run_uuid=ready['run_uuid'],episode_uuid=episode,attempt_id=card['id'],identity=request['identity'],
            deadline_epoch=request['deadline_epoch'],layout_id=card['scene_group'],core_request_relative=relative,
            core_request_sha256=card['relay_request_sha256'],core_request=request,card=card,panel_index=index,
            hopper_session_relative='evidence/'+label+'/resident')
    if len(cases)!=expected_count:raise ValueError('EXACT_E1_PREDECLARED_PANEL_OR_ENGINEERING_INCOMPLETE_ONLY')
    registry=dict(run_uuid=ready['run_uuid'],deadline_epoch=ready['allocation_end_epoch']-2700,cases=cases,
        checkpoint_identity=ready['identity'],**{role+'_'+k:v for role,items in facts.items() for k,v in items.items() if k not in ['cards','prior_256']})
    for role in ['core','hopper']:
        remote(role,r'''
import json,os
from pathlib import Path
root=Path(ROOT);d=root/'plans'/LABEL;d.mkdir(parents=True,exist_ok=True)
def write(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);raw=json.dumps(v,indent=2,allow_nan=False)+'\n'
 if p.exists():assert p.read_text()==raw
 else:p.write_text(raw);p.chmod(0o600)
if ROLE=='core':
 assert not (root/'SOURCE_PHYSICS_STOP.json').exists(),'E1_SOURCE_PHYSICS_STOP_PRESERVED'
 for c in REGISTRY['cases'].values():
  write(root/c['core_request_relative'],c['core_request']);write(d/'cards'/(c['attempt_id']+'.json'),c['card'])
 write(d/'REMOTE_READY.json',READY)
write(d/'REGISTRY.json',REGISTRY)
print(json.dumps(dict(status='E1_FIXED_TRAIN4_REGISTRY_BOUND',label=LABEL)))
''',dict(ROLE=role,LABEL=label,REGISTRY=registry,READY=ready),label='E1-bind-'+label)
    return registry


class KnownIntervention(RuntimeError):pass


def existing_256_recovery_registry(remote,ready):
    values=[]
    for role in ['core','hopper']:
        values.append(remote(role,r'''
import json
from pathlib import Path
r=Path(ROOT);registry=json.loads((r/'plans/M1-256-CANONICAL/REGISTRY.json').read_text());prior=[]
if ROLE=='core':
 original=json.loads((r/'plans/M1-256/REGISTRY.json').read_text())
 prior=[json.loads((r/'episodes'/c['attempt_id']/'E1_ADJUDICATION.json').read_text()) for c in original['cases'].values()]
 target=[c for c in registry['cases'].values() if c['card']['scene_group']=='PI0294/TRAIN/F1/000' and c['card']['task']=='T2']
 assert len(target)==1 and not (r/'worker_processes'/(target[0]['attempt_id']+'.json')).exists()
 assert not (r/'SOURCE_PHYSICS_STOP.json').exists() and not json.loads((r/'RESOURCE_LEDGER.json').read_text())['reservations']
print(json.dumps(dict(registry=registry,prior=prior)))
''',dict(ROLE=role),label='read-only-original-256-incomplete-recovery-'+role))
    if values[0]['registry']!=values[1]['registry']:raise PermissionError('E1_REGISTERED_PANEL_SIDES_DIFFER')
    registry=values[0]['registry']
    if registry['checkpoint_identity']!=ready['identity'] or registry['run_uuid']!=ready['run_uuid']:raise PermissionError('E1_EXACT_EXISTING_MODEL_REQUIRED')
    allowed=recoverable_256_conditions(values[0]['prior'])
    registry['cases']={k:c for k,c in registry['cases'].items() if (c['card']['scene_group'],c['card']['task']) in allowed}
    if len(registry['cases'])!=1:raise PermissionError('E1_ONE_UNSTARTED_ENGINEERING_RECOVERY_ONLY')
    return registry


def routes(remote,base,label,ready,start_worker,*,collection_cards=None,bound_registry=None):
    if bound_registry is not None and (label!='M1-256-CANONICAL' or collection_cards is not None):raise PermissionError('E1_NO_ARBITRARY_REGISTRY_RESUME')
    registry=bound_registry if bound_registry is not None else bind(remote,label,ready,collection_cards=collection_cards)
    journal=Journal(base/'mutation-journal',ready['run_uuid'])
    def rpc(role,args):
        value=remote(role,RPC,dict(ROLE=role,LABEL=label,ARGS=args),label='e1-rpc-'+args['operation'])
        if not value['ok']:raise RuntimeError('E1_RPC_OUTCOME_UNKNOWN_NO_RESEND:'+value['error'])
        result=value['value']
        if isinstance(result,dict) and result.get('known_intervention'):
            if collection_cards is None:raise PermissionError('NO_TAKEOVER_DURING_AUTONOMOUS_EVALUATION')
            raise KnownIntervention('ACTUAL_STUDENT_S1_REJECTION_QUALIFIED_EXPERT_HANDOVER')
        if isinstance(result,dict) and result.get('known_terminal'):raise RuntimeError('KNOWN_CORE_TERMINAL:'+result['known_terminal']['failure_stage'])
        return result
    class Core:
        def observation(self,c,s):return rpc('core',dict(operation='observation',episode=c['episode_uuid'],sequence=s))
        def ack(self,c,s):return rpc('core',dict(operation='ack',episode=c['episode_uuid'],sequence=s))
        def install(self,c,s,b):
            args=dict(operation='install',episode=c['episode_uuid'],sequence=s,bundle=b)
            return journal.once('install-'+unseal(b['command'])['query_uuid'],args,lambda:rpc('core',args))
    class Hopper:
        last=None
        def query(self,c,q,teacher):
            if teacher is not None:raise PermissionError('NO_TEACHER_TAKEOVER_IN_E1_EVAL')
            self.last=q;args=dict(operation='query',episode=c['episode_uuid'],query=q)
            return journal.once('query-'+unseal(q)['query_uuid'],args,lambda:rpc('hopper',args))
        def response(self,c,q):return rpc('hopper',dict(operation='response',episode=c['episode_uuid'],query_uuid=q))
    core,hopper=Core(),Hopper();rows=[]
    for case in sorted(registry['cases'].values(),key=lambda c:c['panel_index']):
        if time.time()+300>=registry['deadline_epoch']:break
        args=['--card','plans/'+label+'/cards/'+case['attempt_id']+'.json']
        journal.once('start-'+case['attempt_id'],dict(case=case['attempt_id']),lambda:start_worker('scripts.pi0294.e1.core',case['attempt_id'],args))
        failure=None;rejection=None;hopper.last=None
        try:run_episode(base/'routes'/case['episode_uuid'],case,core,hopper)
        except KnownIntervention:
            failure=None
            write(base/'routes'/case['episode_uuid']/'KNOWN_INTERVENTION_ADJUDICATION.json',dict(
                status='INTENTIONAL_ONE_WAY_EXPERT_TAKEOVER_NOT_NETWORK_UNCERTAINTY',
                rejected_student_command_executed=False,model_episode_continues=False,
                expert_progress_requires_independent_actual_ACKs=True,no_student_resampling_or_command_replay=True))
        except BaseException as exc:
            failure=dict(type=type(exc).__name__,message=str(exc))
            if hopper.last:
                query=unseal(hopper.last)
                try:
                    action=hopper.response(case,query['query_uuid'])
                    if action:
                        gate=gate_action(unseal(action)['payload']['trajectory_eef10'],query['payload']['state_eef10'],query['tick']*.002,query['reference_uuid'])
                        if gate['decision']=='REJECT':rejection=dict(original_gate=gate,action_sha256=action['sha256'],query_uuid=query['query_uuid'])
                except Exception:pass
            remote('core',r'''
import json,os
from pathlib import Path
root=Path(ROOT);folder=root/'episodes'/ATTEMPT
def write(p,v):
 if not p.exists():p.write_text(json.dumps(v,indent=2)+'\n');p.chmod(0o600)
if REJECTION is not None:write(folder/'E1_MODEL_REJECT.json',REJECTION)
if (folder/'relay').is_dir():write(folder/'relay/STOP.json',dict(run_uuid=RUN,episode_uuid=EPISODE,reason='OWN_E1_ROUTE_TERMINAL_NO_REPLAY'))
print(json.dumps(dict(status='OWN_E1_CASE_STOP_REQUESTED')))
''',dict(ATTEMPT=case['attempt_id'],EPISODE=case['episode_uuid'],REJECTION=rejection),label='e1-stop-case')
        end=time.time()+(900 if collection_cards is not None else 90);terminal={};heartbeat_at=time.time()
        while time.time()<end:
            if time.time()-heartbeat_at>30:
                remote('hopper',r'''
import json,os,time
from pathlib import Path
p=Path(ROOT)/'OWNER_HEARTBEAT.json';t=p.with_name(p.name+'.pending.'+str(os.getpid()));t.write_text(json.dumps(dict(run_uuid=RUN,observed_epoch=time.time()))+'\n');t.chmod(0o600);os.replace(t,p);print(json.dumps(dict(status='OWNER_WAITING_ACTUAL_EXPERT_TERMINAL')))
''',label='e1-expert-progress-heartbeat');heartbeat_at=time.time()
            terminal=remote('core',r'''
import json
from pathlib import Path
root=Path(ROOT);folder=root/'episodes'/ATTEMPT;p=folder/'E1_ADJUDICATION.json'
w=json.loads((root/'worker_processes'/(ATTEMPT+'.json')).read_text());s=Path('/proc',str(w['pid']),'stat')
print(json.dumps(dict(terminal=not s.exists() or s.read_text().split()[2]=='Z',row=json.loads(p.read_text()) if p.exists() else None,source_stop=(root/'SOURCE_PHYSICS_STOP.json').exists(),log_tail=(root/'worker_logs'/(ATTEMPT+'.log')).read_text()[-1500:] if not p.exists() else '')))
''',dict(ATTEMPT=case['attempt_id']),label='e1-case-terminal')
            if terminal['terminal']:break
            time.sleep(1)
        row=terminal.get('row') or dict(attempt_id=case['attempt_id'],status='ENGINEERING_INCOMPLETE',pipeline_trustworthy=False,terminal=terminal)
        row['route_failure']=failure;rows.append(row);write(base/(label+'-RESULTS.private.json'),dict(rows=rows,planned=len(registry['cases'])))
        print(json.dumps(dict(label=label,covered=len(rows),result=row['status'],stage=row.get('failure_stage'))),flush=True)
        if terminal['source_stop'] or not terminal['terminal']:break
    return dict(label=label,rows=rows,planned=len(registry['cases']))
