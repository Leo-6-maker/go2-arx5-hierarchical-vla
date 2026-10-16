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


def bind(remote,label,ready):
    facts={}
    for role in ['core','hopper']:
        facts[role]=remote(role,r'''
import hashlib,json
from pathlib import Path
root=Path(ROOT);raw=(root/'RUN_MANIFEST.json').read_bytes();m=json.loads(raw)
print(json.dumps(dict(source_sha=m['code']['vla_commit'],manifest_sha256=hashlib.sha256(raw).hexdigest(),
 cards=[c['card'] for c in json.loads((Path(E0_ROOT)/'evidence/model-M2048/REGISTRY.json').read_text())['cases'].values() if c['card']['split']=='TRAIN'] if ROLE=='core' else [])))
''',dict(ROLE=role),label='E1-stage-bind-read-'+label)
    cases={}
    for index,original in enumerate(sorted(facts['core']['cards'],key=lambda c:(c['family'],c['task']))):
        card={k:v for k,v in original.items() if not k.startswith('relay_request_')}
        card['id']='PI0294-E1-'+label+'-'+card['family']+'-000-'+card['task'];card['e1_model_label']=label
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
    if len(cases)!=4:raise ValueError('EXACT_E1_ORIGINAL_TRAIN4_REQUIRED')
    registry=dict(run_uuid=ready['run_uuid'],deadline_epoch=ready['allocation_end_epoch']-2700,cases=cases,
        checkpoint_identity=ready['identity'],**{role+'_'+k:v for role,items in facts.items() for k,v in items.items() if k!='cards'})
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


def routes(remote,base,label,ready,start_worker):
    registry=bind(remote,label,ready);journal=Journal(base/'mutation-journal',ready['run_uuid'])
    def rpc(role,args):
        value=remote(role,RPC,dict(ROLE=role,LABEL=label,ARGS=args),label='e1-rpc-'+args['operation'])
        if not value['ok']:raise RuntimeError('E1_RPC_OUTCOME_UNKNOWN_NO_RESEND:'+value['error'])
        result=value['value']
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
        end=time.time()+90;terminal={}
        while time.time()<end:
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
        row['route_failure']=failure;rows.append(row);write(base/(label+'-RESULTS.private.json'),dict(rows=rows,planned=4))
        print(json.dumps(dict(label=label,covered=len(rows),result=row['status'],stage=row.get('failure_stage'))),flush=True)
        if terminal['source_stop'] or not terminal['terminal']:break
    return dict(label=label,rows=rows,planned=4)
