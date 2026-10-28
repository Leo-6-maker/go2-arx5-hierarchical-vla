"""PV1 owner bindings around the original fresh-observation ACK exchange."""
from copy import deepcopy
import hashlib
import json
import time
import uuid

from scripts.pi0292.durable import write
from scripts.pi0292.eval_contract import gate_action
from scripts.pi0292.relay_protocol import unseal
from scripts.pi0294.e1.journal import Journal
from scripts.pi0294.relay_transport import run_episode
from .evaluation_plan import remaining_panel,validate_registry
from .control_budget import PROFILE,POLL_S

LIVE_READINESS='c2ca87bda6942f17b037d47bdacf33f069f2bd1ecb2f8d57126d87ba8a69ed5c'


def registry(spec,ready,peers,prior,*,deadline_epoch,now=None):
    now=time.time() if now is None else now
    proof=ready.get('native_pre_model_qualification',{})
    if (ready['status']!='PV1_NATIVE_MODEL_READY' or ready['label']!=spec['cards'][0]['pv1_session_label']
            or ready['identity']['global_step']!=spec['checkpoint_step']
            or ready['identity']['label']!=spec['candidate']
            or proof.get('status')!='PASS_SAME_JOB_NATIVE_QUALIFICATION_READBACK'
            or proof.get('heartbeat_elapsed_s',0)<600 or proof.get('transport_cycles')!=80
            or proof.get('new_model_queries')!=0 or proof.get('new_physics')!=0
            or not now<deadline_epoch<=ready['allocation_end_epoch']-2700):
        raise PermissionError('PV1_ACTUAL_FROZEN_NATIVE_MODEL_AND_SAME_JOB_PROOF_BEFORE_REGISTRY')
    cases={}
    for index,base in enumerate(remaining_panel(prior,spec)):
        card=deepcopy(base);episode=str(uuid.uuid4())
        relative='plans/'+ready['label']+'/requests/'+card['id']+'.json'
        request=dict(run_uuid=ready['run_uuid'],attempt_id=card['id'],source_sha=peers['core_source_sha'],
            run_manifest_sha256=peers['core_manifest_sha256'],initial_state_sha256=card['initial_state_sha256'],
            profile='B1',category=spec['split'],identity=ready['identity']['relay_identity'],episode_uuid=episode,
            deadline_epoch=deadline_epoch,live_readiness_sha256=LIVE_READINESS)
        digest=hashlib.sha256((json.dumps(request,indent=2,allow_nan=False)+'\n').encode()).hexdigest()
        card.update(relay_request_relative=relative,relay_request_sha256=digest)
        cases[episode]=dict(run_uuid=ready['run_uuid'],episode_uuid=episode,attempt_id=card['id'],
            identity=request['identity'],deadline_epoch=deadline_epoch,layout_id=card['scene_group']+'/'+card['task'],
            core_request_relative=relative,core_request_sha256=digest,core_request=request,card=card,panel_index=index,
            hopper_session_relative='evidence/'+ready['label']+'/resident')
    value=dict(run_uuid=ready['run_uuid'],job_id=ready['job_id'],deadline_epoch=deadline_epoch,cases=cases,
        checkpoint_identity=ready['identity'],native_pre_model_receipt_sha256=proof['receipt_sha256'],
        control_budget_profile=PROFILE,**peers)
    validate_registry(value,spec,ready['identity'],run_uuid=ready['run_uuid'],job_id=ready['job_id'],peer=peers,prior_rows=prior)
    return value


def routes(remote,base,session,ready,bound,start_worker,transport,terminal_audit,heartbeat):
    if bound.get('control_budget_profile')!=PROFILE:
        raise PermissionError('PV1_PREBOUND_FINITE_CONTROL_PROFILE_REQUIRED')
    journal=Journal(base/'mutation-journal',ready['run_uuid']);rows=[]
    def rpc(role,args):
        heartbeat();value=transport.call(role,args)
        if not value['ok']:raise RuntimeError('PV1_UNKNOWN_FILE_RPC_OUTCOME_NO_RESEND:'+value['error'])
        value=value['value']
        if isinstance(value,dict) and value.get('known_intervention'):
            raise PermissionError('PV1_TEACHER_OFF_AUTONOMOUS_EVALUATION')
        if isinstance(value,dict) and value.get('known_terminal'):
            raise RuntimeError('PV1_KNOWN_CORE_TERMINAL:'+value['known_terminal']['failure_stage'])
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
            if teacher is not None:raise PermissionError('PV1_NO_TEACHER_IN_STUDENT_INPUT')
            self.last=q;args=dict(operation='query',episode=c['episode_uuid'],query=q)
            return journal.once('query-'+unseal(q)['query_uuid'],args,lambda:rpc('hopper',args))
        def response(self,c,q):return rpc('hopper',dict(operation='response',episode=c['episode_uuid'],query_uuid=q))
    core,hopper=Core(),Hopper()
    def poisoned():return any(s.failed for s in transport.sessions.values())
    for case in sorted(bound['cases'].values(),key=lambda c:c['panel_index']):
        if poisoned():return dict(status='ENGINEERING_BRANCH_STOP_NO_RETRY',rows=rows,abort_remaining_sessions=True)
        if time.time()+900>=bound['deadline_epoch']:break
        args=['--card','plans/'+session+'/cards/'+case['attempt_id']+'.json']
        journal.once('start-'+case['attempt_id'],dict(attempt_id=case['attempt_id']),
            lambda:start_worker('scripts.pi0294.pv1.core',case['attempt_id'],args))
        failure=None;rejection=None;hopper.last=None
        try:run_episode(base/'routes'/case['episode_uuid'],case,core,hopper,poll_interval=POLL_S)
        except BaseException as exc:
            failure=dict(type=type(exc).__name__,message=str(exc))
            if hopper.last and not poisoned():
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
f=Path(ROOT)/'episodes'/ATTEMPT
def put(p,v):
 if not p.exists():p.write_text(json.dumps(v,indent=2)+'\n');p.chmod(0o600)
if REJECTION is not None:put(f/'MODEL_REJECT.json',REJECTION)
if (f/'relay').is_dir():put(f/'relay/STOP.json',dict(run_uuid=RUN,episode_uuid=EPISODE,reason='PV1_OWN_CASE_STOP_NO_REPLAY'))
print(json.dumps(dict(status='PV1_OWN_CASE_STOP_REQUESTED_NO_REPLAY')))
''',dict(ATTEMPT=case['attempt_id'],EPISODE=case['episode_uuid'],REJECTION=rejection),label='PV1-own-case-terminal-stop-no-replay')
        end=time.time()+90;terminal={}
        while time.time()<end:
            heartbeat()
            terminal=remote('core',r'''
import json
from pathlib import Path
r=Path(ROOT);p=r/'episodes'/ATTEMPT/'PV1_ADJUDICATION.json';w=json.loads((r/'worker_processes'/(ATTEMPT+'.json')).read_text())
s=Path('/proc',str(w['pid']),'stat');f=s.read_text().split() if s.exists() else None
alive=bool(f and f[2]!='Z' and f[21]==str(w['proc_start']))
print(json.dumps(dict(terminal=not alive,row=json.loads(p.read_text()) if p.exists() else None,
 source_stop=(r/'SOURCE_PHYSICS_STOP.json').exists())))
''',dict(ATTEMPT=case['attempt_id']),label='PV1-real-case-terminal-readonly')
            if terminal['terminal']:break
            time.sleep(5)
        if not terminal.get('terminal') or not terminal.get('row'):
            return dict(status='UNKNOWN_NATIVE_OUTCOME_PRESERVED_NOT_RETRIED',rows=rows,attempt_id=case['attempt_id'],
                route_failure=failure,abort_remaining_sessions=True)
        row=terminal['row'];audit=terminal_audit(case['attempt_id'])
        record=dict(row,route_failure=failure,independent_audit=audit)
        rows.append(record);write(base/(session+'-RESULTS.private.json'),dict(session=session,rows=rows,planned=len(bound['cases'])))
        print(json.dumps(dict(session=session,covered=len(rows),status=row['status'])),flush=True)
        if (poisoned() or terminal.get('source_stop') or row['status']=='ACTUAL_SAFETY_FAIL' or row['failure_stage']=='S1_INTEGRITY'
                or audit['status'] not in ('PASS_INDEPENDENT_ACTUAL_EVIDENCE_AUDIT','PASS_ZERO_INTEGRATION_ENGINEERING_EVIDENCE')):
            return dict(status='SOURCE_OR_ENGINEERING_STOP_PRESERVED',rows=rows,abort_remaining_sessions=True)
        if row['status']=='ENGINEERING_INCOMPLETE':
            return dict(status='CONFIRMED_DIAGNOSIS_REQUIRED_BEFORE_ANY_ENGINEERING_REPLACEMENT',rows=rows,abort_remaining_sessions=True)
    return dict(status='FINITE_PANEL_ACTUAL_RESULTS',rows=rows,planned=len(bound['cases']))
