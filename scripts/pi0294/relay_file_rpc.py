"""Finite SSH-stdio file RPC, never a model service or physical executor."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import select
import sys
import time

PROCESS_STARTED_CPU=time.process_time()

from scripts.pi0292.durable import write
from scripts.pi0294.e1.control_read import read
from scripts.pi0294.e1.rpc import call
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.relay_io import PhaseTrace
from scripts.pi0294.rpc_metrics import SessionMetrics

MAX_LINE_BYTES=2*1024**2


def ensure_model_not_terminal(root,label):
    """A published ready marker must not outlive its native model process."""
    root=Path(root);state=read(root/'pbs/MODEL_STATE.json')
    if state.get('label')!=label:
        raise PermissionError('EXACT_CURRENT_NATIVE_MODEL_SESSION')
    for path in (root/'pbs').glob('alloc-*/PBS_PROCESS_FINAL.json'):
        if read(path).get('job_id')==state['job_id']:
            raise PermissionError('NATIVE_MODEL_ALREADY_TERMINAL_NO_NEXT_QUERY')


def metadata(path):
    s=path.stat();return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_mode)


def execute_request(packet,invoke,guard,seen):
    if not isinstance(packet,dict) or set(packet)!={'rpc_id','args'} or not isinstance(packet['rpc_id'],str) or not 1<=len(packet['rpc_id'])<=64:
        raise ValueError('EXACT_STDIO_RPC_ENVELOPE')
    nonce=packet['rpc_id']
    if nonce in seen:raise PermissionError('DUPLICATE_STDIO_RPC_ID_NO_AUTOMATIC_RETRY')
    seen.add(nonce);args=packet['args']
    if not isinstance(args,dict) or args.get('operation') not in ['observation','ack','install','query','response'] or args.get('teacher') is not None:
        raise PermissionError('MODEL_FILE_RPC_ONLY_NO_TEACHER_ACCESS')
    guard(args)
    return dict(rpc_id=nonce,ok=True,value=invoke(args))


def operation_budget(root,manifest,registry,*,pv1):
    if not pv1 or not manifest.get('execution',{}).get('control_budget_profile'):
        return 40000,None
    from scripts.pi0294.pv1.control_budget import validate as control_budget
    control=control_budget(manifest,read(Path(root)/'PV1_STAGE_BINDING.json'))
    if (registry.get('control_budget_profile')!=control['profile']
            or len(registry.get('cases',{}))>control['conditions_per_panel_max']):
        raise PermissionError('PV1_RPC_REGISTRY_FINITE_CONTROL_PROFILE_REQUIRED')
    return control['max_calls'],control


def serve(root,role,label):
    if label.startswith('PV1-PREMODEL'):
        from scripts.pi0294.pv1.transport_fixture import serve as serve_fixture
        return serve_fixture(root,role,label)
    if label.startswith('N1-PREMODEL'):
        from scripts.pi0294.mp1.transport_fixture import serve as serve_fixture
        return serve_fixture(root,role,label)
    root=Path(root);pv1=(root/'PV1_STAGE_BINDING.json').is_file()
    if pv1:
        cpu=read(root/'RUN_MANIFEST.json')['limits']['rpc_cpu']
        if type(cpu) is not int or cpu not in os.sched_getaffinity(0):raise PermissionError('PV1_REGISTERED_RPC_CPU_ONLY')
        os.sched_setaffinity(0,{cpu})
    else:os.sched_setaffinity(0,{sorted(os.sched_getaffinity(0))[3 if role=='core' else 0]})
    resource.setrlimit(resource.RLIMIT_CPU,(7200,7200));resource.setrlimit(resource.RLIMIT_CORE,(0,0));started=PROCESS_STARTED_CPU
    os.environ.pop('E1_RUN_MANIFEST',None);os.environ.pop('MP1_RUN_MANIFEST',None)
    original_manifest=(root/'RUN_MANIFEST.json').read_bytes();m=json.loads(original_manifest)
    if pv1:
        if (root/'MP1_STAGE_BINDING.json').exists():raise PermissionError('PV1_NO_MIXED_OLD_STAGE_AUTHORITY')
        from scripts.pi0294.pv1.visibility import Endpoint
        from scripts.pi0294.pv1.evaluation_plan import session_spec
        from scripts.pi0294.pv1.core_ledger import Ledger as CoreLedger
        from scripts.pi0294.pv1.neural_ledger import Ledger as NeuralLedger
        session_spec(root,label)
        def Ledger(path,side):
            if side=='core':return CoreLedger(path)
            if side=='hopper':return NeuralLedger(path)
            raise PermissionError('PV1_EXACT_SOURCE_POLICY_RPC_ROLE')
        adjudication='PV1_ADJUDICATION.json'
    elif root.name=='20261016T150228Z-localpc-e1':
        if label not in ['M1-1024-CANONICAL-REMAINDER-03','M2-BC-REMAINDER-03','M2-CORR-REMAINDER-03']:raise PermissionError('E1_FAST_FILE_RPC_ONLY_LAST_FIXED_REMAINDER')
        from scripts.pi0294.e1.endpoint import Endpoint
        from scripts.pi0294.e1.scope import Ledger
        adjudication='E1_ADJUDICATION.json'
    elif (root/'MP1_STAGE_BINDING.json').is_file():
        from scripts.pi0294.relay_endpoint import Endpoint
        from scripts.pi0294.mp1.science import Ledger
        from scripts.pi0294.mp1.plans import session_spec
        session_spec(read(root/'plans/SOURCE_EVALUATION_PLAN.json'),label,root=root)
        adjudication='MP1_ADJUDICATION.json'
    else:raise PermissionError('ADOPTED_E1_REMAINDER_OR_MP1_SOURCE_ONLY')
    ledger=Ledger(root,role);registry=root/'plans'/label/'REGISTRY.json';registry_hash=sha(registry)
    endpoint=Endpoint(root,str(registry.relative_to(root)),registry_hash,role)
    binding=read(root/'OUTPUT_BINDING.private.json');seal=root/m['code']['source_seal_relative'];seal_hash=sha(seal)
    code=root/m['code']['capsule_relative'];files=[code/name for name in read(seal)['files']];file_metadata={p:metadata(p) for p in files}
    copy=code.parent/('STDIO_MANIFEST_'+role+'_'+label+'.json')
    if copy.exists():raise FileExistsError('PRESERVE_PRIOR_STDIO_SESSION_NO_BLIND_RESTART')
    copy.write_bytes(original_manifest);copy.chmod(0o400)
    name='file-rpc-'+role+'-'+label;records=root/'aux_worker_processes';records.mkdir(exist_ok=True,mode=0o700);record=records/(name+'.json')
    if record.exists():raise FileExistsError('ONE_FIXED_FILE_SESSION_PER_ROLE_AND_PANEL')
    write(record,dict(pid=os.getpid(),proc_start=Path('/proc/self/stat').read_text().split()[21],source_sha=m['code']['vla_commit'],
        manifest_relative=str(copy.relative_to(root)),manifest_sha256=hashlib.sha256(original_manifest).hexdigest(),source_live_worker=False,
        work_type='CONTROL_FILE_RPC_NO_MODEL_NO_PHYSICS',maximum_CPU_s=7200,deadline_epoch=endpoint.registry['deadline_epoch'],started_epoch=time.time()))
    max_calls,control=operation_budget(root,m,endpoint.registry,pv1=pv1)
    owner_heartbeat_only=False
    if pv1:
        from scripts.pi0294.pv1.control_heartbeat import owner_only,keepalive
        owner_heartbeat_only=owner_only(root,m)
    if control and endpoint.registry['deadline_epoch']-time.time()>control['wall_limit_s']:
        raise PermissionError('PV1_ORIGINAL_MAXIMUM_ALLOCATION_CONTROL_WALL')
    seen=set();metrics=SessionMetrics();status='SESSION_EOF_OR_DEADLINE';error=None;trace=PhaseTrace(sys.stderr)
    def guard(args):
        ledger.guard()
        if time.time()>=endpoint.registry['deadline_epoch']:raise TimeoutError('ORIGINAL_PANEL_DEADLINE_UNCHANGED')
        if (root/'RUN_MANIFEST.json').read_bytes()!=original_manifest or sha(seal)!=seal_hash or sha(registry)!=registry_hash:
            raise PermissionError('IMMUTABLE_FILE_SESSION_BINDING_CHANGED')
        lease=read(binding['lease'])
        if lease['status']!='ACTIVE' or lease['run_uuid']!=root.name:raise PermissionError('ACTUAL_ACTIVE_FILE_SESSION_LEASE')
        if args['operation'] in ['query','install']:
            if any(metadata(p)!=v for p,v in file_metadata.items()):raise PermissionError('SEALED_SOURCE_METADATA_CHANGED_NO_HOT_RELOAD')
            if (root/'SOURCE_PHYSICS_STOP.json').exists() or (root/'SOURCE_SAFETY_STOP_FROM_CORE.json').exists():raise PermissionError('SOURCE_SAFETY_STOP_PRESERVED')
        if role=='hopper':ensure_model_not_terminal(root,label)
        if role=='hopper':
            if pv1:keepalive(root,role,readonly_owner=owner_heartbeat_only)
            else:write(root/'OWNER_HEARTBEAT.json',dict(run_uuid=root.name,observed_epoch=time.time()))
    try:
        while time.time()<endpoint.registry['deadline_epoch'] and metrics.operations<max_calls:
            readable,_,_=select.select([sys.stdin],[],[],min(30,max(0,endpoint.registry['deadline_epoch']-time.time())))
            if not readable:continue
            line=sys.stdin.buffer.readline(MAX_LINE_BYTES+1)
            if not line:break
            if len(line)>MAX_LINE_BYTES or not line.endswith(b'\n'):raise ValueError('BOUNDED_ONE_JSON_LINE_REQUIRED')
            packet=json.loads(line);nonce=packet.get('rpc_id') if isinstance(packet,dict) else None
            trace.bind_rpc(packet)
            trace.received(len(line))
            try:
                def checked(args):
                    with trace.phase('guard'):guard(args)
                def invoke(args):
                    with trace.phase('invoke'):return call(root,role,label,args,endpoint=endpoint,adjudication_name=adjudication)
                value=execute_request(packet,invoke,checked,seen)
            except Exception as exc:
                value=dict(rpc_id=nonce,ok=False,error=type(exc).__name__+': '+str(exc),unknown_effect_not_retried=True)
                print(json.dumps(value,allow_nan=False),flush=True);raise
            with trace.phase('emit'):
                metrics.reply(value,lambda v:print(json.dumps(v,allow_nan=False),flush=True))
        if metrics.operations>=max_calls:status='FINITE_CONTROL_OPERATION_BOUND_REACHED'
    except Exception as exc:status='FAILED_SESSION_PRESERVED';error=type(exc).__name__+': '+str(exc);raise
    finally:
        cpu=time.process_time()-started
        trace.bind_rpc(None)
        with trace.phase('accounting'):settlement=metrics.settle(ledger,cpu)
        terminal=root/'evidence/file-rpc-sessions'/name;terminal.mkdir(parents=True,exist_ok=True,mode=0o700)
        write(terminal/'TERMINAL.json',dict(status=status,error=error,operations=metrics.operations,CPU_s=cpu,
            maximum_operations=max_calls,versioned_control_budget=control,
            metrics_settlement=settlement,per_reply_shared_ledger_transaction=False,
            crash_before_final_settlement_requires_registered_CPU_bound_reconciliation=True,
            new_model_generations_by_this_process=0,new_physical_integrations=0,original_call_and_action_watchdogs_unchanged=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--role',choices=['core','hopper'],required=True);p.add_argument('--label',required=True);a=p.parse_args();serve(a.root,a.role,a.label)
