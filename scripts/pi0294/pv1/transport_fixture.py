"""PV1 bindings around the mature recorded-packet transport primitives."""
import os
import base64
from pathlib import Path
import re
import resource
import select
import sys
import time
import uuid
import json

from scripts.pi0292.durable import OwnerLock,write
from scripts.pi0294.e1.control_read import read
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.relay_endpoint import inside
from scripts.pi0294.rpc_metrics import SessionMetrics
from .visibility import read_visible,ALGORITHM
from scripts.pi0294.mp1.transport_fixture import (
    Endpoint as OriginalEndpoint,Pump as OriginalPump,immutable,check_observation,
    CYCLES,receive_query,response_for)
from .decision import AUTHORITY
from .resources import namespace_submission
from .runtime import validate_runtime
from . import evaluation_platform as evaluation

PLAN='plans/PV1_TRANSPORT_FIXTURE.json'
BASE='PV1-PREMODEL'


def namespace(root,label):
    match=re.fullmatch(r'(PV1-PREMODEL(?:-FAULT-LATE|-FAULT-EXIT)?)-ALLOCATION-([0-9]{2})',label)
    if match is None:
        raise PermissionError('PV1_EXACT_FINITE_TRANSPORT_NAMESPACE')
    number=namespace_submission(root,int(match[2]))
    return inside(Path(root),'engineering/pv1-transport/'+label),number,match[1]


def prepared_plan(root,m):
    root=Path(root);p=inside(root,PLAN);v=immutable(p)
    if (m['execution']['sealed_inputs'].get(PLAN)!=sha(p)
            or v['schema']!='PI0294_PV1_RECORDED_TRANSPORT_FIXTURE_V1'
            or v['run_uuid']!=root.name or v['scope']!='RECORDED_OBSERVATION_NO_MODEL_NO_PHYSICS'
            or v['cycles']!=80 or v.get('NFS_lookup_cache_assistance')!=ALGORITHM
            or v['recorded_observation_relative']!='reference/RECORDED_OBSERVATION.json'
            or v['raw_reference_binding_relative']!='reference/RAW_REFERENCE_BINDING.private.json'
            or not re.fullmatch(r'[0-9a-f]{64}',v['raw_reference_binding_sha256'])
            or not re.fullmatch(r'[0-9a-f]{64}',v['recorded_observation_file_sha256'])
            or not re.fullmatch(r'[0-9a-f]{64}',v['recorded_payload_sha256'])
            or not re.fullmatch(r'[0-9a-f]{64}',v['reference_proof_sha256'])
            or len(v['RGB_png_bytes'])!=2 or any(type(n) is not int or n<=0 for n in v['RGB_png_bytes'])
            or not 0<v['RGB_payload_bytes']<2*1024**2
            or v['RGB_payload_bytes']!=sum(4*((n+2)//3) for n in v['RGB_png_bytes'])):
        raise PermissionError('PV1_SEALED_RECORDED_RGB_PLAN_NOT_POLICY_ACTIONS')
    reference=inside(root,v['recorded_observation_relative'])
    if sha(reference)!=v['recorded_observation_file_sha256']:
        raise ValueError('PV1_RECORDED_REFERENCE_BYTES_CHANGED')
    check_observation(immutable(reference),v)
    return v


def reference_proof(root,plan,m):
    binding=inside(root,plan['raw_reference_binding_relative'])
    if (sha(binding)!=plan['raw_reference_binding_sha256']
            or m['execution']['sealed_inputs'].get(plan['raw_reference_binding_relative'])!=sha(binding)):
        raise PermissionError('PV1_CORE_RAW_REFERENCE_BINDING_MUST_BE_SEALED')
    b=immutable(binding);old=Path(b['original_root']);output=read(old/'OUTPUT_BINDING.private.json')
    if output['uid']!=os.getuid() or read(output['lease'])['status']!='CLOSED':
        raise PermissionError('PV1_RECORDED_REFERENCE_ONLY_OWN_CLOSED_ORIGINAL_SOURCE')
    p=inside(old,b['original_observation_relative']);episode=p.parent.parent.parent
    audit_path=inside(old,b['original_audit_relative']) if 'original_audit_relative' in b else episode/'AUDIT.json'
    audit=read(audit_path)
    if 'original_audit_relative' in b:
        if (audit_path.resolve()!=(old/'evidence/audits'/episode.name/'AUDIT.private.json').resolve()
                or audit['status']!='PASS_INDEPENDENT_ACTUAL_EVIDENCE_AUDIT' or audit['errors']
                or not audit['checks'] or not all(v is True for v in audit['checks'].values())
                or audit['actual_native_safety_flag_rows'] or audit['adjudication']['attempt_id']!=episode.name
                or audit['adjudication']['pipeline_trustworthy'] is not True
                or audit['adjudication']['live_native_failures']
                or audit['adjudication']['result_sha256']!=sha(episode/'RESULT.json')):
            raise PermissionError('PV1_REFERENCE_EXACT_ORIGINAL_COMPLETE_ACTUAL_EVIDENCE_AUDIT')
    elif audit['status']!='PASS_INDEPENDENT_RAW_AUDIT':
        raise PermissionError('PV1_REFERENCE_ORIGINAL_FULL_RAW_AUDIT_REQUIRED')
    if (sha(p)!=plan['recorded_observation_file_sha256']
            or sha(episode/'FILE_MANIFEST.json')!=b['raw_manifest_sha256']
            or sha(audit_path)!=b['audit_sha256']):
        raise PermissionError('PV1_REFERENCE_REAL_ORIGINAL_RAW_AND_AUDIT_IDENTITY')
    result=read(episode/'RESULT.json');s1=read(episode/'S1_RESULT.json')
    if result['status']!='VALID_COMPLETE' or result['physics_steps']!=16000 or s1['live_native_failures']:
        raise PermissionError('PV1_RECORDING_MUST_HAVE_REAL_COMPLETE_SAFE_PHYSICS')
    body=check_observation(immutable(inside(root,plan['recorded_observation_relative'])),plan)
    pixels={}
    for image,camera in zip(body['payload']['images'],('body_rgb','wrist_rgb')):
        original=inside(episode,camera+'/'+str(body['tick']//25).zfill(4)+'.png')
        if sha(original)!=image['sha256'] or original.read_bytes()!=base64.b64decode(image['png_base64'],validate=True):
            raise PermissionError('PV1_FIXTURE_PIXELS_MUST_EQUAL_ACTUAL_ORIGINAL_RENDERED_PNG')
        pixels[camera]=image['sha256']
    return dict(status='PASS_ACTUAL_CLOSED_SOURCE_RECORDED_TWO_CAMERA_REFERENCE',
        original_run_uuid=old.name,original_episode_id=episode.name,original_raw_manifest_sha256=b['raw_manifest_sha256'],
        original_audit_sha256=b['audit_sha256'],recorded_observation_file_sha256=sha(p),
        recorded_payload_sha256=plan['recorded_payload_sha256'],actual_two_camera_pixels=pixels,
        raw_reference_binding_sha256=sha(binding),new_models=0,new_queries=0,new_physics=0)


def ledger(root,role):
    if role=='core':
        from .core_ledger import Ledger
    elif role=='hopper':
        from .neural_ledger import Ledger
    else:raise PermissionError('PV1_EXACT_TRANSPORT_PEER')
    return Ledger(root)


class Endpoint(OriginalEndpoint):
    def __init__(self,root,role,label,*,now=time.time):
        self.root,self.role,self.label,self.now=Path(root),role,label,now
        if role not in ('core','hopper'):raise PermissionError('PV1_TYPED_FIXTURE_ROLE')
        self.folder,number,self.mode=namespace(root,label)
        profile=evaluation.file_RPC_profile({} if role=='core' else read(self.root/'RUN_MANIFEST.json'),role)
        self.manifest=validate_runtime(root,profile)
        self.plan=prepared_plan(root,self.manifest);self.registry_path=inside(self.root,'plans/'+label+'/REGISTRY.json')
        self.registry=immutable(self.registry_path);self.registry_sha=sha(self.registry_path)
        self.manifest_sha=sha(self.root/'RUN_MANIFEST.json');self.ledger=ledger(root,role)
        a=read(self.root.parent.parent/'PV1_ADOPTION_AND_RESOURCE_BINDING.json');r=self.registry
        if (r['schema']!='PI0294_PV1_TRANSPORT_REGISTRY_V1' or r['authority_release']!=AUTHORITY
                or r['run_uuid']!=self.root.name or r['campaign_run_uuid']!=a['run_uuid']
                or r['label']!=label or r['allocation_number']!=number or r['plan_sha256']!=sha(self.root/PLAN)
                or r[role+'_manifest_sha256']!=self.manifest_sha
                or r[role+'_source_sha']!=self.manifest['code']['vla_commit']
                or not self.now()<r['deadline_epoch']<=min(r['allocation_end_epoch']-2700,a['stop_new_physics_epoch'])
                or not re.fullmatch(evaluation.fixture_job_pattern(self.manifest,role),r['job_id'])):
            raise PermissionError('PV1_SAME_ALLOCATION_SOURCE_AUTHORITY_AND_ORIGINAL_DEADLINE')
        uuid.UUID(r['fixture_uuid'])
        self.folder.mkdir(parents=True,exist_ok=True,mode=0o700)
        for name in ('observations','inbox','outbox','installs','completed'):
            p=inside(self.folder,name);p.mkdir(exist_ok=True,mode=0o700)
        self.recorded=immutable(inside(self.root,self.plan['recorded_observation_relative'])) if role=='core' else None
        if role=='core':
            proof=reference_proof(self.root,self.plan,self.manifest);path=self.folder/'REFERENCE_PROOF.json'
            if path.exists():
                if immutable(path)!=proof:raise ValueError('PV1_RECORDED_REFERENCE_PROOF_CHANGED')
            else:write(path,proof);path.chmod(0o400)
            if sha(path)!=self.plan['reference_proof_sha256']:
                raise PermissionError('PV1_REFERENCE_PROOF_MUST_EQUAL_PRESEALED_CORE_RAW_IDENTITY')

    def invoke(self,args):
        if self.role=='hopper' and args.get('operation')=='response':
            self.guard(args)
            lock=OwnerLock(self.folder/('ENDPOINT-'+self.role+'.lock'))
            try:return read_visible(self.path('outbox',args['sequence']))
            finally:lock.close()
        return super().invoke(args)


class Pump(OriginalPump):
    def __init__(self,root,label,allocation,guard):
        self.root,self.label,self.allocation,self.guard=Path(root),label,allocation,guard
        self.folder,number,mode=namespace(root,label)
        if mode!=BASE or os.environ.get('PBS_JOBID')!=allocation['job_id'] or os.uname().nodename!=allocation['hostname']:
            raise PermissionError('PV1_FIXTURE_PUMP_ONLY_ON_OWN_ADMITTED_COMPUTE_NODE')
        r=immutable(self.root/'plans'/label/'REGISTRY.json');self.registry=r
        m=read(self.root/'RUN_MANIFEST.json');self.plan=prepared_plan(root,m)
        admission=self.root/'pbs'/('alloc-'+str(number).zfill(2))/'ALLOCATION_ADMISSION.json'
        typed=evaluation.cold_profile(m)
        expected=evaluation.STATUS if typed else 'PASS_PV1_OWN_NATIVE_H200_ALLOCATION'
        if typed:
            request=self.root/'pbs'/('alloc-'+str(number).zfill(2))/'REQUEST.json'
            evaluation.sealed_identity(m,read(request),allocation,run_uuid=self.root.name,
                manifest_sha256=sha(self.root/'RUN_MANIFEST.json'),request_sha256=sha(request))
        if (read(admission)!=allocation or allocation['status']!=expected
                or r['job_id']!=allocation['job_id'] or r['label']!=label
                or r['allocation_number']!=number or r['hopper_manifest_sha256']!=sha(self.root/'RUN_MANIFEST.json')
                or r['hopper_source_sha']!=m['code']['vla_commit']):
            raise PermissionError('PV1_NATIVE_PUMP_EXACT_CURRENT_ADMISSION_AND_REGISTRY')
        self.folder.mkdir(parents=True,exist_ok=True,mode=0o700)
        self.process=OwnerLock(self.folder/'COMPUTE.lock');self.next=0
        if (self.folder/'COMPUTE_STARTED.json').exists():
            self.process.close();raise FileExistsError('PV1_PRESERVE_PRIOR_COMPUTE_FIXTURE')
        for name in ('inbox','outbox','compute-intents','completed'):inside(self.folder,name).mkdir(exist_ok=True,mode=0o700)
        write(self.folder/'COMPUTE_STARTED.json',dict(job_id=allocation['job_id'],hostname=allocation['hostname'],
            pid=os.getpid(),proc_start=Path('/proc/self/stat').read_text().split()[21],models_loaded=0,new_queries=0,new_physics=0))

    def poll(self):
        if self.next>=CYCLES:return
        path=inside(self.folder,'inbox/'+str(self.next).zfill(3)+'.json')
        query=read_visible(path)
        if query is None:return
        self.guard();q=receive_query(query,self.registry,self.plan)
        if q['sequence']!=self.next:raise ValueError('CONTIGUOUS_EIGHTY_NATIVE_FIXTURE_QUERIES')
        intent=inside(self.folder,'compute-intents/'+path.name);response=inside(self.folder,'outbox/'+path.name)
        if intent.exists() or response.exists():raise PermissionError('UNKNOWN_FIXTURE_COMPUTE_EFFECT_NO_REPLAY')
        write(intent,dict(request_sha256=query['sha256'],sequence=self.next,started_epoch=time.time()))
        write(response,response_for(query,self.registry,self.plan,time.time()));self.next+=1


def serve(root,role,label):
    from scripts.pi0294.relay_file_rpc import execute_request,MAX_LINE_BYTES
    endpoint=Endpoint(root,role,label);folder=endpoint.folder
    records=Path(root)/'aux_worker_processes';records.mkdir(exist_ok=True,mode=0o700)
    record=records/('fixture-rpc-'+role+'-'+label+'.json')
    if record.exists():raise FileExistsError('PV1_ONE_FINITE_FIXTURE_STDIO_PROCESS_NO_RESTART')
    started=time.process_time();code=Path(root)/endpoint.manifest['code']['capsule_relative']
    copy=code.parent/('STDIO_FIXTURE_MANIFEST_'+role+'_'+label+'.json')
    if copy.exists():raise FileExistsError('PV1_IMMUTABLE_STDIO_COPY_ALREADY_EXISTS')
    copy.write_bytes((Path(root)/'RUN_MANIFEST.json').read_bytes());copy.chmod(0o400)
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_CPU,(180,180));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    write(record,dict(pid=os.getpid(),proc_start=Path('/proc/self/stat').read_text().split()[21],
        source_sha=endpoint.manifest['code']['vla_commit'],manifest_relative=str(copy.relative_to(root)),
        manifest_sha256=sha(copy),source_live_worker=False,work_type='PV1_ISOLATED_TRANSPORT_NO_MODEL_NO_PHYSICS',
        maximum_CPU_s=180,deadline_epoch=endpoint.registry['deadline_epoch'],job_id=endpoint.registry['job_id']))
    seen=set();metrics=SessionMetrics();status='FIXTURE_EOF_OR_DEADLINE';error=None
    try:
        while time.time()<endpoint.registry['deadline_epoch'] and metrics.operations<40000:
            if not select.select([sys.stdin],[],[],min(1,max(0,endpoint.registry['deadline_epoch']-time.time())))[0]:continue
            line=sys.stdin.buffer.readline(MAX_LINE_BYTES+1)
            if not line:break
            if len(line)>MAX_LINE_BYTES or not line.endswith(b'\n'):raise ValueError('PV1_BOUNDED_PRODUCTION_JSON_LINE')
            packet=json.loads(line)
            if endpoint.mode!=BASE:
                endpoint.guard(packet['args']);intent=folder/'FAULT_INTENT.json'
                if intent.exists():raise PermissionError('PV1_SINGLE_ISOLATED_FAULT_REQUEST_NO_RETRY')
                import hashlib
                write(intent,dict(job_id=endpoint.registry['job_id'],fixture_uuid=endpoint.registry['fixture_uuid'],
                    mode=endpoint.mode,rpc_id=packet['rpc_id'],request_sha256=hashlib.sha256(line).hexdigest(),
                    started_epoch=time.time(),physical_installs=0,new_model_queries=0))
                if endpoint.mode==BASE+'-FAULT-EXIT':os._exit(72)
                time.sleep(20.25)
                result=execute_request(packet,lambda args:dict(isolated_late_fixture_reply=True,new_model_queries=0,new_physics=0),endpoint.guard,seen)
            else:result=execute_request(packet,endpoint.invoke,endpoint.guard,seen)
            metrics.reply(result,lambda v:print(json.dumps(v,allow_nan=False),flush=True))
    except BaseException as exc:status='FIXTURE_FAILED_PRESERVED_NO_RETRY';error=type(exc).__name__+': '+str(exc);raise
    finally:
        cpu=time.process_time()-started;settled=metrics.settle(endpoint.ledger,cpu)
        write(folder/('STDIO_TERMINAL_'+role+'.json'),dict(status=status,error=error,CPU_s=cpu,metrics_settlement=settled,
            operations=metrics.operations,heartbeat_written=False,new_model_queries=0,new_physics=0,unknown_effects_resent=0))
