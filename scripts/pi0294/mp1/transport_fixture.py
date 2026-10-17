"""Recorded dual-camera transport only, isolated from neural/physical RPC paths."""
import base64
import json
import os
from pathlib import Path
import re
import resource
import select
import sys
import time
import uuid

from scripts.pi0292.durable import OwnerLock,read,write
from scripts.pi0292.relay_protocol import PROTOCOL,digest,sealed,unseal,validate_observation
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.relay_endpoint import inside
from .n1_limits import RELEASE,allocation_number,resolve

PLAN='plans/N1_TRANSPORT_FIXTURE.json'
BASE='N1-PREMODEL'
CYCLES=80


def prepared_plan(root,manifest):
    root=Path(root);path=inside(root,PLAN);plan=immutable(path)
    if (manifest['execution']['sealed_inputs'].get(PLAN)!=sha(path)
            or plan['schema']!='MP1_N1_RECORDED_TRANSPORT_FIXTURE_V1' or plan['run_uuid']!=root.name
            or plan['scope']!='RECORDED_OBSERVATION_NO_MODEL_NO_PHYSICS' or plan['cycles']!=CYCLES
            or not re.fullmatch(r'episodes/[A-Za-z0-9_-]+/relay/observations/[0-9]{3}\.json',plan['recorded_observation_relative'])
            or any(not re.fullmatch('[0-9a-f]{64}',plan[key]) for key in [
                'recorded_observation_file_sha256','recorded_observation_packet_sha256','recorded_payload_sha256'])
            or len(plan['RGB_png_bytes'])!=2 or any(type(n) is not int or n<=0 for n in plan['RGB_png_bytes'])
            or not 0<plan['RGB_payload_bytes']<2*1024**2
            or plan['RGB_payload_bytes']!=sum(4*((n+2)//3) for n in plan['RGB_png_bytes'])):
        raise PermissionError('SEALED_REAL_SIZED_RECORDED_TRANSPORT_PLAN_BEFORE_QSUB')
    return plan


def namespace(root,label):
    base,number=allocation_number(label,12)
    if base not in [BASE,BASE+'-FAULT-LATE',BASE+'-FAULT-EXIT']:
        raise PermissionError('ISOLATED_N1_TRANSPORT_LABEL_ONLY')
    return inside(Path(root),'engineering/n1-transport/'+label),number,base


def immutable(path):
    if path.is_symlink() or path.stat().st_mode&0o222:
        raise PermissionError('IMMUTABLE_FIXTURE_BINDING_REQUIRED')
    return read(path)


def packet_identity(registry,sequence):
    if type(sequence) is not int or not 0<=sequence<CYCLES:
        raise ValueError('FIXED_EIGHTY_FIXTURE_CYCLES')
    return str(uuid.uuid5(uuid.UUID(registry['fixture_uuid']),str(sequence)))


def check_observation(packet,plan):
    if packet['sha256']!=plan['recorded_observation_packet_sha256']:
        raise ValueError('ORIGINAL_RECORDED_OBSERVATION_NOT_REENCODED_OR_PADDED')
    body=unseal(packet)
    if (set(body)!={'protocol','kind','run_uuid','episode_uuid','sequence','tick','payload'}
            or body['protocol']!=PROTOCOL or body['kind']!='observation' or body['run_uuid']!=plan['run_uuid']
            or type(body['sequence']) is not int or not 0<=body['sequence']<80 or body['tick']!=body['sequence']*200):
        raise ValueError('OWN_REGISTERED_RECORDED_OBSERVATION_ONLY')
    validate_observation(body['payload'])
    sizes=[len(base64.b64decode(v['png_base64'],validate=True)) for v in body['payload']['images']]
    if (sizes!=plan['RGB_png_bytes'] or sum(len(v['png_base64']) for v in body['payload']['images'])!=plan['RGB_payload_bytes']
            or digest(body['payload'])!=plan['recorded_payload_sha256']):
        raise ValueError('EXACT_ACTUAL_PRODUCTION_RGB_PAYLOAD_SIZE_AND_HASH')
    return body


def check_query(packet,registry,plan,now):
    q=unseal(packet)
    expected={'kind','run_uuid','fixture_uuid','sequence','query_uuid','recorded_observation','created_epoch','expires_epoch'}
    if (set(q)!=expected or q['kind']!='fixture_checksum_request_no_action'
            or q['run_uuid']!=registry['run_uuid'] or q['fixture_uuid']!=registry['fixture_uuid']
            or q['query_uuid']!=packet_identity(registry,q['sequence'])
            or not q['created_epoch']<=now<q['expires_epoch']<=q['created_epoch']+60):
        raise PermissionError('EXACT_FINITE_FIXTURE_QUERY_NOT_POLICY_ACTION')
    check_observation(q['recorded_observation'],plan)
    return q


def response_for(packet,registry,plan,now):
    q=check_query(packet,registry,plan,now)
    return sealed(dict(kind='fixture_checksum_response_no_action',run_uuid=q['run_uuid'],
        fixture_uuid=q['fixture_uuid'],sequence=q['sequence'],query_uuid=q['query_uuid'],
        request_sha256=packet['sha256'],recorded_observation_sha256=q['recorded_observation']['sha256'],
        payload_sha256=plan['recorded_payload_sha256'],RGB_payload_bytes=plan['RGB_payload_bytes'],
        computed_epoch=now,generated_actions=0,physical_integrations=0))


def check_response(packet,query,registry,plan):
    q=unseal(query);body=unseal(packet)
    expected=response_for(query,registry,plan,body['computed_epoch'])
    if packet!=expected:raise PermissionError('EXACT_COMPUTE_FIXTURE_RESPONSE_NO_ACTION_CHANNEL')
    return body


class Endpoint:
    def __init__(self,root,role,label,*,now=time.time):
        from .science import validate_runtime,Ledger
        self.root,self.role,self.label,self.now=Path(root),role,label,now
        if role not in ['core','hopper']:raise PermissionError('EXACT_FIXTURE_ROLE')
        self.folder,number,self.mode=namespace(root,label)
        self.manifest=validate_runtime(root,'4080-simulation' if role=='core' else 'hopper-neural')
        n1=resolve(root,self.manifest)
        if n1 is None:raise PermissionError('EXPLICIT_SEALED_N1_ONLY')
        self.plan=prepared_plan(root,self.manifest);self.registry_path=inside(self.root,'plans/'+label+'/REGISTRY.json')
        self.registry=immutable(self.registry_path);self.registry_sha=sha(self.registry_path)
        self.manifest_sha=sha(self.root/'RUN_MANIFEST.json');self.ledger=Ledger(root,role)
        p=self.plan;r=self.registry
        from .n1_boundary import target
        if (self.manifest['execution']['sealed_inputs'].get(PLAN)!=sha(self.root/PLAN)
                or target(self.manifest['execution']['cold_engineering_binding_relative'])!=number
                or p['schema']!='MP1_N1_RECORDED_TRANSPORT_FIXTURE_V1' or p['run_uuid']!=self.root.name
                or p['scope']!='RECORDED_OBSERVATION_NO_MODEL_NO_PHYSICS' or p['cycles']!=CYCLES
                or r['schema']!='MP1_N1_TRANSPORT_REGISTRY_V1' or r['run_uuid']!=self.root.name
                or r['label']!=label or r['authority_release']!=RELEASE or r['plan_sha256']!=sha(self.root/PLAN)
                or r[role+'_manifest_sha256']!=self.manifest_sha
                or r[role+'_source_sha']!=self.manifest['code']['vla_commit']
                or not self.now()<r['deadline_epoch']<=min(r['allocation_end_epoch']-2700,n1['new_experiment_cutoff_epoch'])):
            raise PermissionError('EXACT_FRESH_SAME_JOB_FIXTURE_BINDINGS')
        uuid.UUID(r['fixture_uuid'])
        if not re.fullmatch(r'\d+\.hopper-m-\d+',r['job_id']):raise PermissionError('NATIVE_FIXTURE_JOB_ID')
        self.folder.mkdir(parents=True,exist_ok=True,mode=0o700)
        for name in ['observations','inbox','outbox','installs','completed']:
            path=inside(self.folder,name)
            if path.is_symlink():raise PermissionError('NO_FIXTURE_SUBDIRECTORY_SYMLINK')
            path.mkdir(exist_ok=True,mode=0o700)
        self.recorded=None
        if role=='core':
            reference=inside(self.root,p['recorded_observation_relative'])
            if not p['recorded_observation_relative'].startswith('episodes/') or sha(reference)!=p['recorded_observation_file_sha256']:
                raise PermissionError('EXACT_OWNED_CLOSED_RECORDED_OBSERVATION_FILE')
            self.recorded=read(reference);body=check_observation(self.recorded,p)
            episode=reference.parent.parent.parent
            adjudication=read(episode/'MP1_ADJUDICATION.json')
            if adjudication['status'] not in ['SUCCESS','VALID_TASK_FAIL'] or adjudication['live_native_failures']:
                raise PermissionError('FIXTURE_REFERENCE_MUST_BE_A_CLOSED_SAFE_REAL_RECORD')
            for image,camera in zip(body['payload']['images'],['body_rgb','wrist_rgb']):
                if sha(inside(episode,camera+'/'+str(body['tick']//25).zfill(4)+'.png'))!=image['sha256']:
                    raise PermissionError('RECORDED_FIXTURE_PIXELS_MUST_EQUAL_ACTUAL_SAVED_RENDER_FRAMES')

    def guard(self,args):
        self.ledger.guard()
        if (self.now()>=self.registry['deadline_epoch'] or sha(self.root/'RUN_MANIFEST.json')!=self.manifest_sha
                or sha(self.registry_path)!=self.registry_sha):
            raise PermissionError('FIXTURE_IDENTITY_OR_DEADLINE_CHANGED')
        if (set(args)-{'operation','sequence','query','bundle'} or args['operation'] not in ['observation','query','response','install','ack']):
            raise PermissionError('NO_EPISODE_TEACHER_ACTION_OR_PHYSICS_FIELDS_IN_FIXTURE_RPC')
        expected={'operation','sequence'}
        if args['operation']=='query':expected.add('query')
        if args['operation']=='install' or self.role=='hopper' and args['operation']=='ack':expected.add('bundle')
        if set(args)!=expected:raise PermissionError('EXACT_TYPED_FIXTURE_RPC_FIELDS')
        packet_identity(self.registry,args['sequence'])

    def path(self,kind,sequence):return inside(self.folder,kind+'/'+str(sequence).zfill(3)+'.json')

    def invoke(self,args):
        self.guard(args);operation=args['operation'];sequence=args['sequence']
        lock=OwnerLock(self.folder/('ENDPOINT-'+self.role+'.lock'))
        try:
            if self.role=='core':
                if operation=='observation':
                    path=self.path('observations',sequence)
                    if not path.exists():
                        now=self.now();write(path,sealed(dict(kind='fixture_checksum_request_no_action',run_uuid=self.root.name,
                            fixture_uuid=self.registry['fixture_uuid'],sequence=sequence,query_uuid=packet_identity(self.registry,sequence),
                            recorded_observation=self.recorded,created_epoch=now,expires_epoch=min(now+60,self.registry['deadline_epoch']))))
                    return read(path)
                if operation=='ack':
                    p=self.path('installs',sequence);return read(p) if p.exists() else None
                if operation!='install' or set(args)!={'operation','sequence','bundle'}:
                    raise PermissionError('ONLY_ISOLATED_CORE_FIXTURE_INSTALL')
                query=read(self.path('observations',sequence));check_query(query,self.registry,self.plan,self.now())
                body=check_response(args['bundle'],query,self.registry,self.plan)
                path=self.path('installs',sequence)
                if path.exists():raise FileExistsError('NEVER_DUPLICATE_FIXTURE_INSTALL')
                ack=sealed(dict(kind='fixture_install_ack_no_physics',run_uuid=self.root.name,
                    fixture_uuid=self.registry['fixture_uuid'],sequence=sequence,query_uuid=body['query_uuid'],
                    response_sha256=args['bundle']['sha256'],request_sha256=query['sha256'],
                    installed_epoch=self.now(),fixture_install_count=1,physical_installs=0,unknown_effects_resent=0))
                write(path,ack);return ack
            if operation=='response':
                path=self.path('outbox',sequence);return read(path) if path.exists() else None
            if operation=='query':
                if set(args)!={'operation','sequence','query'}:raise PermissionError('EXACT_FIXTURE_QUERY_FIELDS')
                query=args['query'];q=check_query(query,self.registry,self.plan,self.now())
                if q['sequence']!=sequence:raise ValueError('FIXTURE_QUERY_SEQUENCE_MISMATCH')
                path=self.path('inbox',sequence)
                if path.exists():raise FileExistsError('NEVER_REQUERY_FIXTURE_UNKNOWN_OR_KNOWN_EFFECT')
                write(path,query);return dict(status='FIXTURE_QUERY_RESERVED_NO_MODEL',request_sha256=query['sha256'])
            if operation=='ack':
                if set(args)!= {'operation','sequence','bundle'}:raise PermissionError('EXACT_FIXTURE_COMPLETION_FIELDS')
                path=self.path('completed',sequence)
                if path.exists():raise FileExistsError('ONE_FIXTURE_CYCLE_COMPLETION')
                write(path,args['bundle']);return dict(status='FIXTURE_COMPLETION_RECORDED')
            raise PermissionError('NO_POLICY_OPERATIONS_ON_FIXTURE_HOPPER_ENDPOINT')
        finally:lock.close()


class Pump:
    """Runs on the admitted compute node before any model load, no CUDA work."""
    def __init__(self,root,label,allocation,guard):
        self.root,self.label,self.allocation,self.guard=Path(root),label,allocation,guard
        self.folder,number,mode=namespace(root,label)
        if mode!=BASE:raise PermissionError('ONLY_NOMINAL_NATIVE_FIXTURE_PUMP')
        self.registry=immutable(self.root/'plans'/label/'REGISTRY.json');self.plan=immutable(self.root/PLAN)
        from .native_heartbeat_capture import native_identity
        native_identity(self.root,self.root/'pbs'/('alloc-'+str(number).zfill(2)),allocation,read(self.root/'RUN_MANIFEST.json'))
        if self.registry['job_id']!=allocation['job_id'] or self.registry['label']!=label:
            raise PermissionError('SAME_NATIVE_ALLOCATION_FIXTURE_PUMP')
        self.folder.mkdir(parents=True,exist_ok=True,mode=0o700)
        self.process=OwnerLock(self.folder/'COMPUTE.lock');self.next=0
        if (self.folder/'COMPUTE_STARTED.json').exists():
            self.process.close();raise FileExistsError('PRESERVE_PRIOR_COMPUTE_FIXTURE_NO_RESTART')
        for name in ['inbox','outbox','compute-intents','completed']:
            path=inside(self.folder,name)
            if path.is_symlink():raise PermissionError('NO_COMPUTE_FIXTURE_SUBDIRECTORY_SYMLINK')
            path.mkdir(exist_ok=True,mode=0o700)
        write(self.folder/'COMPUTE_STARTED.json',dict(job_id=allocation['job_id'],hostname=allocation['hostname'],
            pid=os.getpid(),proc_start=Path('/proc/self/stat').read_text().split()[21],models_loaded=0,new_queries=0,new_physics=0))

    def poll(self):
        if self.next>=CYCLES:return
        path=inside(self.folder,'inbox/'+str(self.next).zfill(3)+'.json')
        if not path.exists():return
        self.guard();query=read(path);q=check_query(query,self.registry,self.plan,time.time())
        if q['sequence']!=self.next:raise ValueError('CONTIGUOUS_EIGHTY_NATIVE_FIXTURE_QUERIES')
        intent=inside(self.folder,'compute-intents/'+path.name);response=inside(self.folder,'outbox/'+path.name)
        if intent.exists() or response.exists():raise PermissionError('UNKNOWN_FIXTURE_COMPUTE_EFFECT_NO_REPLAY')
        write(intent,dict(request_sha256=query['sha256'],sequence=self.next,started_epoch=time.time()))
        write(response,response_for(query,self.registry,self.plan,time.time()));self.next+=1

    def completed(self):
        rows=[]
        for sequence in range(CYCLES):
            name=str(sequence).zfill(3)+'.json';p=inside(self.folder,'completed/'+name)
            if not p.exists():return None
            value=read(p);query=read(self.folder/'inbox'/name);response=read(self.folder/'outbox'/name)
            ack=unseal(value['ack']);row=value['cycle']
            if (ack['kind']!='fixture_install_ack_no_physics' or ack['run_uuid']!=self.root.name
                    or ack['fixture_uuid']!=self.registry['fixture_uuid'] or ack['sequence']!=sequence
                    or ack['query_uuid']!=packet_identity(self.registry,sequence)
                    or ack['request_sha256']!=query['sha256'] or ack['response_sha256']!=response['sha256']
                    or ack['fixture_install_count']!=1 or ack['physical_installs'] or ack['unknown_effects_resent']
                    or row['sequence']!=sequence or row['query_uuid']!=ack['query_uuid']
                    or row['query_packet_sha256']!=query['sha256'] or row['response_packet_sha256']!=response['sha256']
                    or row['core_ACK_sha256']!=value['ack']['sha256']
                    or row['RGB_payload_bytes']!=self.plan['RGB_payload_bytes']
                    or row['fixture_install_count']!=1 or row['ACK_matches_request'] is not True
                    or not 0<row['maximum_RPC_s']<=20):
                raise PermissionError('EXACT_NATIVE_FIXTURE_RESPONSE_CORE_ACK_AND_WIRE_CYCLE_JOIN')
            rows.append(row)
        return rows

    def close(self):self.process.close()


def serve(root,role,label):
    """Same production stdio envelope, no policy endpoint or heartbeat writer."""
    from scripts.pi0294.relay_file_rpc import execute_request,MAX_LINE_BYTES
    from scripts.pi0294.rpc_metrics import SessionMetrics
    os.environ.pop('E1_RUN_MANIFEST',None);os.environ.pop('MP1_RUN_MANIFEST',None)
    endpoint=Endpoint(root,role,label);folder=endpoint.folder;record=Path(root)/'aux_worker_processes'/('fixture-rpc-'+role+'-'+label+'.json')
    if record.exists():raise FileExistsError('ONE_FINITE_FIXTURE_STDIO_PROCESS_NO_RESTART')
    record.parent.mkdir(exist_ok=True,mode=0o700);started=time.process_time()
    code=Path(root)/endpoint.manifest['code']['capsule_relative'];copy=code.parent/('STDIO_FIXTURE_MANIFEST_'+role+'_'+label+'.json')
    if copy.exists():raise FileExistsError('PRESERVE_PRIOR_FIXTURE_STDIO_MANIFEST_NO_RESTART')
    copy.write_bytes((Path(root)/'RUN_MANIFEST.json').read_bytes());copy.chmod(0o400)
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});resource.setrlimit(resource.RLIMIT_CPU,(180,180));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    write(record,dict(pid=os.getpid(),proc_start=Path('/proc/self/stat').read_text().split()[21],source_sha=endpoint.manifest['code']['vla_commit'],
        manifest_relative=str(copy.relative_to(root)),manifest_sha256=sha(copy),source_live_worker=False,
        work_type='ISOLATED_TRANSPORT_FIXTURE_NO_MODEL_NO_PHYSICS',maximum_CPU_s=180,deadline_epoch=endpoint.registry['deadline_epoch'],job_id=endpoint.registry['job_id']))
    seen=set();metrics=SessionMetrics();status='FIXTURE_EOF_OR_DEADLINE';error=None
    try:
        while time.time()<endpoint.registry['deadline_epoch'] and metrics.operations<40000:
            if not select.select([sys.stdin],[],[],min(1,max(0,endpoint.registry['deadline_epoch']-time.time())))[0]:continue
            line=sys.stdin.buffer.readline(MAX_LINE_BYTES+1)
            if not line:break
            if len(line)>MAX_LINE_BYTES or not line.endswith(b'\n'):raise ValueError('BOUNDED_PRODUCTION_FIXTURE_JSON_LINE')
            packet=json.loads(line)
            if endpoint.mode!=BASE:
                endpoint.guard(packet['args']);intent=folder/'FAULT_INTENT.json'
                if intent.exists():raise PermissionError('SINGLE_ISOLATED_FAULT_REQUEST_NO_RETRY')
                import hashlib
                write(intent,dict(job_id=endpoint.registry['job_id'],fixture_uuid=endpoint.registry['fixture_uuid'],
                    mode=endpoint.mode,rpc_id=packet['rpc_id'],request_sha256=hashlib.sha256(line).hexdigest(),
                    started_epoch=time.time(),physical_installs=0,new_model_queries=0))
                if endpoint.mode==BASE+'-FAULT-EXIT':os._exit(72)
                time.sleep(20.25)
                result=execute_request(packet,lambda args:dict(isolated_late_fixture_reply=True,new_model_queries=0,new_physics=0),endpoint.guard,seen)
            else:result=execute_request(packet,endpoint.invoke,endpoint.guard,seen)
            metrics.reply(result,lambda value:print(json.dumps(value,allow_nan=False),flush=True))
    except BaseException as exc:status='FIXTURE_FAILED_PRESERVED_NO_RETRY';error=type(exc).__name__+': '+str(exc);raise
    finally:
        cpu=time.process_time()-started;settlement=metrics.settle(endpoint.ledger,cpu)
        target=folder/('STDIO_TERMINAL_'+role+'.json')
        write(target,dict(status=status,error=error,CPU_s=cpu,metrics_settlement=settlement,operations=metrics.operations,
            heartbeat_written=False,new_model_queries=0,new_physics=0,unknown_effects_resent=0))
