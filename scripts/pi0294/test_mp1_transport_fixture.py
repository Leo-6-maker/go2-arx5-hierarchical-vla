import base64
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import uuid

from PIL import Image
from scripts.pi0292.durable import read
from scripts.pi0292.eval_contract import FRAME,TCP
from scripts.pi0292.relay_protocol import PROTOCOL,digest,sealed
from .mp1 import transport_fixture as fixture
from .mp1 import science,native_heartbeat_capture
from .mp1.transport_fixture_router import route,verify_cycle
from .mp1.n1_limits import RELEASE
from .native_dataset import sha
from .relay_file_rpc import execute_request


def put(path,value,immutable=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():path.chmod(0o600)
    path.write_text(json.dumps(value)+'\n')
    if immutable:path.chmod(0o400)


def reference(root,now):
    root=Path(root);png=io.BytesIO();Image.new('RGB',(256,256),(32,64,128)).save(png,format='PNG');raw=png.getvalue()
    images=[dict(camera=camera,png_base64=base64.b64encode(raw).decode(),sha256=hashlib.sha256(raw).hexdigest())
        for camera in ['external_rgb','wrist_rgb']]
    payload=dict(snapshot_uuid=str(uuid.uuid4()),language='CPU fixture, not native scientific data',
        state_eef10=[0.,0.,0.,1.,0.,0.,0.,1.,0.,.5],images=images,frame=FRAME,tcp=TCP)
    packet=sealed(dict(protocol=PROTOCOL,kind='observation',run_uuid=root.name,episode_uuid=str(uuid.uuid4()),sequence=0,tick=0,payload=payload))
    relative='episodes/fixture-closed/relay/observations/000.json';put(root/relative,packet)
    episode=root/'episodes/fixture-closed';put(episode/'MP1_ADJUDICATION.json',dict(status='SUCCESS',live_native_failures=0))
    for camera in ['body_rgb','wrist_rgb']:
        p=episode/camera/'0000.png';p.parent.mkdir(exist_ok=True);p.write_bytes(raw)
    plan=dict(schema='MP1_N1_RECORDED_TRANSPORT_FIXTURE_V1',run_uuid=root.name,cycles=80,
        scope='RECORDED_OBSERVATION_NO_MODEL_NO_PHYSICS',recorded_observation_relative=relative,
        recorded_observation_file_sha256=sha(root/relative),recorded_observation_packet_sha256=packet['sha256'],
        recorded_payload_sha256=digest(payload),RGB_png_bytes=[len(raw),len(raw)],
        RGB_payload_bytes=sum(len(image['png_base64']) for image in images))
    put(root/fixture.PLAN,plan,True)
    path=root/'RUN_MANIFEST.json';manifest=read(path) if path.exists() else dict(code=dict(vla_commit='a'*40))
    manifest['execution']=dict(sealed_inputs={fixture.PLAN:sha(root/fixture.PLAN)},cold_engineering_binding_relative='plans/N1/alloc-06/COLD_BOUNDARY.json')
    put(path,manifest)
    label='N1-PREMODEL-ALLOCATION-06'
    registry=dict(schema='MP1_N1_TRANSPORT_REGISTRY_V1',run_uuid=root.name,label=label,
        authority_release=RELEASE,fixture_uuid=str(uuid.uuid4()),job_id='123.hopper-m-02',
        plan_sha256=sha(root/fixture.PLAN),allocation_end_epoch=now+10800,deadline_epoch=now+1000,
        core_manifest_sha256=sha(path),hopper_manifest_sha256=sha(path),core_source_sha='a'*40,hopper_source_sha='a'*40)
    put(root/'plans'/label/'REGISTRY.json',registry,True)
    return plan,registry,packet


def wire(role,args,value,rpc_id=None):
    rpc_id=rpc_id or str(uuid.uuid4());packet=dict(rpc_id=rpc_id,ok=True,value=value)
    request=(json.dumps(dict(rpc_id=rpc_id,args=args),allow_nan=False)+'\n').encode();response=json.dumps(packet,allow_nan=False).encode()
    metrics=dict(role=role,operation=args['operation'],rpc_id=rpc_id,request_sha256=hashlib.sha256(request).hexdigest(),
        response_sha256=hashlib.sha256(response).hexdigest(),request_bytes=len(request),response_bytes=len(response),
        wall_s=.001,effect_retried=False,call_watchdog_s=20)
    return dict(packet=packet,wire=metrics)


def sealed_transport_artifacts(root,directory,cycles,faults):
    """Synthetic raw evidence for verifier tests, never a native measurement."""
    from .mp1.native_transport_qualification import BINDING,OWNER
    root=Path(root);plan,registry,packet=reference(root,1000)
    folder,_,_=fixture.namespace(root,registry['label']);members={};rows=[];acks=[]
    def member(path,value):
        put(path,value,True);members[str(path.relative_to(root))]=sha(path)
    for sequence in range(80):
        query=sealed(dict(kind='fixture_checksum_request_no_action',run_uuid=root.name,
            fixture_uuid=registry['fixture_uuid'],sequence=sequence,query_uuid=fixture.packet_identity(registry,sequence),
            recorded_observation=packet,created_epoch=1000,expires_epoch=1060))
        response=fixture.response_for(query,registry,plan,1001)
        ack=sealed(dict(kind='fixture_install_ack_no_physics',run_uuid=root.name,fixture_uuid=registry['fixture_uuid'],
            sequence=sequence,query_uuid=query['body']['query_uuid'],request_sha256=query['sha256'],response_sha256=response['sha256'],
            installed_epoch=1002,fixture_install_count=1,physical_installs=0,unknown_effects_resent=0))
        calls=[wire('core',dict(operation='observation',sequence=sequence),query)['wire'],
            wire('hopper',dict(operation='query',sequence=sequence,query=query),dict(status='FIXTURE_QUERY_RESERVED_NO_MODEL',request_sha256=query['sha256']))['wire'],
            wire('hopper',dict(operation='response',sequence=sequence),response)['wire'],
            wire('core',dict(operation='install',sequence=sequence,bundle=response),ack)['wire'],
            wire('core',dict(operation='ack',sequence=sequence),ack)['wire']]
        row=deepcopy(cycles[sequence]) if sequence<len(cycles) else dict(fixture_install_count=1,ACK_matches_request=True,maximum_RPC_s=.001)
        row.update(sequence=sequence,query_uuid=query['body']['query_uuid'],RGB_payload_bytes=plan['RGB_payload_bytes'],
            request_sha256=calls[1]['request_sha256'],response_sha256=calls[2]['response_sha256'],wire_calls=calls,
            query_packet_sha256=query['sha256'],response_packet_sha256=response['sha256'],core_ACK_sha256=ack['sha256'],
            physical_installs=0,unknown_effects_resent=0)
        name=str(sequence).zfill(3)+'.json'
        member(folder/'inbox'/name,query);member(folder/'outbox'/name,response)
        member(folder/'completed'/name,dict(cycle=row,ack=ack))
        member(folder/'compute-intents'/name,dict(request_sha256=query['sha256'],sequence=sequence,started_epoch=1000.5))
        rows.append(row);acks.append(ack)
    actual_faults=deepcopy(faults)
    expired=dict(run_uuid=root.name,observed_epoch=1000)
    member(folder/'faults/actual_expired_heartbeat/OWNER_HEARTBEAT.json',expired)
    member(folder/'faults/actual_expired_heartbeat/WRITER_TERMINAL.json',dict(returncode=0,observed_epoch=1000.1,
        heartbeat_sha256=sha(folder/'faults/actual_expired_heartbeat/OWNER_HEARTBEAT.json')))
    future=dict(run_uuid=root.name,observed_epoch=1730)
    member(folder/'faults/actual_future_heartbeat/OWNER_HEARTBEAT.json',future)
    member(folder/'faults/wrong_run_heartbeat/OWNER_HEARTBEAT.json',dict(run_uuid='other',observed_epoch=1700))
    replace_rows=[dict(heartbeat=dict(run_uuid=root.name,observed_epoch=1700+i*.01),heartbeat_written=False,
        clock_sampled_after_read=True,server_revalidation=False,cached_clock_epoch=1700+i*.01+.001) for i in range(40)]
    member(folder/'faults/concurrent_replace/READS.json',dict(rows=replace_rows,writes=[1700+i*.01 for i in range(40)]))
    member(folder/'faults/concurrent_replace/OWNER_HEARTBEAT.json',replace_rows[-1]['heartbeat'])
    member(folder/'faults/metadata_read_failure/RESOURCE.json',dict(fixture=True))
    member(folder/'faults/statistics_lock/COUNTERS.json',dict(RPC_calls=1))
    actual_faults['actual_expired_heartbeat'].update(diagnostics=dict(heartbeat=expired,revalidated_clock_epoch=1700))
    actual_faults['actual_future_heartbeat'].update(diagnostics=dict(heartbeat=future,revalidated_clock_epoch=1700))
    actual_faults['metadata_read_failure'].update(attempts=2,injected_errno=116,wall_s=.01,recovered_only_declared_idempotent_read=True)
    actual_faults['statistics_lock'].update(actual_lock_holder_exit=0,reply_while_lock_held=True,reply_wall_s=.0001)
    for name,value in actual_faults.items():
        if name in ['late_response','process_exit']:
            value.update(native_process_terminal_verified=True,actual_fault_requests=1,second_call_blocked_before_send=True,
                job_id=registry['job_id'],source_sha='a'*40,actual_server_exit=0 if name=='late_response' else 72,
                label=registry['label'].replace(fixture.BASE,fixture.BASE+('-FAULT-LATE' if name=='late_response' else '-FAULT-EXIT'),1))
        else:member(folder/'faults'/name/'RESULT.json',value)
    owner=dict(schema='MP1_N1_OWNER_TRANSPORT_FAULTS_V1',run_uuid=root.name,label=registry['label'],job_id=registry['job_id'],
        registry_sha256=sha(root/'plans'/registry['label']/'REGISTRY.json'),nominal_cycles=80,
        nominal_cycles_sha256=digest(rows[:len(cycles)]),
        nominal_terminals=dict(core=dict(native_process_terminal_verified=True,actual_install_count=80,actual_install_ACKs_sha256=digest(acks)),
            hopper=dict(native_process_terminal_verified=True)),faults={name:actual_faults[name] for name in ['late_response','process_exit']})
    member(folder/OWNER,owner)
    binding=folder/BINDING;put(binding,dict(schema='MP1_N1_TRANSPORT_RAW_BINDING_V1',run_uuid=root.name,label=registry['label'],
        job_id=registry['job_id'],source_sha='a'*40,plan_sha256=sha(root/fixture.PLAN),registry_sha256=owner['registry_sha256'],members=members),True)
    return dict(codec='PRODUCTION_JSON_LINE_STDIO_AND_SEALED_OBSERVATION_V1',scope='ISOLATED_FIXTURE_NO_MODEL_NO_PHYSICS',
        watchdogs_s=[20,60,900],duplicate_fixture_installs=0,physical_installs=0,unknown_effects_resent=0,
        cycles=rows[:len(cycles)],reference_RGB_payload_bytes=plan['RGB_payload_bytes'],reference_payload_sha256=plan['recorded_payload_sha256'],
        raw_binding_relative=str(binding.relative_to(root)),raw_binding_sha256=sha(binding)),actual_faults


class FixtureLedger:
    def __init__(self,*args):pass
    def guard(self):pass


class TransportFixtureTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)/'source-pair-01';self.root.mkdir();now=time.time()
        self.plan,self.registry,self.packet=reference(self.root,now);self.label=self.registry['label']
        self.patchers=[patch.object(science,'validate_runtime',side_effect=lambda *a,**kw:read(self.root/'RUN_MANIFEST.json')),
            patch.object(science,'Ledger',FixtureLedger),patch.object(fixture,'resolve',return_value=dict(new_experiment_cutoff_epoch=now+7200)),
            patch.object(native_heartbeat_capture,'native_identity',return_value={})]
        for p in self.patchers:p.start();self.addCleanup(p.stop)
        self.core=fixture.Endpoint(self.root,'core',self.label);self.hopper=fixture.Endpoint(self.root,'hopper',self.label)
        self.pump=fixture.Pump(self.root,self.label,dict(job_id=self.registry['job_id'],hostname='CPU_FIXTURE_NOT_NATIVE'),lambda:None)
        self.addCleanup(self.pump.close);self.seen={role:set() for role in ['core','hopper']}

    def call(self,role,args):
        endpoint=self.core if role=='core' else self.hopper;rpc_id=str(uuid.uuid4())
        result=execute_request(dict(rpc_id=rpc_id,args=args),endpoint.invoke,endpoint.guard,self.seen[role])
        if role=='hopper' and args['operation']=='query':self.pump.poll()
        return wire(role,args,result['value'],rpc_id)

    def test_eighty_full_real_codec_cycles_have_eighty_distinct_once_only_fixture_acks(self):
        original=sha(self.root/self.plan['recorded_observation_relative']);rows=[]
        result=route(self.call,self.plan,self.registry,rows.append)
        self.assertEqual(rows,result);self.assertEqual(len(rows),80);self.assertEqual(len({r['query_uuid'] for r in rows}),80)
        self.assertEqual(self.pump.next,80);self.assertEqual(self.pump.completed(),rows)
        self.assertEqual(len(list((self.core.folder/'installs').glob('*.json'))),80)
        self.assertEqual(sha(self.root/self.plan['recorded_observation_relative']),original)
        self.assertFalse((self.root/'OWNER_HEARTBEAT.json').exists());self.assertEqual(len(list((self.root/'episodes').iterdir())),1)
        self.assertTrue(all(row['physical_installs']==0 for row in rows))
        self.assertLess(self.plan['RGB_payload_bytes'],65536)

    def test_eighty_cycles_with_ahead_core_clock_keep_original_receiver_and_ttl_checks(self):
        self.core.now=lambda:time.time()+.068
        rows=route(self.call,self.plan,self.registry,lambda row:None)
        self.assertEqual(len(rows),80);self.assertEqual(self.pump.next,80)
        self.assertEqual(self.pump.completed(),rows)
        self.assertEqual(len(list((self.core.folder/'installs').glob('*.json'))),80)
        for path in (self.core.folder/'observations').glob('*.json'):
            body=read(path)['body'];self.assertEqual(body['expires_epoch'],body['created_epoch']+60)

    def test_duplicate_query_install_and_completion_never_repeat_effect(self):
        query=self.core.invoke(dict(operation='observation',sequence=0));self.hopper.invoke(dict(operation='query',sequence=0,query=query));self.pump.poll()
        response=self.hopper.invoke(dict(operation='response',sequence=0));self.core.invoke(dict(operation='install',sequence=0,bundle=response))
        for endpoint,args in [(self.hopper,dict(operation='query',sequence=0,query=query)),
                              (self.core,dict(operation='install',sequence=0,bundle=response))]:
            with self.subTest(operation=args['operation']),self.assertRaises(FileExistsError):endpoint.invoke(args)
        self.assertEqual(len(list((self.core.folder/'installs').glob('*.json'))),1)

    def test_recorded_png_hash_or_real_payload_size_cannot_be_padded(self):
        for key,value in [('RGB_png_bytes',[1,2]),('RGB_payload_bytes',self.plan['RGB_payload_bytes']+1)]:
            bad=dict(self.plan,**{key:value})
            with self.subTest(key=key),self.assertRaises(ValueError):fixture.check_observation(self.packet,bad)
        bad=deepcopy(self.packet);bad['body']['payload']['images'][0]['png_base64']+='A'
        with self.assertRaises(ValueError):fixture.check_observation(bad,self.plan)

    def test_wrong_run_expiry_and_unregistered_action_fields_fail_before_install(self):
        packet=self.core.invoke(dict(operation='observation',sequence=0))
        for key,value in [('run_uuid','other'),('expires_epoch',0),('kind','install_prefix8')]:
            q=deepcopy(packet['body']);q[key]=value
            with self.subTest(key=key),self.assertRaises(PermissionError):self.hopper.invoke(dict(operation='query',sequence=0,query=sealed(q)))
        for args in [dict(operation='install',sequence=0,bundle={},teacher={}),dict(operation='observation',sequence=0,episode='policy')]:
            with self.assertRaises(PermissionError):self.core.invoke(args)
        self.assertEqual(len(list((self.core.folder/'installs').glob('*.json'))),0)

    def test_receiver_waits_for_causal_time_without_resealing_or_extending_ttl(self):
        packet=self.core.invoke(dict(operation='observation',sequence=0));original=deepcopy(packet)
        created=packet['body']['created_epoch'];elapsed=[0.];waits=[]
        def sleep(seconds):
            waits.append(seconds);elapsed[0]+=seconds
        with self.assertRaises(PermissionError):fixture.check_query(packet,self.registry,self.plan,created-.068)
        result=fixture.receive_query(packet,self.registry,self.plan,now=lambda:created-.068+elapsed[0],
            clock=lambda:elapsed[0],sleep=sleep)
        self.assertEqual(result,packet['body']);self.assertEqual(packet,original)
        self.assertGreaterEqual(elapsed[0],.068);self.assertLess(elapsed[0],.08)
        self.assertTrue(waits);self.assertEqual(packet['body']['expires_epoch'],created+60)

    def test_receiver_clock_hold_is_bounded_and_never_accepts_future_or_expired(self):
        packet=self.core.invoke(dict(operation='observation',sequence=0));created=packet['body']['created_epoch']
        elapsed=[0.];waits=[]
        def sleep(seconds):
            waits.append(seconds);elapsed[0]+=seconds
        with self.assertRaises(PermissionError):
            fixture.receive_query(packet,self.registry,self.plan,now=lambda:created-.1,
                clock=lambda:elapsed[0],sleep=sleep)
        self.assertLessEqual(elapsed[0],1);self.assertTrue(waits)
        for sample in [created-2,created+60,created+61]:
            with self.subTest(receiver_time=sample),self.assertRaises(PermissionError):
                fixture.receive_query(packet,self.registry,self.plan,now=lambda:sample,
                    sleep=lambda _:self.fail('Invalid time must fail without waiting'))
        malformed=deepcopy(packet['body']);malformed['run_uuid']='other'
        with self.assertRaises(PermissionError):
            fixture.receive_query(sealed(malformed),self.registry,self.plan,now=lambda:created-.1,
                sleep=lambda _:self.fail('Invalid identity must fail before waiting'))

    def test_wire_record_corruption_is_not_a_successful_transport_cycle(self):
        def corrupt(role,args):
            r=self.call(role,args);r['wire']['response_sha256']='f'*64;return r
        with self.assertRaises(ValueError):route(corrupt,self.plan,self.registry,lambda row:None)
        self.assertEqual(len(list((self.core.folder/'installs').glob('*.json'))),0)

    def test_external_namespace_and_unbound_label_are_rejected(self):
        with self.assertRaises(PermissionError):fixture.namespace(self.root,'N1-PREMODEL-ALLOCATION-13')
        with self.assertRaises(PermissionError):fixture.namespace(self.root,'POLICY-ALLOCATION-06')
        outside=Path(self.tmp.name)/'outside';outside.mkdir();root=Path(self.tmp.name)/'escape';root.mkdir()
        (root/'engineering').symlink_to(outside,target_is_directory=True)
        with self.assertRaises(PermissionError):fixture.namespace(root,self.label)

    def test_missing_or_unsealed_fixture_plan_fails_before_qsub(self):
        manifest=read(self.root/'RUN_MANIFEST.json')
        self.assertEqual(fixture.prepared_plan(self.root,manifest),self.plan)
        changed=deepcopy(manifest);changed['execution']['sealed_inputs'].pop(fixture.PLAN)
        with self.assertRaises(PermissionError):fixture.prepared_plan(self.root,changed)
        (self.root/fixture.PLAN).chmod(0o600)
        with self.assertRaises(PermissionError):fixture.prepared_plan(self.root,manifest)


if __name__=='__main__':unittest.main()
