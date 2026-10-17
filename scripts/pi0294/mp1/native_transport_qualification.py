"""Produce same-job N1 qualification from actual raw packets and native reads."""
from pathlib import Path
import resource
import time

from scripts.pi0292.durable import read,write
from scripts.pi0292.relay_protocol import digest,unseal
from scripts.pi0294.native_dataset import sha
from scripts.pi0294.relay_endpoint import inside
from .native_heartbeat_capture import capture,FILENAME as HEARTBEAT
from .n1_limits import RELEASE
from .transport_fixture import BASE,CYCLES,PLAN,Endpoint,Pump,immutable,namespace,check_query,check_response
from .transport_fixture_faults import start_exit_heartbeat,finish as finish_faults
from .transport_fixture_router import verify_cycle

OWNER='OWNER_TRANSPORT_FAULTS.json'
BINDING='TRANSPORT_RAW_BINDING.json'
FAULT_RAW=[
    'actual_expired_heartbeat/OWNER_HEARTBEAT.json','actual_expired_heartbeat/WRITER_TERMINAL.json',
    'actual_future_heartbeat/OWNER_HEARTBEAT.json','wrong_run_heartbeat/OWNER_HEARTBEAT.json',
    'concurrent_replace/READS.json','concurrent_replace/OWNER_HEARTBEAT.json',
    'metadata_read_failure/RESOURCE.json','statistics_lock/COUNTERS.json']


def verify_faults(folder,run,faults,heartbeat):
    folder=Path(folder)/'faults';expired=read(folder/'actual_expired_heartbeat/OWNER_HEARTBEAT.json')
    writer=read(folder/'actual_expired_heartbeat/WRITER_TERMINAL.json');proof=heartbeat[-1]['proof']
    clock=proof['revalidated_clock_epoch'] if proof['server_revalidation'] else proof['cached_clock_epoch']
    diagnostic=faults['actual_expired_heartbeat']['diagnostics']
    if (expired['run_uuid']!=run or writer['returncode']!=0 or writer['heartbeat_sha256']!=sha(folder/'actual_expired_heartbeat/OWNER_HEARTBEAT.json')
            or not 0<=writer['observed_epoch']-expired['observed_epoch']<20 or clock-expired['observed_epoch']<=300
            or diagnostic['heartbeat']!=expired or diagnostic['revalidated_clock_epoch']<clock
            or diagnostic['revalidated_clock_epoch']-expired['observed_epoch']<=300):
        raise PermissionError('ACTUAL_EXITED_FIXTURE_WRITER_RAW_HEARTBEAT_MUST_TRULY_EXPIRE')
    future=read(folder/'actual_future_heartbeat/OWNER_HEARTBEAT.json');diagnostic=faults['actual_future_heartbeat']['diagnostics']
    if future['run_uuid']!=run or diagnostic['heartbeat']!=future or diagnostic['revalidated_clock_epoch']-future['observed_epoch']>=0:
        raise PermissionError('ACTUAL_FUTURE_HEARTBEAT_RAW_NEGATIVE_AGE_REQUIRED')
    if read(folder/'wrong_run_heartbeat/OWNER_HEARTBEAT.json')['run_uuid']==run:
        raise PermissionError('ACTUAL_WRONG_RUN_FIXTURE_REQUIRED')
    replaced=read(folder/'concurrent_replace/READS.json')
    if len(replaced['rows'])<40 or len(replaced['writes'])<2 or len({r['heartbeat']['observed_epoch'] for r in replaced['rows']})<2:
        raise PermissionError('ACTUAL_CONCURRENT_REPLACE_AND_POST_READ_SERIES_REQUIRED')
    for proof in replaced['rows']:
        clock=proof['revalidated_clock_epoch'] if proof['server_revalidation'] else proof['cached_clock_epoch']
        if (proof['heartbeat_written'] is not False or proof['clock_sampled_after_read'] is not True
                or proof['heartbeat']['run_uuid']!=run or not 0<=clock-proof['heartbeat']['observed_epoch']<=300):
            raise PermissionError('CONCURRENT_FIXTURE_MUST_USE_ORIGINAL_POST_READ_WATCHDOG')
    metadata=faults['metadata_read_failure'];stats=faults['statistics_lock']
    if (read(folder/'metadata_read_failure/RESOURCE.json')!={'fixture':True} or metadata['attempts']!=2
            or metadata['injected_errno']!=116 or not 0<=metadata['wall_s']<=120
            or metadata['recovered_only_declared_idempotent_read'] is not True
            or stats['actual_lock_holder_exit']!=0 or stats['reply_while_lock_held'] is not True
            or not 0<=stats['reply_wall_s']<20 or read(folder/'statistics_lock/COUNTERS.json')['RPC_calls']!=1):
        raise PermissionError('ACTUAL_CLASSIFIED_METADATA_RECOVERY_AND_HELD_STATS_LOCK_REPLY')


def run(root,directory,allocation,guard):
    from .pre_model_qualification import FILENAME,require
    root=Path(root);directory=Path(directory)
    if (directory/FILENAME).exists():return require(root,directory,allocation)
    label=BASE+'-ALLOCATION-'+directory.name.removeprefix('alloc-')
    endpoint=Endpoint(root,'hopper',label)
    pump=Pump(root,label,allocation,guard);folder=pump.folder
    children_before=resource.getrusage(resource.RUSAGE_CHILDREN)
    try:
        start_exit_heartbeat(folder,root.name)
        write(directory/'N1_PREMODEL_READY.json',dict(status='N1_ISOLATED_TRANSPORT_READY_BEFORE_MODEL',
            run_uuid=root.name,label=label,job_id=allocation['job_id'],hostname=allocation['hostname'],
            registry_sha256=sha(endpoint.registry_path),source_sha=endpoint.manifest['code']['vla_commit'],
            observed_epoch=time.time(),deadline_epoch=endpoint.registry['deadline_epoch'],
            models_loaded=0,new_model_queries=0,new_physics_integrations=0,heartbeat_written=False))
        heartbeat=capture(root,directory,allocation,guard,service=pump.poll)
        deadline=min(time.time()+120,endpoint.registry['deadline_epoch'])
        cycles=None;owner_path=folder/OWNER
        while time.time()<deadline:
            guard();pump.poll();cycles=pump.completed()
            if cycles is not None and owner_path.exists():break
            time.sleep(.05)
        if cycles is None or not owner_path.exists():raise TimeoutError('REAL_EIGHTY_TRANSPORT_CYCLES_OR_OWNER_FAULT_PROOFS_MISSING')
        owner=immutable(owner_path)
        if (owner['schema']!='MP1_N1_OWNER_TRANSPORT_FAULTS_V1' or owner['job_id']!=allocation['job_id']
                or owner['run_uuid']!=root.name or owner['label']!=label
                or owner['registry_sha256']!=sha(endpoint.registry_path)
                or owner['nominal_cycles_sha256']!=digest(cycles) or owner['nominal_cycles']!=80
                or set(owner['faults'])!={'late_response','process_exit'} or set(owner['nominal_terminals'])!={'core','hopper'}
                or any(r['native_process_terminal_verified'] is not True for r in owner['nominal_terminals'].values())
                or owner['nominal_terminals']['core']['actual_install_count']!=80
                or owner['nominal_terminals']['core']['actual_install_ACKs_sha256']!=digest([
                    read(folder/'completed'/(str(i).zfill(3)+'.json'))['ack'] for i in range(CYCLES)])):
            raise PermissionError('EXACT_REAL_OWNER_RAW_CYCLES_AND_ISOLATED_FAULT_RESULTS')
        for name,value in owner['faults'].items():
            if (value['status']!='PASS' or value['failed_closed'] is not True or value['unknown_effects_resent']
                    or value['native_process_terminal_verified'] is not True
                    or value['second_call_blocked_before_send'] is not True or value['actual_fault_requests']!=1
                    or value['job_id']!=allocation['job_id'] or value['source_sha']!=endpoint.manifest['code']['vla_commit']
                    or value['label']!=label.replace(BASE,BASE+('-FAULT-LATE' if name=='late_response' else '-FAULT-EXIT'),1)
                    or value['actual_server_exit']!=(0 if name=='late_response' else 72)):
                raise PermissionError('LATE_OR_EXIT_FIXTURE_MUST_FAIL_CLOSED_WITHOUT_REPLAY')
        faults=finish_faults(folder,root.name)
        faults.update(owner['faults']);members={}
        for kind in ['inbox','outbox','completed','compute-intents']:
            for sequence in range(CYCLES):
                path=folder/kind/(str(sequence).zfill(3)+'.json');members[str(path.relative_to(root))]=sha(path)
        for name in faults:
            if name in owner['faults']:continue
            path=folder/'faults'/name/'RESULT.json';members[str(path.relative_to(root))]=sha(path)
        for name in FAULT_RAW:
            path=folder/'faults'/name;members[str(path.relative_to(root))]=sha(path)
        members[str(owner_path.relative_to(root))]=sha(owner_path)
        for name in members:inside(root,name).chmod(0o400)
        binding=folder/BINDING
        write(binding,dict(schema='MP1_N1_TRANSPORT_RAW_BINDING_V1',run_uuid=root.name,label=label,
            job_id=allocation['job_id'],source_sha=endpoint.manifest['code']['vla_commit'],
            plan_sha256=sha(root/PLAN),registry_sha256=sha(endpoint.registry_path),members=members))
        binding.chmod(0o400)
        value=dict(schema='MP1_N1_NATIVE_ENDURANCE_AND_TRANSPORT_V1',
            status='PASS_NATIVE_PREMODEL_HEARTBEAT_AND_REAL_TRANSPORT',authority_release=RELEASE,
            run_uuid=root.name,job_id=allocation['job_id'],hostname=allocation['hostname'],
            source_sha=endpoint.manifest['code']['vla_commit'],manifest_sha256=sha(root/'RUN_MANIFEST.json'),
            models_loaded=0,new_model_queries=0,new_physics_integrations=0,production_heartbeat_or_science_fault_injection=False,
            heartbeat=dict(clock_source=heartbeat['clock_source'],rows=heartbeat['rows'],native_fragment_sha256=sha(directory/HEARTBEAT)),
            transport=dict(codec='PRODUCTION_JSON_LINE_STDIO_AND_SEALED_OBSERVATION_V1',scope='ISOLATED_FIXTURE_NO_MODEL_NO_PHYSICS',
                watchdogs_s=[20,60,900],cycles=cycles,duplicate_fixture_installs=0,physical_installs=0,unknown_effects_resent=0,
                reference_RGB_payload_bytes=endpoint.plan['RGB_payload_bytes'],reference_payload_sha256=endpoint.plan['recorded_payload_sha256'],
                raw_binding_relative=str(binding.relative_to(root)),raw_binding_sha256=sha(binding)),isolated_faults=faults)
        path=directory/FILENAME;write(path,value);path.chmod(0o400)
        return require(root,directory,allocation)
    finally:
        pump.close();after=resource.getrusage(resource.RUSAGE_CHILDREN)
        # Difference monotonic counters separately so equal totals cannot cancel below zero.
        user_extra=after.ru_utime-children_before.ru_utime
        system_extra=after.ru_stime-children_before.ru_stime
        if min(user_extra,system_extra)<0:raise ValueError('NONNEGATIVE_NATIVE_FIXTURE_CHILD_CPU')
        endpoint.ledger.cpu(user_extra+system_extra)


def verify(root,directory,allocation,value):
    root=Path(root);directory=Path(directory);label=BASE+'-ALLOCATION-'+directory.name.removeprefix('alloc-')
    folder,_,mode=namespace(root,label);plan=immutable(root/PLAN);registry=immutable(root/'plans'/label/'REGISTRY.json')
    transport=value['transport'];binding_path=inside(root,transport['raw_binding_relative']);binding=immutable(binding_path)
    if (binding_path.resolve()!=(folder/BINDING).resolve() or sha(binding_path)!=transport['raw_binding_sha256']
            or binding['schema']!='MP1_N1_TRANSPORT_RAW_BINDING_V1' or binding['run_uuid']!=root.name
            or binding['label']!=label or binding['job_id']!=allocation['job_id']
            or binding['source_sha']!=value['source_sha'] or binding['plan_sha256']!=sha(root/PLAN)
            or binding['registry_sha256']!=sha(root/'plans'/label/'REGISTRY.json')
            or registry['schema']!='MP1_N1_TRANSPORT_REGISTRY_V1' or registry['run_uuid']!=root.name
            or registry['label']!=label or registry['job_id']!=allocation['job_id']
            or registry['hopper_manifest_sha256']!=sha(root/'RUN_MANIFEST.json')
            or registry['hopper_source_sha']!=value['source_sha']
            or registry['deadline_epoch']>allocation['allocation_end_epoch']-2700
            or read(root/'RUN_MANIFEST.json')['execution']['sealed_inputs'].get(PLAN)!=sha(root/PLAN)
            or transport['reference_RGB_payload_bytes']!=plan['RGB_payload_bytes']
            or transport['reference_payload_sha256']!=plan['recorded_payload_sha256']):
        raise PermissionError('SAME_JOB_ACTUAL_NATIVE_TRANSPORT_RAW_BINDING_REQUIRED')
    expected={str((folder/kind/(str(i).zfill(3)+'.json')).relative_to(root))
        for kind in ['inbox','outbox','completed','compute-intents'] for i in range(CYCLES)}
    native_faults=set(value['isolated_faults'])-{'late_response','process_exit'}
    expected|={str((folder/'faults'/name/'RESULT.json').relative_to(root)) for name in native_faults}
    expected|={str((folder/'faults'/name).relative_to(root)) for name in FAULT_RAW}
    expected.add(str((folder/OWNER).relative_to(root)))
    if set(binding['members'])!=expected:raise PermissionError('COMPLETE_RAW_EIGHTY_CYCLE_AND_FAULT_MANIFEST')
    for name,digest_ in binding['members'].items():
        path=inside(root,name)
        if path.stat().st_mode&0o222 or sha(path)!=digest_:raise PermissionError('RAW_FIXTURE_MEMBER_CHANGED:'+name)
    for sequence,row in enumerate(transport['cycles']):
        name=str(sequence).zfill(3)+'.json';query=read(folder/'inbox'/name);response=read(folder/'outbox'/name)
        completed=read(folder/'completed'/name);intent=read(folder/'compute-intents'/name)
        body=unseal(query);check_query(query,registry,plan,intent['started_epoch']);check_response(response,query,registry,plan)
        verify_cycle(row,query,response,completed['ack'])
        ack=unseal(completed['ack'])
        if (completed['cycle']!=row or row['query_packet_sha256']!=query['sha256']
                or row['response_packet_sha256']!=response['sha256'] or row['core_ACK_sha256']!=completed['ack']['sha256']
                or ack['request_sha256']!=query['sha256'] or ack['response_sha256']!=response['sha256']
                or not body['created_epoch']<=ack['installed_epoch']<body['expires_epoch']
                or intent['request_sha256']!=query['sha256']):
            raise PermissionError('RAW_NATIVE_INPUT_RESPONSE_INTENT_CORE_ACK_AND_CYCLE_MISMATCH')
    owner=immutable(folder/OWNER)
    if (owner['nominal_cycles_sha256']!=digest(transport['cycles']) or owner['job_id']!=allocation['job_id']
            or owner['nominal_terminals']['core']['actual_install_count']!=80
            or owner['nominal_terminals']['core']['actual_install_ACKs_sha256']!=digest([
                read(folder/'completed'/(str(i).zfill(3)+'.json'))['ack'] for i in range(CYCLES)])
            or any(r['native_process_terminal_verified'] is not True for r in owner['nominal_terminals'].values())):
        raise PermissionError('OWNER_AND_NATIVE_TRANSPORT_CYCLES_OR_PROCESS_CLOSURE_DISAGREE')
    for name,fault in value['isolated_faults'].items():
        raw=owner['faults'][name] if name in ['late_response','process_exit'] else read(folder/'faults'/name/'RESULT.json')
        if raw!=fault:raise PermissionError('NATIVE_ISOLATED_FAULT_MUST_EQUAL_RAW_RECEIPT')
        if name in ['late_response','process_exit']:
            if (fault['job_id']!=allocation['job_id'] or fault['source_sha']!=value['source_sha']
                    or fault['label']!=label.replace(BASE,BASE+('-FAULT-LATE' if name=='late_response' else '-FAULT-EXIT'),1)
                    or fault['actual_server_exit']!=(0 if name=='late_response' else 72)
                    or fault['native_process_terminal_verified'] is not True or fault['actual_fault_requests']!=1
                    or fault['second_call_blocked_before_send'] is not True):
                raise PermissionError('ACTUAL_SAME_JOB_FAULT_PROCESS_EXIT_AND_POISONED_CLIENT_PROOF')
    verify_faults(folder,root.name,value['isolated_faults'],value['heartbeat']['rows'])
    return dict(status='PASS_RAW_NATIVE_TRANSPORT_JOIN',cycles=CYCLES)
