"""PV1 models require the same native allocation, endurance and real wire proof."""
from pathlib import Path
import os
import re
import socket
import time

from scripts.pi0292.durable import read
from scripts.pi0294.native_dataset import sha
from .decision import AUTHORITY as RELEASE
from .heartbeat_capture import FILENAME as HEARTBEAT_CAPTURE,verify as verify_heartbeat
from .resources import namespace_submission
from . import evaluation_platform as evaluation
from . import evaluation_core_identity as Core_native

from .mix_core_identity import STATUS as CD_CORE_STATUS
CORE_STATUSES=(Core_native.STATUS,CD_CORE_STATUS)

FILENAME='PV1_NATIVE_ENDURANCE_AND_TRANSPORT.json'
FAULTS={'concurrent_replace','metadata_read_failure','statistics_lock','late_response','process_exit',
        'actual_expired_heartbeat','actual_future_heartbeat','wrong_run_heartbeat'}


def current_directory(root,allocation):
    root=Path(root)
    if (allocation.get('status') not in ('PASS_PV1_OWN_NATIVE_H200_ALLOCATION',evaluation.STATUS,*CORE_STATUSES)
            or not Core_native.current_handle() or Core_native.current_handle()!=allocation['job_id']
            or socket.gethostname()!=allocation['hostname']
            or not allocation['allocation_start_epoch']<=time.time()<allocation['allocation_end_epoch']-2700):
        raise PermissionError('PV1_NATIVE_CURRENT_ALLOCATION_BEFORE_MODEL_OR_TRAINING')
    paths=sorted((root/'pbs').glob('alloc-[0-9][0-9]'))
    matches=[p for p in paths if (p/'ALLOCATION_ADMISSION.json').exists() and read(p/'ALLOCATION_ADMISSION.json')==allocation]
    if len(matches)!=1:raise PermissionError('PV1_ONE_EXACT_CURRENT_ADMISSION')
    directory=matches[0]
    namespace_submission(root,int(directory.name.removeprefix('alloc-')))
    ordinal=read(directory/'REQUEST.json').get('allocation_number' if allocation['status']in CORE_STATUSES else 'submit_count')
    if type(ordinal) is not int or ordinal!=int(directory.name.removeprefix('alloc-')):
        raise PermissionError('PV1_CURRENT_REQUEST_AND_NAMESPACE_SUBMISSION_MUST_MATCH')
    if (directory.is_symlink() or (directory/'ALLOCATION_ADMISSION.json').is_symlink()
            or (directory/'ALLOCATION_ADMISSION.json').stat().st_mode&0o222
            or (directory/'REQUEST.json').is_symlink()
            or (directory/'REQUEST.json').stat().st_mode&0o222
            or sha(directory/'REQUEST.json')!=allocation['request_sha256']
            or (directory/'PBS_PROCESS_FINAL.json').exists() or (directory/'NATIVE_PROCESS_FINAL.json').exists()):
        raise PermissionError('PV1_CURRENT_IMMUTABLE_REQUEST_NOT_A_TERMINAL_ALLOCATION')
    manifest=read(root/'RUN_MANIFEST.json') if allocation['status'] in (evaluation.STATUS,*CORE_STATUSES) else None
    request=read(directory/'REQUEST.json')
    if allocation['status'] not in (evaluation.STATUS,*CORE_STATUSES) and any('platform_evaluation_version' in x for x in (request,allocation)):
        raise PermissionError('PV1_EVALUATION_IDENTITY_CANNOT_FALL_BACK_TO_H200')
    if allocation['status']in CORE_STATUSES:
        Core_native.current_identity(root,manifest,request,allocation,qualified_model_readback=True)
    elif allocation['status']==evaluation.STATUS:
        evaluation.sealed_identity(manifest,request,allocation,
            run_uuid=root.name,manifest_sha256=sha(root/'RUN_MANIFEST.json'),
            request_sha256=sha(directory/'REQUEST.json'))
    return directory


def require_current(root,allocation):
    return require(root,current_directory(root,allocation),allocation)


def require(root,directory,allocation):
    root=Path(root);directory=Path(directory);path=directory/FILENAME
    if not path.is_file():raise PermissionError('PV1_PRE_MODEL_NATIVE_600S_80_CYCLE_QUALIFICATION_NOT_RUN')
    if path.is_symlink() or directory.is_symlink() or not directory.resolve().is_relative_to((root/'pbs').resolve()) or path.stat().st_mode&0o222:
        raise PermissionError('PV1_EXACT_IMMUTABLE_NATIVE_QUALIFICATION_RECEIPT')
    raw=path.read_bytes();v=read(path);manifest=read(root/'RUN_MANIFEST.json')
    if (v['schema']!='PI0294_PV1_NATIVE_ENDURANCE_AND_TRANSPORT_V1'
            or v['status']!='PASS_NATIVE_PREMODEL_HEARTBEAT_AND_REAL_TRANSPORT'
            or v['authority_release']!=RELEASE or v['run_uuid']!=root.name
            or v['job_id']!=allocation['job_id'] or v['hostname']!=allocation['hostname']
            or v['source_sha']!=manifest['code']['vla_commit']
            or v['manifest_sha256']!=sha(root/'RUN_MANIFEST.json')
            or v['models_loaded'] or v['new_model_queries'] or v['new_physics_integrations']
            or v['production_heartbeat_or_science_fault_injection'] is not False):
        raise PermissionError('PV1_FRESH_SAME_NATIVE_JOB_PREMODEL_IDENTITY_AND_ZERO_SCIENCE')
    heartbeat=v['heartbeat'];rows=heartbeat['rows']
    fragment=verify_heartbeat(root,directory,allocation)
    if (heartbeat['native_fragment_sha256']!=sha(directory/HEARTBEAT_CAPTURE)
            or fragment['rows']!=rows or fragment['clock_source']!=heartbeat['clock_source']):
        raise PermissionError('PV1_INLINE_HEARTBEAT_MUST_JOIN_ACTUAL_NATIVE_RAW_CAPTURE')
    if (heartbeat['clock_source']!='CLOCK_REALTIME_AFTER_EACH_READ_AND_CLOCK_MONOTONIC'
            or len(rows)<81 or rows[-1]['monotonic_s']-rows[0]['monotonic_s']<600):
        raise PermissionError('PV1_REAL_600S_ENDURANCE_NOT_SINGLE_SMOKE')
    previous=None
    for row in rows:
        proof=row['proof'];clock=proof['revalidated_clock_epoch'] if proof['server_revalidation'] else proof['cached_clock_epoch']
        if (proof['clock_sampled_after_read'] is not True or proof['watchdog_s']!=300
                or proof['heartbeat_written'] is not False or proof['heartbeat']['run_uuid']!=root.name
                or not 0<=clock-proof['heartbeat']['observed_epoch']<=300
                or not allocation['allocation_start_epoch']<=clock<=allocation['allocation_end_epoch']
                or previous is not None and not 0<row['monotonic_s']-previous<=300):
            raise PermissionError('PV1_ORIGINAL_POST_READ_HEARTBEAT_WATCHDOG_UNCHANGED')
        previous=row['monotonic_s']
    transport=v['transport'];cycles=transport['cycles']
    if (transport['codec']!='PRODUCTION_JSON_LINE_STDIO_AND_SEALED_OBSERVATION_V1'
            or transport['scope']!='ISOLATED_FIXTURE_NO_MODEL_NO_PHYSICS'
            or transport['watchdogs_s']!=[20,60,900]
            or len(cycles)!=80 or [c['sequence'] for c in cycles]!=list(range(len(cycles)))
            or len({c['query_uuid'] for c in cycles})!=len(cycles)
            or transport['duplicate_fixture_installs'] or transport['physical_installs']
            or transport['unknown_effects_resent']):
        raise PermissionError('PV1_REAL_FINITE_PRODUCTION_TRANSPORT_CYCLES')
    for row in cycles:
        if (not 0<row['RGB_payload_bytes']<2*1024**2
                or row['RGB_payload_bytes']!=transport['reference_RGB_payload_bytes']
                or not 0<row['maximum_RPC_s']<=20 or row['fixture_install_count']!=1
                or row['ACK_matches_request'] is not True
                or any(not re.fullmatch('[0-9a-f]{64}',row[k]) for k in ['request_sha256','response_sha256'])):
            raise PermissionError('PV1_REAL_SIZED_TRANSPORT_IDENTITY_NO_DUPLICATE_INSTALL')
    faults=v['isolated_faults']
    if set(faults)!=FAULTS or any(x['status']!='PASS' or x['unknown_effects_resent'] for x in faults.values()):
        raise PermissionError('PV1_ISOLATED_FAULT_QUALIFICATIONS_REQUIRED')
    for name in ['actual_expired_heartbeat','actual_future_heartbeat','wrong_run_heartbeat','late_response','process_exit']:
        if faults[name]['failed_closed'] is not True:
            raise PermissionError('PV1_TRUE_LOSS_OR_LATE_REPLY_MUST_STILL_STOP')
    from .native_transport_qualification import verify as verify_transport
    verify_transport(root,directory,allocation,v)
    if path.read_bytes()!=raw:raise PermissionError('PV1_NATIVE_QUALIFICATION_CHANGED_DURING_READ')
    return dict(status='PASS_SAME_JOB_NATIVE_QUALIFICATION_READBACK',receipt_sha256=sha(path),
        heartbeat_elapsed_s=rows[-1]['monotonic_s']-rows[0]['monotonic_s'],transport_cycles=len(cycles),
        new_model_queries=0,new_physics=0)
