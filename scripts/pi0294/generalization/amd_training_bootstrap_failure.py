"""Exact1497713 Q-to-R bootstrap failure; preserve original missing effects.

This positive source/native-final audit is not a generic absence-to-zero rule.
It never edits the old process final, invents a native effect journal, refunds
paid costs or authorizes another PBS submission.
"""
import ast,base64,json,math
from copy import deepcopy
from . import amd_training_accounting as a

JOB='1497713.stdct-mgmt-02'
REQUEST='24ce96df-f84a-4b35-b2c8-e932fe6e5c02'
SOURCE='b8eb0c0d2535eb0af1712b1dfeb0cca97773b723'
ERROR='GEN1_AMD_ADOPTION_TRAIN_CURRENT_NATIVE_SERVER_OWNER_HEARTBEAT'
STATUS='PASS_INDEPENDENT_GEN1_PRERELEASE_NATIVE_FAILURE_NO_NEURAL_OUTPUT'

def audit(capture):
    def blob(name):
        x=capture['files'][name];b=base64.b64decode(x['base64'],validate=True)
        a.need(len(b)==x['bytes']and a.sha(b)==x['sha256'],'BOOT_FAIL_CAPTURE_BYTES:'+name);return b
    def read(name):return json.loads(blob(name))
    gb=base64.b64decode(capture['GLOBAL_base64'],validate=True)
    a.need(a.sha(gb)==capture['GLOBAL_sha256'],'BOOT_FAIL_REAL_CURRENT_GLOBAL_BYTES')
    g=json.loads(gb);row=a.rows(g)[0];manifest=read('RUN_MANIFEST.json');seal=read('SOURCE_SEAL.json')
    req=read('PBS/REQUEST.json');allocation=read('PBS/ALLOCATION_ADMISSION.json');beat=read('control/OWNER_HEARTBEAT.json')
    final=read('PBS/PBS_PROCESS_FINAL.json');wrapper=read('PBS/PBS_WRAPPER_FINAL.json');owner=read('control/RESIDENT_OWNER_OS_EXIT.json')
    a.need(row['request_id']==req['request_id']==REQUEST and row['job_id']==JOB
        and row['source_sha']==manifest['code']['vla_commit']==seal['source_sha']==SOURCE
        and row['status']==a.STATES[1]and row['request_sha256']==a.sha(blob('PBS/REQUEST.json'))
        and row['run_manifest_sha256']==a.sha(blob('RUN_MANIFEST.json'))
        and manifest['code']['source_seal_sha256']==a.sha(blob('SOURCE_SEAL.json')),
        'BOOT_FAIL_EXACT_REGISTERED_SOURCE_REQUEST_AND_MANIFEST')
    for name,x in capture['modules'].items():
        b=base64.b64decode(x['base64'],validate=True)
        a.need(len(b)==x['bytes']and a.sha(b)==x['sha256']==seal['files'][name],
            'BOOT_FAIL_REAL_EXECUTED_SOURCE:'+name)
    source=base64.b64decode(capture['modules']['scripts/pi0294/generalization/amd_training_runtime.py']['base64'])
    tree=ast.parse(source);run=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='run')
    guarded=[n for n in ast.walk(run)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='guard']
    load=[n for n in ast.walk(run)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)
        and isinstance(n.func.value,ast.Name)and n.func.value.id=='model'and n.func.attr=='load']
    fork=[n for n in ast.walk(run)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)
        and isinstance(n.func.value,ast.Name)and n.func.value.id=='subprocess'and n.func.attr=='run']
    init=[n for n in ast.walk(run)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)
        and isinstance(n.func.value,ast.Name)and n.func.value.id=='ledger'and n.func.attr=='initialize']
    a.need(len(guarded)==1 and len(fork)==1 and len(init)==1 and len(load)==3
        and guarded[0].lineno<fork[0].lineno<init[0].lineno<min(n.lineno for n in load),
        'BOOT_FAIL_POSITIVE_GUARD_BEFORE_CPU_CHILD_EFFECT_INIT_AND_ANY_MODEL')
    a.need(final['status']=='GEN1_FAILED_OR_PARTIAL_PRESERVED'and final['phase']=='CPU_BANK_PRELUDE'
        and final['error']==dict(type='PermissionError',message=ERROR)and final['result']is None
        and final['job_id']==JOB and final['source_sha']==SOURCE and final['request_sha256']==row['request_sha256']
        and 'effects'not in final and 'effect_summary_error'not in final
        and final['new_physics']==final['new_episodes']==0,
        'BOOT_FAIL_ACTUAL_PRE_CHILD_EXCEPTION_NOT_ASSUMED_FROM_ABSENCE')
    a.need(all(beat[k]==v for k,v in dict(run_uuid=manifest['run_uuid'],job_id=JOB,
        request_id=REQUEST,submission_nonce=req['submission_nonce'],request_sha256=row['request_sha256'],
        source_sha=SOURCE,scope=a.VERSION,grant_sha256=a.authority.GRANT_SHA,
        native_job_state='Q',global_unknown_effects=[]).items()),
        'BOOT_FAIL_ONLY_NATIVE_STATE_WAS_QUEUED_NOT_IDENTITY_OR_UNKNOWN')
    a.need(allocation['status']=='PASS_GEN1_OWN_NATIVE_A40_ALLOCATION'
        and allocation['job_id']==JOB and allocation['request_id']==REQUEST
        and allocation['source_sha']==SOURCE and allocation['request_sha256']==row['request_sha256']
        and 0<allocation['allocation_start_epoch']-beat['observed_epoch']<300,
        'BOOT_FAIL_REAL_R_ALLOCATION_AFTER_FRESH_SAME_OWNER_Q')
    native=capture['native'][JOB];variables=native['Variable_List']
    a.need(set(capture['native'])=={JOB}and native['job_state']=='F'and type(native['Exit_status'])is int
        and native['Exit_status']==1 and native['run_count']==1
        and native['Job_Name']==req['job_name']=='pi0294_dec1_a40_train'
        and native['project']==req['project']=='CFP04-CF-007'
        and native['Job_Owner'].split('@')[0]=='e1192548'and int(native['Resource_List']['ngpus'])==1
        and variables['PV1_REQUEST_ID']==REQUEST and variables['PV1_SUBMISSION_NONCE']==req['submission_nonce']
        and allocation['hostname'].split('.')[0]in {x.split('/')[0]for x in native['exec_host'].split('+')},
        'BOOT_FAIL_REAL_F_ONE_ALLOCATED_GPU_NOT_RESOURCES_USED_NGPUS_ZERO')
    a.need(wrapper['returncode']==owner['returncode']==1 and wrapper['job_id']==JOB
        and wrapper['source_sha']==owner['source_sha']==SOURCE
        and len(capture['actors'])==2 and all(x['actual_procstart']!=x['expected_procstart']for x in capture['actors']),
        'BOOT_FAIL_REAL_WRAPPER_OWNER_OS1_AND_NO_ACTIVE_RESTART')
    a.need(read('control/resident-owner/setup-rpc/RPC_EFFECT_STATE.json')['status']=='CONFIRMED'
        and read('control/resident-owner/running-rpc/RPC_EFFECT_STATE.json')['status']=='CONFIRMED',
        'BOOT_FAIL_NO_UNKNOWN_NATIVE_QSUB_OR_HEARTBEAT_PUBLICATION')
    log=blob('PBS/pbs.log').decode()
    a.need(log.rstrip().endswith('PermissionError: '+ERROR)and 'try:guard(root,allocation);break'in log,
        'BOOT_FAIL_REAL_GUARD_CAUSAL_TRACEBACK')
    a.need(all(x==[]for x in capture['inventory'].values())
        and set(capture['inventory'])=={'training','datasets','reservations','worker_processes'}
        and 'CPU_BANK_PRELUDE_INTENT.json'in capture['absent_PBS']
        and 'START_NEURAL.json'in capture['absent_PBS']and 'A40_NATIVE_EFFECT_LEDGER.json'in capture['absent'],
        'BOOT_FAIL_MATCHING_EMPTY_SCIENTIFIC_NAMESPACES_AND_GUARD_BOUNDARY')
    retained=capture['retained'];a.need(type(retained['logical_bytes'])is int and 0<retained['logical_bytes']<=64*1024**2
        and len(retained['symlinks'])==1
        and retained['symlinks'][0]['path']=='model_inputs/nvidia/Cosmos-Reason2-2B'
        and retained['symlinks'][0]['target'].endswith('/snapshots/9ce19a195e423419c349abfc86fd07178b230561'),
        'BOOT_FAIL_ALL_RETAINED_CONTROL_INSIDE_ALREADY_PAID64M_NO_NEURAL_PAYLOAD')
    from scripts.pi0294.e0_scope import pbs_epoch
    hms=lambda x:sum(int(v)*n for v,n in zip(x.split(':'),(3600,60,1)))
    gpu=max(hms(native['resources_used']['walltime']),pbs_epoch(native['obittime'])-pbs_epoch(native['stime']))+60
    cpu=max(hms(native['resources_used']['cput']),final['CPU_actual_s'])+120
    terminal=dict(job_id=JOB,job_state='F',request_id=REQUEST,submission_nonce=req['submission_nonce'],Exit_status=1,
        conservative_GPU_s=gpu,conservative_CPU_s=cpu,raw_native_sha256=a.sha(capture['native_raw'].encode()),
        process_final_sha256=a.sha(blob('PBS/PBS_PROCESS_FINAL.json')),wrapper_final_sha256=a.sha(blob('PBS/PBS_WRAPPER_FINAL.json')))
    effects=dict(schema=a.EFFECTS,job_id=JOB,request_id=REQUEST,request_sha256=row['request_sha256'],source_sha=SOURCE,
        arm=a.bank.IDENTITY,start_native_step=0,optimizer_updates_committed=0,last_native_step=0,
        query_attempts=0,model_load_attempts=0,pending_updates=[],unknown_query_intents=[],
        derived_from_exact_prerelease_source_and_real_final=True,original_native_effect_ledger_created=False,
        original_process_final_unchanged=True,new_physics=0,new_episodes=0)
    report=dict(status=STATUS,causal_boundary='FRESH_SAME_OWNER_Q_HEARTBEAT_BEFORE_FIRST_R_OWNER_POLL',
        source_sha=SOURCE,request_sha256=row['request_sha256'],all_processes_terminal=True,
        actual_retained_control_bytes=retained['logical_bytes'],already_paid_control_output_upper_bytes=64*1024**2,
        actual_neural_output_bytes=0,declared_neural_output_reservation_released_not_refunded=70000000000,
        original_final_and_missing_ledger_preserved=True,heartbeat_sha256=a.sha(blob('control/OWNER_HEARTBEAT.json')),
        source_runtime_sha256=capture['modules']['scripts/pi0294/generalization/amd_training_runtime.py']['sha256'],
        scientific_endpoint_complete=False,models=0,queries=0,updates=0,physics_ticks=0)
    return dict(native=terminal,effects=effects,audit=report)

def settle(g,capture):
    result=audit(capture);a.need(a.sha(a.raw(g))==capture['GLOBAL_sha256'],
        'BOOT_FAIL_FRESH_EXACT_GLOBAL_NOT_AN_OLD_PROPOSAL')
    # The generic terminal arithmetic is retained. This exact proven phase
    # never reached CPU packaging or a model and used only already-paid64M
    # control output. Its70GB neural reservation was never a paid cost.
    out=a.settle_training(g,REQUEST,**result,now_epoch=capture['epoch'])
    out['new_storage_current_bytes']=g['new_storage_current_bytes']
    out['new_storage_peak_bytes']=g['new_storage_peak_bytes']
    row=a.rows(out)[0];row.update(known_prerelease_failure_version=STATUS,
        neural_output_charge_bytes=0,unused_neural_output_reservation_bytes=70000000000,
        metadata_retained_inside_already_paid64MiB=True)
    out[a.authority.FIELD][a.FIELD]['known_prerelease_failure']=deepcopy(result['audit'])
    a.rows(out);a.original.validate_global(out)
    a.need(out[a.original.PACKAGE]==g[a.original.PACKAGE]and out['new_storage_current_bytes']==g['new_storage_current_bytes']
        and out['new_storage_peak_bytes']==g['new_storage_peak_bytes'],
        'BOOT_FAIL_NO_PAID_COST_OR_ORIGINAL_ESCROW_REFUND')
    return dict(proposed_global=out,**result,full_future_requires_fresh_admission=True)
