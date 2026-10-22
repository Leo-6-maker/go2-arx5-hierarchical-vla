"""Pure A40 source-training escrow in the existing global and DEC1 ledgers.

Callers must persist a once-only intent and both changed ledgers under the
existing global owner lock before qsub. An unknown reply is never retried here.
"""
from copy import deepcopy
import re

from scripts.pi0294.pv1.future_budget import _read
from . import a40_allocation as a
from .alternate_accounting import digest, finite
from .dual_platform import reserve_check
from .lost_save_recovery import lineage,validate_adopted
from .segmented_training import enabled,limits

TERMINAL=('NATIVE_TERMINAL','NATIVE_QSUB_REJECTED_NO_JOB')


def fail(message):
    raise PermissionError('VAP1_A40_ACCOUNTING_'+message)


def validate_item(row):
    lineage(row)
    wall=finite(row.get('reserved_H200_s'),integer=True)
    start=finite(row.get('start_native_step'),integer=True)
    segment=enabled(row);stop=limits(row)[1] if segment else 4096
    expected=dict(native_platform=a.PLATFORM,platform_training_version=a.VERSION,
        scientific_admission_version=a.SCIENCE_VERSION,authority_amendment=a.AUTHORITY,
        stage='P2_PRIMARY',project='CFP04-CF-007',job_name='pi0294_dec1_a40_train',
        parent_checkpoint_manifest_sha256=a.PARENT,statistics_sha256=a.R0,new_bank_manifest_sha256=a.BANK,
        physical_GPU_kind='NVIDIA_A40',physical_GPU_count=1,legacy_GPU_seconds_field_is_projection=True,
        accounting_pad_s=a.PAD,CPU_upper_s=a.cpu_envelope(row,wall),query_limit=29,
        optimizer_update_limit=stop-start)
    if (any(row.get(k)!=v for k,v in expected.items()) or not 2760<wall<=28800
            or (not segment and start>=1024) or row.get('arm') not in ('A_CONTROL','B_COVERAGE')
            or row.get('dec1_work_id')!=('TRAIN:'+row['arm']+f':SEGMENT:{stop}' if segment else 'TRAIN:'+row['arm']+':LONG')):
        fail('EXACT_TYPED_PRE1024_SOURCE_LONG_JOB_ENVELOPE')
    for key,length in (('request_sha256',64),('run_manifest_sha256',64),('source_sha',40)):
        if re.fullmatch('[0-9a-f]{'+str(length)+'}',row.get(key,'')) is None:
            fail('EXACT_SOURCE_REQUEST_AND_MANIFEST_HASHES')
    if row.get('status') not in ('QSUB_INTENT_OUTCOME_UNKNOWN_NO_RETRY','SUBMITTED_PENDING_NATIVE',
                                 'NATIVE_RUNNING_QUALIFICATION','NATIVE_TERMINAL'):
        fail('KNOWN_SOURCE_SUBMISSION_LIFECYCLE_ONLY')
    if row.get('job_id') is None:
        if row['status']!='QSUB_INTENT_OUTCOME_UNKNOWN_NO_RETRY':fail('ONE_UNRESOLVED_INTENT_NO_SECOND_SUBMIT')
    elif re.fullmatch(r'[0-9]+\.stdct-mgmt-02',row['job_id']) is None:
        fail('NATIVE_VANDA_JOB_ID_NOT_A_HOPPER_HANDLE')
    if row['status']=='NATIVE_TERMINAL':
        for name in ('terminal_native_sha256','terminal_effect_summary_sha256'):
            if re.fullmatch('[0-9a-f]{64}',row.get(name,'')) is None:fail('ACTUAL_TERMINAL_PROVENANCE')
        if (finite(row['conservative_H200_s'])>wall+a.PAD
                or finite(row['conservative_CPU_s'])>row['CPU_upper_s']
                or finite(row['committed_optimizer_updates'],integer=True)>row['optimizer_update_limit']
                or finite(row['query_attempts'],integer=True)>29
                or finite(row['reserved_optimizer_updates'],integer=True)>1
                or finite(row['reserved_neural_queries'],integer=True)>row['query_attempts']):
            fail('TERMINAL_COUNTERS_INSIDE_ORIGINAL_FULL_RESERVATION')
    elif (row.get('reserved_optimizer_updates')!=row['optimizer_update_limit']
            or row.get('reserved_neural_queries')!=29):
        fail('WHOLE_JOB_NEURAL_ENVELOPE_HELD_UNTIL_NATIVE_TERMINAL')
    return row


def reserve(value, DEC1_value, *, request_bytes, budget_proof, grant_bytes, campaign_view, now_epoch):
    r,request_sha=_read(request_bytes);a.request_scope(r,now=now_epoch)
    v,d=deepcopy(value),deepcopy(DEC1_value)
    if (v.get('authority_release')!=a.PARENT_AUTHORITY or d.get('authority_release')!=a.AUTHORITY
            or v.get('campaign_run_uuid')!=r['campaign_run_uuid'] or d.get('campaign_run_uuid')!=r['campaign_run_uuid']
            or v.get('unknown_effects')!=[] or v.get('pending_updates')!=[] or d.get('unknown_effects')!=[]
            or v['used']['PBS_submissions']!=len(v['PBS_submissions'])
            or r['submit_count']!=len(v['PBS_submissions'])+1
            or len(d['new_submissions'])>=(11 if enabled(r) else 8) or len(v['PBS_submissions'])>=24
            or any(s.get('request_id')==r['request_id'] for s in v['PBS_submissions'])
            or v.get('DEC1_source_pair_platform',a.PLATFORM)!=a.PLATFORM
            or not now_epoch<a.T0+144*3600
            or not now_epoch+r['walltime_s']+a.PAD<a.T0+156*3600):
        fail('ORIGINAL_SINGLE_LEDGER_AND_SAME_PLATFORM_PAIR_NO_RESET_OR_RETRY')
    if (budget_proof.get('schema')!='VAP1_A40_RECOMPUTED_REMAINING_WORK_ADMISSION_V1'
            or budget_proof.get('purpose')!='PREQUEUE'
            or budget_proof.get('status')!='PASS_ORIGINAL_GLOBAL_AND_DEC1_REMAINING_WORK'
            or budget_proof.get('request_sha256')!=request_sha or budget_proof.get('job_id') is not None
            or budget_proof.get('global_ledger_canonical_sha256')!=digest(value)
            or budget_proof.get('DEC1_ledger_canonical_sha256')!=digest(DEC1_value)
            or not 0<=now_epoch-budget_proof['observed_epoch']<=120
            or budget_proof.get('forecast',{}).get('admissible_forecast') is not True):
        fail('FRESH_RECOMPUTED_WHOLE_WORK_BEFORE_ATOMIC_RESERVATION')
    start=r['expected_native_start_step'];arm=r['training_block']['arm'];count=r['training_block']['stop_after_step']-start
    if enabled(r):
        from .segmented_training import validate_adopted as segment_adopted
        segment_adopted(v,d)
    if (any(v.get('training_runs',{}).get(name,{}).get('last_native_step',0)>0
            for name in ('A_CONTROL','B_COVERAGE')) and v.get('DEC1_source_pair_platform')!=a.PLATFORM):
        fail('DO_NOT_IMPORT_AN_ALREADY_STARTED_HOPPER_SOURCE_ARM')
    run_key=lineage(r)
    if r.get('recovery_authorization_release') is not None:validate_adopted(v,d)
    prior=v.get('training_runs',{}).get(run_key,{})
    if prior.get('last_native_step',0)!=start:
        fail('EXACT_KNOWN_SAME_ARM_PROGRESS_NO_RESTART_OR_ROLLBACK')
    reserve_check(grant_bytes=grant_bytes,view=campaign_view,used=v['used'],reserved=v['reserved'],
        extra=dict(H200_s=r['walltime_s']+a.PAD,PBS_submissions=1,optimizer_updates=count,neural_queries=29),
        platform=a.PLATFORM,job_wall_s=r['walltime_s'],now_epoch=now_epoch)
    if (v['CPU_s']+v.get('CPU_reserved_s',0)+r['CPU_upper_s']>160*3600
            or d['CPU_s']+d.get('CPU_reserved_s',0)+r['CPU_upper_s']>96*3600
            or v['new_storage_current_bytes']+v.get('storage_reserved_bytes',0)+r['additional_output_upper_bytes']+40*a.GIB>250*a.GIB
            or d.get('new_GPU_site_output_allocated_upper_bytes',0)+d.get('GPU_site_output_reserved_bytes',0)
                +r['additional_output_upper_bytes']>(176 if enabled(r) else 160)*a.GIB):
        fail('ACTUAL_CPU_AND_OUTPUT_HEADROOM_NOT_ONLY_GPU_HOURS')
    row=dict(status='QSUB_INTENT_OUTCOME_UNKNOWN_NO_RETRY',job_id=None,
        request_id=r['request_id'],submission_nonce=r['submission_nonce'],request_sha256=request_sha,
        global_submit_count=r['submit_count'],
        source_sha=r['source_sha'],run_manifest_sha256=r['run_manifest_sha256'],run_uuid=r['run_uuid'],
        native_platform=a.PLATFORM,platform_training_version=a.VERSION,scientific_admission_version=a.SCIENCE_VERSION,
        authority_amendment=a.AUTHORITY,stage='P2_PRIMARY',arm=arm,
        dec1_work_id=('TRAIN:'+arm+f':SEGMENT:{r["training_block"]["stop_after_step"]}' if enabled(r) else 'TRAIN:'+arm+':LONG'),
        project=r['project'],job_name=r['job_name'],parent_checkpoint_manifest_sha256=a.PARENT,
        statistics_sha256=a.R0,new_bank_manifest_sha256=a.BANK,start_native_step=start,
        physical_GPU_kind='NVIDIA_A40',physical_GPU_count=1,legacy_GPU_seconds_field_is_projection=True,
        reserved_H200_s=r['walltime_s'],accounting_pad_s=a.PAD,CPU_upper_s=r['CPU_upper_s'],query_limit=29,
        optimizer_update_limit=count,reserved_optimizer_updates=count,reserved_neural_queries=29,
        additional_output_upper_bytes=r['additional_output_upper_bytes'],submitted_intent_epoch=now_epoch)
    if 'CPU_envelope_profile' in r:row['CPU_envelope_profile']=r['CPU_envelope_profile']
    for key in ('recovery_authorization_release','recovery_adoption_sha256','training_lineage_key'):
        if key in r:row[key]=r[key]
    if enabled(r):row.update(segmented_authorization_release=r['segmented_authorization_release'],
        segmented_grant_sha256=r['segmented_grant_sha256'],segment_stop_step=r['training_block']['stop_after_step'])
    validate_item(row);v['PBS_submissions'].append(row);v['used']['PBS_submissions']+=1
    for k,c in dict(H200_s=r['walltime_s']+a.PAD,optimizer_updates=count,neural_queries=29).items():
        v['reserved'][k]=v['reserved'].get(k,0)+c
    v['CPU_reserved_s']=v.get('CPU_reserved_s',0)+r['CPU_upper_s']
    v['storage_reserved_bytes']=v.get('storage_reserved_bytes',0)+r['additional_output_upper_bytes']
    v['DEC1_source_pair_platform']=a.PLATFORM
    d['new_submissions'].append(dict(request_id=r['request_id'],job_id=None,platform=a.PLATFORM,
        platform_training_version=a.VERSION,status=row['status']))
    for k,c in dict(GPU_reserved_s=r['walltime_s']+a.PAD,CPU_reserved_s=r['CPU_upper_s'],
        neural_queries_reserved=29,optimizer_updates_reserved=count,
        GPU_site_output_reserved_bytes=r['additional_output_upper_bytes']).items():d[k]=d.get(k,0)+c
    return v,d


def bind_job(value, DEC1_value, *, request_id, job_id, now_epoch):
    v,d=deepcopy(value),deepcopy(DEC1_value)
    rows=[s for s in v['PBS_submissions'] if s.get('request_id')==request_id]
    dr=[s for s in d['new_submissions'] if s.get('request_id')==request_id]
    if len(rows)!=1 or len(dr)!=1:fail('ONE_EXACT_GLOBAL_AND_DEC1_INTENT')
    row=validate_item(rows[0])
    if (row['status']!='QSUB_INTENT_OUTCOME_UNKNOWN_NO_RETRY' or row.get('job_id') is not None
            or dr[0].get('job_id') is not None or dr[0]['status']!=row['status']
            or re.fullmatch(r'[0-9]+\.stdct-mgmt-02',job_id) is None):
        fail('EXACT_NATIVE_REPLY_ONCE_NO_SUBMIT_OR_REBIND')
    for item in (row,dr[0]):item.update(status='SUBMITTED_PENDING_NATIVE',job_id=job_id,native_reply_epoch=now_epoch)
    validate_item(row);return v,d


def settle(value, DEC1_value, *, request_id, native_proof, effects, now_epoch):
    """Settle once from terminal PBS plus the bounded native effect ledger."""
    v,d=deepcopy(value),deepcopy(DEC1_value)
    rows=[s for s in v['PBS_submissions'] if s.get('request_id')==request_id]
    dr=[s for s in d['new_submissions'] if s.get('request_id')==request_id]
    if len(rows)!=1 or len(dr)!=1:fail('ONE_EXACT_GLOBAL_AND_DEC1_SETTLEMENT')
    row=validate_item(rows[0])
    if row['status']=='NATIVE_TERMINAL':fail('ALREADY_SETTLED_NO_SECOND_CHARGE')
    if v.get('pending_updates') or v.get('unknown_effects') or d.get('unknown_effects'):
        fail('PRESERVE_OTHER_UNKNOWN_EFFECTS_BEFORE_TERMINAL_RECONCILIATION')
    expected=dict(schema='VAP1_NATIVE_A40_SOURCE_TERMINAL_V1',job_id=row['job_id'],job_state='F',
        project=row['project'],job_name=row['job_name'],owner='e1192548',ngpus=1,
        request_id=request_id,submission_nonce=row['submission_nonce'],recorded_process_terminal=True)
    if (any(native_proof.get(k)!=n for k,n in expected.items())
            or not row['job_id'] or dr[0].get('job_id')!=row['job_id']
            or not 0<=now_epoch-native_proof['observed_epoch']<=120):
        fail('EXACT_OWN_NATIVE_TERMINAL_JOB_AND_PROCESS_PROOF')
    gpu=finite(native_proof['conservative_GPU_s']);cpu=finite(native_proof['conservative_CPU_s'])
    if gpu>row['reserved_H200_s']+a.PAD or cpu>row['CPU_upper_s']:
        fail('REAL_BOUND_EXCEEDED_PRESERVE_WHOLE_RESERVATION_FOR_RECONCILIATION')
    if (effects.get('schema')!='VAP1_A40_NATIVE_SOURCE_EFFECT_SUMMARY_V1'
            or effects.get('job_id')!=row['job_id'] or effects.get('request_id')!=request_id
            or effects.get('request_sha256')!=row['request_sha256']
            or effects.get('source_sha')!=row['source_sha'] or effects.get('arm')!=row['arm']
            or effects.get('start_native_step')!=row['start_native_step']
            or effects.get('new_physics')!=0 or effects.get('new_episodes')!=0
            or effects.get('effect_journal_complete') is not True):
        fail('SAME_NATIVE_SOURCE_STEP_QUERY_AND_ZERO_PHYSICS_JOURNAL')
    committed=finite(effects['optimizer_updates_committed'],integer=True)
    queries=finite(effects['query_attempts'],integer=True)
    loads=finite(effects['model_load_attempts'],integer=True)
    if (committed>row['optimizer_update_limit'] or queries>29 or loads>4
            or effects['last_native_step']!=row['start_native_step']+committed
            or (committed or queries) and (not loads or not gpu) or loads and not gpu
            or committed and queries<17 or effects['last_native_step']>1024 and row['start_native_step']<1024 and queries!=29):
        fail('BOUNDED_ACTUAL_COMMITTED_STEPS_AND_QUERY_ATTEMPTS')
    pending=effects['pending_updates'];unknown_queries=effects['unknown_query_intents']
    if (type(pending) is not list or len(pending)>1 or type(unknown_queries) is not list
            or len(set(unknown_queries))!=len(unknown_queries)
            or any(type(i) is not int or not 0<=i<queries for i in unknown_queries)
            or pending and (pending[0].get('arm')!=row['arm'] or pending[0].get('request_id')!=request_id
                or pending[0].get('local_step_before')!=effects['last_native_step'])):
        fail('EXPLICIT_UNSETTLED_INTENTS_NEVER_RETRIED_OR_DROPPED')
    output=finite(native_proof['new_resident_allocated_upper_bytes'],integer=True)
    if output>row['additional_output_upper_bytes']:fail('ACTUAL_OUTPUT_EXCEEDS_REGISTERED_ENVELOPE')
    if committed:
        recipe=effects.get('recipe_sha256','')
        if re.fullmatch('[0-9a-f]{64}',recipe) is None:fail('ACTUAL_RECIPE_HASH_FOR_COMMITTED_RUN')
        if row.get('recovery_authorization_release') is not None:
            from .lost_save_recovery import RECIPE
            if recipe!=RECIPE:fail('RECOVERY_PRESERVES_ORIGINAL_MATCHED_RECIPE')
        runs=v.setdefault('training_runs',{});run_key=lineage(row);prior=runs.get(run_key,{})
        if (prior.get('last_native_step',0)!=row['start_native_step']
                or prior and prior.get('recipe_sha256')!=recipe):
            fail('NO_OTHER_ARM_OR_CHANGED_RECIPE_PROGRESS_OVERWRITE')
        runs[run_key]=dict(last_native_step=effects['last_native_step'],recipe_sha256=recipe)
    v['used']['H200_s']+=gpu;v['used']['optimizer_updates']+=committed;v['used']['neural_queries']+=queries
    v['optimizer_updates']=v['used']['optimizer_updates'];v['generation_queries']=v['used']['neural_queries']
    v['model_loads']=v.get('model_loads',0)+loads;v['CPU_s']+=cpu
    released=dict(H200_s=row['reserved_H200_s']+a.PAD,
        optimizer_updates=row['optimizer_update_limit']-len(pending),neural_queries=29-len(unknown_queries))
    for k,c in released.items():
        v['reserved'][k]-=c
        if v['reserved'][k]<0:fail('PRESERVE_OTHER_RESERVATIONS_NO_NEGATIVE_RELEASE')
    v['CPU_reserved_s']-=row['CPU_upper_s'];v['storage_reserved_bytes']-=row['additional_output_upper_bytes']
    v['new_storage_current_bytes']+=output
    v['new_storage_peak_bytes']=max(v.get('new_storage_peak_bytes',0),v['new_storage_current_bytes'])
    d['GPU_allocated_s']+=gpu;d['CPU_s']+=cpu;d['optimizer_updates']+=committed;d['neural_queries']+=queries
    for key,c in dict(GPU_reserved_s=released['H200_s'],CPU_reserved_s=row['CPU_upper_s'],
        optimizer_updates_reserved=released['optimizer_updates'],neural_queries_reserved=released['neural_queries'],
        GPU_site_output_reserved_bytes=row['additional_output_upper_bytes']).items():
        d[key]-=c
        if d[key]<0:fail('PRESERVE_OTHER_DEC1_RESERVATIONS')
    d['new_GPU_site_output_allocated_upper_bytes']=d.get('new_GPU_site_output_allocated_upper_bytes',0)+output
    if pending or unknown_queries:
        fault=dict(kind='A40_SOURCE_UNKNOWN_NATIVE_INTENTS_NO_RETRY',request_id=request_id,
            pending_updates=pending,unknown_query_intents=unknown_queries)
        v['unknown_effects'].append(fault);d['unknown_effects'].append(deepcopy(fault))
        v['pending_updates']=deepcopy(pending)
    row.update(status='NATIVE_TERMINAL',conservative_H200_s=gpu,physical_A40_allocated_s=gpu,
        conservative_CPU_s=cpu,committed_optimizer_updates=committed,query_attempts=queries,
        reserved_optimizer_updates=len(pending),reserved_neural_queries=len(unknown_queries),
        terminal_native_sha256=digest(native_proof),terminal_effect_summary_sha256=digest(effects),
        native_exit_status=native_proof['Exit_status'],settled_epoch=now_epoch)
    dr[0].update(status='NATIVE_TERMINAL',settled_epoch=now_epoch)
    validate_item(row)
    return v,d
