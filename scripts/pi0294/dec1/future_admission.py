"""Versioned DEC1 source-first branch of the mature fresh-owner budget bridge."""
from pathlib import Path

from scripts.pi0294.pv1 import future_admission as mature
from scripts.pi0294.pv1 import future_budget as old
from .authority import AUTHORITY,PARENT_AUTHORITY,PARENT
from .source_admission import VERSION,COPY
from .future_budget import forecast

INDEX='plans/DEC1_FUTURE_BUDGET_INPUT_INDEX.json'


def fail(message):
    raise PermissionError('DEC1_FUTURE_ADMISSION_'+message)


def inputs(root,request_bytes):
    root=Path(root);manifest_raw=mature._file(root,'RUN_MANIFEST.json')
    manifest,_=old._read(manifest_raw);request,request_sha=old._read(request_bytes)
    if (request.get('scientific_admission_version')!=VERSION
            or request.get('authority_release')!=PARENT_AUTHORITY
            or request.get('schema')!='PI0294_PV1_NATIVE_ALLOCATION_REQUEST_V1'
            or manifest.get('schema')!='PI0294_PV1_NEURAL_CHILD_RUN_MANIFEST_V1'
            or manifest.get('owner')!='EXISTING_LOCAL_PC'
            or request.get('run_uuid')!=root.name or manifest.get('run_uuid')!=root.name
            or request.get('run_manifest_sha256')!=mature._digest(manifest_raw)
            or request.get('source_sha')!=manifest['code']['vla_commit']):
        fail('ORIGINAL_NATIVE_REQUEST_MANIFEST_AND_EXPLICIT_NEW_VERSION')
    index_raw=mature._file(root,INDEX);index,_=old._read(index_raw)
    if (index.get('schema')!='DEC1_SOURCE_FUTURE_BUDGET_INPUT_INDEX_V1'
            or index.get('authority_release')!=AUTHORITY
            or index.get('runtime_refresh_policy')!='SAME_OWNER_SAME_JOB_FRESH_RECOMPUTATION_V1'
            or index.get('campaign_run_uuid')!=request.get('campaign_run_uuid')
            or index.get('T0_epoch')!=manifest['limits']['PV1_T0_epoch']):
        fail('NEW_SEALED_INDEX_SAME_ORIGINAL_CAMPAIGN')
    fields=('plan','request_work_binding','baseline_hopper_ledger','performance_evidence',
            'checkpoint_size_evidence','storage_lifecycle_evidence')
    names=[index[k] for k in fields]
    if len(set(names))!=len(names) or INDEX in names:
        fail('NO_ALIASED_OR_SELF_REFERENTIAL_SEALED_EVIDENCE')
    sealed=manifest['execution']['sealed_inputs']
    raw={n:mature._file(root,n) for n in set(names)|set(sealed)|{INDEX}}
    if (any(mature._digest(v)!=sealed.get(n) for n,v in raw.items())
            or not set(sealed)<=set(request['sealed_inputs'])):
        fail('ALL_INPUTS_IN_ORIGINAL_SEALED_MANIFEST_AND_REQUEST')
    if any(mature._digest(raw[n] if n in raw else mature._file(root,n))!=h for n,h in request['sealed_inputs'].items()):
        fail('REQUEST_INPUT_BYTES_CHANGED')
    from scripts.pi0294.pv1.neural import stage_work
    stage_work(root,request)
    return request,request_sha,manifest,index,raw


def recompute(root,*,request_bytes,snapshot_bytes,progress_bytes,purpose,now_epoch):
    if purpose not in ('PREPARE','SUBMIT','NATIVE_PRETRAIN'):
        fail('ONLY_ORIGINAL_PREQUEUE_OR_FRESH_NATIVE_PRETRAIN_PURPOSE')
    root=Path(root);request,request_sha,manifest,index,files=inputs(root,request_bytes)
    snapshot,snapshot_sha=old._read(snapshot_bytes);progress,progress_sha=old._read(progress_bytes)
    if purpose!='NATIVE_PRETRAIN':old._fresh(request['created_epoch'],now_epoch,120)
    if (snapshot['campaign_run_uuid']!=index['campaign_run_uuid']
            or snapshot['T0_epoch']!=index['T0_epoch']):
        fail('NO_NEW_CLOCK_OR_PARALLEL_CAMPAIGN')
    ledger,ledger_sha=mature._ledger(root,snapshot,progress)
    adopted,_=old._read(files[COPY])
    baseline,_=old._read(files[index['baseline_hopper_ledger']])
    left=adopted['accounting']['original_remaining_after_spent_reservations_and_final']
    original_count=24-left['GPU_submissions']
    if (baseline.get('schema')!='PI0294_PV1_HOPPER_CAMPAIGN_LEDGER_V1'
            or baseline.get('authority_release')!=PARENT_AUTHORITY
            or baseline.get('campaign_run_uuid')!=request['campaign_run_uuid']
            or baseline.get('unknown_effects')!=[] or baseline.get('pending_updates')!=[]
            or len(ledger['PBS_submissions'])<original_count
            or len(baseline['PBS_submissions'])!=original_count
            or baseline['generation_queries']!=48000-9000-left['neural_queries']
            or baseline['optimizer_updates']!=16384-left['optimizer_updates']):
        fail('ACTUAL_ADOPTION_BASELINE_NOT_A_RESET_OR_FAKE_ZERO')
    for before,after in zip(baseline['PBS_submissions'],ledger['PBS_submissions']):
        if any(before.get(k)!=after.get(k) for k in ('job_id','request_id','run_uuid','request_sha256','project')):
            fail('HISTORICAL_NATIVE_JOB_IDENTITIES_PRESERVED')
    new_items=ledger['PBS_submissions'][original_count:]
    from .alternate_accounting import VERSION as ALTERNATE_VERSION,validate_item as alternate_item
    for item in new_items:
        if item.get('scientific_admission_version')==ALTERNATE_VERSION:
            alternate_item(item)
            if item['status']!='NATIVE_TERMINAL':
                fail('ALTERNATE_QUALIFICATION_MUST_TERMINATE_BEFORE_SOURCE_ADMISSION')
        elif item.get('scientific_admission_version')!=VERSION:
            fail('POST_ADOPTION_NATIVE_JOBS_REQUIRE_EXPLICIT_DEC1_CLASSIFICATION')
    active=[s for s in ledger['PBS_submissions'] if s['status'] not in mature.TERMINAL]
    current=[s for s in active if s['request_id']==request['request_id']]
    if len(active)>1 or active and (len(current)!=1 or not active[0].get('job_id')):
        fail('EXCLUSIVE_EXACT_CURRENT_JOB_NOT_A_RECOVERABLE_OLD_HOLD')
    expected=dict(H200_s=sum(old._number(s.get('conservative_H200_s',0)) for s in new_items if s['status']=='NATIVE_TERMINAL'),
        PBS_submissions=len(new_items),optimizer_updates=ledger['optimizer_updates']-baseline['optimizer_updates'],
        neural_queries=ledger['generation_queries']-baseline['generation_queries'],
        Hopper_CPU_s=ledger['CPU_s']-baseline['CPU_s'])
    expected_reserved=dict(H200_s=sum(s['reserved_H200_s']+s['accounting_pad_s'] for s in new_items if s['status'] not in mature.TERMINAL),
        Hopper_storage_bytes=sum(s['additional_output_upper_bytes'] for s in new_items if s['status'] not in mature.TERMINAL))
    for table,values in (('dec1_used',expected),('dec1_reserved',expected_reserved)):
        projected=old._vector(snapshot[table])
        if any(projected[k]!=v for k,v in values.items()):
            fail('NATIVE_DEC1_COSTS_AND_RESERVATIONS_NOT_UNDERCOUNTED_OR_DOUBLE_CHARGED')
    plan,_=old._read(files[index['plan']]);binding,binding_sha=old._read(files[index['request_work_binding']])
    for field,key in (('performance_evidence','performance_evidence_sha256'),
                      ('checkpoint_size_evidence','checkpoint_size_evidence_sha256')):
        if plan[key]!=mature._digest(files[index[field]]):fail('ORIGINAL_PERFORMANCE_AND_SIZE_EVIDENCE_BYTES')
    timing,_=old._read(files[index['performance_evidence']])
    if (timing.get('schema')!='DEC1_PRIOR_PARENT_STEP_TIMING_BOUND_V1'
            or timing.get('parent_checkpoint_manifest_sha256')!=PARENT
            or timing.get('not_current_PV1_measurement') is not True
            or timing.get('new_model_loads')!=0 or timing.get('new_updates')!=0
            or timing.get('new_physics')!=0 or not timing.get('step_logs')):
        fail('ACTUAL_PRIOR_PARENT_TIMING_NOT_A_NEW_TRAINING_MEASUREMENT')
    maximum=0
    for record in timing['step_logs']:
        old._sha(record['file_sha256'])
        first=old._number(record['first_step'],integer=True,positive=True)
        last=old._number(record['last_step'],integer=True,positive=True)
        count=old._number(record['count'],integer=True,positive=True)
        peak=old._number(record['maximum_s'],positive=True)
        if last<first or count!=last-first+1:fail('PRIOR_STEP_LOG_COMPLETE_INTERVAL')
        maximum=max(maximum,peak)
    multiplier=old._number(timing['declared_upper_multiplier'],positive=True)
    if multiplier<1.1 or plan['seconds_per_update_upper']<maximum*multiplier:
        fail('STEP_TIMING_UPPER_COVERS_PRIOR_MAXIMUM_WITH_DECLARED_HEADROOM')
    if plan['storage_lifecycle']['implementation_or_execution_plan_sha256']!=mature._digest(files[index['storage_lifecycle_evidence']]):
        fail('ORIGINAL_BOUNDED_STORAGE_LIFECYCLE_NOT_A_DELETION_RECEIPT')
    size,_=old._read(files[index['checkpoint_size_evidence']])
    if (size.get('new_model_loads')!=0 or not size.get('checkpoint_members')
            or sum(old._number(r['bytes'],integer=True) for r in size['checkpoint_members'])!=size.get('total_payload_bytes')
            or plan['full_checkpoint_payload_upper_bytes']<size['total_payload_bytes']):
        fail('ACTUAL_CHECKPOINT_BYTES_BEFORE_NEW_OUTPUT_RESERVATION')
    block=request['training_block'];arm=block['arm'];start=request['expected_native_start_step'];stop=block['stop_after_step']
    if (binding.get('schema')!='DEC1_FUTURE_REQUEST_WORK_BINDING_V1' or binding.get('stage')!='P2_PRIMARY'
            or binding.get('arm')!=arm or binding.get('replica') is not False
            or binding.get('start_step')!=start or binding.get('stop_step')!=stop
            or binding.get('probe_only',False)!=block.get('probe_only',False)
            or ledger.get('training_runs',{}).get(arm,{}).get('last_native_step',0)!=start
            or progress['candidates'][arm]['optimizer_updates_committed']!=start):
        fail('EXACT_SOURCE_BLOCK_AND_ACTUAL_CHARGED_FULL_RESUME_STEP')
    if current and current[0].get('dec1_work_id')!=binding['work_id']:
        fail('CURRENT_JOB_WORK_ID_CANNOT_BE_REASSIGNED')
    output=forecast(snapshot=snapshot,progress_bytes=progress_bytes,plan=plan,now_epoch=now_epoch)
    jobs=[j for j in output['mandatory_jobs'] if j['work_id']==binding['work_id']]
    if len(jobs)!=1:fail('ONE_REQUIRED_REMAINING_SOURCE_BLOCK')
    job=jobs[0];new_output=request['additional_output_upper_bytes']
    if (job.get('arm')!=arm or job.get('start_step')!=start or job.get('stop_step')!=stop
            or job['walltime_upper_s']!=request['walltime_s']
            or job['resources']['Hopper_storage_bytes']<new_output
            or new_output<plan['full_checkpoint_payload_upper_bytes']
            or request.get('dec1_phase_Hopper_CPU_upper_s')!=job['resources']['Hopper_CPU_s']):
        fail('REQUEST_WALL_CHECKPOINT_OUTPUT_AND_PHASE_ENVELOPE')
    # Before another allocation can write a full checkpoint, retirement/restore
    # transitions must already have produced enough actual resident headroom.
    # A lifecycle plan is never accepted in place of this live pre-write bound.
    already_reserved=expected_reserved['Hopper_storage_bytes'] if current else 0
    if snapshot['dec1_used'].get('Hopper_storage_bytes',0)+new_output+snapshot['dec1_reserved'].get('Hopper_storage_bytes',0)-already_reserved>64*1024**3:
        fail('ACTUAL_CHECKPOINT_RESIDENT_HEADROOM_REQUIRED_BEFORE_QUEUE_OR_UPDATE')
    if not output['admissible_forecast']:fail('NEGATIVE_ORIGINAL_OR_DEC1_REMAINING_RUNWAY')
    return dict(schema='DEC1_RECOMPUTED_SOURCE_FUTURE_ADMISSION_V1',
        status='PASS_RECOMPUTED_DEC1_RESOURCE_ENVELOPE_NOT_NATIVE_ALLOCATION_OR_GO',purpose=purpose,
        authority_amendment=AUTHORITY,request_id=request['request_id'],submission_nonce=request['submission_nonce'],
        request_sha256=request_sha,observed_epoch=now_epoch,snapshot_sha256=snapshot_sha,
        progress_sha256=progress_sha,actual_Hopper_ledger_sha256=ledger_sha,
        request_work_binding_sha256=binding_sha,work_id=binding['work_id'],
        phase_Hopper_CPU_upper_s=job['resources']['Hopper_CPU_s'],forecast=output,
        full_B0_not_fabricated=True,old_PV1_release_or_LOCKBOX_admitted=False)


def declared_phase_cpu(root):
    """Read the already manifest-sealed phase envelope before request hashing."""
    index,_=old._read(mature._file(root,INDEX))
    binding,_=old._read(mature._file(root,index['request_work_binding']))
    plan,_=old._read(mature._file(root,index['plan']))
    matches=[j for j in plan['training_jobs'] if j['work_id']==binding['work_id']]
    if len(matches)!=1:fail('ONE_DECLARED_SOURCE_PHASE_CPU_BOUND')
    return old._number(matches[0]['resources']['Hopper_CPU_s'],integer=True,positive=True)


def bounded_process_cpu(request,original_remaining):
    if request.get('scientific_admission_version')!=VERSION:return original_remaining
    bound=old._number(request['dec1_phase_Hopper_CPU_upper_s'],integer=True,positive=True)
    if bound<6000:fail('PHASE_CPU_INCLUDES_QUALIFICATION_AND_CHECKPOINT_CLOSEOUT')
    return min(original_remaining,bound)
