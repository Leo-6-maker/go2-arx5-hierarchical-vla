"""Full original remaining-resource recomputation for a Core GPU cold request.

No native IO, process start, model permission or caller cap override. The same
raw GLOBAL/LOCAL markers, all original attempts and matrix/final protections
are required. Conservative original PBS/site bounds are retained; Core CPU,
model transfer and Core-resident neural output are added without dedup credit.
"""
import hashlib
import json
from copy import deepcopy

from . import evaluation_core_accounting as c,evaluation_scope as scope,evaluation_core_B24_1000 as late
from . import evaluation_admission as original,evaluation_stage as stage
from .evaluation_work import remaining,MATRIX
from .future_budget import CAPS,FINAL,_read,_vector,_number,_fresh


def need(ok,why):
    if not ok:raise PermissionError('PV1_CORE_RESOURCE_'+why)


def digest(value):return hashlib.sha256(original.canonical(value)).hexdigest()


def request_identity(manifest,request,manifest_sha):
    m,r=manifest,request
    need(m.get('schema')=='PI0294_PV1_NEURAL_CHILD_RUN_MANIFEST_V1'
         and m.get('scope')==c.ROLE and m.get('owner')=='EXISTING_LOCAL_PC'
         and m.get('owner_user')=='liuyu-nfs' and m.get('owner_uid')==2009
         and m.get('platform_evaluation_version')==c.VERSION
         and m.get('execution',{}).get('profile')==c.PROFILE
         and m['execution'].get('uid')==2009 and m['execution'].get('host')=='corelab-4080s',
         'EXACT_COLD_CORE_NEURAL_ROLE_NOT_CPU_OR_VANDA_ALIAS')
    need(r.get('schema')=='PI0294_PV1_CORE_NATIVE_GPU_REQUEST_V1'
         and r.get('platform_evaluation_version')==c.VERSION
         and r.get('native_platform')=='CORELAB_RTX4080'
         and r.get('run_uuid')==m.get('run_uuid')!=scope.CAMPAIGN
         and r.get('campaign_run_uuid')==scope.CAMPAIGN
         and r.get('authority_release')==c.AUTHORITY
         and r.get('run_manifest_sha256')==manifest_sha
         and r.get('source_sha')==m.get('code',{}).get('vla_commit')
         and c.canonical_uuid(r.get('request_id')) and c.canonical_uuid(r.get('submission_nonce')),
         'SAME_ORIGINAL_CAMPAIGN_AND_SEALED_NATIVE_REQUEST')
    need(not any(k in r for k in ('project','submit_count','job_name','PBS_JOBID',
         'platform_training_version','training_block','request_sha256','native_handle',
         'status','charged_effects','reserved_neural_queries')),'NO_PBS_OR_LEDGER_RECORD_AS_REQUEST')
    need(type(r.get('allocation_number')) is int and r['allocation_number'] in (1,2),
         'EXACT_ORIGINAL_STAGE_NATIVE_ORDINAL')
    expected=dict(evaluation_candidate=r.get('evaluation_candidate'),stage='P4_DEV',
        noise_bank=294101,remaining_conditions=24,walltime_s=28800,CPU_upper_s=80000,
        declared_process_CPU_limit_s=79760,active_cpu_limit=4,requested_memory_bytes=48*1024**3,
        minimum_GPU_memory_bytes=30*1024**3,optimizer_updates=0,Teacher_takeover=False,
        normalizer_refit=False,evaluation_data_for_training=False,LOCKBOX_opened=False,
        numerical_mode='DETERMINISTIC_CUDA_MATH_SDP_V1',statistics_sha256=c.old.R0,
        parent_checkpoint_manifest_sha256=c.old.PARENT,original_T0_epoch=c.old.T0,
        hard_close_epoch=c.old.HARD_CLOSE,model_transfer_payload_bytes=c.MODEL_BYTES)
    need(r.get('evaluation_candidate') in stage.CANDIDATES
         and r.get('checkpoint_manifest_sha256')==c.old.ENDPOINTS[r['evaluation_candidate']]
         and all(r.get(k)==v and type(r[k]) is type(v) for k,v in expected.items()),
         'ORIGINAL_AB_R0_WHOLE_PANEL_ZERO_TRAINING_ONLY')
    stage.request(m,r)
    if 'startup_recovery_sha256' in r:
        from .evaluation_core_zero_start_recovery import SCHEMA
        need(r.get('evaluation_candidate')=='B_COVERAGE' and c.canonical_uuid(r.get('control_id'))
             and r['run_uuid'] in ('pv1-v4a-b4096-n294101-zero-start-recovery-01','pv1-v4a-b4096-n294101-read-recovery-02','pv1-v4a-b4096-n294101-shared-gpu0-03','pv1-v4a-b4096-n294101-shared-gpu0-repair-04')
             and m['execution']['sealed_inputs'].get('plans/CORE_B_ZERO_START_COLD_RECOVERY.json')
             ==r['sealed_inputs'].get('plans/CORE_B_ZERO_START_COLD_RECOVERY.json')
             and type(r.get('startup_recovery_sha256')) is str and len(r['startup_recovery_sha256'])==64,
             'EXACT_SEALED_SINGLE_KNOWN_ZERO_COLD_B_REPLACEMENT')
    late.check_request(m,r)
    queue_fields=[k for k in r if k.startswith('queue_')]
    if queue_fields:
        from . import evaluation_core_queue as queue
        need(r.get('queue_dispatch_mode')==queue.MODE and c.canonical_uuid(r.get('queue_id'))
             and type(r.get('queue_position')) is int and r['queue_position']==r['allocation_number']-1
             and m['execution']['sealed_inputs'].get('plans/CORE_RESIDENT_STAGE_QUEUE.json')==r.get('queue_plan_sha256')
             and type(r.get('queue_plan_sha256')) is str
             and len(r['queue_plan_sha256'])==64,'EXACT_PRESEALED_SERIAL_QUEUE_MODE')


def recompute(*,manifest_bytes,request_bytes,population_bytes,evidence_by_candidate,
              global_bytes,local_bytes,snapshot_bytes,now_epoch,purpose,native_facts_bytes=None):
    need(purpose in ('PREPARE','RESERVE','NATIVE_PREMODEL'),'KNOWN_PHASE_NOT_NATIVE_GO')
    native=purpose=='NATIVE_PREMODEL'
    need(native==(native_facts_bytes is not None),'NATIVE_PROCESS_FACTS_REQUIRED_ONLY_FOR_BOUND_PHASE')
    m,msha=_read(manifest_bytes);r,rsha=_read(request_bytes);g,gsha=_read(global_bytes);l,lsha=_read(local_bytes)
    snapshot,_=_read(snapshot_bytes);c.checked(g);scope.validate_pair(g,l)
    need(snapshot.get('heartbeat_only_raw_ledger') is not True,'NO_NATIVE_OR_MODEL_ADMISSION_FROM_UNKNOWN_SCHEDULER_HEARTBEAT')
    request_identity(m,r,msha)
    need(not g['unknown_effects'] and not g['pending_updates'],'GLOBAL_UNKNOWN_OR_PENDING_FENCE')
    need(snapshot.get('schema')==original.SNAPSHOT and snapshot.get('campaign_run_uuid')==scope.CAMPAIGN
         and snapshot.get('original_T0_epoch')==c.old.T0 and snapshot.get('hard_close_epoch')==c.old.HARD_CLOSE
         and snapshot.get('global_ledger_sha256')==gsha and snapshot.get('LOCAL_ledger_sha256')==lsha,
         'CURRENT_EXACT_RAW_LEDGER_AND_ORIGINAL_CLOCK_BINDING')
    _fresh(snapshot.get('observed_epoch'),now_epoch,120)
    if not native:_fresh(r.get('created_epoch'),now_epoch,120)
    limit=late.check_request(m,r,g)
    need(c.old.T0<=now_epoch<(c.old.STOP_PHYSICS if native else limit),'ORIGINAL_CLOCK_OR_EXACT_HUMAN_B24_EXCEPTION')
    need(not any(x['status'] not in ('NATIVE_TERMINAL','NATIVE_QSUB_REJECTED_NO_JOB') for x in g['PBS_submissions'])
         and (native or not any(x['status']!=c.STATES[3] for x in g.get(c.FIELD,{}).values()))
         and (native or r['request_id'] not in g.get(c.FIELD,{}))
         and all(x.get('request_id')!=r['request_id'] for x in g['PBS_submissions']),
         'ONE_NEW_INACTIVE_REQUEST_NOT_REISSUING_UNKNOWN')
    native_process=None
    if native:
        facts,_=_read(native_facts_bytes);native_process=facts
        bound=g.get(c.FIELD,{}).get(r['request_id']);need(type(bound) is dict,'ACTUAL_BOUND_CORE_ROW')
        need(bound['status'] in c.STATES[1:3] and bound['request_sha256']==rsha
             and bound['run_manifest_sha256']==msha and bound['child_run_uuid']==r['run_uuid']
             and bound['source_sha']==r['source_sha'] and bound['submission_nonce']==r['submission_nonce']
             and all(facts.get(k)==v and type(facts[k]) is type(v) for k,v in dict(
                 schema='PV1_CORE_GPU_NATIVE_PROCESS_STARTED_V1',hostname='corelab-4080s',uid=2009,
                 GPU_uuid=r['GPU_uuid'],request_id=r['request_id'],request_sha256=rsha,
                 submission_nonce=r['submission_nonce'],run_manifest_sha256=msha,source_sha=r['source_sha'],
                 process_state='RUNNING',native_handle=bound['native_handle'],boot_id=bound['boot_id'],
                 pid=bound['pid'],proc_start=bound['proc_start'],started_epoch=bound['process_started_epoch'],
                 native_model_permission=False,physics_authorized=False).items()),
             'CURRENT_EXACT_ACTUAL_BOUND_PROCESS_NOT_COLD_INTENT')
        _fresh(facts.get('observed_epoch'),now_epoch,120)
        from .evaluation_core_identity import kernel_limits,parse_GPU_rows
        limits=facts['kernel_limits'];kernel_limits(str(limits['CPU_quota'])+' '+str(limits['CPU_period']),
            str(limits['memory_max_bytes']),str(limits['pids_max']))
        unit=facts['native_unit_properties'];reads=facts['native_GPU_raw']
        need(unit.get('MainPID')==str(bound['pid']) and unit.get('ActiveState')=='active'
             and unit.get('RuntimeMaxUSec')=='8h' and bool(unit.get('ControlGroup'))
             and set(reads)=={'devices','processes'} and all(type(x.get('returncode')) is int
                 and x['returncode']==0 for x in reads.values()),'ACTUAL_UNIT_LIMITS_AND_FULL_NVIDIA_READS')
        need(parse_GPU_rows(reads['devices']['stdout'],reads['processes']['stdout'],GPU_uuid=r['GPU_uuid'],shared_request=r if 'shared_GPU_grant_sha256' in r else None)==facts['device'],
             'ONE_REAL_SELECTED_CORE_DEVICE_BEFORE_CUDA_OR_MODEL')
    work=remaining(population_bytes,evidence_by_candidate=evidence_by_candidate)
    work=stage.first_stage(work,manifest=m,current_request=r,global_ledger=g,
        purpose='NATIVE_PREMODEL' if native else 'PREPARE')
    inventory=snapshot.get('primary_attempt_inventory')
    need(type(inventory) is dict and set(inventory)==set(c.old.ENDPOINTS)
         and all(type(ids) is list and len(ids)==len(set(ids))
             and set(ids)==set(work['histories'][candidate]['all_actual_attempts'])
             for candidate,ids in inventory.items()),'ALL_ORIGINAL_ACTUAL_ATTEMPTS_NOT_SUCCESS_ONLY')
    selected=[j for j in work['jobs'] if j['candidate']==r['evaluation_candidate'] and j['noise_bank']==294101]
    need(len(selected)==1,'ONE_STILL_UNFINISHED_ORIGINAL_PANEL')
    job=dict(selected[0],native_platform='CORELAB_RTX4080')
    job['GPU_site_output_upper_bytes']+=c.MODEL_BYTES
    need(r.get('planned_panel_sha256')==digest(job)
         and r.get('prior_outcomes_sha256')==digest(work['histories'][r['evaluation_candidate']]['prior_rows'])
         and r.get('additional_output_upper_bytes')==job['GPU_site_output_upper_bytes']
         and r.get('new_bank_manifest_sha256')==work['new_bank_manifest_sha256'],
         'EXACT_UNCHANGED_CONDITIONS_AND_PRIOR_HISTORY')
    used,reserved=_vector(snapshot['used']),_vector(snapshot['reserved'])
    floors=original.resource_floor(g,l)
    need(all(used[k]>=floors[0][k] and reserved[k]>=floors[1][k] for k in CAPS),
         'NO_LOST_ACTUAL_PAID_OR_PENDING_RESOURCE')
    need(type(snapshot.get('Core_storage_upper_bytes')) is int
         and snapshot['Core_storage_upper_bytes']>0
         and used['Core_storage_bytes']>=snapshot['Core_storage_upper_bytes']
         and type(snapshot.get('LOCAL_storage_upper_bytes')) is int
         and snapshot['LOCAL_storage_upper_bytes']>0
         and used['local_storage_bytes']>=snapshot['LOCAL_storage_upper_bytes'],
         'INCLUSIVE_ACTUAL_STORAGE_UPPERS')
    future=_vector(work['resources']);jobs=len(work['jobs'])
    # Do not refund a Vanda/PBS bound because a Core process is being prepared.
    # Keeping those original future bounds makes the initial migration fit
    # conservative until actual known terminal costs replace them.
    future['Core_CPU_s']+=jobs*87200+(600 if jobs else 0)
    future['Core_storage_bytes']+=jobs*c.MODEL_BYTES+sum(j['GPU_site_output_upper_bytes'] for j in work['jobs'])
    future['Hopper_storage_bytes']+=jobs*c.MODEL_BYTES
    future['transfer_bytes']+=jobs*2*c.MODEL_BYTES
    if native:
        # Credit only real escrow rows for these unfinished original panels.
        # A queued intent, prepared file or caller amount grants no credit.
        for held in g[c.FIELD].values():
            if held['status']==c.STATES[3]:continue
            need(held['status'] in (c.STATES[1],c.STATES[2],c.STATES[5])
                 and any(j['candidate']==held['evaluation_candidate'] and j['noise_bank']==held['noise_bank']
                     for j in work['jobs']),'NO_UNACCEPTED_INTENT_OR_UNCLASSIFIED_NATIVE_HOLD')
            local_hold=l.get('Core_native_GPU_reservations',{}).get(held['request_id'])
            need(type(local_hold) is dict and local_hold.get('request_sha256')==held['request_sha256']
                 and local_hold.get('status')=='RESERVED_NOT_SPENT'
                 and local_hold.get('Core_CPU_upper_s')==c.CPU_UPPER
                 and local_hold.get('Core_output_upper_bytes')==held['additional_output_upper_bytes']
                 and local_hold.get('transfer_upper_bytes')==2*c.MODEL_BYTES,'REAL_PAIRED_LOCAL_ESCROW_BEFORE_CREDIT')
            credits=dict(H200_s=c.GPU_UPPER,neural_queries=held['query_limit'],
                Core_CPU_s=c.CPU_UPPER,Hopper_CPU_s=c.CPU_UPPER,
                Core_storage_bytes=held['additional_output_upper_bytes'],
                Hopper_storage_bytes=held['additional_output_upper_bytes'],transfer_bytes=2*c.MODEL_BYTES)
            for key,amount in credits.items():
                need(future[key]>=amount,'NO_DOUBLE_OR_EXCESS_RESERVATION_CREDIT:'+key)
                future[key]-=amount
    caps=original.adopted_caps(g);dimensions={}
    for key in CAPS:
        total=used[key]+reserved[key]+future[key]+MATRIX[key]+_vector(FINAL)[key]
        dimensions[key]=dict(cap=caps[key],used=used[key],reserved=reserved[key],
            remaining_stage_upper=future[key],matrix=MATRIX[key],final=_vector(FINAL)[key],
            projected=total,shortfall=max(0,total-caps[key]))
    if native:
        clock=stage.clock(work,now_epoch=now_epoch,current_native_facts=dict(
            allocation_start_epoch=native_process['started_epoch']))
        if 'queue_dispatch_mode' in r:
            from .evaluation_core_queue import FIELD as QUEUE_FIELD
            need(g[QUEUE_FIELD]['phase']=='NATIVE_ACCEPTED'
                 and g[QUEUE_FIELD]['accepted_epoch']<c.old.STOP_NEW,'ACTUAL_PREEXISTING_NATIVE_QUEUE')
            clock['last_stage_submission_fits_original_stop_new']=True
            clock['dispatch_mode']=r['queue_dispatch_mode']
    elif 'queue_dispatch_mode' in r:
        from .evaluation_core_queue import clock as queue_clock
        clock=queue_clock(work,now_epoch=now_epoch)
    else:clock=stage.clock(work,now_epoch=now_epoch)
    if limit!=c.old.STOP_NEW:
        need(len(work['jobs'])==1 and work['jobs'][0]['candidate']=='B_COVERAGE' and work['jobs'][0]['noise_bank']==294101,'ONLY_THE_ONE_B24_LATE_EXCEPTION')
        clock['last_stage_submission_fits_original_stop_new']=(native_process['started_epoch'] if native else now_epoch)<c.old.STOP_NEW
        clock['last_stage_submission_fits_exact_human_B24_exception']=(native_process['started_epoch'] if native else now_epoch)<limit
        clock['exact_human_B24_start_limit_epoch']=limit
        clock['original_stop_new_epoch_unchanged']=c.old.STOP_NEW
    shortfalls={k:x['shortfall'] for k,x in dimensions.items() if x['shortfall']}
    skeleton={k:v for k,v in r.items() if k!='prequeue_future_admission_sha256'}
    context=dict(manifest_sha256=msha,request_skeleton_sha256=digest(skeleton),global_sha256=gsha,
        LOCAL_sha256=lsha,resource_snapshot_sha256=hashlib.sha256(snapshot_bytes).hexdigest(),
        work_sha256=digest(work),future_upper=future,dimensions=dimensions)
    context_sha=digest(context)
    need(not shortfalls and clock['both_stage_panels_fit_original_physics_window']
         and (clock['last_stage_submission_fits_original_stop_new'] or clock.get('last_stage_submission_fits_exact_human_B24_exception') is True),
         'FULL_RESOURCES_AND_BOTH_ORIGINAL_STAGE_CLOCKS')
    if purpose=='RESERVE':need(r.get('prequeue_future_admission_sha256')==context_sha,
         'EXACT_PREPARE_CONTEXT_NO_MUTABLE_CAP_OR_CLOCK')
    return dict(schema='PV1_CORE_NATIVE_GPU_FULL_COLD_RESOURCE_RECOMPUTATION_V1',
        purpose=purpose,request_id=r['request_id'],request_sha256=rsha,
        prequeue_context_sha256=context_sha,global_ledger_sha256=gsha,LOCAL_ledger_sha256=lsha,
        dimensions=dimensions,shortfalls=shortfalls,all_remaining_costs_fit=True,stage_clock=clock,
        work=job,all_original_history_preserved=True,original_matrix_and_final_protected=True,
        original_T0_epoch=c.old.T0,hard_close_epoch=c.old.HARD_CLOSE,
        native_or_scientific_admission=False,new_training_updates=0,new_native_writes=0)


def reserve_pair_proposal(*,inputs,record):
    """Two existing-ledger proposals; persist with original once IO, not here."""
    proof=recompute(**dict(inputs,purpose='RESERVE'))
    g,_=_read(inputs['global_bytes']);l,_=_read(inputs['local_bytes']);r,rsha=_read(inputs['request_bytes'])
    c.row(record)
    need(record['request_id']==r['request_id'] and record['request_sha256']==rsha
         and record['run_manifest_sha256']==r['run_manifest_sha256']
         and record['child_run_uuid']==r['run_uuid']
         and all(record.get(k)==r.get(k) for k in ('source_sha','submission_nonce','evaluation_candidate',
             'checkpoint_manifest_sha256','statistics_sha256','GPU_uuid','planned_panel_sha256',
             'prior_outcomes_sha256','prequeue_future_admission_sha256','created_epoch')),
         'SAME_EXACT_BYTE_BOUND_CURRENT_CORE_REQUEST')
    after=c.reserve(g,record);local=deepcopy(l)
    local['GPU_reserved_s']+=c.GPU_UPPER
    local['CPU_reserved_s']['CoreLab']+=c.CPU_UPPER
    local['CPU_reserved_s']['Hopper']+=c.CPU_UPPER
    local.setdefault('Core_native_GPU_reservations',{})[r['request_id']]=dict(
        request_id=r['request_id'],request_sha256=rsha,source_sha=r['source_sha'],run_uuid=r['run_uuid'],
        status='RESERVED_NOT_SPENT',Core_CPU_upper_s=c.CPU_UPPER,
        Core_output_upper_bytes=record['additional_output_upper_bytes'],
        transfer_upper_bytes=2*c.MODEL_BYTES,charges_DEC1_subcaps=False)
    held=local.setdefault('Core_native_GPU_reserved',dict(CoreLab_output_bytes=0,transfer_bytes=0))
    held['CoreLab_output_bytes']+=record['additional_output_upper_bytes']
    held['transfer_bytes']+=2*c.MODEL_BYTES
    scope.validate_pair(after,local);original.resource_floor(after,local)
    return dict(proposed_global=after,proposed_LOCAL=local,recomputed_proof=proof,
        native_dispatch_admitted=False,IO_performed=False,PBS_submissions_consumed=0)


def reserve_stage_queue_proposal(pair):
    """Full dual-ledger pair intent; neither accepted queue nor GPU dispatch."""
    from . import evaluation_core_queue as queue
    need(type(pair) is list and len(pair)==2,'EXACT_ATOMIC_TWO_REQUEST_QUEUE')
    first=pair[0]['inputs'];records=[];proofs=[]
    g,_=_read(first['global_bytes']);local,_=_read(first['local_bytes']);local=deepcopy(local)
    for position,item in enumerate(pair):
        inputs,record=item['inputs'],item['record']
        need(all(inputs[k]==first[k] for k in ('global_bytes','local_bytes','snapshot_bytes',
             'population_bytes','evidence_by_candidate','now_epoch')),'ONE_FRESH_FULL_SHARED_PAIR_SNAPSHOT')
        proof=recompute(**dict(inputs,purpose='RESERVE'));r,rsha=_read(inputs['request_bytes'])
        c.row(record)
        keys=('source_sha','submission_nonce','evaluation_candidate','checkpoint_manifest_sha256','statistics_sha256',
            'GPU_uuid','planned_panel_sha256','prior_outcomes_sha256','prequeue_future_admission_sha256',
            'created_epoch','queue_id','queue_position','queue_dispatch_mode','queue_plan_sha256')
        need(record['request_id']==r['request_id'] and record['request_sha256']==rsha
             and record['run_manifest_sha256']==r['run_manifest_sha256']
             and record['child_run_uuid']==r['run_uuid'] and record['queue_position']==position
             and record['evaluation_candidate']==stage.CANDIDATES[position]
             and record['status']==c.STATES[4] and r['queue_dispatch_mode']==queue.MODE
             and all(record.get(k)==r.get(k) for k in keys),'SAME_EXACT_TWO_IMMUTABLE_REQUEST_BYTES')
        records.append(record);proofs.append(proof)
        local['GPU_reserved_s']+=c.GPU_UPPER
        local['CPU_reserved_s']['CoreLab']+=c.CPU_UPPER
        local['CPU_reserved_s']['Hopper']+=c.CPU_UPPER
        local.setdefault('Core_native_GPU_reservations',{})[r['request_id']]=dict(
            request_id=r['request_id'],request_sha256=rsha,source_sha=r['source_sha'],run_uuid=r['run_uuid'],
            status='RESERVED_NOT_SPENT',Core_CPU_upper_s=c.CPU_UPPER,
            Core_output_upper_bytes=record['additional_output_upper_bytes'],
            transfer_upper_bytes=2*c.MODEL_BYTES,charges_DEC1_subcaps=False)
        held=local.setdefault('Core_native_GPU_reserved',dict(CoreLab_output_bytes=0,transfer_bytes=0))
        held['CoreLab_output_bytes']+=record['additional_output_upper_bytes'];held['transfer_bytes']+=2*c.MODEL_BYTES
    after=queue.reserve_two(g,records);scope.validate_pair(after,local);original.resource_floor(after,local)
    return dict(proposed_global=after,proposed_LOCAL=local,recomputed_proofs=proofs,
        native_queue_accepted=False,native_dispatch_admitted=False,IO_performed=False,PBS_submissions_consumed=0)
