"""Full original remaining-resource recomputation for a Core GPU cold request.

No native IO, process start, model permission or caller cap override. The same
raw GLOBAL/LOCAL markers, all original attempts and matrix/final protections
are required. Conservative original PBS/site bounds are retained; Core CPU,
model transfer and Core-resident neural output are added without dedup credit.
"""
import hashlib
import json
from copy import deepcopy

from . import evaluation_core_accounting as c,evaluation_scope as scope
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


def recompute(*,manifest_bytes,request_bytes,population_bytes,evidence_by_candidate,
              global_bytes,local_bytes,snapshot_bytes,now_epoch,purpose):
    need(purpose in ('PREPARE','RESERVE'),'COLD_PHASE_ONLY_NOT_NATIVE_GO')
    m,msha=_read(manifest_bytes);r,rsha=_read(request_bytes);g,gsha=_read(global_bytes);l,lsha=_read(local_bytes)
    snapshot,_=_read(snapshot_bytes);c.checked(g);scope.validate_pair(g,l)
    request_identity(m,r,msha)
    need(not g['unknown_effects'] and not g['pending_updates'],'GLOBAL_UNKNOWN_OR_PENDING_FENCE')
    need(snapshot.get('schema')==original.SNAPSHOT and snapshot.get('campaign_run_uuid')==scope.CAMPAIGN
         and snapshot.get('original_T0_epoch')==c.old.T0 and snapshot.get('hard_close_epoch')==c.old.HARD_CLOSE
         and snapshot.get('global_ledger_sha256')==gsha and snapshot.get('LOCAL_ledger_sha256')==lsha,
         'CURRENT_EXACT_RAW_LEDGER_AND_ORIGINAL_CLOCK_BINDING')
    _fresh(snapshot.get('observed_epoch'),now_epoch,120);_fresh(r.get('created_epoch'),now_epoch,120)
    need(c.old.T0<=now_epoch<c.old.STOP_NEW,'ORIGINAL_STOP_NEW_NO_EXTENSION')
    need(not any(x['status'] not in ('NATIVE_TERMINAL','NATIVE_QSUB_REJECTED_NO_JOB') for x in g['PBS_submissions'])
         and not any(x['status']!=c.STATES[3] for x in g.get(c.FIELD,{}).values())
         and r['request_id'] not in g.get(c.FIELD,{})
         and all(x.get('request_id')!=r['request_id'] for x in g['PBS_submissions']),
         'ONE_NEW_INACTIVE_REQUEST_NOT_REISSUING_UNKNOWN')
    work=remaining(population_bytes,evidence_by_candidate=evidence_by_candidate)
    work=stage.first_stage(work,manifest=m,current_request=r,global_ledger=g,purpose='PREPARE')
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
    caps=original.adopted_caps(g);dimensions={}
    for key in CAPS:
        total=used[key]+reserved[key]+future[key]+MATRIX[key]+_vector(FINAL)[key]
        dimensions[key]=dict(cap=caps[key],used=used[key],reserved=reserved[key],
            remaining_stage_upper=future[key],matrix=MATRIX[key],final=_vector(FINAL)[key],
            projected=total,shortfall=max(0,total-caps[key]))
    clock=stage.clock(work,now_epoch=now_epoch)
    shortfalls={k:x['shortfall'] for k,x in dimensions.items() if x['shortfall']}
    skeleton={k:v for k,v in r.items() if k!='prequeue_future_admission_sha256'}
    context=dict(manifest_sha256=msha,request_skeleton_sha256=digest(skeleton),global_sha256=gsha,
        LOCAL_sha256=lsha,resource_snapshot_sha256=hashlib.sha256(snapshot_bytes).hexdigest(),
        work_sha256=digest(work),future_upper=future,dimensions=dimensions)
    context_sha=digest(context)
    need(not shortfalls and clock['both_stage_panels_fit_original_physics_window']
         and clock['last_stage_submission_fits_original_stop_new'],
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
