"""Byte-bound V4 DEV resource recomputation, not native or scientific GO.

Uses the existing global and LOCAL campaign ledgers. The sole owner must collect
the actual fresh Core/LOCAL projections and complete historical inventory.
Project/quota/lease/source/native qualification remain separate. No caller cap
override, retry, transport, lock, or write is performed by this module.
"""
import hashlib
import json
from copy import deepcopy

from scripts.pi0294.dec1.storage_expansion import global_cap_from_ledger
from . import evaluation_accounting as accounting
from . import evaluation_platform as platform
from .evaluation_work import remaining,MATRIX
from .future_budget import CAPS,FINAL,_read,_vector,_fresh,_number
from .resources import pbs_limit

SNAPSHOT='PI0294_V4_ALL_SITE_RESOURCE_SNAPSHOT_V1'
CAMPAIGN='20261018T182956Z-localpc-pv1'
PURPOSES=('PREPARE','RESERVE','NATIVE_PREMODEL')
INDEX='plans/V4_DEV_HISTORY_INPUTS.json'
FRESH_SNAPSHOT='evidence/V4_DEV_FRESH_SNAPSHOT.private.json'
FRESH_LOCAL='evidence/V4_DEV_FRESH_LOCAL_LEDGER.private.json'
FRESH_SITE='evidence/V4_DEV_PREQUEUE_SITE.private.json'


def need(ok,reason):
    if not ok:raise PermissionError('PV1_V4_DEV_ADMISSION_'+reason)


def canonical(value):
    return (json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def adopted_caps(global_ledger):
    """Existing exact adopted markers only; proposals cannot change these."""
    accounting.checked(global_ledger)
    caps=dict(CAPS)
    caps.update(PBS_submissions=pbs_limit(global_ledger),
        Hopper_storage_bytes=global_cap_from_ledger(global_ledger))
    from . import evaluation_scope as scope
    if scope.FIELD in global_ledger:caps.update(scope.caps(global_ledger))
    from . import successor_cd as successor
    if successor.FIELD in global_ledger:caps.update(successor.validated(global_ledger)['authorized_storage_caps'])
    from scripts.pi0294.generalization import amd_adoption as gen1
    if gen1.FIELD in global_ledger:caps.update(gen1.validate(global_ledger)['caps'])
    if 'ARX5_minimal_first512_seven_cap_authority' in global_ledger:
        from scripts.pi0294.generalization.minimal_adoption import global_limits
        caps=global_limits(global_ledger,caps)
    return caps


def resource_floor(global_ledger,local):
    """Minimum projections from actual paid/reserved counters, not headroom."""
    need(local.get('run_uuid')==CAMPAIGN and local.get('status')=='ACTIVE_CONTROL_PLANE'
         and local.get('T0_epoch')==platform.T0
         and local.get('hard_closeout_epoch')==platform.HARD_CLOSE
         and local.get('unknown_effects')==[] and local.get('Hopper_unknown_effects',[])==[]
         and local.get('pending_updates')==[] and local.get('pending_queries')==[],
         'SAME_ORIGINAL_LOCAL_CAMPAIGN_WITHOUT_UNKNOWN_EFFECTS')
    used=_vector(global_ledger['used']);reserved=_vector(global_ledger['reserved'])
    cpu=local['CPU_s'];held=local['CPU_reserved_s']
    for key,site in (('Core_CPU_s','CoreLab'),('Hopper_CPU_s','Hopper'),('local_CPU_s','local_PC')):
        used[key]=_number(cpu[site]);reserved[key]=_number(held[site])
    used['Hopper_CPU_s']=max(used['Hopper_CPU_s'],_number(global_ledger['CPU_s']))
    reserved['Hopper_CPU_s']=max(reserved['Hopper_CPU_s'],_number(global_ledger.get('CPU_reserved_s',0)))
    used['H200_s']=max(used['H200_s'],_number(local['GPU_allocated_s']))
    reserved['H200_s']=max(reserved['H200_s'],_number(local['GPU_reserved_s']))
    used['PBS_submissions']=max(used['PBS_submissions'],len(local['PBS_submissions']))
    need(type(local['PBS_submissions']) is list,'LOCAL_COMPLETE_PBS_INVENTORY')
    for key,field in (('optimizer_updates','optimizer_updates'),('neural_queries','neural_queries'),
            ('episode_starts','episode_starts'),('engine_starts','engine_starts')):
        used[key]=max(used[key],_number(local[field],integer=True))
    ticks=local['physics_ticks']
    need(type(ticks) is dict,'ALL_ACTUAL_AND_OLD_FINANCIAL_PHYSICS_COUNTERS')
    used['physics_ticks']=max(used['physics_ticks'],sum(_number(x,integer=True) for x in ticks.values()))
    used['transfer_bytes']=_number(local['network_transfer_accounted_bytes'],integer=True)
    used['Hopper_storage_bytes']=max(_number(global_ledger.get('new_storage_current_bytes',0),integer=True),
        _number(global_ledger['new_storage_peak_bytes'],integer=True))
    reserved['Hopper_storage_bytes']=_number(global_ledger.get('storage_reserved_bytes',0),integer=True)
    used['Core_storage_bytes']=max(used['Core_storage_bytes'],
        _number(local.get('Core_native_GPU_current_storage_bytes',0),integer=True))
    # These separate namespaces are already represented in the original LOCAL
    # campaign. Sum non-CPU holds; CPU_s/CPU_reserved_s may already aggregate them.
    pending={}
    names=('source_cpu_reserved','target_zero_counter_reserved','target_sdk_client_reserved','target_native_IO_reserved',
           'Core_native_GPU_reserved')
    mapping=dict(CoreLab_CPU_s='Core_CPU_s',local_CPU_s='local_CPU_s',episode_starts='episode_starts',
        physics_ticks='physics_ticks',constructors='engine_starts',CoreLab_output_bytes='Core_storage_bytes',
        local_output_bytes='local_storage_bytes',transfer_bytes='transfer_bytes')
    for name in names:
        for key,value in local.get(name,{}).items():
            need(key in mapping,'KNOWN_EXISTING_CONTROL_RESOURCE_DIMENSION')
            target=mapping[key]
            pending[target]=pending.get(target,0)+_number(value,integer=not target.endswith('CPU_s'))
    for key,value in pending.items():reserved[key]=max(reserved[key],value)
    return used,reserved


def recompute(*,manifest_bytes,request_bytes,population_bytes,evidence_by_candidate,
              global_bytes,local_bytes,snapshot_bytes,now_epoch,purpose,queue_proof=None):
    """Recompute all remaining regular work and exact current GPU-side credit.

    PREPARE returns the prequeue context digest; the owner adds it to the sealed
    request. RESERVE requires that digest and no existing request. Neither path
    writes or submits. NATIVE_PREMODEL needs an already bound R job and a fresh
    complete queue; unbound UNKNOWN intents are never made retryable here.
    No control prepayment credit is invented; such costs remain in the full
    conservative future until a separate exact paid record can be joined.
    """
    need(purpose in PURPOSES,'EXPLICIT_CURRENT_PHASE')
    manifest,manifest_sha=_read(manifest_bytes);request,request_sha=_read(request_bytes)
    global_ledger,global_sha=_read(global_bytes);local,local_sha=_read(local_bytes)
    from . import evaluation_scope as scope
    if scope.FIELD in global_ledger or scope.FIELD in local:scope.validate_pair(global_ledger,local)
    snapshot,snapshot_sha=_read(snapshot_bytes)
    platform.request_identity(manifest,request,run_uuid=request['run_uuid'],manifest_sha256=manifest_sha)
    accounting.checked(global_ledger)
    need(global_ledger['unknown_effects']==[] and global_ledger['pending_updates']==[],
         'NO_NEW_EFFECT_DURING_GLOBAL_UNKNOWN_OR_PENDING')
    need(global_ledger['campaign_run_uuid']==CAMPAIGN
         and snapshot.get('schema')==SNAPSHOT and snapshot.get('campaign_run_uuid')==CAMPAIGN
         and snapshot.get('original_T0_epoch')==platform.T0
         and snapshot.get('hard_close_epoch')==platform.HARD_CLOSE
         and snapshot.get('global_ledger_sha256')==global_sha and snapshot.get('LOCAL_ledger_sha256')==local_sha,
         'FRESH_RAW_LEDGER_BINDING_NOT_PASS_FLAGS_OR_RESET_CLOCK')
    _fresh(snapshot.get('observed_epoch'),now_epoch,120)
    need(platform.T0<=now_epoch<platform.STOP_PHYSICS,'ORIGINAL_EVALUATION_CUTOFF')
    if purpose!='NATIVE_PREMODEL':
        _fresh(request['created_epoch'],now_epoch,120)
        need(now_epoch<platform.STOP_NEW,'ORIGINAL_NEW_REQUEST_CUTOFF')
    work=remaining(population_bytes,evidence_by_candidate=evidence_by_candidate)
    staged=scope.FIELD in global_ledger
    if staged:
        from . import evaluation_stage as execution_stage
        work=execution_stage.first_stage(work,manifest=manifest,current_request=request,
            global_ledger=global_ledger,purpose=purpose)
    inventory=snapshot.get('primary_attempt_inventory')
    need(type(inventory) is dict and set(inventory)==set(platform.ENDPOINTS)
         and all(type(ids) is list and len(ids)==len(set(ids))
             and set(ids)==set(work['histories'][candidate]['all_actual_attempts'])
             for candidate,ids in inventory.items()),'EVERY_ORIGINAL_ATTEMPT_IN_COMPLETE_OWNER_INVENTORY')
    candidate=request['evaluation_candidate'];noise=request.get('noise_bank')
    jobs=[j for j in work['jobs'] if j['candidate']==candidate and j['noise_bank']==noise]
    need(len(jobs)==1,'ONE_ORIGINAL_UNFINISHED_NOISE_PANEL')
    job=jobs[0];prior=work['histories'][candidate]['prior_rows']
    need(request.get('remaining_conditions')==job['conditions']
         and request.get('scientific_admission_version')==platform.VERSION
         and request.get('walltime_s')==job['walltime_s']
         and request.get('additional_output_upper_bytes')==job['GPU_site_output_upper_bytes']
         and request.get('planned_panel_sha256')==digest(job)
         and request.get('prior_outcomes_sha256')==digest(prior)
         and request.get('new_bank_manifest_sha256')==work['new_bank_manifest_sha256']
         and request.get('CPU_upper_s')==job['process_CPU_envelope_s'],
         'REQUEST_MATCHES_WHOLE_REMAINING_ORIGINAL_PANEL_AND_PRIOR_BYTES')
    platform.process_cpu_limit(request,80000)
    from scripts.pi0294.dec1.a40_allocation import CAPACITY_PROFILES
    need(request.get('allowed_native_capacity_profiles')==[
        dict(ncpus=c,memory_bytes=m) for c,m in CAPACITY_PROFILES],
        'ORIGINAL_EXPLICIT_ALLOWED_CAPACITY_PROFILES')
    used,reserved=_vector(snapshot['used']),_vector(snapshot['reserved'])
    floor_used,floor_reserved=resource_floor(global_ledger,local)
    need(all(used[k]>=floor_used[k] and reserved[k]>=floor_reserved[k] for k in CAPS),
         'PROJECTION_CANNOT_DROP_PAID_OR_RESERVED_NATIVE_COSTS')
    need(type(snapshot.get('Core_storage_upper_bytes')) is int
         and type(snapshot.get('LOCAL_storage_upper_bytes')) is int
         and snapshot['Core_storage_upper_bytes']>0 and snapshot['LOCAL_storage_upper_bytes']>0
         and used['Core_storage_bytes']>=snapshot['Core_storage_upper_bytes']
         and used['local_storage_bytes']>=snapshot['LOCAL_storage_upper_bytes'],
         'INCLUSIVE_OWNER_STORAGE_UPPERS_NOT_ZERO_OR_WEIGHT_ALIAS_REFUND')
    caps=adopted_caps(global_ledger)
    credit=_vector({});current=[r for r in global_ledger['PBS_submissions'] if r.get('request_id')==request['request_id']]
    if purpose=='NATIVE_PREMODEL':
        need(len(current)==1,'ONE_ALREADY_BOUND_CURRENT_REQUEST')
        row=accounting.validate_item(current[0])
        need(row['status'] in accounting.STATES[1:3] and row.get('job_id')
             and row['request_sha256']==request_sha and row['child_run_uuid']==request['run_uuid']
             and global_ledger['PBS_submissions'].index(row)+1==request['submit_count']
             and all(row.get(k)==request.get(k) for k in ('source_sha','run_manifest_sha256','submission_nonce',
                 'stage','evaluation_candidate','checkpoint_manifest_sha256','statistics_sha256','noise_bank',
                 'remaining_conditions','planned_panel_sha256','prior_outcomes_sha256','prequeue_future_admission_sha256')),
             'EXACT_BOUND_LIVE_REQUEST_NOT_UNBOUND_UNKNOWN_OR_REASSIGNED_WORK')
        if staged:
            need(all(row.get(k)==request.get(k) for k in ('evaluation_execution_stage',
                'evaluation_stage_plan_sha256','evaluation_stage_card_sha256','control_prepayment_id')),
                'SAME_BOUND_V4A_STAGE_AND_ONCE_CONTROL_PAYMENT')
        need(queue_proof is not None,'ACTUAL_COMPLETE_CURRENT_QUEUE_FOR_BOUND_CREDIT')
        from scripts.pi0294.dec1.vanda_campaign import queue_rows,classify
        rows=queue_rows(queue_proof,now_epoch=now_epoch)
        campaign,_,unknown=classify(global_ledger,rows)
        need(not unknown and len(campaign)==1 and campaign[0]['job_id']==row['job_id']
             and campaign[0]['state']=='R','ONE_KNOWN_RUNNING_CURRENT_CHAIN')
        jobs_by_id=json.loads(queue_proof['qstat']['stdout'])['Jobs'];native=jobs_by_id[row['job_id']]
        host=native['exec_host'].split('+')[0].split('/')[0]
        facts=platform.native_job(request,now=now_epoch,user='e1192548',hostname=host,
            job_id=row['job_id'],queue_proof=queue_proof)
        need(now_epoch<facts['allocation_end_epoch']-2700,'ORIGINAL_NATIVE_CLOSEOUT_RESERVE')
        charged,_,cpu=accounting.progress(row)
        need(row['reserved_neural_queries']+charged==row['query_limit']
             and reserved['H200_s']>=28860 and reserved['Hopper_CPU_s']>=80000-cpu
             and reserved['Hopper_storage_bytes']>=row['additional_output_upper_bytes'],
             'CURRENT_CREDIT_EXISTS_IN_ACTUAL_ESCROW_AND_DEBITS')
        credit.update(H200_s=28860,PBS_submissions=1,neural_queries=row['query_limit'],
            Hopper_CPU_s=80000,Hopper_storage_bytes=row['additional_output_upper_bytes'])
    else:
        need(not current and all(r['status'] in (accounting.TERMINAL,'NATIVE_QSUB_REJECTED_NO_JOB')
             for r in global_ledger['PBS_submissions'])
             and request['submit_count']==len(global_ledger['PBS_submissions'])+1,
             'NEW_UNIQUE_REQUEST_ONLY_NO_UNKNOWN_WRITE_RETRY_OR_SECOND_OWNER')
    from . import evaluation_control as control
    control_credit=control.credit(global_ledger,local,request,work)
    for key,value in control_credit.items():credit[key]+=value
    future=_vector(work['resources'])
    need(all(credit[k]<=future[k] and credit[k]<=used[k]+reserved[k] for k in CAPS),
         'ONCE_CREDIT_CANNOT_EXCEED_EXISTING_OR_REMAINING_WORK')
    rows={}
    for key in CAPS:
        total=used[key]+reserved[key]+future[key]-credit[key]+MATRIX[key]+_vector(FINAL)[key]
        rows[key]=dict(cap=caps[key],used=used[key],reserved=reserved[key],remaining_S2=future[key],
            exact_current_credit=credit[key],matrix=MATRIX[key],final=_vector(FINAL)[key],
            projected=total,shortfall=max(0,total-caps[key]))
    shortfalls={k:r['shortfall'] for k,r in rows.items() if r['shortfall']}
    # Exclude only this field to avoid request -> proof -> request hash cycles.
    # Raw full request SHA is still recorded separately and bound at native R.
    skeleton={k:v for k,v in request.items() if k!='prequeue_future_admission_sha256'}
    context=dict(schema='PI0294_V4_PREQUEUE_RESOURCE_CONTEXT_V1',request_skeleton_sha256=digest(skeleton),
        snapshot_sha256=snapshot_sha,global_ledger_sha256=global_sha,LOCAL_ledger_sha256=local_sha,
        manifest_sha256=manifest_sha,remaining_work_sha256=digest(work),adopted_caps=caps)
    context_sha=digest(context)
    if purpose=='RESERVE':need(request.get('prequeue_future_admission_sha256')==context_sha,
        'EXACT_RECOMPUTED_PREQUEUE_CONTEXT_NO_CIRCULAR_OR_STALE_PROOF')
    stage_clock=(execution_stage.clock(work,now_epoch=now_epoch,
        current_native_facts=facts if purpose=='NATIVE_PREMODEL' else None) if staged else {})
    return dict(schema='PI0294_V4_RECOMPUTED_DEV_RESOURCE_ENVELOPE_V1',
        execution_stage=work.get('execution_stage','LEGACY_FULL_WORK_DIAGNOSTIC_NOT_V4A_DISPATCH'),
        stage_clock=stage_clock,current_stage_clock_fits=all(stage_clock.get(k,True) for k in
            ('both_stage_panels_fit_original_physics_window','last_stage_submission_fits_original_stop_new')),
        status='PASS_RECOMPUTED_COST_ENVELOPE_NOT_NATIVE_PERMISSION' if not shortfalls else 'RESOURCE_SHORTFALL_NO_NEW_INTENT',
        purpose=purpose,request_sha256=request_sha,request_id=request['request_id'],
        submission_nonce=request['submission_nonce'],observed_epoch=now_epoch,work_id=job['work_id'],
        global_ledger_sha256=global_sha,LOCAL_ledger_sha256=local_sha,snapshot_sha256=snapshot_sha,
        prequeue_context_sha256=context_sha,remaining_work_sha256=digest(work),dimensions=rows,
        shortfalls=shortfalls,all_remaining_costs_fit=not shortfalls,
        current_credit=credit,control_prepayment_credit_applied=any(control_credit.values()),
        control_prepayment_credit=control_credit,
        core_LOCAL_storage_projections_require_actual_owner_readback=True,
        native_or_scientific_admission=False,original_matrix_and_final_protected=True,
        original_T0_epoch=platform.T0,hard_close_epoch=platform.HARD_CLOSE,
        full_remaining_wall_upper_s=sum(j['walltime_s']+j['accounting_pad_s'] for j in work['jobs']),
        all_remaining_wall_upper_fits_current_window=sum(j['walltime_s']+j['accounting_pad_s'] for j in work['jobs'])<=platform.STOP_PHYSICS-now_epoch,
        combined_matrix_and_S2_wall_upper_s=future['H200_s']+MATRIX['H200_s'],
        combined_matrix_and_S2_wall_upper_fits_current_window=future['H200_s']+MATRIX['H200_s']<=platform.STOP_PHYSICS-now_epoch,
        current_job_wall_fits_current_window=request['walltime_s']+60<=platform.STOP_PHYSICS-now_epoch,
        new_training_updates=0,new_native_writes=0)


def reserve_proposal(global_bytes,*,recompute_inputs):
    """Pure original-global proposal; caller still owns lock/once IO/dispatch.

    Recompute from raw inputs, never trust an incoming PASS summary. A shortfall
    is rejected before creating a new intent; existing UNKNOWN is not retried.
    """
    need(recompute_inputs.get('global_bytes')==global_bytes,'SAME_EXACT_BEFORE_GLOBAL_BYTES')
    proof=recompute(**dict(recompute_inputs,purpose='RESERVE'))
    need(proof['all_remaining_costs_fit'] and proof['current_job_wall_fits_current_window']
         and proof['current_stage_clock_fits'],
         'COMPLETE_REMAINING_RESOURCES_AND_CURRENT_STAGE_CLOCK_BEFORE_INTENT')
    old,_=_read(global_bytes);request,_=_read(recompute_inputs['request_bytes'])
    row={k:request[k] for k in ('request_id','submission_nonce','source_sha','run_manifest_sha256',
        'campaign_run_uuid','stage','project','job_name','evaluation_candidate','checkpoint_manifest_sha256',
        'parent_checkpoint_manifest_sha256','statistics_sha256','created_epoch','noise_bank','remaining_conditions',
        'cold_input_audit_sha256','planned_panel_sha256','prior_outcomes_sha256','checkpoint_reuse_proof_sha256',
        'prequeue_future_admission_sha256','additional_output_upper_bytes','CPU_upper_s','CPU_envelope_profile',
        'declared_process_CPU_limit_s','original_T0_epoch','hard_close_epoch')}
    row.update(request_sha256=proof['request_sha256'],child_run_uuid=request['run_uuid'],
        owner_user='e1192548',authority_release=platform.AUTHORITY,native_platform=platform.PLATFORM,
        platform_evaluation_version=platform.VERSION,scientific_admission_version=platform.VERSION,
        physical_GPU_kind='NVIDIA_A40',physical_GPU_count=1,legacy_GPU_seconds_field_is_projection=True,
        status=accounting.STATES[0],job_id=None,reserved_H200_s=28800,accounting_pad_s=60,
        query_limit=80*request['remaining_conditions']+17,reserved_neural_queries=80*request['remaining_conditions']+17,
        optimizer_update_limit=0,reserved_optimizer_updates=0,charges_DEC1_subcaps=False,new_robot_physics=0)
    for key in ('evaluation_execution_stage','evaluation_stage_plan_sha256','evaluation_stage_card_sha256','control_prepayment_id'):
        if key in request:row[key]=request[key]
    accounting.validate_item(row)
    proposal=deepcopy(old);proposal['PBS_submissions'].append(row);proposal['used']['PBS_submissions']+=1
    proposal['reserved']['H200_s']+=28860;proposal['reserved']['neural_queries']+=row['query_limit']
    proposal['CPU_reserved_s']=proposal.get('CPU_reserved_s',0)+80000
    proposal['storage_reserved_bytes']=proposal.get('storage_reserved_bytes',0)+row['additional_output_upper_bytes']
    accounting.checked(proposal)
    return dict(proposed_global=proposal,recomputed_proof=proof,revision_increment_or_IO_performed=False,
        qsub_performed=False,native_or_scientific_admission=False)


def file_inputs(root,request_path,*,now_epoch):
    """Read actual registered-root inputs; original evidence must be presealed.

    Fresh sole-owner projections remain outside immutable scientific inputs,
    as in the existing future-admission bridge. The original native scope must
    still be validated by the calling owner before persisting any proposal.
    """
    from pathlib import Path
    from .future_admission import _file
    from scripts.pi0294.dec1.source_cpu_once_io import _file as ledger_file
    root,request_path=Path(root),Path(request_path)
    need(root.is_absolute() and request_path.is_absolute()
         and request_path.parent.parent.resolve()==(root/'pbs').resolve()
         and request_path.name=='REQUEST.json','ONE_REGISTERED_REQUEST_PATH')
    manifest_raw=_file(root,'RUN_MANIFEST.json');manifest,_=_read(manifest_raw)
    request_raw=_file(root,str(request_path.relative_to(root)));request,_=_read(request_raw)
    need(request_path.parent.name=='alloc-'+str(request['submit_count']).zfill(2)
         and manifest['run_uuid']==root.name,'ACTUAL_REQUEST_NAMESPACE')
    sealed=manifest['execution']['sealed_inputs']
    def original(name):
        data=_file(root,name);pin=hashlib.sha256(data).hexdigest()
        need(sealed.get(name)==pin and request['sealed_inputs'].get(name)==pin,
             'ORIGINAL_HISTORY_FILE_SEALED_IN_MANIFEST_AND_REQUEST:'+name)
        return data
    index_raw=original(INDEX);index,_=_read(index_raw)
    need(index.get('schema')=='PI0294_V4_PRIMARY_HISTORY_INPUT_INDEX_V1'
         and index.get('campaign_run_uuid')==CAMPAIGN
         and index.get('authority_release')==platform.AUTHORITY
         and set(index.get('evidence',{}))==set(platform.ENDPOINTS),
         'ORIGINAL_EXPLICIT_ALL_CANDIDATE_HISTORY_INDEX')
    population=original(index['population'])
    evidence={};names=[INDEX,index['population']]
    for candidate,items in index['evidence'].items():
        need(type(items) is list,'ORIGINAL_EVIDENCE_PATH_RECORDS')
        evidence[candidate]=[]
        for record in items:
            need(type(record) is dict and {'adjudication','result','audit','model_ready'}<=set(record)
                 and set(record)<= {'adjudication','result','audit','model_ready','causal_chain','engineering_diagnosis'},
                 'EXACT_PRIMARY_EVIDENCE_KINDS')
            names.extend(record.values());evidence[candidate].append({kind:original(name) for kind,name in record.items()})
    need(len(names)==len(set(names)),'NO_ALIASED_OR_SELF_REFERENTIAL_HISTORY_FILES')
    global_bytes=ledger_file(root.parent.parent/'HOPPER_RESOURCE_LEDGER.json')
    from . import evaluation_scope as scope,evaluation_stage as stage
    if scope.FIELD in _read(global_bytes)[0]:stage.sealed(root,manifest,request)
    return dict(manifest_bytes=manifest_raw,request_bytes=request_raw,population_bytes=population,
        evidence_by_candidate=evidence,global_bytes=global_bytes,
        local_bytes=_file(root,FRESH_LOCAL),snapshot_bytes=_file(root,FRESH_SNAPSHOT),now_epoch=now_epoch)


def prequeue_site(site_bytes,*,request_bytes,global_bytes,resource_proof,now_epoch,independent_job_bytes=None):
    """Bound current real-site inputs before the once-only metadata effect."""
    from datetime import date,datetime,timedelta,timezone
    from scripts.pi0294.dec1.vanda_campaign import queue_rows,classify
    site,_=_read(site_bytes);request,request_sha=_read(request_bytes);g,_=_read(global_bytes)
    need(site.get('schema')=='PI0294_V4_NATIVE_PREQUEUE_SITE_V1'
         and site.get('source')=='VANDA_LOGIN_NATIVE_AMGR_QUEUE_AND_QUOTA_READBACK'
         and site.get('request_id')==request['request_id'] and site.get('submission_nonce')==request['submission_nonce']
         and site.get('request_sha256')==request_sha
         and resource_proof['request_sha256']==request_sha and resource_proof['all_remaining_costs_fit'] is True,
         'EXACT_OWNER_NATIVE_SITE_INPUTS_NOT_A_GENERIC_PASS')
    _fresh(site['observed_epoch'],now_epoch,120)
    rows=queue_rows(site['queue_proof'],now_epoch=now_epoch)
    campaign,outside,unknown=classify(g,rows)
    from .evaluation_account_jobs import checked_outside
    independent=checked_outside(rows,outside,independent_job_bytes)
    need(not campaign and not unknown,'COMPLETE_KNOWN_INACTIVE_QUEUE_BEFORE_NEW_RESERVATION')
    project=site['project'];today=datetime.fromtimestamp(now_epoch,timezone(timedelta(hours=8))).date()
    wall=request['walltime_s']+60
    capacity=max(x['ncpus'] for x in request['allowed_native_capacity_profiles'])
    need(project.get('name')==request['project'] and project.get('active') is True
         and 'e1192548' in project.get('users',[]) and 'stdct-mgmt-02' in project.get('cluster',[])
         and date.fromisoformat(project['start_date'])<=today<=date.fromisoformat(project['end_date'])
         and _number(site['balance_GPU_hours'])>=wall/3600
         and _number(site['balance_CPU_hours'])>=capacity*wall/3600,
         'NATIVE_ACTIVE_PROJECT_DATE_MEMBERSHIP_AND_DECLARED_CAPACITY_BALANCE')
    queue=site['effective_queue_limit']
    need(queue.get('source')=='VANDA_NATIVE_QSTAT_EFFECTIVE_QUEUE_LIMITS'
         and queue.get('requested_queue')=='auto' and queue.get('destination')=='batch_gpu'
         and _number(queue['maximum_wall_s'],integer=True)>=request['walltime_s'],
         'ORIGINAL_AUTO_AND_ACTUAL_NATIVE_QUEUE_LIMIT')
    _fresh(queue['observed_epoch'],now_epoch,120)
    quota=site['quota'];_fresh(quota['observed_epoch'],now_epoch,900)
    dim=resource_proof['dimensions']['Hopper_storage_bytes']
    required=dim['reserved']+dim['remaining_S2']-dim['exact_current_credit']+dim['matrix']+dim['final']
    need(quota.get('user')=='e1192548' and quota.get('scope')=='OWN_PROJECT_AND_FILESYSTEM_MINIMUM'
         and _number(quota['available_bytes'],integer=True)>=required,
         'ACTUAL_QUOTA_FOR_ALL_REMAINING_OUTPUT_AND_ORIGINAL_FINAL_PROTECTION')
    return dict(site_input_sha256=hashlib.sha256(site_bytes).hexdigest(),
        project_and_queue_and_quota_checked=True,independent_outside_jobs=independent,native_or_scientific_admission=False)
