"""Pure native admission for source-only long training on Vanda.

This is not a launcher, a model loader, or a replacement for the original
H200 admission. Native evidence and both-site reservations must precede it.
"""
from datetime import date, datetime, timedelta, timezone
import math
import re
import uuid

from scripts.pi0294.e0_scope import pbs_epoch
from scripts.pi0294.pv1.resources import admit_resources
from .authority import AUTHORITY, PARENT_AUTHORITY, PARENT, R0, BANK, HARD_CLOSE
from .alternate_accounting import digest
from .dual_platform import GRANT_SHA, grant
from .source_admission import VERSION as SCIENCE_VERSION

VERSION = 'VAP1_VANDA_A40_SOURCE_LONG_TRAIN_V1'
STATUS = 'PASS_DEC1_OWN_NATIVE_A40_SOURCE_TRAIN_ALLOCATION'
PLATFORM = 'VANDA_A40'
T0 = 1792348196
PAD = 60
GIB = 1024**3
CAPACITY_PROFILES = ((4,32*GIB), (36,240*GIB))
BOUNDED_CPU_PROFILE = 'A40_EIGHT_HOUR_PROCESS_CPU_80000_V1'
BOUNDED_CPU_ACCOUNTING_MARGIN = 240


def fail(reason):
    raise PermissionError('VAP1_A40_TRAIN_'+reason)


def number(value, *, integer=False, positive=False):
    if (type(value) not in (int,float) or not math.isfinite(value)
            or value < 0 or integer and type(value) is not int or positive and value <= 0):
        fail('FINITE_NONNEGATIVE_EXACT_RESOURCE')
    return value


def cpu_envelope(value, wall):
    """A stricter cold process limit, not a smaller claimed site allocation."""
    original=4*(wall+PAD)+120
    profile=value.get('CPU_envelope_profile')
    if profile is None:return original
    from .segmented_training import CPU_PROFILE,enabled
    if profile==CPU_PROFILE:
        if not enabled(value) or wall!=22500:fail('EXACT_AUTHORIZED_SEGMENT_CPU_PROFILE')
        return 54000
    if profile!=BOUNDED_CPU_PROFILE or wall!=28800 or 80000>=original:
        fail('EXACT_STRICTER_EIGHT_HOUR_PROCESS_CPU_PROFILE')
    return 80000


def process_cpu_limit(request):
    upper=request['CPU_upper_s']
    if request.get('CPU_envelope_profile') is None:return upper
    if upper!=cpu_envelope(request,request['walltime_s']):fail('EXACT_ENFORCED_CPU_ACCOUNTING_UPPER')
    # Leave room for the native wrapper/observer and terminal accounting pad.
    return upper-BOUNDED_CPU_ACCOUNTING_MARGIN


def memory_bytes(text):
    match = re.fullmatch(r'([1-9][0-9]*)([kmgt]?)b', str(text).lower())
    if not match:
        fail('EXACT_NATIVE_MEMORY_UNIT')
    return int(match[1])*1024**('kmgt'.index(match[2])+1 if match[2] else 0)


def select_profile(limits, request):
    parts = limits.get('select','').split(':')
    if not parts or parts[0] != '1' or '+' in limits.get('select',''):
        fail('ONE_NATIVE_SELECT_CHUNK')
    fields = {}
    for part in parts[1:]:
        pair = part.split('=',1)
        if len(pair)!=2 or pair[0] in fields:
            fail('EXACT_NATIVE_SELECT_FIELDS')
        fields[pair[0]]=pair[1]
    try:
        cpus=int(fields['ncpus']);ngpus=int(fields['ngpus']);mem=memory_bytes(fields['mem'])
    except (KeyError, ValueError):
        fail('NATIVE_SELECT_CPU_GPU_MEMORY_PRESENT')
    declared = request['allowed_native_capacity_profiles']
    if (declared != [dict(ncpus=c,memory_bytes=m) for c,m in CAPACITY_PROFILES]
            or (cpus,mem) not in CAPACITY_PROFILES or ngpus != 1
            or limits.get('ncpus') != cpus or int(limits.get('ngpus',0)) != 1
            or memory_bytes(limits.get('mem','')) != mem
            or fields.get('gpu_model', 'A40') != 'A40'):
        fail('COLD_DECLARED_A40_CAPACITY_PROFILE_NOT_ARBITRARY_REWRITE')
    return cpus,mem


def request_scope(request, *, now):
    r=request;block=r.get('training_block',{})
    from .segmented_training import enabled,limits
    segment=enabled(r)
    from .storage_expansion import enabled as storage_enabled
    storage_enabled(r)
    from . import zero_update_recovery as zero
    if zero.enabled(r):zero.caps(r)
    stop=limits(r)[1] if segment else 4096
    if (r.get('schema')!='PI0294_PV1_NATIVE_ALLOCATION_REQUEST_V1'
            or r.get('platform_training_version')!=VERSION or r.get('native_platform')!=PLATFORM
            or r.get('authority_release')!=PARENT_AUTHORITY or r.get('authority_amendment')!=AUTHORITY
            or r.get('scientific_admission_version')!=SCIENCE_VERSION
            or r.get('campaign_run_uuid')!='20261018T182956Z-localpc-pv1'
            or r.get('original_T0_epoch')!=T0 or r.get('hard_close_epoch')!=HARD_CLOSE
            or r.get('stage')!='P2_PRIMARY' or r.get('project')!='CFP04-CF-007'
            or r.get('job_name')!='pi0294_dec1_a40_train'
            or r.get('data_bank_status')!='SEALED_FULL_SUCCESS_OFFICIAL_CODEC_ADMITTED'
            or r.get('parent_checkpoint_manifest_sha256')!=PARENT
            or r.get('statistics_sha256')!=R0 or r.get('new_bank_manifest_sha256')!=BANK
            or r.get('numerical_mode')!='DETERMINISTIC_CUDA_MATH_SDP_V1'
            or r.get('active_cpu_limit')!=4 or r.get('requested_ncpus')!=4
            or r.get('requested_memory_bytes')!=32*GIB
            or r.get('compatibility_queries')!=17 or r.get('TRAIN_probe_queries')!=12
            or r.get('new_robot_physics')!=0 or r.get('LOCKBOX_opened') is not False
            or block.get('arm') not in ('A_CONTROL','B_COVERAGE')
            or block.get('replica') is not False or block.get('same_allocation_1024_probes') is not True
            or block.get('stop_after_step')!=stop or block.get('probe_only',False)
            or not T0<=r.get('created_epoch',0)<T0+144*3600
            or not T0<=now<T0+156*3600):
        fail('EXPLICIT_SOURCE_ONLY_ORIGINAL_PAIR_CLOCK_RECIPE_AND_PLATFORM')
    from . import b_heartbeat_recovery as b
    new_b=b.enabled(r)
    if new_b:b.lineage(r)
    from . import b_saved_state_reuse as saved
    saved.request(r)
    if (type(r.get('submit_count')) is not int or not 1<=r['submit_count']<=(25 if new_b else 24)
            or re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,127}',r.get('run_uuid','')) is None
            or r['run_uuid']==r['campaign_run_uuid']):
        fail('EXACT_CHILD_RUN_AND_GLOBAL_SUBMISSION_ORDINAL')
    for key in ('request_id','submission_nonce'):
        try:parsed=uuid.UUID(r[key])
        except (ValueError, KeyError, TypeError, AttributeError):fail('CANONICAL_REQUEST_UUID')
        if str(parsed)!=r[key] or parsed.version!=4:fail('CANONICAL_REQUEST_UUID')
    for key,length in (('source_sha',40),('run_manifest_sha256',64)):
        if not re.fullmatch('[0-9a-f]{'+str(length)+'}',r.get(key,'')):
            fail('EXACT_SOURCE_REQUEST_MANIFEST_HASH')
    start=number(r['expected_native_start_step'],integer=True)
    from .heartbeat_recovery import lineage
    lineage(r)
    expected_resume=f"training/{block['arm']}/checkpoint-{start}" if start else None
    if (not segment and start>=1024) or block.get('resume_checkpoint_relative')!=expected_resume:
        fail('ORIGINAL_PRE1024_START_OR_EXACT_SAME_ARM_FULL_RESUME')
    wall=number(r['walltime_s'],integer=True,positive=True)
    if not 2760<wall<=8*3600 or now+2700>=HARD_CLOSE:
        fail('ORIGINAL_EIGHT_HOUR_OR_SHORTER_SITE_LIMIT_WITH_SAVE_RESERVE')
    if r.get('CPU_upper_s')!=cpu_envelope(r,wall):
        fail('FULL_ACTIVE_PROCESS_CPU_AND_PERFORMANCE_OBSERVER_ENVELOPE')
    return r


def validate_native(*, now, user, hostname, job_id, job, request_bytes, ledger,
                    project_proof, quota, queue_proof, budget_proof,
                    grant_bytes, campaign_view):
    """Reject stale or synthetic metadata; return a distinct A40 allocation type."""
    from .segmented_training import enabled
    grant(grant_bytes)
    if type(request_bytes) is not bytes or len(request_bytes)>1024**2:
        fail('BOUNDED_ORIGINAL_SEALED_REQUEST_BYTES')
    from scripts.pi0294.pv1.future_budget import _read
    r,request_sha=_read(request_bytes);request_scope(r,now=now)
    if (campaign_view.get('status')!='PASS_TWO_PLATFORM_CAMPAIGN_VIEW'
            or campaign_view.get('grant_sha256')!=GRANT_SHA
            or not 0<=now-campaign_view['observed_epoch']<=120):
        fail('FRESH_BOTH_SITE_UNFILTERED_CAMPAIGN_VIEW')
    observed=[j for j in campaign_view['campaign_jobs'] if j['job_id']==job_id]
    if (len(observed)!=1 or observed[0]!=dict(platform=PLATFORM,job_id=job_id,state='R',
            project=r['project'],name=r['job_name'])
            or sum(j['platform']==PLATFORM for j in campaign_view['campaign_jobs'])!=1
            or len(campaign_view['campaign_jobs'])>2
            or re.fullmatch(r'[0-9]+\.stdct-mgmt-02',job_id) is None
            or not hostname.split('.')[0].lower().startswith('gn-a40-')
            or job.get('job_state')!='R' or job.get('Job_Owner','').split('@')[0]!=user
            or job.get('project')!=r['project'] or job.get('Job_Name')!=r['job_name']
            or job.get('queue')!='batch_gpu' or job.get('Rerunable') not in (False,'False')
            or int(job.get('run_count',1))!=1):
        fail('ONE_OWN_RUNNING_A40_JOB_NOT_HOPPER_OR_OTHER_PROJECT_ALLOCATION')
    variables=job.get('Variable_List',{})
    if (not isinstance(variables,dict) or variables.get('PV1_REQUEST_ID')!=r['request_id']
            or variables.get('PV1_SUBMISSION_NONCE')!=r['submission_nonce']):
        fail('SAME_NATIVE_REQUEST_NONCE_NO_SERVER_RESTART')
    limits=job['Resource_List'];cpus,mem=select_profile(limits,r)
    if re.fullmatch(r'[0-9]{2,3}:[0-5][0-9]:[0-5][0-9]',limits.get('walltime','')) is None:
        fail('EXACT_NATIVE_WALLTIME_FORMAT')
    native_wall=sum(int(x)*scale for x,scale in zip(limits['walltime'].split(':'),(3600,60,1)))
    started=pbs_epoch(job['stime']);nodes={p.split('/')[0].split('.')[0] for p in job['exec_host'].split('+')}
    allocation_end=min(started+native_wall,T0+156*3600)
    if (native_wall!=r['walltime_s'] or hostname.split('.')[0] not in nodes or len(nodes)!=1
            or not T0<=started<=now<allocation_end-2700):
        fail('EXACT_NATIVE_COMPUTE_HOST_WALL_AND_FULL_SAVE_HEADROOM')
    if (queue_proof.get('source')!='VANDA_NATIVE_QSTAT_EFFECTIVE_QUEUE_LIMITS'
            or queue_proof.get('requested_queue')!='auto' or queue_proof.get('destination')!='batch_gpu'
            or not 0<=r['created_epoch']-queue_proof['observed_epoch']<=120
            or not native_wall<=number(queue_proof['maximum_wall_s'],integer=True,positive=True)):
        fail('PROJECT_AUTO_AND_STRICTER_NATIVE_QUEUE_BOUND')
    proof=project_proof;project=proof.get('project',{})
    today=datetime.fromtimestamp(now,timezone(timedelta(hours=8))).date()
    if (proof.get('schema')!='VAP1_A40_NATIVE_RUNNING_PROJECT_ATTESTATION_V1'
            or proof.get('job_id')!=job_id or proof.get('run_uuid')!=r['run_uuid']
            or proof.get('request_id')!=r['request_id'] or proof.get('submission_nonce')!=r['submission_nonce']
            or proof.get('source')!='VANDA_LOGIN_NATIVE_AMGR_LS_AND_CHECKBALANCE'
            or not started<=proof['observed_epoch']<=now or now-proof['observed_epoch']>120
            or project.get('name')!=r['project'] or project.get('active') is not True
            or user not in project.get('users',[]) or 'stdct-mgmt-02' not in project.get('cluster',[])
            or not date.fromisoformat(project['start_date'])<=today<=date.fromisoformat(project['end_date'])
            or number(proof['balance_GPU_hours'])<(native_wall+PAD)/3600
            or number(proof['balance_CPU_hours'])<cpus*(native_wall+PAD)/3600):
        fail('FRESH_AFTER_ALLOCATION_PROJECT_AUTH_AND_ACTUAL_CAPACITY_BILLING_BALANCE')
    if (ledger.get('authority_release')!=PARENT_AUTHORITY
            or ledger.get('campaign_run_uuid')!=r['campaign_run_uuid']
            or ledger.get('unknown_effects')!=[] or ledger.get('pending_updates')!=[]
            or ledger['used']['PBS_submissions']!=len(ledger['PBS_submissions'])):
        fail('ONE_ORIGINAL_GLOBAL_LEDGER_NO_UNKNOWN_OR_RESET')
    rows=[s for s in ledger['PBS_submissions'] if s['request_id']==r['request_id']]
    if (len(rows)!=1 or r['submit_count']>len(ledger['PBS_submissions'])
            or ledger['PBS_submissions'][r['submit_count']-1]['request_id']!=r['request_id']):
        fail('ONE_DURABLE_BEFORE_QSUB_RESERVATION')
    row=rows[0];updates=r['training_block']['stop_after_step']-r['expected_native_start_step']
    expected=dict(job_id=job_id,native_platform=PLATFORM,platform_training_version=VERSION,
        scientific_admission_version=SCIENCE_VERSION,project=r['project'],job_name=r['job_name'],
        source_sha=r['source_sha'],request_sha256=request_sha,
        run_manifest_sha256=r['run_manifest_sha256'],submission_nonce=r['submission_nonce'],
        arm=r['training_block']['arm'],reserved_H200_s=native_wall,accounting_pad_s=PAD,
        global_submit_count=r['submit_count'],
        reserved_optimizer_updates=updates,reserved_neural_queries=29,
        CPU_upper_s=r['CPU_upper_s'],physical_GPU_kind='NVIDIA_A40',physical_GPU_count=1,
        CPU_envelope_profile=r.get('CPU_envelope_profile'),
        legacy_GPU_seconds_field_is_projection=True,parent_checkpoint_manifest_sha256=PARENT,
        statistics_sha256=R0,new_bank_manifest_sha256=BANK,
        additional_output_upper_bytes=r['additional_output_upper_bytes'])
    if (any(row.get(k)!=value for k,value in expected.items())
            or row.get('status') not in ('SUBMITTED_PENDING_NATIVE','NATIVE_RUNNING_QUALIFICATION')
            or ledger['reserved'].get('H200_s',0)<native_wall+PAD
            or ledger['reserved'].get('optimizer_updates',0)<updates
            or ledger['reserved'].get('neural_queries',0)<29
            or ledger.get('CPU_reserved_s',0)<r['CPU_upper_s']
            or number(ledger['CPU_s'])+number(ledger['CPU_reserved_s'])>160*3600):
        fail('FULL_WALL_UPDATE_QUERY_OUTPUT_ENVELOPE_RESERVED_IN_SAME_GLOBAL_TOTAL')
    from . import b_heartbeat_recovery as b
    b_scope=b.enabled(r)
    if b_scope:
        b.validate_adopted(ledger,budget_proof['DEC1_native_recovery_view'])
        if not b.enabled(budget_proof):fail('B_RECOVERY_SCOPE_IN_NATIVE_BUDGET_PROOF')
    admit_resources(ledger['used'],ledger['reserved'],{},
        b_recovery_ledgers=(ledger,budget_proof['DEC1_native_recovery_view']) if b_scope else None)
    if (budget_proof.get('schema')!='VAP1_A40_RECOMPUTED_REMAINING_WORK_ADMISSION_V1'
            or budget_proof.get('status')!='PASS_ORIGINAL_GLOBAL_AND_DEC1_REMAINING_WORK'
            or budget_proof.get('purpose')!='NATIVE_PRETRAIN'
            or budget_proof.get('job_id')!=job_id or budget_proof.get('request_sha256')!=request_sha
            or not 0<=now-budget_proof['observed_epoch']<=120
            or budget_proof.get('final_GPU_protected_s')!=72000
            or budget_proof.get('original_T0_epoch')!=T0 or budget_proof.get('hard_close_epoch')!=HARD_CLOSE
            or budget_proof.get('all_old_spent_and_reserved_preserved') is not True
            or budget_proof.get('global_ledger_canonical_sha256')!=digest(ledger)
            or budget_proof.get('same_platform_AB')!=PLATFORM
            or budget_proof.get('new_R1_in_AB') is not False
            or budget_proof.get('uses_H200_timing_as_A40_measurement') is not False
            or budget_proof.get('full_remaining_work_admissible') is not True
            or budget_proof.get('forecast',{}).get('admissible_forecast') is not True):
        fail('FRESH_NATIVE_FULL_REMAINING_WORK_NOT_ONLY_CURRENT_JOB_FITS')
    if r.get('recovery_authorization_release') is not None:
        from .lost_save_recovery import validate_adopted,GRANT_SHA as RECOVERY_SHA
        validate_adopted(ledger,budget_proof['DEC1_native_recovery_view'])
        if (budget_proof.get('source_recovery_grant_sha256')!=RECOVERY_SHA
                or digest(budget_proof['DEC1_native_recovery_view'])!=budget_proof.get('DEC1_ledger_canonical_sha256')):
            fail('EXACT_ADOPTED_RECOVERY_IN_NATIVE_BUDGET_PROOF')
    if enabled(r):
        from .segmented_training import validate_adopted,GRANT_SHA as SEGMENT_SHA
        validate_adopted(ledger,budget_proof['DEC1_native_recovery_view'])
        if budget_proof.get('segmented_grant_sha256')!=SEGMENT_SHA:fail('NATIVE_SEGMENT_GRANT_IN_BUDGET')
    planned=[j for j in budget_proof['forecast'].get('mandatory_jobs',[])
        if j.get('work_id')==row.get('dec1_work_id')]
    if (len(planned)!=1 or planned[0].get('arm')!=r['training_block']['arm']
            or planned[0].get('start_step')!=r['expected_native_start_step']
            or planned[0].get('stop_step')!=r['training_block']['stop_after_step'] or planned[0].get('same_allocation_1024_probes') is not True
            or planned[0].get('walltime_upper_s')!=native_wall
            or planned[0].get('resources',{}).get('optimizer_updates')!=updates
            or planned[0]['resources'].get('Hopper_CPU_s')!=r['CPU_upper_s']
            or planned[0].get('CPU_envelope_profile')!=r.get('CPU_envelope_profile')
            or planned[0]['resources'].get('neural_queries',0)<29):
        fail('EXACT_CURRENT_JOB_IN_COMPLETE_SOURCE_PAIR_AND_MATRIX_FORECAST')
    from . import storage_expansion as storage
    from . import heartbeat_recovery as heartbeat
    global_output_cap,dec1_output_cap=heartbeat.caps(r)
    if heartbeat.enabled(r):
        heartbeat.validate_adopted(ledger,budget_proof['DEC1_native_recovery_view'])
        if not heartbeat.enabled(budget_proof):fail('HEARTBEAT_SCOPE_IN_NATIVE_BUDGET_PROOF')
    from . import zero_update_recovery as zero
    if zero.enabled(r):
        zero.validate_adopted(ledger,budget_proof['DEC1_native_recovery_view'])
        if not zero.enabled(budget_proof):fail('EXACT_ZERO_UPDATE_RESUME_SCOPE_IN_NATIVE_PROOF')
    if storage.enabled(r):
        storage.validate_adopted(ledger,budget_proof['DEC1_native_recovery_view'])
        if budget_proof.get('storage_authorization_release')!=storage.RELEASE or budget_proof.get('storage_grant_sha256')!=storage.GRANT_SHA:
            fail('EXACT_STORAGE_SCOPE_IN_NATIVE_BUDGET_PROOF')
    extra=number(r['additional_output_upper_bytes'],integer=True,positive=True)
    current=number(ledger['new_storage_current_bytes'],integer=True)
    reserved=number(ledger['storage_reserved_bytes'],integer=True)
    if (reserved<extra or current+reserved+40*GIB>global_output_cap
            or quota.get('user')!=user or quota.get('scope')!='OWN_PROJECT_AND_FILESYSTEM_MINIMUM'
            or not 0<=now-quota['observed_epoch']<=900
            or number(quota['available_bytes'],integer=True)<reserved+40*GIB
            or number(budget_proof['DEC1_GPU_site_output_used_and_reserved_bytes'],integer=True)>dec1_output_cap):
        fail('ACTUAL_PROJECT_AND_FILESYSTEM_HEADROOM_WITH_BOTH_OUTPUT_CAPS_AND_FINAL_RESERVE')
    return dict(status=STATUS,job_id=job_id,submit_count=r['submit_count'],request_sha256=request_sha,
        request_id=r['request_id'],submission_nonce=r['submission_nonce'],run_uuid=r['run_uuid'],
        source_sha=r['source_sha'],manifest_sha256=r['run_manifest_sha256'],
        native_platform=PLATFORM,physical_GPU='NVIDIA_A40',hostname=hostname,project=r['project'],
        observed_epoch=now,allocation_start_epoch=started,
        allocation_end_epoch=allocation_end,
        native_select=limits['select'],native_allocated_ncpus=cpus,native_allocated_memory_bytes=mem,
        actual_process_CPU_limit=4,site_capacity_is_not_actual_process_CPU=True,
        process_CPU_time_upper_s=process_cpu_limit(r),CPU_accounting_upper_s=r['CPU_upper_s'],
        CPU_envelope_profile=r.get('CPU_envelope_profile'),
        request_or_native_resources_modified=False,model_loaded=False,new_optimizer_updates=0)
