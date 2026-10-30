"""One raw-parent E1@512 allocation in the existing campaign ledgers.

Additional conservative escrow preserves the live E2's SHA-pinned old package,
old D25 row and every historical cost. No IO or runtime permission is supplied.
The existing native process, finite delegation and onceIO implementations own
execution; this module defines the new scientific allocation family only.
"""
from copy import deepcopy
import hashlib, json, math, re, uuid

from . import minimal_evaluation as plan
from scripts.pi0294.pv1 import mix_accounting as old, successor_cd as campaign
from scripts.pi0294.pv1.evaluation_accounting import progress
from scripts.pi0294.pv1.evaluation_core_accounting import process_handle

VERSION=plan.VERSION
FIELD='ARX5_minimal_Core_fixed512_allocations'
CANDIDATES='ARX5_minimal_Core_fixed512_candidates'
ROLE='ARX5_MINIMAL_CORE_FIXED512_NEURAL_V1'
OWNER_ROLE='ARX5_MINIMAL_CORE_FIXED512_RESIDENT_OWNER_V1'
TASK=plan.STAGE
TASK_SOURCE=campaign.TASK_SOURCE
PHYSICAL_ROLE='V5_CD_CORE_POLICY_EVALUATION_V1'  # Existing CPU evaluator role.
PROFILE='4080-neural-evaluation'
STATES=('PRESTART_INTENT_UNKNOWN_NO_RETRY','BOUND_QUALIFICATION','NATIVE_RUNNING','NATIVE_TERMINAL')
CHECKPOINT='383ad9b8fc3b81efffde132d4f06de92fb96fbdf9ec0b6c92c097e0f58eb2a8d'
DIRECTIVE_SOURCE='28c6eb0fe97b7cb852e321bbcd19f8ed047abc34'
DIRECTIVE_SHA='c8c62b86ceaaf83b5eed2f6f89ed387e72761c52d9fd113b5160be47e1a72198'
CONTROL=dict(Hopper_CPU_s=1800,Core_CPU_s=9000,local_CPU_s=7200,
             transfer_bytes=256*1024**2,Core_storage_bytes=128*1024**2,
             Hopper_storage_bytes=64*1024**2)
HOLD=dict(H200_s=28860,neural_queries=2097,Hopper_CPU_s=80000,
          Core_CPU_s=173900,Hopper_storage_bytes=36*1024**3,
          Core_storage_bytes=36*1024**3)
HOLD['transfer_bytes']=2*HOLD['Core_storage_bytes']
CPU_UPPER=80000
GPU_UPPER=28860


def raw(v): return (json.dumps(v,indent=2,allow_nan=False)+'\n').encode()
def sha(b): return hashlib.sha256(b).hexdigest()
def digest(v): return sha(raw(v))
def need(ok,why):
    if not ok: raise PermissionError('ARX5_MINIMAL_CORE_'+why)
def typed(v): return plan.typed(v)
def canonical_uuid(v):
    try: return type(v) is str and str(uuid.UUID(v))==v
    except (ValueError,TypeError,AttributeError): return False


def grant():
    return dict(schema='ARX5_MINIMAL_E1_FIXED512_DIRECT_SCIENCE_FIRST_SCOPE_V1',
        directive_source=DIRECTIVE_SOURCE,directive_sha256=DIRECTIVE_SHA,
        purpose='Original E1 fixed512 first valid condition then original24, without modifying live E2 or D25',
        candidate='E1',checkpoint_manifest_sha256=CHECKPOINT,checkpoint_step=512,
        raw_parent_files_identity=plan.RAW_PARENT_FILES_ID,statistics_sha256=plan.R0,
        population_plan_sha256=plan.POPULATION,noise_bank=294101,conditions=24,
        task_duration_s=32,H40=40,prefix=8,denoise_steps=4,
        optimizer_updates=0,PBS_submissions=0,new_target_WBC_physics=0,
        additional_conservative_hold=deepcopy(HOLD),control_upper_paid_once=deepcopy(CONTROL),
        old_escrows_and_costs_retained=True,old_package_and_D25_row_unchanged=True,
        project_internal_total_budget_limit=None,automatic_retry=False,
        same_fault_after_evidenced_fix='STOP_AFFECTED_BRANCH',
        same_job_native_qualification=dict(heartbeat_s=600,transport_cycles=80),
        deployment_requires_actual_devices_quota_space_and_original_onceIO=True)


GRANT_SHA=digest(grant())
GRANT_FILE='plans/ARX5_MINIMAL_E1_FIXED512_SCOPE.json'


def validated(value):
    rows=validate(value)
    need(len(rows)==1,'ONE_ACTUAL_FIXED512_CLOCK')
    row=next(iter(rows.values()))
    return dict(task=TASK,task_source=TASK_SOURCE,grant_sha256=GRANT_SHA,
        T_CD=row['T_CD'],deadline_epoch=row['deadline_epoch'])


def endpoint_reference():
    return dict(schema='ARX5_E1_FIXED512_CLOSED_ENDPOINT_REFERENCE_V1',candidate='E1',global_step=512,
        checkpoint_manifest_sha256=CHECKPOINT,raw_parent_files_identity=plan.RAW_PARENT_FILES_ID,
        statistics_sha256=plan.R0,HF_revision='f410cd2b92cd543606d54b46f710197454606d9b',
        original_qualified_endpoint_reused=True,new_training_updates=0)


def request_fields():
    return dict(schema='ARX5_MINIMAL_CORE_FIXED512_REQUEST_V1',
        platform_evaluation_version=VERSION,native_platform='CORELAB_RTX4080',
        task=plan.STAGE,authority_release=campaign.TASK_SOURCE,scope_grant_sha256=GRANT_SHA,
        stage=plan.STAGE,evaluation_candidate='E1',noise_bank=294101,remaining_conditions=24,
        checkpoint_step=512,checkpoint_manifest_sha256=CHECKPOINT,
        raw_parent_files_identity=plan.RAW_PARENT_FILES_ID,statistics_sha256=plan.R0,
        walltime_s=28800,CPU_upper_s=80000,declared_process_CPU_limit_s=79760,
        active_cpu_limit=4,requested_memory_bytes=48*1024**3,
        minimum_GPU_memory_bytes=30*1024**3,optimizer_updates=0,
        additional_output_upper_bytes=HOLD['Hopper_storage_bytes'],
        Teacher_takeover=False,normalizer_refit=False,evaluation_data_for_training=False,
        LOCKBOX_opened=False,numerical_mode='DETERMINISTIC_CUDA_MATH_SDP_V1',allocation_number=1,
        global_accounting_mode='V5_CD_CORE_FINITE_RESERVED_LOCAL_EFFECTS_V1')


def request(r):
    need(all(r.get(k)==v and type(r.get(k)) is type(v) for k,v in request_fields().items()),
         'EXACT_E1_RAW512_ORIGINAL24_ZERO_TRAINING_REQUEST')
    need(canonical_uuid(r.get('request_id')) and canonical_uuid(r.get('submission_nonce')),
         'NEW_CANONICAL_REQUEST_AND_NONCE')
    for k,n in [('source_sha',40),('run_manifest_sha256',64),('cold_input_audit_sha256',64),
                ('planned_panel_sha256',64),('training_handoff_sha256',64)]:
        need(type(r.get(k)) is str and re.fullmatch('[0-9a-f]{'+str(n)+'}',r[k]),'SEALED_'+k)
    need(type(r.get('run_uuid')) is str and re.fullmatch('arx5-e1-fixed512-n294101-[0-9]{2}',r['run_uuid'])
        and r.get('native_unit')==r['run_uuid']+'-neural.service', 'DISTINCT_OWN_NATIVE_UNIT')
    need(type(r.get('GPU_uuid')) is str and re.fullmatch(r'GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}',r['GPU_uuid']),
         'ACTUAL_DEVICE_UUID')
    need(not any(k in r for k in ('request_sha256','native_handle','status','project','PBS_JOBID',
        'platform_training_version','shared_GPU_grant_sha256','startup_recovery_sha256',
        'parent_checkpoint_manifest_sha256')), 'NO_CORR_PARENT_OR_OLD_SCOPE_ALIAS')
    clock=campaign.scope(r.get('T_CD'))
    need(r.get('deadline_epoch')==clock['deadline_epoch'] and type(r.get('created_epoch')) in (int,float)
        and math.isfinite(r['created_epoch']) and clock['T_CD']<=r['created_epoch']<clock['deadline_epoch']-28860,
        'FINITE_RUN_INSIDE_DECLARED_WINDOW')
    return r


def validate_row(row):
    saved=row.get('request');need(type(saved) is dict,'COMPLETE_ORIGINAL_REQUEST_NOT_A_ROW_ALIAS')
    request(saved)
    need(row.get('request_sha256')==digest(saved)
        and all(row.get(k)==v and type(row[k]) is type(v) for k,v in saved.items())
        and row.get('status') in STATES and row.get('child_run_uuid')==saved['run_uuid']
        and row.get('reserved_GPU_s')==HOLD['H200_s'] and row.get('query_limit')==HOLD['neural_queries']
        and row.get('additional_output_upper_bytes')==HOLD['Hopper_storage_bytes'],
        'IMMUTABLE_TYPED_ROW_AND_OWN_ESCROW')
    q,loads,cpu=progress(row)
    need(q<=HOLD['neural_queries'] and loads<=2 and cpu<=80000,
         'PER_JOB_FINITE_ACTUAL_EFFECTS')
    need(row.get('reserved_neural_queries')==(len(row.get('unknown_query_intents',[])) if row['status']==STATES[3] else HOLD['neural_queries']-q),
         'ALL_UNSPENT_QUERIES_HELD')
    if row['status']==STATES[0]: need(row.get('native_handle') is None and (q,loads,cpu)==(0,0,0),'NO_UNBOUND_EFFECT')
    else:
        from scripts.pi0294.pv1.evaluation_core_accounting import process_handle
        need(row.get('native_handle')==process_handle(row.get('boot_id'),row.get('pid'),row.get('proc_start')),
             'REAL_BOUND_BOOT_PID_PROCSTART')
    return row


def validate(g):
    rows=g.get(FIELD,{})
    need(type(rows) is dict and len(rows)<=1,'ONE_E1_PANEL_NO_AUTOMATIC_RETRY')
    for rid,row in rows.items():
        validate_row(row);need(row['request_id']==rid,'CANONICAL_INDEX')
        need(row.get('control_payment')==CONTROL and row.get('scope_grant_sha256')==GRANT_SHA,
             'DIRECT_AUTHORITY_AND_ONCE_PAID_CONTROL')
    from .minimal_core_cold import validate as cold_rebind
    cold_rebind(g)
    active=[v for v in rows.values()if v['status']!=STATES[3]]
    if not active:return rows
    need(g['unknown_effects']==g['pending_updates']==[], 'GLOBAL_UNKNOWN_AND_PENDING_FENCES')
    # The legacy D25 reservation is not an actual GPU process. It stays exact
    # while the healthy E2 source still validates it; no old role is revived.
    other=[r for r in g.get(old.CORE_ROWS,{}).values()if r['status']!='NATIVE_TERMINAL']
    need(all(r.get('native_handle') is None and progress(r)==(0,0,0) for r in other),
         'NO_COMPETING_CORE_EVALUATION')
    pbs=[r for r in g['PBS_submissions']if r['status'] not in old.TERMINAL]
    need(len(pbs)<=1 and all(r.get('platform_training_version')=='ARX5_MINIMAL_E2_ONLY_FRESH_RAW512_V1'
        and r.get('job_id')=='1508345.stdct-mgmt-02' for r in pbs),
        'ONLY_EXISTING_INDEPENDENT_VANDA_E2_NOT_ANOTHER_TRAINER')
    row=active[0]
    for key,amount in [('H200_s',HOLD['H200_s']),('neural_queries',row['reserved_neural_queries'])]:
        need(g['reserved'][key]>=amount,'OWN_HOLD_'+key)
    need(g['CPU_reserved_s']>=80000-progress(row)[2]
        and g['storage_reserved_bytes']>=HOLD['Hopper_storage_bytes'], 'OWN_CPU_STORAGE_HOLDS')
    return rows


def checked(g):
    from scripts.pi0294.pv1.neural_ledger import check
    check(g,allow_unknown=True);rows=validate(g)
    active=[r for r in rows.values()if r['status']!=STATES[3]]
    if active:
        # Validation-only subtraction proves that our hold cannot hide an old
        # underfunded allocation. Never persist this projection. Old E2 can
        # consume its own escrow meanwhile, so the original numeric floor is
        # not incorrectly treated as a permanent frozen resource counter.
        v=deepcopy(g)
        for row in active:
            v['reserved']['H200_s']-=HOLD['H200_s']
            v['reserved']['neural_queries']-=row['reserved_neural_queries']
            v['CPU_reserved_s']-=80000-progress(row)[2]
            v['storage_reserved_bytes']-=HOLD['Hopper_storage_bytes']
        del v[FIELD]
        check(v,allow_unknown=True)
    return g


def paired(g,l):
    checked(g)
    from .minimal_core_cold import FIELD as cold_field
    need(l.get(FIELD,{})==g.get(FIELD,{}) and l['unknown_effects']==l['pending_updates']==l['pending_queries']==[],
         'SAME_CANONICAL_NEW_FAMILY_AND_LOCAL_FENCES')
    need(l.get(cold_field)==g.get(cold_field),'SAME_CANONICAL_EXACT_COLD_REBIND')


def register(global_bytes,local_bytes,*,request_bytes,manifest_bytes,physical_manifest_bytes,now_epoch):
    g,l,r,m,p=map(json.loads,(global_bytes,local_bytes,request_bytes,manifest_bytes,physical_manifest_bytes))
    checked(g);request(r);need(FIELD not in g and FIELD not in l,'ONE_NEW_FAMILY_REGISTRATION')
    from .minimal_core_identity import request_identity
    request_identity(m,r,sha(manifest_bytes))
    need(request_bytes==raw(r) and manifest_bytes==raw(m) and r['created_epoch']<=now_epoch<r['deadline_epoch']-28860,
         'EXACT_CANONICAL_BYTES_AND_ACTUAL_WINDOW')
    need(plan.typed(p) and p['candidate']=='E1' and p['run_uuid']==r['run_uuid']
        and p['code']['vla_commit']==r['source_sha'] and p['authority']['scope_grant_sha256']==GRANT_SHA
        and p['limits']['physics_ticks']==832000 and p['limits']['CPU_s']==93900
        and p['limits']['policy_episodes']==26 and p['limits']['engine_starts']==2108,
         'ORIGINAL_FULL_PHYSICAL_ENVELOPE_AND_FIXED_PEER')
    floor=dict(reserved=deepcopy(g['reserved']),CPU_reserved_s=g['CPU_reserved_s'],storage_reserved_bytes=g['storage_reserved_bytes'])
    row=dict(r,request=deepcopy(r),request_sha256=sha(request_bytes),child_run_uuid=r['run_uuid'],
        status=STATES[0],native_handle=None,reserved_GPU_s=HOLD['H200_s'],query_limit=HOLD['neural_queries'],
        hostname='corelab-4080s',owner_uid=2009,
        reserved_neural_queries=HOLD['neural_queries'],additional_output_upper_bytes=HOLD['Hopper_storage_bytes'],
        reservation_floor_before=floor,control_payment=deepcopy(CONTROL),
        physical_manifest_sha256=sha(physical_manifest_bytes),created_epoch=r['created_epoch'])
    v,w=deepcopy(g),deepcopy(l)
    for x in (v,w):x[FIELD]={r['request_id']:deepcopy(row)}
    for key in ('H200_s','neural_queries'):v['reserved'][key]+=HOLD[key]
    v['CPU_reserved_s']+=80000;v['storage_reserved_bytes']+=HOLD['Hopper_storage_bytes']
    v['CPU_s']+=CONTROL['Hopper_CPU_s'];v['new_storage_current_bytes']+=CONTROL['Hopper_storage_bytes']
    v['new_storage_peak_bytes']=max(v['new_storage_peak_bytes'],v['new_storage_current_bytes'])
    w['GPU_reserved_s']+=HOLD['H200_s']
    w['CPU_reserved_s']['CoreLab']+=HOLD['Core_CPU_s']
    w['CPU_reserved_s']['Hopper']+=HOLD['Hopper_CPU_s']
    held=w.setdefault('Core_native_GPU_reserved',{})
    for name,amount in dict(episode_starts=26,physics_ticks=832000,constructors=2108,
        CoreLab_output_bytes=HOLD['Core_storage_bytes'],transfer_bytes=HOLD['transfer_bytes']).items():
        held[name]=held.get(name,0)+amount
    for name,site in [('Core_CPU_s','CoreLab'),('Hopper_CPU_s','Hopper'),('local_CPU_s','local_PC')]:w['CPU_s'][site]+=CONTROL[name]
    w['network_transfer_accounted_bytes']+=CONTROL['transfer_bytes']
    v['revision']+=1;paired(v,w)
    need(v[old.CORE_ROWS]==g[old.CORE_ROWS] and v['PBS_submissions']==g['PBS_submissions']
        and v['ARX5_minimal_first512_execution_package']==g['ARX5_minimal_first512_execution_package'],
        'LIVE_E2_D25_AND_OLD_PACKAGE_BYTES_UNCHANGED')
    return dict(proposed_global=v,proposed_LOCAL=w,original_onceIO_required=True,native_admission=False)


def bind(g,l,request_id,process,*,now_epoch):
    paired(g,l);v,w=deepcopy(g),deepcopy(l);row=v[FIELD][request_id]
    need(row['status']==STATES[0] and 0<=now_epoch-process['observed_epoch']<120,'ONE_FRESH_ACTUAL_BIND')
    from .minimal_core_identity import check_clock
    check_clock(row['request'],started_epoch=process['started_epoch'],now_epoch=now_epoch)
    need(process['request_id']==request_id and process['request_sha256']==row['request_sha256']
        and process['source_sha']==row['source_sha'] and process['run_manifest_sha256']==row['run_manifest_sha256'],
         'NATIVE_PROCESS_EXACT_REQUEST_NOT_A_CLAIMED_PID')
    row.update(status=STATES[1],native_handle=process['native_handle'],boot_id=process['boot_id'],pid=process['pid'],
               proc_start=process['proc_start'],process_started_epoch=process['started_epoch'])
    w[FIELD][request_id]=deepcopy(row);v['revision']+=1;paired(v,w)
    return dict(proposed_global=v,proposed_LOCAL=w,native_admission=False)


def old_future_projection(g,l):
    """Remove only THIS unspent hold for future double-count accounting.

    Never persisted; every actual paid cost remains. The final forecast adds
    our entire finite output bound back. Original package bytes are untouched.
    """
    v,w=deepcopy(g),deepcopy(l)
    rows=validate(g)
    for r in rows.values():
        if r['status']==STATES[3]:continue
        v['reserved']['H200_s']-=HOLD['H200_s'];v['reserved']['neural_queries']-=r['reserved_neural_queries']
        v['CPU_reserved_s']-=80000-progress(r)[2];v['storage_reserved_bytes']-=HOLD['Hopper_storage_bytes']
        w['GPU_reserved_s']-=HOLD['H200_s'];w['CPU_reserved_s']['CoreLab']-=HOLD['Core_CPU_s']
        w['CPU_reserved_s']['Hopper']-=HOLD['Hopper_CPU_s']
        for name,amount in dict(episode_starts=26,physics_ticks=832000,constructors=2108,
            CoreLab_output_bytes=HOLD['Core_storage_bytes'],transfer_bytes=HOLD['transfer_bytes']).items():
            w['Core_native_GPU_reserved'][name]-=amount
    v.pop(FIELD,None);w.pop(FIELD,None)
    return v,w


def finite_control_scope(g,rid):
    row=validate(g)[rid]
    return dict(schema='ARX5_MINIMAL_CORE_FIXED512_FINITE_CONTROL_SCOPE_V1',
        marker_sha256=digest(row['control_payment']),request_id=rid,
        CPU_s=7200,RPC_calls=150000,source_preparation_CPU_s=600,
        total_control_CPU_s=9000,owner_process_CPU_limit_s=1200,
        owner_output_limit_bytes=3*1024**3,closing_output_reserve_bytes=512*1024**2,
        financial_costs=deepcopy(CONTROL),
        Core_CPU_components_s=dict(cold_prepare=600,owner=1200,original_RPC=6600,closeout=600),
        wire_phase_upper_bytes=dict(fixed512_preparation=CONTROL['transfer_bytes']))
