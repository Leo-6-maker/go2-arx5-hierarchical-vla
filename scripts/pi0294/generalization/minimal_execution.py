"""First512 typed PBS/session index inside the original GLOBAL reservation.

Pure transitions only. They neither submit nor supply native admission. The
original owner/onceIO/queue/lease/native process remain authoritative. Local
A40 effect journals retain each session's pending updates and query intents;
this module never clears a global or local UNKNOWN/pending latch.
"""
import base64,json,re,uuid
from copy import deepcopy
from . import minimal_package as package,minimal_adoption as authority,minimal_scope as scope
from . import minimal_training as training
from .minimal_dataset import PANEL_SHA,CURSOR_SHA
from .minimal_study import R0
from scripts.pi0294.pv1 import mix_accounting as original
from scripts.pi0294.pv1.future_budget import _vector

FIELD='ARX5_minimal_first512_native_execution'
VERSION='ARX5_MINIMAL_RAW_PARENT_FIRST512_VANDA_NATIVE_V1'
PROFILE='vanda-arx5-minimal-raw-parent-first512'
CAPACITY_PROFILES=((8,64*1024**3),(36,240*1024**3))
PREPARATION_CPU_UPPER=900
STATES=('QSUB_INTENT_OUTCOME_UNKNOWN_NO_RETRY','SUBMITTED_PENDING_NATIVE','NATIVE_TERMINAL')
CONTROL=_vector(dict(Hopper_CPU_s=3600,Hopper_storage_bytes=128*1024**2,
                     transfer_bytes=768*1024**2,local_CPU_s=630))
raw=authority.raw;sha=authority.sha;need=authority.need


def native_sessions():
    """Preserve147 queries, including actual official-export comparisons.

    The reserved session_plan is immutable. Its two unmatched E0 engineering
    seeds29415/16 become one reconstruction draw per formal arm. All fifteen
    paired zero-step seeds, both full-restore pairs, original12 TRAIN probes
    and every language24 case remain. No new observation or label is added.
    """
    rows=deepcopy(package.sessions())
    rows[0]['query_limit']=39
    for row in rows[3:]:row['query_limit']=37
    need(sum(r['query_limit']for r in rows)==147,'EXECUTION_FIXED_TOTAL_QUERY_ENVELOPE')
    return rows


def select_profile(limits,r):
    """Typed extension of the original own-job reader, not an allocation.

    Only this new immutable request may use the approved16h30m/eight-active
    profile, with either requested8/64GB or actual site36/240GB capacity.
    Existing4/36-CPU and eight-hour requests keep their old parser.
    """
    request(r)
    from scripts.pi0294.dec1.a40_allocation import memory_bytes
    parts=limits.get('select','').split(':');fields={}
    need(parts[0]=='1'and '+'not in limits.get('select',''),'EXECUTION_ONE_A40_SELECT_CHUNK')
    for part in parts[1:]:
        pair=part.split('=',1)
        need(len(pair)==2 and pair[0]not in fields,'EXECUTION_EXACT_SELECT_FIELDS')
        fields[pair[0]]=pair[1]
    cpus=int(fields.get('ncpus','0'));mem=memory_bytes(fields.get('mem',''))
    need((cpus,mem)in CAPACITY_PROFILES and fields.get('ngpus')=='1'
        and fields.get('gpu_model','A40')=='A40'
        and limits.get('ncpus')==cpus and int(limits.get('ngpus',0))==1
        and memory_bytes(limits.get('mem',''))==mem
        and limits.get('walltime')=='16:30:00',
        'EXECUTION_PREDECLARED_EIGHT_ACTIVE_CPUS_AND_ACTUAL_SITE_CAPACITY')
    return cpus,mem


def project_envelope(r):
    """Site project balance covers its physical allocation charging class.

    Native process CPU is bounded separately by CPU_upper_s. The site has
    actually granted36cores/240GB to prior successful jobs; acknowledge that
    capacity and project hold without calling36 cores actively used or raising
    the experiment's8-active-core/165600s process envelope.
    """
    request(r)
    return dict(GPU_hours=r['reserved_H200_s']/3600,
        CPU_hours=max(c for c,_ in CAPACITY_PROFILES)*r['reserved_H200_s']/3600,
        reserved_CPU_including_prior_preparation_s=r['CPU_upper_s'],
        actual_process_CPU_upper_s=r['CPU_upper_s']-PREPARATION_CPU_UPPER,active_CPU_limit=8,
        project_hold_is_not_experiment_CPU_spend=True)


def process_CPU_limits(r):
    """Keep the prior uncertain readonly900s in its original diagnostic pool.

    It is neither refunded nor silently treated as0. The native parent/child
    limits leave room for that cost and the original240s terminal observer.
    Terminal settlement includes it once, inside CPU_upper_s, never on top.
    """
    request(r);upper=r['CPU_upper_s'];child=600;observer=240
    return dict(parent_RLIMIT_CPU_s=upper-PREPARATION_CPU_UPPER-child-observer,
        isolated_child_CPU_upper_s=child,terminal_observer_CPU_upper_s=observer,
        prior_readonly_preparation_CPU_upper_s=PREPARATION_CPU_UPPER,
        whole_reserved_CPU_upper_s=upper,new_CPU_budget=0)


def main_resources():
    parts=scope.components()
    rows=[v for k,v in parts.items()if k in ('E1_QUALIFICATION_AND_FORMAL512',
        'E2_QUALIFICATION_AND_FORMAL512','FINITE_OFFLINE_DIAGNOSTIC_AND_SHARED_PBS')]
    return _vector({k:sum(x[k]for x in rows)for k in _vector({})})


def native_resources():
    return _vector({k:v-CONTROL[k]for k,v in main_resources().items()})


def subtract(vector,amount):
    need(all(vector[k]>=v for k,v in amount.items()),'EXECUTION_NO_BORROWED_RESOURCE:'+str(amount))
    for k,v in amount.items():vector[k]-=v


def _uuid(v):
    try:return type(v)is str and str(uuid.UUID(v))==v
    except (ValueError,TypeError,AttributeError):return False


def request(r,*,central=False):
    """Whole serial package, not a C/D/GEN row or five new allocations."""
    need(type(central)is bool,'EXECUTION_EXPLICIT_CENTRAL_ROW')
    resources=native_resources()
    fixed=dict(schema='ARX5_MINIMAL_FIRST512_NATIVE_REQUEST_V1',scope=scope.SCOPE,
        platform_training_version=VERSION,native_platform='VANDA_A40',
        profile=PROFILE,project='CFP04-CF-007',job_name='pi0294_dec1_a40_train',
        raw_parent_files_identity=training.RAW_PARENT_FILES_ID,
        statistics_sha256=R0,panel_sha256=PANEL_SHA,full_cursor_sha256=CURSOR_SHA,
        scheduler_horizon=1024,formal_terminal_steps=512,optimizer_update_limit=1028,
        query_limit=147,model_load_limit=11,expected_native_start_step=0,
        physical_GPU_count=1,physical_GPU_kind='NVIDIA_A40',active_cpu_limit=8,
        allowed_native_capacity_profiles=[dict(ncpus=c,memory_bytes=m)for c,m in CAPACITY_PROFILES],
        walltime_s=59400,accounting_pad_s=120,reserved_H200_s=59520,
        CPU_upper_s=resources['Hopper_CPU_s'],additional_output_upper_bytes=resources['Hopper_storage_bytes'],
        prior_readonly_preparation_CPU_upper_s=PREPARATION_CPU_UPPER,
        scope_grant_sha256=authority.GRANT_SHA,conditional1024_authorized=False,
        new_physics=0,new_constructors=0,session_plan=native_sessions(),
        reserved_session_plan_sha256=sha(raw(package.sessions())),
        native_session_plan_version='MATCHED15_AND_ACTUAL_OFFICIAL_EXPORT_WITHIN147_V1')
    need(all(r.get(k)==v and type(r.get(k))is type(v)for k,v in fixed.items()),
         'EXECUTION_EXACT_OWN_SEQUENTIAL_FIVE_SESSIONS_AND_MAIN_ENVELOPE')
    need(_uuid(r.get('request_id'))and _uuid(r.get('submission_nonce')),
         'EXECUTION_NEW_CANONICAL_REQUEST_AND_NONCE')
    for k,n in [('source_sha',40),('run_manifest_sha256',64),('cold_input_audit_sha256',64),
                ('raw_parent_audit_plan_sha256',64),('frozen_input_packet_sha256',64),('Core_measurement_capture_sha256',64)]:
        need(type(r.get(k))is str and re.fullmatch('[0-9a-f]{'+str(n)+'}',r[k]),
             'EXECUTION_FULL_SEALED_IDENTITY:'+k)
    if central:
        need(type(r.get('request_sha256'))is str and re.fullmatch('[0-9a-f]{64}',r['request_sha256'])
             and r.get('status')in STATES,'EXECUTION_KNOWN_CENTRAL_STATE')
        job=r.get('job_id')
        need(job is None and r['status']==STATES[0]or type(job)is str and
             re.fullmatch(r'[0-9]+\.stdct-mgmt-02',job),'EXECUTION_ACTUAL_UNIQUE_PBS_HANDLE')
    else:
        need('request_sha256'not in r and 'status'not in r and 'job_id'not in r,
             'EXECUTION_REQUEST_NOT_A_LEDGER_OR_SELF_HASH')
    return r


def validate(g):
    """An adopted row is never inferred from a request-shaped dictionary."""
    marker=g.get(FIELD)
    if marker is None:return None
    cap=authority.validate(g);p=g[package.FIELD]
    need(cap is not None and marker.get('schema')=='ARX5_MINIMAL_FIRST512_NATIVE_INDEX_V1'
        and marker['grant_sha256']==authority.GRANT_SHA
        and marker['original_resources']==package.resources()
        and marker['main_resources']==main_resources()
        and marker['control_paid_once']==CONTROL,
        'EXECUTION_ORIGINAL_RESERVED_PACKAGE_AND_PAID_CONTROL')
    ids=marker['registered_requests'];items=[r for r in g['PBS_submissions']
        if r.get('platform_training_version')==VERSION]
    need(type(ids)is list and len(ids)==len(items)==1 and ids==[items[0]['request_id']]
        and p['registered_requests']==ids,'EXECUTION_ONE_MAIN_NOT_FIVE_PBS_OR_REPLAY')
    r=items[0];request(r,central=True)
    request_bytes=base64.b64decode(marker['request_base64'],validate=True);saved=json.loads(request_bytes);request(saved)
    need(sha(request_bytes)==r['request_sha256']and all(r.get(k)==v and type(r.get(k))is type(v)for k,v in saved.items())
        and set(r)-set(saved)<=set(('request_sha256','status','job_id','submitted_epoch','native_reply_epoch')),
        'EXECUTION_ACTUAL_REQUEST_BYTES_NOT_RESTAMPED_ROW')
    from .minimal_measurement import validate as measurement
    proof=measurement(g,base64.b64decode(marker['Core_measurement_capture_base64'],validate=True))
    need(proof==marker['Core_measurement_proof']and proof['capture_sha256']==r['Core_measurement_capture_sha256'],
         'EXECUTION_ORIGINAL_MEASUREMENT_GATE_BEFORE_FORMAL_TRAINING')
    need(all(r[k]==cap[k]for k in ('T_CD','deadline_epoch','stop_new_training_epoch'))
        and r['source_sha']==marker['source_sha']
        and r['request_sha256']==marker['request_sha256']
        and r['run_manifest_sha256']==marker['run_manifest_sha256']
        and type(marker['registered_epoch'])in(int,float)
        and cap['T_CD']<=marker['registered_epoch']<cap['stop_new_training_epoch'],
        'EXECUTION_REAL_REGISTERED_BINDING_AND_UNRESET_CLOCK')
    need(g['used']['PBS_submissions']>=marker['before']['PBS_submissions']+1
        and g['CPU_s']>=marker['before']['CPU_s']+CONTROL['Hopper_CPU_s']
        and g['new_storage_current_bytes']>=marker['before']['new_storage_current_bytes']+CONTROL['Hopper_storage_bytes'],
        'EXECUTION_OLD_COSTS_AND_ONCE_CONTROL_RETAINED')
    need(type(p['completed_sessions'])is list and p['completed_sessions']==[],
        'EXECUTION_SESSION_TERMINAL_IMPORT_NOT_IMPLEMENTED_NO_FAKE_COMPLETION')
    need(r['status']in STATES[:2]and p['status']=='MAIN_NATIVE_REGISTERED',
        'EXECUTION_REGISTER_BIND_ONLY_NO_UNIMPLEMENTED_TERMINAL')
    remaining=_vector(p['remaining_resources']);expected=package.resources()
    subtract(expected,main_resources())
    need(remaining==expected,'EXECUTION_EXACT_UNUSED_CORE_EVAL_AND_REPLACEMENT_ESCROW')
    return marker


def rows(g):
    if validate(g)is None:return []
    return [r for r in g['PBS_submissions']if r.get('platform_training_version')==VERSION]


def forecast_remaining(g):
    p=g[package.FIELD]
    if FIELD not in g:return dict(p['remaining_resources'])
    validate(g);v=dict(p['remaining_resources'])
    for k,n in native_resources().items():
        # PBS was consumed at registration; transfer/LOCAL CPU were part of
        # the original once control, not a new projected metadata allowance.
        v[k]+=n if k!='PBS_submissions'else 0
    return _vector(v)


def budget_LOCAL(g,local):
    """Conservative once-control projection; never a canonical LOCAL write.

    GLOBAL has no transfer/local-PC counter. Removing those costs from its
    future package must project the same paid amounts into the existing LOCAL
    view. Its pre-existing reservation is untouched; it is not our new hold.
    """
    marker=validate(g)
    if marker is None:return local
    key='ARX5_minimal_first512_control_projection'
    proof=dict(request_sha256=marker['request_sha256'],costs=CONTROL,
               canonical_LOCAL_write_proven=False,original_once_control_only=True)
    if key in local:
        need(local[key]==proof,'EXECUTION_NO_OTHER_OR_CHANGED_LOCAL_CONTROL_PROJECTION')
        return local
    v=deepcopy(local)
    v['CPU_s']['local_PC']+=CONTROL['local_CPU_s']
    v['network_transfer_accounted_bytes']+=CONTROL['transfer_bytes']
    v[key]=proof
    return v


def register(global_bytes,*,request_bytes,Core_measurement_capture_bytes,now_epoch):
    """Reuse held resources and original PBS list, never reserve a second pack.

    The first join is deliberately serial. Later explicit cross-scope native
    admission must qualify a new Core capsule before allowing overlap; this
    function cannot make the current sealed C50 owner accept unknown rows.
    """
    g=json.loads(global_bytes);r=json.loads(request_bytes);request(r)
    from .minimal_measurement import validate as measurement
    measurement_proof=measurement(g,Core_measurement_capture_bytes)
    need(measurement_proof['capture_sha256']==r['Core_measurement_capture_sha256'],
         'EXECUTION_SEALED_MEASUREMENT_CAPTURE')
    need(FIELD not in g and g['unknown_effects']==g['pending_updates']==[],
         'EXECUTION_NO_EXISTING_MAIN_OR_UNKNOWN_LATCH')
    original.validate_global(g);p=package.validate(g);cap=authority.validate(g)
    need(p is not None and all(x['status']in original.TERMINAL for x in g['PBS_submissions'])
        and all(x['status']=='NATIVE_TERMINAL'for x in g.get(original.CORE_ROWS,{}).values()),
         'EXECUTION_COLD_SERIAL_BOUNDARY_NO_HOT_CORE_OVERLAP')
    need(cap['T_CD']<=now_epoch and now_epoch+59520<cap['stop_new_training_epoch']
        and all(r[k]==cap[k]for k in ('T_CD','deadline_epoch','stop_new_training_epoch')),
         'EXECUTION_COMPLETE_OWN_MAIN_INSIDE_ORIGINAL_WINDOW')
    need(not any(x.get('request_id')==r['request_id']or x.get('submission_nonce')==r['submission_nonce']
        for x in[*g['PBS_submissions'],*g.get(original.CORE_ROWS,{}).values()]),
        'EXECUTION_NEW_IDENTITY_ACROSS_BOTH_SITES')
    v=deepcopy(g);pv=v[package.FIELD];need(pv['remaining_resources']==package.resources(),
                                         'EXECUTION_ORIGINAL_FULL_PACKAGE_STILL_HELD')
    subtract(pv['remaining_resources'],main_resources())
    row=dict(r,request_sha256=sha(request_bytes),status=STATES[0],job_id=None,
             submitted_epoch=now_epoch)
    v['PBS_submissions'].append(row);v['used']['PBS_submissions']+=1
    v['CPU_reserved_s']-=CONTROL['Hopper_CPU_s'];v['CPU_s']+=CONTROL['Hopper_CPU_s']
    v['storage_reserved_bytes']-=CONTROL['Hopper_storage_bytes']
    v['new_storage_current_bytes']+=CONTROL['Hopper_storage_bytes']
    v['new_storage_peak_bytes']=max(v['new_storage_peak_bytes'],v['new_storage_current_bytes'])
    marker=dict(schema='ARX5_MINIMAL_FIRST512_NATIVE_INDEX_V1',grant_sha256=authority.GRANT_SHA,
        original_resources=package.resources(),main_resources=main_resources(),control_paid_once=CONTROL,
        before={k:g[k]for k in('CPU_s','new_storage_current_bytes')},registered_requests=[r['request_id']],
        source_sha=r['source_sha'],request_sha256=sha(request_bytes),request_base64=base64.b64encode(request_bytes).decode(),
        Core_measurement_capture_base64=base64.b64encode(Core_measurement_capture_bytes).decode(),Core_measurement_proof=measurement_proof,run_manifest_sha256=r['run_manifest_sha256'],
        registered_epoch=now_epoch,global_pending_and_unknown_latches_not_cleared=True,
        local_effect_journals_are_original_allocation_effects_not_another_budget=True)
    marker['before']['PBS_submissions']=g['used']['PBS_submissions'];v[FIELD]=marker
    pv.update(registered_requests=[r['request_id']],status='MAIN_NATIVE_REGISTERED')
    from scripts.pi0294.pv1.neural_ledger import check
    v['revision']+=1;validate(v);check(v,allow_unknown=True)
    return dict(proposed_global=v,native_admission=False,new_PBS_intents=1,
                new_model_queries_updates_physics=0,original_onceIO_required=True)


def bind(g,request_id,job_id,*,now_epoch):
    v=deepcopy(g);matches=[r for r in rows(v)if r['request_id']==request_id]
    need(len(matches)==1 and matches[0]['status']==STATES[0]and matches[0]['job_id']is None,
         'EXECUTION_EXACT_MAIN_NOT_ALREADY_BOUND')
    matches[0].update(status=STATES[1],job_id=job_id,native_reply_epoch=now_epoch)
    from scripts.pi0294.pv1.neural_ledger import check
    v['revision']+=1;validate(v);check(v,allow_unknown=True);return v
